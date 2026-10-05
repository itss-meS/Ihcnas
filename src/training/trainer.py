import os
import time
import torch
from collections import deque
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import build_dataloader
from src.model.satsrflow import SatSRFlow
from src.training.checkpoint import FullStateCheckpoint
from src.metrics.metrics import compute_all_metrics


def format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f'{hours:02d}h {minutes:02d}m {secs:02d}s'
    if minutes > 0:
        return f'{minutes:02d}m {secs:02d}s'
    return f'{secs:02d}s'


def render_progress_bar(fraction, width=30):
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(width * fraction))
    if filled >= width:
        bar = '=' * width
    elif filled == 0:
        bar = '-' * width
    else:
        bar = '=' * (filled - 1) + '>' + '-' * (width - filled)
    percent = int(round(fraction * 100))
    return f'[{bar}] {percent:3d}%'


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stage = cfg.get('stage', 'stage2')
        self.run_name = cfg.get('run_name', self.stage)
        self.run_dir = os.path.join('checkpoints', self.run_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader, self.val_loader = build_dataloader(
            root=cfg.get('data_root', 'dataset'),
            split='train',
            batch_size=cfg.get('batch_size', 8),
            num_workers=cfg.get('num_workers', 2),
            val_fraction=cfg.get('val_fraction', 0.1),
            seed=cfg.get('seed', 42),
            norm_stats_path=cfg.get('norm_stats_path', None),
        )

        self.model = SatSRFlow(
            in_ch=4,
            out_ch=4,
            latent_dim=cfg.get('latent_dim', 64),
            cond_dim=cfg.get('cond_dim', 128),
            hidden_dim=cfg.get('hidden_dim', 128),
            flow_depth=cfg.get('flow_depth', 6),
            hat_depth=cfg.get('hat_depth', 4),
            scale_factor=cfg.get('scale_factor', 4),
        ).to(self.device)

        init_path = cfg.get('init_from_checkpoint', None) or cfg.get('init_from_stage1', None)
        if self.stage == 'stage2' and init_path:
            if os.path.exists(init_path):
                init_ckpt = torch.load(init_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(init_ckpt['model_state_dict'], strict=False)
                print(f'initialized weights from {init_path}')

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.get('learning_rate', 2e-4),
            weight_decay=cfg.get('weight_decay', 1e-4),
        )
        total_steps = cfg.get('num_epochs', 50) * max(len(self.train_loader), 1)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(total_steps, 1))
        self.amp_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler = GradScaler(self.amp_device_type, enabled=cfg.get('use_amp', True) and torch.cuda.is_available())
        self.checkpointer = FullStateCheckpoint(self.run_dir)
        self.writer = SummaryWriter(log_dir=self.run_dir)

        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = None

    def resume(self, resume_path):
        self.start_epoch, self.global_step, self.best_metric = self.checkpointer.load(
            resume_path, self.model, self.optimizer, self.scheduler, self.scaler,
            map_location=self.device,
        )
        print(f'resumed from {resume_path} at epoch {self.start_epoch} step {self.global_step}')

    def _forward(self, lr_img, gt_img):
        if self.stage == 'stage1':
            return self.model.forward_stage1(lr_img, gt_img)
        return self.model.forward_train(lr_img, gt_img)

    def run_validation(self):
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'psnr': 0.0, 'ssim': 0.0, 'sam': 0.0, 'ergas': 0.0, 'ndvi_mae': 0.0}
        n_batches = 0
        with torch.no_grad():
            for lr_img, gt_img in self.val_loader:
                lr_img = lr_img.to(self.device)
                gt_img = gt_img.to(self.device)
                loss, _ = self._forward(lr_img, gt_img)
                num_steps = 1 if self.stage == 'stage1' else self.cfg.get('num_euler_steps', 5)
                pred, _ = self.model.infer(lr_img, gt_img.shape[-2:], num_steps=num_steps, ensemble_size=1)
                total_loss += loss.item()
                metrics = compute_all_metrics(pred, gt_img, self.cfg.get('scale_factor', 4))
                for k in total_metrics:
                    total_metrics[k] += metrics[k]
                n_batches += 1
        self.model.train()
        avg_loss = total_loss / max(n_batches, 1)
        avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
        return avg_loss, avg_metrics

    def fit(self):
        num_epochs = self.cfg.get('num_epochs', 50)
        save_every_steps = self.cfg.get('save_every_steps', 500)
        log_every_steps = self.cfg.get('log_every_steps', 50)
        use_amp = self.cfg.get('use_amp', True) and torch.cuda.is_available()

        steps_per_epoch = max(len(self.train_loader), 1)
        total_steps = num_epochs * steps_per_epoch
        step_time_window = deque(maxlen=50)
        run_start_time = time.time()

        run_label = f'{self.run_name} [{self.stage}]'
        print(f'\n>> starting run: {run_label}')
        print(f'total planned steps: {total_steps} ({num_epochs} epochs x {steps_per_epoch} steps/epoch)\n')

        self.model.train()
        for epoch in range(self.start_epoch, num_epochs):
            for lr_img, gt_img in self.train_loader:
                step_start = time.time()

                lr_img = lr_img.to(self.device)
                gt_img = gt_img.to(self.device)

                self.optimizer.zero_grad()
                with autocast(self.amp_device_type, enabled=use_amp):
                    loss, loss_dict = self._forward(lr_img, gt_img)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                self.global_step += 1
                step_time_window.append(time.time() - step_start)

                if self.global_step % log_every_steps == 0:
                    avg_step_time = sum(step_time_window) / len(step_time_window)
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
                    remaining_steps = max(total_steps - self.global_step, 0)
                    eta_seconds = remaining_steps * avg_step_time
                    elapsed_seconds = time.time() - run_start_time

                    epoch_fraction = ((self.global_step - 1) % steps_per_epoch + 1) / steps_per_epoch
                    bar = render_progress_bar(epoch_fraction)

                    loss_str = ' '.join(f'{k}={v:.5f}' for k, v in loss_dict.items())

                    print(
                        f'epoch {epoch + 1:03d}/{num_epochs:03d} '
                        f'{bar} '
                        f'step {self.global_step:06d}/{total_steps:06d}  '
                        f'{loss_str}  '
                        f'{steps_per_sec:.2f} it/s  '
                        f'elapsed {format_duration(elapsed_seconds)}  '
                        f'ETA {format_duration(eta_seconds)}'
                    )

                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
                    self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                    self.writer.add_scalar('train/steps_per_sec', steps_per_sec, self.global_step)

                if self.global_step % save_every_steps == 0:
                    self.checkpointer.save(
                        self.model, self.optimizer, self.scheduler, self.scaler,
                        epoch, self.global_step, self.best_metric, 'last.ckpt',
                        config=self.cfg,
                    )

            val_loss, val_metrics = self.run_validation()
            metric_str = ' '.join(f'{k}={v:.5f}' for k, v in val_metrics.items())
            print(f'\n== epoch {epoch + 1:03d} complete ==  val_loss={val_loss:.5f}  {metric_str}\n')
            self.writer.add_scalar('val/loss', val_loss, self.global_step)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, self.global_step)

            if self.best_metric is None or val_metrics['psnr'] > self.best_metric:
                self.best_metric = val_metrics['psnr']

            self.checkpointer.save(
                self.model, self.optimizer, self.scheduler, self.scaler,
                epoch + 1, self.global_step, self.best_metric, 'last.ckpt',
                config=self.cfg,
            )
            self.checkpointer.save(
                self.model, self.optimizer, self.scheduler, self.scaler,
                epoch + 1, self.global_step, self.best_metric, f'epoch_{epoch + 1:04d}.ckpt',
                config=self.cfg,
            )

            keep_last_n = self.cfg.get('keep_last_n_epoch_ckpts', 3)
            self._cleanup_old_epoch_ckpts(keep_last_n)

        total_elapsed = time.time() - run_start_time
        print(f'training complete: {run_label}')
        print(f'total training time this run: {format_duration(total_elapsed)}')
        self.writer.close()

    def _cleanup_old_epoch_ckpts(self, keep_last_n):
        epoch_ckpts = sorted(
            f for f in os.listdir(self.run_dir)
            if f.startswith('epoch_') and f.endswith('.ckpt')
        )
        if len(epoch_ckpts) > keep_last_n:
            for old_ckpt in epoch_ckpts[:-keep_last_n]:
                os.remove(os.path.join(self.run_dir, old_ckpt))