import os
import sys
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer


def make_dummy_dataset(root, n_train=8):
    for split in ['train', 'test']:
        for sub in ['LR', 'GT']:
            os.makedirs(os.path.join(root, split, sub), exist_ok=True)
    for i in range(n_train):
        lr = np.random.rand(4, 64, 64).astype(np.float32)
        gt = np.random.rand(4, 256, 256).astype(np.float32)
        np.save(os.path.join(root, 'train', 'LR', f'{i:04d}.npy'), lr)
        np.save(os.path.join(root, 'train', 'GT', f'{i:04d}.npy'), gt)


def test_resume_continuity():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = os.path.join(base_dir, '_test_resume_tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    data_root = os.path.join(tmp_dir, 'dataset')
    make_dummy_dataset(data_root)

    cfg = {
        'stage': 'stage1',
        'run_name': 'test_resume_run',
        'data_root': data_root,
        'batch_size': 2,
        'num_workers': 0,
        'use_amp': False,
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'latent_dim': 16,
        'cond_dim': 32,
        'hidden_dim': 32,
        'flow_depth': 2,
        'hat_depth': 2,
        'num_epochs': 1,
        'val_fraction': 0.25,
        'seed': 0,
        'save_every_steps': 1,
        'log_every_steps': 1,
    }

    original_cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        trainer1 = Trainer(cfg)
        trainer1.fit()
        step_after_full = trainer1.global_step

        latest = trainer1.checkpointer.find_latest()
        assert latest is not None

        trainer2 = Trainer(cfg)
        trainer2.resume(latest)
        assert trainer2.global_step == step_after_full
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    test_resume_continuity()
    print('resume test passed')
