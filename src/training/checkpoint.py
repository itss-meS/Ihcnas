import os
import random
import numpy as np
import torch


class FullStateCheckpoint:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, scaler, epoch, global_step, best_metric, filename, config=None):
        path = os.path.join(self.run_dir, filename)
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'epoch': epoch,
            'global_step': global_step,
            'best_metric': best_metric,
            'config': config,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        torch.save(state, path)
        return path

    def find_latest(self):
        last_path = os.path.join(self.run_dir, 'last.ckpt')
        if os.path.exists(last_path):
            return last_path
        return None

    def load(self, path, model, optimizer=None, scheduler=None, scaler=None, map_location='cpu'):
        state = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(state['model_state_dict'])

        if optimizer is not None and state.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if scheduler is not None and state.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        if scaler is not None and state.get('scaler_state_dict') is not None:
            scaler.load_state_dict(state['scaler_state_dict'])

        if state.get('torch_rng_state') is not None:
            rng_state = state['torch_rng_state']
            if not torch.is_tensor(rng_state):
                rng_state = torch.tensor(rng_state, dtype=torch.uint8)
            torch.set_rng_state(rng_state.cpu().to(torch.uint8))
        if state.get('cuda_rng_state') is not None and torch.cuda.is_available():
            cuda_rng_state = [t.cpu().to(torch.uint8) for t in state['cuda_rng_state']]
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if state.get('numpy_rng_state') is not None:
            np.random.set_state(state['numpy_rng_state'])
        if state.get('python_rng_state') is not None:
            random.setstate(state['python_rng_state'])

        epoch = state.get('epoch', 0)
        global_step = state.get('global_step', 0)
        best_metric = state.get('best_metric', None)
        return epoch, global_step, best_metric