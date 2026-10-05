import os
import glob
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class SatelliteSRDataset(Dataset):
    def __init__(self, root, split, norm_stats_path=None, augment=True):
        self.lr_dir = os.path.join(root, split, 'LR')
        self.gt_dir = os.path.join(root, split, 'GT')
        lr_files = sorted(glob.glob(os.path.join(self.lr_dir, '*.npy')))
        filenames = [os.path.basename(f) for f in lr_files]
        self.filenames = [
            f for f in filenames if os.path.exists(os.path.join(self.gt_dir, f))
        ]
        self.augment = augment
        self.norm_stats = None
        if norm_stats_path is not None and os.path.exists(norm_stats_path):
            with open(norm_stats_path, 'r') as f:
                self.norm_stats = json.load(f)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                lr = np.flip(lr, axis=2).copy()
                gt = np.flip(gt, axis=2).copy()
            if random.random() < 0.5:
                lr = np.flip(lr, axis=1).copy()
                gt = np.flip(gt, axis=1).copy()
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                lr = np.rot90(lr, k, axes=(1, 2)).copy()
                gt = np.rot90(gt, k, axes=(1, 2)).copy()

        lr_t = torch.from_numpy(lr).float().clamp(0.0, 1.0)
        gt_t = torch.from_numpy(gt).float().clamp(0.0, 1.0)
        return lr_t, gt_t


def build_dataloader(root, split, batch_size, num_workers, val_fraction=0.1, seed=42, norm_stats_path=None):
    if split == 'train':
        base_dataset = SatelliteSRDataset(root, 'train', norm_stats_path, augment=True)
        n = len(base_dataset)
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)
        n_val = max(1, int(n * val_fraction)) if n > 0 else 0
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_subset = Subset(SatelliteSRDataset(root, 'train', norm_stats_path, augment=True), train_indices)
        val_subset = Subset(SatelliteSRDataset(root, 'train', norm_stats_path, augment=False), val_indices)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        )
        return train_loader, val_loader

    test_dataset = SatelliteSRDataset(root, 'test', norm_stats_path, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return test_loader
