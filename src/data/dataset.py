import os
import glob
import random
import json
import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from PIL import Image


class SatelliteSRDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        norm_stats_path=None,
        augment=True,
        degrade_on_the_fly=None,
        scale_factor=4,
        blur_sigma_range=(0.4, 1.6),
        noise_std_range=(0.0, 0.01),
        jpeg_artifact_prob=0.2,
        jpeg_quality_range=(70, 95),
        downsample_jitter_range=(1.0, 1.25),
    ):
        self.lr_dir = os.path.join(root, split, 'LR')
        self.gt_dir = os.path.join(root, split, 'GT')
        self.augment = augment
        self.scale_factor = max(float(scale_factor), 1.0)
        self.blur_sigma_range = blur_sigma_range
        self.noise_std_range = noise_std_range
        self.jpeg_artifact_prob = jpeg_artifact_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.downsample_jitter_range = downsample_jitter_range
        self.degrade_on_the_fly = (
            split == 'train' and augment if degrade_on_the_fly is None else degrade_on_the_fly
        )

        gt_files = sorted(glob.glob(os.path.join(self.gt_dir, '*.npy')))
        gt_filenames = [os.path.basename(f) for f in gt_files]
        if self.degrade_on_the_fly:
            self.filenames = gt_filenames
        else:
            lr_files = sorted(glob.glob(os.path.join(self.lr_dir, '*.npy')))
            lr_filenames = [os.path.basename(f) for f in lr_files]
            self.filenames = [
                f for f in lr_filenames if os.path.exists(os.path.join(self.gt_dir, f))
            ]

        self.norm_stats = None
        if norm_stats_path is not None and os.path.exists(norm_stats_path):
            with open(norm_stats_path, 'r') as f:
                self.norm_stats = json.load(f)

    def __len__(self):
        return len(self.filenames)

    def _sample_range(self, bounds):
        lo, hi = bounds
        if lo > hi:
            lo, hi = hi, lo
        return random.uniform(lo, hi)

    def _apply_jpeg_artifacts(self, lr):
        if random.random() >= self.jpeg_artifact_prob:
            return lr
        lo, hi = self.jpeg_quality_range
        if lo > hi:
            lo, hi = hi, lo
        quality = int(round(random.uniform(lo, hi)))
        degraded = np.empty_like(lr)
        for c in range(lr.shape[0]):
            chan = np.clip(lr[c] * 255.0, 0.0, 255.0).astype(np.uint8)
            img = Image.fromarray(chan, mode='L')
            buff = io.BytesIO()
            img.save(buff, format='JPEG', quality=quality, optimize=False)
            buff.seek(0)
            decoded = np.array(Image.open(buff), dtype=np.float32) / 255.0
            degraded[c] = decoded
        return degraded

    def _degrade_from_gt(self, gt):
        c, h, w = gt.shape
        lr_h = max(1, int(round(h / self.scale_factor)))
        lr_w = max(1, int(round(w / self.scale_factor)))
        jitter = max(self._sample_range(self.downsample_jitter_range), 1.0)
        mid_h = max(1, int(round(lr_h * jitter)))
        mid_w = max(1, int(round(lr_w * jitter)))
        sigma = self._sample_range(self.blur_sigma_range)
        noise_std = self._sample_range(self.noise_std_range)
        interp_order = random.choice([1, 2, 3])

        lr = np.zeros((c, lr_h, lr_w), dtype=np.float32)
        for band_idx in range(c):
            band = gaussian_filter(gt[band_idx], sigma=sigma)
            band = resize(
                band, (mid_h, mid_w), order=interp_order,
                anti_aliasing=True, preserve_range=True,
            )
            if (mid_h, mid_w) != (lr_h, lr_w):
                band = resize(
                    band, (lr_h, lr_w), order=interp_order,
                    anti_aliasing=True, preserve_range=True,
                )
            lr[band_idx] = band.astype(np.float32)

        if noise_std > 0:
            lr = lr + np.random.normal(0.0, noise_std, size=lr.shape).astype(np.float32)

        lr = self._apply_jpeg_artifacts(lr)
        return np.clip(lr, 0.0, 1.0)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        if self.degrade_on_the_fly:
            lr = self._degrade_from_gt(gt)
        else:
            lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)

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
        base_dataset = SatelliteSRDataset(root, 'train', norm_stats_path, augment=True, degrade_on_the_fly=True)
        n = len(base_dataset)
        indices = list(range(n))
        rng = random.Random(seed)
        rng.shuffle(indices)
        n_val = max(1, int(n * val_fraction)) if n > 0 else 0
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_subset = Subset(
            SatelliteSRDataset(root, 'train', norm_stats_path, augment=True, degrade_on_the_fly=True),
            train_indices
        )
        val_subset = Subset(
            SatelliteSRDataset(root, 'train', norm_stats_path, augment=False, degrade_on_the_fly=False),
            val_indices
        )

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        )
        return train_loader, val_loader

    test_dataset = SatelliteSRDataset(root, 'test', norm_stats_path, augment=False, degrade_on_the_fly=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return test_loader
