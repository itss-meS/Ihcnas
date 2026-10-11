import os
import sys
import tempfile
import shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import SatelliteSRDataset, build_dataloader


def _make_split_dirs(root, split):
    os.makedirs(os.path.join(root, split, 'GT'), exist_ok=True)
    os.makedirs(os.path.join(root, split, 'LR'), exist_ok=True)


def test_train_degradation_is_stochastic_from_gt_only():
    tmp_dir = tempfile.mkdtemp(prefix='dataset_degrade_')
    try:
        _make_split_dirs(tmp_dir, 'train')
        gt = np.random.rand(4, 256, 256).astype(np.float32)
        np.save(os.path.join(tmp_dir, 'train', 'GT', '0000.npy'), gt)

        dataset = SatelliteSRDataset(
            tmp_dir,
            'train',
            augment=False,
            degrade_on_the_fly=True,
            noise_std_range=(0.01, 0.02),
            jpeg_artifact_prob=0.0,
        )

        lr1, gt1 = dataset[0]
        lr2, gt2 = dataset[0]

        assert gt1.shape == (4, 256, 256)
        assert lr1.shape == (4, 64, 64)
        assert torch.allclose(gt1, gt2)
        assert not torch.allclose(lr1, lr2)
    finally:
        shutil.rmtree(tmp_dir)


def test_eval_path_is_deterministic_with_saved_lr():
    tmp_dir = tempfile.mkdtemp(prefix='dataset_eval_')
    try:
        _make_split_dirs(tmp_dir, 'test')
        gt = np.random.rand(4, 256, 256).astype(np.float32)
        lr = np.random.rand(4, 64, 64).astype(np.float32)
        np.save(os.path.join(tmp_dir, 'test', 'GT', '0000.npy'), gt)
        np.save(os.path.join(tmp_dir, 'test', 'LR', '0000.npy'), lr)

        dataset = SatelliteSRDataset(
            tmp_dir,
            'test',
            augment=False,
            degrade_on_the_fly=False,
        )

        lr1, gt1 = dataset[0]
        lr2, gt2 = dataset[0]

        assert torch.allclose(lr1, lr2)
        assert torch.allclose(gt1, gt2)
    finally:
        shutil.rmtree(tmp_dir)


def test_build_dataloader_train_works_without_saved_lr():
    tmp_dir = tempfile.mkdtemp(prefix='dataset_loader_')
    try:
        _make_split_dirs(tmp_dir, 'train')
        for idx in range(4):
            gt = np.random.rand(4, 256, 256).astype(np.float32)
            np.save(os.path.join(tmp_dir, 'train', 'GT', f'{idx:04d}.npy'), gt)

        train_loader, val_loader = build_dataloader(
            root=tmp_dir,
            split='train',
            batch_size=2,
            num_workers=0,
            val_fraction=0.25,
            seed=0,
        )

        train_lr, train_gt = next(iter(train_loader))
        assert train_lr.shape[-2:] == (64, 64)
        assert train_gt.shape[-2:] == (256, 256)
        assert len(val_loader) == 1
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    test_train_degradation_is_stochastic_from_gt_only()
    test_eval_path_is_deterministic_with_saved_lr()
    test_build_dataloader_train_works_without_saved_lr()
    print('dataset degradation tests passed')
