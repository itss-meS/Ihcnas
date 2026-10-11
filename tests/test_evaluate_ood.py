import os
import sys
import tempfile
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_ood import (
    aggregate_rows,
    bicubic_resize_to_gt,
    choose_safe_ssim_window,
    deterministic_gaussian_downsample,
    list_paired_files,
    load_and_validate_npy,
    metric_to_text,
    write_markdown_matrix,
)


def test_bicubic_resize_to_target_shape():
    arr = np.random.rand(4, 8, 8).astype(np.float32)
    out = bicubic_resize_to_gt(arr, (32, 24))
    assert out.shape == (4, 32, 24)
    assert np.isfinite(out).all()


def test_deterministic_gaussian_downsample_reproducible():
    gt = np.random.rand(4, 32, 32).astype(np.float32)
    lr1, clip1 = deterministic_gaussian_downsample(gt, sigma=0.8, scale_factor=4)
    lr2, clip2 = deterministic_gaussian_downsample(gt, sigma=0.8, scale_factor=4)
    assert lr1.shape == (4, 8, 8)
    assert np.allclose(lr1, lr2)
    assert clip1 == clip2


def test_pairing_and_validation_checks():
    tmp = tempfile.mkdtemp(prefix="ood_pairs_")
    try:
        gt_dir = os.path.join(tmp, "GT")
        lr_dir = os.path.join(tmp, "LR")
        os.makedirs(gt_dir)
        os.makedirs(lr_dir)

        np.save(os.path.join(gt_dir, "a.npy"), np.zeros((4, 16, 16), dtype=np.float32))
        np.save(os.path.join(gt_dir, "b.npy"), np.zeros((4, 16, 16), dtype=np.float32))
        np.save(os.path.join(lr_dir, "a.npy"), np.zeros((4, 4, 4), dtype=np.float32))

        pairs = list_paired_files(gt_dir, lr_dir)
        assert pairs == ["a.npy"]

        loaded = load_and_validate_npy(os.path.join(gt_dir, "a.npy"))
        assert loaded.shape == (4, 16, 16)

        bad_path = os.path.join(gt_dir, "bad.npy")
        np.save(bad_path, np.zeros((3, 16, 16), dtype=np.float32))
        try:
            load_and_validate_npy(bad_path)
            assert False, "Expected validation error for wrong channel count"
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp)


def test_safe_ssim_window_selection():
    assert choose_safe_ssim_window(2, 7) is None
    assert choose_safe_ssim_window(3, 7) == 3
    assert choose_safe_ssim_window(10, 10) == 9
    assert choose_safe_ssim_window(20, 20) == 11


def test_metric_aggregation_and_markdown_format():
    per_image = [
        {
            "condition": "synthetic_sigma",
            "sigma": "0.4",
            "method": "Bicubic",
            "image": "x.npy",
            "psnr": 20.0,
            "ssim": 0.5,
            "lpips": 0.2,
            "status": "ok",
            "error": "",
            "clip_applied": False,
        },
        {
            "condition": "synthetic_sigma",
            "sigma": "0.4",
            "method": "Bicubic",
            "image": "y.npy",
            "psnr": 22.0,
            "ssim": 0.7,
            "lpips": 0.1,
            "status": "ok",
            "error": "",
            "clip_applied": False,
        },
    ]
    agg = aggregate_rows(per_image)
    assert len(agg) == 1
    row = agg[0]
    assert row["num_images"] == 2
    assert abs(row["psnr"] - 21.0) < 1e-6
    assert metric_to_text(None) == "NA"

    tmp = tempfile.mkdtemp(prefix="ood_md_")
    try:
        md_path = os.path.join(tmp, "matrix.md")
        write_markdown_matrix(md_path, agg)
        text = open(md_path, "r").read()
        assert "| Condition | Blur sigma | Method" in text
        assert "synthetic_sigma" in text
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_bicubic_resize_to_target_shape()
    test_deterministic_gaussian_downsample_reproducible()
    test_pairing_and_validation_checks()
    test_safe_ssim_window_selection()
    test_metric_aggregation_and_markdown_format()
    print("evaluate_ood tests passed")
