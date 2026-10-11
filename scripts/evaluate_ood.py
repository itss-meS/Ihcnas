import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.metrics import psnr as metric_psnr
from src.metrics.metrics import ssim as metric_ssim
from src.model.satsrflow import SatSRFlow


@dataclass
class EvalSettings:
    data_root: str
    blur_sigmas: List[float]
    scale_factor: int
    checkpoint: Optional[str]
    device: str
    num_steps: int
    ensemble_size: int
    output_dir: str
    seed: int
    max_images: Optional[int]
    disable_lpips: bool
    force: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: Dict, device: torch.device) -> SatSRFlow:
    model = SatSRFlow(
        in_ch=4,
        out_ch=4,
        latent_dim=cfg.get("latent_dim", 64),
        cond_dim=cfg.get("cond_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 128),
        flow_depth=cfg.get("flow_depth", 6),
        hat_depth=cfg.get("hat_depth", 4),
        scale_factor=cfg.get("scale_factor", 4),
    ).to(device)
    return model


def load_model(checkpoint_path: str, device: torch.device) -> SatSRFlow:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def list_paired_files(gt_dir: str, lr_dir: Optional[str]) -> List[str]:
    if not os.path.isdir(gt_dir):
        return []
    gt_names = sorted(name for name in os.listdir(gt_dir) if name.endswith(".npy"))
    if lr_dir is None or not os.path.isdir(lr_dir):
        return gt_names
    return [name for name in gt_names if os.path.exists(os.path.join(lr_dir, name))]


def load_and_validate_npy(path: str, expected_channels: int = 4) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [C,H,W] in {path}, got shape {arr.shape}")
    if arr.shape[0] != expected_channels:
        raise ValueError(f"Expected {expected_channels} bands in {path}, got {arr.shape[0]}")
    if arr.shape[1] < 1 or arr.shape[2] < 1:
        raise ValueError(f"Invalid spatial size in {path}: {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values found in {path}")
    return arr.astype(np.float32)


def deterministic_gaussian_downsample(
    gt: np.ndarray,
    sigma: float,
    scale_factor: int,
) -> Tuple[np.ndarray, bool]:
    c, h, w = gt.shape
    lr_h = max(1, int(round(h / float(scale_factor))))
    lr_w = max(1, int(round(w / float(scale_factor))))
    lr = np.empty((c, lr_h, lr_w), dtype=np.float32)

    for band_idx in range(c):
        blurred = gaussian_filter(gt[band_idx], sigma=float(sigma), mode="reflect")
        resized = resize(
            blurred,
            (lr_h, lr_w),
            order=3,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )
        lr[band_idx] = resized.astype(np.float32)

    clipped = bool(np.any((lr < 0.0) | (lr > 1.0)))
    lr = np.clip(lr, 0.0, 1.0).astype(np.float32)
    return lr, clipped


def bicubic_resize_to_gt(arr_chw: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(arr_chw).unsqueeze(0)
    up = F.interpolate(tensor, size=target_hw, mode="bicubic", align_corners=False)
    return up.squeeze(0).numpy().astype(np.float32)


def choose_safe_ssim_window(h: int, w: int) -> Optional[int]:
    min_side = min(h, w)
    if min_side < 3:
        return None
    win = min(11, min_side)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return None
    return win


def compute_psnr_ssim(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, Optional[float], Optional[str]]:
    pred_t = torch.from_numpy(np.clip(pred, 0.0, 1.0)).unsqueeze(0)
    gt_t = torch.from_numpy(np.clip(gt, 0.0, 1.0)).unsqueeze(0)

    psnr_value = float(metric_psnr(pred_t, gt_t, max_val=1.0).item())

    h, w = gt.shape[-2:]
    win = choose_safe_ssim_window(h, w)
    if win is None:
        return psnr_value, None, "ssim_skipped_small_image"

    ssim_value = float(metric_ssim(pred_t, gt_t, window_size=win, max_val=1.0).item())
    if not math.isfinite(ssim_value):
        return psnr_value, None, "ssim_non_finite"

    return psnr_value, ssim_value, None


def to_lpips_tensor(arr_chw: np.ndarray) -> torch.Tensor:
    rgb = arr_chw[[2, 1, 0], :, :]
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = rgb * 2.0 - 1.0
    return torch.from_numpy(rgb).unsqueeze(0).float()


def maybe_create_lpips(disable_lpips: bool, device: torch.device):
    if disable_lpips:
        return None
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS is required unless --disable_lpips is set. Install with: pip install lpips"
        ) from exc
    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    return metric


def compute_lpips(
    lpips_metric,
    pred: np.ndarray,
    gt: np.ndarray,
    device: torch.device,
) -> Optional[float]:
    if lpips_metric is None:
        return None
    pred_t = to_lpips_tensor(pred).to(device)
    gt_t = to_lpips_tensor(gt).to(device)
    with torch.inference_mode():
        value = float(lpips_metric(pred_t, gt_t).item())
    if not math.isfinite(value):
        raise ValueError("LPIPS returned non-finite value")
    return value


def run_model_infer(
    model: SatSRFlow,
    lr: np.ndarray,
    target_hw: Tuple[int, int],
    num_steps: int,
    ensemble_size: int,
    device: torch.device,
) -> np.ndarray:
    lr_t = torch.from_numpy(lr).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        mean_sr, _ = model.infer(
            lr_t,
            target_shape=target_hw,
            num_steps=num_steps,
            ensemble_size=ensemble_size,
        )
    out = mean_sr.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def ensure_output_paths(output_dir: str, force: bool) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "per_image_csv": os.path.join(output_dir, "ood_per_image_metrics.csv"),
        "matrix_csv": os.path.join(output_dir, "ood_baseline_matrix.csv"),
        "matrix_md": os.path.join(output_dir, "ood_baseline_matrix.md"),
        "meta_json": os.path.join(output_dir, "ood_metadata.json"),
    }
    if not force:
        existing = [p for p in paths.values() if os.path.exists(p)]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --force to overwrite: " + ", ".join(existing)
            )
    return paths


def metric_to_text(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def aggregate_rows(per_image_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str], List[Dict]] = {}
    for row in per_image_rows:
        key = (row["condition"], row["sigma"], row["method"])
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for (condition, sigma, method), rows in sorted(grouped.items()):
        done_rows = [r for r in rows if r["status"] == "ok"]
        failed_rows = [r for r in rows if r["status"] != "ok"]

        def mean_metric(name: str) -> Optional[float]:
            vals = [r[name] for r in done_rows if r[name] is not None]
            if not vals:
                return None
            return float(np.mean(vals))

        status = "ok" if done_rows else "not_run"
        note_parts = []
        if failed_rows:
            note_parts.append(f"errors={len(failed_rows)}")
            reasons = sorted({r.get("error", "") for r in failed_rows if r.get("error")})
            if reasons:
                note_parts.append("; ".join(reasons))
        if status != "ok" and not note_parts:
            note_parts.append("no successful samples")

        aggregates.append(
            {
                "condition": condition,
                "sigma": sigma,
                "method": method,
                "psnr": mean_metric("psnr"),
                "ssim": mean_metric("ssim"),
                "lpips": mean_metric("lpips"),
                "num_images": len(done_rows),
                "status": status,
                "notes": " | ".join(note_parts) if note_parts else "",
            }
        )
    return aggregates


def add_missing_required_matrix_rows(
    rows: List[Dict],
    blur_sigmas: Sequence[float],
    include_model: bool,
    bicubic_missing_reason: str,
    model_missing_reason: str,
) -> List[Dict]:
    by_key = {(r["condition"], r["sigma"], r["method"]): r for r in rows}
    methods = ["Bicubic", "IHCNAS"]
    for sigma in blur_sigmas:
        sigma_key = f"{sigma:.1f}" if sigma in (0.4, 0.8, 2.0) else str(sigma)
        for method in methods:
            status = "not_run"
            if method == "IHCNAS":
                notes = model_missing_reason if not include_model else model_missing_reason
            else:
                notes = bicubic_missing_reason
            key = ("synthetic_sigma", sigma_key, method)
            if key not in by_key:
                rows.append(
                    {
                        "condition": "synthetic_sigma",
                        "sigma": sigma_key,
                        "method": method,
                        "psnr": None,
                        "ssim": None,
                        "lpips": None,
                        "num_images": 0,
                        "status": status,
                        "notes": notes,
                    }
                )
    return rows


def write_per_image_csv(path: str, rows: List[Dict]) -> None:
    fieldnames = [
        "condition",
        "sigma",
        "method",
        "image",
        "psnr",
        "ssim",
        "lpips",
        "status",
        "error",
        "clip_applied",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "condition": row["condition"],
                    "sigma": row["sigma"],
                    "method": row["method"],
                    "image": row["image"],
                    "psnr": metric_to_text(row["psnr"]),
                    "ssim": metric_to_text(row["ssim"]),
                    "lpips": metric_to_text(row["lpips"]),
                    "status": row["status"],
                    "error": row.get("error", ""),
                    "clip_applied": str(bool(row.get("clip_applied", False))),
                }
            )


def write_aggregate_csv(path: str, rows: List[Dict]) -> None:
    fieldnames = ["condition", "sigma", "method", "psnr", "ssim", "lpips", "num_images", "status", "notes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "condition": row["condition"],
                    "sigma": row["sigma"],
                    "method": row["method"],
                    "psnr": metric_to_text(row["psnr"]),
                    "ssim": metric_to_text(row["ssim"]),
                    "lpips": metric_to_text(row["lpips"]),
                    "num_images": row["num_images"],
                    "status": row["status"],
                    "notes": row.get("notes", ""),
                }
            )


def write_markdown_matrix(path: str, rows: List[Dict]) -> None:
    ordered = sorted(
        rows,
        key=lambda r: (
            0 if r["condition"] == "synthetic_sigma" else 1,
            float(r["sigma"]) if r["sigma"] not in ("NA", "") else 999.0,
            0 if r["method"] == "Bicubic" else 1,
        ),
    )

    lines = [
        "| Condition | Blur sigma | Method | PSNR (dB) | SSIM | LPIPS | Number of images | Status | Notes |",
        "|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in ordered:
        lines.append(
            "| {condition} | {sigma} | {method} | {psnr} | {ssim} | {lpips} | {num} | {status} | {notes} |".format(
                condition=row["condition"],
                sigma=row["sigma"],
                method=row["method"],
                psnr=metric_to_text(row["psnr"]),
                ssim=metric_to_text(row["ssim"]),
                lpips=metric_to_text(row["lpips"]),
                num=row["num_images"],
                status=row["status"],
                notes=row.get("notes", ""),
            )
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def evaluate_dataset(settings: EvalSettings) -> Dict[str, str]:
    seed_everything(settings.seed)
    device = resolve_device(settings.device)
    paths = ensure_output_paths(settings.output_dir, settings.force)

    model = None
    if settings.checkpoint is not None:
        model = load_model(settings.checkpoint, device)

    lpips_metric = maybe_create_lpips(settings.disable_lpips, device)

    test_root = os.path.join(settings.data_root, "test")
    gt_dir = os.path.join(test_root, "GT")
    lr_dir = os.path.join(test_root, "LR")

    gt_files = list_paired_files(gt_dir, None)
    fixed_pairs = list_paired_files(gt_dir, lr_dir)

    if settings.max_images is not None:
        gt_files = gt_files[: settings.max_images]
        fixed_pairs = fixed_pairs[: settings.max_images]

    per_image_rows: List[Dict] = []

    if gt_files:
        for sigma in settings.blur_sigmas:
            sigma_str = f"{sigma:.1f}" if sigma in (0.4, 0.8, 2.0) else str(sigma)
            for fname in gt_files:
                gt_path = os.path.join(gt_dir, fname)
                try:
                    gt = load_and_validate_npy(gt_path)
                    lr_syn, clip_applied = deterministic_gaussian_downsample(
                        gt=gt,
                        sigma=float(sigma),
                        scale_factor=settings.scale_factor,
                    )
                    target_hw = tuple(gt.shape[-2:])

                    bicubic = bicubic_resize_to_gt(lr_syn, target_hw)
                    bic_psnr, bic_ssim, bic_err = compute_psnr_ssim(bicubic, gt)
                    bic_lpips = compute_lpips(lpips_metric, bicubic, gt, device)
                    per_image_rows.append(
                        {
                            "condition": "synthetic_sigma",
                            "sigma": sigma_str,
                            "method": "Bicubic",
                            "image": fname,
                            "psnr": bic_psnr,
                            "ssim": bic_ssim,
                            "lpips": bic_lpips,
                            "status": "ok" if bic_err is None else "error",
                            "error": bic_err or "",
                            "clip_applied": clip_applied,
                        }
                    )

                    if model is not None:
                        sr = run_model_infer(
                            model=model,
                            lr=lr_syn,
                            target_hw=target_hw,
                            num_steps=settings.num_steps,
                            ensemble_size=settings.ensemble_size,
                            device=device,
                        )
                        m_psnr, m_ssim, m_err = compute_psnr_ssim(sr, gt)
                        m_lpips = compute_lpips(lpips_metric, sr, gt, device)
                        per_image_rows.append(
                            {
                                "condition": "synthetic_sigma",
                                "sigma": sigma_str,
                                "method": "IHCNAS",
                                "image": fname,
                                "psnr": m_psnr,
                                "ssim": m_ssim,
                                "lpips": m_lpips,
                                "status": "ok" if m_err is None else "error",
                                "error": m_err or "",
                                "clip_applied": clip_applied,
                            }
                        )
                except Exception as exc:
                    per_image_rows.append(
                        {
                            "condition": "synthetic_sigma",
                            "sigma": sigma_str,
                            "method": "Bicubic",
                            "image": fname,
                            "psnr": None,
                            "ssim": None,
                            "lpips": None,
                            "status": "error",
                            "error": str(exc),
                            "clip_applied": False,
                        }
                    )
                    if model is not None:
                        per_image_rows.append(
                            {
                                "condition": "synthetic_sigma",
                                "sigma": sigma_str,
                                "method": "IHCNAS",
                                "image": fname,
                                "psnr": None,
                                "ssim": None,
                                "lpips": None,
                                "status": "error",
                                "error": str(exc),
                                "clip_applied": False,
                            }
                        )

    if fixed_pairs:
        for fname in fixed_pairs:
            gt_path = os.path.join(gt_dir, fname)
            lr_path = os.path.join(lr_dir, fname)
            try:
                gt = load_and_validate_npy(gt_path)
                lr = load_and_validate_npy(lr_path)
                target_hw = tuple(gt.shape[-2:])

                bicubic = bicubic_resize_to_gt(lr, target_hw)
                bic_psnr, bic_ssim, bic_err = compute_psnr_ssim(bicubic, gt)
                bic_lpips = compute_lpips(lpips_metric, bicubic, gt, device)
                per_image_rows.append(
                    {
                        "condition": "fixed_test_lr",
                        "sigma": "NA",
                        "method": "Bicubic",
                        "image": fname,
                        "psnr": bic_psnr,
                        "ssim": bic_ssim,
                        "lpips": bic_lpips,
                        "status": "ok" if bic_err is None else "error",
                        "error": bic_err or "",
                        "clip_applied": False,
                    }
                )

                if model is not None:
                    sr = run_model_infer(
                        model=model,
                        lr=lr,
                        target_hw=target_hw,
                        num_steps=settings.num_steps,
                        ensemble_size=settings.ensemble_size,
                        device=device,
                    )
                    m_psnr, m_ssim, m_err = compute_psnr_ssim(sr, gt)
                    m_lpips = compute_lpips(lpips_metric, sr, gt, device)
                    per_image_rows.append(
                        {
                            "condition": "fixed_test_lr",
                            "sigma": "NA",
                            "method": "IHCNAS",
                            "image": fname,
                            "psnr": m_psnr,
                            "ssim": m_ssim,
                            "lpips": m_lpips,
                            "status": "ok" if m_err is None else "error",
                            "error": m_err or "",
                            "clip_applied": False,
                        }
                    )
            except Exception as exc:
                per_image_rows.append(
                    {
                        "condition": "fixed_test_lr",
                        "sigma": "NA",
                        "method": "Bicubic",
                        "image": fname,
                        "psnr": None,
                        "ssim": None,
                        "lpips": None,
                        "status": "error",
                        "error": str(exc),
                        "clip_applied": False,
                    }
                )
                if model is not None:
                    per_image_rows.append(
                        {
                            "condition": "fixed_test_lr",
                            "sigma": "NA",
                            "method": "IHCNAS",
                            "image": fname,
                            "psnr": None,
                            "ssim": None,
                            "lpips": None,
                            "status": "error",
                            "error": str(exc),
                            "clip_applied": False,
                        }
                    )

    aggregated_rows = aggregate_rows(per_image_rows)

    bicubic_reason = "dataset/test/GT unavailable" if not gt_files else "no successful samples"
    model_reason = (
        "dataset/test/GT unavailable; checkpoint not provided"
        if not gt_files and model is None
        else ("checkpoint not provided" if model is None else ("dataset/test/GT unavailable" if not gt_files else "no successful samples"))
    )
    aggregated_rows = add_missing_required_matrix_rows(
        aggregated_rows,
        settings.blur_sigmas,
        include_model=(model is not None),
        bicubic_missing_reason=bicubic_reason,
        model_missing_reason=model_reason,
    )

    if not fixed_pairs:
        for method in ["Bicubic", "IHCNAS"]:
            if method == "IHCNAS" and model is None:
                notes = "checkpoint not provided"
            else:
                notes = "dataset/test/LR unavailable or no paired files"
            aggregated_rows.append(
                {
                    "condition": "fixed_test_lr",
                    "sigma": "NA",
                    "method": method,
                    "psnr": None,
                    "ssim": None,
                    "lpips": None,
                    "num_images": 0,
                    "status": "not_run",
                    "notes": notes,
                }
            )

    write_per_image_csv(paths["per_image_csv"], per_image_rows)
    write_aggregate_csv(paths["matrix_csv"], aggregated_rows)
    write_markdown_matrix(paths["matrix_md"], aggregated_rows)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "data_root": settings.data_root,
            "blur_sigmas": settings.blur_sigmas,
            "scale_factor": settings.scale_factor,
            "checkpoint": settings.checkpoint,
            "device_requested": settings.device,
            "device_resolved": str(device),
            "num_steps": settings.num_steps,
            "ensemble_size": settings.ensemble_size,
            "max_images": settings.max_images,
            "disable_lpips": settings.disable_lpips,
            "force": settings.force,
            "deterministic_degradation": {
                "gaussian_filter_mode": "reflect",
                "resize_order": 3,
                "resize_mode": "reflect",
                "anti_aliasing": True,
                "preserve_range": True,
                "clip_to_0_1": True,
            },
            "lpips_rgb_mapping": "B04,B03,B02 from indices [2,1,0] scaled to [-1,1]",
            "range_notes": "Dataset arrays are treated as reflectance in [0,1].",
            "fixed_lr_notes": "fixed_test_lr condition uses existing dataset/test/LR files and is separate from sigma rows.",
        },
        "counts": {
            "gt_files": len(gt_files),
            "fixed_pairs": len(fixed_pairs),
            "per_image_rows": len(per_image_rows),
            "aggregate_rows": len(aggregated_rows),
        },
        "paths": paths,
    }
    with open(paths["meta_json"], "w") as f:
        json.dump(metadata, f, indent=2)

    return paths


def parse_args() -> EvalSettings:
    parser = argparse.ArgumentParser(description="OOD evaluation for bicubic and IHCNAS")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--blur_sigmas", type=float, nargs="+", default=[0.4, 0.8, 2.0])
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--disable_lpips", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite existing result files")
    args = parser.parse_args()

    if args.scale_factor < 1:
        raise ValueError("--scale_factor must be >= 1")
    if args.num_steps < 1:
        raise ValueError("--num_steps must be >= 1")
    if args.ensemble_size < 1:
        raise ValueError("--ensemble_size must be >= 1")
    if not args.blur_sigmas:
        raise ValueError("--blur_sigmas must contain at least one value")

    return EvalSettings(
        data_root=args.data_root,
        blur_sigmas=[float(v) for v in args.blur_sigmas],
        scale_factor=int(args.scale_factor),
        checkpoint=args.checkpoint,
        device=args.device,
        num_steps=int(args.num_steps),
        ensemble_size=int(args.ensemble_size),
        output_dir=args.output_dir,
        seed=int(args.seed),
        max_images=args.max_images,
        disable_lpips=bool(args.disable_lpips),
        force=bool(args.force),
    )


def main() -> None:
    settings = parse_args()
    paths = evaluate_dataset(settings)
    print("Wrote:")
    for key, value in paths.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
