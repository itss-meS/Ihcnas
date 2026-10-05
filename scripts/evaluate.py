import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from rasterio.transform import Affine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.satsrflow import SatSRFlow
from src.utils.geo_io import read_geotiff, write_geotiff, rescale_transform
from src.metrics.metrics import compute_all_metrics


def build_model(cfg, device):
    model = SatSRFlow(
        in_ch=4,
        out_ch=4,
        latent_dim=cfg.get('latent_dim', 64),
        cond_dim=cfg.get('cond_dim', 128),
        hidden_dim=cfg.get('hidden_dim', 128),
        flow_depth=cfg.get('flow_depth', 6),
        hat_depth=cfg.get('hat_depth', 4),
        scale_factor=cfg.get('scale_factor', 4),
    ).to(device)
    return model


def save_visual_preview(lr_img, sr_img, output_path):
    def to_uint8_rgb(arr):
        rgb = arr[[2, 1, 0], :, :]
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = (rgb * 255.0).astype(np.uint8)
        return np.transpose(rgb, (1, 2, 0))

    lr_rgb = to_uint8_rgb(lr_img)
    sr_rgb = to_uint8_rgb(sr_img)

    lr_pil = Image.fromarray(lr_rgb).resize((sr_rgb.shape[1], sr_rgb.shape[0]), Image.NEAREST)
    sr_pil = Image.fromarray(sr_rgb)

    separator_width = 4
    separator_x = lr_pil.width + separator_width // 2

    combined = Image.new('RGB', (lr_pil.width + separator_width + sr_pil.width, sr_pil.height), 'white')
    combined.paste(lr_pil, (0, 0))
    combined.paste(sr_pil, (lr_pil.width + separator_width, 0))

    draw = ImageDraw.Draw(combined)
    draw.line((separator_x, 0, separator_x, combined.height), fill=(80, 80, 80), width=1)
    combined.save(output_path)


def run_single_scene(model, input_path, output_dir, scale, num_steps, device, gt_path=None):
    os.makedirs(output_dir, exist_ok=True)
    arr, profile, transform, crs = read_geotiff(input_path)

    if arr.max() > 1.5:
        arr = np.clip(arr.astype(np.float32) / 10000.0, 0.0, 1.0)

    lr_tensor = torch.from_numpy(arr).unsqueeze(0).float().to(device)
    H_in, W_in = arr.shape[-2:]
    H_out, W_out = H_in * scale, W_in * scale

    mean_sr, uncertainty_map = model.infer(lr_tensor, (H_out, W_out), num_steps=num_steps, ensemble_size=3)

    sr_np = mean_sr.squeeze(0).cpu().numpy()
    unc_np = uncertainty_map.squeeze(0).squeeze(0).cpu().numpy()

    out_transform = rescale_transform(transform, scale)

    sr_path = os.path.join(output_dir, 'ihcnaS_super_resolved_2.5m.tif')
    write_geotiff(sr_path, sr_np, out_transform, crs)

    unc_path = os.path.join(output_dir, 'ihcnaS_uncertainty_map.tif')
    write_geotiff(unc_path, unc_np, out_transform, crs)

    red = sr_np[2]
    nir = sr_np[3]
    ndvi_arr = (nir - red) / (nir + red + 1e-6)
    ndvi_path = os.path.join(output_dir, 'ihcnaS_ndvi_2.5m.tif')
    write_geotiff(ndvi_path, ndvi_arr, out_transform, crs)

    preview_path = os.path.join(output_dir, 'ihcnaS_visual_preview.png')
    save_visual_preview(arr, sr_np, preview_path)

    metrics_report = {
        'scale_factor': scale,
        'thresholds': {
            'psnr_db': 32.0,
            'ssim': 0.88,
            'sam_deg': 2.8,
            'ergas': 2.5,
            'ndvi_mae': 0.025,
        },
    }

    if gt_path is not None and os.path.exists(gt_path):
        gt_arr, _, _, _ = read_geotiff(gt_path)
        if gt_arr.max() > 1.5:
            gt_arr = np.clip(gt_arr.astype(np.float32) / 10000.0, 0.0, 1.0)
        gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0).float().to(device)
        metrics = compute_all_metrics(mean_sr, gt_tensor, scale)
        metrics_report['metrics'] = metrics
        metrics_report['pass'] = {
            'psnr': metrics['psnr'] > 32.0,
            'ssim': metrics['ssim'] > 0.88,
            'sam': metrics['sam'] < 2.8,
            'ergas': metrics['ergas'] < 2.5,
            'ndvi_mae': metrics['ndvi_mae'] < 0.025,
        }

    metrics_path = os.path.join(output_dir, 'ihcnaS_metrics_report.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_report, f, indent=2)

    return sr_path, unc_path, ndvi_path, preview_path, metrics_path


def load_source_scene_map(dataset_root):
    manifest_path = os.path.join(dataset_root, 'manifest.json')
    mapping = {}
    if not os.path.exists(manifest_path):
        return mapping
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    for entry in manifest.get('test', []):
        source_scene = entry.get('source_scene', entry['filename'])
        mapping[entry['filename']] = os.path.splitext(source_scene)[0]
    return mapping


def unique_folder_name(base_name, used_names):
    name = base_name
    suffix = 1
    while name in used_names:
        suffix += 1
        name = f'{base_name}_{suffix}'
    used_names.add(name)
    return name


def run_npy_test_batch(model, test_dir, output_dir, scale, num_steps, device):
    lr_dir = os.path.join(test_dir, 'LR')
    gt_dir = os.path.join(test_dir, 'GT')
    dataset_root = os.path.dirname(test_dir.rstrip('/'))
    source_scene_map = load_source_scene_map(dataset_root)

    filenames = sorted(f for f in os.listdir(lr_dir) if f.endswith('.npy'))
    os.makedirs(output_dir, exist_ok=True)
    used_names = set()
    dummy_transform = Affine(2.5, 0.0, 0.0, 0.0, -2.5, 0.0)
    dummy_crs = 'EPSG:4326'
    summary = []

    for filename in filenames:
        lr_path = os.path.join(lr_dir, filename)
        gt_path = os.path.join(gt_dir, filename)
        if not os.path.exists(gt_path):
            print(f'skipping {filename}: no matching GT file')
            continue

        lr_np = np.load(lr_path).astype(np.float32)
        gt_np = np.load(gt_path).astype(np.float32)

        lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).to(device)

        mean_sr, uncertainty_map = model.infer(
            lr_tensor, gt_np.shape[-2:], num_steps=num_steps, ensemble_size=3
        )

        sr_np = mean_sr.squeeze(0).cpu().numpy()
        unc_np = uncertainty_map.squeeze(0).squeeze(0).cpu().numpy()

        base_name = source_scene_map.get(filename, os.path.splitext(filename)[0])
        folder_name = unique_folder_name(base_name, used_names)
        aoi_out_dir = os.path.join(output_dir, folder_name)
        os.makedirs(aoi_out_dir, exist_ok=True)

        np.save(os.path.join(aoi_out_dir, 'ihcnaS_input_lr.npy'), lr_np)

        write_geotiff(os.path.join(aoi_out_dir, 'ihcnaS_super_resolved_2.5m.tif'), sr_np, dummy_transform, dummy_crs)
        write_geotiff(os.path.join(aoi_out_dir, 'ihcnaS_uncertainty_map.tif'), unc_np, dummy_transform, dummy_crs)

        red = sr_np[2]
        nir = sr_np[3]
        ndvi_arr = (nir - red) / (nir + red + 1e-6)
        write_geotiff(os.path.join(aoi_out_dir, 'ihcnaS_ndvi_2.5m.tif'), ndvi_arr, dummy_transform, dummy_crs)

        save_visual_preview(lr_np, sr_np, os.path.join(aoi_out_dir, 'ihcnaS_visual_preview.png'))

        metrics = compute_all_metrics(mean_sr, gt_tensor, scale)
        report = {
            'aoi': folder_name,
            'source_file': filename,
            'scale_factor': scale,
            'thresholds': {
                'psnr_db': 32.0,
                'ssim': 0.88,
                'sam_deg': 2.8,
                'ergas': 2.5,
                'ndvi_mae': 0.025,
            },
            'metrics': metrics,
            'pass': {
                'psnr': metrics['psnr'] > 32.0,
                'ssim': metrics['ssim'] > 0.88,
                'sam': metrics['sam'] < 2.8,
                'ergas': metrics['ergas'] < 2.5,
                'ndvi_mae': metrics['ndvi_mae'] < 0.025,
            },
        }
        with open(os.path.join(aoi_out_dir, 'ihcnaS_metrics_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        summary.append({'aoi': folder_name, 'source_file': filename, **metrics})
        print(f'{folder_name}: psnr={metrics["psnr"]:.3f} ssim={metrics["ssim"]:.3f} sam={metrics["sam"]:.3f} ergas={metrics["ergas"]:.3f} ndvi_mae={metrics["ndvi_mae"]:.4f}')

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'processed {len(summary)} test pairs')


def run_single_npy(model, lr_npy_path, output_dir, scale, num_steps, device, gt_npy_path=None):
    os.makedirs(output_dir, exist_ok=True)
    lr_np = np.load(lr_npy_path).astype(np.float32)
    lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).to(device)

    dummy_transform = Affine(2.5, 0.0, 0.0, 0.0, -2.5, 0.0)
    dummy_crs = 'EPSG:4326'

    if gt_npy_path is not None and os.path.exists(gt_npy_path):
        gt_np = np.load(gt_npy_path).astype(np.float32)
        target_shape = gt_np.shape[-2:]
    else:
        gt_np = None
        H_in, W_in = lr_np.shape[-2:]
        target_shape = (H_in * scale, W_in * scale)

    with torch.no_grad():
        mean_sr, uncertainty_map = model.infer(lr_tensor, target_shape, num_steps=num_steps, ensemble_size=3)

    sr_np = mean_sr.squeeze(0).cpu().numpy()
    unc_np = uncertainty_map.squeeze(0).squeeze(0).cpu().numpy()

    write_geotiff(os.path.join(output_dir, 'ihcnaS_super_resolved_2.5m.tif'), sr_np, dummy_transform, dummy_crs)
    write_geotiff(os.path.join(output_dir, 'ihcnaS_uncertainty_map.tif'), unc_np, dummy_transform, dummy_crs)

    red = sr_np[2]
    nir = sr_np[3]
    ndvi_arr = (nir - red) / (nir + red + 1e-6)
    write_geotiff(os.path.join(output_dir, 'ihcnaS_ndvi_2.5m.tif'), ndvi_arr, dummy_transform, dummy_crs)

    save_visual_preview(lr_np, sr_np, os.path.join(output_dir, 'ihcnaS_visual_preview.png'))

    report = {
        'scale_factor': scale,
        'thresholds': {
            'psnr_db': 32.0,
            'ssim': 0.88,
            'sam_deg': 2.8,
            'ergas': 2.5,
            'ndvi_mae': 0.025,
        },
        'ground_truth_available': gt_np is not None,
    }

    if gt_np is not None:
        gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).to(device)
        metrics = compute_all_metrics(mean_sr, gt_tensor, scale)
        report['metrics'] = metrics
        report['pass'] = {
            'psnr': metrics['psnr'] > 32.0,
            'ssim': metrics['ssim'] > 0.88,
            'sam': metrics['sam'] < 2.8,
            'ergas': metrics['ergas'] < 2.5,
            'ndvi_mae': metrics['ndvi_mae'] < 0.025,
        }
        print(f'psnr={metrics["psnr"]:.3f} ssim={metrics["ssim"]:.3f} sam={metrics["sam"]:.3f} ergas={metrics["ergas"]:.3f} ndvi_mae={metrics["ndvi_mae"]:.4f}')
    else:
        print('no ground truth provided, metrics skipped')

    with open(os.path.join(output_dir, 'ihcnaS_metrics_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    return report.get('metrics', None)


def run_npy_folder(model, lr_dir, output_dir, scale, num_steps, device, gt_dir=None):
    filenames = sorted(f for f in os.listdir(lr_dir) if f.endswith('.npy'))
    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for index, filename in enumerate(filenames, start=1):
        name = os.path.splitext(filename)[0]
        lr_path = os.path.join(lr_dir, filename)
        gt_path = os.path.join(gt_dir, filename) if gt_dir is not None else None

        print(f'[{index}/{len(filenames)}] {name}')
        aoi_out_dir = os.path.join(output_dir, name)
        metrics = run_single_npy(model, lr_path, aoi_out_dir, scale, num_steps, device, gt_path)
        if metrics is not None:
            summary.append({'name': name, **metrics})

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'processed {len(filenames)} LR files, {len(summary)} had ground truth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--gt_dir', type=str, default=None)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {}) or {}

    model = build_model(cfg, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    is_npy_test_dir = os.path.isdir(args.input) and os.path.isdir(os.path.join(args.input, 'LR')) and os.path.isdir(os.path.join(args.input, 'GT'))
    is_single_npy = os.path.isfile(args.input) and args.input.lower().endswith('.npy')
    is_npy_folder = os.path.isdir(args.input) and not is_npy_test_dir and any(f.endswith('.npy') for f in os.listdir(args.input))

    if is_npy_test_dir:
        with torch.no_grad():
            run_npy_test_batch(model, args.input, args.output_dir, args.scale, args.num_steps, device)
    elif is_single_npy:
        with torch.no_grad():
            run_single_npy(model, args.input, args.output_dir, args.scale, args.num_steps, device, args.gt)
    elif is_npy_folder:
        with torch.no_grad():
            run_npy_folder(model, args.input, args.output_dir, args.scale, args.num_steps, device, args.gt_dir)
    else:
        run_single_scene(model, args.input, args.output_dir, args.scale, args.num_steps, device, args.gt)

    print(f'wrote outputs to {args.output_dir}')


if __name__ == '__main__':
    main()