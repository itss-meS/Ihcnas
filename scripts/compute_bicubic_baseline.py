import os
import glob
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_metrics(lr_dir, gt_dir, output_file):
    # Find all TIF files
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.tif')))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.tif')))
    
    if not lr_files or not gt_files:
        print(f"Error: No .tif files found. LR: {len(lr_files)}, GT: {len(gt_files)}.")
        print("Please run scripts/buildData.py first to generate the dataset.")
        return

    # Match files by exact name
    lr_dict = {os.path.basename(f): f for f in lr_files}
    gt_dict = {os.path.basename(f): f for f in gt_files}
    common_files = sorted(list(set(lr_dict.keys()).intersection(set(gt_dict.keys()))))
    
    if not common_files:
        print("Error: No matching LR and GT filenames found.")
        return

    results = {}
    psnr_list, ssim_list = [], []

    print(f"Starting evaluation on {len(common_files)} real test samples...")
    
    for filename in common_files:
        # Load TIF using rasterio (outputs Channel-first: C, H, W)
        with rasterio.open(lr_dict[filename]) as src_lr:
            lr = src_lr.read().astype(np.float32)
        with rasterio.open(gt_dict[filename]) as src_gt:
            gt = src_gt.read().astype(np.float32)
            
        # Normalize to [0, 1] if data is uint16 (typical for Sentinel-2)
        if gt.max() > 2.0:
            lr = np.clip(lr / 10000.0, 0, 1)
            gt = np.clip(gt / 10000.0, 0, 1)

        if lr.shape[0] != 4:
            print(f"Skipping {filename}: Does not have 4 bands (Found {lr.shape[0]}).")
            continue

        # Bicubic Upsampling (4x)
        lr_tensor = torch.from_numpy(lr).unsqueeze(0)
        pred_tensor = F.interpolate(lr_tensor, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
        pred = pred_tensor.squeeze(0).numpy()

        if pred.shape != gt.shape:
            print(f"Skipping {filename}: Shape mismatch after upsampling. Pred {pred.shape} vs GT {gt.shape}")
            continue

        # Convert to (H, W, C) for scikit-image metrics
        gt_hwc = np.moveaxis(gt, 0, -1)
        pred_hwc = np.moveaxis(pred, 0, -1)

        psnr_val = peak_signal_noise_ratio(gt_hwc, pred_hwc, data_range=1.0)
        ssim_val = structural_similarity(gt_hwc, pred_hwc, data_range=1.0, channel_axis=-1)
        
        results[filename] = {"PSNR": float(psnr_val), "SSIM": float(ssim_val)}
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    if not psnr_list:
        print("No valid samples were evaluated. Check your data formats.")
        return

    summary = {
        "Method": "Bicubic (4x)",
        "Valid Samples": len(psnr_list),
        "Average PSNR (dB)": float(np.mean(psnr_list)),
        "Average SSIM": float(np.mean(ssim_list))
    }

    # Make docs dir if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({"summary": summary, "per_sample": results}, f, indent=4)
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Total Samples: {summary['Valid Samples']}")
    print(f"Average PSNR:  {summary['Average PSNR (dB)']:.2f} dB")
    print(f"Average SSIM:  {summary['Average SSIM']:.4f}")
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_dir', type=str, required=True, help='Path to test LR .tif files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to test GT .tif files')
    parser.add_argument('--output', type=str, default='docs/bicubic_baseline.json')
    args = parser.parse_args()
    
    compute_metrics(args.lr_dir, args.gt_dir, args.output)