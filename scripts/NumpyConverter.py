import os
import sys
import glob
import argparse
import numpy as np
import rasterio


def read_tif_as_array(path):
    """Reads a rasterio-compatible TIFF file and returns it as a float32 numpy array."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Convert Sentinel-2 GeoTIFF patches to sequential NumPy arrays.")
    parser.add_argument('--data_root', type=str, default='dataset', help='Root directory containing the train folder')
    args = parser.parse_args()

    # Define input directories (where .tif files are located)
    gt_tif_dir = os.path.join(args.data_root, 'train', 'tif_GT')
    lr_tif_dir = os.path.join(args.data_root, 'train', 'tif_LR')

    # Define output directories (where .npy files will be saved)
    gt_npy_dir = os.path.join(args.data_root, 'train', 'GT')
    lr_npy_dir = os.path.join(args.data_root, 'train', 'LR')

    # Create output directories if they don't exist
    os.makedirs(gt_npy_dir, exist_ok=True)
    os.makedirs(lr_npy_dir, exist_ok=True)

    # Find all ground truth TIFF files
    gt_files = sorted(glob.glob(os.path.join(gt_tif_dir, '*.tif')))

    if not gt_files:
        print(f"Error: No .tif files found in '{gt_tif_dir}'. Please ensure buildData.py has been run first.")
        sys.exit(1)

    converted = 0
    skipped = 0
    saved_idx = 0  # Counter for sequential naming (0000.npy, 0001.npy, etc.)

    print(f"Found {len(gt_files)} Ground Truth TIF files. Starting sequential conversion...")

    for gt_path in gt_files:
        name = os.path.splitext(os.path.basename(gt_path))[0]
        lr_path = os.path.join(lr_tif_dir, f'{name}.tif')

        # Ensure corresponding LR file exists
        if not os.path.exists(lr_path):
            print(f"[{name}] Missing matching LR tif, skipping...")
            skipped += 1
            continue

        # Read images into numpy arrays
        gt_arr = read_tif_as_array(gt_path)
        lr_arr = read_tif_as_array(lr_path)

        # Format filename as 4-digit zero-padded index (e.g., 0000.npy)
        file_name = f"{saved_idx:04d}.npy"

        # Save as .npy files in sequential order
        np.save(os.path.join(gt_npy_dir, file_name), gt_arr)
        np.save(os.path.join(lr_npy_dir, file_name), lr_arr)

        print(f"[{name}] -> Saved as {file_name}")

        converted += 1
        saved_idx += 1

    print("\n--- Conversion Summary ---")
    print(f"Successfully converted: {converted} pairs")
    print(f"Skipped (missing LR): {skipped}")
    print(f"Saved sequential files in: {gt_npy_dir} & {lr_npy_dir}")


if __name__ == '__main__':
    main()