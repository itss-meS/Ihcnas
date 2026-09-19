import os
import sys
import glob
import argparse
import numpy as np
import rasterio


def read_tif_as_array(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset')
    args = parser.parse_args()

    gt_tif_dir = os.path.join(args.data_root, 'tiff_gt')
    lr_tif_dir = os.path.join(args.data_root, 'tiff_lr')
    gt_npy_dir = os.path.join(args.data_root, 'GT')
    lr_npy_dir = os.path.join(args.data_root, 'LR')
    os.makedirs(gt_npy_dir, exist_ok=True)
    os.makedirs(lr_npy_dir, exist_ok=True)

    gt_files = sorted(glob.glob(os.path.join(gt_tif_dir, '*.tif')))
    converted = 0
    skipped = 0

    for gt_path in gt_files:
        name = os.path.splitext(os.path.basename(gt_path))[0]
        lr_path = os.path.join(lr_tif_dir, f'{name}.tif')
        if not os.path.exists(lr_path):
            print(f'{name}: missing matching LR tif, skipping')
            skipped += 1
            continue

        gt_arr = read_tif_as_array(gt_path)
        lr_arr = read_tif_as_array(lr_path)

        np.save(os.path.join(gt_npy_dir, f'{name}.npy'), gt_arr)
        np.save(os.path.join(lr_npy_dir, f'{name}.npy'), lr_arr)
        converted += 1
        print(f'{name}: converted')

    print(f'converted {converted} pairs, skipped {skipped}')


if __name__ == '__main__':
    main()
