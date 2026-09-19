import os
import sys
import glob
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


def to_rgb(arr):
    rgb = arr[[2, 1, 0], :, :]
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    gt_dir = os.path.join(args.data_root, 'GT')
    lr_dir = os.path.join(args.data_root, 'LR')

    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.npy')))
    if not gt_files:
        print(f'no .npy files found in {gt_dir}')
        sys.exit(1)

    gt_path = random.choice(gt_files)
    name = os.path.basename(gt_path)
    lr_path = os.path.join(lr_dir, name)

    if not os.path.exists(lr_path):
        print(f'no matching LR file for {name}')
        sys.exit(1)

    gt_arr = np.load(gt_path)
    lr_arr = np.load(lr_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(to_rgb(gt_arr))
    axes[0].set_title(f'GT - {name}\nShape: {gt_arr.shape}')
    axes[0].axis('off')

    axes[1].imshow(to_rgb(lr_arr))
    axes[1].set_title(f'LR - {name}\nShape: {lr_arr.shape}')
    axes[1].axis('off')

    fig.suptitle(f'GT vs LR Comparison - {name}')
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
        print(f'saved figure to {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
