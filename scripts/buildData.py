import os
import sys
import json
import argparse
import requests
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

CDSE_CLIENT_ID = "sh-664a8c40-bb9b-41a9-b4ab-26512350d3ee"
CDSE_CLIENT_SECRET = "pHul4nZB3REURPGRQG0TaGCvHcDyjoF1"




def get_access_token(client_id, client_secret):
    token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    resp = requests.post(token_url, data={
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials',
    })
    resp.raise_for_status()
    return resp.json()['access_token']


def fetch_process_api_patch(bbox, start_date, end_date, size, access_token):
    url = 'https://sh.dataspace.copernicus.eu/api/v1/process'

    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B02", "B03", "B04", "B08"],
            output: { bands: 4, sampleType: "UINT16" }
        }
    }
    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B08];
    }
    """

    payload = {
        "input": {
            "bounds": {
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date}T00:00:00Z",
                            "to": f"{end_date}T23:59:59Z"
                        },
                        "mosaickingOrder": "leastCC"
                    }
                }
            ]
        },
        "output": {
            "width": size,
            "height": size,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }
            ]
        },
        "evalscript": evalscript
    }

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'Accept': 'image/tiff'
    }

    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        print(f"Process API failed with status {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
        return None, None, None

    with MemoryFile(resp.content) as memfile:
        with memfile.open() as dataset:
            arr = dataset.read()  # Shape: (C, H, W)
            transform = dataset.transform
            crs = dataset.crs

    return arr, transform, crs


def degrade_to_lr(gt_patch, lr_size, blur_sigma=0.8, noise_std=0.003):
    C, H, W = gt_patch.shape
    lr = np.zeros((C, lr_size, lr_size), dtype=np.float32)
    for c in range(C):
        band = gaussian_filter(gt_patch[c], sigma=blur_sigma)
        band = resize(band, (lr_size, lr_size), order=3, anti_aliasing=True, preserve_range=True)
        lr[c] = band
    if noise_std > 0:
        lr = lr + np.random.normal(0.0, noise_std, size=lr.shape).astype(np.float32)
    return np.clip(lr, 0.0, 1.0)


def write_tif(path, arr, transform, crs):
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[1],
        'width': arr.shape[2],
        'count': arr.shape[0],
        'dtype': 'float32',
        'crs': crs,
        'transform': transform,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        for i in range(arr.shape[0]):
            dst.write(arr[i].astype('float32'), i + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aoi_config', type=str, required=True)
    parser.add_argument('--output_root', type=str, default='dataset')
    parser.add_argument('--gt_size', type=int, default=256)
    parser.add_argument('--lr_size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    # Updated paths to dataset/train/tif_GT and dataset/train/tif_LR
    gt_dir = os.path.join(args.output_root, 'train', 'tif_GT')
    lr_dir = os.path.join(args.output_root, 'train', 'tif_LR')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    with open(args.aoi_config, 'r') as f:
        aois = json.load(f)
    if args.limit > 0:
        aois = aois[:args.limit]

    if not CDSE_CLIENT_ID or CDSE_CLIENT_ID == "your_client_id_here":
        print('Please set your CDSE_CLIENT_ID and CDSE_CLIENT_SECRET directly at the top of the script.')
        sys.exit(1)

    print("Authenticating via Client Credentials...")
    access_token = get_access_token(CDSE_CLIENT_ID, CDSE_CLIENT_SECRET)

    done = 0
    for entry in aois:
        name = entry['name']
        bbox = entry['bbox']
        start_date = entry.get('start_date', '2024-06-01')
        end_date = entry.get('end_date', '2024-09-30')

        gt_path = os.path.join(gt_dir, f'{name}.tif')
        lr_path = os.path.join(lr_dir, f'{name}.tif')
        if os.path.exists(gt_path) and os.path.exists(lr_path):
            print(f'{name}: already exists, skipping')
            continue

        gt_arr, transform, crs = fetch_process_api_patch(bbox, start_date, end_date, args.gt_size, access_token)
        if gt_arr is None:
            print(f'{name}: failed to fetch from Process API, skipping')
            continue

        gt_arr = np.clip(gt_arr.astype(np.float32) / 10000.0, 0.0, 1.0)
        lr_arr = degrade_to_lr(gt_arr, args.lr_size)
        lr_transform = transform * transform.scale(args.gt_size / args.lr_size, args.gt_size / args.lr_size)

        write_tif(gt_path, gt_arr, transform, crs)
        write_tif(lr_path, lr_arr, lr_transform, crs)

        done += 1
        print(f'{name}: wrote {gt_path} and {lr_path} ({done}/{len(aois)})')

    print(f'completed {done} AOIs')


if __name__ == '__main__':
    main()