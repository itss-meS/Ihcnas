import os
import sys
import json
import zipfile
import argparse
import tempfile
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


BAND_FILE_SUFFIX = {
    'B02': '_B02_10m.jp2',
    'B03': '_B03_10m.jp2',
    'B04': '_B04_10m.jp2',
    'B08': '_B08_10m.jp2',
}


def get_access_token(client_id, client_secret):
    import requests
    token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    resp = requests.post(token_url, data={
        'client_id': client_id,
        'client_secret': client_secret,
        'grant_type': 'client_credentials',
    })
    resp.raise_for_status()
    return resp.json()['access_token']


def search_product(bbox, start_date, end_date, access_token):
    import requests
    polygon = (
        f'POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},'
        f'{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))'
    )
    search_url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
    filter_str = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T00:00:00.000Z and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
    )
    params = {'$filter': filter_str, '$top': 1, '$orderby': 'ContentDate/Start desc'}
    resp = requests.get(search_url, params=params, headers={'Authorization': f'Bearer {access_token}'})
    resp.raise_for_status()
    results = resp.json().get('value', [])
    if not results:
        return None
    return results[0]


def download_product_zip(product_id, access_token, dest_path):
    import requests
    download_url = f'https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value'
    with requests.get(download_url, headers={'Authorization': f'Bearer {access_token}'}, stream=True) as r:
        if r.status_code != 200:
            print(f'download failed with status {r.status_code}: {r.text[:500]}')
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def find_band_paths(extract_dir):
    band_paths = {}
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            for band, suffix in BAND_FILE_SUFFIX.items():
                if fname.endswith(suffix):
                    band_paths[band] = os.path.join(root, fname)
    return band_paths


def crop_bands_to_array(band_paths, bbox, out_size):
    arrays = []
    transform_out = None
    crs_out = None
    for band in ['B02', 'B03', 'B04', 'B08']:
        with rasterio.open(band_paths[band]) as src:
            crs_out = src.crs
            window_bbox = transform_bounds('EPSG:4326', src.crs, *bbox)
            window = from_bounds(*window_bbox, transform=src.transform)
            data = src.read(
                1,
                window=window,
                out_shape=(out_size, out_size),
                resampling=Resampling.bilinear,
            )
            if transform_out is None:
                base_transform = src.window_transform(window)
                scale_x = window.width / out_size
                scale_y = window.height / out_size
                transform_out = base_transform * base_transform.scale(scale_x, scale_y)
            arrays.append(data)
    stacked = np.stack(arrays, axis=0)
    return stacked, transform_out, crs_out


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

    gt_dir = os.path.join(args.output_root, 'tiff_gt')
    lr_dir = os.path.join(args.output_root, 'tiff_lr')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    with open(args.aoi_config, 'r') as f:
        aois = json.load(f)
    if args.limit > 0:
        aois = aois[:args.limit]

    client_id = os.environ.get('CDSE_CLIENT_ID')
    client_secret = os.environ.get('CDSE_CLIENT_SECRET')
    if not client_id or not client_secret:
        print('CDSE_CLIENT_ID / CDSE_CLIENT_SECRET environment variables are required')
        sys.exit(1)

    access_token = get_access_token(client_id, client_secret)

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

        access_token = get_access_token(client_id, client_secret)
        product = search_product(bbox, start_date, end_date, access_token)
        if product is None:
            print(f'{name}: no scene found, skipping')
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, 'product.zip')
            fresh_token = get_access_token(client_id, client_secret)
            download_product_zip(product['Id'], fresh_token, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmp_dir)

            band_paths = find_band_paths(tmp_dir)
            if len(band_paths) < 4:
                print(f'{name}: could not locate all 4 bands, skipping')
                continue

            gt_arr, transform, crs = crop_bands_to_array(band_paths, bbox, args.gt_size)
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
