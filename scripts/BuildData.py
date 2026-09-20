import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

BANDS = ["B02", "B03", "B04", "B08"]

EVALSCRIPT = """//VERSION=3
function setup() {
    return {
        input: [{ bands: ["B02", "B03", "B04", "B08"], units: "DN" }],
        output: { bands: 4, sampleType: "UINT16" }
    };
}
function evaluatePixel(sample) {
    return [sample.B02, sample.B03, sample.B04, sample.B08];
}
"""

CLIENT_ID = "CLIENT_ID"
CLIENT_SECRET = "CLIENT_SECRET_ID"


class CDSEAuthSession:
    """Session that automatically refreshes the OAuth token before expiration."""

    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.expires_at = 0
        self.session = requests.Session()

    def get_token(self):
        if time.time() < self.expires_at - 60:
            return self.token

        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }
        res = requests.post(TOKEN_URL, data=data, timeout=30)
        if res.status_code != 200:
            raise RuntimeError(f"Token API error {res.status_code}: {res.text[:500]}")
        payload = res.json()
        self.token = payload["access_token"]
        self.expires_at = time.time() + payload.get("expires_in", 600)
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        return self.token

    def post(self, *args, **kwargs):
        self.get_token()
        return self.session.post(*args, **kwargs)


def search_best_scene(auth, aoi):
    west, south, east, north = aoi["bbox"]
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [west, south, east, north],
        "datetime": f'{aoi["start_date"]}T00:00:00Z/{aoi["end_date"]}T23:59:59Z',
        "limit": 100,
    }
    response = auth.post(CATALOG_URL, json=body, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Catalog API error {response.status_code}: {response.text[:500]}")

    features = response.json().get("features", [])
    if not features:
        raise RuntimeError("No Sentinel-2 L2A scene found in this date range.")

    def get_cloud(f):
        val = f.get("properties", {}).get("eo:cloud_cover")
        return float(val) if val is not None else 100.0

    # Sort strictly by cloud coverage and pick the clearest
    best = min(features, key=get_cloud)
    return best


def download_gt(auth, aoi, scene, out_path, gt_size):
    west, south, east, north = aoi["bbox"]
    gt_h, gt_w = gt_size

    # Target the single best day discovered by search_best_scene
    scene_datetime = scene["properties"]["datetime"]
    date_str = scene_datetime.split("T")[0]
    time_from = f"{date_str}T00:00:00Z"
    time_to = f"{date_str}T23:59:59Z"

    body = {
        "input": {
            "bounds": {
                "bbox": [west, south, east, north],
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": time_from, "to": time_to},
                    "mosaickingOrder": "mostRecent"
                },
            }],
        },
        "output": {
            "width": gt_w,
            "height": gt_h,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }

    response = auth.post(
        PROCESS_URL,
        headers={"Content-Type": "application/json", "Accept": "image/tiff"},
        json=body,
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Process API error {response.status_code}: {response.text[:500]}")

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            arr = src.read()
            transform = src.transform
            crs = src.crs

    # Verification: check if output is truly blank/nodata (all zeros)
    if np.all(arr == 0):
        raise RuntimeError("API returned completely empty (zero-filled) raster.")

    arr = arr[:4].astype(np.uint16, copy=False)

    profile = {
        "driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2],
        "count": 4, "dtype": "uint16", "crs": crs, "transform": transform,
        "compress": "deflate", "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(4):
            dst.write(arr[i], i + 1)
            dst.set_band_description(i + 1, BANDS[i])

    # Save an 8-bit visual check preview (RGB stretched)
    save_rgb_preview(arr, out_path.with_suffix(".png"))


def save_rgb_preview(arr_uint16, preview_path):
    """Saves an 8-bit RGB preview with reflectance visual stretch."""
    from PIL import Image
    # Band order in your arr: 0: B02 (Blue), 1: B03 (Green), 2: B04 (Red)
    rgb = np.stack([arr_uint16[2], arr_uint16[1], arr_uint16[0]], axis=-1).astype(np.float32)

    # 2.5x visual stretch on 0-10000 surface reflectance
    rgb = np.clip((rgb / 10000.0) * 2.5 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(preview_path)


def make_lr(gt_path, lr_path, lr_size):
    out_h, out_w = lr_size
    with rasterio.open(gt_path) as src:
        profile = src.profile.copy()
        profile.update(height=out_h, width=out_w)
        profile["transform"] = src.transform * src.transform.scale(
            src.width / out_w, src.height / out_h
        )
        with rasterio.open(lr_path, "w", **profile) as dst:
            for i in range(1, 5):
                # Use bilinear or average to avoid negative underflow artifacts
                data = src.read(i, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
                dst.write(data, i)
                dst.set_band_description(i, BANDS[i - 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi_config", type=str, required=True)
    parser.add_argument("--output_root", type=str, default=".")
    parser.add_argument("--gt_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--lr_size", type=int, nargs=2, default=[64, 64])
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    config_path = Path(args.aoi_config).resolve()
    gt_dir = root / "tiff_GT"
    lr_dir = root / "tiff_LR"
    failed_path = root / "configs" / "failed_aois.json"

    gt_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        aois = json.load(f)

    auth = CDSEAuthSession(CLIENT_ID, CLIENT_SECRET)

    processed = 0
    failed = []

    for index, aoi in enumerate(aois, start=1):
        name = aoi["name"]
        gt_path = gt_dir / f"{name}.tif"
        lr_path = lr_dir / f"{name}.tif"

        if gt_path.exists() and lr_path.exists():
            print(f"[{index}/{len(aois)}] {name}: already exists, skipping")
            continue

        print(f"[{index}/{len(aois)}] {name}")
        try:
            best_scene = search_best_scene(auth, aoi)
            scene_dt = best_scene["properties"]["datetime"]
            cloud = best_scene["properties"].get("eo:cloud_cover", "unknown")
            print(f"  Best scene found: {scene_dt} (cloud: {cloud}%)")

            download_gt(auth, aoi, best_scene, gt_path, tuple(args.gt_size))
            make_lr(gt_path, lr_path, tuple(args.lr_size))
            print(f"  Saved {gt_path.name} and {lr_path.name}")
            processed += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")
            if gt_path.exists():
                gt_path.unlink()
            if lr_path.exists():
                lr_path.unlink()
            failed.append({"name": name, "error": str(exc)})

    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
    print(f"Done! Processed: {processed}/{len(aois)}, Failed: {len(failed)}")


if __name__ == "__main__":
    main()
