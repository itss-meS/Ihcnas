import numpy as np
import rasterio
from rasterio.transform import Affine


def read_geotiff(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return arr, profile, transform, crs


def write_geotiff(path, arr, transform, crs, dtype='float32'):
    if arr.ndim == 2:
        count = 1
        height, width = arr.shape
    else:
        count = arr.shape[0]
        height, width = arr.shape[-2:]

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': count,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        if arr.ndim == 2:
            dst.write(arr.astype(dtype), 1)
        else:
            for i in range(count):
                dst.write(arr[i].astype(dtype), i + 1)


def rescale_transform(transform, scale_factor):
    return Affine(
        transform.a / scale_factor,
        transform.b,
        transform.c,
        transform.d,
        transform.e / scale_factor,
        transform.f,
    )


def dn_to_reflectance(dn_array, scale=10000.0):
    return np.clip(dn_array.astype(np.float32) / scale, 0.0, 1.0)
