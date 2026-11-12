"""Shared terrain-window encoding for seasonal preparation and app inference."""
import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


def patch(source, x, y, resolution, elevation):
    half = 32*resolution
    window = from_bounds(x-half, y-half, x+half, y+half, source.transform)
    raw = source.read(1, window=window, out_shape=(64, 64), boundless=True, masked=True,
                      out_dtype="float32", resampling=Resampling.average)
    valid = ~np.ma.getmaskarray(raw) & np.isfinite(raw.data)
    height = np.where(valid, (raw.data.astype(float)-elevation)/1000, 0)
    encoded = height.astype(np.float16)
    if not np.isfinite(encoded).all():
        raise ValueError("Terrain cannot be represented as finite float16 kilometre offsets")
    return encoded, valid.astype(np.uint8)
