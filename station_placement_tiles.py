"""Bounded caches for fixed placement-gain rasters and XYZ map tiles."""
from collections import OrderedDict
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window, from_bounds, transform

from file_hash import code_sha256


def tile_bounds(z, x, y):
    if not 0 <= z <= 18 or not 0 <= x < 2 ** z or not 0 <= y < 2 ** z:
        raise ValueError("Ugyldig plasseringskartflis")
    extent = 20037508.342789244
    span = 2 * extent / 2 ** z
    west, north = -extent + x * span, extent - y * span
    return list(Transformer.from_crs(3857, 4326, always_xy=True).transform_bounds(
        west, north - span, west + span, north))


def raster_payload(pixels, values):
    height, width = values.shape
    header = np.asarray([width, height], dtype="<u4").tobytes()
    values = np.where(pixels[:, :, 3], values, np.nan).astype("<f4")
    # Dimensions, float32 values, then original RGBA colours; clients only change alpha.
    return gzip.compress(header + values.tobytes() + pixels.tobytes(), compresslevel=1, mtime=0)


class PlacementTiles:
    def __init__(self, folder=None):
        self.folder = None if folder is None else Path(folder)
        self.lock = threading.Lock()
        self.images = OrderedDict()
        self.source_key = None
        self.source = None
        self.renderer = code_sha256(Path(__file__), Path(__file__).with_name("station_app.py"))

    def version(self, active, layer):
        signature = [self.renderer, layer["source"], layer.get("source_version"),
                     layer["minimum"], layer["maximum"], layer.get("reduction_max", 2), active.cell_m,
                     active.raster_shape, tuple(active.transform)]
        coast = getattr(active, "norway_path", None)
        if coast is not None:
            stat = Path(coast).stat()
            signature.extend([str(coast), stat.st_size, stat.st_mtime_ns])
        return hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:24]

    def image(self, active, layer, version, z, x, y, cutoff, settings, counties, road_mask, clip, numeric=False,
              map_scale="log", render_mode="placement"):
        bounds = tile_bounds(z, x, y)
        source_key = (id(active), id(layer["values"]), version)
        key = (source_key, z, x, y, cutoff, numeric, map_scale, render_mode,
               json.dumps(settings, sort_keys=True) if settings is not None else None, counties)
        with self.lock:
            if key in self.images:
                self.images.move_to_end(key)
                return self.images[key], True
            suffix = "bin.gz" if numeric else "png"
            path = (self.folder / version / f"{z}-{x}-{y}.{suffix}" if self.folder is not None
                    and render_mode == "placement" and map_scale == "log" and z <= 6 and cutoff == 0 and
                    settings is None and counties is None else None)
            if path is not None and path.is_file():
                result = path.read_bytes()
                self.images[key] = result
                if len(self.images) > 512:
                    self.images.popitem(last=False)
                return result, True
            if self.source_key != source_key:
                source = np.full(active.raster_shape, -9999, dtype=np.float32)
                values = layer["values"]
                missing_value = (layer["maximum"] if render_mode in ("uncertainty", "baseline") else -9999)
                source[active.rows, active.columns] = (
                    missing_value if values is None else np.where(np.isfinite(values), values, missing_value)
                )
                self.source, self.source_key = source, source_key
            source = self.source
            south, west = active.bounds[0]
            north, east = active.bounds[1]
            clipped_bounds = [max(bounds[0], west), max(bounds[1], south),
                              min(bounds[2], east), min(bounds[3], north)]
            if clipped_bounds[0] >= clipped_bounds[2] or clipped_bounds[1] >= clipped_bounds[3]:
                row0 = col0 = row1 = col1 = 0
            else:
                native = Transformer.from_crs(4326, 25833, always_xy=True).transform_bounds(*clipped_bounds, densify_pts=41)
                window = from_bounds(*native, transform=active.transform)
                row0 = max(0, int(np.floor(window.row_off)) - 2)
                col0 = max(0, int(np.floor(window.col_off)) - 2)
                row1 = min(source.shape[0], int(np.ceil(window.row_off + window.height)) + 2)
                col1 = min(source.shape[1], int(np.ceil(window.col_off + window.width)) + 2)
            if row1 <= row0 or col1 <= col0:
                cropped = np.full((1, 1), -9999, dtype=np.float32)
                source_transform = active.transform
            else:
                cropped = source[row0:row1, col0:col1].copy()
                source_transform = transform(Window(col0, row0, col1-col0, row1-row0), active.transform)
                if cutoff > 0:
                    cropped[cropped <= cutoff] = -9999
                if settings is not None:
                    mask = road_mask(active, dict(settings, bounds=clipped_bounds))
                    indices = np.flatnonzero(mask)
                    rows, columns = active.rows[indices] - row0, active.columns[indices] - col0
                    inside = (rows >= 0) & (rows < cropped.shape[0]) & (columns >= 0) & (columns < cropped.shape[1])
                    allowed = np.zeros(cropped.shape, dtype=bool)
                    allowed[rows[inside], columns[inside]] = True
                    cropped[~allowed] = -9999
            options = dict(display_size=256, viewport=bounds, raster_source=(cropped, source_transform))
            if numeric:
                pixels, values = active._render_pixels(layer["values"], render_mode,
                                                       layer["minimum"], layer["maximum"],
                                                       layer.get("reduction_max", 2), map_scale=map_scale,
                                                       filter_missing=render_mode in ("uncertainty", "baseline"),
                                                       **options)
                if counties is not None:
                    image = BytesIO()
                    Image.fromarray(pixels).save(image, format="PNG")
                    image.seek(0)
                    with Image.open(clip(image, bounds)) as clipped:
                        pixels = np.asarray(clipped).copy()
                result = raster_payload(pixels, values)
            else:
                image = active.render(layer["values"], render_mode, layer["minimum"], layer["maximum"],
                                      layer.get("reduction_max", 2),
                                      map_scale=map_scale, **options)
                result = clip(image, bounds).getvalue()
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pending", delete=False) as handle:
                    pending = Path(handle.name)
                    handle.write(result)
                try:
                    pending.replace(path)
                finally:
                    pending.unlink(missing_ok=True)
            self.images[key] = result
            if len(self.images) > 512:
                self.images.popitem(last=False)
            return result, False

    def warm(self, active, layer):
        version = self.version(active, layer)
        south, west = active.bounds[0]
        north, east = active.bounds[1]
        ready = 0
        for z in range(7):
            count = 2 ** z
            left = max(0, int(np.floor((west + 180) / 360 * count)))
            right = min(count - 1, int(np.floor((east + 180) / 360 * count)))
            top = max(0, int(np.floor((1 - np.arcsinh(np.tan(np.radians(north))) / np.pi) / 2 * count)))
            bottom = min(count - 1, int(np.floor((1 - np.arcsinh(np.tan(np.radians(south))) / np.pi) / 2 * count)))
            for x in range(left, right + 1):
                for y in range(top, bottom + 1):
                    self.image(active, layer, version, z, x, y, 0, None, None, None,
                               lambda image, bounds: image, numeric=True)
                    ready += 1
        return ready
