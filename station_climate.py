"""Categorical climate zones rendered directly from the native raster."""
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from pyproj import Transformer


COLORS = ("#8c6bb1", "#4292c6", "#331b9e", "#e6ab02",
          "#66c2a5", "#d73027", "#a6761d", "#e78ac3")


class ClimateZones:
    def __init__(self, path, cache):
        self.path, self.cache = Path(path), Path(cache)

    @lru_cache(maxsize=1)
    def metadata(self):
        with rasterio.open(self.path) as source:
            if source.crs is None or source.count != 1:
                raise ValueError("Klimasoner krever ett rasterbånd med koordinatsystem")
            tags = source.tags()
            names = json.loads(tags["sonenavn"])
            values = np.unique(source.read(1, masked=True).compressed())
            if any(str(int(value)) not in names for value in values):
                raise ValueError("Klimasoner mangler navn for en rasterklasse")
            if len(values) > len(COLORS):
                raise ValueError("For mange klimasoner for kartets kategoripalett")
            west, south, east, north = transform_bounds(source.crs, 4326, *source.bounds, densify_pts=41)
            return dict(label=tags.get("modellnavn", "Klimasoner"), bounds=[[south, west], [north, east]],
                resolution_m=abs(source.res[0]), method=tags.get("metode"),
                zones=[dict(value=int(value), name=names[str(int(value))], color=COLORS[index])
                       for index, value in enumerate(values)],
                statistics=json.loads(tags.get("sone_statistikk", "{}")))

    def transform(self, size):
        south, west = self.metadata()["bounds"][0]
        north, east = self.metadata()["bounds"][1]
        bounds = Transformer.from_crs(4326, 3857, always_xy=True).transform_bounds(west, south, east, north)
        width = max(1, round(size * (bounds[2]-bounds[0])/(bounds[3]-bounds[1])))
        return from_bounds(*bounds, width, size), width

    def image(self, size):
        if size not in (400, 800, 1600):
            raise ValueError("Ugyldig bildeoppløsning")
        stat = self.path.stat()
        key = hashlib.sha256(json.dumps([str(self.path.resolve()), stat.st_size, stat.st_mtime_ns,
            size, COLORS, "climate-png-v1"]).encode()).hexdigest()
        path = self.cache / f"{key}.png"
        if path.is_file():
            return BytesIO(path.read_bytes())
        transform, width = self.transform(size)
        values = np.zeros((size, width), dtype=np.uint8)
        with rasterio.open(self.path) as source:
            reproject(rasterio.band(source, 1), values, src_transform=source.transform,
                src_crs=source.crs, src_nodata=source.nodata, dst_transform=transform,
                dst_crs=3857, dst_nodata=0, resampling=Resampling.nearest)
        pixels = np.zeros((*values.shape, 4), dtype=np.uint8)
        for zone in self.metadata()["zones"]:
            color = zone["color"].lstrip("#")
            pixels[values == zone["value"]] = [int(color[i:i+2], 16) for i in (0, 2, 4)] + [255]
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        self.cache.mkdir(parents=True, exist_ok=True)
        import tempfile
        with tempfile.NamedTemporaryFile(dir=self.cache, suffix=".pending", delete=False) as handle:
            pending = Path(handle.name)
            handle.write(output.getvalue())
        try:
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        output.seek(0)
        return output

    def clip(self, image, polygons, size):
        with Image.open(image) as source:
            pixels = np.asarray(source.convert("RGBA")).copy()
        transform, _ = self.transform(size)
        geometries = [geometry for geometry in polygons.to_crs(3857).geometry if not geometry.is_empty]
        mask = (rasterize([(geometry, 1) for geometry in geometries], out_shape=pixels.shape[:2],
                         transform=transform, dtype="uint8") if geometries
                else np.zeros(pixels.shape[:2], dtype=np.uint8))
        pixels[:, :, 3] *= mask
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        output.seek(0)
        return output

    def point(self, lon, lat):
        if not np.isfinite([lon, lat]).all() or not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError("Ugyldige koordinater")
        with rasterio.open(self.path) as source:
            x, y = Transformer.from_crs(4326, source.crs, always_xy=True).transform(lon, lat)
            row, col = source.index(x, y)
            if not (0 <= row < source.height and 0 <= col < source.width):
                return dict(zone=None)
            value = int(source.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
        zone = next((zone for zone in self.metadata()["zones"] if zone["value"] == value), None)
        return dict(zone=zone, statistics=self.metadata()["statistics"].get(str(value), {}))
