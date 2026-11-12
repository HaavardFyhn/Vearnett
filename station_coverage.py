"""Display the saved notebook-method road status analysis without recalculating it."""
from functools import lru_cache
from contextlib import closing
from io import BytesIO
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds, transform as transform_coordinates


RELATIVE = Path("outputs") / "road_climate_coverage_new"
COLORS = ("#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027")
STATUSES = ("Godt representert", "Relativt god dekning", "Mulig dekningsmangel",
            "Interessant kandidatområde", "Svært interessant kandidatområde")


def save_status_raster(path, occurrences, road_mask, relevant, profile):
    lookup = np.zeros(int(occurrences.max()) + 1, dtype=np.uint8)
    lookup[relevant.ForekomstID.to_numpy()] = relevant.Kandidatklasse.to_numpy(dtype=np.uint8)
    values = lookup[occurrences]
    values[~road_mask] = 0
    with rasterio.open(path, "w", **dict(profile, driver="GTiff", count=1, dtype="uint8",
                                       nodata=0, compress="lzw")) as output:
        output.write(values, 1)


class CoverageLayer:
    def __init__(self, *roots):
        self.folder = next((Path(root) / RELATIVE for root in roots
                            if (Path(root) / RELATIVE / "status.tif").is_file()),
                           Path(roots[0]) / RELATIVE)
        self.path = self.folder / "status.tif"

    @lru_cache(maxsize=1)
    def metadata(self):
        if not self.path.is_file():
            return dict(available=False)
        summary = self.folder / "status_summary.json"
        if summary.is_file():
            return json.loads(summary.read_text(encoding="utf-8"))
        manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        with closing(sqlite3.connect((self.folder / "road_climate_coverage.gpkg").resolve().as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT Kandidatklasse,count(*),count(distinct ForekomstID),"
                              "sum(length_m)/1000 FROM scored_roads GROUP BY Kandidatklasse").fetchall()
        counts = {int(row[0]): row[1:] for row in rows}
        statuses = manifest.get("classification", {}).get("statuses", STATUSES)
        if (len(statuses) != 5 or
                any(not isinstance(status, str) or not status.strip() for status in statuses)):
            raise ValueError("Ugyldige statusnavn i dekningsanalysen")
        return dict(available=True, created=manifest["created"], stations=manifest["stations"],
            owners=manifest["owners"],
            classes=[dict(value=value, label=label, color=color,
                          lines=counts.get(value, (0, 0, 0))[0],
                          occurrences=counts.get(value, (0, 0, 0))[1],
                          length_km=counts.get(value, (0, 0, 0))[2])
                     for value, (label, color) in enumerate(zip(statuses, COLORS), 1)])

    @lru_cache(maxsize=1)
    def source(self):
        if not self.path.is_file():
            raise ValueError("Ingen lagret kandidatanalyse tilgjengelig")
        with rasterio.open(self.path) as source:
            return source.read(1), source.transform, source.crs

    @staticmethod
    def classes(values):
        if (not isinstance(values, list) or any(type(value) is not int or value not in range(1, 6)
                                               for value in values)):
            raise ValueError("Ugyldige kandidatklasser")
        return sorted(set(values))

    def image(self, bounds, width, height, classes):
        classes = self.classes(classes)
        if (not isinstance(bounds, list) or len(bounds) != 4 or
                not np.isfinite(bounds).all() or
                not (-180 <= bounds[0] < bounds[2] <= 180 and -85 <= bounds[1] < bounds[3] <= 85)):
            raise ValueError("Ugyldig kartutsnitt")
        if not 1 <= width <= 2000 or not 1 <= height <= 2000:
            raise ValueError("Ugyldig bildeoppløsning")
        values, transform, crs = self.source()
        filtered = np.where(np.isin(values, classes), values, 0).astype("uint8")
        projected = transform_bounds(4326, 3857, *bounds, densify_pts=21)
        target = np.zeros((height, width), dtype=np.uint8)
        reproject(filtered, target, src_transform=transform, src_crs=crs, src_nodata=0,
                  dst_transform=from_bounds(*projected, width, height), dst_crs=3857,
                  dst_nodata=0, resampling=Resampling.max)
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        for value in classes:
            color = COLORS[value - 1].lstrip("#")
            pixels[target == value] = [int(color[i:i + 2], 16) for i in (0, 2, 4)] + [255]
        output = BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        output.seek(0)
        return output

    def point(self, longitude, latitude, classes):
        classes = self.classes(classes)
        longitude, latitude = float(longitude), float(latitude)
        if not np.isfinite([longitude, latitude]).all() or not -180 <= longitude <= 180 or not -85 <= latitude <= 85:
            raise ValueError("Ugyldige koordinater")
        values, transform, crs = self.source()
        xs, ys = transform_coordinates(4326, crs, [longitude], [latitude])
        row, col = rasterio.transform.rowcol(transform, xs[0], ys[0])
        if not 0 <= row < values.shape[0] or not 0 <= col < values.shape[1]:
            return dict(found=False)
        value = int(values[row, col])
        return (dict(found=True, value=value, label=self.metadata()["classes"][value - 1]["label"],
                     color=COLORS[value - 1])
                if value in classes else dict(found=False))
