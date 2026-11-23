"""Cached, non-interactive road tiles independent of scenario calculations."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pyogrio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import binary_dilation
import shapely

from station_roads import normalize_reference, road_categories, road_references


ROAD_BACKGROUND = Path("data") / "komprimert_nettverk_med_blindveg_10.gpkg"


def _raster_shapes(geometries):
    parts = shapely.get_parts(geometries)
    if len(parts) and np.all(shapely.get_type_id(parts) == shapely.GeometryType.LINESTRING):
        _, coordinates, (offsets,) = shapely.to_ragged_array(parts)
        return [({"type": "MultiLineString", "coordinates": [
            coordinates[start:stop].tolist() for start, stop in zip(offsets[:-1], offsets[1:])
        ]}, 1)]
    return [(geometry, 1) for geometry in geometries]


class RoadBackground:
    def __init__(self, path, counties=None):
        self.path = Path(path)
        self.counties = counties
        self.lock = threading.Lock()
        self.tree = None

    def prepare(self):
        with self.lock:
            if self.tree is not None:
                return
            frame = pyogrio.read_dataframe(
                self.path, columns=["roadid"], force_2d=True,
                where="roadid LIKE '%E%' OR roadid LIKE '%R%' OR roadid LIKE '%F%'")
            if frame.crs is None or frame.crs.to_epsg() != 25833:
                raise ValueError("Bakgrunnsvegnettet må være i EPSG:25833")
            masks = {value: sum({"E": 1, "R": 2, "F": 4}.get(code, 0)
                                for code in road_categories(value)) for value in frame.roadid.unique()}
            selected = frame.roadid.map(masks) > 0
            frame = frame.loc[selected & frame.geometry.notna() & ~frame.geometry.is_empty]
            geometries = frame.geometry.to_numpy()
            references = {}
            for value in frame.roadid.unique():
                normalized = set()
                for reference in road_references(value):
                    try:
                        normalized.add(normalize_reference(reference))
                    except ValueError:
                        continue
                references[value] = normalized
            self.category_masks = frame.roadid.map(masks).to_numpy(dtype=np.uint8)
            self.references = {}
            for index, value in enumerate(frame.roadid):
                for reference in references[value]:
                    self.references.setdefault(reference, []).append(index)
            transform = Transformer.from_crs(25833, 3857, always_xy=True)
            self.geometries = shapely.transform(
                geometries, lambda coordinates: np.column_stack(
                    transform.transform(coordinates[:, 0], coordinates[:, 1])))
            self.tree = shapely.STRtree(self.geometries)

    @staticmethod
    def filters(data):
        categories = data.get("categories", ["E", "R", "F"])
        if not isinstance(categories, list) or not all(isinstance(code, str) for code in categories):
            raise ValueError("Vegkategorier må være en liste")
        if not set(categories) <= {"E", "R", "F"}:
            raise ValueError("Bakgrunnsvegnettet støtter bare E-, R- og F-veger")
        reference = data.get("reference", "")
        if not isinstance(reference, str) or len(reference) > 30:
            raise ValueError("Ugyldig vegnavn")
        return tuple(sorted(set(categories))), normalize_reference(reference) if reference.strip() else ""

    @lru_cache(maxsize=32)
    def county_geometry(self, codes):
        if self.counties is None:
            raise ValueError("Fylkesdata mangler for bakgrunnsvegnettet")
        transform = Transformer.from_crs(25833, 3857, always_xy=True)
        return shapely.transform(
            self.counties.geometry(codes), lambda coordinates: np.column_stack(
                transform.transform(coordinates[:, 0], coordinates[:, 1])))

    @lru_cache(maxsize=512)
    def tile(self, z, x, y, categories=("E", "F", "R"), reference="", counties=None):
        if not 0 <= z <= 18 or not 0 <= x < 2 ** z or not 0 <= y < 2 ** z:
            raise ValueError("Ugyldig vegkartflis")
        self.prepare()
        extent = 20037508.342789244
        span = 2 * extent / 2 ** z
        west, north = -extent + x * span, extent - y * span
        pixel = span / 512
        window = shapely.box(west - pixel * 2, north - span - pixel * 2,
                             west + span + pixel * 2, north + pixel * 2)
        indices = self.tree.query(window)
        category_mask = sum({"E": 1, "R": 2, "F": 4}[code] for code in categories)
        indices = indices[(self.category_masks[indices] & category_mask) != 0]
        if reference:
            indices = indices[np.isin(indices, self.references.get(reference, []))]
        shapes = self.geometries[indices]
        shapes = shapely.simplify(shapes, pixel / 2)
        simple = shapely.is_simple(shapes)
        clipped = shapely.clip_by_rect(shapes, *window.bounds)
        if not simple.all():
            # Preserve intersection noding for self-crossing roads.
            clipped[~simple] = shapely.intersection(shapes[~simple], window)
        shapes = clipped
        if counties is not None:
            shapes = shapely.intersection(shapes, self.county_geometry(counties))
        shapes = shapes[~shapely.is_empty(shapes)]
        mask = (rasterize(_raster_shapes(shapes), out_shape=(516, 516),
                          transform=from_bounds(west - pixel * 2, north - span - pixel * 2,
                                                west + span + pixel * 2, north + pixel * 2,
                                                516, 516), dtype="uint8")
                if len(shapes) else np.zeros((516, 516), dtype=np.uint8))
        pixels = np.zeros((512, 512, 4), dtype=np.uint8)
        pixels[:, :, 3] = binary_dilation(mask)[2:-2, 2:-2].astype(np.uint8) * 230
        image = Image.fromarray(pixels).resize((256, 256), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
