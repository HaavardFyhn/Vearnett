"""County geometry shared by display, road selection and vector exports."""
from functools import lru_cache
from io import BytesIO
import json
import threading

import geopandas as gpd
import numpy as np
from PIL import Image
import pyogrio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely import box, intersects, line_merge, prepare, union_all


class CountyMask:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.frame = None
        self.cell_masks = {}

    def load(self):
        with self.lock:
            if self.frame is None:
                frame = pyogrio.read_dataframe(self.path, layer="fylke",
                                              columns=["fylkesnummer", "fylkesnavn"]).to_crs(25833)
                frame["code"] = frame.fylkesnummer.astype(str)
                if frame.empty or frame.code.duplicated().any() or frame.geometry.isna().any():
                    raise ValueError("Ugyldige fylkespolygoner")
                self.frame = frame.set_index("code")
        return self.frame

    def options(self):
        frame = self.load()
        return sorted([dict(code=str(code), name=str(row.fylkesnavn))
                       for code, row in frame.iterrows()], key=lambda row: row["name"])

    @lru_cache(maxsize=1)
    def borders(self):
        # Merge shared edges before display-only simplification, avoiding doubled borders.
        lines = line_merge(union_all(self.load().geometry.boundary.to_numpy())).simplify(100)
        frame = gpd.GeoDataFrame(geometry=[lines], crs=25833).to_crs(4326)
        return json.loads(frame.to_json(drop_id=True))

    def selection(self, codes):
        frame = self.load()
        if codes is None:
            return None
        if not isinstance(codes, (list, tuple)) or not all(isinstance(code, str) for code in codes):
            raise ValueError("Fylker må være en liste av fylkesnummer")
        if not set(codes).issubset(frame.index):
            raise ValueError("Ukjent fylkesnummer")
        return None if set(codes) == set(frame.index) else tuple(sorted(set(codes)))

    @lru_cache(maxsize=32)
    def geometry(self, codes):
        geometry = union_all(self.load().loc[list(codes)].geometry.to_numpy())
        prepare(geometry)
        return geometry

    def polygons(self, codes):
        codes = self.selection(codes)
        if codes is None:
            return None
        return gpd.GeoDataFrame(geometry=[self.geometry(codes)], crs=25833)

    def cells(self, engine, codes):
        codes = self.selection(codes)
        if codes is None:
            return None
        key = (id(engine.grid), engine.cell_m, codes)
        with self.lock:
            cached = self.cell_masks.get(key)
            if cached is not None and cached[0] is engine.grid:
                return cached[1]
        half = engine.cell_m/2
        x, y = engine.grid.x.to_numpy(), engine.grid.y.to_numpy()
        mask = intersects(self.geometry(codes), box(x-half, y-half, x+half, y+half))
        with self.lock:
            if len(self.cell_masks) >= 16:
                self.cell_masks.pop(next(iter(self.cell_masks)))
            self.cell_masks[key] = (engine.grid, mask)
        return mask

    def clip_image(self, image, bounds, codes):
        codes = self.selection(codes)
        if codes is None:
            return image
        polygon = self.polygons(codes).to_crs(3857).geometry.iloc[0]
        from pyproj import Transformer
        west, south, east, north = bounds
        left, bottom, right, top = Transformer.from_crs(4326, 3857, always_xy=True).transform_bounds(
            west, south, east, north)
        with Image.open(image) as source:
            rgba = np.asarray(source.convert("RGBA")).copy()
        mask = np.zeros(rgba.shape[:2], np.uint8)
        if not polygon.is_empty:
            mask = rasterize([(polygon, 1)], out_shape=mask.shape,
                             transform=from_bounds(left, bottom, right, top, mask.shape[1], mask.shape[0]), dtype="uint8")
        rgba[:, :, 3] *= mask
        output = BytesIO()
        Image.fromarray(rgba).save(output, format="PNG")
        output.seek(0)
        return output
