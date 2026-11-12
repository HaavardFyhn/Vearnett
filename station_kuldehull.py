"""Read potential-basin polygons as a display-only Kuldehull layer."""
from collections import OrderedDict
from io import BytesIO
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import threading

import geopandas as gpd
import numpy as np
from PIL import Image
import pyogrio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from shapely import from_wkb, to_wkb
from shapely.geometry import Point, box


DEFAULT_PATH = (Path(__file__).resolve().parent / "outputs"
                / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg")
DETAIL_LEVELS = (0, 25, 100, 500, 2000)
OVERVIEW_CELL_M = 1000
PROPERTY_FIELDS = ("lake_id", "max_depth_m", "mean_depth_m", "area_m2", "volume_m3",
                   "spill_level_m", "group_id", "member_basin_count")


class KuldehullLayer:
    def __init__(self, path, cache_dir=None):
        self.path = Path(path)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.path.parent / ".kuldehull_cache"
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.polygons = None
        self.signature = None
        self.detail_layers = {}
        self.overview = None
        self.overview_transform = None
        self.to_web = Transformer.from_crs(4326, 3857, always_xy=True)

    @staticmethod
    def depth(value):
        result = float(value)
        if not np.isfinite(result) or not 0 <= result <= 10000:
            raise ValueError("Kuldehull-dybde må være mellom 0 og 10000 meter")
        return result

    @property
    def metadata(self):
        return dict(available=self.path.is_file(), label="Kuldehull",
                    depth_field="max_depth_m", depth_unit="m")

    def prepare(self):
        """Build persistent display geometry offline, or load it and index it at startup."""
        if not self.path.is_file():
            return False
        stat = self.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with self.lock:
            if self.signature == signature and self.detail_layers:
                return True
            fingerprint = dict(schema="kuldehull-display-v2", path=str(self.path.resolve()),
                               mtime_ns=stat.st_mtime_ns, bytes=stat.st_size,
                               detail_levels=DETAIL_LEVELS)
            key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
            folder = self.cache_dir / key
            manifest = folder / "ready.json"
            if not manifest.is_file():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print("Forbereder Kuldehull: projeksjon og detaljnivåer...", flush=True)
                info = pyogrio.read_info(self.path, layer="potential_lakes")
                fields = [field for field in PROPERTY_FIELDS if field in info["fields"]]
                polygons = pyogrio.read_dataframe(
                    self.path, layer="potential_lakes", columns=fields,
                ).to_crs(3857)
                with tempfile.TemporaryDirectory(dir=self.cache_dir, prefix="prepare_") as temporary:
                    staging = Path(temporary) / "cache"
                    staging.mkdir()
                    for i, tolerance in enumerate(DETAIL_LEVELS):
                        geometry = (polygons.geometry if tolerance == 0 else
                                    polygons.geometry.simplify(tolerance, preserve_topology=True))
                        blobs = to_wkb(geometry.to_numpy())
                        lengths = np.asarray([len(blob) for blob in blobs], dtype=np.int64)
                        arrays = {f"field_{field}": polygons[field].to_numpy() for field in fields}
                        np.savez_compressed(
                            staging / f"detail_{tolerance}.npz",
                            wkb=np.frombuffer(b"".join(blobs), dtype=np.uint8),
                            offsets=np.concatenate(([0], np.cumsum(lengths))), **arrays,
                        )
                        print(f"Kuldehull: {i+1}/{len(DETAIL_LEVELS)}", flush=True)
                    left, bottom, right, top = polygons.total_bounds
                    left = np.floor(left / OVERVIEW_CELL_M) * OVERVIEW_CELL_M
                    bottom = np.floor(bottom / OVERVIEW_CELL_M) * OVERVIEW_CELL_M
                    right = np.ceil(right / OVERVIEW_CELL_M) * OVERVIEW_CELL_M
                    top = np.ceil(top / OVERVIEW_CELL_M) * OVERVIEW_CELL_M
                    width = int(round((right - left) / OVERVIEW_CELL_M))
                    height = int(round((top - bottom) / OVERVIEW_CELL_M))
                    ordered = polygons.sort_values("max_depth_m")
                    overview = rasterize(
                        ((geometry, depth) for geometry, depth in zip(ordered.geometry, ordered.max_depth_m)),
                        out_shape=(height, width), dtype="float32", fill=0,
                        transform=from_bounds(left, bottom, right, top, width, height),
                    )
                    np.savez_compressed(staging / "overview.npz", depth=overview,
                                        bounds=np.asarray([left, bottom, right, top]))
                    print("Kuldehull: landsoversikt lagret", flush=True)
                    (staging / "ready.json").write_text(json.dumps(fingerprint), encoding="utf-8")
                    try:
                        staging.rename(folder)
                    except OSError:
                        if not manifest.is_file():
                            raise
            prepared = {}
            for tolerance in DETAIL_LEVELS:
                with np.load(folder / f"detail_{tolerance}.npz", allow_pickle=False) as stored:
                    data = {field.removeprefix("field_"): stored[field]
                            for field in stored.files if field.startswith("field_")}
                    payload, offsets = stored["wkb"], stored["offsets"]
                    blobs = [payload[first:last].tobytes() for first, last in zip(offsets[:-1], offsets[1:])]
                    frame = gpd.GeoDataFrame(data, geometry=from_wkb(blobs), crs=3857)
                    frame.sindex
                    prepared[tolerance] = frame
            with np.load(folder / "overview.npz", allow_pickle=False) as stored:
                self.overview = stored["depth"]
                self.overview_transform = from_bounds(
                    *stored["bounds"], self.overview.shape[1], self.overview.shape[0],
                )
            self.polygons = prepared[0]
            self.detail_layers = prepared
            self.signature = signature
            self.cache.clear()
            print(f"Kuldehull klar: {len(self.polygons):,}, alle detaljnivåer lastet", flush=True)
        return True

    def mask(self, bounds, width, height, minimum_depth):
        minimum_depth = self.depth(minimum_depth)
        if not self.path.is_file():
            raise ValueError("Kuldehull-laget mangler")
        west, south, east, north = map(float, bounds)
        if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
            raise ValueError("Ugyldig kartutsnitt for Kuldehull")
        if not 1 <= width <= 2000 or not 1 <= height <= 2000:
            raise ValueError("Ugyldig bildeoppløsning for Kuldehull")
        stat = self.path.stat()
        key = (tuple(bounds), width, height, minimum_depth, stat.st_mtime_ns, stat.st_size)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            self.prepare()
            left, bottom = self.to_web.transform(west, south)
            right, top = self.to_web.transform(east, north)
            pixel_m = min((right - left) / width, (top - bottom) / height)
            mask = np.zeros((height, width), dtype=np.uint8)
            transform = from_bounds(left, bottom, right, top, width, height)
            if pixel_m >= OVERVIEW_CELL_M * 4:
                depths = np.zeros((height, width), dtype=np.float32)
                reproject(self.overview, depths, src_transform=self.overview_transform,
                          src_crs=3857, dst_transform=transform, dst_crs=3857,
                          src_nodata=0, dst_nodata=0, resampling=Resampling.nearest)
                mask[:] = depths > minimum_depth
            else:
                tolerance = max(level for level in DETAIL_LEVELS if level <= pixel_m / 4)
                display = self.detail_layers[tolerance]
                indices = display.sindex.query(box(left, bottom, right, top), predicate="intersects")
                polygons = display.iloc[indices]
                polygons = polygons.loc[polygons.max_depth_m > minimum_depth]
                if len(polygons):
                    rasterize(((geometry, 1) for geometry in polygons.geometry), out=mask,
                              transform=transform, all_touched=False)
            mask.setflags(write=False)
            self.cache[key] = mask
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
            return mask

    def point(self, lon, lat, minimum_depth=0):
        minimum_depth = self.depth(minimum_depth)
        lon, lat = float(lon), float(lat)
        if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -85 <= lat <= 85):
            raise ValueError("Ugyldige koordinater for Kuldehull")
        if not self.path.is_file():
            raise ValueError("Kuldehull-laget mangler")
        self.prepare()
        x, y = self.to_web.transform(lon, lat)
        location = Point(x, y)
        indices = self.polygons.sindex.query(location, predicate="intersects")
        polygons = self.polygons.iloc[indices]
        matches = polygons.loc[polygons.max_depth_m > minimum_depth].sort_values("lake_id")
        if matches.empty:
            return dict(found=False, label="Kuldehull", lon=lon, lat=lat)
        row = matches.iloc[0]
        fields = ("lake_id", "max_depth_m", "mean_depth_m", "area_m2", "volume_m3",
                  "spill_level_m", "group_id", "member_basin_count")
        properties = {}
        for field in fields:
            if field in row and np.isfinite(float(row[field])):
                properties[field] = (int(row[field]) if field in ("lake_id", "group_id", "member_basin_count")
                                     else float(row[field]))
        return dict(found=True, label="Kuldehull", lon=lon, lat=lat, **properties)

    def overlay(self, bounds, width, height, minimum_depth):
        mask = self.mask(bounds, width, height, minimum_depth)
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask.astype(bool)] = [0, 130, 170, 140]
        output = BytesIO()
        Image.fromarray(rgba).save(output, format="PNG")
        output.seek(0)
        return output

    def clip(self, image, engine, minimum_depth, viewport=None):
        image.seek(0)
        with Image.open(image) as source:
            rendered = source.convert("RGBA")
        south, west = engine.bounds[0]
        north, east = engine.bounds[1]
        bounds = (west, south, east, north)
        if viewport is not None:
            transformer = Transformer.from_crs(4326, 3857, always_xy=True)
            left, bottom = transformer.transform(west, south)
            right, top = transformer.transform(east, north)
            view_west, view_south, view_east, view_north = viewport["bounds"]
            view_left, view_bottom = transformer.transform(view_west, view_south)
            view_right, view_top = transformer.transform(view_east, view_north)
            extent = ((view_left - left) / (right - left) * rendered.width,
                      (top - view_top) / (top - bottom) * rendered.height,
                      (view_right - left) / (right - left) * rendered.width,
                      (top - view_bottom) / (top - bottom) * rendered.height)
            rendered = rendered.transform(
                (viewport["width"], viewport["height"]), Image.Transform.EXTENT, extent,
                resample=Image.Resampling.NEAREST,
            )
            bounds = viewport["bounds"]
        rgba = np.array(rendered)
        mask = self.mask(bounds, rgba.shape[1], rgba.shape[0], minimum_depth)
        rgba[:, :, 3][mask == 0] = 0
        output = BytesIO()
        Image.fromarray(rgba).save(output, format="PNG")
        output.seek(0)
        return output


def lake_settings(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Ugyldig Kuldehull-filter")
    if not isinstance(value.get("only", False), bool):
        raise ValueError("Ugyldig Kuldehull-filter")
    result = dict(only=value.get("only", False), depth=KuldehullLayer.depth(value.get("depth", 0)))
    if "bounds" in value:
        bounds = tuple(map(float, value["bounds"]))
        if len(bounds) != 4:
            raise ValueError("Ugyldig kartutsnitt for Kuldehull")
        west, south, east, north = bounds
        if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
            raise ValueError("Ugyldig kartutsnitt for Kuldehull")
        width, height = int(value["width"]), int(value["height"])
        if not 1 <= width <= 2000 or not 1 <= height <= 2000:
            raise ValueError("Ugyldig bildeoppløsning for Kuldehull")
        result["viewport"] = dict(bounds=bounds, width=width, height=height)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare persistent Kuldehull display caches before starting Station Lab")
    parser.add_argument("--source", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if not KuldehullLayer(args.source, args.cache_dir).prepare():
        raise FileNotFoundError(f"Kuldehull source missing: {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())