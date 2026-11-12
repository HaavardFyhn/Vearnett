"""Lokalt kartverktøy for foreslåtte værstasjoner og forventet prediksjonsfeil."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import OrderedDict
import gzip
from contextlib import ExitStack, contextmanager
from io import BytesIO
import os
import secrets
import socket
import tempfile
import threading
import time
from pathlib import Path
from zipfile import ZipFile

_RASTERIO_SPEC = importlib.util.find_spec("rasterio")
if _RASTERIO_SPEC is None or _RASTERIO_SPEC.origin is None:
    raise ImportError("Rasterio is required by Station Lab")
_PROJ_DATA = Path(_RASTERIO_SPEC.origin).resolve().parent / "proj_data"
if not (_PROJ_DATA / "proj.db").is_file():
    raise FileNotFoundError(f"Rasterio's bundled PROJ database is missing: {_PROJ_DATA / 'proj.db'}")
os.environ["PROJ_DATA"] = str(_PROJ_DATA)
os.environ["PROJ_LIB"] = str(_PROJ_DATA)

import joblib
import numpy as np
import pandas as pd
import pyogrio
from flask import Flask, jsonify, request, send_file, session
from PIL import Image
from pyproj import Transformer
from scipy.spatial import cKDTree

from greedy_400_stations_pruned import make_cache, predict, subset_cache
from make_uncertainty_map_gif import (
    HERE, OUT, MODEL, SOURCE_GRID, STATIONS, load_grid_all_fields, load_surfaces,
)
from station_weather import WeatherRegistry, WEATHER, exact_network
from station_weather import PHENOMENON_DATASETS as STATION_LAB_DATASETS
from station_pross import pross_excluded, svv_mask
from station_roads import RoadNetwork
from station_kuldehull import KuldehullLayer, lake_settings
from station_weather_maps import DISPLAY_SIZES, WeatherMapCache, colour_reference, frost_limits
import matplotlib
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation

ASSET_ROOT = HERE

TERRAIN_LAYERS = {
    "elevation": ("Høyde over havet", "m", "terrain", 1),
    "tpi_10km": ("TPI · 10 km", "m", "RdBu_r", 1),
    "tpi_25km": ("TPI · 25 km", "m", "RdBu_r", 1),
    "relief_10km": ("Relieff · 10 km", "m", "YlOrBr", 1),
    "relief_25km": ("Relieff · 25 km", "m", "YlOrBr", 1),
    "rugstd_10km": ("Terrengruhet · 10 km", "m", "cividis", 1),
    "rugstd_25km": ("Terrengruhet · 25 km", "m", "cividis", 1),
    "coast_dist_fjord": ("Avstand til fjord", "km", "Blues", 1000),
    "coast_dist_ocean": ("Avstand til åpent hav", "km", "Blues", 1000),
}


class GreedySurfaces:
    def __init__(self, engine, dataset="temperature:continuous"):
        self.engine = engine
        self.dataset = dataset
        self.shape = (len(engine.stations), len(engine.grid))
        self.values = []
        self.lock = threading.RLock()

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += self.shape[0]
        if not 0 <= index < self.shape[0]:
            raise IndexError("Ugyldig stasjonsnummer")
        with self.lock:
            while len(self.values) <= index:
                prediction = (self.engine.greedy_prediction(len(self.values)) if self.dataset == "temperature:continuous"
                              else self.engine.greedy_prediction(len(self.values), self.dataset))
                current = prediction.copy() if not self.values else np.minimum(self.values[-1], prediction)
                self.values.append(current)
            return self.values[index]


class StationEngine:
    def __init__(self, surface_path, cell_m=5000, frost_neighbors=0, shared=None):
        if frost_neighbors != 0:
            raise ValueError("Frost-nettet støtter bare eksakt beregning over alle stasjoner")
        self.cell_m = cell_m
        from grid_resolution import source_spacing
        self.candidate_cell_m = source_spacing(cell_m)
        self.default_pross_only = True
        self.norway_path = OUT / "norway_polygon.gpkg"
        self.root = self if shared is None else shared.root
        self.surface_path = surface_path
        self.frost_neighbors = frost_neighbors
        self.resources = ExitStack()
        self.grid = load_grid_all_fields(OUT / f"norway_grid_{cell_m}m.gpkg")
        portable_path = OUT / "portable_weather.json"
        portable = json.loads(portable_path.read_text(encoding="utf-8")) if portable_path.is_file() else None
        original = self.grid if portable is not None or cell_m == 5000 else load_grid_all_fields(SOURCE_GRID)
        same_candidates = shared is not None and shared.candidate_cell_m == self.candidate_cell_m
        self.candidate_grid = (shared.candidate_grid if same_candidates else
                               self.grid if cell_m == self.candidate_cell_m else None)
        self.candidate_categories = (shared.candidate_categories if same_candidates else
            pd.Index(pyogrio.read_dataframe(OUT / f"norway_grid_{self.candidate_cell_m}m.gpkg", columns=["objtype"], read_geometry=False)["objtype"].dropna().unique()))
        categories = (pd.Index(portable["categories"]) if portable is not None else
                  pd.Index(original["objtype"].unique()).union(self.grid["objtype"].unique()).union(self.candidate_categories))
        self.categories = categories
        self.grid["objtype_code"] = pd.Categorical(self.grid["objtype"], categories=categories).codes
        self.cache = make_cache(self.grid)
        self.candidate_cache = None
        self.stations = (self.grid.iloc[:0].copy() if portable is not None else
                 pd.read_csv(STATIONS).sort_values("rank").reset_index(drop=True)
                         if shared is None else shared.stations)
        greedy_sources = self.stations.copy()
        greedy_sources["objtype_code"] = pd.Categorical(greedy_sources["objtype"], categories=categories).codes
        self.greedy_cache = make_cache(greedy_sources)
        self.frames = (GreedySurfaces(self) if portable is not None else
                   load_surfaces(surface_path, (len(self.stations), len(self.grid)), self.resources)
                   if cell_m == 1000 else GreedySurfaces(self))
        self.model_frames = {"temperature:continuous": self.frames}
        self.model = None if portable is not None else joblib.load(MODEL) if shared is None else shared.model
        self.weather_images = (WeatherMapCache(OUT / "station_app_cache" / "weather_maps")
                       if shared is None else shared.root.weather_images)
        self.tree = shared.tree if same_candidates else None
        self.to_xy = Transformer.from_crs(4326, 25833, always_xy=True)
        self.to_lonlat = Transformer.from_crs(25833, 4326, always_xy=True)
        self.predictions = OrderedDict()
        self.partial_predictions = OrderedDict()
        self.lock = threading.RLock()
        self.minimum = (0.5 if portable is not None else shared.minimum if shared is not None else
                        max(float(np.percentile(self.frames[-1], 0.5)), 0.01) if cell_m == 1000 else 0.5)
        self.maximum = (6.5 if portable is not None else shared.maximum if shared is not None else
                        max(float(np.percentile(self.frames[0], 99.5)), self.minimum * 2) if cell_m == 1000 else 6.5)
        models = shared.weather.models if shared is not None and shared.weather is not None else None
        self.weather = (WeatherRegistry(HERE, self.grid, self.cache, categories, frost_neighbors, models=models,
                                       cell_m=cell_m, allow_precompute=False)
                if portable is not None or (HERE / "frost_history.db").exists() else None)
        self._prepare_raster()
        if shared is None:
            self.engines = {cell_m: self}
            self.resolution_lock = threading.RLock()
            self.candidate_lock = threading.RLock()

    def ensure_candidates(self):
        if self.candidate_cache is not None:
            return
        with self.root.candidate_lock:
            if self.candidate_grid is None:
                self.candidate_grid = load_grid_all_fields(OUT / f"norway_grid_{self.candidate_cell_m}m.gpkg")
            if self.tree is None:
                self.tree = cKDTree(self.candidate_grid[["x", "y"]].to_numpy())
            features = self.candidate_grid.copy(deep=False)
            features["objtype_code"] = pd.Categorical(features["objtype"], categories=self.categories).codes
            self.candidate_cache = make_cache(features)

    def at_resolution(self, cell_m):
        if cell_m not in (500, 1000, 2500, 5000):
            raise ValueError("Velg 500, 1000, 2500 eller 5000 meter")
        with self.root.resolution_lock:
            if cell_m not in self.root.engines:
                engine = StationEngine(self.surface_path, cell_m, self.frost_neighbors, shared=self.root)
                self.root.resources.callback(engine.resources.close)
                self.root.engines[cell_m] = engine
            return self.root.engines[cell_m]

    def frames_for(self, dataset):
        with self.lock:
            if dataset not in self.model_frames:
                self.model_frames[dataset] = GreedySurfaces(self, dataset)
            return self.model_frames[dataset]

    def frost_statuses(self):
        result = []
        for engine in list(self.root.engines.values()):
            if engine.weather is not None:
                result.extend(dict(profile["status"]) for profile in list(engine.weather.loaded.values()))
        return result

    def sensor_catalog(self):
        """Phenomena per Frost station id, taken from the already loaded model station lists."""
        if getattr(self, "weather", None) is None:
            return {}
        available = {item["id"] for item in self.weather.available}
        profiles = {phenomenon: self.weather.profile(dataset) for phenomenon, dataset in PHENOMENON_DATASETS.items()
                    if dataset in available}
        key = tuple((phenomenon, id(profile["stations"])) for phenomenon, profile in profiles.items())
        with self.lock:
            if getattr(self, "_sensor_catalog_key", None) != key:
                catalog = {}
                for phenomenon, profile in profiles.items():
                    for station_id in profile["stations"]["station_id"].astype(str):
                        catalog.setdefault(station_id, []).append(phenomenon)
                self._sensor_catalog, self._sensor_catalog_key = catalog, key
            return self._sensor_catalog

    def saved_weather_map(self, profile, display_size, limits):
        return self.weather_images.get(self, profile, display_size, limits)

    def station_owner_catalog(self):
        from station_owners import load_catalog
        return load_catalog(ASSET_ROOT, HERE)

    def warm_weather_maps(self, resolutions=(1000,)):
        """Load fixed weather profiles, grids, and available prepared images before serving."""
        self.ensure_candidates()
        ready = 0
        missing = 0
        for spacing in resolutions:
            if not (OUT / f"norway_grid_{spacing}m.gpkg").is_file():
                continue
            active = self.at_resolution(spacing)
            if active.weather is None:
                continue
            for config in active.weather.available:
                if config["id"] not in PHENOMENON_DATASETS.values():
                    continue
                profile = active.weather.profile(config["id"])
                if profile["status"]["state"] != "ready":
                    missing += 1
                    continue
                limits = frost_limits(colour_reference(active.weather, config["id"], profile))
                for size in DISPLAY_SIZES:
                    if active.saved_weather_map(profile, size, limits) is None:
                        missing += 1
                    else:
                        ready += 1
        print(f"Laster værkart", flush=True)

    def _prepare_raster(self):
        from rasterio.transform import from_origin, array_bounds
        from rasterio.warp import calculate_default_transform, transform_bounds

        xs, ys = self.grid["x"].to_numpy(), self.grid["y"].to_numpy()
        self.columns = np.rint((xs - xs.min()) / self.cell_m).astype(int)
        self.rows = np.rint((ys.max() - ys) / self.cell_m).astype(int)
        self.raster_shape = (int(self.rows.max()) + 1, int(self.columns.max()) + 1)
        self.transform = from_origin(xs.min() - self.cell_m / 2,
                                     ys.max() + self.cell_m / 2, self.cell_m, self.cell_m)
        bounds = array_bounds(*self.raster_shape, self.transform)
        _, width, height = calculate_default_transform(25833, 3857,
                                                      self.raster_shape[1], self.raster_shape[0], *bounds)
        scale = min(1, 1600 / max(width, height))
        self.web_transform, self.web_width, self.web_height = calculate_default_transform(
            25833, 3857, self.raster_shape[1], self.raster_shape[0], *bounds,
            dst_width=max(1, int(width * scale)), dst_height=max(1, int(height * scale)))
        web_bounds = array_bounds(self.web_height, self.web_width, self.web_transform)
        west, south, east, north = transform_bounds(3857, 4326, *web_bounds)
        self.bounds = [[south, west], [north, east]]

    def station(self, index):
        self.ensure_candidates()
        row = self.candidate_grid.iloc[index]
        lon, lat = self.to_lonlat.transform(float(row.x), float(row.y))
        return dict(id=int(index), x=float(row.x), y=float(row.y), lon=lon, lat=lat,
                    elevation=float(row.elevation))

    def snap(self, lon, lat):
        if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Ugyldige kartkoordinater")
        self.ensure_candidates()
        distance, index = self.tree.query(self.to_xy.transform(lon, lat))
        if distance > 1500:
            raise ValueError("Velg et punkt innen 1500 meter fra terrengrutenettet for stasjonsplassering")
        return int(index)

    def prediction(self, index, dataset=None):
        self.ensure_candidates()
        key = (dataset, index)
        with self.lock:
            if key not in self.predictions:
                profile = self.weather.profile(dataset) if dataset is not None else None
                model = self.model if profile is None else profile["model"]
                source = subset_cache(self.candidate_cache, np.array([index]))
                values = np.empty(len(self.grid), dtype=np.float32)
                for start in range(0, len(self.grid), 50000):
                    targets = np.arange(start, min(start + 50000, len(self.grid)))
                    values[targets] = predict(model, source, self.cache, targets)[0]
                if profile is not None and profile["kind"] == "class":
                    values = np.minimum(values, 1)
                self.predictions[key] = values
                if len(self.predictions) > 64:
                    self.predictions.popitem(last=False)
            self.predictions.move_to_end(key)
            return self.predictions[key]

    def prediction_at(self, index, target_ids, dataset=None):
        target_ids = np.asarray(target_ids, dtype=np.int64)
        if target_ids.ndim != 1 or (target_ids < 0).any() or (target_ids >= len(self.grid)).any():
            raise ValueError("Ugyldige ruteindekser for forhåndsvisning")
        if not len(target_ids):
            return np.empty(0, dtype=np.float32)
        self.ensure_candidates()
        key = (dataset, index, hashlib.sha256(target_ids.tobytes()).digest())
        with self.lock:
            full = self.predictions.get((dataset, index))
            if full is not None:
                return full[target_ids]
            if key not in self.partial_predictions:
                profile = self.weather.profile(dataset) if dataset is not None else None
                model = self.model if profile is None else profile["model"]
                source = subset_cache(self.candidate_cache, np.array([index]))
                values = np.empty(len(target_ids), dtype=np.float32)
                for start in range(0, len(target_ids), 50000):
                    targets = target_ids[start:start + 50000]
                    values[start:start + len(targets)] = predict(model, source, self.cache, targets)[0]
                if profile is not None and profile["kind"] == "class":
                    values = np.minimum(values, 1)
                self.partial_predictions[key] = values
                if len(self.partial_predictions) > 32:
                    self.partial_predictions.popitem(last=False)
            self.partial_predictions.move_to_end(key)
            return self.partial_predictions[key]

    def greedy_prediction(self, index, dataset="temperature:continuous"):
        key = ("greedy", dataset, index)
        with self.lock:
            if key not in self.predictions:
                profile = self.weather.profile(dataset) if self.weather is not None else None
                model = self.model if dataset == "temperature:continuous" or profile is None else profile["model"]
                source = subset_cache(self.greedy_cache, np.array([index]))
                values = np.empty(len(self.grid), dtype=np.float32)
                for start in range(0, len(self.grid), 50000):
                    targets = np.arange(start, min(start + 50000, len(self.grid)))
                    values[targets] = predict(model, source, self.cache, targets)[0]
                if profile is not None and profile["kind"] == "class":
                    values = np.minimum(values, 1)
                self.predictions[key] = values
                if len(self.predictions) > 64:
                    self.predictions.popitem(last=False)
            self.predictions.move_to_end(key)
            return self.predictions[key]

    def greedy_prediction_at(self, index, target_ids, dataset="temperature:continuous"):
        with self.lock:
            cached = self.predictions.get(("greedy", dataset, index))
            if cached is not None:
                return cached[target_ids]
            source = subset_cache(self.greedy_cache, np.array([index]))
            profile = self.weather.profile(dataset) if self.weather is not None else None
            model = self.model if dataset == "temperature:continuous" or profile is None else profile["model"]
            values = np.empty(len(target_ids), dtype=np.float32)
            for start in range(0, len(target_ids), 50000):
                targets = target_ids[start:start + 50000]
                values[start:start + len(targets)] = predict(model, source, self.cache, targets)[0]
            return np.minimum(values, 1) if profile is not None and profile["kind"] == "class" else values

    def terrain_layer(self, field):
        import matplotlib

        if field not in TERRAIN_LAYERS or field not in self.grid:
            raise ValueError("Ukjent terrenglag")
        label, unit, palette, divisor = TERRAIN_LAYERS[field]
        values = self.grid[field].to_numpy(np.float32) / divisor
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError("Terrenglaget mangler gyldige verdier")
        minimum, maximum = float(finite.min()), float(finite.max())
        if field.startswith("tpi_"):
            maximum = max(abs(minimum), abs(maximum), 1)
            minimum = -maximum
        elif minimum == maximum:
            maximum = minimum + 1
        colors = matplotlib.colormaps[palette](np.linspace(0, 1, 7))[:, :3]
        stops = ["rgb(" + ",".join(str(int(channel * 255)) for channel in color) + ")" for color in colors]
        return dict(field=field, label=label, unit=unit, palette=palette,
                    minimum=minimum, maximum=maximum, values=values,
                    gradient="linear-gradient(to right," + ",".join(stops) + ")")

    def placement_layer(self, dataset):
        active = self.at_resolution(5000)
        if active is not self:
            return active.placement_layer(dataset)
        config = next((item for item in self.weather.available if item["id"] == dataset), None)
        if config is None:
            raise ValueError("Ingen lagret femårsmodell for dette værvalget")
        prefix = Path(config["prefix"])
        path = prefix.with_name(prefix.name + "_error_reduction.gpkg")
        if not path.exists():
            raise ValueError("Lagret plasseringsgevinst på 5000 m mangler for dette værvalget")
        stat = path.stat()
        key = (dataset, stat.st_size, stat.st_mtime_ns)
        if not hasattr(self, "placement_layers"):
            self.placement_layers = {}
        if key not in self.placement_layers:
            frame = pyogrio.read_dataframe(path, layer="error_reduction",
                                           columns=["x", "y", "total_error_reduction"], read_geometry=False)
            expected = pd.MultiIndex.from_frame(self.grid[["x", "y"]])
            indexed = frame.set_index(["x", "y"])["total_error_reduction"]
            if not indexed.index.is_unique or len(indexed) != len(expected) or len(indexed.index.difference(expected)):
                raise ValueError("Plasseringskartet samsvarer ikke med terrengrutenettet på 5000 m")
            values = indexed.reindex(expected).to_numpy(dtype=np.float32)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Plasseringskartet inneholder ugyldig total feilreduksjon")
            self.placement_layers[key] = dict(dataset=dataset, label="Plasseringsgevinst", field="total_error_reduction",
                unit=config["unit"], metric=config["metric"], weather_label=config["label"], cell_m=5000,
                minimum=0., maximum=max(float(values.max()), .0001), values=values,
                bounds=self.bounds, source=path.name)
        return self.placement_layers[key]

    def terrain_catalog(self):
        result = []
        for field in TERRAIN_LAYERS:
            if field in self.grid and np.isfinite(self.grid[field].to_numpy(float)).any():
                result.append({key: value for key, value in self.terrain_layer(field).items() if key != "values"})
        return dict(cell_m=self.cell_m, layers=result)

    def target_index(self, lon, lat):
        if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Ugyldige kartkoordinater")
        coordinates = self.grid[["x", "y"]].to_numpy()
        if not hasattr(self, "terrain_tree"):
            self.terrain_tree = cKDTree(coordinates)
        point = np.asarray(self.to_xy.transform(lon, lat))
        if not np.isfinite(point).all():
            raise ValueError("Punktet ligger utenfor evaluerte terrengruter")
        _, index = self.terrain_tree.query(point)
        if (np.abs(coordinates[index] - point) > self.cell_m / 2 + 0.01).any():
            raise ValueError("Punktet ligger utenfor evaluerte terrengruter")
        return int(index)

    def terrain_point(self, lon, lat):
        index = self.target_index(lon, lat)
        row = self.grid.iloc[index]
        values = []
        for field, (label, unit, _, divisor) in TERRAIN_LAYERS.items():
            if field in row:
                value = float(row[field]) / divisor
                values.append(dict(field=field, label=label, unit=unit,
                                   value=value if np.isfinite(value) else None))
        return dict(cell_m=self.cell_m, x=float(row.x), y=float(row.y), values=values)

    def render(self, values, mode, minimum=None, maximum=None, reduction_max=2, cell_mask=None, display_size=None):


        scale = 1 if display_size is None else min(1, display_size / max(self.web_height, self.web_width))
        height, width = max(1, round(self.web_height * scale)), max(1, round(self.web_width * scale))
        web_transform = self.web_transform * self.web_transform.scale(self.web_width / width, self.web_height / height)
        image = np.zeros((height, width, 4), dtype=np.uint8)
        minimum = self.minimum if minimum is None else minimum
        maximum = self.maximum if maximum is None else maximum
        if mode in ("uncertainty", "baseline", "combined"):
            values = (np.full(len(self.grid), maximum, dtype=np.float32) if values is None else
                      np.where(np.isfinite(values), values, maximum))
        if values is not None:
            source = np.full(self.raster_shape, -9999, dtype=np.float32)
            source[self.rows, self.columns] = values if cell_mask is None else np.where(cell_mask, values, -9999)
            target = np.full((height, width), -9999, dtype=np.float32)
            reproject(source, target, src_transform=self.transform, src_crs=25833,
                      dst_transform=web_transform, dst_crs=3857,
                      src_nodata=-9999, dst_nodata=-9999,
                      resampling=Resampling.average if scale < 1 and mode in ("uncertainty", "baseline") else Resampling.nearest)
            valid = (target != -9999) & np.isfinite(target)
            if mode.startswith("terrain:"):
                terrain = self.terrain_layer(mode.split(":", 1)[1])
                scaled = (target - terrain["minimum"]) / (terrain["maximum"] - terrain["minimum"])
                colors = matplotlib.colormaps[terrain["palette"]](np.clip(scaled, 0, 1))
            elif mode == "placement":
                scaled = np.log1p(np.maximum(target, 0)) / np.log1p(maximum)
                colors = matplotlib.colormaps["YlOrRd"](np.clip(scaled, 0, 1))
            elif mode in ("uncertainty", "baseline"):
                scaled = (np.log(np.maximum(target, minimum)) - np.log(minimum)) / np.log(maximum / minimum)
                colors = matplotlib.colormaps["RdBu_r"](np.clip(scaled, 0, 1))
            elif mode in ("change", "affected_change"):
                magnitude = np.log1p(np.abs(target) / (reduction_max / 2000)) / np.log1p(2000)
                scaled = np.sign(target) if mode == "affected_change" else np.sign(target) * np.clip(magnitude, 0, 1)
                colors = matplotlib.colormaps["RdBu"]((scaled + 1) / 2)
                valid &= target != 0
            elif mode == "affected":
                colors = np.zeros((*target.shape, 4))
                colors[:, :, :3] = [0.85, 0, 0.48]
                valid &= target > 0
            elif mode == "combined":
                scaled = (target - minimum) / (maximum - minimum)
                colors = matplotlib.colormaps["RdBu_r"](np.clip(scaled, 0, 1))
            elif mode == "combined_change":
                colors = matplotlib.colormaps["RdBu"]((np.clip(target / maximum, -1, 1) + 1) / 2)
                valid &= target != 0
            elif mode == "combined_reduction":
                colors = matplotlib.colormaps["YlGnBu"](np.clip(target / maximum, 0, 1))
                valid &= target > 0
            else:
                scaled = np.log1p(np.maximum(target, 0) / (reduction_max / 2000)) / np.log1p(2000)
                colors = matplotlib.colormaps["YlGnBu"](np.clip(scaled, 0, 1))
                valid &= target > 0
            image = (colors * 255).astype(np.uint8)
            image[:, :, 3] = valid.astype(np.uint8) * 210
            if mode in ("affected", "affected_change"):
                image[:, :, 3] = valid.astype(np.uint8) * 255
                outline = binary_dilation(valid) & ~valid
                image[outline] = [255, 255, 255, 255]
        if getattr(self, "norway_path", None) is not None:
            from rasterio.features import geometry_mask

            if not hasattr(self, "_norway_shapes"):
                self._norway_shapes = pyogrio.read_dataframe(self.norway_path).to_crs(3857).geometry.tolist()
            if not hasattr(self, "_norway_masks"):
                self._norway_masks = {}
            key = (height, width)
            if key not in self._norway_masks:
                self._norway_masks[key] = geometry_mask(
                    self._norway_shapes, out_shape=(height, width), transform=web_transform, invert=True)
            image[:, :, 3] = np.where(self._norway_masks[key], image[:, :, 3], 0)
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        buffer.seek(0)
        return buffer


PHENOMENON_DATASETS = STATION_LAB_DATASETS


def default_threshold(profile):
    return 0.001 if profile["kind"] == "class" else 0.01 if profile["phenomenon"] == "temperature" else 0.1


def combine_layers(parts, method="sum"):
    """Sum, or root-sum-square, of normalised layers; signed parts are combined per sign."""
    stack = np.stack(parts).astype(np.float32)
    finite = np.isfinite(stack).any(axis=0)
    stack = np.nan_to_num(stack)
    if method == "rss":
        total = (np.sqrt((np.maximum(stack, 0) ** 2).sum(axis=0)) -
                 np.sqrt((np.minimum(stack, 0) ** 2).sum(axis=0)))
    else:
        total = stack.sum(axis=0)
    return np.where(finite, total, np.nan).astype(np.float32)


def combined_range(values, kind):
    finite = np.empty(0) if values is None else values[np.isfinite(values)]
    if kind == "combined_change":
        bound = float(np.abs(finite).max()) if len(finite) else 0.0
        return -max(bound, 1e-6), max(bound, 1e-6)
    if kind == "combined_reduction":
        positive = finite[finite > 0]
        return 0.0, max(float(positive.max()) if len(positive) else 0.0, 1e-6)
    if not len(finite):
        return 0.0, 1.0
    lower, upper = float(finite.min()), float(finite.max())
    return lower, max(upper, lower + 1e-6)


class Scenario:
    def __init__(self, engine, baseline=None, dataset="temperature:continuous"):
        self.engine = engine
        self.baseline = min(12, len(engine.stations)) if baseline is None else baseline
        dataset = "temperature:senorge30" if dataset == "temperature:senorge30_pross" else dataset
        self.dataset = dataset
        self.initial_dataset = dataset
        self.initial_baseline = self.baseline
        self.layers = [dataset]
        self._combined = {}
        self.combine = "sum"
        self.added_sensors = {}
        self.added_positions = {}
        self.pross_only = getattr(engine, "default_pross_only", False)
        self.station_since = ""
        self.counties = None
        self.network = "frost" if getattr(engine, "weather", None) is not None else "greedy"
        self.added = []
        self.selected = None
        self.comparison_scope = "selected"
        self.frost_edits = {}
        self.station_edits = {}
        self.station_sensor_removals = {}
        self._frost_fallbacks = {}
        self._restricted_profiles = {}
        self.threshold = 0.001 if dataset.endswith(":class") else 0.01 if dataset.startswith("temperature:") else 0.1
        self.revision = 0
        self.updated = time.monotonic()
        self.lock = threading.RLock()
        self._change_key = None
        self._change_values = None

    def profile(self):
        profile = self.engine.weather.profile(self.dataset) if getattr(self.engine, "weather", None) is not None else None
        if (profile is not None and self.pross_only and self.network == "frost"
                and profile.get("baseline") is not None and profile.get("cache_path") is not None):
            if getattr(self.engine.weather, "portable_manifest", None) is not None:
                return self.engine.weather.svv_profile(self.dataset)
            from station_weather import restricted_profile
            excluded = pross_excluded(profile["stations"])
            key = (self.dataset, self.engine.cell_m, id(profile["baseline"]), excluded)
            if key not in self._restricted_profiles:
                cached = restricted_profile(profile, self.engine.cache, excluded)
                if cached is not None:
                    self._restricted_profiles[key] = cached
            return self._restricted_profiles.get(key, profile)
        return profile

    def phenomenon(self):
        profile = self.profile()
        return None if profile is None else profile["phenomenon"]

    def layer_phenomena(self):
        return [dataset.split(":")[0] for dataset in self.layers]

    def active_added(self, additions=None):
        """Own stations that carry a sensor for the current phenomenon."""
        additions = self.added if additions is None else additions
        phenomenon = self.phenomenon()
        if phenomenon is None:
            return list(additions)
        return [index for index in additions if phenomenon in self.added_sensors.get(index, (phenomenon,))]

    def masked(self):
        if self.network != "frost" or not self.ready():
            return frozenset()
        stations = self.profile()["stations"]
        excluded = set(pross_excluded(stations)) if self.pross_only else set()
        if self.station_since:
            dates = pd.to_datetime(stations.get("last_time", pd.Series("", index=stations.index)),
                                   errors="coerce", utc=True)
            excluded.update(np.flatnonzero(
                (dates.isna() | (dates < pd.Timestamp(self.station_since, tz="UTC"))).to_numpy()).tolist())
        return frozenset(excluded)

    @staticmethod
    def validate_station_date(value):
        if not isinstance(value, str):
            raise ValueError("Velg en gyldig dato (ÅÅÅÅ-MM-DD)")
        if value:
            from datetime import date
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Velg en gyldig dato (ÅÅÅÅ-MM-DD)")
        return value

    def station_date_preview(self, value):
        value = self.validate_station_date(value)
        if self.network != "frost":
            raise ValueError("Datofilteret krever Frost-nett")
        layers = []
        for dataset in self.layers:
            with self.as_dataset(dataset):
                profile = self.profile()
                stations = profile["stations"]
                dates = pd.to_datetime(stations.get("last_time", pd.Series("", index=stations.index)),
                                       errors="coerce", utc=True)
                eligible = set(range(len(stations)))
                if self.pross_only:
                    eligible -= set(pross_excluded(stations))
                removed = (set(np.flatnonzero((dates.isna() | (dates < pd.Timestamp(value, tz="UTC"))).to_numpy()))
                           & eligible) if value else set()
                current = set(self.masked()) & eligible
                layers.append(dict(dataset=dataset, label=profile["label"], total=len(eligible),
                    removed=len(removed), remaining=len(eligible - removed),
                    unknown=int(dates.iloc[sorted(eligible)].isna().sum()),
                    additional_removed=len(removed - current), restored=len(current - removed)))
        return dict(value=value, revision=self.revision, layers=layers)

    def reference(self):
        """Unedited Frost surface, after the station mask."""
        profile = self.profile()
        if not self.ready():
            return None
        if not self.masked():
            return profile["baseline"]
        saved = self.frost_edits
        self.frost_edits = {}
        try:
            return self.frost_surface()
        finally:
            self.frost_edits = saved

    def frost_selection(self, index=None, station_id=None):
        """Frost selection keyed by station id; index is None when the current layer lacks the station."""
        stations = self.profile()["stations"]
        if index is None and station_id is not None:
            matches = np.flatnonzero(stations["station_id"].astype(str).to_numpy() == str(station_id))
            index = int(matches[0]) if len(matches) and int(matches[0]) not in self.masked() else None
        elif index is not None:
            station_id = str(stations.iloc[index]["station_id"])
        return dict(kind="frost", index=index, station_id=station_id)

    def edits_for_current(self):
        """Map station-id edits onto the current layer's station indices."""
        if self.network != "frost" or self.profile() is None:
            return {}
        ids = self.profile()["stations"]["station_id"].astype(str).to_numpy()
        lookup = {station_id: index for index, station_id in enumerate(ids)}
        masked = self.masked()
        edits = {lookup[station_id]: dict(edit) for station_id, edit in self.station_edits.items()
                 if station_id in lookup and lookup[station_id] not in masked}
        for station_id, sensors in self.station_sensor_removals.items():
            if self.phenomenon() in sensors and station_id in lookup and lookup[station_id] not in masked:
                edits[lookup[station_id]] = dict(removed=True)
        return edits

    def sync_edits(self):
        self.frost_edits = self.edits_for_current()

    def frost_station(self, station_id):
        """Original row of a Frost station from the first checked layer that contains it."""
        for dataset in self.station_datasets():
            with self.as_dataset(dataset):
                if self.profile() is None:
                    continue
                stations = self.profile()["stations"]
                matches = np.flatnonzero(stations["station_id"].astype(str).to_numpy() == str(station_id))
                if len(matches) and int(matches[0]) not in self.masked():
                    return stations.iloc[int(matches[0])]
        return None

    def station_datasets(self):
        available = {item["id"] for item in self.engine.weather.available}
        return list(dict.fromkeys([*self.layers, *(
            dataset for dataset in PHENOMENON_DATASETS.values() if dataset in available)]))

    def attached_proposal(self, station_id):
        original = self.frost_station(station_id)
        if original is None:
            return None

        edit = self.station_edits.get(station_id, {})
        if edit.get("removed"):
            return None
        longitude, latitude = (float(original.lon), float(original.lat)) if (
            "lon" in original and "lat" in original) else self.engine.to_lonlat.transform(original.x, original.y)
        longitude, latitude = edit.get("display_lon", longitude), edit.get("display_lat", latitude)
        for index in self.added:
            position = self.added_positions.get(index, {})
            if (abs(position.get("lon", float("inf")) - longitude) < 1e-9 and
                    abs(position.get("lat", float("inf")) - latitude) < 1e-9):
                return index
        return None

    def restore_attached_proposals(self, station_ids):
        moves = []
        for station_id in station_ids:
            index = self.attached_proposal(station_id)
            if index is None or "candidate_index" not in self.station_edits.get(station_id, {}):
                continue
            original = self.frost_station(station_id)
            longitude, latitude = (float(original.lon), float(original.lat)) if (
                "lon" in original and "lat" in original) else self.engine.to_lonlat.transform(original.x, original.y)
            destination = self.engine.snap(longitude, latitude)
            if destination != index and destination in self.added:
                raise ValueError("Et eget forslag finnes allerede ved den opprinnelige stasjonen")
            if self.phenomenon() in self.added_sensors[index]:
                self.prediction(destination)
            moves.append((index, destination, longitude, latitude))
        if len({destination for _, destination, _, _ in moves}) != len(moves):
            raise ValueError("Flere forslag kan ikke gjenopprettes i samme terrengrute")
        for index, destination, longitude, latitude in moves:
            self.added[self.added.index(index)] = destination
            self.added_sensors[destination] = self.added_sensors.pop(index)
            self.added_positions.pop(index, None)
            self.added_positions[destination] = dict(lon=longitude, lat=latitude)

    @contextmanager
    def as_dataset(self, dataset):
        saved = (self.dataset, self.frost_edits, self.selected, self.threshold, self._change_key, self._change_values)
        if dataset != saved[0]:
            self.dataset, self._change_key = dataset, None
            self.frost_edits = self.edits_for_current()
            self.selected = saved[2] if saved[2] is not None and saved[2]["kind"] != "frost" else None
            if saved[2] is not None and saved[2]["kind"] == "frost" and saved[2].get("station_id"):
                self.selected = self.frost_selection(station_id=saved[2]["station_id"])
            self.threshold = default_threshold(self.profile())
        try:
            yield
        finally:
            (self.dataset, self.frost_edits, self.selected, self.threshold,
             self._change_key, self._change_values) = saved

    def frames(self):
        return self.engine.frames_for(self.dataset) if hasattr(self.engine, "frames_for") else self.engine.frames

    def ready(self):
        profile = self.profile()
        return (self.network != "frost" or profile is None or
                (profile.get("baseline") is not None and profile.get("status", {}).get("state", "ready") == "ready"))

    def greedy_values(self, index, targets=None):
        if targets is None:
            return (self.engine.greedy_prediction(index) if self.dataset == "temperature:continuous"
                    else self.engine.greedy_prediction(index, self.dataset))
        if hasattr(self.engine, "greedy_prediction_at"):
            return (self.engine.greedy_prediction_at(index, targets) if self.dataset == "temperature:continuous"
                    else self.engine.greedy_prediction_at(index, targets, self.dataset))
        return self.greedy_values(index)[targets]

    def base(self):
        if self.network == "empty":
            return None
        if self.network == "frost":
            if not self.ready():
                return None
            return self.frost_surface()
        return None if self.baseline == 0 else self.frames()[self.baseline - 1].copy()

    def fixed(self, include_removed=False):
        if self.network == "frost":
            stations = self.profile()["stations"].copy()
            stations["source_index"] = np.arange(len(stations), dtype=np.int32)
            masked = self.masked()
            if masked:
                stations = stations.loc[~stations.source_index.isin(masked)].copy()
            for index, edit in self.frost_edits.items():
                if edit.get("removed"):
                    if not include_removed:
                        stations = stations.loc[stations.source_index != index].copy()
                else:
                    location = self.engine.station(edit["candidate_index"])
                    for field in ("x", "y", "lon", "lat", "elevation"):
                        stations.loc[stations.source_index == index, field] = location[field]
            return stations.reset_index(drop=True)
        return self.engine.stations.iloc[:0 if self.network == "empty" else self.baseline]

    def frost_surface(self, omit_index=None, restore_index=None):
        profile = self.profile()
        masked = self.masked()
        permanent = set(profile.get("network_excluded", ()))
        excluded = (set(self.frost_edits) | masked) - permanent
        if restore_index is not None:
            excluded.discard(restore_index)
        if omit_index is not None:
            excluded.add(omit_index)
        if not excluded:
            current = profile["baseline"].copy()
        elif len(excluded) >= len(profile["stations"]) - len(permanent):
            current = None
        elif len(excluded) == 1:
            only_index = next(iter(excluded))
            if len(profile["stations"]) <= 1:
                current = None
            else:
                current = np.where(profile["winner"] == only_index,
                                   profile["second"], profile["baseline"]).copy()
        else:
            cache_key = (self.dataset, self.engine.cell_m, tuple(sorted(excluded)), id(profile["baseline"]))
            cached = self._frost_fallbacks.get(cache_key)
            disk = None
            if cached is None and masked and profile.get("cache_path") is not None:
                digest = hashlib.sha256(json.dumps([Path(profile["cache_path"]).name, sorted(excluded)]).encode())
                disk = Path(profile["cache_path"]).with_name(f"frost_masked_{digest.hexdigest()[:20]}.npy")
                if disk.is_file():
                    stored = np.load(disk, allow_pickle=False)
                    if stored.shape == profile["baseline"].shape:
                        cached = stored
                        self._frost_fallbacks[cache_key] = cached
            if cached is None:
                current = profile["baseline"].copy()
                # Unedited original winners remain optimal; only edited winners need a fallback search.
                targets = np.flatnonzero(np.isin(profile["winner"], np.fromiter(excluded, dtype=np.int32)))
                if len(targets):
                    sources = profile["stations"].iloc[
                        [index not in excluded | permanent for index in range(len(profile["stations"]))]
                    ]
                    target = subset_cache(self.engine.cache, targets)
                    best, _, _ = exact_network(
                        profile["model"], make_cache(sources), target,
                        profile["kind"] == "class",
                    )
                    current[targets] = best
                self._frost_fallbacks[cache_key] = current
                if disk is not None:
                    staged = disk.with_suffix(f".{os.getpid()}.tmp.npy")
                    np.save(staged, current)
                    os.replace(staged, disk)
                if len(self._frost_fallbacks) > 8:
                    self._frost_fallbacks.pop(next(iter(self._frost_fallbacks)))
            else:
                current = cached.copy()
        for index, edit in self.frost_edits.items():
            if edit.get("removed") or index in (omit_index, restore_index):
                continue
            moved = self.prediction(edit["candidate_index"])
            current = moved.copy() if current is None else np.minimum(current, moved)
        return current

    def prediction(self, index):
        return (self.engine.prediction(index, self.dataset) if getattr(self.engine, "weather", None) is not None
                else self.engine.prediction(index))

    def limits(self, cell_mask=None):
        profile = self.profile()
        if profile is None:
            return self.engine.minimum, self.engine.maximum, 2
        profile = colour_reference(self.engine.weather, self.dataset, profile)
        self._colour_reference_warning = profile.get("colour_reference_warning")
        return frost_limits(profile)

    def selection(self):
        if self.selected is not None:
            return self.selected
        if self.added:
            return dict(kind="manual", index=self.added[-1])
        if self.network == "greedy" and self.baseline:
            return dict(kind="greedy", index=self.baseline - 1)
        return None

    def surface(self, exclude_last=False, omit=None, restore_index=None):
        if not self.ready():
            return None
        current = self.base() if restore_index is None else self.frost_surface(restore_index=restore_index)
        if omit is not None and omit["kind"] == "frost":
            current = self.frost_surface(omit_index=omit["index"])
        if omit is not None and omit["kind"] == "greedy":
            if omit["index"] == self.baseline - 1:
                current = None if self.baseline == 1 else self.frames()[self.baseline - 2].copy()
            else:
                current = None
                for rank in range(self.baseline):
                    if rank != omit["index"]:
                        values = self.greedy_values(rank)
                        current = values.copy() if current is None else np.minimum(current, values)
        additions = self.active_added(self.added[:-1] if exclude_last else self.added)
        for index in additions:
            if omit is not None and omit["kind"] == "manual" and omit["index"] == index:
                continue
            prediction = self.prediction(index)
            current = prediction.copy() if current is None else np.minimum(current, prediction)
        return current

    def surface_at(self, target_ids):
        if not self.ready():
            return None
        base = self.base()
        current = None if base is None else base[target_ids].copy()
        for index in self.active_added():
            prediction = (self.engine.prediction_at(index, target_ids, self.dataset)
                          if hasattr(self.engine, "prediction_at") and getattr(self.engine, "weather", None) is not None else
                          self.engine.prediction_at(index, target_ids) if hasattr(self.engine, "prediction_at") else
                          self.prediction(index)[target_ids])
            current = prediction.copy() if current is None else np.minimum(current, prediction)
        return current

    def best_station(self, lon, lat):
        if not self.ready():
            raise ValueError("Eksakt Frost-grunnflate mangler. Kjør precompute_frost.py separat")
        target_index = self.engine.target_index(lon, lat)
        targets = np.array([target_index], dtype=np.int64)
        profile = self.profile()
        candidates = []

        def append(kind, index, label, value, x, y):
            if profile is not None and profile.get("allow_nodata") and np.isposinf(value):
                return
            if not np.isfinite(value) or value < 0:
                raise ValueError("Stasjonsmodellen ga ugyldig prediksjonsfeil")
            longitude, latitude = self.engine.to_lonlat.transform(x, y)
            candidates.append(dict(kind=kind, index=int(index), label=label, error=float(value),
                                   lon=float(longitude), lat=float(latitude)))

        if self.network == "frost":
            originals = profile["stations"]
            masked = self.masked()
            indices = [index for index in range(len(originals)) if index not in self.frost_edits and index not in masked]
            if indices:
                sources = make_cache(originals.iloc[indices])
                for start in range(0, len(indices), 4096):
                    batch = np.arange(start, min(start + 4096, len(indices)))
                    values = predict(profile["model"], subset_cache(sources, batch), self.engine.cache, targets)[:, 0]
                    if profile["kind"] == "class":
                        values = np.minimum(values, 1)
                    for index, value in zip(np.asarray(indices)[batch], values):
                        row = originals.iloc[index]
                        append("frost", index, f"{row.station_id} · {row['name']}", value, row.x, row.y)
            for index, edit in self.frost_edits.items():
                if edit.get("removed"):
                    continue
                location = self.engine.station(edit["candidate_index"])
                value = self.engine.prediction_at(edit["candidate_index"], targets, self.dataset)[0]
                row = originals.iloc[index]
                append("frost", index, f"{row.station_id} · {row['name']} (flyttet)", value, location["x"], location["y"])
        elif self.network == "greedy":
            for index in range(self.baseline):
                row = self.engine.stations.iloc[index]
                append("greedy", index, f"Foreslått stasjon {index + 1}",
                       self.greedy_values(index, targets)[0], row.x, row.y)
        for number, index in enumerate(self.added, 1):
            if index not in self.active_added():
                continue
            location = self.engine.station(index)
            value = (self.engine.prediction_at(index, targets, self.dataset) if profile is not None
                     else self.engine.prediction_at(index, targets))[0]
            append("manual", index, f"Ny stasjon {number}", value, location["x"], location["y"])
        if not candidates:
            if profile is not None and profile.get("allow_nodata"):
                raise ValueError("Ingen aktive seNorge-referanser innen 46 km ved denne støttede ruten")
            raise ValueError("Scenariet har ingen aktive stasjoner å undersøke")
        candidates.sort(key=lambda station: station["error"])
        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        third = candidates[2] if len(candidates) > 2 else None
        row = self.engine.grid.iloc[target_index]
        def station_features(station):
            if station["kind"] == "frost" and station["index"] not in self.frost_edits:
                return profile["stations"].iloc[station["index"]]
            if station["kind"] == "greedy":
                return self.engine.stations.iloc[station["index"]]
            index = (self.frost_edits[station["index"]]["candidate_index"]
                     if station["kind"] == "frost" else station["index"])
            return self.engine.candidate_grid.iloc[index]

        source = station_features(best)
        runner_source = None if runner_up is None else station_features(runner_up)
        third_source = None if third is None else station_features(third)
        terrain = []
        for field, (label, unit, _, divisor) in TERRAIN_LAYERS.items():
            if field not in row and field not in source and (runner_source is None or field not in runner_source):
                continue
            values = {}
            for name, features in (("place", row), ("station", source), ("runner_up", runner_source), ("third", third_source)):
                value = float(features.get(field, np.nan)) / divisor if features is not None else np.nan
                values[name] = value if np.isfinite(value) else None
            terrain.append(dict(field=field, label=label, unit=unit, **values))
        return dict(revision=self.revision, dataset=self.dataset, cell_m=self.engine.cell_m,
                    x=float(row.x), y=float(row.y), station_count=len(candidates),
                    distance_km=float(np.hypot(source.x - row.x, source.y - row.y)) / 1000,
                    terrain=terrain,
                    unit="\u00b0C" if profile is None else profile["unit"],
                    metric="RMSE" if profile is None else profile["metric"],
                    best=best, runner_up=runner_up, third=third,
                    removal_increase=None if runner_up is None else runner_up["error"] - best["error"],
                    tied_count=sum(bool(np.isclose(station["error"], best["error"], rtol=0, atol=1e-6))
                                   for station in candidates))

    def change(self):
        profile = self.profile()
        selected = self.selection()
        key = (id(self.engine), self.engine.cell_m, self.dataset, self.network, self.baseline, self.comparison_scope,
               tuple((index, self.added_sensors.get(index)) for index in self.added), self.pross_only, self.station_since,
               None if selected is None else (selected["kind"], selected["index"], selected.get("station_id")),
                             tuple((index, tuple(sorted(edit.items()))) for index, edit in sorted(self.frost_edits.items())),
               None if profile is None else id(profile.get("baseline")),
               None if profile is None else profile.get("status", {}).get("state", "ready"))
        if key != self._change_key:
            values = self._calculate_change()
            self._change_key, self._change_values = key, values
        return self._change_values

    def compares_baseline(self):
        return self.network == "frost" and self.comparison_scope == "baseline"

    def comparison_kind(self):
        if self.compares_baseline():
            return "baseline"
        selected = self.selection()
        if selected is not None and selected["kind"] == "frost":
            edit = self.frost_edits.get(selected["index"])
            if edit is not None:
                return "removal" if edit.get("removed") else "move"
        return "contribution"

    def signed_comparison(self):
        return self.comparison_kind() != "contribution"

    def comparison_method(self):
        return {
            "baseline": "Opprinnelig Frost-grunnnett minus scenariet",
            "move": "Bare valgt flytting: opprinnelig plassering minus ny plassering; andre endringer beholdes",
            "removal": "Bare valgt fjerning: scenario med stasjonen tilbake minus dagens scenario; andre endringer beholdes",
            "contribution": "Nettet med og uten valgt stasjon",
        }[self.comparison_kind()]

    def affected(self, reduction):
        if reduction is None:
            return None
        values = np.abs(reduction) if self.signed_comparison() else reduction
        return values > self.threshold

    def map_change(self, reduction, mode):
        signed = self.signed_comparison()
        if mode == "affected" and reduction is not None:
            mask = (np.abs(reduction) if signed else reduction) > self.threshold
            reduction = np.where(mask, np.sign(reduction), 0).astype(np.float32)
        if signed and mode in ("reduction", "affected"):
            mode = "change" if mode == "reduction" else "affected_change"
        return reduction, mode

    def _normalised(self, mode):
        if not self.ready():
            return None
        minimum, maximum, reduction_max = self.limits()
        if mode in ("uncertainty", "baseline"):
            if mode == "baseline" and self.network == "frost":
                values = self.reference()
            else:
                values = self.base() if mode == "baseline" or not self.active_added() else self.surface()
            if values is None:
                return None
            with np.errstate(invalid="ignore", divide="ignore"):
                scaled = (np.log(np.maximum(values, minimum)) - np.log(minimum)) / np.log(maximum / minimum)
            return np.where(np.isfinite(values), np.clip(scaled, 0, 1), np.nan)
        _, _, reduction = self.change()
        if reduction is None:
            return None
        if mode == "affected":
            return self.map_change(reduction, "affected")[0].astype(np.float32)
        with np.errstate(invalid="ignore"):
            magnitude = np.log1p(np.abs(reduction) / (reduction_max / 2000)) / np.log1p(2000)
        return np.sign(reduction) * np.clip(magnitude, 0, 1)

    def combined(self, mode):
        """Combine the checked phenomena, each scaled to 0..1 (or -1..1) with its own colour scale."""
        key = (mode, self.revision, tuple(self.layers), id(self.engine), self.combine)
        if key in self._combined:
            return self._combined[key]
        parts, signed = [], False
        for dataset in self.layers:
            with self.as_dataset(dataset):
                values = self._normalised(mode)
                signed |= mode in ("reduction", "affected") and self.signed_comparison()
            if values is None and mode in ("uncertainty", "baseline"):
                values = np.full(len(self.engine.grid), np.nan, dtype=np.float32)
            if values is not None:
                parts.append(values)
        total = combine_layers(parts, self.combine) if parts else None
        if total is not None and mode in ("uncertainty", "baseline"):
            total = np.where(np.isfinite(np.stack(parts)).all(axis=0), total, np.nan)
        kind = ("combined" if mode in ("uncertainty", "baseline") else
                "combined_change" if signed else "combined_reduction")
        if len(self._combined) >= 8:
            self._combined.pop(next(iter(self._combined)))
        self._combined[key] = (total, kind)
        return self._combined[key]

    def _calculate_change(self):
        profile = self.profile()
        def difference(before, current):
            if before is None or current is None:
                return None
            if profile is not None and profile.get("allow_nodata"):
                result = np.full(current.shape, np.nan, np.float32)
                np.subtract(before, current, out=result, where=np.isfinite(before) & np.isfinite(current))
                return result
            return before - current

        if not self.ready():
            return None, None, None
        current = self.surface()
        if self.compares_baseline():
            before = self.reference()
            before = None if before is None else before.copy()
            return current, before, difference(before, current)
        selected = self.selection()
        if self.comparison_kind() in ("move", "removal"):
            restored = self.surface(restore_index=selected["index"])
            return current, restored, difference(restored, current)
        if selected is not None and selected["kind"] == "greedy":
            index = selected["index"]
            if self.baseline == 1 and not self.added:
                return current, None, None
            insertion = self.frames()[index]
            previous = None if index == 0 else self.frames()[index - 1]
            possible = insertion == current
            if previous is not None:
                possible &= insertion < previous
            targets = np.flatnonzero(possible)
            before = current.copy()
            if len(targets):
                competing = np.full(len(targets), np.inf, dtype=np.float32) if previous is None else previous[targets].copy()
                for rank in range(index + 1, self.baseline):
                    values = self.greedy_values(rank, targets)
                    competing = np.minimum(competing, values)
                for identifier in self.active_added():
                    competing = np.minimum(competing, self.prediction(identifier)[targets])
                before[targets] = competing
        else:
            before = self.surface(omit=selected) if selected is not None else current
        delta = difference(before, current)
        reduction = None if delta is None else np.maximum(delta, 0)
        return current, before, reduction

    def apply(self, data):
        action = data.get("action")
        if action == "remove_sensor":
            sensor = data.get("sensor")
            if sensor not in PHENOMENON_DATASETS:
                raise ValueError("Velg en gyldig sensor")
            station_id = str(data.get("station_id", ""))
            if station_id:
                if self.network != "frost" or self.frost_station(station_id) is None:
                    raise ValueError("Velg en stasjon som finnes i nettet")
                index = self.attached_proposal(station_id)
            else:
                index = int(data["id"])
                if index not in self.added:
                    raise ValueError("Velg en av dine egne stasjoner")
            if index is not None and sensor in self.added_sensors.get(index, ()):
                sensors = tuple(value for value in self.added_sensors[index] if value != sensor)
                if sensors:
                    self.added_sensors[index] = sensors
                else:
                    self.added.remove(index)
                    self.added_sensors.pop(index, None)
                    self.added_positions.pop(index, None)
                self.selected = self.frost_selection(station_id=station_id) if station_id else None
            elif station_id:
                catalog = self.engine.sensor_catalog()
                if (sensor not in catalog.get(station_id, ()) or
                        sensor in self.station_sensor_removals.get(station_id, ())):
                    raise ValueError("Stasjonen har ikke denne sensoren")
                self.station_sensor_removals[station_id] = (
                    *self.station_sensor_removals.get(station_id, ()), sensor)
            else:
                raise ValueError("Stasjonen har ikke denne sensoren")
            self.sync_edits()
        elif action == "add_sensor":
            station_id = str(data.get("station_id", ""))
            original = self.frost_station(station_id) if self.network == "frost" else None
            if original is None:
                raise ValueError("Velg en stasjon som finnes i nettet")
            sensors = data.get("sensors")
            if (not isinstance(sensors, list) or not sensors or
                    any(sensor not in PHENOMENON_DATASETS for sensor in sensors)):
                raise ValueError("Velg minst én gyldig sensor")
            catalog = self.engine.sensor_catalog() if hasattr(self.engine, "sensor_catalog") else {}
            removed_sensors = self.station_sensor_removals.get(station_id, ())
            if any(sensor in catalog.get(station_id, []) and sensor not in removed_sensors for sensor in sensors):
                raise ValueError("Stasjonen har allerede en av de valgte sensorene")
            edit = self.station_edits.get(station_id, {})
            if edit.get("removed"):
                raise ValueError("Gjenopprett stasjonen før du legger til sensorer")
            if "candidate_index" in edit:
                position = self.engine.station(edit["candidate_index"])
                longitude = edit.get("display_lon", position["lon"])
                latitude = edit.get("display_lat", position["lat"])
            else:
                longitude, latitude = (float(original.lon), float(original.lat)) if (
                    "lon" in original and "lat" in original) else self.engine.to_lonlat.transform(original.x, original.y)
            index = self.engine.snap(longitude, latitude)
            if (any(sensor not in removed_sensors for sensor in sensors) and
                    index not in self.added and len(self.added) >= 40):
                raise ValueError("Du kan legge til maksimalt 40 egne stasjoner i dette scenariet")
            previous = self.added_sensors.get(index, ()) if index in self.added else ()
            merged = tuple(dict.fromkeys([*previous, *(sensor for sensor in sensors if sensor not in removed_sensors)]))
            if self.phenomenon() in merged:
                self.prediction(index)
            remaining = tuple(sensor for sensor in removed_sensors if sensor not in sensors)
            if remaining:
                self.station_sensor_removals[station_id] = remaining
            else:
                self.station_sensor_removals.pop(station_id, None)
            if merged:
                self.added_sensors[index] = merged
            if merged and index not in self.added:
                self.added_positions[index] = dict(lon=longitude, lat=latitude)
                self.added.append(index)
            self.sync_edits()
            self.selected = dict(kind="manual", index=index) if merged else self.frost_selection(station_id=station_id)
        elif action == "add":
            index = self.engine.snap(float(data["lon"]), float(data["lat"]))
            if index in self.added:
                raise ValueError("Det finnes allerede en foreslått stasjon i denne terrengruten")
            location = self.engine.station(index)
            fixed = self.fixed()
            if ((fixed["x"] == location["x"]) & (fixed["y"] == location["y"])).any():
                raise ValueError("Denne plasseringen finnes allerede i grunnettet")
            if len(self.added) >= 40:
                raise ValueError("Du kan legge til maksimalt 40 egne stasjoner i dette scenariet")
            sensors = data.get("sensors")
            if sensors is None:
                sensors = self.layer_phenomena()
            if (not isinstance(sensors, list) or not sensors or
                    any(sensor not in PHENOMENON_DATASETS for sensor in sensors)):
                raise ValueError("Velg minst én gyldig sensor for den nye stasjonen")
            self.added_sensors[index] = tuple(dict.fromkeys(sensors))
            if index in self.active_added([index]):
                self.prediction(index)
            self.added.append(index)
            self.added_positions[index] = dict(lon=float(data["lon"]), lat=float(data["lat"]))
        elif action == "move_manual":
            index = int(data["id"])
            if index not in self.added:
                raise ValueError("Velg en av dine egne stasjoner")
            longitude, latitude = float(data["lon"]), float(data["lat"])
            destination = self.engine.snap(longitude, latitude)
            if destination != index and destination in self.added:
                raise ValueError("Det finnes allerede et eget forslag i denne terrengruten")
            location = self.engine.station(destination)
            fixed = self.fixed()
            if destination != index and ((fixed["x"] == location["x"]) & (fixed["y"] == location["y"])).any():
                raise ValueError("Denne plasseringen finnes allerede i grunnettet")
            self.prediction(destination)
            sensors = self.added_sensors.get(index, tuple(self.layer_phenomena()))
            self.added[self.added.index(index)] = destination
            self.added_sensors.pop(index, None)
            self.added_positions.pop(index, None)
            self.added_sensors[destination] = sensors
            self.added_positions[destination] = dict(lon=longitude, lat=latitude)
            self.selected = dict(kind="manual", index=destination)
        elif action == "sensors":
            index = int(data["id"])
            if index not in self.added:
                raise ValueError("Velg en av dine egne stasjoner")
            sensors = data.get("sensors")
            if (not isinstance(sensors, list) or not sensors or
                    any(sensor not in PHENOMENON_DATASETS for sensor in sensors)):
                raise ValueError("Velg minst én gyldig sensor for stasjonen")
            if self.phenomenon() in sensors:
                self.prediction(index)
            self.added_sensors[index] = tuple(dict.fromkeys(sensors))
            self.selected = dict(kind="manual", index=index)
        elif action == "undo":
            if self.added:
                self.added.pop()
        elif action == "remove":
            if int(data["id"]) not in self.added:
                raise ValueError("Stasjonen finnes ikke blant de egne forslagene")
            self.added.remove(int(data["id"]))
        elif action in ("remove_frost", "move_frost"):
            verb = "fjerner" if action == "remove_frost" else "flytter"
            if self.network != "frost" or not self.ready():
                raise ValueError(f"Velg et ferdig beregnet Frost-nett før du {verb} en stasjon")
            if data.get("station_id"):
                edited_station = str(data["station_id"])
            else:
                index = int(data["index"])
                if not 0 <= index < len(self.profile()["stations"]) or index in self.masked():
                    raise ValueError("Frost-stasjonen finnes ikke i dette scenariet")
                edited_station = str(self.profile()["stations"].iloc[index]["station_id"])
            original = self.frost_station(edited_station)
            if original is None:
                raise ValueError("Frost-stasjonen finnes ikke i dette scenariet")
            existing_edit = self.station_edits.get(edited_station, {})
            proposal = self.attached_proposal(edited_station)
            if action == "remove_frost":
                if existing_edit.get("removed"):
                    raise ValueError("Frost-stasjonen er allerede fjernet")
                self.station_edits[edited_station] = dict(removed=True)
                if proposal is not None:
                    self.added.remove(proposal)
                    self.added_sensors.pop(proposal, None)
                    self.added_positions.pop(proposal, None)
            else:
                if existing_edit.get("removed"):
                    raise ValueError("Gjenopprett Frost-stasjonen før du flytter den igjen")
                candidate_index = self.engine.snap(float(data["lon"]), float(data["lat"]))
                location = self.engine.station(candidate_index)
                if (np.isclose(location["x"], original.x, rtol=0, atol=1e-6) and
                        np.isclose(location["y"], original.y, rtol=0, atol=1e-6)):
                    raise ValueError("Velg en annen terrengrute for den flyttede stasjonen")
                fixed = self.fixed()
                other_fixed = fixed.loc[fixed["station_id"].astype(str) != edited_station]
                if ((other_fixed.x == location["x"]) & (other_fixed.y == location["y"])).any():
                    raise ValueError("En Frost-stasjon finnes allerede på denne plasseringen")
                for added in self.added:
                    if added == proposal:
                        continue
                    existing = self.engine.station(added)
                    if existing["x"] == location["x"] and existing["y"] == location["y"]:
                        raise ValueError("Et eget forslag finnes allerede på denne plasseringen")
                if proposal is not None:
                    if self.phenomenon() in self.added_sensors[proposal]:
                        self.prediction(candidate_index)
                    self.added[self.added.index(proposal)] = candidate_index
                    self.added_sensors[candidate_index] = self.added_sensors.pop(proposal)
                    self.added_positions.pop(proposal, None)
                    self.added_positions[candidate_index] = dict(lon=float(data["lon"]), lat=float(data["lat"]))
                self.station_edits[edited_station] = dict(candidate_index=candidate_index, removed=False,
                                                         display_lon=float(data["lon"]), display_lat=float(data["lat"]))
            self.sync_edits()
        elif action == "restore_frost":
            station_id = data.get("station_id")
            if station_id is not None:
                station_id = str(station_id)
                if station_id not in self.station_edits and not self.station_sensor_removals.get(station_id):
                    raise ValueError("Denne stasjonen er ikke endret")
                self.restore_attached_proposals([station_id])
                self.station_edits.pop(station_id, None)
                self.station_sensor_removals.pop(station_id, None)
                self.sync_edits()
                self.selected = self.frost_selection(station_id=station_id)
            else:
                if not self.station_edits and not any(self.station_sensor_removals.values()):
                    raise ValueError("Frost-nettet er ikke endret")
                self.restore_attached_proposals(self.station_edits)
                self.station_edits.clear()
                self.station_sensor_removals.clear()
                self.frost_edits.clear()
                self.selected = None
        elif action == "reset_defaults":
            defaults = Scenario(self.engine, baseline=self.initial_baseline, dataset=self.initial_dataset)
            for name, value in defaults.__dict__.items():
                if name not in ("engine", "lock", "revision", "updated"):
                    setattr(self, name, value)
        elif action == "reset":
            self.added.clear()
        elif action == "baseline":
            baseline = int(data["count"])
            if not 0 <= baseline <= len(self.engine.stations):
                raise ValueError("Stasjonsantallet er utenfor den lagrede stasjonsrekken")
            self.baseline = baseline
            self.network = "greedy"
            self.added.clear()
            self.frost_edits.clear()
            self.station_edits.clear()
            self.station_sensor_removals.clear()
        elif action == "network":
            network = data["value"]
            pross_only = data.get("pross_only", self.pross_only)
            if network == "svv":
                network, pross_only = "frost", True
            if network not in ("frost", "greedy", "empty"):
                raise ValueError("Ukjent grunnett")
            if not isinstance(pross_only, bool):
                raise ValueError("Ugyldig valg av stasjonsnett")
            if network == "frost" and self.profile() is None:
                raise ValueError("Ingen Frost-database er tilgjengelig")
            self.network = network
            self.pross_only = pross_only
            self.added.clear()
            self.frost_edits.clear()
            self.station_edits.clear()
            self.station_sensor_removals.clear()
        elif action == "dataset":
            profile = self.engine.weather.profile(data["value"])
            self.dataset = profile["id"]
            self.layers = [self.dataset]
            self.sync_edits()
            self.selected = None
            self.threshold = default_threshold(profile)
        elif action == "layers":
            phenomena = data.get("value")
            if getattr(self.engine, "weather", None) is None:
                raise ValueError("Ingen værmodeller er tilgjengelige")
            if not isinstance(phenomena, list):
                raise ValueError("Værfenomen må være en liste")
            available = {item["id"] for item in self.engine.weather.available}
            layers = []
            for phenomenon in phenomena:
                # if phenomenon not in PHENOMENON_DATASETS:
                #     raise ValueError("Ukjent værfenomen")
                # if PHENOMENON_DATASETS[phenomenon] not in available:
                #     raise ValueError("Ingen ferdig modell for valgt værfenomen")
                if PHENOMENON_DATASETS[phenomenon] not in layers:
                    layers.append(PHENOMENON_DATASETS[phenomenon])
            for dataset in layers:
                self.engine.weather.profile(dataset)
            if self.dataset in layers:
                layers.remove(self.dataset)
                layers.insert(0, self.dataset)
            elif layers:
                profile = self.engine.weather.profile(layers[0])
                self.dataset = profile["id"]
                self.selected = None
                self.threshold = default_threshold(profile)
            self.layers = layers
            self.sync_edits()
        elif action == "combine":
            if data.get("value") not in ("sum", "rss"):
                raise ValueError("Ukjent kombinasjonsmetode")
            self.combine = data["value"]
        elif action == "pross_mask":
            if self.network != "frost":
                raise ValueError("SVV-filteret krever MET-stasjonsnettet")
            self.pross_only = bool(data["value"])
            self.sync_edits()
            self.selected = None
        elif action == "station_since":
            value = self.validate_station_date(data.get("value", ""))
            if data.get("expected_revision", self.revision) != self.revision:
                raise ValueError("Nettet er endret. Tell stasjonene på nytt før du bekrefter")
            self.station_since = value
            self.sync_edits()
            self.selected = None
        elif action == "counties":
            self.counties = data["value"]
        elif action == "threshold":
            threshold = float(data["value"])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError("Terskelen må være et endelig tall som er null eller større")
            self.threshold = threshold
        elif action == "comparison":
            if data["value"] not in ("baseline", "selected"):
                raise ValueError("Ukjent sammenligning")
            self.comparison_scope = data["value"]
        elif action == "resolution":
            engine = self.engine.at_resolution(int(data["cell_m"]))
            if engine.weather is not None:
                for dataset in self.layers:
                    engine.weather.profile(dataset)
            self.engine = engine
            self.sync_edits()
        elif action == "select":
            kind = data["kind"]
            if kind == "frost" and self.network == "frost" and data.get("station_id"):
                station_id = str(data["station_id"])
                selection = self.frost_selection(station_id=station_id)
                known = selection["index"] is not None
                for dataset in self.layers[1:]:
                    with self.as_dataset(dataset):
                        known = known or (self.ready() and self.frost_selection(station_id=station_id)["index"] is not None)
                known = known or self.frost_station(station_id) is not None
                if not known:
                    raise ValueError("Velg en stasjon som finnes i nettet")
                self.selected = selection
            else:
                index = int(data["index"])
                if not ((kind == "manual" and index in self.added) or
                        (kind == "greedy" and self.network == "greedy" and 0 <= index < self.baseline) or
                        (kind == "frost" and self.network == "frost" and 0 <= index < len(self.profile()["stations"])
                         and index not in self.masked())):
                    raise ValueError("Velg en stasjon som finnes i nettet")
                self.selected = self.frost_selection(index) if kind == "frost" else dict(kind=kind, index=index)
        else:
            raise ValueError("Ukjent scenariohandling")
        if action in ("add", "undo", "remove", "remove_frost", "restore_frost", "reset", "baseline", "network"):
            self.selected = None
        if action in ("move_frost", "remove_frost"):
            self.selected = self.frost_selection(station_id=edited_station)
        self.revision += 1
        self.updated = time.monotonic()

    def state(self):
        current, before, reduction = self.change()
        profile = self.profile()
        fixed = self.fixed()
        visible = self.fixed(include_removed=True)
        lon, lat = self.engine.to_lonlat.transform(visible["x"].to_numpy(), visible["y"].to_numpy())
        markers = []
        def display_position(row):
            if "lon" in row and "lat" in row:
                return float(row["lon"]), float(row["lat"])
            return self.engine.to_lonlat.transform(row["x"], row["y"])

        def add_calculation_position(marker, row, active_profile, edit):
            spacing = getattr(self.engine, "candidate_cell_m", 1000) if edit and not edit.get("removed") else active_profile.get("source_grid_m")
            if spacing is not None:
                if edit and not edit.get("removed"):
                    row = self.engine.station(edit["candidate_index"])
                x, y = (row["x"], row["y"]) if edit and not edit.get("removed") else (
                    row.get("calculation_x", row["x"]), row.get("calculation_y", row["y"]))
                longitude, latitude = self.engine.to_lonlat.transform(x, y)
                marker.setdefault("calculation_positions", []).append(dict(
                    dataset=active_profile["id"], lon=float(longitude), lat=float(latitude), cell_m=spacing))

        for rank, (longitude, latitude, row) in enumerate(zip(lon, lat, visible.to_dict("records"))):
            if self.network == "frost":
                longitude, latitude = display_position(row)
            source_index = int(row.get("source_index", rank))
            markers.append(dict(lon=float(longitude), lat=float(latitude), rank=rank + 1, index=source_index,
                                removed=bool(self.station_edits.get(str(row.get("station_id", "")), {}).get("removed")) if self.network == "frost" else False,
                                kind="observation" if self.network == "frost" else "proposed",
                                station_id=str(row.get("station_id", "")), name=str(row.get("name", "")),
                                first_time=str(row.get("first_time", "")), last_time=str(row.get("last_time", ""))))
            edit = self.station_edits.get(str(row.get("station_id", ""))) if self.network == "frost" else None
            if self.network == "frost":
                add_calculation_position(markers[-1], row, profile, edit)
            if edit is not None and not edit.get("removed"):
                original = profile["stations"].iloc[source_index]
                original_lon, original_lat = display_position(original)
                markers[-1]["original_position"] = dict(lon=float(original_lon), lat=float(original_lat))
                markers[-1]["lon"] = edit.get("display_lon", markers[-1]["lon"])
                markers[-1]["lat"] = edit.get("display_lat", markers[-1]["lat"])
        catalog = self.engine.sensor_catalog() if hasattr(self.engine, "sensor_catalog") else {}
        primary = self.phenomenon()
        for marker in markers:
            marker.update(editable=True, layers=[] if primary is None else [primary])
        if self.network == "frost":
            by_id = {marker["station_id"]: marker for marker in markers}
            for dataset in self.station_datasets():
                if dataset == self.dataset:
                    continue
                with self.as_dataset(dataset):
                    phenomenon, stations = self.phenomenon(), self.fixed(include_removed=True)
                    layer_profile = self.profile()
                for row in stations.to_dict("records"):
                    station_id = str(row["station_id"])
                    marker = by_id.get(station_id)
                    edit = self.station_edits.get(station_id, {})
                    if marker is None:
                        longitude, latitude = display_position(row)
                        marker = dict(lon=float(longitude), lat=float(latitude), rank=None, index=None,
                                      removed=bool(edit.get("removed")),
                                      kind="observation", station_id=station_id, name=str(row["name"]),
                                      first_time=str(row.get("first_time", "")), last_time=str(row.get("last_time", "")),
                                      editable=True, layers=[])
                        if edit and not edit.get("removed"):
                            original = self.frost_station(station_id)
                            original_lon, original_lat = display_position(original)
                            marker["original_position"] = dict(lon=float(original_lon), lat=float(original_lat))
                            marker["lon"] = edit.get("display_lon", marker["lon"])
                            marker["lat"] = edit.get("display_lat", marker["lat"])
                        markers.append(marker)
                        by_id[station_id] = marker
                    add_calculation_position(marker, row, layer_profile, edit)
                    if dataset in self.layers:
                        marker["layers"].append(phenomenon)
        for marker in markers:
            removed_sensors = self.station_sensor_removals.get(marker["station_id"], ())
            marker["layers"] = [sensor for sensor in marker["layers"] if sensor not in removed_sensors]
            marker["sensors"] = [sensor for sensor in dict.fromkeys(
                catalog.get(marker["station_id"], []) + marker["layers"]) if sensor not in removed_sensors]
            marker["sensors_changed"] = bool(removed_sensors)
            marker["svv"] = False
        if markers:
            for marker, svv in zip(markers, svv_mask([marker["name"] for marker in markers])):
                marker["svv"] = bool(svv)
        from station_owners import owner_counts
        owner_catalog = self.engine.station_owner_catalog() if hasattr(self.engine, "station_owner_catalog") else None
        for marker in markers:
            marker["owners"] = ([] if owner_catalog is None else
                                owner_catalog["stations"].get(marker["station_id"], {}).get("owners", []))
        owners = owner_counts(markers)
        owners.update(updated=None if owner_catalog is None else owner_catalog["updated"],
                      available=owner_catalog is not None)
        minimum, maximum, reduction_max = self.limits()
        selected = self.selection()
        selected_label = None
        if selected is not None:
            if selected["kind"] == "frost" and selected["index"] is None:
                marker = next((item for item in markers if item["station_id"] == selected.get("station_id")), None)
                selected_label = str(selected.get("station_id")) + ("" if marker is None else " · " + marker["name"])
            elif selected["kind"] == "frost":
                row = profile["stations"].iloc[selected["index"]]
                selected_label = str(row["station_id"]) + " · " + str(row["name"])
            else:
                number = selected["index"] + 1 if selected["kind"] == "greedy" else self.added.index(selected["index"]) + 1
                selected_label = ("Foreslått stasjon " if selected["kind"] == "greedy" else "Ny stasjon ") + str(number)
        status = None if profile is None else dict(profile.get("status", {"state": "ready", "percent": 100}))
        frost_edits = []
        for station_id in sorted(set(self.station_edits) | {
                station_id for station_id, sensors in self.station_sensor_removals.items() if sensors}):
            edit = self.station_edits.get(station_id, {})
            marker = next((item for item in markers if item["station_id"] == station_id), None)
            frost_edits.append(dict(index=None if marker is None else marker["index"], removed=bool(edit.get("removed")),
                                    station_id=station_id, name="" if marker is None else marker["name"]))
            if self.station_sensor_removals.get(station_id):
                frost_edits[-1]["removed_sensors"] = list(self.station_sensor_removals[station_id])
        display_mask = (self.county_layer.cells(self.engine, self.counties)
                        if self.counties is not None and hasattr(self, "county_layer") else None)
        def average(values):
            if values is None:
                return None
            if display_mask is not None:
                values = values[display_mask]
                values = values[np.isfinite(values)]
                return float(values.mean()) if len(values) else None
            if profile is not None and profile.get("allow_nodata"):
                finite = values[np.isfinite(values)]
                return float(finite.mean()) if len(finite) else None
            return float(values.mean())

        def manual_station(index):
            location = self.engine.station(index)
            return dict(location, **self.added_positions.get(index, {}),
                        calculation_positions=[dict(lon=location["lon"], lat=location["lat"],
                                                    cell_m=getattr(self.engine, "candidate_cell_m", 1000))],
                        sensors=list(self.added_sensors.get(index, self.layer_phenomena())))

        return dict(
            station_owners=owners,
            revision=self.revision, baseline=self.baseline, maximum_baseline=len(self.engine.stations),
            threshold=self.threshold, cell_m=self.engine.cell_m, bounds=self.engine.bounds,
            resolutions=[1000, 2500, 5000] if hasattr(self.engine, "at_resolution") else [self.engine.cell_m],
            dataset=self.dataset, layers=list(self.layers),
            layer_labels=[] if profile is None else [next(item["label"] for item in self.engine.weather.available
                                                          if item["id"] == dataset) for dataset in self.layers],
            layer_options=[] if profile is None else [
                dict(phenomenon=phenomenon, dataset=dataset,
                     available=any(item["id"] == dataset for item in self.engine.weather.available))
                for phenomenon, dataset in PHENOMENON_DATASETS.items()],
            network=self.network, fixed_count=len(fixed), frost_edits=frost_edits,
            frost_ready=self.ready(), frost_precompute=status,
            precompute_jobs=self.engine.frost_statuses() if hasattr(self.engine, "frost_statuses") else [],
            proposed_design_note="Plasseringer og rekkefølge valgt med temperatur-RMSE; ikke optimalisert for denne værmodellen",
            weather_modes=[] if profile is None else [{key: item[key] for key in ("id", "phenomenon", "kind", "label", "metric", "unit")} for item in self.engine.weather.available],
            unit="\u00b0C" if profile is None else profile["unit"],
            metric="RMSE" if profile is None else profile["metric"],
            weather_label="Temperatur" if profile is None else profile["label"],
            model_grid_m=None if profile is None else profile.get("model_grid_m"),
            model_warning=None if profile is None else profile.get("model_warning"),
            colour_reference_warning=getattr(self, "_colour_reference_warning", None),
            training_period=None if profile is None else profile.get("training_period"),
            baseline_method=(profile["method"] + (f" · {len(frost_edits)} endringer i scenariet" if frost_edits else "")
                             if self.network == "frost" else
                             "Lagret foreslått nett" if self.network == "greedy" else "Tomt nett"),
            observation_count=len(fixed) if self.network == "frost" else 0 if profile is None else len(profile["stations"]),
            period_start=None if profile is None else profile["start"],
            period_end=None if profile is None else profile["end"],
            missing_station_features=0 if profile is None else profile["missing_features"],
            classes=None if profile is None else profile["classes"],
            color_min=minimum, color_max=maximum, reduction_max=reduction_max,
            comparison=self.comparison_kind(), comparison_scope=self.comparison_scope,
            comparison_method=self.comparison_method(),
            selected_station=selected, selected_station_label=selected_label,
            stations=[manual_station(index) for index in self.added],
            pross_only=self.pross_only, pross_available=True,
            station_since=self.station_since, counties=self.counties,
            masked_count=len(self.masked()), combine=self.combine,
            missing_layers=[] if profile is None else [
                dict(label=self.engine.weather.profile(dataset)["label"],
                     error=self.engine.weather.profile(dataset).get("status", {}).get("error"))
                for dataset in self.layers
                if self.engine.weather.profile(dataset).get("status", {}).get("state", "ready") != "ready"
                or self.engine.weather.profile(dataset).get("baseline") is None],
            fixed_stations=markers,
            mean_rmse=average(current),
            previous_rmse=average(before),
            mean_reduction=average(reduction),
            affected_area_m2=None if reduction is None else int(np.count_nonzero(self.affected(reduction) &
                (True if display_mask is None else display_mask))) * self.engine.cell_m ** 2,
            affected_area_km2=None if reduction is None else float(np.count_nonzero(self.affected(reduction) &
                (True if display_mask is None else display_mask))) * self.engine.cell_m ** 2 / 1_000_000,
        )

    def export(self, cell_mask=None):
        import geopandas as gpd
        import pyogrio
        import rasterio

        if not self.ready():
            raise ValueError("Den eksakte Frost-grunnflaten er ikke ferdig forhåndsberegnet")
        current, _, reduction = self.change()
        if current is None:
            raise ValueError("Legg til en stasjon før du eksporterer et tomt nett")
        engine = self.engine
        profile = self.profile()
        unit = "degrees C" if profile is None else profile["unit"]
        fixed = self.fixed().copy()
        fields = [key for key in ("station_id", "name", "x", "y", "first_time", "last_time") if key in fixed]
        fixed = fixed[fields]
        fixed["kind"] = "observation" if self.network == "frost" else "proposed"
        raster_metadata = dict(station_count=len(fixed) + len(self.added),
                               comparison=self.comparison_kind(), comparison_scope=self.comparison_scope,
                               threshold=self.threshold, weather_mode=self.dataset, cell_size_m=engine.cell_m)
        if profile is not None and profile.get("allow_nodata"):
            raster_metadata["model_warning"] = profile["model_warning"]
        buffer = BytesIO()
        with tempfile.TemporaryDirectory(prefix="station_scenario_") as directory:
            root = Path(directory)
            layers = {"uncertainty": current}
            if reduction is not None:
                layers["selected_reduction"] = reduction
                layers["selected_affected"] = self.affected(reduction).astype(np.float32)
            for name, values in layers.items():
                if cell_mask is not None:
                    values = np.where(cell_mask, values, -9999)
                raster = np.full(engine.raster_shape, -9999, dtype=np.float32)
                raster[engine.rows, engine.columns] = (
                    np.where(np.isfinite(values), values, -9999)
                    if profile is not None and profile.get("allow_nodata") else values)
                with rasterio.open(root / f"{name}.tif", "w", driver="GTiff", dtype="float32",
                                   count=1, width=raster.shape[1], height=raster.shape[0],
                                   crs="EPSG:25833", transform=engine.transform,
                                   nodata=-9999, compress="deflate") as dataset:
                    dataset.write(raster, 1)
                    units = "binary 0/1" if name == "selected_affected" else unit
                    dataset.update_tags(units=units, **raster_metadata)
            manual = pd.DataFrame([engine.station(index) for index in self.added],
                                  columns=["id", "x", "y", "lon", "lat", "elevation"])
            manual["kind"] = "manual"
            stations = pd.concat([fixed, manual], ignore_index=True)
            stations["rank"] = np.arange(1, len(stations) + 1)
            stations.to_csv(root / "stations.csv", index=False)
            features = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.x, stations.y), crs=25833)
            pyogrio.write_dataframe(features, root / "stations.gpkg", layer="stations", driver="GPKG")
            (root / "scenario.json").write_text(json.dumps(dict(
                dataset=self.dataset, unit=unit, threshold=self.threshold, network=self.network,
                model_warning=self.state()["model_warning"], training_period=self.state()["training_period"],
                baseline_method=self.state()["baseline_method"],
                selected_station=self.selection(), affected_area_km2=self.state()["affected_area_km2"],
                frost_edits=self.state()["frost_edits"],
                cell_size_m=engine.cell_m,
                comparison_scope=self.comparison_scope, contribution_method=self.comparison_method(),
                fixed_stations=len(self.fixed()), manual_stations=len(self.added)), indent=2), encoding="utf-8")
            with ZipFile(buffer, "w") as archive:
                for path in root.iterdir():
                    archive.write(path, path.name)
        buffer.seek(0)
        return buffer


def create_app(engine, road_network=None, initial_dataset="temperature:continuous", kuldehull_layer=None):
    from station_coverage import CoverageLayer
    coverage = CoverageLayer(ASSET_ROOT, HERE)
    from station_climate import ClimateZones
    climate = ClimateZones(HERE / "data" / "klimasoner.tif", OUT / "station_app_cache" / "climate")
    from station_counties import CountyMask
    county_layer = CountyMask(HERE / "data" / "Basisdata_0000_Norge_25833_Fylker_FGDB.gdb")
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict", MAX_CONTENT_LENGTH=4096)
    if (OUT / "portable_weather.json").is_file():
        app.add_url_rule("/api/footprint/available", "portable_footprint_unavailable",
                         lambda: jsonify(available=False))
    else:
        from station_footprint import register_footprint_view
        register_footprint_view(app, HERE)
    scenarios = {}
    registry_lock = threading.Lock()
    road_lock = threading.Lock()
    map_images = OrderedDict()
    image_lock = threading.Lock()
    cold_basins = (kuldehull_layer if kuldehull_layer is not None else
                   KuldehullLayer(OUT / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg"))

    def roads():
        nonlocal road_network
        with road_lock:
            if road_network is None:
                source = HERE / "data" / "ProssExport.gdb"
                portable_path = OUT / "portable_weather.json"
                if portable_path.is_file():
                    bundled = json.loads(portable_path.read_text(encoding="utf-8"))
                    if bundled.get("roads", {}).get("indexed_source"):
                        source = HERE / bundled["roads"]["indexed_source"]
                road_network = RoadNetwork(source, cache_dir=OUT / "station_app_cache" / "roads",
                                           portable_cache=engine.cell_m == 500)
            return road_network

    def road_mask(current, settings):
        county = county_layer.cells(current.engine, current.counties) if current.counties is not None else None
        if settings is None or not settings.get("corridor", False):
            return county
        if current.counties is not None:
            settings = dict(settings, _county_key=tuple(current.counties),
                            _county_geometry=county_layer.geometry(tuple(current.counties)))
        road = roads().corridor_mask(current.engine, settings["bounds"], settings,
                                      settings.get("distance_m", 1000))
        return road if county is None else road & county

    def county_image(current, image, kuldehull=None, active=None):
        if current.counties is None:
            return image
        active = current.engine if active is None else active
        bounds = (kuldehull["viewport"]["bounds"] if kuldehull and kuldehull.get("only") and
                  "viewport" in kuldehull else
                  [active.bounds[0][1], active.bounds[0][0], active.bounds[1][1], active.bounds[1][0]])
        return county_layer.clip_image(image, bounds, current.counties)

    def minimum_rmse(value):
        try:
            result = float(value)
        except (ValueError, TypeError):
            raise ValueError("Ugyldig RMSE-grense") from None
        if not np.isfinite(result) or result < 0:
            raise ValueError("RMSE-grensen må være endelig og ikke negativ")
        return result

    def draw(current, values, mode, settings=None, display_size=None, kuldehull=None,
             rmse_min=0, rmse_values=None):
        mask = road_mask(current, settings)
        if rmse_min > 0 and not mode.startswith("terrain:"):
            errors = rmse_values if rmse_values is not None else (
                values if mode in ("uncertainty", "baseline") else current.surface())
            error_mask = (np.ones(len(current.engine.grid), dtype=bool) if errors is None else
                          ~np.isfinite(errors) | (errors > rmse_min))
            mask = error_mask if mask is None else mask & error_mask
        if mode in ("reduction", "affected"):
            values, mode = current.map_change(values, mode)
        limits = () if mode.startswith("terrain:") else current.limits(mask)
        options = {} if display_size is None else dict(display_size=display_size)
        if mask is None:
            image = current.engine.render(values, mode, *limits, **options)
        else:
            image = current.engine.render(values, mode, *limits, cell_mask=mask, **options)
        if kuldehull and kuldehull["only"] and not mode.startswith("terrain:"):
            image = cold_basins.clip(image, current.engine, kuldehull["depth"],
                                     **({"viewport": kuldehull["viewport"]} if "viewport" in kuldehull else {}))
        return county_image(current, image, kuldehull)

    def parse_road_settings(settings):
        if settings is None:
            return None
        if len(settings) > 4096:
            raise ValueError("Vegfilteret er for stort")
        try:
            settings = json.loads(settings)
        except ValueError:
            raise ValueError("Ugyldig vegfilter") from None
        if not isinstance(settings, dict):
            raise ValueError("Vegfilteret må være et JSON-objekt")
        return settings

    def combined_image(current, mode, settings, display_size, kuldehull):
        values, kind = current.combined(mode)
        mask = road_mask(current, parse_road_settings(settings))
        cutoff = minimum_rmse(request.args.get("rmse_min", 0))
        if cutoff > 0:
            errors = values if mode in ("uncertainty", "baseline") else current.combined("uncertainty")[0]
            error_mask = (np.ones(len(current.engine.grid), dtype=bool) if errors is None else
                          ~np.isfinite(errors) | (errors > cutoff))
            mask = error_mask if mask is None else mask & error_mask
        options = dict(cell_mask=mask) if mask is not None else {}
        if display_size is not None:
            options["display_size"] = display_size
        image = current.engine.render(values, kind, *combined_range(values, kind), 1, **options)
        if kuldehull and kuldehull["only"]:
            image = cold_basins.clip(image, current.engine, kuldehull["depth"],
                                     **({"viewport": kuldehull["viewport"]} if "viewport" in kuldehull else {}))
        response = send_file(county_image(current, image, kuldehull), mimetype="image/png", max_age=0)
        response.headers["X-Map-Cache"] = "MISS"
        response.headers["Cache-Control"] = "no-store"
        return response

    def combined_placement(current):
        active = current.engine.at_resolution(5000)
        parts = []
        for dataset in current.layers:
            try:
                layer = active.placement_layer(dataset)
            except (OSError, ValueError, KeyError):
                continue
            parts.append(np.clip(np.log1p(np.maximum(layer["values"], 0)) / np.log1p(layer["maximum"]), 0, 1))
        if not parts:
            raise ValueError("Ingen lagret plasseringsgevinst for valgte værfenomen")
        return active, combine_layers(parts, current.combine)

    @app.get("/api/combined-range")
    def combined_value_range():
        mode = request.args.get("mode", "uncertainty")
        if mode not in ("uncertainty", "baseline", "reduction", "affected", "placement"):
            raise ValueError("Ukjent kartlag")
        current = scenario()
        with current.lock:
            if mode == "placement":
                active, values = combined_placement(current)
                kind = "combined_reduction"
            else:
                values, kind = current.combined(mode)
                active = current.engine
            mask = county_layer.cells(active, current.counties) if current.counties is not None else None
            if mask is not None and values is not None:
                values = values[mask]
            minimum, maximum = combined_range(values, kind)
            return jsonify(minimum=minimum, maximum=maximum, kind=kind, method=current.combine,
                           revision=current.revision)

    def scenario():
        with registry_lock:
            now = time.monotonic()
            expired = [key for key, value in scenarios.items() if now - value.updated > 21600]
            for key in expired:
                del scenarios[key]
            identifier = session.get("scenario")
            if identifier not in scenarios:
                if len(scenarios) >= 16:
                    raise ValueError("For mange åpne scenarier. Lukk nettleserøkter som ikke er i bruk")
                identifier = secrets.token_urlsafe(24)
                session["scenario"] = identifier
                scenarios[identifier] = Scenario(engine, dataset=initial_dataset)
                scenarios[identifier].county_layer = county_layer
            result = scenarios[identifier]
            result.updated = now
            return result

    @app.get("/")
    def index():
        return send_file(ASSET_ROOT / "station_app.html")

    @app.get("/station_app.js")
    def javascript():
        return send_file(ASSET_ROOT / "station_app.js", mimetype="text/javascript")

    @app.get("/api/state")
    def state():
        current = scenario()
        with current.lock:
            return jsonify(current.state())

    @app.get("/api/precompute")
    def precompute_status():
        from calculation_progress import read_progress
        current = scenario()
        with current.lock:
            current.profile()
            return jsonify(jobs=engine.frost_statuses() if hasattr(engine, "frost_statuses") else [],
                           calculation=read_progress(HERE))

    @app.get("/api/placement")
    def placement_metadata():
        current = scenario()
        with current.lock:
            active = current.engine.at_resolution(5000)
            layer = active.placement_layer(current.dataset)
            return jsonify({key: value for key, value in layer.items() if key != "values"})

    @app.get("/api/counties")
    def county_options():
        return jsonify(counties=county_layer.options())

    @app.get("/api/counties/borders")
    def county_borders():
        response = jsonify(county_layer.borders())
        response.vary.add("Accept-Encoding")
        if request.accept_encodings["gzip"]:
            response.set_data(gzip.compress(response.get_data(), compresslevel=1))
            response.headers["Content-Encoding"] = "gzip"
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response

    @app.get("/api/placement.png")
    def placement_image():
        current = scenario()
        size = request.args.get("size", 800, type=int)
        if size not in (400, 800, 1600):
            raise ValueError("Ugyldig bildeoppløsning")
        with current.lock:
            active = current.engine.at_resolution(5000)
            if len(current.layers) > 1:
                active, total = combined_placement(current)
                response = send_file(county_image(current, active.render(total, "combined_reduction",
                                                   *combined_range(total, "combined_reduction"), 1, display_size=size), active=active),
                                     mimetype="image/png", max_age=0)
                response.headers["Cache-Control"] = "no-store"
                return response
            layer = active.placement_layer(current.dataset)
            key = ("placement", id(active), id(layer["values"]), size)
            with image_lock:
                hit = key in map_images
                if not hit:
                    map_images[key] = active.render(layer["values"], "placement", layer["minimum"],
                                                   layer["maximum"], display_size=size).getvalue()
                    if len(map_images) > 16:
                        map_images.popitem(last=False)
                map_images.move_to_end(key)
                response = send_file(county_image(current, BytesIO(map_images[key]), active=active), mimetype="image/png", max_age=0)
            response.headers["X-Map-Cache"] = "HIT" if hit else "MISS"
            response.headers["Cache-Control"] = "no-store"
            return response

    @app.get("/api/climate/options")
    def climate_options():
        return jsonify(climate.metadata())

    @app.get("/api/climate/point")
    def climate_point():
        return jsonify(climate.point(float(request.args["lon"]), float(request.args["lat"])))

    @app.get("/api/climate.png")
    def climate_image():
        size = request.args.get("size", 1600, type=int)
        image = climate.image(size)
        current = scenario()
        with current.lock:
            settings = request.args.get("roads")
            if settings is not None:
                settings = json.loads(settings)
                if not isinstance(settings, dict):
                    raise ValueError("Vegfilteret må være et JSON-objekt")
                if settings.get("corridor"):
                    mask = road_mask(current, settings)
                    if mask is not None:
                        import geopandas as gpd
                        from shapely import box
                        grid = current.engine.grid
                        half = current.engine.cell_m / 2
                        x, y = grid.x.to_numpy()[mask], grid.y.to_numpy()[mask]
                        polygons = gpd.GeoDataFrame(geometry=box(x-half, y-half, x+half, y+half), crs=25833)
                        image = climate.clip(image, polygons, size)
            south, west = climate.metadata()["bounds"][0]
            north, east = climate.metadata()["bounds"][1]
            image = county_layer.clip_image(image, [west, south, east, north], current.counties)
        response = send_file(image, mimetype="image/png", max_age=0)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/terrain/options")
    def terrain_options():
        current = scenario()
        with current.lock:
            return jsonify(current.engine.terrain_catalog())

    @app.get("/api/terrain/point")
    def terrain_point():
        current = scenario()
        with current.lock:
            return jsonify(current.engine.terrain_point(float(request.args["lon"]), float(request.args["lat"])))

    @app.get("/api/stations/best")
    def best_station():
        current = scenario()
        with current.lock:
            return jsonify(current.best_station(float(request.args["lon"]), float(request.args["lat"])))

    @app.get("/api/roads/options")
    def road_options():
        return jsonify(roads().metadata)

    @app.post("/api/roads/components")
    def road_components():
        if request.headers.get("Origin") not in (None, request.host_url.rstrip("/")):
            return jsonify(error="Vegspørringer fra andre nettsteder er ikke tillatt"), 403
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("Vegspørringen må inneholde et JSON-objekt")
        current = scenario()
        with current.lock:
            codes = current.counties
        settings = dict(data)
        if codes is not None:
            settings["_county_key"] = tuple(codes)
        return jsonify(roads().components(settings))

    @app.get("/api/kuldehull/options")
    def kuldehull_options():
        return jsonify(cold_basins.metadata)

    @app.get("/api/coverage/options")
    def coverage_options():
        return jsonify(coverage.metadata())

    @app.get("/api/coverage.png")
    def coverage_image():
        bounds = json.loads(request.args["bounds"])
        classes = json.loads(request.args.get("classes", "[1,2,3,4,5]"))
        image = coverage.image(bounds, int(request.args.get("width", 1000)),
                               int(request.args.get("height", 800)), classes)
        current = scenario()
        with current.lock:
            if current.counties is not None:
                image = county_layer.clip_image(image, bounds, current.counties)
        return send_file(image, mimetype="image/png", max_age=0)

    @app.get("/api/coverage/point")
    def coverage_point():
        return jsonify(coverage.point(request.args["lon"], request.args["lat"],
                                      json.loads(request.args.get("classes", "[1,2,3,4,5]"))))

    @app.get("/api/kuldehull/point")
    def kuldehull_point():
        return jsonify(cold_basins.point(
            request.args["lon"], request.args["lat"], request.args.get("depth", 0),
        ))

    @app.get("/api/kuldehull.png")
    def kuldehull_image():
        bounds = json.loads(request.args["bounds"])
        width = int(request.args.get("width", 1000))
        height = int(request.args.get("height", 800))
        minimum_depth = cold_basins.depth(request.args.get("depth", 0))
        image = cold_basins.overlay(bounds, width, height, minimum_depth)
        current = scenario()
        with current.lock:
            if current.counties is not None:
                image = county_layer.clip_image(image, bounds, current.counties)
        return send_file(image, mimetype="image/png", max_age=0)

    @app.post("/api/roads/view")
    def road_view():
        if request.headers.get("Origin") not in (None, request.host_url.rstrip("/")):
            return jsonify(error="Vegspørringer fra andre nettsteder er ikke tillatt"), 403
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("Vegspørringen må inneholde et JSON-objekt")
        current = scenario()
        with current.lock:
            geometry_only = data.get("geometry_only") is True
            surface, reduction = None, None
            if not geometry_only:
                surface, _, reduction = current.change()
            active, revision, threshold = current.engine, current.revision, current.threshold
            if geometry_only:
                unit, metric, minimum, maximum = "", "VS_Total", None, None
            else:
                profile = current.profile()
                unit = "\u00b0C" if profile is None else profile["unit"]
                metric = "RMSE" if profile is None else profile["metric"]
                minimum, maximum, _ = current.limits()
            county_codes = current.counties
        settings = dict(data, colour_min=minimum, colour_max=maximum)
        if county_codes is not None:
            settings["_county_key"] = tuple(county_codes)
            settings["_county_geometry"] = county_layer.geometry(tuple(county_codes))
        result = roads().view(active, surface, reduction, threshold, data["bounds"], settings,
                      zoom=data.get("zoom", 8))
        result.update(revision=revision, cell_m=active.cell_m, unit=unit, metric=metric,
                      weather_ready=surface is not None, color_min=minimum, color_max=maximum)
        response = jsonify(result)
        response.vary.add("Accept-Encoding")
        if request.accept_encodings["gzip"] and len(response.get_data()) > 65536:
            response.set_data(gzip.compress(response.get_data(), compresslevel=1))
            response.headers["Content-Encoding"] = "gzip"
        return response

    @app.post("/api/station-date-preview")
    def station_date_preview():
        if request.headers.get("Origin") not in (None, request.host_url.rstrip("/")):
            return jsonify(error="Stasjonsspørringer fra andre nettsteder er ikke tillatt"), 403
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("Velg en gyldig dato")
        current = scenario()
        with current.lock:
            return jsonify(current.station_date_preview(data.get("value", "")))

    @app.post("/api/action")
    def action():
        if request.headers.get("Origin") not in (None, request.host_url.rstrip("/")):
            return jsonify(error="Endringer fra andre nettsteder er ikke tillatt"), 403
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("Forespørselen må inneholde en JSON-handling")
        current = scenario()
        with current.lock:
            if data.get("action") == "counties":
                selected = county_layer.selection(data.get("value"))
                data = dict(data, value=None if selected is None else list(selected))
            current.apply(data)
            return jsonify(current.state())

    @app.get("/api/scenario")
    def save_scenario():
        from station_scenario import export_scenario
        return jsonify(format="vaernett-scenario", version=1, scenario=export_scenario(scenario()))

    @app.post("/api/snapshot-data")
    def snapshot_data():
        from station_scenario import export_scenario, restore_scenario
        from station_snapshot import road_chunks, basin_chunks, coverage_chunks, weather_maps
        if request.headers.get("Origin") not in (None, request.host_url.rstrip("/")):
            return jsonify(error="Eksport fra andre nettsteder er ikke tillatt"), 403
        data = request.get_json()
        if not isinstance(data, dict) or not isinstance(data.get("roads"), dict):
            raise ValueError("Eksporten krever valgte vegfiltre")
        current = scenario()
        with current.lock:
            revision = data.get("revision")
            if revision != current.revision:
                raise ValueError("Scenariet ble endret. Prøv eksporten igjen.")
            frozen = restore_scenario(current, export_scenario(current), county_layer)
        settings = {key: value for key, value in data["roads"].items()
                    if key in ("categories", "classes", "reference", "corridor", "distance_m")}
        if not isinstance(data.get("basins_only", False), bool) or not isinstance(settings.get("corridor", False), bool):
            raise ValueError("Ugyldig lagfilter")
        network = roads()
        if not network.metadata["available"]:
            raise ValueError("Vegnett mangler")
        network.filters(settings)
        geometry = None if frozen.counties is None else county_layer.geometry(tuple(frozen.counties))
        # All data is prepared from the isolated scenario, never the active viewport.
        weather_settings = dict(settings, bounds=list(network.geographic_bounds))
        depth = cold_basins.depth(data.get("depth", 0))
        coverage_classes = coverage.classes(data.get("coverage_classes", []))
        result = dict(revision=revision,
            weather=weather_maps(frozen, draw, weather_settings, depth, bool(data.get("basins_only"))),
            roads=road_chunks(network, settings, geometry, frozen.counties),
            basins=basin_chunks(cold_basins, depth, geometry))
        result["coverage"] = (coverage_chunks(coverage, coverage_classes, geometry)
                              if coverage_classes else dict(chunks=[], count=0))
        result["compressed_bytes"] = sum(len(chunk["data"]) * 3 // 4
            for kind in ("roads", "basins", "coverage") for chunk in result[kind]["chunks"])
        return jsonify(result)

    @app.post("/api/scenario")
    def load_scenario():
        from station_scenario import restore_scenario
        request.max_content_length = 2 * 1024 * 1024
        if request.headers.get("Origin") not in (None, request.host_url.rstrip("/")):
            return jsonify(error="Endringer fra andre nettsteder er ikke tillatt"), 403
        if request.content_length is None or request.content_length > 2 * 1024 * 1024:
            raise ValueError("Scenariofilen er for stor")
        data = request.get_json()
        if not isinstance(data, dict) or data.get("format") != "vaernett-scenario" or data.get("version") != 1:
            raise ValueError("Ukjent scenarioformat eller versjon")
        current = scenario()
        with current.lock:
            restored = restore_scenario(current, data.get("scenario"), county_layer)
            result = restored.state()
            with registry_lock:
                scenarios[session["scenario"]] = restored
            return jsonify(result)

    @app.post("/api/rmse-range")
    def rmse_range():
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("RMSE-filteret må være et JSON-objekt")
        mode = data.get("mode", "uncertainty")
        if mode not in ("uncertainty", "baseline", "reduction", "affected"):
            raise ValueError("Ukjent kartlag")
        settings = data.get("roads")
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("Vegfilteret må være et JSON-objekt")
        current = scenario()
        with current.lock:
            values = ((current.reference() if current.network == "frost" else current.base())
                      if mode == "baseline" else current.surface())
            mask = road_mask(current, settings)
            finite = np.empty(0) if values is None else values[np.isfinite(values) &
                (np.ones(len(values), dtype=bool) if mask is None else mask)]
            return jsonify(maximum=float(finite.max()) if len(finite) else 0,
                           revision=current.revision, cells=len(finite))

    @app.get("/api/map.png")
    def image():
        mode = request.args.get("mode", "uncertainty")
        display_size = request.args.get("size", type=int)
        if "size" in request.args and display_size not in (400, 800, 1600):
            raise ValueError("Ugyldig bildeoppløsning")
        terrain = mode.startswith("terrain:")
        if not terrain and mode not in ("uncertainty", "baseline", "reduction", "affected"):
            raise ValueError("Ukjent kartlag")
        current = scenario()
        with current.lock:
            if not terrain and len(current.layers) > 1:
                return combined_image(current, mode, request.args.get("roads"), display_size,
                                      lake_settings(request.args.get("kuldehull")))
            if terrain:
                surface = current.engine.terrain_layer(mode.split(":", 1)[1])["values"]
            elif mode in ("uncertainty", "baseline"):
                if mode == "baseline" and current.network == "frost":
                    surface = current.reference()
                    surface = None if surface is None else surface.copy()
                else:
                    surface = current.base() if mode == "baseline" or not current.added else current.surface()
            else:
                surface, _, reduction = current.change()
                if mode == "reduction":
                    surface = reduction
                elif mode == "affected":
                    surface = reduction
            settings = request.args.get("roads")
            if settings is not None:
                if len(settings) > 4096:
                    raise ValueError("Vegfilteret er for stort")
                try:
                    settings = json.loads(settings)
                except ValueError:
                    raise ValueError("Ugyldig vegfilter") from None
                if not isinstance(settings, dict):
                    raise ValueError("Vegfilteret må være et JSON-objekt")
            kuldehull = lake_settings(request.args.get("kuldehull"))
            rmse_min = minimum_rmse(request.args.get("rmse_min", 0))
            reusable = (current.network == "frost" and mode in ("uncertainty", "baseline") and not current.station_since and current.counties is None and
                        (not current.pross_only or "network_excluded" in current.profile() or not current.masked()) and
                        (mode == "baseline" or (not current.frost_edits and not current.added)) and
                        settings is None and surface is not None and not rmse_min and not (kuldehull and kuldehull["only"]))
            cache_hit = False
            if reusable:
                key = (id(current.engine), id(current.profile()["baseline"]), display_size, current.limits())
                with image_lock:
                    cache_hit = key in map_images
                    if not cache_hit:
                        saved = (current.engine.saved_weather_map(current.profile(), display_size, current.limits())
                                 if hasattr(current.engine, "saved_weather_map") else None)
                        cache_hit = saved is not None
                        map_images[key] = (saved if saved is not None else
                                           draw(current, surface, mode, display_size=display_size).getvalue())
                        if len(map_images) > 16:
                            map_images.popitem(last=False)
                    map_images.move_to_end(key)
                    image_data = BytesIO(map_images[key])
            else:
                saved = None
                if (not rmse_min and not current.station_since and current.counties is None and settings is None and kuldehull and kuldehull["only"]
                        and current.network == "frost" and mode in ("uncertainty", "baseline")
                        and (not current.pross_only or "network_excluded" in current.profile() or not current.masked())
                        and (mode == "baseline" or (not current.frost_edits and not current.added))
                        and surface is not None and hasattr(current.engine, "saved_weather_map")):
                    saved = current.engine.saved_weather_map(current.profile(), display_size, current.limits())
                if saved is not None:
                    cache_hit = True
                    image_data = cold_basins.clip(
                        BytesIO(saved), current.engine, kuldehull["depth"],
                        **({"viewport": kuldehull["viewport"]} if "viewport" in kuldehull else {}),
                    )
                else:
                    image_data = draw(current, surface, mode, settings, display_size, kuldehull, rmse_min)
            response = send_file(image_data, mimetype="image/png", max_age=0)
            response.headers["X-Map-Cache"] = "HIT" if cache_hit else "MISS"
            response.headers["Cache-Control"] = "no-store"
            return response

    @app.post("/api/preview")
    def preview():
        if request.headers.get("Origin") not in (None, request.host_url.rstrip("/")):
            return jsonify(error="Forhåndsvisning fra andre nettsteder er ikke tillatt"), 403
        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("Forhåndsvisningen må inneholde koordinater i JSON-format")
        mode = data.get("mode", "affected")
        if mode not in ("uncertainty", "affected", "reduction"):
            raise ValueError("Ukjent kartlag for forhåndsvisning")
        current = scenario()
        with current.lock:
            revision = current.revision
            if not current.ready():
                return jsonify(error="Eksakt Frost-grunnflate mangler. Kjør precompute_frost.py separat"), 409
            if int(data.get("revision", revision)) != revision:
                return jsonify(error="Scenariet er endret"), 409
            active = current.engine
            index = active.snap(float(data["lon"]), float(data["lat"]))
            settings = data.get("roads")
            if settings is not None and not isinstance(settings, dict):
                raise ValueError("Vegfilteret må være et JSON-objekt")
            kuldehull = lake_settings(data.get("kuldehull"))
            rmse_min = minimum_rmse(data.get("rmse_min", 0))
            mask = road_mask(current, None if settings is None else {**settings, "corridor": True})
            targets = None if mask is None else np.flatnonzero(mask)
            if targets is None:
                before = current.surface()
                prediction = current.prediction(index)
            elif len(targets):
                before = current.surface_at(targets)
                prediction = (active.prediction_at(index, targets, current.dataset)
                              if getattr(active, "weather", None) is not None else active.prediction_at(index, targets))
            else:
                before = None
                prediction = np.empty(0, dtype=np.float32)
            after = prediction if before is None else np.minimum(before, prediction)
            if current.compares_baseline():
                before = current.reference()
                if targets is not None:
                    before = before[targets]
            reduction = None if before is None else np.maximum(before - after, 0)
            if current.compares_baseline():
                reduction = before - after
            values = after if mode == "uncertainty" else reduction
            if targets is None:
                image = draw(current, values, mode, settings, kuldehull=kuldehull,
                             rmse_min=rmse_min, rmse_values=after)
            else:
                if mode in ("reduction", "affected"):
                    values, mode = current.map_change(values, mode)
                display = np.zeros(len(active.grid), dtype=np.float32)
                if values is not None:
                    display[targets] = values
                if rmse_min > 0:
                    mask = mask.copy()
                    mask[targets] &= np.isfinite(after) & (after > rmse_min)
                image = active.render(display, mode, *current.limits(), cell_mask=mask)
                if kuldehull and kuldehull["only"]:
                    image = cold_basins.clip(image, active, kuldehull["depth"],
                                             **({"viewport": kuldehull["viewport"]} if "viewport" in kuldehull else {}))
            response = send_file(county_image(current, image, kuldehull), mimetype="image/png", max_age=0)
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Station"] = json.dumps(active.station(index))
            response.headers["X-Revision"] = str(revision)
            response.headers["X-Mean-Rmse"] = "" if not len(after) else str(float(after.mean()))
            response.headers["X-Mean-Reduction"] = "" if reduction is None or not len(reduction) else str(float(reduction.mean()))
            response.headers["X-Affected-Area-M2"] = "" if reduction is None or not len(reduction) else str(int(np.count_nonzero(current.affected(reduction))) * active.cell_m ** 2)
            response.headers["X-Affected-Area-Km2"] = "" if reduction is None or not len(reduction) else str(float(np.count_nonzero(current.affected(reduction))) * active.cell_m ** 2 / 1_000_000)
            response.headers["X-Preview-Scope"] = "network" if targets is None else "counties" if settings is None else "roads"
            response.headers["X-Evaluated-Cells"] = str(len(after))
            response.headers["X-Total-Cells"] = str(len(active.grid))
            return response

    @app.post("/api/export.gpkg")
    def export_geopackage():
        from station_rmse_export import export_rmse
        import geopandas as gpd
        from shapely.geometry import box

        data = request.get_json()
        if not isinstance(data, dict):
            raise ValueError("Eksportfilteret må være et JSON-objekt")
        mode = data.get("mode", "uncertainty")
        if mode not in ("uncertainty", "baseline", "reduction", "affected"):
            raise ValueError("Velg et RMSE-kartlag før eksport")
        settings = data.get("roads")
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("Vegfilteret må være et JSON-objekt")
        threshold = minimum_rmse(data.get("rmse_min", 0))
        kuldehull = lake_settings(data.get("kuldehull"))
        current = scenario()
        with current.lock:
            if not current.ready():
                raise ValueError("RMSE-kartet er ikke klart for eksport")
            if data.get("revision") != current.revision:
                raise ValueError("Scenariet er endret. Prøv eksporten på nytt")
            profile = current.profile()
            if profile is not None and profile["metric"] != "RMSE":
                raise ValueError("Velg et RMSE-kartlag før eksport")
            values = ((current.reference() if current.network == "frost" else current.base())
                      if mode == "baseline" else current.surface())
            if values is None:
                raise ValueError("RMSE-kartet mangler")
            mask = road_mask(current, settings)
            clip = None
            if kuldehull and kuldehull["only"]:
                if not cold_basins.prepare():
                    raise ValueError("Kuldehull-laget mangler")
                polygons = cold_basins.polygons
                clip = polygons.loc[polygons.max_depth_m > kuldehull["depth"], ["geometry"]]
                if "viewport" in kuldehull:
                    bounds = kuldehull["viewport"]["bounds"]
                    viewport = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=4326).to_crs(clip.crs)
                    clip = gpd.clip(clip, viewport)
            if current.counties is not None:
                county_clip = county_layer.polygons(current.counties)
                clip = county_clip if clip is None else gpd.clip(clip, county_clip.to_crs(clip.crs))
            result = export_rmse(current.engine, values, threshold, mask, clip,
                metadata=dict(dataset=current.dataset, mode=mode, revision=current.revision,
                              unit="degrees C" if profile is None else profile["unit"],
                              roads=json.dumps(settings), kuldehull=json.dumps(kuldehull),
                              counties=json.dumps(current.counties), station_since=current.station_since))
            return send_file(result, mimetype="application/geopackage+sqlite3", as_attachment=True,
                             download_name="norway_rmse.gpkg")

    @app.get("/api/export")
    def export():
        current = scenario()
        with current.lock:
            mask = county_layer.cells(current.engine, current.counties) if current.counties is not None else None
            return send_file(current.export(cell_mask=mask), mimetype="application/zip", as_attachment=True,
                             download_name="stasjonsforslag_arcgis.zip")

    @app.errorhandler(ValueError)
    @app.errorhandler(KeyError)
    @app.errorhandler(TypeError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    return app


def main():
    global HERE, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        help="Bruk en ferdig portable datamappe uten å kopiere den til prosjektet")
    parser.add_argument("--surfaces", type=Path, default=OUT / "uncertainty_surfaces_1000m.npy")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--phenomenon", choices=WEATHER, default="temperature",
                        help="Værfenomen ved oppstart (kontinuerlig modell)")
    parser.add_argument("--cell-m", type=int, choices=[500, 1000],
                        help="Evalueringsrutenett ved oppstart")
    parser.add_argument("--frost-neighbors", type=int, choices=[0], default=0,
                        help="Bare 0 støttes: eksakt minimum over alle kvalifiserte Frost-stasjoner")
    parser.add_argument("--footprint-only", action="store_true",
                        help="Start bare den eksperimentelle native1 km footprint-visningen, uten nasjonale scenarioberegninger")
    args = parser.parse_args()
    returned_stage = ASSET_ROOT / "outputs" / "portable_build" / "Remote_500m" / "data"
    if (args.data_root is None and not args.footprint_only and args.cell_m != 1000
            and not (OUT / "portable_weather.json").is_file()
            and (returned_stage / "outputs" / "portable_weather.json").is_file()):
        args.data_root = returned_stage
    if args.data_root is not None:
        root = args.data_root.resolve()
        manifest_path = root / "outputs" / "portable_weather.json"
        if not manifest_path.is_file():
            parser.error("--data-root må inneholde outputs/portable_weather.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        spacing = manifest["cell_m"]
        if spacing not in (500, 1000) or args.cell_m is not None and args.cell_m != spacing:
            parser.error("--cell-m må samsvare med datamappens oppløsning")
        if args.footprint_only:
            parser.error("--data-root støtter ikke --footprint-only")
        HERE, OUT = root, root / "outputs"
        args.cell_m = spacing
    elif args.cell_m is None:
        args.cell_m = 1000
    print(f"Datamappe: {HERE} · rutenett: {args.cell_m} m", flush=True)
    port = args.port
    while port < args.port + 20:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                break
        port += 1
    else:
        parser.error("Ingen ledig lokal port i det valgte området")
    if args.footprint_only:
        from flask import redirect
        from station_footprint import register_footprint_view
        app = Flask(__name__)
        get_backend = register_footprint_view(app, HERE, standalone=True)
        app.add_url_rule("/", "index", lambda: redirect("/footprint"))
        backend = get_backend()
        print(f"Værnett footprint1 km: http://127.0.0.1:{port}/footprint", flush=True)
        try:
            app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
        finally:
            backend.close()
        return
    print("Laster terreng, modeller og lagrede flater ...", flush=True)
    engine = StationEngine(args.surfaces, cell_m=args.cell_m, frost_neighbors=args.frost_neighbors)
    try:
        engine.warm_weather_maps(resolutions=(args.cell_m,))
        cold_basins = KuldehullLayer(OUT / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg")
        cold_basins.prepare()
        print(f"Værnett: http://127.0.0.1:{port}", flush=True)
        initial = PHENOMENON_DATASETS[args.phenomenon]
        if engine.weather is None or not any(item["id"] == initial for item in engine.weather.available):
            initial = f"{args.phenomenon}:continuous"
        app = create_app(engine, initial_dataset=initial, kuldehull_layer=cold_basins)
        app.config["SESSION_COOKIE_NAME"] = f"station_lab_{port}"
        app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
    finally:
        engine.resources.close()


if __name__ == "__main__":
    main()