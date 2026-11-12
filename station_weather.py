"""Five-year weather models and read-only Frost observation availability."""
from __future__ import annotations

import hashlib
import json
import os
from io import StringIO
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from zipfile import BadZipFile

import joblib
import numpy as np
import pandas as pd
import pyogrio

from greedy_400_stations_pruned import FEATURES, TERRAIN, make_cache
from pipeline_config import element_update_marker


WEATHER = {
    "temperature": ("Temperatur", "air_temperature", "\u00b0C", "temperature_class_error", "Klassefeil"),
    "wind": ("Vindhastighet", "wind_speed", "m/s", None, None),
    "precipitation": ("Nedbør", "sum(precipitation_amount P1D)", "mm/døgn", "precipitation_occurrence_error", "Brier-skår"),
    "snow": ("Snødybde", "surface_snow_thickness", "cm", "snow_occurrence_error", "Brier-skår"),
    "humidity": ("Luftfuktighet", "relative_humidity", "prosentpoeng", None, None),
    "road_temperature": ("Vegbanetemperatur", "mean(road_surface_temperature PT1M)", "\u00b0C", None, None),
}

# The one model per phenomenon offered in Station Lab.
PHENOMENON_DATASETS = {
    "temperature": "temperature:senorge30",
    "wind": "wind:continuous",
    "precipitation": "precipitation:class",
    "snow": "snow:class",
    "humidity": "humidity:continuous",
    "road_temperature": "road_temperature:continuous",
}


AVAILABILITY_LOCK = threading.RLock()


def observation_stations(database, report, element, station_ids=None):
    database, report = Path(database).resolve(), Path(report).resolve()
    markers = element_update_marker(database, element).parent
    if markers.is_dir():
        # Downloads touch one marker per element, so writes for other elements keep this cache valid.
        paths = [element_update_marker(database, element), report]
    else:
        paths = [database, Path(str(database) + "-wal"), report]
    signature = tuple((path.stat().st_mtime_ns, path.stat().st_size) if path.exists() else None for path in paths)
    station_filter = None if station_ids is None else tuple(sorted(set(map(str, station_ids))))
    with AVAILABILITY_LOCK:
        stations, start, end = _observation_stations(database, report, element, signature, station_filter)
    return stations.copy(), start, end


@lru_cache(maxsize=8)
def _observation_stations(database, report, element, signature, station_filter=None):
    key = hashlib.sha256(json.dumps(
        # v3: drops entries saved while a stale report was trusted, which shifted the five-year window.
        ["availability-v3", str(database), str(report), element, signature, station_filter]
    ).encode()).hexdigest()[:24]
    folder = database.parent / "outputs" / "station_app_cache"
    path = folder / f"availability_{key}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.read_json(StringIO(payload["stations"]), orient="split", convert_dates=False)
            if not {"station_id", "name", "lat", "lon", "elevation", "inside_norway"} <= set(frame.columns):
                raise ValueError("Ugyldig stasjonsmellomlager")
            return frame, payload["start"], payload["end"]
        except (OSError, ValueError, KeyError):
            pass
    frame, start, end = _read_observation_stations(database, report, element, station_filter)
    staged = None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".json", mode="w", encoding="utf-8", delete=False) as temporary:
            staged = Path(temporary.name)
            json.dump(dict(stations=frame.to_json(orient="split", index=False, force_ascii=False),
                           start=start, end=end, element=element, station_filter=station_filter), temporary)
        staged.replace(path)
    except OSError:
        pass
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return frame, start, end


def _read_observation_stations(database, report, element, station_filter=None):
    wal = Path(str(database) + "-wal")
    modified = max(database.stat().st_mtime, wal.stat().st_mtime if wal.exists() else 0)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        catalogue = pd.read_sql_query("SELECT station_id,name,lat,lon,elevation,inside_norway FROM stations", connection)
        records = None
        # A report older than the database would shift the five-year window and invalidate saved Frost surfaces.
        if Path(report).exists() and Path(report).stat().st_mtime >= modified:
            reported = pd.read_csv(report)
            records = reported.loc[(reported["element"] == element) & (reported["rows"] > reported["null_values"])].copy()
            if station_filter is not None:
                records = records.loc[records["station_id"].astype(str).isin(station_filter)]
        if records is None:
            print(f"Leser datatilgjengelighet for {element} fra SQLite ...", flush=True)
            candidates = catalogue.loc[catalogue["inside_norway"] == 1, "station_id"]
            if station_filter is not None:
                candidates = candidates.loc[candidates.astype(str).isin(station_filter)]
            available = []
            for identifier in candidates:
                latest = connection.execute(
                    "SELECT reference_time FROM observations WHERE station_id=? AND element=? "
                    "AND value IS NOT NULL ORDER BY reference_time DESC LIMIT 1", (identifier, element)).fetchone()
                if latest is not None:
                    earliest = connection.execute(
                        "SELECT reference_time FROM observations WHERE station_id=? AND element=? "
                        "AND value IS NOT NULL ORDER BY reference_time LIMIT 1", (identifier, element)).fetchone()
                    available.append(dict(station_id=identifier, first_time=earliest[0], last_time=latest[0]))
            records = pd.DataFrame(available, columns=["station_id", "first_time", "last_time"])
        if records.empty:
            return catalogue.iloc[:0].copy(), None, None
        last_times = pd.to_datetime(records["last_time"], format="ISO8601", utc=True)
        last = last_times.max().normalize()
        start = (last - pd.DateOffset(years=5)).strftime("%Y-%m-%dT00:00:00.000Z")
        end = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00.000Z")
        records = records.loc[last_times >= pd.Timestamp(start)]
        records = records.merge(catalogue, on="station_id", how="inner")
        records = records.loc[records["inside_norway"] == 1].drop_duplicates("station_id")
        return records.reset_index(drop=True), start, end


def paired_predictions(model, source, target, source_ids, target_ids, classification):
    if hasattr(model, "predict_pairs"):
        values = model.predict_pairs(source, target, source_ids, target_ids)
        return np.minimum(values, 1) if classification else values
    columns = [
        np.hypot(source["x"][source_ids] - target["x"][target_ids], source["y"][source_ids] - target["y"][target_ids]),
        np.abs(source["elevation"][source_ids] - target["elevation"][target_ids]),
        np.abs(source["coast_dist_fjord"][source_ids] - target["coast_dist_fjord"][target_ids]),
        np.abs(source["coast_dist_ocean"][source_ids] - target["coast_dist_ocean"][target_ids]),
        (source["objtype_code"][source_ids] == target["objtype_code"][target_ids]).astype(np.float32),
        np.abs(source["x"][source_ids] - target["x"][target_ids]),
        np.abs(source["y"][source_ids] - target["y"][target_ids]),
    ]
    columns.extend(np.abs(source[key][source_ids] - target[key][target_ids]) for key in TERRAIN)
    values = np.maximum(model.predict(np.column_stack(columns).astype(np.float32)), 0).astype(np.float32)
    return np.minimum(values, 1) if classification else values


def save_exact_cache(path, best, second, winner, completed, extra=None):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f"{path.stem}.pending_", suffix=".npz", delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        np.savez_compressed(staged, baseline=best, second=second, winner=winner,
                            completed=np.asarray(completed, dtype=np.int32), **({} if extra is None else extra))
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    for attempt in range(8):
        try:
            staged.replace(path)
            return
        except PermissionError as error:
            if attempt == 7:
                raise PermissionError(
                    f"Kunne ikke erstatte sjekkpunktet etter 8 forsøk: {path}. "
                    f"Det nye sjekkpunktet er bevart i {staged}") from error
            time.sleep(min(0.25 * 2 ** attempt, 2))


def exact_network(model, source, target, classification, checkpoint=None, progress=None,
                  source_batch=8, pair_batch=100000):
    station_count, cell_count = len(source["x"]), len(target["x"])
    if station_count == 0 or cell_count == 0:
        raise ValueError("Eksakt beregning krever minst én stasjon og én rute")
    best = np.full(cell_count, np.inf, dtype=np.float32)
    second = best.copy()
    winner = np.full(cell_count, -1, dtype=np.int32)
    completed = 0
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        candidates = ([checkpoint] if checkpoint.exists() else []) + list(
            checkpoint.parent.glob(f"{checkpoint.stem}.pending_*.npz"))
        for candidate in candidates:
            try:
                with np.load(candidate, allow_pickle=False) as saved:
                    saved_best = saved["baseline"].copy()
                    saved_second, saved_winner = saved["second"].copy(), saved["winner"].copy()
                    saved_completed = int(saved["completed"])
                if (saved_best.shape != (cell_count,) or saved_second.shape != saved_best.shape or
                        saved_winner.shape != saved_best.shape or not 0 <= saved_completed <= station_count or
                        np.isnan(saved_best).any() or np.isnan(saved_second).any()):
                    raise ValueError("Ugyldig sjekkpunkt for eksakt Frost-beregning")
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                if candidate == checkpoint:
                    raise
                continue
            if saved_completed > completed:
                best, second, winner = saved_best, saved_second, saved_winner
                completed = saved_completed
    if progress is not None:
        progress(completed, station_count)
    for start in range(completed, station_count, source_batch):
        station_ids = np.arange(start, min(start + source_batch, station_count))
        target_chunk = max(1, pair_batch // len(station_ids))
        for offset in range(0, cell_count, target_chunk):
            targets = np.arange(offset, min(offset + target_chunk, cell_count))
            values = paired_predictions(model, source, target, np.repeat(station_ids, len(targets)),
                                        np.tile(targets, len(station_ids)), classification)
            values = values.reshape(len(station_ids), len(targets))
            if (np.isnan(values).any() or np.isneginf(values).any() or
                    (not getattr(model, "allow_nodata", False) and not np.isfinite(values).all())):
                raise ValueError("Modellen ga ugyldige verdier under forhåndsberegning")
            for row, station_id in enumerate(station_ids):
                prediction = values[row]
                better = prediction < best[targets]
                second[targets] = np.where(better, best[targets], np.minimum(second[targets], prediction))
                best[targets] = np.minimum(best[targets], prediction)
                winner[targets] = np.where(better, station_id, winner[targets])
        completed = int(station_ids[-1]) + 1
        if checkpoint is not None:
            save_exact_cache(checkpoint, best, second, winner, completed)
        if progress is not None:
            progress(completed, station_count)
    return best, second, winner


def restricted_profile(profile, target_cache, excluded, build=False):
    """Cache an exact subset network, retaining original station indices for edits."""
    excluded = frozenset(int(index) for index in excluded)
    if not excluded:
        return profile
    kept = np.array([index for index in range(len(profile["stations"])) if index not in excluded],
                    dtype=np.int32)
    if not len(kept):
        return None
    source = Path(profile["cache_path"])
    stat = source.stat()
    signature = dict(schema="restricted-frost-v1", source=str(source.resolve()),
                     bytes=stat.st_size, modified=stat.st_mtime_ns, excluded=sorted(excluded))
    if profile.get("cell_m") == 500:
        from file_hash import sha256
        signature = dict(schema="restricted-frost-500m-v1", source=sha256(source), excluded=sorted(excluded))
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    path = source.with_name(f"frost_svv_{digest}.npz")
    if not path.is_file():
        if not build:
            return None
        checkpoint = path.with_suffix(".partial.npz")
        calculation_started = time.monotonic()

        def progress(completed, total):
            from calculation_progress import report
            report(f"Vegvær-kart {profile['phenomenon']} · {profile['cell_m']} m",
                   completed, total, calculation_started)
            print(f"SVV {profile['id']} {profile['cell_m']} m: {completed}/{total} stasjoner", flush=True)

        best, second, winner = exact_network(profile["model"], make_cache(profile["stations"].iloc[kept]),
            target_cache, profile["kind"] == "class", checkpoint=checkpoint, progress=progress)
        original_winner = np.full(winner.shape, -1, dtype=np.int32)
        valid = winner >= 0
        original_winner[valid] = kept[winner[valid]]
        save_exact_cache(path, best, second, original_winner, len(kept))
        checkpoint.unlink(missing_ok=True)
    with np.load(path, allow_pickle=False) as stored:
        best, second, winner = (stored[key].copy() for key in ("baseline", "second", "winner"))
        completed = int(stored["completed"])
    shape = profile["baseline"].shape
    if (best.shape != shape or second.shape != shape or winner.shape != shape or completed != len(kept)
            or np.isnan(best).any() or np.isnan(second).any() or (best < 0).any()
            or (second < best).any() or not np.isin(winner, np.append(kept, -1)).all()
            or ((winner == -1) != ~np.isfinite(best)).any()):
        raise ValueError(f"Ugyldig lagret SVV-grunnflate: {path}")
    finite = best[np.isfinite(best)]
    if not len(finite):
        raise ValueError(f"SVV-grunnflaten har ingen gyldige verdier: {path}")
    lower, upper = np.percentile(finite, [0.5, 99.5])
    minimum = max(float(lower), 0.0001)
    return dict(profile, baseline=best, second=second, winner=winner, cache_path=path,
                network_excluded=excluded, minimum=minimum,
                frost_maximum=max(float(upper), minimum * 1.01))


class WeatherRegistry:
    def __init__(self, root, grid, target_cache, categories, neighbors=0, models=None,
                 cell_m=1000, allow_precompute=False):
        if neighbors != 0:
            raise ValueError("Frost-nettet beregnes bare eksakt. Bruk --frost-neighbors 0 eller utelat valget")
        self.root, self.grid, self.target_cache = Path(root), grid, target_cache
        self.categories, self.neighbors = categories, neighbors
        self.loaded = {}
        self.svv_loaded = {}
        self.models = {} if models is None else models
        self.allow_precompute = allow_precompute
        self.cell_m = cell_m
        self.lock = threading.RLock()
        self.available = []
        self.portable_manifest = None
        self.station_grid = None
        self.terrain_grids = {}
        manifest_path = self.root / "outputs" / "portable_weather.json"
        if manifest_path.is_file():
            self.portable_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cell_m != self.portable_manifest["cell_m"]:
                raise ValueError("Portable package supports only its bundled grid resolution")
            self.available = self.portable_manifest["available"]
            return
        for phenomenon, (label, element, unit, class_metric, class_label) in WEATHER.items():
            metrics = [("continuous", "prediction_rmse")]
            if class_metric is not None:
                metrics.append(("class", class_metric))
            for kind, metric in metrics:
                tag = f"{phenomenon}_5y_{metric}"
                candidates = [(self.root / "outputs" / tag, None)]
                if phenomenon == "humidity":
                    candidates = [(self.root / "outputs" / f"humidity_5y_{spacing}m" / tag, spacing)
                                  for spacing in dict.fromkeys([cell_m, 5000])]
                    candidates.append((self.root / "outputs" / tag, None))
                elif phenomenon in ("wind", "road_temperature"):
                    candidates = [(self.root / "outputs" / f"{phenomenon}_5y_5000m" / tag, 5000),
                                  (self.root / "outputs" / tag, None)]
                from grid_resolution import source_spacing, weather_folder
                station_spacing = source_spacing(cell_m)
                cell_folder = weather_folder(self.root, station_spacing) / phenomenon
                if station_spacing == 500:
                    candidates = []
                if (cell_folder / "ready.json").is_file() or os.environ.get("WEATHER_MODEL_STAGING") == phenomenon:
                    candidates.insert(0, (cell_folder / tag, station_spacing))
                selected = next(((prefix, spacing) for prefix, spacing in candidates
                                 if all(prefix.with_name(prefix.name + suffix).exists() for suffix in
                                        ("_model.joblib", "_metadata.json", "_station_features.gpkg"))), None)
                if selected is not None:
                    prefix, model_grid_m = selected
                    self.available.append(dict(id=f"{phenomenon}:{kind}", phenomenon=phenomenon,
                                               kind=kind, label=label, metric="RMSE" if kind == "continuous" else class_label,
                                               unit=unit if kind == "continuous" else "skår" if phenomenon != "temperature" else "andel",
                                               prefix=str(prefix), element=element, model_grid_m=model_grid_m))
        senorge_prefix = self.root / "outputs" / "temperature_5y_prediction_rmse_senorge"
        if all(senorge_prefix.with_name(senorge_prefix.name + suffix).is_file() for suffix in
               ("_model.joblib", "_metadata.json", "_station_features.gpkg")):
            self.available.append(dict(
                id="temperature:senorge", phenomenon="temperature", kind="senorge",
                label="Temperatur (seNorge-pilot)", metric="RMSE", unit="°C",
                prefix=str(senorge_prefix), element=WEATHER["temperature"][1],
                model_grid_m=1000,
            ))
        portable = self.root / "outputs" / "senorge_30yr_station_model"
        station_features = self.root / "outputs" / "temperature_5y_prediction_rmse_station_features.gpkg"
        if (all((portable / name).is_file() for name in
                ("xgboost_with_basins.ubj", "metadata.json", "grid_features.csv.gz")) and station_features.is_file()):
            self.available.append(dict(
                id="temperature:senorge30", phenomenon="temperature", kind="senorge30",
                label="Temperatur", metric="RMSE", unit="°C",
                prefix=str(portable), element=WEATHER["temperature"][1], model_grid_m=1000,
                station_features=str(station_features),
            ))
        seasonal = self.root / "outputs" / "senorge_seasonal_station_model"
        if station_features.is_file() and all((seasonal / name).is_file() for name in
                ("metadata.json", "combined.pt", "combined_normalization.npz", "seasonal_manifest.json")):
            from station_seasonal import TARGET_LABELS
            for target, label in TARGET_LABELS.items():
                self.available.append(dict(
                    id=f"temperature:seasonal_{target}", phenomenon="temperature", kind=f"seasonal_{target}",
                    label=f"Temperatur ({label} · terreng-CNN)", metric="RMSE", unit="°C",
                    prefix=str(seasonal), element=WEATHER["temperature"][1], model_grid_m=1000,
                    station_features=str(station_features), seasonal_target=target))

        gpu = self.root / "outputs" / "senorge_transfer_gpu_returned"
        if all((gpu / name).is_file() for name in
               ("verification.json", "portable_ridge.json", "portable_ridge.npz",
                "senorge_transfer_gpu_results/report.json")):
            for variable, target, label in ((0, "tg", "døgnmiddel"), (1, "tn", "døgnminimum")):
                self.available.append(dict(
                    id=f"temperature:senorge_gpu_{target}", phenomenon="temperature", kind=f"senorge_gpu_{target}",
                    label=f"Temperatur (seNorge GPU · langtids-RMSE · {label} · 1996–2025)",
                    metric="RMSE", unit="°C", prefix=str(gpu), model_grid_m=1000,
                    long_term=True, variable=variable))

    def svv_profile(self, identifier, build=False):
        """Unedited national SVV baseline, independent of the interactive scenario."""
        if self.portable_manifest is not None:
            profile = self.profile(identifier)
            if identifier not in self.svv_loaded:
                entry = self.portable_manifest["profiles"][identifier]
                reference = joblib.load(self.root / entry["svv"])
                reference["model"] = profile["model"]
                reference["cache_path"] = self.root / entry["svv"]
                self.svv_loaded[identifier] = reference
            return self.svv_loaded[identifier]
        from station_pross import pross_excluded

        with self.lock:
            profile = self.profile(identifier)
            if profile.get("baseline") is None:
                raise ValueError("SVV-grunnflate mangler. Kjør python precompute_frost.py --station-network both")
            stations = profile["stations"]
            excluded = pross_excluded(stations)
            if len(excluded) == len(stations):
                if profile["phenomenon"] == "precipitation":
                    return dict(profile, colour_reference_warning=
                        "Ingen nedbørstasjoner har vegnummer først i navnet. "
                        "Vegvær-nettet for nedbør er tomt; fargeskalaen bruker alle MET-stasjoner.")
                raise ValueError(f"Ingen SVV-stasjoner (MET-navn med vegnummer først) for {identifier}")
            key = (identifier, id(profile["baseline"]), excluded)
            if key not in self.svv_loaded:
                reference = restricted_profile(profile, self.target_cache, excluded, build=build)
                if reference is None:
                    raise ValueError("SVV-grunnflate mangler. Kjør python precompute_frost.py --station-network both")
                self.svv_loaded[key] = reference
            return self.svv_loaded[key]

    def profile(self, identifier):
        if identifier == "temperature:senorge30_pross":
            identifier = "temperature:senorge30"
        with self.lock:
            cached = self.loaded.get(identifier)
            pending_long_term = cached is not None and cached.get("long_term") and cached.get("baseline") is None
            if pending_long_term:
                from station_senorge_gpu import refresh_progress
                refresh_progress(cached)
            if cached is not None and (pending_long_term or cached.get("status", {}).get("state") == "missing") and cached["cache_path"].exists():
                del self.loaded[identifier]
            if identifier not in self.loaded:
                config = next((item for item in self.available if item["id"] == identifier), None)
                if config is None:
                    raise ValueError("Ingen lagret femårsmodell finnes for dette værvalget")
                self.loaded[identifier] = self._load(config)
            return self.loaded[identifier]

    def _load(self, config):
        if self.portable_manifest is not None:
            entry = self.portable_manifest["profiles"][config["id"]]
            profile = joblib.load(self.root / entry["profile"])
            if entry.get("model_kind") == "senorge":
                from station_senorge import BasinLookup, SeNorgePairModel
                profile["model"] = SeNorgePairModel(self.root / entry["model"], BasinLookup(
                    self.root / entry["basin_features"], self.root / entry["basins"]))
            else:
                profile["model"] = joblib.load(self.root / entry["model"])
            profile["cache_path"] = self.root / entry["profile"]
            return profile
        if config["kind"].startswith("senorge_gpu_"):
            from station_senorge_gpu import load_profile
            return load_profile(self, config)
        prefix = Path(config["prefix"])
        seasonal = config["kind"].startswith("seasonal_")
        portable = config["kind"] in ("senorge30", "senorge30_pross") or seasonal
        model_path = prefix / ("combined.pt" if seasonal else "xgboost_with_basins.ubj") if portable else prefix.with_name(prefix.name + "_model.joblib")
        feature_path = Path(config["station_features"]) if portable else prefix.with_name(prefix.name + "_station_features.gpkg")
        metadata_path = prefix / "metadata.json" if portable else prefix.with_name(prefix.name + "_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_features = FEATURES
        if portable:
            from station_senorge import PAIR_FEATURES, BasinLookup, SeNorgePairModel
            from file_hash import sha256
            expected_features = PAIR_FEATURES
            if sha256(model_path) != metadata["model_sha256"]:
                raise ValueError("seNorge-modellens kontrollsum stemmer ikke")
            basin_prefix = self.root / "outputs" / "senorge_30yr_station_model" if seasonal else prefix
            basin_metadata = json.loads((basin_prefix / "metadata.json").read_text(encoding="utf-8"))
            if sha256(basin_prefix / "grid_features.csv.gz") != basin_metadata["basin_grid_sha256"]:
                raise ValueError("seNorge-bassengtabellens kontrollsum stemmer ikke")
        if metadata["features"] != expected_features:
            raise ValueError("Værmodellens variabler samsvarer ikke med beregningsmotoren")
        features = pyogrio.read_dataframe(feature_path).drop_duplicates("station_id")
        station_grid_signature = None
        if portable:
            from make_uncertainty_map_gif import load_grid_all_fields
            from station_senorge import station_cell_features
            from grid_resolution import source_spacing
            station_spacing = source_spacing(self.cell_m)
            station_grid_path = self.root / "outputs" / f"norway_grid_{station_spacing}m.gpkg"
            if self.station_grid is None:
                self.station_grid = load_grid_all_fields(station_grid_path)
            features = station_cell_features(features, self.station_grid)
            station_grid_signature = (station_grid_path.stat().st_size, station_grid_path.stat().st_mtime_ns)
        fields = ["station_id", "x", "y", "elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype", *TERRAIN]
        from grid_resolution import source_spacing, cell_policy
        source_grid_m = (source_spacing(self.cell_m) if portable else
                         500 if metadata.get("station_input_policy") == cell_policy(500) else
                         1000 if metadata.get("station_input_policy") == cell_policy(1000) else
                         config.get("model_grid_m") or 5000)
        from scipy.spatial import cKDTree
        from make_uncertainty_map_gif import load_grid_all_fields
        if source_grid_m not in self.terrain_grids:
            self.terrain_grids[source_grid_m] = (self.station_grid if portable else self.grid
                if self.cell_m == source_grid_m else load_grid_all_fields(
                    self.root / "outputs" / f"norway_grid_{source_grid_m}m.gpkg"))
        terrain_grid = self.terrain_grids[source_grid_m]
        _, terrain_ids = cKDTree(terrain_grid[["x", "y"]].to_numpy()).query(features[["x", "y"]].to_numpy())
        features["calculation_x"] = terrain_grid.iloc[terrain_ids]["x"].to_numpy()
        features["calculation_y"] = terrain_grid.iloc[terrain_ids]["y"].to_numpy()
        fields.extend(["calculation_x", "calculation_y"])
        grid_hash = hashlib.sha256(pd.util.hash_pandas_object(
            self.grid[["x", "y", "elevation", "objtype", "coast_dist_fjord", "coast_dist_ocean", *TERRAIN]],
            index=False).to_numpy().tobytes()).hexdigest()
        folder = self.root / "outputs" / "station_app_cache"
        folder.mkdir(exist_ok=True)

        def source_frame(observations):
            return observations[["station_id", "name", "lat", "lon", "first_time", "last_time"]].merge(
                features[fields], on="station_id", how="inner").dropna(subset=fields[1:]).reset_index(drop=True)

        def cache_path(sources, start, end):
            signature = dict(schema="exact-best-two-v1", mode=config["id"],
                             stations=sources["station_id"].tolist(),
                             model=(model_path.stat().st_size, model_path.stat().st_mtime_ns),
                             features=(feature_path.stat().st_size, feature_path.stat().st_mtime_ns),
                             grid_size=len(self.grid), grid_hash=grid_hash, start=start, end=end)
            if portable:
                basin_path = self.root / "outputs" / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg"
                signature["portable"] = dict(schema="station-cell-1km-v3",
                                             station_grid=station_grid_signature,
                                             metadata=metadata,
                                             basins=(basin_path.stat().st_size, basin_path.stat().st_mtime_ns))
            if self.cell_m == 500:
                from file_hash import sha256
                signature["model"] = sha256(model_path)
                signature["features"] = sha256(feature_path)
                if portable:
                    signature["portable"].update(schema="station-cell-500m-v1",
                        station_grid=grid_hash, basins=sha256(basin_path))
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
            return folder / f"frost_exact_{key}.npz"

        prepared = None
        if not self.allow_precompute:
            # Only reuse a snapshot whose full model/grid/source signature has a completed surface.
            for snapshot in sorted(folder.glob("availability_*.json"),
                                   key=lambda item: item.stat().st_mtime_ns, reverse=True):
                payload = json.loads(snapshot.read_text(encoding="utf-8"))
                if payload.get("element", config["element"]) != config["element"]:
                    continue
                observations = pd.read_json(StringIO(payload["stations"]), orient="split", convert_dates=False)
                sources = source_frame(observations)
                start, end = payload["start"], payload["end"]
                path = cache_path(sources, start, end)
                if path.is_file():
                    prepared = observations, sources, start, end, path
                    break
        if prepared is None:
            observations, start, end = observation_stations(
                self.root / "frost_history.db", self.root / "frost_db_report" / "station_element_all.csv",
                config["element"], station_ids=features["station_id"],
            )
            sources = source_frame(observations)
            path = cache_path(sources, start, end)
        else:
            observations, sources, start, end, path = prepared
        sources["objtype_code"] = pd.Categorical(sources["objtype"], categories=self.categories).codes
        if (sources["objtype_code"] < 0).any():
            raise ValueError("En observasjonsstasjon har en terrengkategori som mangler i rutenettet")
        if sources.empty:
            raise ValueError("Ingen Frost-stasjoner har brukbare observasjoner og terrengverdier for dette værvalget")
        model_key = str(model_path) + (":" + config["seasonal_target"] if seasonal else "")
        if model_key not in self.models:
            if seasonal:
                from station_seasonal import SeasonalBackend, SeasonalPairModel
                backend_key = str(prefix / "shared-backend")
                if backend_key not in self.models:
                    self.models[backend_key] = SeasonalBackend(self.root, prefix)
                self.models[model_key] = SeasonalPairModel(
                    self.models[backend_key], BasinLookup(
                        basin_prefix / "grid_features.csv.gz",
                        self.root / "outputs" / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg"),
                    config["seasonal_target"])
            else:
                self.models[model_key] = (
                    SeNorgePairModel(model_path, BasinLookup(
                        prefix / "grid_features.csv.gz",
                        self.root / "outputs" / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg"))
                    if portable else joblib.load(model_path))
        model = self.models[model_key]
        if os.environ.get("WEATHER_CPU_THREADS") and hasattr(model, "set_params"):
            model.set_params(n_jobs=int(os.environ["WEATHER_CPU_THREADS"]))
        if int(model.n_features_in_) != len(expected_features):
            raise ValueError("Den lagrede modellen støtter ikke de forventede terrengvariablene")
        maximum = 1.0 if config["kind"] == "class" else {
            "temperature": 6.5, "wind": 20, "precipitation": 25, "snow": 50, "humidity": 25,
            "road_temperature": 8,
        }[config["phenomenon"]]
        profile = dict(**config, model=model, baseline=None, second=None, winner=None, stations=sources,
                       source_grid_m=source_grid_m,
                       cache_path=path, cell_m=self.cell_m,
                       missing_features=len(observations) - len(sources), minimum=0.0001,
                       maximum=maximum, start=start, end=end, method="Eksakt minimum over alle Frost-stasjoner",
                       classes=metadata.get("temperature_class_limits") or metadata.get("precip_wet_threshold_mm") or metadata.get("snow_present_threshold_cm"))
        if portable:
            if seasonal:
                from station_seasonal import WARNING
            else:
                from station_senorge import WARNING
            profile.update(model_warning=WARNING, training_period="1996–2025", source_grid_m=source_grid_m,
                           method="Eksakt minimum over Frost-stasjoner · seNorge-likhet, ikke målestasjonsnøyaktighet")
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                baseline = stored["baseline"].copy()
                second, winner = stored["second"].copy(), stored["winner"].copy()
                completed = int(stored["completed"])
            if (baseline.shape != (len(self.grid),) or second.shape != baseline.shape or winner.shape != baseline.shape or
                    completed != len(sources) or not np.isfinite(baseline).all() or
                    (len(sources) > 1 and not np.isfinite(second).all()) or
                    (second < baseline).any() or (winner < 0).any() or (winner >= len(sources)).any()):
                raise ValueError("Ugyldig mellomlagret Frost-flate. Fjern den aktuelle mellomlagerfilen og start på nytt")
            lower, upper = np.percentile(baseline, [0.5, 99.5])
            minimum = max(float(lower), 0.0001)
            profile.update(baseline=baseline, second=second, winner=winner,
                           minimum=minimum, frost_maximum=max(float(upper), minimum * 1.01),
                           status=dict(state="ready", completed=len(sources), total=len(sources), percent=100,
                                       label=f"{config['label']} · {config['metric']}", cell_m=self.cell_m, error=None))
        else:
            if not self.allow_precompute:
                profile["status"] = dict(state="missing", completed=0, total=len(sources), percent=0,
                                         label=f"{config['label']} · {config['metric']}", cell_m=self.cell_m,
                                         error=f"Kjør separat: python precompute_frost.py --phenomenon {config['phenomenon']} "
                                               f"--cell-m {self.cell_m} --kind {config['kind']}")
                return profile
            checkpoint = path.with_suffix(".partial.npz")

            def progress(completed, total):
                from calculation_progress import report
                report(f"Kart {config['phenomenon']} · {self.cell_m} m", completed, total, calculation_started)
                profile["status"].update(completed=completed, total=total, percent=round(100 * completed / total, 1))
                print(f"Eksakt Frost {config['id']} {self.cell_m} m: {completed}/{total} stasjoner", flush=True)

            calculation_started = time.monotonic()
            def prepare():
                best, runner_up, best_station = exact_network(
                    model, make_cache(sources), self.target_cache, config["kind"] == "class", checkpoint, progress)
                save_exact_cache(path, best, runner_up, best_station, len(sources))
                with self.lock:
                    lower, upper = np.percentile(best, [0.5, 99.5])
                    minimum = max(float(lower), 0.0001)
                    profile.update(baseline=best, second=runner_up, winner=best_station,
                                   minimum=minimum, frost_maximum=max(float(upper), minimum * 1.01))
                checkpoint.unlink(missing_ok=True)

            profile["status"] = dict(state="running", completed=0, total=len(sources), percent=0,
                                     label=f"{config['label']} · {config['metric']}", cell_m=self.cell_m, error=None)
            prepare()
            profile["status"].update(state="ready", completed=len(sources), percent=100)
        return profile