"""Experimental single-source national temperature transfer at 1 km land cells."""
from collections import OrderedDict
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
from pyproj import Transformer
import requests

from national_temperature_features import temperature_features
from national_temperature_uncertainty import TemperatureUncertainty
from prepare_national_dataset import (
    POLICY, atomic_npz, identity, load_config, load_splits, rectangles,
    regime_metadata, selected_temperatures, validate_regime,
)
from prepare_valley_pilot import VARIABLES, http, validate_day
from remote_senorge.train_valley_neural import create_model, sha256, write_json
from valley_pilot import HOURS, REGIME_FIELDS, minimax_heights, parse_ascii


WARNING = (
    "Experimental temperature transfer, not observed weather or a future forecast. "
    "1 km land cells use a 2.5 km archived MEPS background and the national single-source model. "
    "Unobserved cells and arbitrary slider temperatures are not validated. "
    "Experimental station-derived interval radii do not guarantee coverage at grid cells, "
    "synthetic temperatures or sources outside the training network. No domino propagation. "
    "Sea, unsupported terrain and land disconnected from the source are transparent."
)


def map_geometry(table, positions, height, valid, origin, source):
    """Use the same 115 km local minimax graph and straight-route features as training."""
    r, c = positions[source]
    radius = int((POLICY["source_radius_m"]+POLICY["graph_halo_m"])/1000)
    r0, c0 = max(0, r-radius), max(0, c-radius)
    r1, c1 = min(height.shape[0], r+radius+1), min(height.shape[1], c+radius+1)
    peaks = minimax_heights(height[r0:r1, c0:c1], valid[r0:r1, c0:c1], (r-r0, c-c0))
    atlas_positions = np.rint((table[:, :2]-origin)*[1, -1]/1000).astype(int)[:, ::-1]
    selected = np.flatnonzero(
        (atlas_positions[:, 0] >= r-32) & (atlas_positions[:, 0] < r+32) &
        (atlas_positions[:, 1] >= c-32) & (atlas_positions[:, 1] < c+32))
    destinations = atlas_positions[selected]
    supported = ((destinations >= 0) & (destinations < height.shape)).all(axis=1)
    selected, destinations = selected[supported], destinations[supported]
    keep = valid[destinations[:, 0], destinations[:, 1]]
    keep &= np.isfinite(peaks[destinations[:, 0]-r0, destinations[:, 1]-c0])
    selected, destinations = selected[keep], destinations[keep]
    if not len(selected):
        raise ValueError("No connected supported land in this 64 km window")
    source_z = height[r, c]
    graph = []
    for tr, tc in destinations:
        peak, target_z = peaks[tr-r0, tc-c0], height[tr, tc]
        steps = int(max(abs(tr-r), abs(tc-c)))+1
        line = np.rint(np.linspace([r, c], [tr, tc], steps)).astype(int)
        straight = bool(valid[line[:, 0], line[:, 1]].all())
        ridge = max(0., float(height[line[:, 0], line[:, 1]].max())-max(source_z, target_z)) if straight else 0.
        graph.append([1., (peak-source_z)/1000, (peak-target_z)/1000, ridge/1000, float(straight)])
    local_positions = destinations-[r-32, c-32]
    top_left = origin+np.array([(c-32)*1000-500, -(r-32)*1000+500])
    return selected, local_positions, np.asarray(graph, np.float32), top_left


def download_weather(day, points, progress):
    with requests.Session() as session:
        progress("Checking archived MEPS coordinates, units and issue time")
        metadata = regime_metadata(session, day, points)
        ix, iy = np.asarray(metadata["ix"]), np.asarray(metadata["iy"])
        fields = {name: np.full((3, len(points)), np.nan) for name in REGIME_FIELDS}
        chunks = rectangles(ix, iy)
        for chunk, (indices, c0, c1, r0, r1) in enumerate(chunks):
            progress(f"Downloading bounded MEPS subset {chunk+1}/{len(chunks)}")
            selection = f"[12:3:18][0:1:0][{r0}:1:{r1}][{c0}:1:{c1}]"
            text = http(session, metadata["url"]+".ascii?"+
                        ",".join(name+selection for name in VARIABLES))
            for variable, (name, _) in VARIABLES.items():
                array = parse_ascii(text, variable+"."+variable)
                if array.shape != (3, 1, r1-r0+1, c1-c0+1):
                    raise ValueError(f"Unexpected map weather shape: {variable}")
                fields[name][:, indices] = array[:, 0, iy[indices]-r0, ix[indices]-c0]
        fields["temperature_c"] -= 273.15
        metadata["fields"] = {name: values.tolist() for name, values in fields.items()}
        validate_regime(metadata, day, points)
        return metadata


class TemperatureMap:
    def __init__(self, root):
        import torch
        self.torch = torch
        self.root = Path(root)
        self.folder = self.root / "outputs" / "national_station_dataset"
        self.config = load_config(self.folder)
        load_splits(self.folder)
        self.stations = pd.read_csv(self.folder / "station_splits.csv")
        self.catalogue = self.stations.copy().reset_index(drop=True)
        self.manifest = json.loads((self.folder / "dataset_manifest.json").read_text())
        self.receipts = json.loads((self.folder / "regimes_progress.json").read_text())
        self.observation_receipts = json.loads((self.folder / "observations_progress.json").read_text())
        observation_config = json.loads((self.folder / "observations_config.json").read_text())
        if observation_config["station_ids"] != self.stations["station_id"].tolist():
            raise ValueError("Observation and terrain catalogue order differ")
        excluded = {row["date"] for row in self.manifest["excluded_forecast_nights"]}
        self.days = sorted(set(self.manifest["shards"])-excluded)
        with np.load(self.folder / "terrain.npz", allow_pickle=False) as terrain:
            self.station_table = terrain["table"].copy()
            self.positions = terrain["positions"].copy()
            self.height, self.valid = terrain["height"].copy(), terrain["valid"].copy()
            self.origin = terrain["origin"].copy()
        atlas = self.root / "outputs" / "senorge_footprint_expanded"
        if sha256(atlas / "footprint_manifest.json") != self.config["atlas_manifest_sha256"]:
            raise ValueError("Terrain atlas provenance changed")
        self.table = np.load(atlas / "grid_table.npy", mmap_mode="r", allow_pickle=False)
        result = self.root / "outputs" / "national_transfer_returned" / "national_transfer_results"
        summary = json.loads((result / "summary.json").read_text())
        run_config = json.loads((result / "run_config.json").read_text())
        if sha256(result.parent / "manifest.json") != run_config["manifest_sha256"]:
            raise ValueError("National temperature model manifest checksum mismatch")
        for name in ("selected_pair.pt", "normalization.npz"):
            if sha256(result / name) != summary["checksums"][name]:
                raise ValueError(f"National temperature model checksum mismatch: {name}")
        self.selected = summary["pair_selected"]["name"]
        model_manifest = json.loads((result.parent / "manifest.json").read_text())
        if model_manifest["features"] != self.manifest["features"]:
            raise ValueError("National temperature feature ordering changed")
        self.model = create_model(torch, summary["pair_selected"]["config"], model_manifest)
        self.model.load_state_dict(torch.load(result / "selected_pair.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        with np.load(result / "normalization.npz", allow_pickle=False) as norm:
            self.mean, self.std = norm["mean"].copy(), norm["std"].copy()
        uncertainty_folder = self.root / "outputs" / "national_uncertainty_returned"
        self.uncertainty = None
        if uncertainty_folder.exists():
            self.uncertainty = TemperatureUncertainty(self.root)
            if (self.uncertainty.run["selected_weights_sha256"] != sha256(result / "selected_pair.pt") or
                    self.uncertainty.run["normalization_sha256"] != sha256(result / "normalization.npz")):
                raise ValueError("Uncertainty was trained for a different temperature predictor")
        self.cache = self.root / "outputs" / "national_temperature_map"
        self.cache.mkdir(exist_ok=True)
        self.contexts = OrderedDict()
        self.to_lonlat = Transformer.from_crs(25833, 4326, always_xy=True)
        self.fingerprint = identity(dict(schema="temperature-map-v1", code=sha256(Path(__file__)),
                                        features=sha256(Path(__file__).with_name("national_temperature_features.py")),
                                        station_catalogue=self.config["checksums"]["stations.csv"],
                                        dataset_helpers=sha256(Path(__file__).with_name("prepare_national_dataset.py")),
                                        weather_helpers=sha256(Path(__file__).with_name("prepare_valley_pilot.py")),
                                        graph_helpers=sha256(Path(__file__).with_name("valley_pilot.py")),
                                        terrain=self.config["checksums"]["terrain.npz"],
                                        atlas=sha256(atlas / "grid_table.npy")))

    def options(self):
        return dict(stations=self.catalogue[["station_id", "name", "lat", "lon", "group"]].to_dict("records"),
                    days=self.days, hours=list(HOURS), selected=self.selected, warning=WARNING,
                    uncertainty=dict(available=self.uncertainty is not None,
                                     test=self.uncertainty.report["test"] if self.uncertainty else None),
                    resolution_m=1000, width_km=64)

    def validate(self, source, day):
        if source not in set(self.catalogue["station_id"]) or day not in self.days:
            raise ValueError("Choose a terrain-supported station and an available archived winter date")
        return pd.Timestamp(day).date()

    @lru_cache(maxsize=128)
    def observation(self, source, timestamp):
        day = timestamp[:10]
        index = self.stations.index[self.stations["station_id"].eq(source)][0]
        batch = int(index)//64
        key = f"{day}_b{batch:03d}"
        path = self.folder / "observations" / f"{key}.json"
        if sha256(path) != self.observation_receipts[key]:
            raise ValueError("National QC observation cache checksum mismatch")
        payload = json.loads(path.read_text())
        validate_day(payload, pd.Timestamp(day).date(), self.stations.iloc[batch*64:(batch+1)*64], "observations")
        temperatures = selected_temperatures([payload], self.stations)
        return temperatures.get((source, timestamp))

    def prepare(self, source, day, progress):
        date = self.validate(source, day)
        key = (source, day)
        if key in self.contexts:
            return
        cache_key = identity(dict(source=source, day=day, fingerprint=self.fingerprint,
                                  station_weather=self.receipts[day]))
        path, receipt = self.cache / f"{cache_key}.npz", self.cache / f"{cache_key}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if sha256(path) != saved["sha256"]:
                raise ValueError("Cached map weather/terrain checksum mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                context = {name: arrays[name].copy() for name in arrays.files}
        else:
            progress("Computing terrain-connected 1 km land cells")
            source_index = self.stations.index[self.stations["station_id"].eq(source)][0]
            indices, positions, graph, top_left = map_geometry(
                self.table, self.positions, self.height, self.valid, self.origin, source_index)
            table = np.vstack([self.station_table[source_index], self.table[indices]])
            lon, lat = self.to_lonlat.transform(table[:, 0], table[:, 1])
            # Station weather is sampled at its true thermometer coordinates, as during fitting.
            station = self.stations.loc[source_index]
            lon[0], lat[0] = station.lon, station.lat
            points = pd.DataFrame(dict(station_id=[source]+[f"cell{i}" for i in indices], lon=lon, lat=lat))
            forecast = download_weather(date, points, progress)
            fields = np.array([forecast["fields"][name] for name in REGIME_FIELDS]).transpose(1, 2, 0)
            station_path = self.folder / "regimes" / f"{day}.json"
            if sha256(station_path) != self.receipts[day]:
                raise ValueError("National station weather checksum mismatch")
            station_weather = json.loads(station_path.read_text())
            if source in station_weather["station_ids"]:
                index = station_weather["station_ids"].index(source)
                for field, name in enumerate(REGIME_FIELDS):
                    if not np.allclose(fields[:, 0, field], np.array(station_weather["fields"][name])[:, index], atol=1e-5):
                        raise ValueError("Map source weather differs from training archive")
            context = dict(table=table, positions=positions, graph=graph, top_left=top_left,
                           fields=fields, projection_xy=np.asarray(forecast["projection_xy"]),
                           url=np.array(forecast["url"]), run_time=np.array(forecast["run_time"]))
            atomic_npz(path, **context)
            write_json(receipt, dict(source=source, day=day, fingerprint=self.fingerprint, sha256=sha256(path)))
        self.contexts[key] = context
        while len(self.contexts) > 3:
            self.contexts.popitem(last=False)

    def raster(self, source, day, hour, temperature, include_uncertainty=False):
        self.validate(source, day)
        if hour not in HOURS:
            raise ValueError("Choose UTC hour 00, 03 or 06")
        context = self.contexts.get((source, day))
        if context is None:
            raise ValueError("Prepare this station/date map first")
        stamp = pd.Timestamp(day, tz="UTC")+pd.Timedelta(hours=hour)
        observed = self.observation(source, stamp.isoformat())
        synthetic = temperature is not None
        if temperature is None:
            if observed is None:
                raise ValueError("No QC source observation for this time; choose another time or explicitly use the synthetic slider")
            temperature = observed
        elif not np.isfinite(temperature) or not -20 <= temperature <= 20:
            raise ValueError("Synthetic source temperature must be between -20 and +20 C")
        count = len(context["graph"])
        x, delta = temperature_features(context["table"], 0, np.arange(1, count+1),
                                        context["graph"], temperature, stamp,
                                        context["fields"][HOURS.index(hour)], context["projection_xy"])
        with self.torch.inference_mode():
            correction = self.model(self.torch.from_numpy(((x-self.mean)/self.std).astype(np.float32))).numpy()
        prediction = temperature+delta+correction
        if not np.isfinite(prediction).all():
            raise ValueError("National model produced nonfinite map temperatures")
        raster = np.full((64, 64), np.nan, np.float32)
        positions = context["positions"]
        raster[positions[:, 0], positions[:, 1]] = prediction
        metadata = dict(source=source, time=stamp.isoformat(), source_temperature_c=float(temperature),
                        observed_source_c=observed, model=self.selected,
                        synthetic_source=synthetic,
                        weather_url=str(context["url"]), run_time=str(context["run_time"]),
                        cells=count, minimum_c=float(prediction.min()), maximum_c=float(prediction.max()),
                        warning=WARNING, uncertainty_validated=False)
        groups = self.catalogue.set_index("station_id")["group"]
        metadata["source_group"] = str(groups[source])
        metadata["source_extrapolation"] = bool(groups[source] != "train")
        if include_uncertainty:
            if self.uncertainty is None:
                raise ValueError("Verified returned uncertainty model is unavailable")
            radius = self.uncertainty.radii(x, prediction)
            if radius.shape != prediction.shape or not np.isfinite(radius).all() or (radius <= 0).any():
                raise ValueError("Error model produced invalid interval radii")
            radius_raster = np.full_like(raster, np.nan)
            radius_raster[positions[:, 0], positions[:, 1]] = radius
            raster = np.stack([raster, radius_raster, raster-radius_raster, raster+radius_raster])
            metadata.update(
                interval_radius_median_c=float(np.median(radius)), interval_radius_p90_c=float(np.quantile(radius, .9)),
                uncertainty_model_sha256=self.uncertainty.model_sha256,
                uncertainty_factor=self.uncertainty.factor,
                uncertainty_test_night_coverage=self.uncertainty.report["test"]["night_coverage"],
                interval_label="Experimental station-derived interval; not calibrated grid coverage")
        return raster, context["top_left"], metadata

    def geojson(self, raster, top_left, metadata):
        bands = raster[None] if raster.ndim == 2 else raster
        rows, cols = np.nonzero(np.isfinite(bands[0]))
        xy = top_left+np.column_stack([cols*1000, -rows*1000])
        rings = xy[:, None, :]+np.array([[0, 0], [1000, 0], [1000, -1000], [0, -1000], [0, 0]])
        lon, lat = self.to_lonlat.transform(rings[:, :, 0], rings[:, :, 1])
        features = [
            dict(type="Feature", properties=dict(
                temperature_c=float(bands[0, r, c]), row=int(r), col=int(c),
                **(dict(interval_radius_c=float(bands[1, r, c]), lower_c=float(bands[2, r, c]),
                        upper_c=float(bands[3, r, c])) if len(bands) == 4 else {})),
                 geometry=dict(type="Polygon", coordinates=[np.column_stack([lon[i], lat[i]]).tolist()]))
            for i, (r, c) in enumerate(zip(rows, cols))]
        return dict(type="FeatureCollection", features=features, metadata=metadata)

    def geotiff(self, raster, top_left, metadata):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        bands = raster[None] if raster.ndim == 2 else raster
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=64, height=64, count=len(bands), dtype="float32",
                             crs="EPSG:25833", transform=from_origin(*top_left, 1000, 1000),
                             nodata=-9999., compress="deflate") as dataset:
                dataset.write(np.where(np.isfinite(bands), bands, -9999.).astype(np.float32))
                labels = ("Experimental transferred temperature (degC)", "Experimental interval radius (degC)",
                          "Experimental lower interval bound (degC)", "Experimental upper interval bound (degC)")
                for band in range(len(bands)):
                    dataset.set_band_description(band+1, labels[band])
                dataset.update_tags(**{key: str(value) for key, value in metadata.items()})
            return memory.read()


def register_temperature_map(app, root, backend=None):
    from flask import Blueprint, jsonify, request, send_file
    view = Blueprint("temperature_map", __name__)
    root = Path(root)
    lock = threading.RLock()
    status = dict(state="idle")
    worker = None

    def get_backend():
        nonlocal backend
        if backend is None:
            backend = TemperatureMap(root)
        return backend

    @view.get("/temperature-map")
    def index():
        return send_file(root / "station_temperature_map.html")

    @view.get("/station_temperature_map.js")
    def javascript():
        return send_file(root / "station_temperature_map.js", mimetype="text/javascript")

    @view.get("/api/temperature-map/options")
    def options():
        with lock:
            return jsonify(get_backend().options())

    @view.get("/api/temperature-map/status")
    def state():
        with lock:
            return jsonify(status)

    @view.get("/api/temperature-map/station")
    def station():
        source, day, hour = request.args["source"], request.args["day"], int(request.args["hour"])
        with lock:
            model = get_backend()
            model.validate(source, day)
            if hour not in HOURS:
                raise ValueError("Choose UTC hour 00, 03 or 06")
            stamp = pd.Timestamp(day, tz="UTC")+pd.Timedelta(hours=hour)
            row = model.catalogue.loc[model.catalogue.station_id.eq(source)].iloc[0]
            return jsonify(source=source, time=stamp.isoformat(), group=row["group"],
                           observed_source_c=model.observation(source, stamp.isoformat()),
                           source_extrapolation=bool(row["group"] != "train"))

    @view.post("/api/temperature-map/prepare")
    def prepare():
        nonlocal worker, status
        source, day = request.args["source"], request.args["day"]
        with lock:
            model = get_backend()
            model.validate(source, day)
            if (source, day) in model.contexts:
                return jsonify(state="ready", source=source, day=day)
            if worker is not None and worker.is_alive():
                if status.get("source") != source or status.get("day") != day:
                    return jsonify(error="Another station/date is being prepared; wait for it to finish"), 409
                return jsonify(status), 202
            status = dict(state="running", source=source, day=day, detail="Starting map preparation")

            def progress(detail):
                with lock:
                    status["detail"] = detail

            def work():
                try:
                    model.prepare(source, day, progress)
                    with lock:
                        status.update(state="ready", detail="Map terrain and weather ready")
                except Exception as error:
                    app.logger.exception("Temperature map preparation failed")
                    with lock:
                        status.update(state="failed", error=str(error))

            worker = threading.Thread(target=work, name="temperature-map-preparation", daemon=True)
            worker.start()
            return jsonify(status), 202

    def raster_args():
        temperature = request.args.get("temperature")
        uncertainty = request.args.get("uncertainty", "0")
        if uncertainty not in ("0", "1"):
            raise ValueError("Uncertainty must be 0 or 1")
        return (request.args["source"], request.args["day"], int(request.args["hour"]),
                None if temperature is None else float(temperature), uncertainty == "1")

    @view.get("/api/temperature-map/grid")
    def grid():
        args = raster_args()
        with lock:
            model = get_backend()
            raster, top_left, metadata = model.raster(*args)
            return jsonify(model.geojson(raster, top_left, metadata))

    @view.get("/api/temperature-map/export")
    def export():
        args = raster_args()
        with lock:
            model = get_backend()
            raster, top_left, metadata = model.raster(*args)
            data = model.geotiff(raster, top_left, metadata)
            return send_file(BytesIO(data), mimetype="image/tiff", as_attachment=True,
                             download_name=f"experimental_temperature_{metadata['source']}_{metadata['time'][:10]}.tif")

    @view.errorhandler(ValueError)
    def invalid(error):
        app.logger.warning("Invalid temperature map input/data: %s", error)
        return jsonify(error=str(error)), 400

    @view.errorhandler(Exception)
    def failed(error):
        from werkzeug.exceptions import HTTPException
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        app.logger.exception("Temperature map request failed")
        return jsonify(error=str(error)), 500

    app.register_blueprint(view)
    from station_model_evaluation import register_model_evaluation
    register_model_evaluation(app, root)
