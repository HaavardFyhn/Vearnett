"""Original best-source Frost network calculation with new seNorge transfer weights."""
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from senorge_corrected_transfer import features
from senorge_transfer_map import SeNorgeTransferMap
from remote_senorge.train_valley_neural import sha256


WARNING = (
    "New seNorge GPU corrected-RMSE model, original app's minimum-over-stations calculation. "
    "Frost catalogue locations define the network; temperatures come from the historical seNorge "
    "grid, NOT actual Frost QC readings. No MEPS. Each cell uses the lowest predicted corrected "
    "RMSE among eligible station references within 46 km. This is best-source coverage, NOT "
    "multi-station fusion or a calibrated confidence interval. Removing a station uses the "
    "next-best remaining source, so RMSE cannot decrease; without another eligible source the "
    "cell becomes NoData. seNorge-to-real-station accuracy is not independently validated."
)


def network_scenario(context, removed=-1):
    full = np.isfinite(context["best"])
    affected = (context["winner"] == removed) & full if removed >= 0 else np.zeros(len(full), bool)
    radius = context["best"].copy()
    temperature = context["temperature"].copy()
    if removed >= 0:
        radius[affected] = context["second"][affected]
        temperature[affected] = context["second_temperature"][affected]
    valid = np.isfinite(radius)
    lost = full & ~valid
    change = np.full(len(full), np.nan, np.float32)
    change[full & valid] = radius[full & valid]-context["best"][full & valid]
    if np.any(change[full & valid] < 0):
        raise ValueError("Best-source station removal unexpectedly decreased RMSE")
    return dict(radius=radius, temperature=temperature, change=change, lost=lost,
                affected=affected, full=full, valid=valid)


class SeNorgeNetworkMap(SeNorgeTransferMap):
    def __init__(self, root):
        super().__init__(root)
        self.root = Path(root)
        catalogue = self.root / "outputs" / "national_station_dataset" / "station_splits.csv"
        self.stations = pd.read_csv(catalogue)
        self.folder = self.root / "outputs" / "senorge_network_map"
        self.folder.mkdir(exist_ok=True)
        self.fingerprint = sha256(Path(__file__))
        self.catalogue_sha = sha256(catalogue)
        self.raster_cache = OrderedDict()
        self.context = None

    def prepare_network(self, day, variable):
        if day not in self.days or variable not in (0, 1):
            raise ValueError("Choose a cached seNorge date and tg/tn")
        import hashlib
        payload = dict(day=day, variable=variable, code=self.fingerprint,
                       adapter=sha256(self.root / "senorge_transfer_map.py"),
                       catalogue=self.catalogue_sha, provenance=self.model.manifest["provenance"],
                       selected=self.model.selected,
                       models=sha256(self.root / "outputs" / "senorge_transfer_gpu_returned" / "portable_ridge.json"))
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        receipt, path = self.folder / f"{key}.json", self.folder / f"{key}.npz"
        if receipt.exists():
            metadata = json.loads(receipt.read_text())
            if sha256(path) != metadata["sha256"]:
                raise ValueError("Changed saved seNorge network prediction")
            with np.load(path, allow_pickle=False) as saved:
                context = {name: saved[name].copy() for name in saved.files}
        else:
            n = len(self.table)
            best, second = np.full(n, np.inf, np.float32), np.full(n, np.inf, np.float32)
            temperature, second_temperature = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
            winner, second_winner = np.full(n, -1, np.int32), np.full(n, -1, np.int32)
            source_cell = np.full(len(self.stations), -1, np.int32)
            source_c = np.full(len(self.stations), np.nan, np.float32)
            reasons = []
            date_index = self.days.index(day)
            for station, row in self.stations.iterrows():
                x, y = self.to_xy.transform(row.lon, row.lat)
                _, cell = self.tree.query([x, y])
                if np.max(np.abs(self.table[cell, :2]-[x, y])) > 500:
                    reasons.append("Station coordinate is outside a supported native land cell")
                    continue
                value = float(self.weather[variable, cell, date_index])
                if not np.isfinite(value):
                    reasons.append("No archived seNorge temperature at this location/date")
                    continue
                reasons.append("")
                source_cell[station], source_c[station] = cell, value
                targets = np.sort(self.tree.query_ball_point(self.table[cell, :2], 46000)).astype(int)
                inputs = features(self.table, np.full(len(targets), cell), targets,
                    np.full(len(targets), value), np.repeat(day, len(targets)), np.full(len(targets), variable))
                prediction, rmse = self.model.predict(inputs, value)
                prediction, rmse = prediction.astype(np.float32), rmse.astype(np.float32)
                identical = targets == cell
                prediction[identical], rmse[identical] = value, 0
                better = rmse < best[targets]
                improve_second = ~better & (rmse < second[targets])
                promoted, alternate = targets[better], targets[improve_second]
                second[promoted], second_temperature[promoted], second_winner[promoted] = (
                    best[promoted], temperature[promoted], winner[promoted])
                best[promoted], temperature[promoted], winner[promoted] = (
                    rmse[better], prediction[better], station)
                second[alternate], second_temperature[alternate], second_winner[alternate] = (
                    rmse[improve_second], prediction[improve_second], station)
            context = dict(best=best, second=second, temperature=temperature,
                second_temperature=second_temperature, winner=winner, second_winner=second_winner,
                source_cell=source_cell, source_c=source_c, reasons=np.asarray(reasons, dtype=str))
            full = np.isfinite(best)
            if not full.any():
                raise ValueError("No eligible seNorge station references for this date")
            if np.any(second[full] < best[full]):
                raise ValueError("Invalid next-best source predictions")
            pending = path.with_suffix(".pending.npz")
            np.savez_compressed(pending, **context)
            pending.replace(path)
            metadata = dict(key=key, day=day, variable=variable, stations=len(self.stations),
                eligible_stations=int((source_cell >= 0).sum()), cells=int(full.sum()),
                no_prediction_cells=int((~full).sum()),
                median_predicted_rmse_c=float(np.median(best[full])),
                sha256=sha256(path), warning=WARNING, selected=self.model.selected)
            pending_receipt = receipt.with_suffix(".pending.json")
            pending_receipt.write_text(json.dumps(metadata, indent=2))
            pending_receipt.replace(receipt)
        self.context = context
        self.metadata = metadata
        self.raster_cache.clear()
        self.key = key
        return dict(metadata=metadata, catalogue=self.catalogue())

    def catalogue(self):
        c = self.context
        return [dict(station_id=row.station_id, name=row["name"], lon=float(row.lon), lat=float(row.lat),
                     available=int(c["source_cell"][i]) >= 0, reason=str(c["reasons"][i]),
                     source_c=float(c["source_c"][i]) if np.isfinite(c["source_c"][i]) else None,
                     source_cell=int(c["source_cell"][i]))
                for i, row in self.stations.iterrows()]

    def require(self, key):
        if self.context is None or key != self.key:
            raise ValueError("Network prediction is not ready or has changed; prepare this date again")

    def selected(self, station):
        if not station:
            return -1
        matches = np.flatnonzero(self.stations.station_id.to_numpy() == station)
        if len(matches) != 1:
            raise ValueError("Unknown Frost catalogue station")
        return int(matches[0])

    def scenario(self, key, station, mode):
        self.require(key)
        if mode not in ("full", "without", "change"):
            raise ValueError("Choose full, without or change")
        removed = self.selected(station)
        cache_key = (station, mode)
        if cache_key not in self.raster_cache:
            values = network_scenario(self.context, removed if mode != "full" else -1)
            radius = values["change"] if mode == "change" else values["radius"]
            if mode == "change":
                temperature = values["temperature"]-self.context["temperature"]
            else:
                temperature = values["temperature"]
            self.rasters = {}
            for metric, data in (("rmse_c", radius), ("temperature_c", temperature)):
                raster = np.full(self.shape, np.nan, np.float32)
                keep = np.isfinite(data)
                ij = self.ij[keep]
                raster[ij[:, 0], ij[:, 1]] = data[keep]
                self.rasters[metric] = raster
            loss = np.zeros(self.shape, np.uint8)
            ij = self.ij[values["lost"]]
            loss[ij[:, 0], ij[:, 1]] = 1
            self.raster_cache[cache_key] = (self.rasters, loss)
            while len(self.raster_cache) > 4:
                self.raster_cache.popitem(last=False)
        self.rasters, self.loss_raster = self.raster_cache[cache_key]

    def summary(self, key, station):
        self.require(key)
        removed = self.selected(station)
        values = network_scenario(self.context, removed)
        return dict(affected_cells=int(values["affected"].sum()), lost_cells=int(values["lost"].sum()),
                    increased_cells=int((values["change"] > 0).sum()),
                    unchanged_cells=int((values["change"] == 0).sum()),
                    max_increase_c=float(np.nanmax(values["change"])),
                    station=station, metadata=self.metadata)

    def network_detail(self, lon, lat, key, station):
        self.require(key)
        cell = self.locate(lon, lat)
        values = network_scenario(self.context, self.selected(station))
        if not values["full"][cell]:
            raise ValueError("No eligible reference within 46 km: no seNorge network prediction here")

        def source(index):
            if index < 0:
                return None
            row = self.stations.iloc[index]
            return dict(station_id=row.station_id, name=row["name"],
                        source_c=float(self.context["source_c"][index]))

        return dict(cell=cell, full_rmse_c=float(self.context["best"][cell]),
            full_temperature_c=float(self.context["temperature"][cell]),
            source=source(int(self.context["winner"][cell])),
            next_source=source(int(self.context["second_winner"][cell])),
            without_rmse_c=float(values["radius"][cell]) if values["valid"][cell] else None,
            increase_c=float(values["change"][cell]) if values["valid"][cell] else None,
            lost_prediction=bool(values["lost"][cell]))

    def network_export(self, key, station, mode):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        self.scenario(key, station, mode)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=self.shape[1], height=self.shape[0], count=2,
                dtype="float32", crs="EPSG:25833", nodata=-9999, compress="deflate", predictor=3,
                transform=from_origin(self.origin[0]-500, self.origin[1]+500, 1000, 1000)) as image:
                for band, metric in enumerate(("temperature_c", "rmse_c"), 1):
                    raster = self.rasters[metric]
                    image.write(np.where(np.isfinite(raster), raster, -9999), band)
                    image.set_band_description(band, metric if mode != "change" else "change_"+metric)
                image.update_tags(model="new-seNorge-GPU-best-source-network", warning=WARNING,
                    day=self.metadata["day"], variable=str(self.metadata["variable"]),
                    removed_station=station, scenario=mode)
            return memory.read()


def register_senorge_network_map(app, root):
    from flask import Blueprint, jsonify, request, send_file
    from werkzeug.exceptions import HTTPException
    view = Blueprint("senorge_network_map", __name__)
    lock = threading.RLock()
    backend = None
    state = dict(state="idle", detail="Prepare a seNorge network date")

    def model():
        nonlocal backend
        if backend is None:
            backend = SeNorgeNetworkMap(root)
        return backend

    @view.get("/senorge-network")
    def page():
        return send_file(Path(root) / "senorge_network_map.html")

    @view.get("/senorge_network_map.js")
    def script():
        return send_file(Path(root) / "senorge_network_map.js", mimetype="application/javascript")

    @view.get("/api/senorge-network/options")
    def options():
        plan = pd.read_csv(Path(root) / "outputs" / "senorge_footprint_expanded" / "selected_dates.csv")
        return jsonify(days=plan.date.tolist(), warning=WARNING)

    @view.post("/api/senorge-network/prepare")
    def prepare():
        day, variable = request.args["day"], int(request.args["variable"])
        if variable not in (0, 1):
            raise ValueError("Choose tg or tn")
        with lock:
            if state["state"] == "running":
                raise ValueError("A network calculation is already running")
            state.update(state="running", detail="Checking seNorge models/archive and computing all Frost references",
                         day=day, variable=variable)

        def build():
            try:
                data = model().prepare_network(day, variable)
                with lock:
                    state.clear()
                    state.update(state="ready", detail="seNorge network ready", **data)
            except Exception as error:
                app.logger.exception("seNorge network prediction failed")
                with lock:
                    state.clear()
                    state.update(state="failed", detail=str(error))

        threading.Thread(target=build, daemon=True).start()
        return jsonify(state), 202

    @view.get("/api/senorge-network/status")
    def status():
        with lock:
            return jsonify(state)

    def current():
        if state["state"] != "ready":
            raise ValueError("Prepare the network and wait until it is ready")
        return model(), request.args["key"], request.args.get("station", "")

    @view.get("/api/senorge-network/summary")
    def summary():
        with lock:
            current_model, key, station = current()
            return jsonify(current_model.summary(key, station))

    @view.get("/api/senorge-network/cell")
    def cell():
        with lock:
            current_model, key, station = current()
            return jsonify(current_model.network_detail(float(request.args["lon"]), float(request.args["lat"]),
                                                        key, station))

    @view.get("/api/senorge-network/tile/<int:z>/<int:x>/<int:y>.png")
    def tile(z, x, y):
        with lock:
            current_model, key, station = current()
            current_model.scenario(key, station, request.args.get("mode", "full"))
            return send_file(BytesIO(current_model.tile(z, x, y, request.args.get("metric", "rmse_c"),
                                                        key, lost=current_model.loss_raster)),
                             mimetype="image/png")

    @view.get("/api/senorge-network/export")
    def export():
        with lock:
            current_model, key, station = current()
            return send_file(BytesIO(current_model.network_export(key, station, request.args.get("mode", "full"))),
                             mimetype="image/tiff", as_attachment=True,
                             download_name="senorge_gpu_frost_network_1km.tif")

    @view.errorhandler(ValueError)
    def invalid(error):
        app.logger.warning("Invalid seNorge network request: %s", error)
        return jsonify(error=str(error)), 400

    @view.errorhandler(Exception)
    def failed(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        app.logger.exception("seNorge network map request failed")
        return jsonify(error=str(error)), 500

    app.register_blueprint(view)
