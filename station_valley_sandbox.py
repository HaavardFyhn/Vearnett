"""Synthetic station-transfer sandbox; never installs or assimilates predictions."""
from collections import deque
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from prepare_valley_pilot import validate_day, forecast_url
from remote_senorge.train_valley_neural import create_model
from remote_senorge.train_remote import sha256
from train_valley_pilot import static_features
from valley_pilot import HOURS, REGIME_FIELDS


def transfer_features(table, graph, source, target, temperature, stamp, forecast):
    hour = HOURS.index(stamp.hour)
    fields = np.array([forecast["fields"][field][hour] for field in REGIME_FIELDS]).T
    a, b = fields[source], fields[target]
    speed_a, speed_b = np.linalg.norm(a[1:3]), np.linalg.norm(b[1:3])
    delta = b[0]-a[0]
    along = float(np.dot(b[1:3], forecast["bearing"][source][target]))
    barrier = graph[source, target]
    features = [
        temperature, np.sin(2*np.pi*stamp.dayofyear/365.25),
        np.cos(2*np.pi*stamp.dayofyear/365.25), stamp.hour/6,
        *static_features(table, source, target), *a, *b, delta,
        temperature-a[0], speed_a, speed_b, *barrier, along,
        abs(along)*barrier[2], speed_a*barrier[1], speed_b*barrier[2],
        (1-b[3])*barrier[2], (temperature-a[0])*barrier[1],
    ]
    return np.asarray(features, np.float32), float(delta)


def propagate(stations, table, graph, root, temperature, max_hops, predict):
    """Breadth-first rooted tree; nearest parents first, no revisits or feedback."""
    ids = stations["station_id"].tolist()
    if root not in ids or not np.isfinite(temperature) or not -20 <= temperature <= 20:
        raise ValueError("Choose a known source and a temperature between -20 and20C")
    if type(max_hops) is not int or not 1 <= max_hops <= 3:
        raise ValueError("Maximum hops must be an integer from1to3")
    index = ids.index(root)
    rows = [dict(station_id=root, parent=None, hop=0, temperature_c=float(temperature),
                 delta_c=0., distance_km=0., synthetic=True)]
    visited, queue = {index}, deque([(index, float(temperature), 0)])
    while queue:
        source, source_c, hop = queue.popleft()
        if hop >= max_hops:
            continue
        distance = np.linalg.norm(table[:, :2]-table[source, :2], axis=1)
        targets = sorted(range(len(ids)), key=lambda i: (distance[i], ids[i]))
        for target in targets:
            if target in visited or distance[target] > 45000 or graph[source, target, 0] != 1:
                continue
            prediction = float(predict(source, target, source_c))
            if not np.isfinite(prediction):
                raise ValueError("Model produced a nonfinite synthetic temperature")
            visited.add(target)
            rows.append(dict(station_id=ids[target], parent=ids[source], hop=hop+1,
                             temperature_c=prediction, delta_c=prediction-source_c,
                             distance_km=float(distance[target]/1000), synthetic=True))
            queue.append((target, prediction, hop+1))
    return dict(nodes=rows, unreachable=[identifier for i, identifier in enumerate(ids) if i not in visited])


class ValleySandbox:
    def __init__(self, root):
        import torch
        self.torch = torch
        root = Path(root)
        self.data = root / "outputs" / "valley_regime_pilot"
        result = root / "outputs" / "valley_neural_returned" / "valley_neural_results"
        self.summary = json.loads((result / "summary.json").read_text())
        manifest = json.loads((result.parent / "manifest.json").read_text())
        for name, digest in self.summary["checksums"].items():
            if Path(name).name != name or sha256(result / name) != digest:
                raise ValueError(f"Neural sandbox checksum mismatch: {name}")
        config = json.loads((result / "run_config.json").read_text())
        if sha256(result.parent / "manifest.json") != config["manifest_sha256"]:
            raise ValueError("Neural sandbox manifest mismatch")
        pilot = json.loads((self.data / "config.json").read_text())
        for name, digest in pilot["terrain_checksums"].items():
            if sha256(self.data / name) != digest:
                raise ValueError(f"Pilot terrain checksum mismatch: {name}")
        for name, digest in manifest["input_checksums"].items():
            if Path(name).name != name or sha256(self.data / name) != digest:
                raise ValueError(f"Neural pilot input changed: {name}")
        self.catalogue = pd.read_csv(self.data / "stations.csv")
        ids = {row["station_id"] for row in manifest["sensor_choices"]}
        self.indices = np.flatnonzero(self.catalogue["station_id"].isin(ids))
        self.stations = self.catalogue.iloc[self.indices].reset_index(drop=True)
        with np.load(self.data / "terrain.npz", allow_pickle=False) as terrain:
            self.table = terrain["table"].copy()
            self.graph = terrain["graph"].copy()
        with np.load(result / "normalization.npz", allow_pickle=False) as norm:
            self.mean, self.std = norm["mean"].copy(), norm["std"].copy()
        self.model = create_model(torch, self.summary["selected"]["config"], manifest)
        self.model.load_state_dict(torch.load(result / "selected_model.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.receipts = json.loads((self.data / "regimes_progress.json").read_text())
        if sha256(self.data / "regimes_progress.json") != manifest["input_checksums"]["regimes_progress.json"]:
            raise ValueError("Pilot forecast provenance changed")
        self.times = [f"{day}T{hour:02d}:00:00+00:00" for day in sorted(self.receipts) for hour in HOURS]
        self.cached_day, self.forecast = None, None

    def options(self):
        return dict(stations=self.stations[["station_id", "name", "lat", "lon"]].to_dict("records"),
                    times=self.times, selected=self.summary["selected"]["config"]["name"],
                    warning="Synthetic what-if experiment,not observations or a validated weather map. "
                    "One-target validation only; reversed/held-out sources,synthetic source values and multiple hops are extrapolations. "
                    "45km maximum per edge; no calibrated domino uncertainty. No interpolation between stations.")

    def simulate(self, source, temperature, hops, timestamp):
        if timestamp not in self.times:
            raise ValueError("Choose an archived pilot UTC timestamp")
        stamp = pd.Timestamp(timestamp)
        day = str(stamp.date())
        if day != self.cached_day:
            path = self.data / "regimes" / f"{day}.json"
            if sha256(path) != self.receipts[day]:
                raise ValueError("Pilot forecast day checksum mismatch")
            forecast = json.loads(path.read_text())
            validate_day(forecast, stamp.date(), self.catalogue, "regimes")
            if forecast["run_time"] != forecast_url(stamp.date())[1].isoformat():
                raise ValueError("Pilot forecast issue time mismatch")
            self.forecast, self.cached_day = forecast, day

        def predict(a, b, source_c):
            a, b = self.indices[a], self.indices[b]
            features, delta = transfer_features(self.table, self.graph, a, b, source_c, stamp, self.forecast)
            x = self.torch.from_numpy(((features-self.mean)/self.std).astype(np.float32)[None])
            with self.torch.inference_mode():
                correction = float(self.model(x)[0])
            return source_c+delta+correction

        result = propagate(
            self.stations, self.table[self.indices], self.graph[np.ix_(self.indices, self.indices)],
            source, temperature, hops, predict)
        result.update(time=timestamp, source=source, max_hops=hops)
        return result


def register_sandbox_view(app, root):
    from flask import Blueprint, jsonify, request, send_file
    root = Path(root)
    view = Blueprint("valley_sandbox", __name__)
    lock, backend = threading.RLock(), None

    def model():
        nonlocal backend
        if backend is None:
            backend = ValleySandbox(root)
        return backend

    @view.get("/valley-sandbox")
    def index():
        return send_file(root / "station_valley_sandbox.html")

    @view.get("/station_valley_sandbox.js")
    def javascript():
        return send_file(root / "station_valley_sandbox.js", mimetype="text/javascript")

    @view.get("/api/valley-sandbox/options")
    def options():
        with lock:
            return jsonify(model().options())

    @view.get("/api/valley-sandbox/simulate")
    def simulate():
        with lock:
            return jsonify(model().simulate(request.args["source"], float(request.args["temperature"]),
                                            int(request.args["hops"]), request.args["time"]))

    @view.errorhandler(ValueError)
    @view.errorhandler(KeyError)
    def invalid(error):
        app.logger.warning("Invalid valley sandbox input/data: %s", error)
        return jsonify(error=str(error)), 400

    @view.errorhandler(OSError)
    @view.errorhandler(RuntimeError)
    def unavailable(error):
        app.logger.exception("Valley sandbox unavailable")
        return jsonify(error=str(error)), 503

    app.register_blueprint(view)
