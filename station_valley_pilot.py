"""Read-only visualization of saved, held-out valley-pilot station transfers."""
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from remote_senorge.train_remote import sha256
from train_valley_pilot import scores


METHODS = {
    "forecast_transfer": "Overføring av MEPS-differanse",
    "terrain": "Terreng",
    "regime": "Terreng + værregime",
    "graph_regime": "Graf + værregime",
}


class ValleyResults:
    def __init__(self, root):
        self.folder = Path(root) / "outputs" / "valley_regime_pilot_results"
        data = Path(root) / "outputs" / "valley_regime_pilot"
        self.report = json.loads((self.folder / "report.json").read_text())
        for name, digest in self.report["input_checksums"].items():
            if Path(name).name != name or sha256(data / name) != digest:
                raise ValueError(f"Pilot input checksum mismatch: {name}")
        for name, digest in self.report["model_checksums"].items():
            if Path(name).name != name or sha256(self.folder / name) != digest:
                raise ValueError(f"Pilot model checksum mismatch: {name}")
        self.rows = pd.read_csv(self.folder / "test_predictions.csv")
        columns = {"source", "target", "time", "night", "split", "source_c", "target_c", "calm_clear"}
        columns.update(name+"_prediction_c" for name in METHODS)
        if not columns <= set(self.rows.columns) or self.rows.empty:
            raise ValueError("Missing pilot test prediction columns")
        numeric = ["source_c", "target_c", *(name+"_prediction_c" for name in METHODS)]
        if (not np.isfinite(self.rows[numeric].to_numpy()).all() or
                not self.rows["split"].eq(2).all() or
                not self.rows["calm_clear"].isin([True, False]).all() or
                self.rows.duplicated(["source", "target", "time"]).any()):
            raise ValueError("Invalid or duplicate held-out pilot predictions")
        stamps = pd.to_datetime(self.rows["time"], utc=True)
        if (not (stamps.dt.year-stamps.dt.month.lt(12).astype(int)).eq(2024).all() or
                not stamps.dt.month.isin([12, 1, 2]).all() or
                not stamps.dt.hour.isin([0, 3, 6]).all() or
                not stamps.dt.minute.eq(0).all() or
                not stamps.dt.second.eq(0).all() or
                not (stamps.dt.strftime("%Y-%m-%d") == self.rows["night"]).all()):
            raise ValueError("Pilot test timestamps differ from declared nighttime samples")
        self.rows["time"] = stamps.map(lambda stamp: stamp.isoformat())
        if (self.rows.groupby(["target", "time"])["target_c"].nunique() != 1).any():
            raise ValueError("Conflicting target observations")
        catalogue = pd.read_csv(data / "stations.csv")
        ids = set(self.rows["source"]) | set(self.rows["target"])
        if catalogue["station_id"].duplicated().any() or not ids <= set(catalogue["station_id"]):
            raise ValueError("Missing or ambiguous pilot station coordinates")
        selected = catalogue.loc[catalogue["station_id"].isin(ids)]
        if not np.isfinite(selected[["lat", "lon"]].to_numpy()).all():
            raise ValueError("Invalid pilot station coordinates")
        self.stations = selected[["station_id", "name", "lat", "lon", "group"]].to_dict("records")
        groups = catalogue.set_index("station_id")["group"]
        if (not all(groups[source] == 0 for source in self.rows["source"]) or
                not all(groups[target] == 2 for target in self.rows["target"])):
            raise ValueError("Pilot display contains non-held-out sources/targets")
        for method in METHODS:
            computed = scores(self.rows, self.rows[method+"_prediction_c"].to_numpy())
            expected = self.report["variants"][method]["test"]
            for metric in ("rows", "nights", "rmse_c", "mae_c", "bias_c"):
                if not np.isclose(computed[metric], expected[metric], atol=1e-8, rtol=1e-8):
                    raise ValueError(f"Saved pilot prediction/score mismatch: {method}/{metric}")

    def subset(self, target, condition):
        if target not in set(self.rows["target"]):
            raise ValueError("Ukjent teststasjon")
        if condition not in ("all", "calm_clear", "cold"):
            raise ValueError("Ugyldig hendelsesfilter")
        result = self.rows.loc[self.rows["target"] == target]
        if condition == "calm_clear":
            result = result.loc[result["calm_clear"]]
        elif condition == "cold":
            result = result.loc[result["target_c"] <= -5]
        return result

    def options(self, target=None, condition="all"):
        target = target or sorted(self.rows["target"].unique())[0]
        rows = self.subset(target, condition)
        return dict(
            methods=[dict(id=name, label=label) for name, label in METHODS.items()],
            selected_by_validation=self.report["selected_by_validation"],
            targets=sorted(self.rows["target"].unique()), target=target, condition=condition,
            times=sorted(rows["time"].unique()), stations=self.stations,
            sources=sorted(rows["source"].unique()), rows=len(rows), nights=rows["night"].nunique(),
            summary={name: scores(rows, rows[name+"_prediction_c"].to_numpy())
                     for name in METHODS} if not rows.empty else {},
            intervals={name: self.report["variants"][name]["test"]["interval_radius_c"] for name in METHODS},
            limitations=self.report["limitations"], prediction_sha256=sha256(self.folder / "test_predictions.csv"))

    def case(self, target, timestamp, condition):
        rows = self.subset(target, condition)
        part = rows.loc[rows["time"] == timestamp]
        if part.empty:
            raise ValueError("Ingen sammenlignbare testdata for dette tidspunktet/filteret")
        return dict(target=target, time=timestamp, observed_c=float(part["target_c"].iloc[0]),
                    transfers=part.sort_values("source").to_dict("records"))

    def timeline(self, target, source, condition):
        rows = self.subset(target, condition)
        if source not in set(self.rows["source"]):
            raise ValueError("Ukjent kildestasjon")
        rows = rows.loc[rows["source"] == source].sort_values("time")
        return dict(target=target, source=source, points=rows.to_dict("records"),
                    summary={name: scores(rows, rows[name+"_prediction_c"].to_numpy())
                             for name in METHODS} if not rows.empty else {})


def register_valley_view(app, root):
    from flask import Blueprint, jsonify, request, send_file
    root = Path(root)
    view = Blueprint("valley_pilot", __name__)
    backend = None
    lock = threading.RLock()

    def results():
        nonlocal backend
        with lock:
            if backend is None:
                backend = ValleyResults(root)
            return backend

    def args():
        model = results()
        target = request.args.get("target", sorted(model.rows["target"].unique())[0])
        return target, request.args.get("condition", "all")

    @view.get("/valley-pilot")
    def index():
        return send_file(root / "station_valley_pilot.html")

    @view.get("/station_valley_pilot.js")
    def javascript():
        return send_file(root / "station_valley_pilot.js", mimetype="text/javascript")

    @view.get("/api/valley-pilot/options")
    def options():
        return jsonify(results().options(*args()))

    @view.get("/api/valley-pilot/case")
    def case():
        target, condition = args()
        return jsonify(results().case(target, request.args["time"], condition))

    @view.get("/api/valley-pilot/timeline")
    def timeline():
        target, condition = args()
        return jsonify(results().timeline(target, request.args["source"], condition))

    @view.get("/api/valley-pilot/export")
    def export():
        results()
        return send_file(root / "outputs" / "valley_regime_pilot_results" / "test_predictions.csv",
                         as_attachment=True, download_name="valley_pilot_heldout_predictions.csv")

    @view.errorhandler(ValueError)
    @view.errorhandler(KeyError)
    def invalid(error):
        app.logger.warning("Invalid valley pilot data/request: %s", error)
        return jsonify(error=str(error)), 400

    @view.errorhandler(OSError)
    def missing(error):
        app.logger.exception("Valley pilot results unavailable")
        return jsonify(error=str(error)), 503

    app.register_blueprint(view)
