"""Read-only evaluation maps for verified joint-station and seNorge GPU results."""
from io import BytesIO
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
from pyproj import Transformer

from remote_senorge.train_valley_neural import sha256


def metrics(frame, model):
    error = frame.prediction_c.to_numpy()-frame.target_c.to_numpy()
    result = dict(rows=len(frame), targets=frame.target.nunique(),
                  rmse_c=float(np.sqrt(np.mean(error**2))), mae_c=float(np.abs(error).mean()),
                  bias_c=float(error.mean()))
    if model == "senorge":
        result["predicted_pooled_rmse_c"] = float(np.sqrt(np.mean(frame.predicted_rmse_c.to_numpy()**2)))
    else:
        result.update(interval_radius_median_c=float(frame.radius_c.median()),
                      individual_coverage=float((np.abs(error) <= frame.radius_c).mean()))
    return result


class ModelEvaluation:
    def __init__(self, root):
        self.root = Path(root)
        self.transform = Transformer.from_crs(25833, 4326, always_xy=True)
        joint = self.root / "outputs" / "joint_uncertainty_returned"
        senorge = self.root / "outputs" / "senorge_transfer_gpu_returned"
        self.frames, self.reports, self.receipts = {}, {}, {}
        for key, folder, result_name in (
                ("joint", joint, "joint_uncertainty_results"),
                ("senorge", senorge, "senorge_transfer_gpu_results")):
            receipt = json.loads((folder / "verification.json").read_text())
            result = folder / result_name
            if receipt["report_sha256"] != sha256(result / "report.json"):
                raise ValueError(f"Changed verified {key} report")
            self.reports[key] = json.loads((result / "report.json").read_text())
            self.receipts[key] = receipt
            if key == "joint":
                if not receipt["all_test_configurations_verified"]:
                    raise ValueError("Joint predictions not verified")
                file = joint / "verified_full_predictions.csv"
                if sha256(file) != receipt["verified_full_predictions_sha256"]:
                    raise ValueError("Changed verified joint predictions")
                contributions = result / "test_contributions.csv"
                if sha256(contributions) != self.reports[key]["checksums"]["test_contributions.csv"]:
                    raise ValueError("Changed joint contribution diagnostics")
                self.deletions = pd.read_csv(contributions, keep_default_na=False)
                frame = pd.read_csv(file, keep_default_na=False)
                stamps = pd.to_datetime(frame.time, utc=True)
                frame["date"], frame["hour"] = stamps.dt.strftime("%Y-%m-%d"), stamps.dt.hour
                frame["split"] = "test"
                self.frames[key] = frame
            else:
                if not receipt["all_predictions_verified"]:
                    raise ValueError("seNorge predictions not verified")
                parts = []
                for split, digest in receipt["csv_sha256"].items():
                    file = result / f"{split}_predictions.csv"
                    if sha256(file) != digest:
                        raise ValueError("Changed verified seNorge predictions")
                    frame = pd.read_csv(file, dtype={"source": str, "target": str})
                    frame["split"] = split
                    parts.append(frame)
                self.frames[key] = pd.concat(parts, ignore_index=True)
        spec_file = joint / "joint_spec.json"
        run = json.loads((joint / "joint_uncertainty_results" / "run_config.json").read_text())
        if sha256(spec_file) != run["inputs"]["joint_spec.json"]:
            raise ValueError("Changed joint evaluation coordinates")
        spec = json.loads(spec_file.read_text())
        catalogue = pd.read_csv(self.root / "outputs" / "national_station_dataset" / "station_splits.csv")
        names = dict(zip(catalogue.station_id, catalogue.name))
        self.stations = {key: dict(x=value[0], y=value[1], name=names.get(key, key))
                         for key, value in spec["station_xy"].items()}
        atlas = self.root / "outputs" / "senorge_footprint_expanded" / "grid_table.npy"
        manifest_file = senorge / "manifest.json"
        if sha256(manifest_file) != self.receipts["senorge"]["manifest_sha256"]:
            raise ValueError("Changed seNorge manifest")
        manifest = json.loads(manifest_file.read_text())
        if sha256(atlas) != manifest["provenance"]["grid_table.npy"]:
            raise ValueError("Changed seNorge evaluation grid")
        self.table = np.load(atlas, mmap_mode="r", allow_pickle=False)

    def options(self):
        models = {}
        for key, frame in self.frames.items():
            models[key] = dict(splits=sorted(frame.split.unique()), dates={
                split: sorted(frame.loc[frame.split.eq(split), "date"].unique())
                for split in sorted(frame.split.unique())},
                sources=sorted(frame.source.unique()) if key == "senorge" else [],
                report=self.reports[key], overall=metrics(frame.loc[frame.split.eq("test")], key),
                warning=(
                    "Held-out real-station evaluation with MEPS background. Interval radius is NOT RMSE. "
                    "Full joint night coverage87.8%; full-and-deletion73.3%,below90%target. "
                    "Station removal is a signed model sensitivity,not causal or additive importance. "
                    "Only17test targets/two test regions; no nationwide grid coverage guarantee."
                    if key == "joint" else
                    "Held-out seNorge1km grid-to-grid evaluation,not real-station accuracy or a forecast. "
                    "Predicted RMSE=sqrt(predicted MSE),NOT an interval radius. Only sampled dates/cells "
                    "in four already-inspected test blocks. Multiple source evaluations at a target are "
                    "summaries,not a joint temperature model. Environment version file was missing; "
                    "all assessment/test predictions have nevertheless been reproduced.")
                )
        return dict(models=models, default_date=dict(joint="2024-12-15", senorge="2024-12-15"),
                    evaluation_only=True)

    def select(self, model, split, date, hour="3", variable="0", source=""):
        if model not in self.frames:
            raise ValueError("Choose joint or senorge")
        frame = self.frames[model]
        if split not in set(frame.split) or date not in set(frame.loc[frame.split.eq(split), "date"]):
            raise ValueError("Choose an available evaluation split/date")
        selected = frame.split.eq(split) & frame.date.eq(date)
        if model == "joint":
            if hour not in ("0", "3", "6"):
                raise ValueError("Choose00,03or06UTC")
            if source:
                raise ValueError("Joint evaluation does not accept a single source filter")
            selected &= frame.hour.eq(int(hour))
        else:
            if variable not in ("0", "1"):
                raise ValueError("Choose daily mean tg or daily minimum tn")
            if source and source not in set(frame.source):
                raise ValueError("Unknown seNorge source cell")
            selected &= frame.variable.eq(int(variable))
            if source:
                selected &= frame.source.eq(source)
        part = frame.loc[selected].copy()
        if part.empty:
            raise ValueError("No evaluated predictions for these filters; no interpolation or fallback")
        return part

    def position(self, model, target):
        if model == "joint":
            row = self.stations[target]
            return float(row["x"]), float(row["y"])
        index = int(target)
        if index < 0 or index >= len(self.table):
            raise ValueError("Invalid seNorge cell index")
        return tuple(map(float, self.table[index, :2]))

    def grid(self, model, part):
        features = []
        for target, rows in part.groupby("target", sort=True):
            x, y = self.position(model, target)
            summary = metrics(rows, model)
            summary.update(target=str(target), prediction_c=float(rows.prediction_c.mean()),
                           target_c=float(rows.target_c.mean()), x=x, y=y)
            summary["name"] = self.stations[target]["name"] if model == "joint" else f"Cell {target}"
            if model == "joint":
                summary.update(radius_c=float(rows.radius_c.iloc[0]), lower_c=float(rows.lower_c.iloc[0]),
                               upper_c=float(rows.upper_c.iloc[0]),
                               source_count=int(rows.source_count.iloc[0]))
                lon, lat = self.transform.transform(x, y)
                geometry = dict(type="Point", coordinates=[lon, lat])
            else:
                corners = np.array([[x-500, y+500], [x+500, y+500], [x+500, y-500],
                                    [x-500, y-500], [x-500, y+500]])
                lon, lat = self.transform.transform(corners[:, 0], corners[:, 1])
                geometry = dict(type="Polygon", coordinates=[np.column_stack([lon, lat]).tolist()])
            features.append(dict(type="Feature", properties=summary, geometry=geometry))
        return dict(type="FeatureCollection", features=features, summary=metrics(part, model),
                    evaluation_only=True, aggregation=(
                        "Each station has one full joint prediction" if model == "joint" else
                        "Per-target summaries of separate source-to-target evaluations; NOT a fusion prediction. "
                        "Temperature shown is the mean of evaluated predictions. Actual RMSE and predicted RMSE "
                        "are pooled across rows; click a target for separate source predictions."))

    def detail(self, model, part, target):
        rows = part.loc[part.target.eq(target)]
        if rows.empty:
            raise ValueError("Target is not in the selected evaluation view")
        if model == "senorge":
            records = []
            for row in rows.itertuples():
                x, y = self.position(model, row.source)
                lon, lat = self.transform.transform(x, y)
                records.append(dict(source=row.source, prediction_c=row.prediction_c,
                                    source_c=row.source_c, target_c=row.target_c,
                                    error_c=row.prediction_c-row.target_c, predicted_rmse_c=row.predicted_rmse_c,
                                    lat=lat, lon=lon, name=f"Cell {row.source}"))
        else:
            row = rows.iloc[0]
            deletions = self.deletions.loc[self.deletions.target.eq(target) &
                         self.deletions.time.eq(row.time) & self.deletions.removed_source.ne("")]
            records = []
            readings = json.loads(row.source_readings_json)
            temperatures = {reading["source"]: reading["source_c"] for reading in readings}
            if len(temperatures) != int(row.source_count) or not all(
                    np.isfinite(value) for value in temperatures.values()):
                raise ValueError("Invalid verified source readings")
            for removed in deletions.itertuples():
                xy = self.stations[removed.removed_source]
                lon, lat = self.transform.transform(xy["x"], xy["y"])
                records.append(dict(source=removed.removed_source, name=xy["name"], lat=lat, lon=lon,
                                    prediction_c=removed.prediction_c, radius_c=removed.radius_c,
                                    radius_change_c=removed.radius_change_when_removed_c,
                                    temperature_change_c=removed.temperature_change_when_removed_c,
                                    lower_c=removed.lower_c, upper_c=removed.upper_c,
                                    source_c=temperatures[removed.removed_source], deletion_available=True))
            if int(row.source_count) == 1 and readings:
                reading = readings[0]
                xy = self.stations[reading["source"]]
                lon, lat = self.transform.transform(xy["x"], xy["y"])
                records.append(dict(source=reading["source"], name=xy["name"], lat=lat, lon=lon,
                                    source_c=reading["source_c"], deletion_available=False))
        return dict(target=target, sources=records, summary=metrics(rows, model),
                    sole_source_no_deletion=model == "joint" and int(rows.source_count.iloc[0]) == 1,
                    explanation=("Source rows are separate evaluations; not uncertainty contributions"
                                 if model == "senorge" else
                                 "Changes after removing a reading,without replacement. Positive radius change "
                                 "means wider without it; negative means narrower. Not causal/additive importance."))


def register_model_evaluation(app, root, backend=None):
    from flask import Blueprint, jsonify, request, send_file
    from werkzeug.exceptions import HTTPException
    root = Path(root)
    view = Blueprint("model_evaluation", __name__)
    lock = threading.RLock()

    def model():
        nonlocal backend
        with lock:
            if backend is None:
                backend = ModelEvaluation(root)
            return backend

    @view.get("/model-evaluation")
    def index():
        return send_file(root / "station_model_evaluation.html")

    @view.get("/station_model_evaluation.js")
    def javascript():
        return send_file(root / "station_model_evaluation.js", mimetype="text/javascript")

    @view.get("/api/model-evaluation/options")
    def options():
        return jsonify(model().options())

    def selection():
        key = request.args.get("model", "joint")
        return key, model().select(key, request.args.get("split", "test"), request.args.get("date", ""),
                                  request.args.get("hour", "3"), request.args.get("variable", "0"),
                                  request.args.get("source", ""))

    @view.get("/api/model-evaluation/grid")
    def grid():
        key, rows = selection()
        return jsonify(model().grid(key, rows))

    @view.get("/api/model-evaluation/detail")
    def detail():
        key, rows = selection()
        return jsonify(model().detail(key, rows, request.args.get("target", "")))

    @view.get("/api/model-evaluation/export")
    def export():
        key, rows = selection()
        rows["error_c"] = rows.prediction_c-rows.target_c
        data = rows.to_csv(index=False).encode("utf-8")
        return send_file(BytesIO(data), mimetype="text/csv", as_attachment=True,
                         download_name=f"verified_{key}_evaluation_{request.args['date']}.csv")

    @view.errorhandler(ValueError)
    def invalid(error):
        app.logger.warning("Invalid evaluation request/data: %s", error)
        return jsonify(error=str(error)), 400

    @view.errorhandler(Exception)
    def failed(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        app.logger.exception("Evaluation request failed")
        return jsonify(error=str(error)), 500

    app.register_blueprint(view)
    from national_joint_map import register_national_joint_map
    register_national_joint_map(app, root)
    from national_station_removal import register_station_removal
    register_station_removal(app, root)
    from senorge_transfer_map import register_senorge_transfer_map
    register_senorge_transfer_map(app, root)
    from senorge_network_map import register_senorge_network_map
    register_senorge_network_map(app, root)
