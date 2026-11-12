"""Virtual-reference maps using the returned, verified seNorge GPU models."""
from io import BytesIO
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from remote_senorge import train_senorge_transfer_gpu as trainer
from senorge_corrected_transfer import FEATURES, features


WARNING = (
    "New seNorge-only GPU temperature transfer and corrected RMSE models. No Frost stations or MEPS. "
    "Click a virtual reference anywhere on the supported 1 km land grid. Predictions use its temperature, "
    "date, daily mean/minimum and source/target terrain. Display limited to 46 km, the sampling range "
    "of this new training, not the old model's 100 km. RMSE is sqrt(predicted squared corrected error), "
    "NOT an interval radius or a forecast. Real-station accuracy and national coverage are not certified. "
    "The reference cell itself copies its known temperature with zero transfer error (identity rule)."
)


class SeNorgeTransferModel:
    def __init__(self, root):
        import torch
        self.torch = torch
        torch.set_num_threads(4)
        folder = Path(root) / "outputs" / "senorge_transfer_gpu_returned"
        result = folder / "senorge_transfer_gpu_results"
        verified = json.loads((folder / "verification.json").read_text())
        if not verified["all_predictions_verified"] or trainer.sha256(result / "report.json") != verified["report_sha256"]:
            raise ValueError("Verified completed seNorge results required")
        report = json.loads((result / "report.json").read_text())
        if report["state"] != "complete" or report["validation_only"]:
            raise ValueError("Completed full seNorge training required")
        if trainer.sha256(result / "run_config.json") != report["checksums"]["run_config.json"]:
            raise ValueError("Changed seNorge training run provenance")
        run = json.loads((result / "run_config.json").read_text())
        for name in ("senorge_corrected_transfer.py", "train_senorge_transfer_gpu.py"):
            path = (Path(root) / name if name == "senorge_corrected_transfer.py"
                    else Path(root) / "remote_senorge" / name)
            if trainer.sha256(path) != run["code"][name]:
                raise ValueError(f"Changed seNorge training feature/inference code: {name}")
        self.selected = report["selected"]
        receipt = json.loads((folder / "portable_ridge.json").read_text())
        if (trainer.sha256(result / "ridge.joblib") != receipt["model_sha256"] or
                receipt["model_sha256"] != report["checksums"]["ridge.joblib"] or
                trainer.sha256(folder / "portable_ridge.npz") != receipt["arrays_sha256"]):
            raise ValueError("Changed portable seNorge ridge")
        with np.load(folder / "portable_ridge.npz", allow_pickle=False) as saved:
            self.ridge = {name: saved[name].copy() for name in saved.files}
        self.networks = {}
        for name in (self.selected["mean_checkpoint"], self.selected["error_checkpoint"]):
            if name is None:
                continue
            info = receipt["networks"][name]
            if (trainer.sha256(result / name) != info["original_sha256"] or
                    info["original_sha256"] != report["checksums"][name]):
                raise ValueError("Changed original seNorge network")
            for filename, digest in info["checksums"].items():
                if trainer.sha256(folder / filename) != digest:
                    raise ValueError("Changed portable seNorge network")
            config = info["config"]
            count = len(FEATURES) + int(config["error_stage"])
            model = trainer.build_model(torch, count, config["width"], config["error_stage"])
            model.load_state_dict(torch.load(folder / info["weights"], map_location="cpu", weights_only=True))
            with np.load(folder / info["normalization"], allow_pickle=False) as norm:
                self.networks[name] = (model.eval(), norm["mean"].copy(), norm["std"].copy())
        if trainer.sha256(folder / "manifest.json") != verified["manifest_sha256"]:
            raise ValueError("Changed seNorge training manifest")
        self.manifest = json.loads((folder / "manifest.json").read_text())
        if self.manifest["features"] != FEATURES:
            raise ValueError("seNorge feature order differs from training")

    def network(self, name, x):
        model, mean, std = self.networks[name]
        from types import SimpleNamespace
        return trainer.infer(self.torch, model, x, mean, std,
                             SimpleNamespace(device="cpu", batch_size=4096))

    def predict(self, x, source_c):
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
            raise ValueError("Invalid seNorge transfer features")
        source_c = np.broadcast_to(np.asarray(source_c, dtype=np.float64), (len(x),))
        if not np.isfinite(source_c).all():
            raise ValueError("Invalid reference temperature")
        r = self.ridge
        if self.selected["mean_anchor"] == "offset":
            prediction = source_c+r["offset"]
        elif self.selected["mean_anchor"] == "ridge":
            prediction = source_c+((x.astype(np.float64)-r["mean"])/r["std"])@r["coef"]+r["intercept"]
        else:
            raise ValueError("Unknown selected seNorge mean anchor")
        if self.selected["mean_checkpoint"]:
            prediction += self.network(self.selected["mean_checkpoint"], x)
        mse = (self.network(self.selected["error_checkpoint"],
                           np.column_stack([x, prediction]).astype(np.float32))
               if self.selected["error_checkpoint"] else np.full(len(x), self.selected["pooled_mse"]))
        if not np.isfinite(prediction).all() or not np.isfinite(mse).all() or (mse < 0).any():
            raise ValueError("Invalid seNorge temperature/error prediction")
        return prediction, np.sqrt(mse)


class SeNorgeTransferMap:
    def __init__(self, root):
        self.model = SeNorgeTransferModel(root)
        atlas = Path(root) / "outputs" / "senorge_footprint_expanded"
        provenance = self.model.manifest["provenance"]
        for name in ("grid_table.npy", "selected_dates.csv", "cache/sampled_weather.npy"):
            if trainer.sha256(atlas / name) != provenance[name]:
                raise ValueError(f"Changed seNorge inference input: {name}")
        self.table = np.load(atlas / "grid_table.npy", mmap_mode="r", allow_pickle=False)
        self.weather = np.load(atlas / "cache" / "sampled_weather.npy", mmap_mode="r", allow_pickle=False)
        self.plan = pd.read_csv(atlas / "selected_dates.csv")
        if self.weather.shape != (2, len(self.table), len(self.plan)):
            raise ValueError("seNorge historical cache/date shape mismatch")
        self.days = self.plan.date.tolist()
        self.to_xy = Transformer.from_crs(4326, 25833, always_xy=True)
        self.to_lonlat = Transformer.from_crs(25833, 4326, always_xy=True)
        self.tree = cKDTree(self.table[:, :2])
        self.origin = np.array([self.table[:, 0].min(), self.table[:, 1].max()])
        self.ij = np.rint((self.table[:, :2]-self.origin)*[1, -1]/1000).astype(int)[:, ::-1]
        self.shape = tuple((self.ij.max(0)+1).tolist())
        self.context = None
        self.key = 0

    def locate(self, lon, lat):
        if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Invalid map coordinates")
        x, y = self.to_xy.transform(lon, lat)
        _, cell = self.tree.query([x, y])
        if np.max(np.abs(self.table[cell, :2]-[x, y])) > 500:
            raise ValueError("No supported seNorge 1 km land cell at this location")
        return int(cell)

    def reference(self, lon, lat, day, variable):
        if day not in self.days or variable not in (0, 1):
            raise ValueError("Choose a cached seNorge date and tg/tn")
        cell = self.locate(lon, lat)
        temperature = float(self.weather[variable, cell, self.days.index(day)])
        if not np.isfinite(temperature):
            raise ValueError("No archived seNorge value for this reference cell/date/variable")
        lon, lat = self.to_lonlat.transform(*self.table[cell, :2])
        return dict(cell=cell, lon=lon, lat=lat, day=day, variable=variable, archive_c=temperature)

    def prepare(self, lon, lat, day, variable, override=None):
        reference = self.reference(lon, lat, day, variable)
        temperature = reference["archive_c"] if override is None else float(override)
        if not np.isfinite(temperature):
            raise ValueError("Reference temperature must be finite")
        source = reference["cell"]
        targets = np.sort(self.tree.query_ball_point(self.table[source, :2], 46000)).astype(int)
        x = features(self.table, np.full(len(targets), source), targets,
                     np.full(len(targets), temperature), np.repeat(day, len(targets)),
                     np.full(len(targets), variable))
        prediction, rmse = self.model.predict(x, temperature)
        identity = targets == source
        prediction[identity], rmse[identity] = temperature, 0
        observed = np.asarray(self.weather[variable, targets, self.days.index(day)])
        self.key += 1
        self.context = dict(targets=targets, temperature=prediction, rmse=rmse, observed=observed,
            reference=dict(reference, source_c=temperature, manual_override=override is not None), key=self.key)
        self.rasters = {}
        for metric, values in (("temperature_c", prediction), ("rmse_c", rmse)):
            raster = np.full(self.shape, np.nan, np.float32)
            ij = self.ij[targets]
            raster[ij[:, 0], ij[:, 1]] = values
            self.rasters[metric] = raster
        return dict(key=self.key, reference=self.context["reference"], cells=len(targets),
                    radius_km=46, warning=WARNING,
                    temperature_min_c=float(prediction.min()), temperature_max_c=float(prediction.max()),
                    median_predicted_rmse_c=float(np.median(rmse)),
                    pooled_predicted_rmse_c=float(np.sqrt(np.mean(rmse**2))),
                    selected=self.model.selected)

    def detail(self, lon, lat, key):
        self.require(key)
        cell = self.locate(lon, lat)
        slot = np.flatnonzero(self.context["targets"] == cell)
        if not len(slot):
            raise ValueError("Cell is beyond this reference's 46 km prediction area")
        i = int(slot[0])
        c = self.context
        observed = float(c["observed"][i])
        temperature = float(c["temperature"][i])
        return dict(cell=cell, temperature_c=temperature, predicted_rmse_c=float(c["rmse"][i]),
                    archived_target_c=observed if np.isfinite(observed) else None,
                    signed_error_c=temperature-observed if np.isfinite(observed) else None,
                    reference=c["reference"], identity_rule=cell == c["reference"]["cell"])

    def require(self, key):
        if self.context is None or int(key) != self.key:
            raise ValueError("Prediction changed or is not ready; prepare the reference again")

    def tile(self, z, x, y, metric, key, lost=None):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin, from_bounds
        from rasterio.warp import reproject, Resampling
        self.require(key)
        if not (0 <= z <= 19 and 0 <= x < 2**z and 0 <= y < 2**z):
            raise ValueError("Invalid map tile")
        if metric not in self.rasters:
            raise ValueError("Choose temperature_c or rmse_c")
        limit = 20037508.342789244
        size = 2*limit/2**z
        west, north = -limit+x*size, limit-y*size
        output = np.full((256, 256), np.nan, np.float32)
        reproject(self.rasters[metric], output, src_crs="EPSG:25833",
            src_transform=from_origin(self.origin[0]-500, self.origin[1]+500, 1000, 1000),
            src_nodata=np.nan, dst_crs="EPSG:3857",
            dst_transform=from_bounds(west, north-size, west+size, north, 256, 256),
            dst_nodata=np.nan, resampling=Resampling.nearest)
        visible = np.isfinite(output)
        values = np.where(visible, output, 0)
        if metric == "temperature_c":
            amount = np.minimum(1, np.abs(values)/20)
            end = np.where((values < 0)[:, :, None], [33, 102, 172], [178, 24, 43])
            rgb = 247+(end-247)*amount[:, :, None]
        else:
            amount = np.clip(values/5, 0, 1)
            rgb = np.array([255, 247, 236])+(
                np.array([127, 0, 0])-np.array([255, 247, 236]))*amount[:, :, None]
        rgba = np.concatenate([np.rint(rgb).astype(np.uint8),
                               np.where(visible, 200, 0).astype(np.uint8)[:, :, None]], axis=2)
        if lost is not None:
            lost_output = np.zeros((256, 256), np.uint8)
            reproject(lost, lost_output, src_crs="EPSG:25833",
                src_transform=from_origin(self.origin[0]-500, self.origin[1]+500, 1000, 1000),
                src_nodata=0, dst_crs="EPSG:3857",
                dst_transform=from_bounds(west, north-size, west+size, north, 256, 256),
                dst_nodata=0, resampling=Resampling.nearest)
            rgba[lost_output == 1] = [85, 85, 85, 200]
        with MemoryFile() as memory:
            with memory.open(driver="PNG", width=256, height=256, count=4, dtype="uint8") as image:
                image.write(rgba.transpose(2, 0, 1))
            return memory.read()

    def geotiff(self, key):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        self.require(key)
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=self.shape[1], height=self.shape[0], count=2,
                dtype="float32", crs="EPSG:25833", nodata=-9999, compress="deflate", predictor=3,
                transform=from_origin(self.origin[0]-500, self.origin[1]+500, 1000, 1000)) as image:
                for band, metric in enumerate(("temperature_c", "rmse_c"), 1):
                    image.write(np.where(np.isfinite(self.rasters[metric]), self.rasters[metric], -9999), band)
                    image.set_band_description(band, metric)
                image.update_tags(model="senorge-transfer-gpu-v1", warning=WARNING,
                                  reference=json.dumps(self.context["reference"]), radius_km="46")
            return memory.read()


def register_senorge_transfer_map(app, root):
    from flask import Blueprint, jsonify, request, send_file
    from werkzeug.exceptions import HTTPException
    view = Blueprint("senorge_transfer_map", __name__)
    lock = threading.RLock()
    backend = None

    def model():
        nonlocal backend
        if backend is None:
            backend = SeNorgeTransferMap(root)
        return backend

    def inputs():
        return (float(request.args["lon"]), float(request.args["lat"]),
                request.args["day"], int(request.args["variable"]))

    @view.get("/senorge-transfer")
    def page():
        return send_file(Path(root) / "senorge_transfer_map.html")

    @view.get("/senorge_transfer_map.js")
    def script():
        return send_file(Path(root) / "senorge_transfer_map.js", mimetype="application/javascript")

    @view.get("/api/senorge-transfer/options")
    def options():
        with lock:
            current = model()
            return jsonify(days=current.days, warning=WARNING, selected=current.model.selected)

    @view.get("/api/senorge-transfer/reference")
    def reference():
        with lock:
            return jsonify(model().reference(*inputs()))

    @view.post("/api/senorge-transfer/prepare")
    def prepare():
        with lock:
            override = request.args.get("temperature")
            return jsonify(model().prepare(*inputs(), override=None if override is None else float(override)))

    @view.get("/api/senorge-transfer/cell")
    def detail():
        with lock:
            return jsonify(model().detail(float(request.args["lon"]), float(request.args["lat"]),
                                          request.args["key"]))

    @view.get("/api/senorge-transfer/tile/<int:z>/<int:x>/<int:y>.png")
    def tile(z, x, y):
        with lock:
            return send_file(BytesIO(model().tile(z, x, y, request.args.get("metric", "rmse_c"),
                                                  request.args["key"])), mimetype="image/png")

    @view.get("/api/senorge-transfer/export")
    def export():
        with lock:
            return send_file(BytesIO(model().geotiff(request.args["key"])), mimetype="image/tiff",
                             as_attachment=True, download_name="senorge_gpu_virtual_reference_1km.tif")

    @view.errorhandler(ValueError)
    def invalid(error):
        app.logger.warning("Invalid seNorge transfer request: %s", error)
        return jsonify(error=str(error)), 400

    @view.errorhandler(Exception)
    def failed(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        app.logger.exception("seNorge transfer map failed")
        return jsonify(error=str(error)), 500

    app.register_blueprint(view)
