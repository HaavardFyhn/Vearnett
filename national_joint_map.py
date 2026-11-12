"""National1km joint temperature/interval prediction with signed source-removal diagnostics."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import threading
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from national_joint_uncertainty import JointUncertainty
from national_temperature_features import temperature_features
from prepare_national_dataset import POLICY, atomic_npz, identity, selected_temperatures
from remote_senorge.train_national_transfer import fusion_model
from remote_senorge.train_valley_neural import sha256, write_json
from station_temperature_map import TemperatureMap, download_weather
from valley_pilot import HOURS, REGIME_FIELDS, minimax_heights
from prepare_valley_pilot import validate_day


WARNING = (
    "Experimental NATIONAL joint interval prediction,not observed temperature or a forecast. "
    "Native1km land cells,archived2.5km MEPS background,up to six connected observed sources per cell. "
    "Radius is NOT RMSE. Test night coverage87.8%full,73.3%full+deletions,below90%target. "
    "All unobserved cells and nontraining sources are extrapolation. No grid coverage guarantee. "
    "Sea,unsupported terrain and cells without a connected observed source within100km are NoData. "
    "Source removal without replacement may increase OR decrease predicted radius; not causal/additive value."
)


def connections(table, station_table, station_positions, height, valid, origin, progress):
    """Training-equivalent nearest12shortlist,local115km graph,nearest6connected sources."""
    destinations = np.rint((table[:, :2]-origin)*[1, -1]/1000).astype(int)[:, ::-1]
    supported = ((destinations >= 0) & (destinations < height.shape)).all(1)
    supported[supported] &= valid[destinations[supported, 0], destinations[supported, 1]]
    distances, candidates = cKDTree(station_table[:, :2]).query(table[:, :2], k=min(12, len(station_table)))
    if distances.ndim == 1:
        distances, candidates = distances[:, None], candidates[:, None]
    # Stable source ordering breaks exact distance ties deterministically.
    order = np.lexsort((candidates, distances), axis=1)
    distances, candidates = np.take_along_axis(distances, order, 1), np.take_along_axis(candidates, order, 1)
    eligible = (distances <= POLICY["source_radius_m"]) & supported[:, None]
    graph = np.full((*candidates.shape, 5), np.nan, np.float32)
    radius = int((POLICY["source_radius_m"]+POLICY["graph_halo_m"])/1000)
    for source in range(len(station_table)):
        ids, slots = np.nonzero(eligible & (candidates == source))
        if not len(ids):
            continue
        r, c = station_positions[source]
        r0, c0 = max(0, r-radius), max(0, c-radius)
        r1, c1 = min(height.shape[0], r+radius+1), min(height.shape[1], c+radius+1)
        peaks = minimax_heights(height[r0:r1, c0:c1], valid[r0:r1, c0:c1], (r-r0, c-c0))
        ends = destinations[ids]
        peak = peaks[ends[:, 0]-r0, ends[:, 1]-c0]
        keep = np.isfinite(peak)
        ids, slots, ends, peak = ids[keep], slots[keep], ends[keep], peak[keep]
        source_z, target_z = height[r, c], height[ends[:, 0], ends[:, 1]]
        straight, ridge = np.zeros(len(ids), bool), np.zeros(len(ids), np.float32)
        steps = np.max(np.abs(ends-[r, c]), axis=1)+1
        for count in np.unique(steps):
            local = np.flatnonzero(steps == count)
            line = np.rint(np.linspace([r, c], ends[local], int(count))).astype(int)
            ok = valid[line[:, :, 0], line[:, :, 1]].all(0)
            straight[local] = ok
            maximum = height[line[:, :, 0], line[:, :, 1]].max(0)
            ridge[local] = np.where(ok, np.maximum(0, maximum-np.maximum(source_z, target_z[local])), 0)
        graph[ids, slots] = np.column_stack([np.ones(len(ids)), (peak-source_z)/1000,
                           (peak-target_z)/1000, ridge/1000, straight])
        progress(f"National terrain graph: source{source+1}/{len(station_table)}")
    connected = np.isfinite(graph[:, :, 0])
    selected = np.flatnonzero(connected.any(1))
    output_source = np.full((len(selected), 6), -1, np.int32)
    output_graph = np.zeros((len(selected), 6, 5), np.float32)
    rank = np.cumsum(connected[selected], axis=1)-1
    rows, columns = np.nonzero(connected[selected] & (rank < 6))
    output_source[rows, rank[rows, columns]] = candidates[selected[rows], columns]
    output_graph[rows, rank[rows, columns]] = graph[selected[rows], columns]
    return dict(cell=selected, source=output_source, graph=output_graph)


class NationalJointMap:
    def __init__(self, root):
        self.root = Path(root)
        self.base = TemperatureMap(root)
        self.uncertainty = JointUncertainty(root)
        self.folder = self.root / "outputs" / "national_joint_map"
        self.folder.mkdir(exist_ok=True)
        self.result_folder = self.root / "outputs" / "joint_uncertainty_returned"
        self.run = json.loads((self.result_folder / "joint_uncertainty_results" / "run_config.json").read_text())
        self.fingerprint = identity(dict(code=sha256(Path(__file__)), base=self.base.fingerprint,
            joint_report=sha256(self.result_folder / "joint_uncertainty_results" / "report.json"),
            features=sha256(Path(__file__).with_name("national_temperature_features.py"))))
        self.torch = self.base.torch
        with ZipFile(self.root / "outputs" / "joint_uncertainty_training_v1.zip") as archive:
            from io import BytesIO
            spec = json.loads(archive.read("joint_spec.json"))
            for name in ("selected_fusion.pt", "selected_normalization.npz"):
                content = archive.read(name)
                import hashlib
                if hashlib.sha256(content).hexdigest() != self.run["inputs"][name]:
                    raise ValueError("Joint mean model/normalization provenance mismatch")
                if name.endswith(".pt"):
                    weights = self.torch.load(BytesIO(content), map_location="cpu", weights_only=True)
                else:
                    with np.load(BytesIO(content), allow_pickle=False) as norm:
                        self.mean, self.std = norm["mean"].copy(), norm["std"].copy()
            self.model = fusion_model(self.torch, spec["config"]["width"]).eval()
            self.model.load_state_dict(weights)
        self.torch.set_num_threads(4)
        self.context = None

    def observations(self, day, hour):
        stamp = (pd.Timestamp(day, tz="UTC")+pd.Timedelta(hours=hour)).isoformat()
        payloads = []
        for batch in range((len(self.base.stations)+63)//64):
            key = f"{day}_b{batch:03d}"
            path = self.base.folder / "observations" / f"{key}.json"
            if sha256(path) != self.base.observation_receipts[key]:
                raise ValueError("QC observation checksum mismatch")
            payload = json.loads(path.read_text())
            validate_day(payload, pd.Timestamp(day).date(), self.base.stations.iloc[batch*64:(batch+1)*64], "observations")
            payloads.append(payload)
        temperatures = selected_temperatures(payloads, self.base.stations)
        ids = [i for i, row in self.base.stations.iterrows() if (row.station_id, stamp) in temperatures]
        if not ids:
            raise ValueError("No selected-sensor QC readings for this timestamp")
        values = np.array([temperatures[(self.base.stations.iloc[i].station_id, stamp)] for i in ids], np.float32)
        return np.array(ids), values

    def cache(self, name, build):
        path, receipt = self.folder / f"{name}.npz", self.folder / f"{name}.json"
        if receipt.exists():
            if sha256(path) != json.loads(receipt.read_text())["sha256"]:
                raise ValueError("National map cache checksum mismatch")
            with np.load(path, allow_pickle=False) as saved:
                return {key: saved[key].copy() for key in saved.files}
        values = build()
        atomic_npz(path, **values)
        write_json(receipt, dict(sha256=sha256(path), fingerprint=self.fingerprint))
        return values

    def prepare(self, day, hour, progress):
        if day not in self.base.days or hour not in HOURS:
            raise ValueError("Choose an available archived winter day and00/03/06UTC")
        station_ids, temperatures = self.observations(day, hour)
        rows = self.base.stations.iloc[station_ids]
        key = identity(dict(fingerprint=self.fingerprint, day=day, hour=hour,
                            sources=rows.station_id.tolist(), temperatures=temperatures.tolist()))
        receipt_file = self.folder / f"{key}_prediction.json"
        if receipt_file.exists():
            meta = json.loads(receipt_file.read_text())
            path = self.folder / f"{key}_prediction.npz"
            if sha256(path) != meta["sha256"]:
                raise ValueError("National prediction checksum mismatch")
            with np.load(path, allow_pickle=False) as saved:
                self.context = {name: saved[name].copy() for name in saved.files}
            self.context.update(metadata=meta, station_ids=station_ids, temperatures=temperatures)
            return meta
        connection_key = identity(dict(fingerprint=self.fingerprint, sources=rows.station_id.tolist()))
        links = self.cache(connection_key+"_links", lambda: connections(
            self.base.table, self.base.station_table[station_ids], self.base.positions[station_ids],
            self.base.height, self.base.valid, self.base.origin, progress))
        cell = links["cell"]
        if not len(cell):
            raise ValueError("No national terrain cells connected to available sources")
        progress(f"National connected land cells:{len(cell)}; downloading archived background")
        combined = np.vstack([self.base.station_table[station_ids], self.base.table[cell]])
        lon, lat = self.base.to_lonlat.transform(combined[:, 0], combined[:, 1])
        lon[:len(rows)], lat[:len(rows)] = rows.lon, rows.lat
        points = pd.DataFrame(dict(station_id=rows.station_id.tolist()+[f"cell{i}" for i in cell], lon=lon, lat=lat))

        def weather():
            result = download_weather(pd.Timestamp(day).date(), points, progress)
            return dict(fields=np.array([result["fields"][name] for name in REGIME_FIELDS]).transpose(1, 2, 0),
                        projection_xy=np.asarray(result["projection_xy"]), url=np.array(result["url"]),
                        run_time=np.array(result["run_time"]))
        forecast = self.cache(key+"_weather", weather)
        fields = forecast["fields"][HOURS.index(hour)]
        n = len(cell)
        prediction, radius = np.zeros(n, np.float32), np.zeros(n, np.float32)
        removed_temperature, removed_radius = np.full((n, 6), np.nan, np.float32), np.full((n, 6), np.nan, np.float32)
        stamp = pd.Timestamp(day, tz="UTC")+pd.Timedelta(hours=hour)
        for start in range(0, n, 2048):
            ids = np.arange(start, min(n, start+2048))
            sources = links["source"][ids]
            mask = sources >= 0
            x, anchor = np.zeros((*sources.shape, 39), np.float32), np.zeros(sources.shape, np.float32)
            for source in np.unique(sources[mask]):
                rr, ss = np.nonzero(mask & (sources == source))
                features, delta = temperature_features(combined, int(source), len(rows)+ids[rr],
                    links["graph"][ids[rr], ss], temperatures[source], stamp, fields, forecast["projection_xy"])
                x[rr, ss], anchor[rr, ss] = features, temperatures[source]+delta
            normalized = self.torch.from_numpy(((x-self.mean)/self.std).astype(np.float32))
            tensor_anchor, tensor_mask = self.torch.from_numpy(anchor), self.torch.from_numpy(mask)
            with self.torch.inference_mode():
                full = self.model(normalized, tensor_anchor, tensor_mask).numpy()
                radii = self.uncertainty.radii(x, anchor, mask, full)
                prediction[ids], radius[ids] = full, radii
                for slot in range(6):
                    eligible = mask[:, slot] & (mask.sum(1) > 1)
                    if not eligible.any():
                        continue
                    current = mask[eligible].copy()
                    current[:, slot] = False
                    without = self.model(normalized[eligible], tensor_anchor[eligible],
                                         self.torch.from_numpy(current)).numpy()
                    without_radius = self.uncertainty.radii(x[eligible], anchor[eligible], current, without)
                    removed_temperature[ids[eligible], slot] = without-full[eligible]
                    removed_radius[ids[eligible], slot] = without_radius-radii[eligible]
            progress(f"Predicting national joint intervals and removals:{min(start+2048,n)}/{n}cells")
        if not np.isfinite(prediction).all() or not np.isfinite(radius).all() or (radius < 0).any():
            raise ValueError("Invalid national joint prediction")
        valid_deletions=(links["source"]>=0) & ((links["source"]>=0).sum(1)[:,None]>1)
        if (not np.isfinite(removed_temperature[valid_deletions]).all() or
                not np.isfinite(removed_radius[valid_deletions]).all()):
            raise ValueError("Invalid national source-removal predictions")
        arrays = dict(cell=cell, source=links["source"], temperature=prediction, radius=radius,
                      removed_temperature=removed_temperature, removed_radius=removed_radius)
        atomic_npz(self.folder / f"{key}_prediction.npz", **arrays)
        meta = dict(day=day, hour=hour, cells=n, atlas_cells=len(self.base.table),
                    no_prediction_cells=len(self.base.table)-n, sources=len(rows),
                    training_sources=int(rows.group.eq("train").sum()),
                    nontraining_sources=int(rows.group.ne("train").sum()), time=stamp.isoformat(),
                    model="national six-source fusion + verified joint error scale", resolution_m=1000,
                    interval_radius_median_c=float(np.median(radius)), warning=WARNING,
                    uncertainty_map_validated=False, key=key, fingerprint=self.fingerprint,
                    sha256=sha256(self.folder / f"{key}_prediction.npz"),
                    weather_url=str(forecast["url"]), run_time=str(forecast["run_time"]))
        meta["source_selection"]="Nearest12available QC sources within100km,then nearest6terrain-connected. "
        meta["source_selection"]+="Uses available catalogue readings,not just frozen training sources; extrapolation."
        write_json(receipt_file, meta)
        self.context = dict(**arrays, metadata=meta, station_ids=station_ids, temperatures=temperatures)
        return meta

    def geojson(self, bounds=None):
        if self.context is None:
            raise ValueError("Prepare the national prediction first")
        context = self.context
        xy = self.base.table[context["cell"], :2]
        selected = np.arange(len(xy))
        if bounds is not None:
            west, south, east, north = bounds
            if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
                raise ValueError("Invalid viewport bounds")
            lon, lat = self.base.to_lonlat.transform(xy[:, 0], xy[:, 1])
            selected = np.flatnonzero((lon >= west) & (lon <= east) & (lat >= south) & (lat <= north))
        # Exact1km polygons are shown on zoom. National overview uses aggregate tiles,not interpolation.
        aggregate = len(selected) > 6000
        groups = {}
        for i in selected:
            tile = tuple(np.floor(xy[i]/20000).astype(int)) if aggregate else (int(i),)
            groups.setdefault(tile, []).append(int(i))
        features = []
        for tile, ids in groups.items():
            ids = np.asarray(ids)
            if aggregate:
                corner = np.asarray(tile)*20000
                size = 20000
                x0, y0 = corner
            else:
                x0, y0 = xy[ids[0]]-500
                size = 1000
            ring = np.array([[x0,y0], [x0+size,y0], [x0+size,y0+size], [x0,y0+size], [x0,y0]])
            lon, lat = self.base.to_lonlat.transform(ring[:, 0], ring[:, 1])
            features.append(dict(type="Feature", geometry=dict(type="Polygon",
                coordinates=[np.column_stack([lon,lat]).tolist()]), properties=dict(
                temperature_c=float(np.mean(context["temperature"][ids])),
                radius_c=float(np.median(context["radius"][ids])), cells=len(ids),
                cell=None if aggregate else int(context["cell"][ids[0]]),
                overview=aggregate, source_count=float(np.mean((context["source"][ids]>=0).sum(1))))))
        return dict(type="FeatureCollection", features=features, metadata=context["metadata"],
                    overview=aggregate, overview_resolution_m=20000 if aggregate else 1000,
                    explanation="20km display tiles aggregate predicted1km land cells; tile includes unpredicted/sea "
                                "areas and does not imply full tile coverage. Zoom for exact predicted1km cells."
                                if aggregate else "Exact predicted1km cells; all others transparent.")

    def detail(self, cell):
        if self.context is None:
            raise ValueError("Prepare the national prediction first")
        c = self.context
        found = np.flatnonzero(c["cell"] == cell)
        if not len(found):
            raise ValueError("No prediction for this cell")
        i = int(found[0])
        sources = []
        count = int((c["source"][i]>=0).sum())
        for slot, source in enumerate(c["source"][i]):
            if source < 0:
                continue
            station = self.base.stations.iloc[c["station_ids"][source]]
            sources.append(dict(station_id=station.station_id, name=station["name"], lon=station.lon,
                lat=station.lat, source_c=float(c["temperatures"][source]), extrapolation=station.group!="train",
                radius_change_c=None if count==1 else float(c["removed_radius"][i,slot]),
                temperature_change_c=None if count==1 else float(c["removed_temperature"][i,slot]),
                deletion_prediction_available=count>1))
        return dict(cell=cell, temperature_c=float(c["temperature"][i]), radius_c=float(c["radius"][i]),
                    lower_c=float(c["temperature"][i]-c["radius"][i]),
                    upper_c=float(c["temperature"][i]+c["radius"][i]), sources=sources, warning=WARNING)

    def geotiff(self):
        if self.context is None:
            raise ValueError("Prepare the national prediction first")
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        c = self.context
        xy = self.base.table[c["cell"], :2]
        positions = np.rint((xy-self.base.origin)*[1,-1]/1000).astype(int)[:,::-1]
        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=self.base.height.shape[1],height=self.base.height.shape[0],
                count=5,dtype="float32",crs="EPSG:25833",nodata=-9999.,compress="deflate",
                transform=from_origin(self.base.origin[0]-500,self.base.origin[1]+500,1000,1000)) as dataset:
                values=[c["temperature"],c["radius"],c["temperature"]-c["radius"],
                        c["temperature"]+c["radius"],(c["source"]>=0).sum(1)]
                for band, (value,label) in enumerate(zip(values,["Temperature degC","Experimental interval radius degC",
                    "Lower interval degC","Upper interval degC","Connected observed source count"]),1):
                    raster=np.full(self.base.height.shape,-9999.,np.float32)
                    raster[positions[:,0],positions[:,1]]=value
                    dataset.write(raster,band);dataset.set_band_description(band,label)
                dataset.update_tags(**{key:str(value) for key,value in c["metadata"].items()})
            return memory.read()


def register_national_joint_map(app, root):
    from flask import Blueprint, jsonify, request, send_file
    from werkzeug.exceptions import HTTPException
    view = Blueprint("national_joint_map", __name__)
    backend, worker = None, None
    lock = threading.RLock()
    status = dict(state="idle")

    def model():
        nonlocal backend
        with lock:
            if backend is None:
                backend=NationalJointMap(root)
                for path in backend.folder.glob("*_prediction.json"):
                    meta=json.loads(path.read_text())
                    if (meta.get("fingerprint")==backend.fingerprint and meta.get("day")=="2024-12-15" and
                            meta.get("hour")==3):
                        verified=backend.prepare("2024-12-15",3,lambda _:None)
                        status.update(state="ready",day="2024-12-15",hour=3,metadata=verified)
                        break
        return backend

    @view.get("/national-uncertainty")
    def index():
        return send_file(Path(root)/"national_joint_map.html")

    @view.get("/national_joint_map.js")
    def script():
        return send_file(Path(root)/"national_joint_map.js",mimetype="text/javascript")

    @view.get("/api/national-joint/options")
    def options():
        return jsonify(days=model().base.days,warning=WARNING)

    @view.get("/api/national-joint/status")
    def state():
        with lock:
            return jsonify(status)

    @view.post("/api/national-joint/prepare")
    def prepare():
        nonlocal worker
        day,hour=request.args.get("day","2024-12-15"),int(request.args.get("hour","3"))
        with lock:
            if worker is not None and worker.is_alive():
                return jsonify(error="National prediction already running; wait for completion"),409
            current=model()
            if day not in current.base.days or hour not in HOURS:
                raise ValueError("Choose an available archived winter day and00/03/06UTC")
            status.clear();status.update(state="running",day=day,hour=hour,detail="Starting national prediction")
            current.context=None
            def progress(detail):
                with lock:
                    status["detail"]=detail
            def work():
                try:
                    meta=current.prepare(day,hour,progress)
                    with lock:status.update(state="ready",metadata=meta)
                except Exception as error:
                    app.logger.exception("National joint prediction failed")
                    with lock:status.update(state="failed",error=str(error))
            worker=threading.Thread(target=work,daemon=True,name="national-joint-map")
            worker.start()
            return jsonify(status),202

    @view.get("/api/national-joint/grid")
    def grid():
        bounds=request.args.get("bounds")
        return jsonify(model().geojson(None if bounds is None else list(map(float,bounds.split(",")))))

    @view.get("/api/national-joint/detail")
    def detail():
        return jsonify(model().detail(int(request.args["cell"])))

    @view.get("/api/national-joint/export")
    def export():
        return send_file(BytesIO(model().geotiff()),mimetype="image/tiff",as_attachment=True,
                         download_name="experimental_national_joint_1km.tif")

    @view.errorhandler(ValueError)
    def invalid(error):
        app.logger.warning("Invalid national prediction request: %s",error)
        return jsonify(error=str(error)),400

    @view.errorhandler(Exception)
    def failed(error):
        if isinstance(error,HTTPException):return jsonify(error=error.description),error.code
        app.logger.exception("National prediction request failed")
        return jsonify(error=str(error)),500
    app.register_blueprint(view)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day",default="2024-12-15");parser.add_argument("--hour",type=int,default=3)
    args=parser.parse_args()
    backend=NationalJointMap(Path(__file__).resolve().parent)
    print(json.dumps(backend.prepare(args.day,args.hour,lambda text:print(text,flush=True)),indent=2))
