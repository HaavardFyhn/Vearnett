"""Nationwide station-removal views from saved, verified no-replacement predictions."""
import copy
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import threading

import numpy as np

from national_joint_map import NationalJointMap


def removal_arrays(context, source):
    matches = context["source"] == source
    used = matches.any(1)
    count = (context["source"] >= 0).sum(1)
    lost = used & (count == 1)
    valid = used & ~lost
    radius_change = np.zeros(len(count), np.float32)
    temperature_change = np.zeros(len(count), np.float32)
    rows, slots = np.nonzero(matches & valid[:, None])
    radius_change[rows] = context["removed_radius"][rows, slots]
    temperature_change[rows] = context["removed_temperature"][rows, slots]
    if not np.isfinite(radius_change[valid]).all() or not np.isfinite(temperature_change[valid]).all():
        raise ValueError("Invalid saved station-removal predictions")
    radius_change[lost] = np.nan
    temperature_change[lost] = np.nan
    return dict(used=used, lost=lost, valid=valid, radius_change=radius_change,
                temperature_change=temperature_change)


def scenario_backend(backend, changes, mode):
    if mode not in ("full", "without", "change"):
        raise ValueError("Choose full, without or change")
    view = copy.copy(backend)
    c = backend.context
    keep = np.ones(len(c["cell"]), bool) if mode == "full" else ~changes["lost"]
    view.context = {key: value for key, value in c.items()}
    for key in ("cell", "source", "temperature", "radius", "removed_temperature", "removed_radius"):
        view.context[key] = c[key][keep].copy()
    if mode == "without":
        view.context["temperature"] += changes["temperature_change"][keep]
        view.context["radius"] += changes["radius_change"][keep]
    elif mode == "change":
        view.context["temperature"] = changes["temperature_change"][keep]
        view.context["radius"] = changes["radius_change"][keep]
    return view


def station_catalogue(backend):
    c = backend.context
    readings = {int(index): float(c["temperatures"][i]) for i, index in enumerate(c["station_ids"])}
    return [dict(station_id=row.station_id, name=row["name"], lat=float(row.lat), lon=float(row.lon),
                 source_c=readings.get(int(index)), available=int(index) in readings,
                 extrapolation=row.group != "train")
            for index, row in backend.base.stations.iterrows()]


def register_station_removal(app, root):
    from flask import Blueprint, jsonify, request, send_file
    from werkzeug.exceptions import HTTPException
    view = Blueprint("national_station_removal", __name__)
    lock = threading.RLock()
    backend = None
    rasters = OrderedDict()

    def model():
        nonlocal backend
        if backend is None:
            backend = NationalJointMap(root)
        day, hour = request.args.get("day", "2024-12-15"), int(request.args.get("hour", "3"))
        if backend.context is None or (backend.context["metadata"]["day"], backend.context["metadata"]["hour"]) != (day, hour):
            # Never start a duplicate national calculation from a display request.
            import json
            completed = any(
                (meta.get("day"), meta.get("hour"), meta.get("fingerprint")) == (day, hour, backend.fingerprint)
                for meta in (json.loads(path.read_text()) for path in backend.folder.glob("*_prediction.json")))
            if not completed:
                raise ValueError("Prepare this national timestamp before displaying station-removal effects")
            backend.prepare(day, hour, lambda _: None)
        return backend

    def selected():
        current = model()
        stations = current.base.stations.iloc[current.context["station_ids"]]
        station = request.args.get("station", "")
        found = np.flatnonzero(stations.station_id.to_numpy() == station)
        if len(found) != 1:
            raise ValueError("Choose a station with a QC reading at this timestamp")
        source = int(found[0])
        changes = removal_arrays(current.context, source)
        return current, stations.iloc[source], source, changes

    def removal_summary(station, changes, mode):
        if mode not in ("full", "without", "change"):
            raise ValueError("Choose full, without or change")
        valid = changes["valid"]
        return dict(station=station.station_id, name=station["name"], lat=station.lat, lon=station.lon,
            mode=mode, affected_cells=int(changes["used"].sum()), lost_cells=int(changes["lost"].sum()),
            unchanged_cells=int((~changes["used"]).sum()),
            positive_radius_cells=int((changes["radius_change"][valid] > 0).sum()),
            negative_radius_cells=int((changes["radius_change"][valid] < 0).sum()),
            radius_change_min_c=float(np.min(changes["radius_change"][valid])) if valid.any() else None,
            radius_change_max_c=float(np.max(changes["radius_change"][valid])) if valid.any() else None,
            explanation="Without minus full; no replacement. Unused station means exactly zero change. "
                        "Grey cells lose their sole source and become NoData. "
                        "Native 1km effects, not causal/additive station importance.")

    @view.get("/api/national-removal/summary")
    def summary():
        with lock:
            if not request.args.get("station", ""):
                return jsonify(metadata=model().context["metadata"], removal=None)
            current, station, _, changes = selected()
            return jsonify(metadata=current.context["metadata"],
                removal=removal_summary(station, changes, request.args.get("mode", "change")))

    @view.get("/api/national-removal/tile/<int:z>/<int:x>/<int:y>.png")
    def tile(z, x, y):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin, from_bounds
        from rasterio.warp import reproject, Resampling
        with lock:
            if z > 19 or x >= 2**z or y >= 2**z:
                raise ValueError("Invalid map tile")
            current = model()
            metric = request.args.get("metric", "radius_c")
            if metric not in ("radius_c", "temperature_c", "source_count"):
                raise ValueError("Invalid tile metric")
            station = request.args.get("station", "")
            mode = request.args.get("mode", "full") if station else "full"
            key = (current.context["metadata"]["key"], station, mode, metric)
            if key not in rasters:
                scenario = current
                lost = None
                if station:
                    current, _, source, changes = selected()
                    scenario = scenario_backend(current, changes, mode)
                    if mode == "without":
                        scenario.context["source"][scenario.context["source"] == source] = -1
                    if mode != "full":
                        lost = changes["lost"]
                if metric == "source_count" and mode == "change":
                    raise ValueError("Source-count change display is unavailable; choose radius or temperature")
                c = scenario.context
                values = (c["radius"] if metric == "radius_c" else c["temperature"]
                          if metric == "temperature_c" else (c["source"] >= 0).sum(1))
                xy = current.base.table[c["cell"], :2]
                ij = np.rint((xy-current.base.origin)*[1, -1]/1000).astype(int)[:, ::-1]
                raster = np.full(current.base.height.shape, np.nan, np.float32)
                raster[ij[:, 0], ij[:, 1]] = values
                loss = np.zeros(current.base.height.shape, np.uint8)
                if lost is not None and lost.any():
                    xy = current.base.table[current.context["cell"][lost], :2]
                    ij = np.rint((xy-current.base.origin)*[1, -1]/1000).astype(int)[:, ::-1]
                    loss[ij[:, 0], ij[:, 1]] = 1
                rasters[key] = (raster, loss)
                while len(rasters) > 6:
                    rasters.popitem(last=False)
            raster, loss = rasters[key]
            limit = 20037508.342789244
            size = 2*limit/(2**z)
            west, north = -limit+x*size, limit-y*size
            destination_transform = from_bounds(west, north-size, west+size, north, 256, 256)
            transform = from_origin(current.base.origin[0]-500, current.base.origin[1]+500, 1000, 1000)
            values = np.full((256, 256), np.nan, np.float32)
            lost_values = np.zeros((256, 256), np.uint8)
            for source_array, output, nodata in ((raster, values, np.nan), (loss, lost_values, 0)):
                reproject(source_array, output, src_transform=transform, src_crs="EPSG:25833",
                    src_nodata=nodata, dst_transform=destination_transform, dst_crs="EPSG:3857",
                    dst_nodata=nodata, resampling=Resampling.nearest)
            visible = np.isfinite(values)
            numeric = np.where(visible, values, 0)
            signed = mode == "change" or metric == "temperature_c"
            if signed:
                amount = np.minimum(1, np.abs(numeric)/(2 if mode == "change" else 20))
                low, high = np.array([33, 102, 172]), np.array([178, 24, 43])
                end = np.where((numeric < 0)[:, :, None], low, high)
                rgb = 247+(end-247)*amount[:, :, None]
            else:
                amount = np.minimum(1, numeric/(6 if metric == "source_count" else 25))
                rgb = np.array([255, 247, 236])+(np.array([127, 0, 0])-np.array([255, 247, 236]))*amount[:, :, None]
            rgb[lost_values == 1] = 85
            rgba = np.concatenate([np.rint(rgb).astype(np.uint8),
                np.where(visible | (lost_values == 1), 200, 0).astype(np.uint8)[:, :, None]], axis=2)
            with MemoryFile() as memory:
                with memory.open(driver="PNG", width=256, height=256, count=4, dtype="uint8") as dataset:
                    dataset.write(rgba.transpose(2, 0, 1))
                return send_file(BytesIO(memory.read()), mimetype="image/png")

    @view.get("/api/national-removal/cell")
    def cell():
        with lock:
            current = model()
            lon, lat = float(request.args["lon"]), float(request.args["lat"])
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError("Invalid cell coordinates")
            from pyproj import Transformer
            x, y = Transformer.from_crs(4326, 25833, always_xy=True).transform(lon, lat)
            table = current.base.table
            distance = np.max(np.abs(table[:, :2]-[x, y]), axis=1)
            found = np.flatnonzero(distance <= 500)
            if not len(found):
                raise ValueError("No supported native1km prediction cell at this location")
            return jsonify(current.detail(int(found[0])))

    @view.get("/api/national-removal/options")
    def options():
        with lock:
            current = model()
            rows = current.base.stations.iloc[current.context["station_ids"]]
            return jsonify(stations=[dict(station_id=row.station_id, name=row["name"],
                lat=row.lat, lon=row.lon, source_c=float(current.context["temperatures"][i]),
                extrapolation=row.group != "train") for i, (_, row) in enumerate(rows.iterrows())],
                catalogue=station_catalogue(current))

    @view.get("/api/national-removal/grid")
    def grid():
        with lock:
            current, station, source, changes = selected()
            mode = request.args.get("mode", "change")
            scenario = scenario_backend(current, changes, mode)
            if mode == "without":
                scenario.context["source"][scenario.context["source"] == source] = -1
            bounds = request.args.get("bounds")
            bounds = None if bounds is None else list(map(float, bounds.split(",")))
            data = scenario.geojson(bounds)
            for feature in data["features"]:
                feature["properties"]["lost_prediction"] = False
            lost = np.flatnonzero(changes["lost"])
            if len(lost) and mode != "full":
                lost_view = copy.copy(current)
                lost_view.context = dict(current.context)
                for key in ("cell", "source", "temperature", "radius", "removed_temperature", "removed_radius"):
                    lost_view.context[key] = current.context[key][lost].copy()
                for feature in lost_view.geojson(bounds)["features"]:
                    feature["properties"].update(lost_prediction=True, radius_c=None, temperature_c=None)
                    data["features"].append(feature)
            data["removal"] = removal_summary(station, changes, mode)
            return jsonify(data)

    @view.get("/api/national-removal/export")
    def export():
        with lock:
            current, station, source, changes = selected()
            # Export the physically meaningful without-station scenario, not delta bounds.
            scenario = scenario_backend(current, changes, "without")
            scenario.context["source"][scenario.context["source"] == source] = -1
            scenario.context["metadata"] = dict(current.context["metadata"], removed_station=station.station_id,
                                               scenario="Without this reading; no replacement")
            return send_file(BytesIO(scenario.geotiff()), mimetype="image/tiff", as_attachment=True,
                             download_name=f"national_without_{station.station_id}_1km.tif")

    @view.errorhandler(ValueError)
    def invalid(error):
        app.logger.warning("Invalid national station-removal request: %s", error)
        return jsonify(error=str(error)), 400

    @view.errorhandler(Exception)
    def failed(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        app.logger.exception("National station-removal display failed")
        return jsonify(error=str(error)), 500

    app.register_blueprint(view)
