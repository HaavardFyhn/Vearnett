"""Native1km experimental reference-conditioned footprint maps for Station Lab."""
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path
import threading

import numpy as np

from remote_senorge.footprint_atlas import AtlasDataset, SceneArray
from remote_senorge.footprint_model import TARGETS, create_model
from remote_senorge.train_remote import sha256
from station_senorge import BasinLookup


TARGET_LABELS = dict(zip(TARGETS, (
    "År · døgnmiddel", "År · døgnminimum", "Vinter · døgnmiddel",
    "Vinter · døgnminimum", "Sommer · døgnmiddel", "Sommer · døgnminimum")))
WARNING = (
    "Eksperimentell footprint-CNN: historisk seNorge-representasjonsfeil 1996–2025, "
    "ikke temperatur eller kalibrert usikkerhet mot uavhengige stasjoner. "
    "64 × 64 km kart ved 1 km; ingen prediksjon utenfor footprint eller utenfor seNorge-landruter. "
    "Referansen knyttes til nærmeste 1 km-rutesenter; samme rute får 0 som identitetsregel. "
    "Kartet er retningsavhengig og gjelder én referanse, ikke samlet assimilering av stasjonsnettet. "
    "Testen dekker fire regioner med overlappende kart; tidligere forsøk har brukt samme geografi. "
    "Bedre enn XGBoost i sommertesten, men ikke i års-/vintertesten; vinterminimum har "
    "positiv bias ca. 0,24 °C. Gjennomsnittsresultater beviser ikke riktig oppførsel mellom nabodaler."
)


class FootprintBackend:
    def __init__(self, root, folder=None, atlas=None):
        import torch
        self.root = Path(root)
        self.folder = Path(folder) if folder is not None else self.root / "outputs" / "senorge_footprint_station_model"
        self.metadata = json.loads((self.folder / "metadata.json").read_text())
        for name, digest in self.metadata["checksums"].items():
            if Path(name).name != name or sha256(self.folder / name) != digest:
                raise ValueError(f"Footprint app checksum mismatch: {name}")
        if self.metadata["architecture"] != "reference-resunet-spatial10-absolute-v2":
            raise ValueError("Unsupported footprint app model")
        prepared = self.root / "outputs" / "senorge_footprint_expanded"
        if sha256(prepared / "footprint_manifest.json") != self.metadata["manifest_sha256"]:
            raise ValueError("Local footprint dataset differs from returned model")
        self.owns_atlas = atlas is None
        self.atlas = AtlasDataset(prepared) if atlas is None else atlas
        try:
            if self.atlas.feature_count != 31:
                raise ValueError("Footprint app needs31 endpoint inputs")
            self.torch = torch
            torch.set_num_threads(4)
            self.lock = threading.RLock()
            selected = self.metadata["selected"]
            self.model = create_model(torch, selected["width"], True, 64, 11, 31)
            self.model.load_state_dict(torch.load(self.folder / "selected_model.pt",
                                                 map_location="cpu", weights_only=True))
            self.model.eval()
            with np.load(self.folder / "normalization.npz", allow_pickle=False) as norm:
                self.mean, self.std = norm["mean"].copy(), norm["std"].copy()
            if (self.mean.shape != (31,) or self.std.shape != (31,) or
                    not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any()):
                raise ValueError("Invalid footprint app normalization")
            self.keys = {tuple(key): i for i, key in enumerate(
                BasinLookup.cell_keys(self.atlas.table[:, 0], self.atlas.table[:, 1]))}
            self.maps = OrderedDict()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.owns_atlas:
            self.atlas.close()

    def reference_at(self, x, y):
        if not np.isfinite([x, y]).all():
            raise ValueError("Ugyldige referansekoordinater")
        key = tuple(BasinLookup.cell_keys([x], [y])[0])
        ref = self.keys.get(key)
        if ref is None:
            raise ValueError("Referansen ligger utenfor seNorge-landrutenettet; velg en annen 1 km-rute")
        return ref

    def mapping(self, ref):
        x, y = BasinLookup.cell_keys(self.atlas.table[ref:ref+1, 0], self.atlas.table[ref:ref+1, 1])[0]
        return np.array([[self.keys.get((x+c-32, y+32-r), -1) for c in range(64)]
                         for r in range(64)], np.int32)

    def predict_reference(self, ref):
        if not 0 <= ref < len(self.atlas.table):
            raise ValueError("Invalid footprint reference index")
        with self.lock:
            if ref in self.maps:
                self.maps.move_to_end(ref)
                return self.maps[ref]
            mapping = self.mapping(ref)
            scene = SceneArray(self.atlas, "scenes").for_reference(ref)
            local = SceneArray(self.atlas, "local10").for_reference(ref)
            features = SceneArray(self.atlas, "features").for_reference(ref, mapping)
            features = (features-self.mean[:, None, None])/self.std[:, None, None]
            if not all(np.isfinite(value).all() for value in (scene, local, features)):
                raise ValueError("Invalid footprint inference inputs")
            with self.torch.inference_mode():
                values = self.model(*[self.torch.from_numpy(value[None]) for value in
                                      (scene, local, features)]).numpy()[0]
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid footprint model predictions")
            values[:, 32, 32] = 0
            values.setflags(write=False)
            mapping.setflags(write=False)
            self.maps[ref] = values, mapping
            while len(self.maps) > 12:
                self.maps.popitem(last=False)
            return values, mapping

    def describe(self, ref, target):
        from pyproj import Transformer
        if target not in TARGETS:
            raise ValueError("Ugyldig footprint-mål")
        values, mapping = self.predict_reference(ref)
        data = values[TARGETS.index(target)]
        mask = mapping >= 0
        x, y = self.atlas.table[ref, :2]
        left, top = x-32500, y+32500
        to_geo = Transformer.from_crs(25833, 4326, always_xy=True)
        from rasterio.warp import transform_bounds
        bounds = transform_bounds(25833, 3857, left, top-64000, left+64000, top, densify_pts=21)
        west, south = Transformer.from_crs(3857, 4326, always_xy=True).transform(bounds[0], bounds[1])
        east, north = Transformer.from_crs(3857, 4326, always_xy=True).transform(bounds[2], bounds[3])
        lon, lat = to_geo.transform(x, y)
        subset = data[mask]
        return dict(reference=ref, target=target, label=TARGET_LABELS[target], cell_m=1000,
                    shape=[64, 64], context_km=80, reference_pixel=[32, 32],
                    x=float(x), y=float(y), lon=lon, lat=lat,
                    elevation_m=float(self.atlas.table[ref, 2]),
                    coast_ocean_distance_m=float(self.atlas.table[ref, 4]),
                    coast_fjord_distance_m=float(self.atlas.table[ref, 3]),
                    bounds=[[south, west], [north, east]], web_bounds=list(bounds),
                    projected_bounds=[left, top-64000, left+64000, top],
                    valid_cells=int(mask.sum()), minimum_c=float(subset.min()), maximum_c=float(subset.max()),
                    mean_c=float(subset.mean()), warning=WARNING, unit="°C", metric="RMSE",
                    model_sha256=self.metadata["checksums"]["selected_model.pt"],
                    dataset_sha256=self.metadata["manifest_sha256"], selected=self.metadata["selected"]["name"],
                    test=self.metadata["test"][target], baseline_test=self.metadata["baseline_test"][target])

    def point(self, ref, target, x, y):
        if target not in TARGETS or not np.isfinite([x, y]).all():
            raise ValueError("Ugyldig footprint-punkt eller mål")
        cx, cy = self.atlas.table[ref, :2]
        key = BasinLookup.cell_keys([x], [y])[0]*1000
        row, col = int(round((cy-key[1])/1000))+32, int(round((key[0]-cx)/1000))+32
        values, mapping = self.predict_reference(ref)
        supported = 0 <= row < 64 and 0 <= col < 64 and mapping[row, col] >= 0
        return dict(supported=bool(supported), value_c=float(values[TARGETS.index(target), row, col])
                    if supported else None, x=int(key[0]), y=int(key[1]), cell_m=1000,
                    reason=None if supported else "Ingen prediksjon utenfor footprint eller seNorge-landruter")

    def raster(self, ref, target):
        from rasterio.transform import from_origin
        description = self.describe(ref, target)
        values, mapping = self.predict_reference(ref)
        raster = np.where(mapping >= 0, values[TARGETS.index(target)], -9999).astype(np.float32)
        x, y = self.atlas.table[ref, :2]
        return raster, from_origin(x-32500, y+32500, 1000, 1000), description

    def png(self, ref, target):
        raster, transform, description = self.raster(ref, target)
        return raster_png(raster, transform, description)

    def export(self, ref, target):
        raster, transform, description = self.raster(ref, target)
        return raster_export({f"footprint_{target}_1000m.tif": raster}, transform, description)


def raster_png(raster, transform, description, maximum=3.):
    import matplotlib
    from PIL import Image
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    west, south, east, north = description["web_bounds"]
    # Keep native-nearest values; cap national display dimensions, not prediction resolution.
    pixel = max(750., (east-west)/2200, (north-south)/2200)
    width, height = int(np.ceil((east-west)/pixel)), int(np.ceil((north-south)/pixel))
    web = np.full((height, width), -9999, np.float32)
    reproject(raster, web, src_transform=transform, src_crs=25833,
              dst_transform=from_bounds(west, south, east, north, width, height), dst_crs=3857,
              src_nodata=-9999, dst_nodata=-9999, resampling=Resampling.nearest)
    colors = (matplotlib.colormaps["RdBu_r"](np.clip(np.maximum(web, 0)/maximum, 0, 1))*255).astype(np.uint8)
    colors[:, :, 3] = np.where(web == -9999, 0, 220)
    buffer = BytesIO()
    Image.fromarray(colors).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def raster_export(rasters, transform, description):
    from rasterio.io import MemoryFile
    from zipfile import ZipFile
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, raster in rasters.items():
            with MemoryFile() as memory:
                with memory.open(driver="GTiff", width=raster.shape[1], height=raster.shape[0],
                                 count=1, dtype="float32", crs=25833, transform=transform,
                                 nodata=-9999, compress="deflate") as output:
                    output.write(raster.astype(np.float32), 1)
                    output.update_tags(target=description["target"], cell_m=1000,
                                       model_sha256=description["model_sha256"],
                                       interpretation=description.get("interpretation",
                                           "Historical seNorge representation RMSE, not temperature"))
                archive.writestr(name, memory.read())
        archive.writestr("footprint.json", json.dumps(description, indent=2, ensure_ascii=False))
    buffer.seek(0)
    return buffer


def frost_catalogue(root, model):
    import pyogrio
    from station_weather import observation_stations
    root = Path(root)
    features = pyogrio.read_dataframe(
        root / "outputs" / "temperature_5y_prediction_rmse_station_features.gpkg",
        columns=["station_id", "x", "y"], read_geometry=False).drop_duplicates("station_id")
    observed, _, _ = observation_stations(
        root / "frost_history.db", root / "frost_db_report" / "station_element_all.csv",
        "air_temperature", station_ids=features.station_id)
    frame = observed[["station_id", "name", "lat", "lon"]].merge(features, on="station_id").dropna()
    result = []
    for record in frame.to_dict("records"):
        key = tuple(BasinLookup.cell_keys([record["x"]], [record["y"]])[0])
        ref = model.keys.get(key)
        record.update(supported=ref is not None, reference=ref)
        result.append(record)
    return sorted(result, key=lambda item: item["station_id"])


def register_footprint_view(app, root, backend=None, stations=None, standalone=False):
    from flask import Blueprint, jsonify, request, send_file
    from pyproj import Transformer
    from station_valley_pilot import register_valley_view
    register_valley_view(app, root)
    from station_valley_sandbox import register_sandbox_view
    register_sandbox_view(app, root)
    from station_temperature_map import register_temperature_map
    register_temperature_map(app, root)
    view = Blueprint("footprint", __name__)
    root = Path(root)
    lock = threading.RLock()
    catalogue = stations
    network = None
    preparation = None
    to_xy = Transformer.from_crs(4326, 25833, always_xy=True)

    def get_backend():
        nonlocal backend
        with lock:
            if backend is None:
                backend = FootprintBackend(root)
            return backend

    def get_stations():
        nonlocal catalogue
        with lock:
            if catalogue is None:
                catalogue = frost_catalogue(root, get_backend())
            return catalogue

    def target_arg():
        target = request.args.get("target", "winter_tn")
        if target not in TARGETS:
            raise ValueError("Velg et gyldig års-/sesongmål")
        if request.args.get("cell_m", "1000") != "1000":
            raise ValueError("Footprint-visningen støtter bare native1 km-ruter")
        return target

    def get_network():
        nonlocal network
        from station_footprint_network import FootprintNetwork
        with lock:
            if network is None:
                network = FootprintNetwork(get_backend(), get_stations())
            return network

    def network_args():
        target = target_arg()
        mode = request.args.get("mode", "network")
        station = request.args.get("station")
        if mode not in ("network", "reduction", "lost"):
            raise ValueError("Ugyldig nettkart")
        if mode != "network":
            get_network().station_index(station)
        return target, mode, station

    def reference_arg():
        model = get_backend()
        identifier = request.args.get("station")
        if identifier:
            record = next((item for item in get_stations() if item["station_id"] == identifier), None)
            if record is None:
                raise ValueError("Ukjent eller utilgjengelig Frost-stasjon")
            if not record["supported"]:
                raise ValueError("Stasjonens nærmeste1 km-rute finnes ikke i seNorge-landrutenettet")
            return int(record["reference"])
        if "reference" in request.args:
            ref = int(request.args["reference"])
            if not 0 <= ref < len(model.atlas.table):
                raise ValueError("Ugyldig footprint-referanse")
            return ref
        lon, lat = float(request.args["lon"]), float(request.args["lat"])
        if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Ugyldig lengde-/breddegrad")
        return model.reference_at(*to_xy.transform(lon, lat))

    @view.get("/footprint")
    def index():
        return send_file(root / "station_footprint.html")

    @view.get("/station_footprint.js")
    def javascript():
        return send_file(root / "station_footprint.js", mimetype="text/javascript")

    @view.get("/api/footprint/options")
    def options():
        model = get_backend()
        return jsonify(targets=[dict(id=key, label=label) for key, label in TARGET_LABELS.items()],
                       cell_m=1000, standalone=standalone, warning=WARNING, stations=get_stations(),
                       selected=model.metadata["selected"]["name"], test=model.metadata["test"],
                       baseline_test=model.metadata["baseline_test"])

    @view.get("/api/footprint/available")
    def available():
        folder = root / "outputs" / "senorge_footprint_station_model"
        return jsonify(available=all((folder / name).is_file() for name in
                                     ("metadata.json", "selected_model.pt", "normalization.npz")),
                       cell_m=1000)

    @view.get("/api/footprint/network/status")
    def network_status():
        return jsonify(get_network().status())

    @view.post("/api/footprint/network/prepare")
    def network_prepare():
        nonlocal preparation
        with lock:
            net = get_network()
            if net.status()["state"] == "ready":
                return jsonify(net.status())
            if preparation is None or not preparation.is_alive():
                def work():
                    try:
                        net.prepare()
                    except Exception:
                        app.logger.exception("Frost footprint preparation failed; see status")
                preparation = threading.Thread(target=work, name="frost-footprint-preparation", daemon=True)
                preparation.start()
            return jsonify(state="running", stations=len(net.stations)), 202

    @view.get("/api/footprint/network")
    def network_metadata():
        return jsonify(get_network().describe(*network_args()))

    @view.get("/api/footprint/network.png")
    def network_image():
        response = send_file(get_network().png(*network_args()), mimetype="image/png")
        response.headers["X-Cell-Size-M"] = "1000"
        response.headers["Cache-Control"] = "no-store"
        return response

    @view.get("/api/footprint/network/point")
    def network_point():
        target = target_arg()
        lon, lat = float(request.args["point_lon"]), float(request.args["point_lat"])
        if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Ugyldig kartpunkt")
        return jsonify(get_network().point(target, *to_xy.transform(lon, lat), request.args.get("station")))

    @view.get("/api/footprint/network/export")
    def network_export():
        target, mode, station = network_args()
        return send_file(get_network().export(target, mode, station), mimetype="application/zip",
                         as_attachment=True, download_name=f"frost_{mode}_{target}_1000m.zip")

    @view.get("/api/footprint")
    def metadata():
        target = target_arg()
        return jsonify(get_backend().describe(reference_arg(), target))

    @view.get("/api/footprint.png")
    def image():
        target = target_arg()
        response = send_file(get_backend().png(reference_arg(), target), mimetype="image/png")
        response.headers["X-Cell-Size-M"] = "1000"
        response.headers["Cache-Control"] = "no-store"
        return response

    @view.get("/api/footprint/point")
    def point():
        target = target_arg()
        ref = reference_arg()
        lon, lat = float(request.args["point_lon"]), float(request.args["point_lat"])
        if not np.isfinite([lon, lat]).all() or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Ugyldig kartpunkt")
        return jsonify(get_backend().point(ref, target, *to_xy.transform(lon, lat)))

    @view.get("/api/footprint/export")
    def export():
        target = target_arg()
        ref = reference_arg()
        return send_file(get_backend().export(ref, target), mimetype="application/zip",
                         as_attachment=True, download_name=f"footprint_{target}_{ref}_1000m.zip")

    @view.errorhandler(ValueError)
    @view.errorhandler(KeyError)
    @view.errorhandler(TypeError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @view.errorhandler(OSError)
    def failed(error):
        app.logger.exception("Footprint model/data I/O error")
        return jsonify(error=str(error)), 503

    app.register_blueprint(view)
    return get_backend
