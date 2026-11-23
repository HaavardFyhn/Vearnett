"""Persist fixed Frost weather-map images separately from interactive scenarios."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile

from PIL import Image

from file_hash import code_sha256


DISPLAY_SIZES = (400, 800, 1600)


def colour_reference(weather, dataset, profile):
    if profile["phenomenon"] == "road_temperature":
        from station_weather import PHENOMENON_DATASETS
        dataset = PHENOMENON_DATASETS["temperature"]
        ids = weather.station_dataset_ids() if hasattr(weather, "station_dataset_ids") else ()
        if isinstance(ids, set) and dataset not in ids and "temperature:senorge30" in ids:
            dataset = "temperature:senorge30"
        profile = weather.profile(dataset)
    if profile.get("baseline") is not None:
        profile = weather.svv_profile(dataset)
    return profile


def frost_limits(profile):
    reduction = 1 if profile["kind"] == "class" else {
        "temperature": 2, "wind": 5, "precipitation": 10, "snow": 20, "humidity": 10,
        "road_temperature": 2,
    }[profile["phenomenon"]]
    from station_weather import DIRECT_P95
    if profile.get("id") == DIRECT_P95:
        return 0., 3., reduction
    return profile["minimum"], profile.get("frost_maximum", profile["maximum"]), reduction


class WeatherMapCache:
    def __init__(self, folder, renderer_version=None):
        self.folder = Path(folder)
        self.renderer_version = renderer_version or code_sha256(Path(__file__).with_name("station_app.py"))

    def get(self, engine, profile, size, limits, build=False, numeric=False):
        if profile.get("status", {}).get("state") != "ready" or profile.get("baseline") is None:
            return None
        surface = Path(profile["cache_path"])
        if not surface.is_file():
            return None
        size = 1600 if size is None else size
        if size not in DISPLAY_SIZES:
            raise ValueError("Invalid weather image size")
        stat = surface.stat()
        signature = dict(schema="frost-weather-raster-v1" if numeric else "frost-weather-png-v1", renderer=self.renderer_version,
                         source=str(surface.resolve()), source_bytes=stat.st_size,
                         source_mtime=stat.st_mtime_ns, dataset=profile["id"],
                         cell_m=engine.cell_m, size=size, limits=list(limits))
        if engine.cell_m == 500:
            from file_hash import sha256
            signature.update(source=sha256(surface), source_mtime=None)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        suffix = "bin.gz" if numeric else "png"
        path = self.folder / f"{key}.{suffix}"
        if path.is_file():
            return path.read_bytes()
        if not build:
            return None
        from station_weather import DIRECT_P95
        options = {"map_scale": "direct_p95"} if profile["id"] == DIRECT_P95 else {}
        if numeric:
            from station_placement_tiles import raster_payload
            pixels, values = engine._render_pixels(profile["baseline"], "uncertainty", *limits,
                                                   display_size=size, filter_missing=True, **options)
            image = raster_payload(pixels, values)
        else:
            image = engine.render(profile["baseline"], "uncertainty", *limits, display_size=size, **options).getvalue()
            with Image.open(BytesIO(image)) as check:
                check.verify()
        self.folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.folder, suffix=".pending", delete=False) as handle:
            pending = Path(handle.name)
            handle.write(image)
        try:
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        return image