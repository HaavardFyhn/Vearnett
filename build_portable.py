"""Build a relocatable Windows ZIP with ready 1 km models, not observation history."""
import argparse
import ast
from datetime import datetime
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent


def copy_input(source, destination):
    if not source.exists():
        raise FileNotFoundError(f"Required runtime input is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.lock", "__pycache__"))
    else:
        shutil.copy2(source, destination)


def replaced_road_input(path):
    parts = Path(path).parts
    return (len(parts) >= 2 and parts[:2] in (
        ("data", "ProssExport.gdb"), ("data", "Vegnett_VegScore_full.gpkg"))
        or len(parts) == 4 and parts[:3] == ("outputs", "station_app_cache", "roads")
        and parts[3].lower().endswith(".gpkg"))


def copy_runtime_inputs(data, destination, excluded=()):
    excluded = {Path(path) for path in excluded}
    def exclude_replaced_roads(folder, names):
        return {name for name in names if name == "__pycache__" or name.lower().endswith(".lock") or
                (Path(folder) / name).relative_to(data) in excluded or
                replaced_road_input((Path(folder) / name).relative_to(data))}
    shutil.copytree(data, destination, dirs_exist_ok=True, ignore=exclude_replaced_roads)


def copy_temperature_runtime(data, destination, manifest):
    import joblib
    from station_weather import PHENOMENON_DATASETS, station_metadata

    profiles = manifest["profiles"]
    temperature = PHENOMENON_DATASETS["temperature"]
    if temperature not in profiles:
        temperature = "temperature:senorge30"
    if temperature not in profiles:
        raise ValueError("Portable temperature profile is missing")
    retained = {Path(value) for key, value in profiles[temperature].items()
                if key in ("profile", "svv", "svv_met", "model", "basin_features", "basins")}
    excluded = {Path(value) for identifier, entry in profiles.items() if identifier != temperature
                for key, value in entry.items() if key in ("profile", "svv", "svv_met", "model", "basin_features", "basins")
                and Path(value) not in retained}
    for identifier in manifest.get("station_profiles", {}):
        if identifier != temperature:
            tag = identifier.replace(":", "_")
            excluded.update(Path("outputs") / "portable" / f"{tag}_{kind}.joblib"
                            for kind in ("profile", "svv", "model"))
    copy_runtime_inputs(data, destination, excluded)
    stations = dict(manifest.get("station_profiles", {}))
    for identifier, entry in profiles.items():
        if identifier == temperature or identifier in stations:
            continue
        tag = identifier.replace(":", "_")
        metadata_paths = {}
        for kind in ("profile", "svv"):
            relative = Path("outputs") / "portable" / f"{tag}_stations_{kind}.joblib"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(station_metadata(joblib.load(data / entry[kind])), target, compress=3)
            metadata_paths[kind] = relative.as_posix()
        stations[identifier] = metadata_paths
    result = dict(manifest, profiles={temperature: profiles[temperature]},
                  available=[item for item in manifest["available"] if item["id"] == temperature],
                  station_profiles=stations)
    (destination / "outputs" / "portable_weather.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, excluded


def prepare_roads(destination, cell_m=1000):
    import pyogrio
    from station_roads import ROAD_LAYER, RoadNetwork

    source = ROOT / "data" / "ProssExport.gdb"
    info = pyogrio.read_info(source, layer=ROAD_LAYER)
    if int(info["features"]) <= 0:
        raise ValueError(f"Required road layer is empty: {ROAD_LAYER}")
    source_hash = None
    existing = None
    if cell_m == 500:
        import hashlib
        from file_hash import sha256
        entries = [
            (path.name, sha256(path)) for path in source.iterdir()
            if path.is_file() and not path.name.lower().endswith(".lock")
        ]
        source_hash = hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()
        # Older Windows stages used case-insensitive Path ordering; Linux used name ordering.
        legacy_windows_hash = hashlib.sha256(json.dumps(sorted(entries, key=lambda entry: entry[0].lower())).encode()).hexdigest()
        manifest_path = destination / "outputs" / "portable_weather.json"
        if manifest_path.is_file():
            saved = json.loads(manifest_path.read_text(encoding="utf-8")).get("roads", {})
            if saved.get("indexed_source"):
                if saved.get("source_sha256") not in (source_hash, legacy_windows_hash) or saved.get("layer") != ROAD_LAYER:
                    raise ValueError("Road inputs changed; rebuild the remote 500 m stage before packaging")
                existing = destination / saved["indexed_source"]
                indexed = pyogrio.read_info(existing, layer=ROAD_LAYER)
                if int(indexed["features"]) != int(saved["features"]):
                    raise ValueError("Incomplete indexed road source in the 500 m stage")
    cache = ((destination if existing is not None else ROOT) / "outputs" / "station_app_cache" / "roads")
    roads = RoadNetwork(existing if existing is not None else source, cache_dir=cache, portable_cache=cell_m == 500)
    compact = Path("data") / "Vegnett_VegScore_uten_kommunal.gpkg"
    if roads.path.resolve() != (destination / compact).resolve():
        copy_input(roads.path, destination / compact)
    bundled = pyogrio.read_info(destination / compact, layer=ROAD_LAYER)
    if int(bundled["features"]) != roads.metadata["features"]:
        raise ValueError(f"Incomplete portable road layer: {ROAD_LAYER}")
    result = dict(source=compact.as_posix(), indexed_source=compact.as_posix(), layer=ROAD_LAYER,
                  features=int(bundled["features"]), source_features=int(info["features"]),
                  excluded_road_categories=["K"])
    if cell_m == 500:
        for path in cache.glob("*.npz"):
            target = destination / "outputs" / "station_app_cache" / "roads" / path.name
            if path.resolve() != target.resolve():
                copy_input(path, target)
        result["source_sha256"] = source_hash
    return result


def snapshot(destination, cell_m=1000):
    import station_app
    import joblib
    import pandas as pd
    from greedy_400_stations_pruned import make_cache
    from make_uncertainty_map_gif import load_grid_all_fields
    from station_weather import PHENOMENON_DATASETS, WeatherRegistry

    grid = load_grid_all_fields(ROOT / f"outputs/norway_grid_{cell_m}m.gpkg")
    original = load_grid_all_fields(ROOT / "outputs/norway_grid_5000m.gpkg")
    categories = pd.Index(original.objtype.unique()).union(grid.objtype.unique())
    grid["objtype_code"] = pd.Categorical(grid.objtype, categories=categories).codes
    registry = WeatherRegistry(ROOT, grid, make_cache(grid), categories, cell_m=cell_m)
    available = {config["id"]: config for config in registry.available}
    temperature = PHENOMENON_DATASETS["temperature"]
    if temperature not in available and "temperature:senorge30" in available:
        temperature = "temperature:senorge30"
    manifest = dict(schema=1, cell_m=cell_m, categories=categories.tolist(), available=[], profiles={},
                    station_profiles={}, skipped=[])
    portable = destination / "outputs/portable"
    portable.mkdir(parents=True)
    for identifier in dict(PHENOMENON_DATASETS, temperature=temperature).values():
        if identifier not in available:
            manifest["skipped"].append(dict(id=identifier, reason="No trained model"))
            print(f"SKIP {identifier}: no trained model", flush=True)
            continue
        print(f"SNAPSHOT {identifier}", flush=True)
        config = available[identifier]
        tag = identifier.replace(":", "_")
        if identifier != temperature:
            try:
                metadata = registry.station_metadata(identifier)
            except ValueError as error:
                manifest["skipped"].append(dict(id=identifier, reason=str(error)))
                print(f"SKIP {identifier}: {error}", flush=True)
                continue
            relative = f"outputs/portable/{tag}_stations.joblib"
            joblib.dump(metadata, destination / relative, compress=3)
            manifest["station_profiles"][identifier] = dict(profile=relative, svv=relative)
            registry.station_profiles.clear()
            del metadata
            gc.collect()
            continue
        try:
            profile = registry.profile(identifier)
            if profile.get("baseline") is None:
                raise ValueError("No ready 1 km baseline")
            svv = registry.svv_profile(identifier)
        except ValueError as error:
            manifest["skipped"].append(dict(id=identifier, reason=str(error)))
            print(f"SKIP {identifier}: {error}", flush=True)
            continue
        entry = dict(profile=f"outputs/portable/{tag}_profile.joblib",
                     svv=f"outputs/portable/{tag}_svv.joblib", model=f"outputs/portable/{tag}_model.joblib")
        for key, value in (("profile", profile), ("svv", svv)):
            stored = {name: item for name, item in value.items() if name not in ("model", "cache_path")}
            stored["prefix"] = "outputs/portable/" + tag
            joblib.dump(stored, destination / entry[key], compress=3)
        if profile["kind"] in ("senorge30", "direct_p95"):
            entry.update(model_kind="senorge", model=f"outputs/portable/{tag}.ubj",
                         basin_features="outputs/portable/basin_features.csv.gz",
                         basins="outputs/potential_lakes_norway_DTM10_sill20_tiled/potential_lakes.gpkg")
            prefix = Path(config["prefix"])
            model_path = prefix / "results" / "direct-p95" if profile["kind"] == "direct_p95" else prefix
            copy_input(model_path / "xgboost_with_basins.ubj", destination / entry["model"])
            copy_input(prefix / "grid_features.csv.gz", destination / entry["basin_features"])
        else:
            joblib.dump(profile["model"], destination / entry["model"], compress=3)
        manifest["available"].append({name: config[name] for name in
            ("id", "phenomenon", "kind", "label", "metric", "unit", "element", "model_grid_m") if name in config})
        manifest["profiles"][identifier] = entry
        registry.loaded.clear()
        registry.svv_loaded.clear()
        registry.models.clear()
        del profile, svv
        gc.collect()
    if temperature not in manifest["profiles"]:
        raise RuntimeError(f"Prepare the temperature baseline and SVV reference at {cell_m} m before packaging")
    (destination / "outputs/portable_weather.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for relative in (
        "station_app.html", "station_app.js", f"outputs/norway_grid_{cell_m}m.gpkg", "outputs/norway_polygon.gpkg",
        "data/klimasoner.tif",
        "data/Basisdata_0000_Norge_25833_Fylker_FGDB.gdb",
        "outputs/potential_lakes_norway_DTM10_sill20_tiled/potential_lakes.gpkg",
    ):
        copy_input(ROOT / relative, destination / relative)
    return manifest


def sanitize_config(folder):
    tree = ast.parse((ROOT / "pipeline_config.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
            ("FROST_CLIENT_ID", "FROST_CLIENT_SECRET") for target in node.targets):
            node.value = ast.Constant(None)
    (folder / "pipeline_config.py").write_text(ast.unparse(tree) + "\n", encoding="utf-8")


def copy_licenses(application):
    names = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "xgboost", "geopandas", "shapely",
             "pyogrio", "rasterio", "pyproj", "Flask", "Werkzeug", "Jinja2", "Pillow", "matplotlib",
             "PyInstaller", "threadpoolctl", "packaging", "python-dateutil", "certifi", "click", "blinker",
             "itsdangerous", "MarkupSafe", "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing", "pytz", "tzdata")
    for name in names:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for relative in distribution.files or ():
            if ".dist-info/" in str(relative).replace("\\", "/") and any(
                    token in str(relative).lower() for token in ("license", "copying", "notice")):
                source = Path(distribution.locate_file(relative))
                if source.is_file():
                    copy_input(source, application / "LICENSES" / name / relative.name)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        copy_input(python_license, application / "LICENSES/Python.txt")
    native_licenses(application)


def native_licenses(application):
    sources = {
        "OpenSSL.txt": "https://raw.githubusercontent.com/openssl/openssl/openssl-3.0.15/LICENSE.txt",
        "zlib.txt": "https://raw.githubusercontent.com/madler/zlib/v1.3.1/LICENSE",
    }
    cache = ROOT / "outputs/portable_build/license_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in sources.items():
        path = cache / name
        if not path.is_file():
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        copy_input(path, application / "LICENSES" / name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="Vaernett_Windows_" + datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--stage-only", action="store_true", help="Prepare and validate data without building the executable")
    parser.add_argument("--cell-m", type=int, choices=[500, 1000], default=1000)
    parser.add_argument("--reuse-stage", type=Path, help="Reuse an already prepared snapshot folder")
    parser.add_argument("--reuse-executable", action="store_true", help="Reuse an unchanged executable for a data-only repack")
    args = parser.parse_args(argv)
    if Path(args.name).name != args.name:
        parser.error("--name must be a filename, not a path")
    work = ROOT / "outputs/portable_build" / args.name
    if args.reuse_executable:
        for name in ("station_app.py", "station_scenario.py", "station_roads.py", "station_weather.py",
                     "station_weather_maps.py", "station_placement_tiles.py", "station_snapshot.py", "station_owners.py", "station_coverage.py", "file_hash.py", "portable_launcher.py"):
            previous = work / "sources" / name
            if not previous.is_file() or previous.read_bytes() != (ROOT / name).read_bytes():
                parser.error(f"{name} changed or has no build record; rebuild without --reuse-executable")
    data = args.reuse_stage.resolve() if args.reuse_stage else work / "data"
    if args.reuse_stage:
        manifest = json.loads((data / "outputs/portable_weather.json").read_text(encoding="utf-8"))
        if manifest["cell_m"] != args.cell_m:
            parser.error("--cell-m must match the reused stage")
    else:
        if work.exists():
            raise FileExistsError(f"Build folder already exists: {work}; choose another --name")
        manifest = snapshot(data, cell_m=args.cell_m)
    manifest["roads"] = prepare_roads(data, cell_m=args.cell_m)
    from station_road_background import ROAD_BACKGROUND
    copy_input(ROOT / ROAD_BACKGROUND, data / ROAD_BACKGROUND)
    from station_owners import OWNER_FILE, load_catalog
    owners = load_catalog(ROOT, data)
    if owners is None:
        print("WARNING: Station owners missing; run python station_owners.py before packaging", flush=True)
    else:
        if (ROOT / OWNER_FILE).is_file():
            copy_input(ROOT / OWNER_FILE, data / OWNER_FILE)
        manifest["station_owners"] = dict(path=str(OWNER_FILE), updated=owners["updated"],
                                        stations=len(owners["stations"]))
    from station_coverage import CoverageLayer, RELATIVE
    coverage = CoverageLayer(ROOT, data)
    if coverage.path.is_file():
        target = data / RELATIVE
        target.mkdir(parents=True, exist_ok=True)
        if coverage.path.resolve() != (target / "status.tif").resolve():
            copy_input(coverage.path, target / "status.tif")
        (target / "status_summary.json").write_text(json.dumps(coverage.metadata()), encoding="utf-8")
        manifest["coverage"] = dict(path=str(RELATIVE / "status.tif"))
    else:
        print("WARNING: No saved candidate-status map; candidates will be unavailable", flush=True)
    (data / "outputs/portable_weather.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in ("station_app.py", "station_scenario.py", "station_app.html", "station_app.js"):
        copy_input(ROOT / name, data / name)
    print("INCLUDED " + ", ".join(manifest["profiles"]), flush=True)
    if args.stage_only:
        print(f"STAGE READY {data}", flush=True)
        return 0
    sources = work / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    for source in ROOT.glob("*.py"):
        if not source.name.startswith("test_"):
            shutil.copy2(source, sources / source.name)
    sanitize_config(sources)
    excluded = ["torch", "tensorflow", "arcgis", "arcpy", "IPython", "notebook", "pytest", "tkinter", "xgboost.testing",
                "station_footprint", "station_seasonal", "station_senorge_gpu", "remote_senorge"]
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console",
               "--name", "Vaernett", "--distpath", str(work / "dist"), "--workpath", str(work / "build"),
               "--specpath", str(work), "--collect-all", "pyogrio", "--collect-all", "rasterio",
               "--collect-data", "xgboost", "--collect-binaries", "xgboost",
               "--collect-all", "pyproj", "--hidden-import", "sklearn.ensemble._forest",
               "--hidden-import", "sklearn.tree._tree", "--hidden-import", "xgboost",
               "--hidden-import", "matplotlib.backends.backend_agg"]
    for module in excluded:
        command.extend(["--exclude-module", module])
    command.append(str(sources / "portable_launcher.py"))
    if not args.reuse_executable:
        subprocess.run(command, cwd=sources, check=True)
    application = work / "dist/Vaernett"
    manifest, excluded_weather = copy_temperature_runtime(data, application / "_internal", manifest)
    copy_input(Path(sysconfig.get_paths()["purelib"]) / "xgboost/lib", application / "_internal/xgboost/lib")
    copy_licenses(application)

    (application / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"SMOKE TEST {application / 'Vaernett.exe'}", flush=True)
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "GDAL_DATA", "PROJ_DATA", "PROJ_LIB"):
        environment.pop(name, None)
    windows = Path(environment.get("SystemRoot", r"C:\Windows"))
    environment["PATH"] = str(windows / "System32") + os.pathsep + str(windows)
    subprocess.run([str(application / "Vaernett.exe"), "--smoke-test"], cwd=application,
                   env=environment, check=True)
    archive_path = ROOT / "outputs" / (args.name + ".zip")
    print(f"COMPRESS {archive_path}", flush=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
        for path in sorted(application.rglob("*")):
            relative = path.relative_to(application)
            if path.is_file() and not (relative.parts[0] == "_internal" and
                    (replaced_road_input(Path(*relative.parts[1:])) or Path(*relative.parts[1:]) in excluded_weather)):
                archive.write(path, Path("Vaernett") / path.relative_to(application))
    print(f"READY {archive_path} ({archive_path.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())