"""Export an offline source bundle and transfer scripts without duplicating large inputs."""
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import tempfile
import zipfile

from remote_senorge.train_remote import sha256


ROOT = Path(__file__).resolve().parent
INPUTS = [
    "frost_history.db",
    "data/Basisdata_0000_Norge_25833_N100Kartdata_FGDB.gdb",
    "data/Basisdata_0000_Norge_25833_NorgesMaritimeGrenser_FGDB.gdb",
    "data/ProssExport.gdb", "data/klimasoner.tif",
    "data/Basisdata_0000_Norge_25833_Fylker_FGDB.gdb",
    "outputs/norway_polygon.gpkg", "outputs/norway_grid_5000m.gpkg",
    "outputs/temperature_5y_prediction_rmse_station_features.gpkg",
    "outputs/senorge_30yr_station_model/xgboost_with_basins.ubj",
    "outputs/senorge_30yr_station_model/metadata.json",
    "outputs/senorge_30yr_station_model/grid_features.csv.gz",
    "outputs/potential_lakes_norway_DTM10_sill20_tiled/potential_lakes.gpkg",
]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "geopandas", "pyogrio",
            "shapely", "pyproj", "rasterio", "xgboost", "Flask", "Pillow", "matplotlib", "requests"]


def source_files(root):
    pending = ["run_500m", "build_portable", "main", "build_trainmap", "precompute_road_grid",
               "portable_launcher"]
    found = set()
    while pending:
        module = pending.pop()
        path = root.joinpath(*module.split(".")).with_suffix(".py")
        if not path.is_file() or path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                pending.append(node.module)
                pending.extend(node.module + "." + alias.name for alias in node.names)
    return sorted(found)


def sanitized_config(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
                ("FROST_CLIENT_ID", "FROST_CLIENT_ID_ENV", "FROST_CLIENT_SECRET") for target in node.targets):
            node.value = ast.Constant(None)
    return ast.unparse(tree) + "\n"


def prepare(output, force=False):
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; use --force to replace it")
    for suffix in ("-wal", "-journal"):
        path = ROOT / ("frost_history.db" + suffix)
        if path.exists() and path.stat().st_size:
            raise ValueError("Stop observation writers and checkpoint SQLite before transferring frost_history.db")
    files = []
    for relative in INPUTS:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        paths = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in paths:
            if not path.is_file() or path.name.lower().endswith(".lock"):
                continue
            before = path.stat()
            print(f"HASH {path.relative_to(ROOT)}", flush=True)
            digest = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Input changed while hashing: {path}")
            files.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=after.st_size, sha256=digest))
    sources = source_files(ROOT)
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        pending = Path(handle.name)
    try:
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                relative = path.relative_to(ROOT).as_posix()
                if path.name == "pipeline_config.py":
                    archive.writestr(relative, sanitized_config(path))
                else:
                    archive.write(path, relative)
            for name in ("station_app.html", "station_app.js"):
                archive.write(ROOT / name, name)
            archive.writestr("remote_500m_inputs.json", json.dumps(dict(files=files), indent=2))
            archive.writestr("requirements-500m.txt", "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n")
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    upload = output.with_name("upload_500m.ps1")
    lines = ["$ErrorActionPreference = 'Stop'",
             "$remote = 'havfyh@its.npra.io'",
             "ssh -p 1959 $remote 'mkdir -p ~/vaernett_500m/data ~/vaernett_500m/outputs/senorge_30yr_station_model ~/vaernett_500m/outputs/potential_lakes_norway_DTM10_sill20_tiled'",
             "if ($LASTEXITCODE -ne 0) { throw 'SSH connection failed' }",
             f"scp -P 1959 '{output.resolve()}' \"${{remote}}:~/vaernett_500m/\"",
             "if ($LASTEXITCODE -ne 0) { throw 'Source upload failed' }"]
    for relative in INPUTS:
        source = ROOT / relative
        parent = Path(relative).parent.as_posix()
        recursive = "-r " if source.is_dir() else ""
        lines.extend([f"scp -p -P 1959 {recursive}'{source}' \"${{remote}}:~/vaernett_500m/{parent}/\"",
                      "if ($LASTEXITCODE -ne 0) { throw 'Input upload failed' }"])
    upload.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"READY {output}; transfer inputs with {upload}", flush=True)
    print(f"Input data: {sum(item['bytes'] for item in files) / 1024**3:.1f} GiB; not duplicated locally", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "remote_500m_code.zip")
    parser.add_argument("--force", action="store_true", help="Atomically replace an existing source bundle")
    args = parser.parse_args(argv)
    prepare(args.output, force=args.force)


if __name__ == "__main__":
    main()
