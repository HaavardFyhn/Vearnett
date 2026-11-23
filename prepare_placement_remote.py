"""Create a code-only bundle for the 500 m remote placement calculation."""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile

from prepare_remote_500m import ROOT, sanitized_config, source_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "placement_gain_code.zip")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-owners", action="store_true",
                        help="Include the owner catalog required by --station-network svv-met")
    parser.add_argument("--direct-p95-stage", type=Path,
                        help="Include the Direct P95 inputs for the existing 500 m SSH staging root")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} finnes allerede; bruk --force for å erstatte den")
    sources = set(source_files(ROOT))
    sources.add(ROOT / "placement_gain.py")
    sources.add(ROOT / "compare_station_radius.py")
    sources.add(ROOT / "run_placement_empty_all.sh")
    if args.include_owners:
        from station_owners import OWNER_FILE, load_catalog
        catalog = load_catalog(ROOT)
        if catalog is None or not any("MET.NO" in station["owners"] for station in catalog["stations"].values()):
            raise ValueError("MET.NO-eierdata mangler; kjør python station_owners.py før pakking")
        sources.add(ROOT / OWNER_FILE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".zip", delete=False) as handle:
        pending = Path(handle.name)
    try:
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(sources):
                name = path.relative_to(ROOT).as_posix()
                if path.name == "pipeline_config.py":
                    archive.writestr(name, sanitized_config(path))
                elif path.suffix == ".sh":
                    archive.writestr(name, path.read_text(encoding="utf-8").replace("\r\n", "\n"))
                else:
                    archive.write(path, name)
            if args.direct_p95_stage is not None:
                from file_hash import sha256

                stage = args.direct_p95_stage
                manifest = json.loads((stage / "outputs" / "portable_weather.json").read_text(encoding="utf-8"))
                identifier = "temperature:direct_p95"
                entry = manifest["profiles"][identifier]
                bundled = [entry[key] for key in ("profile", "svv", "svv_met", "model", "basin_features")]
                bundled.append(manifest["roads"]["indexed_source"])
                shared = ["outputs/norway_grid_500m.gpkg", "outputs/norway_polygon.gpkg", entry["basins"]]
                if manifest["cell_m"] != 500:
                    raise ValueError("Expected the 500 m staging root")
                for path in bundled:
                    archive.write(stage / path, f"stage/{path}")
                manifest = dict(manifest, profiles={identifier: entry}, station_profiles={},
                                available=[config for config in manifest["available"] if config["id"] == identifier])
                archive.writestr("stage/outputs/portable_weather.json", json.dumps(manifest, indent=2))
                archive.writestr("shared_inputs.json", json.dumps({path: sha256(stage / path) for path in shared}, indent=2))
                archive.write(ROOT / "run_direct_p95_placement.py", "run_direct_p95_placement.py")
        pending.replace(args.output)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Code bundle: {args.output}")


if __name__ == "__main__":
    main()
