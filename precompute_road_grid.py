"""Forhåndsberegn eksakte koblinger mellom veglenker og terrengruter."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from make_uncertainty_map_gif import HERE, OUT, load_grid_all_fields
from station_roads import RoadNetwork


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-m", nargs="+", type=int, choices=[5000, 2500, 1000, 500],
                        default=[5000, 2500, 1000])
    parser.add_argument("--distance-m", type=float,
                        help="Største forhåndsberegnede avstand (standard 1000 m, eller 2000 m med --nearest)")
    parser.add_argument("--nearest", action="store_true",
                        help="Lagre bare avstand til nærmeste veg per rute for værkartets ufiltrerte korridor")
    parser.add_argument("--data-root", type=Path, help="Bruk en ferdig portable datamappe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.distance_m is None:
        args.distance_m = 2000 if args.nearest else 1000
    if not 0 <= args.distance_m <= 5000:
        parser.error("Avstanden må være mellom 0 og 5000 meter")
    if args.nearest and args.distance_m == 0:
        parser.error("--nearest krever en avstand større enn 0")
    spacings = sorted(set(args.cell_m), reverse=True)
    root = args.data_root.resolve() if args.data_root else HERE
    out = root / "outputs" if args.data_root else OUT
    manifest_path = out / "portable_weather.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if args.data_root and manifest is None:
        parser.error("--data-root må inneholde outputs/portable_weather.json")
    if manifest is not None and spacings != [manifest["cell_m"]]:
        parser.error("--cell-m må samsvare med datamappens oppløsning")
    source = root / (manifest.get("roads", {}).get("indexed_source", "data/ProssExport.gdb")
                     if manifest is not None else "data/ProssExport.gdb")
    if not source.exists():
        raise FileNotFoundError(source)
    for spacing in spacings:
        path = out / f"norway_grid_{spacing}m.gpkg"
        if not path.exists():
            raise FileNotFoundError(path)
        if args.dry_run:
            print(f"{spacing} m · {'nærmeste veg' if args.nearest else 'alle veglenker'} · avstand 0–{args.distance_m:g} m")
    if args.dry_run:
        return 0
    roads = RoadNetwork(source, cache_dir=out / "station_app_cache" / "roads",
                        portable_cache=spacings == [500])
    for spacing in spacings:
        grid = load_grid_all_fields(out / f"norway_grid_{spacing}m.gpkg")
        engine = SimpleNamespace(grid=grid, cell_m=spacing)

        def progress(completed, total):
            if completed % 32768 == 0 or completed == total:
                print(f"Veg-ruter {spacing} m: {completed}/{total}", flush=True)

        path = (roads.prepare_nearest_distances(engine, args.distance_m, progress) if args.nearest else
                roads.prepare_grid(engine, args.distance_m, progress))
        print(f"KLAR {path}", flush=True)
        if not args.nearest:
            path = roads.prepare_display(engine)
            print(f"KLAR vegvisning {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())