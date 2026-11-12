"""Forhåndsberegn eksakte koblinger mellom veglenker og terrengruter."""
import argparse
from types import SimpleNamespace

from make_uncertainty_map_gif import HERE, OUT, load_grid_all_fields
from station_roads import RoadNetwork


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-m", nargs="+", type=int, choices=[5000, 2500, 1000, 500],
                        default=[5000, 2500, 1000])
    parser.add_argument("--distance-m", type=float, default=1000,
                        help="Største forhåndsberegnede korridoravstand; standard 1000 m")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.distance_m <= 5000:
        parser.error("Avstanden må være mellom 0 og 5000 meter")
    spacings = sorted(set(args.cell_m), reverse=True)
    source = HERE / "data" / "ProssExport.gdb"
    if not source.exists():
        raise FileNotFoundError(source)
    for spacing in spacings:
        path = OUT / f"norway_grid_{spacing}m.gpkg"
        if not path.exists():
            raise FileNotFoundError(path)
        if args.dry_run:
            print(f"{spacing} m · alle vegklasser/kategorier · avstand 0–{args.distance_m:g} m")
    if args.dry_run:
        return 0
    roads = RoadNetwork(source, cache_dir=OUT / "station_app_cache" / "roads",
                        portable_cache=args.cell_m == [500])
    for spacing in spacings:
        grid = load_grid_all_fields(OUT / f"norway_grid_{spacing}m.gpkg")
        engine = SimpleNamespace(grid=grid, cell_m=spacing)

        def progress(completed, total):
            if completed % 32768 == 0 or completed == total:
                print(f"Veg-ruter {spacing} m: {completed}/{total}", flush=True)

        path = roads.prepare_grid(engine, args.distance_m, progress)
        print(f"KLAR {path}", flush=True)
        path = roads.prepare_display(engine)
        print(f"KLAR vegvisning {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())