"""Compare new-station prediction time and uncertainty with/without a radius cutoff."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radius-km", type=float, default=100)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.radius_km) or args.radius_km <= 0 or args.candidates < 1:
        parser.error("Use a positive radius and candidate count")
    if args.output.exists():
        raise FileExistsError(args.output)
    import station_app
    from station_roads import RoadNetwork

    root = args.data_root.resolve()
    station_app.HERE, station_app.OUT = root, root / "outputs"
    manifest = json.loads((root / "outputs" / "portable_weather.json").read_text(encoding="utf-8"))
    if manifest["cell_m"] != 500:
        raise ValueError("The comparison requires 500 m data")
    engine = station_app.StationEngine(root / "outputs" / "unused.npy", cell_m=500)
    dataset = "temperature:senorge30"
    try:
        engine.ensure_candidates()
        baseline = engine.weather.svv_profile(dataset)["baseline"]
        if not np.isfinite(baseline).all():
            raise ValueError("The SVV baseline must be finite for this comparison")
        roads = RoadNetwork(root / manifest["roads"]["indexed_source"],
                            cache_dir=root / "outputs" / "station_app_cache" / "roads", portable_cache=True)
        mask = roads.national_corridor_mask(engine, 200)
        if not mask.any():
            raise ValueError("No cells in the 200 m road corridor")
        indices = np.linspace(0, len(engine.grid) - 1, min(args.candidates, len(engine.grid)), dtype=int)
        engine.prediction_at(int(indices[0]), np.arange(min(1000, len(engine.grid))), dataset)
        results = []
        for index in indices:
            row = {"candidate": int(index), "x": float(engine.grid.iloc[index].x),
                   "y": float(engine.grid.iloc[index].y), "radius_km": args.radius_km}
            values = []
            for label, radius in (("full", 0), ("cutoff", args.radius_km * 1000)):
                engine.predictions.clear()
                engine.partial_predictions.clear()
                engine.station_radius_m = radius
                started = time.perf_counter()
                prediction = engine.prediction(int(index), dataset)
                row[f"{label}_seconds"] = time.perf_counter() - started
                values.append(np.minimum(baseline, prediction))
                if label == "cutoff":
                    row["predicted_cells"] = int(np.isfinite(prediction).sum())
            delta = values[1] - values[0]
            row.update(speedup=row["full_seconds"] / row["cutoff_seconds"],
                       mean_rmse_increase=float(delta.mean()), max_rmse_increase=float(delta.max()),
                       road_mean_rmse_increase=float(delta[mask].mean()),
                       road_cells_changed_001=int((delta[mask] > .01).sum()))
            results.append(row)
            print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Comparison saved: {args.output}", flush=True)
    finally:
        engine.resources.close()


if __name__ == "__main__":
    main()
