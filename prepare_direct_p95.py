"""Rebuild Direct P95 baselines for a staged Værnett data root."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np

from file_hash import sha256
from greedy_400_stations_pruned import make_cache
from precompute_frost import computation_lock
from station_senorge import PAIR_FEATURES, BasinLookup, SeNorgePairModel
from station_weather import DIRECT_P95, direct_p95_config, exact_network, restricted_profile, save_exact_cache


ROOT = Path(__file__).resolve().parent


def prepare(root, assets, device):
    from station_app import StationEngine
    from station_pross import pross_excluded, svv_met_excluded

    manifest_path = root / "outputs" / "portable_weather.json"
    with computation_lock(manifest_path.parent / "portable"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = direct_p95_config(assets / "outputs" / "senorge_results_10m")
        trained = Path(config["prefix"])
        receipt = json.loads((trained / "results/direct-p95/xgboost_with_basins.json").read_text(encoding="utf-8"))
        training = json.loads((trained / "manifest.json").read_text(encoding="utf-8"))
        model_path = trained / "results/direct-p95/xgboost_with_basins.ubj"
        basin_features = trained / "grid_features.csv.gz"
        if (training["features"] != PAIR_FEATURES or "direct-p95" not in training["targets"] or
                sha256(model_path) != receipt["file_sha256"][model_path.name]):
            raise ValueError("Ugyldige Direct P95-modellfiler")
        old_entry = manifest["profiles"]["temperature:senorge30"]
        basins_path = root / old_entry["basins"]
        token = sha256(model_path)[:12]
        prefix = f"outputs/portable/temperature_direct_p95_{token}"
        entry = dict(profile=f"{prefix}_profile.joblib", svv=f"{prefix}_svv.joblib",
                     svv_met=f"{prefix}_svv_met.joblib", model_kind="senorge", model=f"{prefix}.ubj",
                     basin_features=f"{prefix}_basins.csv.gz", basins=old_entry["basins"])
        engine = StationEngine(root / "outputs" / "unused.npy", cell_m=manifest["cell_m"], data_root=root)
        try:
            profile = joblib.load(root / old_entry["profile"])
            model = SeNorgePairModel(model_path, BasinLookup(basin_features, basins_path))
            if device == "cuda":
                import xgboost as xgb
                booster = model.model.get_booster()
                booster.set_param({"device": "cuda"})
                model.model.predict = lambda values: booster.predict(xgb.DMatrix(values, nthread=4))
            cache_path = root / f"{prefix}_exact.npz"
            stations = profile["stations"]
            if not cache_path.is_file():
                checkpoint = cache_path.with_suffix(".partial.npz")
                baseline, second, winner = exact_network(
                    model, make_cache(stations), engine.cache, False, checkpoint,
                    lambda n, total: print(f"Direct P95 · {engine.cell_m} m · alle: {n}/{total}", flush=True),
                    pair_batch=600_000,
                )
                save_exact_cache(cache_path, baseline, second, winner, len(stations))
                checkpoint.unlink(missing_ok=True)
            with np.load(cache_path, allow_pickle=False) as stored:
                baseline, second, winner = (stored[key].copy() for key in ("baseline", "second", "winner"))
                if int(stored["completed"]) != len(stations) or baseline.shape != (len(engine.grid),):
                    raise ValueError("Direct P95-mellomlageret samsvarer ikke med grunnnettet")
            lo, hi = np.percentile(baseline, [.5, 99.5])
            profile.update({key: value for key, value in config.items() if key != "prefix"})
            profile.update(model=model, cache_path=cache_path, baseline=baseline, second=second, winner=winner,
                           prefix=prefix, minimum=max(float(lo), .0001), frost_maximum=max(float(hi), .000101),
                           status=dict(profile["status"], label="Temperatur · direkte P95"))
            for key, excluded in (
                ("profile", frozenset()), ("svv", pross_excluded(stations)),
                ("svv_met", svv_met_excluded(stations, engine.station_owner_catalog())),
            ):
                if (root / entry[key]).is_file():
                    continue
                ready = restricted_profile(profile, engine.cache, excluded, build=True)
                if ready is None:
                    raise ValueError(f"Ingen kvalifiserte stasjoner i {key}")
                pending = (root / entry[key]).with_suffix(".pending")
                joblib.dump({k: v for k, v in ready.items() if k not in ("model", "cache_path")}, pending, compress=3)
                pending.replace(root / entry[key])
            shutil.copyfile(model_path, root / entry["model"])
            shutil.copyfile(basin_features, root / entry["basin_features"])
            manifest["profiles"][DIRECT_P95] = entry
            manifest["available"] = [item for item in manifest["available"] if item["id"] != DIRECT_P95]
            manifest["available"].append({key: config[key] for key in
                                          ("id", "phenomenon", "kind", "label", "metric", "unit", "element", "model_grid_m")})
            pending_manifest = manifest_path.with_suffix(".pending")
            pending_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            pending_manifest.replace(manifest_path)
            print(f"Direct P95 klar i {root}", flush=True)
        finally:
            engine.resources.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--placement-workers", type=int, default=0,
                        help="After baselines, build all four placement maps with this many workers")
    args = parser.parse_args()
    if args.placement_workers < 0:
        parser.error("--placement-workers must be non-negative")
    os.environ.setdefault("WEATHER_CPU_THREADS", "8")
    root = args.data_root.resolve()
    prepare(root, args.assets.resolve(), args.device)
    if args.placement_workers:
        from station_app import placement_stem
        from station_pross import PLACEMENT_SUFFIXES

        for network, suffix in PLACEMENT_SUFFIXES.items():
            path = root / "outputs" / f"{placement_stem(DIRECT_P95)}{suffix}.npz"
            if path.is_file() and path.with_suffix(".tif").is_file():
                continue
            subprocess.run([sys.executable, "-u", str(ROOT / "placement_gain.py"),
                            "--dataset", DIRECT_P95, "--data-root", str(root),
                            "--station-network", network, "--workers", str(args.placement_workers),
                            "--work-dir", str(root / "outputs" / f"placement_direct_p95_work{suffix}"),
                            "--overwrite"], check=True)
