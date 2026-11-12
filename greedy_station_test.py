"""Benchmark greedy design of a new weather-station network.

Uses an existing ExtraTrees joblib model; it does not read observations or retrain.
The first station minimizes total predicted error. Every later station maximizes
reduction in the current nearest-station error.

Examples:
  python greedy_station_test.py --random-candidates 1000 --random-targets 3000 --iterations 10
  python greedy_station_test.py --random-candidates 3000 --random-targets 5000 --iterations 25 --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np

from pipeline_config import WORK, RANDOM_STATE, PAIR_FEATURES

TERRAIN = [
    "relief_10km", "rugstd_10km", "tpi_10km",
    "relief_25km", "rugstd_25km", "tpi_25km",
]
EXPECTED_FEATURES = [
    "distance_m", "elev_diff_m", "coast_fjord_diff_m",
    "coast_ocean_diff_m", "same_objtype", "x_diff", "y_diff",
    "relief_diff_10km", "rugstd_diff_10km", "tpi_diff_10km",
    "relief_diff_25km", "rugstd_diff_25km", "tpi_diff_25km",
]


def parse_args():
    p = argparse.ArgumentParser(description="Time a greedy zero-to-N station design")
    p.add_argument("--model", type=Path, default=WORK / "temperature_5y_prediction_rmse_model.joblib")
    p.add_argument("--grid-file", type=Path, default=WORK / "norway_grid_5000m.gpkg")
    p.add_argument("--random-candidates", type=int, default=1000)
    p.add_argument("--random-targets", type=int, default=3000)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--full-stations", type=int, default=400)
    p.add_argument("--candidate-batch", type=int, default=50)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--output", type=Path, default=WORK / "greedy_station_test.gpkg")
    p.add_argument("--checkpoint", action="store_true", help="Write output after every selected station")
    return p.parse_args()


def load_grid(path):
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError("Grid CRS is missing")
    if g.crs.to_epsg() != 25833:
        g = g.to_crs(25833)
    required = ["elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype", *TERRAIN, "geometry"]
    missing = sorted(set(required) - set(g.columns))
    if missing:
        raise ValueError(f"Grid missing fields: {missing}")
    g = g.dropna(subset=required).copy().reset_index(drop=True)
    g["x"] = g.geometry.x.astype(np.float32)
    g["y"] = g.geometry.y.astype(np.float32)
    g["source_grid_id"] = np.arange(len(g), dtype=np.int32)
    # One shared category mapping is essential; do not encode samples independently.
    g["objtype_code"] = g["objtype"].astype("category").cat.codes.astype(np.int16)
    return g


def cache(frame):
    keys = ["x", "y", "elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype_code", *TERRAIN]
    result = {}
    for key in keys:
        dtype = np.int16 if key == "objtype_code" else np.float32
        result[key] = frame[key].to_numpy(dtype=dtype)
    return result


def feature_matrix(src, target, target_ids):
    sx, sy = src["x"][:, None], src["y"][:, None]
    tx, ty = target["x"][target_ids][None, :], target["y"][target_ids][None, :]
    columns = [
        np.hypot(sx - tx, sy - ty).ravel(),
        np.abs(src["elevation"][:, None] - target["elevation"][target_ids][None, :]).ravel(),
        np.abs(src["coast_dist_fjord"][:, None] - target["coast_dist_fjord"][target_ids][None, :]).ravel(),
        np.abs(src["coast_dist_ocean"][:, None] - target["coast_dist_ocean"][target_ids][None, :]).ravel(),
        (src["objtype_code"][:, None] == target["objtype_code"][target_ids][None, :]).astype(np.float32).ravel(),
        np.abs(sx - tx).ravel(),
        np.abs(sy - ty).ravel(),
    ]
    for key in TERRAIN:
        columns.append(np.abs(src[key][:, None] - target[key][target_ids][None, :]).ravel())
    X = np.column_stack(columns).astype(np.float32, copy=False)
    if X.shape[1] != len(EXPECTED_FEATURES):
        raise ValueError(f"Generated {X.shape[1]} features, expected {len(EXPECTED_FEATURES)}")
    return X


def subset_cache(values, ids):
    return {key: value[ids] for key, value in values.items()}


def predict_batch(model, candidate_cache, target_cache, target_ids):
    X = feature_matrix(candidate_cache, target_cache, target_ids)
    pred = model.predict(X).astype(np.float32, copy=False)
    return np.maximum(pred, 0).reshape(len(candidate_cache["x"]), len(target_ids))


def save_selection(candidate_frame, selected_local, gains, errors, times_s, output):
    chosen = candidate_frame.iloc[selected_local].copy()
    chosen["rank"] = np.arange(1, len(chosen) + 1, dtype=np.int32)
    chosen["gain"] = gains
    chosen["total_error"] = errors
    chosen["iteration_s"] = times_s
    output.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_file(output, layer="greedy_stations", driver="GPKG")
    chosen.drop(columns="geometry").to_csv(output.with_suffix(".csv"), index=False)


def main():
    a = parse_args()
    if a.iterations < 1 or a.full_stations < 1:
        raise ValueError("iterations and full-stations must be positive")
    if EXPECTED_FEATURES != list(PAIR_FEATURES):
        raise ValueError("PAIR_FEATURES order differs from the model feature order")

    overall = time.perf_counter()
    grid = load_grid(a.grid_file)
    full_count = len(grid)
    rng = np.random.default_rng(a.seed)
    nc = min(a.random_candidates, full_count)
    nt = min(a.random_targets, full_count)
    if a.iterations > nc:
        raise ValueError("iterations cannot exceed sampled candidates")

    candidate_ids = np.sort(rng.choice(full_count, nc, replace=False))
    target_ids_grid = np.sort(rng.choice(full_count, nt, replace=False))
    candidates = grid.iloc[candidate_ids].copy().reset_index(drop=True)
    targets = grid.iloc[target_ids_grid].copy().reset_index(drop=True)
    candidate_cache = cache(candidates)
    target_cache = cache(targets)
    all_targets = np.arange(nt, dtype=np.int64)

    print(f"Grid: {full_count:,}; test candidates: {nc:,}; test targets: {nt:,}")
    print(f"Loading model: {a.model.name}")
    model = joblib.load(a.model)
    if getattr(model, "n_features_in_", len(EXPECTED_FEATURES)) != len(EXPECTED_FEATURES):
        raise ValueError(f"Model expects {model.n_features_in_} features, script builds {len(EXPECTED_FEATURES)}")

    selected = []
    selected_mask = np.zeros(nc, dtype=bool)
    current_error = None
    gains, totals, iteration_times = [], [], []

    for iteration in range(a.iterations):
        t_iter = time.perf_counter()
        best_idx, best_score, best_pred = None, None, None
        available = np.flatnonzero(~selected_mask)

        for b0 in range(0, len(available), a.candidate_batch):
            local_ids = available[b0:b0 + a.candidate_batch]
            pred = predict_batch(model, subset_cache(candidate_cache, local_ids), target_cache, all_targets)
            if current_error is None:
                # With no stations, choose the candidate giving the lowest national error.
                scores = pred.sum(axis=1)
                pos = int(np.argmin(scores))
                is_better = best_score is None or scores[pos] < best_score
            else:
                scores = np.maximum(current_error[None, :] - pred, 0).sum(axis=1)
                pos = int(np.argmax(scores))
                is_better = best_score is None or scores[pos] > best_score
            if is_better:
                best_idx = int(local_ids[pos])
                best_score = float(scores[pos])
                best_pred = pred[pos].copy()

        if current_error is None:
            current_error = best_pred
            gain = float("nan")
        else:
            gain = best_score
            current_error = np.minimum(current_error, best_pred)

        selected.append(best_idx)
        selected_mask[best_idx] = True
        elapsed_iter = time.perf_counter() - t_iter
        gains.append(gain)
        totals.append(float(current_error.sum()))
        iteration_times.append(elapsed_iter)
        eta_test = np.mean(iteration_times) * (a.iterations - iteration - 1)
        print(
            f"{iteration + 1:3d}/{a.iterations}: grid_id={int(candidates.iloc[best_idx].source_grid_id):6d} "
            f"gain={gain:,.2f} total_error={totals[-1]:,.2f} "
            f"iteration={elapsed_iter:.1f}s test_ETA={eta_test/60:.1f}m"
        )
        if a.checkpoint:
            save_selection(candidates, selected, gains, totals, iteration_times, a.output)

    save_selection(candidates, selected, gains, totals, iteration_times, a.output)
    elapsed = time.perf_counter() - overall
    avg = float(np.mean(iteration_times))
    # Linear estimate for 400 selections on this sampled problem.
    sampled_400_s = avg * a.full_stations
    # Pair-work scaling estimate for all grid cells as both candidates and targets.
    candidate_factor = full_count / nc
    target_factor = full_count / nt
    full_400_s = sampled_400_s * candidate_factor * target_factor

    summary = {
        "grid_cells": full_count,
        "test_candidates": nc,
        "test_targets": nt,
        "test_iterations": a.iterations,
        "mean_iteration_seconds": avg,
        "test_elapsed_seconds": elapsed,
        "estimated_sampled_400_hours": sampled_400_s / 3600,
        "estimated_full_grid_400_hours": full_400_s / 3600,
        "note": "Estimates are linear extrapolations; later iterations have slightly fewer candidates.",
    }
    a.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTIMING ESTIMATE")
    print(f"Mean greedy iteration: {avg:.1f}s")
    print(f"400 stations on the same sample: {sampled_400_s/3600:.2f}h")
    print(f"400 stations with full grid as candidates and targets: {full_400_s/3600:.2f}h")
    print(f"Saved: {a.output.name}, {a.output.with_suffix('.csv').name}, {a.output.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
