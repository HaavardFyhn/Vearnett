"""Greedy design of a 400-station Norwegian weather network.

Uses an existing trained joblib model. Checkpoints after every station and resumes
when the same command is run again.

From --prune-after onward, candidates are ranked only against target cells whose
current error is above --target-percentile. After choosing the winner, its error
is still predicted for ALL grid cells, so the national error surface remains exact.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import geopandas as gpd
import joblib
import numpy as np

FEATURES = [
    "distance_m", "elev_diff_m", "coast_fjord_diff_m",
    "coast_ocean_diff_m", "same_objtype", "x_diff", "y_diff",
    "relief_diff_10km", "rugstd_diff_10km", "tpi_diff_10km",
    "relief_diff_25km", "rugstd_diff_25km", "tpi_diff_25km",
]
TERRAIN = ["relief_10km", "rugstd_10km", "tpi_10km",
           "relief_25km", "rugstd_25km", "tpi_25km"]
CACHE_KEYS = ["x", "y", "elevation", "coast_dist_fjord",
              "coast_dist_ocean", "objtype_code", *TERRAIN]


def parse_args():
    p = argparse.ArgumentParser(description="Greedy station placement with checkpoint/resume")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--grid-file", type=Path, required=True)
    p.add_argument("--stations", type=int, default=400)
    p.add_argument("--candidate-batch", type=int, default=50)
    p.add_argument("--output-prefix", type=Path,
                   default=Path("outputs/greedy_400_stations"))
    p.add_argument("--prune-after", type=int, default=101,
                   help="First station number using percentile pruning; 0 disables pruning")
    p.add_argument("--target-percentile", type=float, default=90.0,
                   help="After pruning starts, rank candidates using targets above this current-error percentile")
    p.add_argument("--save-error-surface-every", type=int, default=10)
    p.add_argument("--restart", action="store_true")
    return p.parse_args()


def fingerprint(path):
    s = path.stat()
    raw = f"{path.resolve()}|{s.st_size}|{s.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()


def load_grid(path):
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError("Grid CRS is missing")
    if g.crs.to_epsg() != 25833:
        g = g.to_crs(25833)
    required = ["elevation", "coast_dist_fjord", "coast_dist_ocean",
                "objtype", *TERRAIN, "geometry"]
    missing = sorted(set(required) - set(g.columns))
    if missing:
        raise ValueError(f"Grid missing fields: {missing}")
    g = g.dropna(subset=required).copy().reset_index(drop=True)
    g["x"] = g.geometry.x.astype(np.float32)
    g["y"] = g.geometry.y.astype(np.float32)
    g["grid_id"] = np.arange(len(g), dtype=np.int32)
    # Encode once on the complete grid to keep category codes consistent.
    g["objtype_code"] = g["objtype"].astype("category").cat.codes.astype(np.int16)
    return g


def make_cache(g):
    cache = {k: g[k].to_numpy(np.int16 if k == "objtype_code" else np.float32)
             for k in CACHE_KEYS}
    # The grid-pair model subtracts in float64 before the final float32 feature matrix.
    cache.update({f"senorge_{k}": g[k].to_numpy(np.float64) for k in CACHE_KEYS if k != "objtype_code"})
    return cache


def subset_cache(cache, ids):
    return {k: v[ids] for k, v in cache.items()}


def feature_matrix(src, target, target_ids):
    sx, sy = src["x"][:, None], src["y"][:, None]
    tx = target["x"][target_ids][None, :]
    ty = target["y"][target_ids][None, :]
    cols = [
        np.hypot(sx - tx, sy - ty).ravel(),
        np.abs(src["elevation"][:, None] - target["elevation"][target_ids][None, :]).ravel(),
        np.abs(src["coast_dist_fjord"][:, None] - target["coast_dist_fjord"][target_ids][None, :]).ravel(),
        np.abs(src["coast_dist_ocean"][:, None] - target["coast_dist_ocean"][target_ids][None, :]).ravel(),
        (src["objtype_code"][:, None] == target["objtype_code"][target_ids][None, :]).astype(np.float32).ravel(),
        np.abs(sx - tx).ravel(), np.abs(sy - ty).ravel(),
    ]
    for key in TERRAIN:
        cols.append(np.abs(src[key][:, None] - target[key][target_ids][None, :]).ravel())
    X = np.column_stack(cols).astype(np.float32, copy=False)
    if X.shape[1] != len(FEATURES):
        raise RuntimeError(f"Built {X.shape[1]} features, expected {len(FEATURES)}")
    return X


def predict(model, src, targets, target_ids):
    if hasattr(model, "predict_pairs"):
        source_ids = np.repeat(np.arange(len(src["x"])), len(target_ids))
        paired_targets = np.tile(target_ids, len(src["x"]))
        return model.predict_pairs(src, targets, source_ids, paired_targets).reshape(len(src["x"]), len(target_ids))
    values = model.predict(feature_matrix(src, targets, target_ids))
    return np.maximum(values, 0).astype(np.float32).reshape(len(src["x"]), len(target_ids))


def output_paths(prefix):
    return {
        "state": prefix.with_name(prefix.name + "_checkpoint.npz"),
        "meta": prefix.with_name(prefix.name + "_checkpoint.json"),
        "stations": prefix.with_name(prefix.name + "_stations.gpkg"),
        "csv": prefix.with_name(prefix.name + "_stations.csv"),
        "error": prefix.with_name(prefix.name + "_current_error.gpkg"),
    }


def atomic_npz(path, **arrays):
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def save_checkpoint(out, grid, selected, gains, totals, seconds, current_error,
                    selected_mask, thresholds, target_counts, metadata, save_surface):
    out["state"].parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(out["state"], selected=np.asarray(selected, np.int32),
               gains=np.asarray(gains, np.float64), totals=np.asarray(totals, np.float64),
               iteration_seconds=np.asarray(seconds, np.float64),
               current_error=np.asarray(current_error, np.float32),
               selected_mask=np.asarray(selected_mask, bool),
               prune_thresholds=np.asarray(thresholds, np.float64),
               evaluated_target_counts=np.asarray(target_counts, np.int32))
    tmp = out["meta"].with_name(out["meta"].name + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(tmp, out["meta"])

    chosen = grid.iloc[selected].copy()
    chosen["rank"] = np.arange(1, len(chosen) + 1, dtype=np.int32)
    chosen["gain"] = gains
    chosen["total_error"] = totals
    chosen["iteration_s"] = seconds
    chosen["prune_threshold"] = thresholds
    chosen["evaluated_targets"] = target_counts
    chosen.to_file(out["stations"], layer="stations", driver="GPKG")
    chosen.drop(columns="geometry").to_csv(out["csv"], index=False)
    if save_surface:
        surface = grid[["grid_id", "geometry"]].copy()
        surface["current_error"] = current_error
        surface.to_file(out["error"], layer="current_error", driver="GPKG")


def load_checkpoint(out, expected, n):
    if not (out["state"].exists() and out["meta"].exists()):
        return None
    meta = json.loads(out["meta"].read_text(encoding="utf-8"))
    for key in ("model_fingerprint", "grid_fingerprint", "feature_names",
                "prune_after", "target_percentile"):
        if meta.get(key) != expected.get(key):
            raise RuntimeError(f"Checkpoint differs in {key}; use --restart or original settings")
    z = np.load(out["state"], allow_pickle=False)
    if len(z["current_error"]) != n:
        raise RuntimeError("Checkpoint grid size differs; use --restart")
    return dict(
        selected=z["selected"].astype(int).tolist(),
        gains=z["gains"].astype(float).tolist(), totals=z["totals"].astype(float).tolist(),
        seconds=z["iteration_seconds"].astype(float).tolist(),
        current_error=z["current_error"].astype(np.float32),
        selected_mask=z["selected_mask"].astype(bool),
        thresholds=z["prune_thresholds"].astype(float).tolist(),
        target_counts=z["evaluated_target_counts"].astype(int).tolist(), metadata=meta)


def main():
    a = parse_args()
    if not 0 <= a.target_percentile < 100:
        raise ValueError("--target-percentile must be >= 0 and < 100")
    if a.stations < 1 or a.candidate_batch < 1:
        raise ValueError("--stations and --candidate-batch must be positive")
    out = output_paths(a.output_prefix)
    if a.restart:
        for p in out.values():
            if p.exists(): p.unlink()

    grid = load_grid(a.grid_file)
    n = len(grid)
    cache = make_cache(grid)
    all_targets = np.arange(n, dtype=np.int64)
    model = joblib.load(a.model)
    if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
        raise ValueError(f"Model expects {model.n_features_in_} features; script builds {len(FEATURES)}")

    meta = dict(model_fingerprint=fingerprint(a.model), grid_fingerprint=fingerprint(a.grid_file),
                feature_names=FEATURES, prune_after=a.prune_after,
                target_percentile=a.target_percentile, requested_stations=a.stations,
                grid_cells=n, model=str(a.model.resolve()), grid_file=str(a.grid_file.resolve()))
    state = load_checkpoint(out, meta, n)
    if state:
        selected, gains, totals = state["selected"], state["gains"], state["totals"]
        seconds, current_error = state["seconds"], state["current_error"]
        selected_mask = state["selected_mask"]
        thresholds, target_counts, meta = state["thresholds"], state["target_counts"], state["metadata"]
        meta["requested_stations"] = a.stations
        print(f"RESUME: {len(selected)} stations already selected")
    else:
        selected, gains, totals, seconds, thresholds, target_counts = [], [], [], [], [], []
        current_error = None
        selected_mask = np.zeros(n, dtype=bool)
        print(f"START: {n:,} candidates and targets; no checkpoint")

    run_start = time.perf_counter()
    while len(selected) < a.stations:
        station_no = len(selected) + 1
        t0 = time.perf_counter()
        available = np.flatnonzero(~selected_mask)

        # 90-percentile pruning: rank candidates only on the worst-error 10%.
        pruning = (a.prune_after > 0 and station_no >= a.prune_after and current_error is not None)
        if pruning:
            threshold = float(np.percentile(current_error, a.target_percentile))
            eval_targets = np.flatnonzero(current_error > threshold)
            # Ties can theoretically leave none; use >= or all cells as safe fallbacks.
            if len(eval_targets) == 0:
                eval_targets = np.flatnonzero(current_error >= threshold)
            if len(eval_targets) == 0:
                eval_targets = all_targets
        else:
            threshold = float("nan")
            eval_targets = all_targets

        best_idx = None
        best_score = None
        for b0 in range(0, len(available), a.candidate_batch):
            ids = available[b0:b0 + a.candidate_batch]
            batch_pred = predict(model, subset_cache(cache, ids), cache, eval_targets)
            if current_error is None:
                scores = batch_pred.sum(axis=1, dtype=np.float64)
                local = int(np.argmin(scores))
                better = best_score is None or scores[local] < best_score
            else:
                scores = np.maximum(current_error[eval_targets][None, :] - batch_pred, 0).sum(axis=1, dtype=np.float64)
                local = int(np.argmax(scores))
                better = best_score is None or scores[local] > best_score
            if better:
                best_idx, best_score = int(ids[local]), float(scores[local])

        if best_idx is None:
            raise RuntimeError("No candidate selected")

        # Important: update the winner against the FULL grid, not just pruned targets.
        winner_pred = predict(model, subset_cache(cache, np.array([best_idx])), cache, all_targets)[0]
        if current_error is None:
            current_error = winner_pred
            gain = float("nan")
        else:
            old_total = float(current_error.sum(dtype=np.float64))
            current_error = np.minimum(current_error, winner_pred)
            gain = old_total - float(current_error.sum(dtype=np.float64))

        selected.append(best_idx); selected_mask[best_idx] = True
        gains.append(gain); totals.append(float(current_error.sum(dtype=np.float64)))
        seconds.append(time.perf_counter() - t0)
        thresholds.append(threshold); target_counts.append(len(eval_targets))
        meta.update(updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    completed_stations=len(selected), elapsed_this_run_seconds=time.perf_counter()-run_start)
        save_surface = (station_no == 1 or station_no % a.save_error_surface_every == 0 or station_no == a.stations)
        save_checkpoint(out, grid, selected, gains, totals, seconds, current_error,
                        selected_mask, thresholds, target_counts, meta, save_surface)
        avg = float(np.mean(seconds))
        eta = avg * (a.stations - len(selected)) / 3600
        mode = f"PRUNED>{a.target_percentile:g}%" if pruning else "FULL"
        print(f"{station_no:3d}/{a.stations}: grid_id={best_idx:6d} gain={gain:,.3f} "
              f"total_error={totals[-1]:,.2f} targets={len(eval_targets):,} {mode} "
              f"iteration={seconds[-1]/60:.1f}m ETA={eta:.1f}h CHECKPOINT SAVED", flush=True)

    print(f"Completed {len(selected)} stations; result: {out['stations'].name}")

if __name__ == "__main__":
    main()
