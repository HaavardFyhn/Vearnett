"""Prepare one shared pair dataset for direct and OLS seNorge error models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd

import train_senorge_grid_pairs as training
from remote_senorge.train_remote import sha256

TARGETS = {
    "direct-rmse": "RMSE of same-day temperature differences; no regression",
    "direct-p95": "95th percentile of absolute same-day temperature differences; no regression",
    "rmse": "Mean of directional OLS residual RMSEs",
    "absolute-error-p95": "Mean of directional 95th percentiles of absolute OLS residuals",
}


def p95(errors: np.ndarray) -> np.ndarray:
    labels = np.empty(len(errors))
    complete = np.isfinite(errors).all(axis=1)
    if complete.any():
        labels[complete] = np.quantile(errors[complete], .95, axis=1)
    if not complete.all():
        labels[~complete] = np.nanquantile(errors[~complete], .95, axis=1)
    return labels


def pair_labels(fitting: np.ndarray, evaluation: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    a, b = pairs.T
    result = np.full((len(pairs), len(TARGETS)), np.nan, dtype=np.float32)
    fit_a, fit_b = fitting[a], fitting[b]
    ev_a, ev_b = evaluation[a], evaluation[b]
    valid_fit = np.isfinite(fit_a) & np.isfinite(fit_b)
    valid_eval = np.isfinite(ev_a) & np.isfinite(ev_b)
    fit_count, eval_count = valid_fit.sum(axis=1), valid_eval.sum(axis=1)
    keep = ((fit_count >= max(60, np.ceil(fitting.shape[1] * .9))) &
            (eval_count >= max(20, np.ceil(evaluation.shape[1] * .9))))
    if not keep.any():
        return result
    # Keep pairwise-complete fitting dates in both directions. Float64 moments
    # preserve the existing OLS definition even when the source has low variance.
    valid = valid_fit[keep]
    x = np.where(valid, fit_a[keep], 0).astype(np.float64)
    y = np.where(valid, fit_b[keep], 0).astype(np.float64)
    stats = np.array([fit_count[keep], x.sum(axis=1), y.sum(axis=1),
                      np.einsum("ij,ij->i", x, x), np.einsum("ij,ij->i", y, y),
                      np.einsum("ij,ij->i", x, y)])
    del x, y, valid
    ev_a, ev_b = ev_a[keep], ev_b[keep]
    observed = valid_eval[keep]
    count = eval_count[keep]

    def metrics(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        errors = np.where(observed, np.abs(errors), np.nan)
        return np.sqrt(np.nansum(errors * errors, axis=1, dtype=np.float64) / count), p95(errors)

    direct_rmse, direct_p95 = metrics(ev_b - ev_a)
    slope, intercept = training.ols_coefficients(stats)
    forward_rmse, forward_p95 = metrics(ev_b - slope[:, None] * ev_a - intercept[:, None])
    slope, intercept = training.ols_coefficients(stats[[0, 2, 1, 4, 3, 5]])
    reverse_rmse, reverse_p95 = metrics(ev_a - slope[:, None] * ev_b - intercept[:, None])
    result[keep] = np.column_stack((direct_rmse, direct_p95,
                                    (forward_rmse + reverse_rmse) / 2,
                                    (forward_p95 + reverse_p95) / 2))
    return result


def load_temperatures(archives: list[Path], grid: gpd.GeoDataFrame,
                      plan: pd.DataFrame, path: Path) -> np.ndarray:
    if not path.exists():
        pending = path.with_suffix(".pending.npy")
        # Partition first, so pair workers get contiguous fitting/evaluation series.
        ordered = plan.sort_values(["partition", "date"]).reset_index(drop=True)
        data = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32,
                                         shape=(len(ordered), len(grid)))
        rows, cols = grid.nc_row.to_numpy(), grid.nc_col.to_numpy()
        axes = None
        try:
            for archive in archives:
                year = int(archive.stem.split("_")[-1])
                with netCDF4.Dataset(archive) as ds:
                    var = ds["tg"]
                    current = np.asarray(ds["X"][:]), np.asarray(ds["Y"][:])
                    if (var.units != "Celsius" or var.dimensions != ("time", "Y", "X") or
                            training.RasterCRS.from_string(ds[var.grid_mapping].proj4).to_epsg() != 32633):
                        raise ValueError(f"Unexpected temperature grid: {archive}")
                    if axes is not None and not all(np.array_equal(a, b) for a, b in zip(axes, current)):
                        raise ValueError(f"Temperature axes changed: {archive}")
                    axes = current
                    time = ds["time"]
                    days = ordered.loc[ordered.year == year]
                    actual = netCDF4.num2date(time[days.day_index.to_numpy()], time.units,
                                              getattr(time, "calendar", "standard"))
                    if [day.strftime("%Y-%m-%d") for day in actual] != days.date.tolist():
                        raise ValueError(f"Archive dates differ from the date plan: {archive}")
                    for index, day in zip(days.index, days.day_index):
                        data[index] = np.ma.masked_invalid(var[day]).filled(np.nan)[rows, cols]
                print(f"Temperatures: {year}", flush=True)
            data.flush()
        finally:
            del data
        pending.replace(path)
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.shape != (len(plan), len(grid)) or data.dtype != np.float32:
        raise ValueError("Invalid cached temperature matrix")
    return data


def prepare(source: Path, cache: Path, output: Path, count: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    grid_path = source / "grid_features.gpkg"
    archives = [Path(item["path"]) for item in config["archives"]]
    plan = training.date_plan(config["years"], config["date_step"], config["days_per_month"],
                              config["temporal_split"], config["seed"])
    split = int((plan.partition == 0).sum())
    if count < 1 or split < 60 or len(plan) - split < 20:
        raise ValueError("Need positive pair count and at least 60 fitting and 20 evaluation dates")
    identity = dict(schema=1, pairs=count, config=config, grid_sha256=sha256(grid_path),
                    dates=plan.to_dict("records"), archives=[training.fingerprint(p) for p in archives])
    cache.mkdir(parents=True, exist_ok=True)
    metadata = cache / "inputs.json"
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != identity:
            raise ValueError("The preparation cache belongs to different inputs; use a new cache")
    else:
        training.write_json(metadata, identity)
    grid = gpd.read_file(grid_path)
    pairs_path = cache / "pairs.npy"
    if not pairs_path.exists():
        pairs = training.sample_pairs(grid, count, maximum=config["maximum_distance_m"],
                                      scale=config["distance_scale_m"], seed=config["seed"],
                                      distribution=config["distance_distribution"])
        pending = pairs_path.with_suffix(".pending.npy")
        np.save(pending, pairs)
        pending.replace(pairs_path)
    pairs = np.load(pairs_path, allow_pickle=False)
    if pairs.shape != (count, 2) or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("Invalid cached pair index")
    print(f"Sample ready: {len(pairs):,} unique grid pairs", flush=True)
    temperatures = load_temperatures(archives, grid, plan, cache / "temperatures.npy")
    fitting = np.ascontiguousarray(temperatures[:split].T)
    evaluation = np.ascontiguousarray(temperatures[split:].T)
    del temperatures
    labels = np.empty((len(pairs), len(TARGETS)), dtype=np.float32)
    for start in range(0, len(pairs), 8192):
        stop = min(start + 8192, len(pairs))
        checkpoint = cache / f"labels_{start:08d}.npy"
        if not checkpoint.exists():
            chunk = pair_labels(fitting, evaluation, pairs[start:stop])
            pending = checkpoint.with_suffix(".pending.npy")
            np.save(pending, chunk)
            pending.replace(checkpoint)
        labels[start:stop] = np.load(checkpoint, allow_pickle=False)
        if start % (32 * 8192) == 0 or stop == len(pairs):
            print(f"Labels: {stop:,}/{len(pairs):,}", flush=True)
    del fitting, evaluation
    keep = np.isfinite(labels).all(axis=1)
    pairs, labels = pairs[keep], labels[keep]
    train, test = training.spatial_split(grid, pairs, config["seed"])
    _, blocks = np.unique(np.floor(grid[["x", "y"]].to_numpy() / 100_000).astype(int),
                          axis=0, return_inverse=True)
    ids = grid.grid_id.to_numpy()
    features = np.empty((len(pairs), len(training.FEATURES)), dtype=np.float32)
    for start in range(0, len(pairs), 100_000):
        stop = start + 100_000
        features[start:stop] = training.pair_features(grid, pairs[start:stop]).to_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="senorge_multi_") as temporary:
        work = Path(temporary)
        np.savez(work / "training.npz", x=features, train=train, test=test,
                 grid_id_1=ids[pairs[:, 0]], grid_id_2=ids[pairs[:, 1]],
                 block_1=blocks[pairs[:, 0]], block_2=blocks[pairs[:, 1]],
                 **{f"y_{key}": labels[:, i] for i, key in enumerate(TARGETS)})
        plan.to_csv(work / "selected_dates.csv", index=False)
        grid.drop(columns="geometry").to_csv(work / "grid_features.csv.gz", index=False)
        manifest = dict(schema=2, dataset_sha256=sha256(work / "training.npz"),
                        features=training.FEATURES, base_feature_count=len(training.PAIR_FEATURES),
                        targets=TARGETS, pairs=len(pairs), requested_pairs=count,
                        spatial_train=int(train.sum()), spatial_test=int(test.sum()),
                        years=config["years"], train_days=split, test_days=len(plan)-split,
                        maximum_distance_m=config["maximum_distance_m"],
                        sampling=dict(distribution=config["distance_distribution"],
                                      scale_m=config["distance_scale_m"], seed=config["seed"],
                                      temporal_split=config["temporal_split"]),
                        training_source="seNorge_2018 daily means; no Frost calibration",
                        limitations=["Same historical evaluation dates for all four targets.",
                                     "Direct targets use the other cell's daily mean without adjustment.",
                                     "No station-observation or independent interval-coverage validation."])
        training.write_json(work / "manifest.json", manifest)
        for name in ("train_remote.py", "test_train_remote.py", "requirements.txt", "SERVER_README.md"):
            shutil.copyfile(Path(__file__).with_name("remote_senorge") / name, work / name)
        training.write_json(work / "package_checksums.json", {p.name: sha256(p) for p in work.iterdir()})
        pending = output.with_suffix(".pending")
        archive = Path(shutil.make_archive(str(pending), "zip", work))
        archive.replace(output)
    print(f"Packaged {len(pairs):,}/{count:,} accepted pairs: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=training.WORK / "senorge_grid_pairs_1m_1996_2025_random")
    parser.add_argument("--pairs", type=int, default=10_000_000)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".zip":
        parser.error("--output must end in .zip")
    prepare(args.source, args.cache, args.output, args.pairs)


if __name__ == "__main__":
    main()
