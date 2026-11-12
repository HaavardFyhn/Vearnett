"""Bounded seNorge-only signed temperature transfer and OOF squared-error pilot."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

try:
    from remote_senorge.train_valley_neural import sha256, write_json
except ModuleNotFoundError as error:
    if error.name != "remote_senorge":
        raise
    from train_valley_neural import sha256, write_json


ROOT = Path(__file__).resolve().parent
FEATURES = ["source_temperature_c", "day_sin", "day_cos", "is_daily_minimum",
            "east_difference_km", "north_difference_km", "distance_km"]
FEATURES += [f"{prefix}_terrain_{i}" for prefix in ("source", "target", "signed_difference") for i in range(2, 16)]


def features(table, source, target, temperature, dates, variable):
    a, b = table[source], table[target]
    dx, dy = (b[:, 0]-a[:, 0])/1000, (b[:, 1]-a[:, 1])/1000
    doy = pd.to_datetime(dates).dayofyear.to_numpy()
    x = np.column_stack([temperature, np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                         variable, dx, dy, np.hypot(dx, dy), a[:, 2:], b[:, 2:], b[:, 2:]-a[:, 2:]])
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite signed transfer inputs")
    return x.astype(np.float32)


def geography_pairs(table, refs, per_split=(120, 30, 30), targets_per_source=8):
    pairs, splits, blocks = [], [], []
    all_refs, reference_split, mapping = refs["refs"], refs["split"], refs["mapping"]
    for split, requested in enumerate(per_split):
        candidates = all_refs[reference_split == split]
        codes = np.floor(table[candidates, :2]/150000).astype(int)
        unique = sorted(set(map(tuple, codes)))
        if requested < len(unique):
            raise ValueError("Reference budget must include every existing regional block")
        buckets = [list(candidates[(codes == block).all(1)]) for block in unique]
        chosen = []
        while len(chosen) < requested:
            before = len(chosen)
            for bucket in buckets:
                if bucket and len(chosen) < requested:
                    chosen.append(int(bucket.pop(0)))
            if len(chosen) == before:
                raise ValueError("Insufficient unique scene references")
        for source in chosen:
            scene = int(np.flatnonzero(all_refs == source)[0])
            targets = np.unique(mapping[scene])
            targets = targets[(targets >= 0) & (targets != source)]
            distance = np.linalg.norm(table[targets, :2]-table[source, :2], axis=1)/1000
            bands = []
            for lower, upper in ((0, 5), (5, 15), (15, 30), (30, 46)):
                bucket = targets[(distance > lower) & (distance <= upper)]
                bands.append(list(np.random.default_rng(2026+source).permutation(bucket)))
            selected = []
            while len(selected) < targets_per_source:
                before = len(selected)
                for bucket in bands:
                    if bucket and len(selected) < targets_per_source:
                        selected.append(int(bucket.pop()))
                if len(selected) == before:
                    raise ValueError("Insufficient supported targets in a reference scene")
            block = tuple(np.floor(table[source, :2]/150000).astype(int))
            for target in selected:
                if tuple(np.floor(table[target, :2]/150000).astype(int)) != block:
                    raise ValueError("Scene crosses frozen geographic boundary")
                pairs.append((source, target))
                splits.append(split)
                blocks.append(f"{block[0]}:{block[1]}")
    result = dict(pairs=np.asarray(pairs, np.int32), split=np.asarray(splits), block=np.asarray(blocks))
    for a in range(3):
        endpoints = set(result["pairs"][result["split"] == a].ravel())
        for b in range(a+1, 3):
            if endpoints & set(result["pairs"][result["split"] == b].ravel()):
                raise ValueError("Shared geographic endpoint across pilot splits")
    return result


def sampled_rows(table, weather, plan, pairs, pair_ids, date_ids, count, seed):
    """Sample unique pair/date/variable combinations; missing temperatures are counted."""
    rng = np.random.default_rng(seed)
    combinations = len(pair_ids)*len(date_ids)*2
    count = min(count, combinations)
    selected = rng.choice(combinations, size=count, replace=False)
    variable = selected % 2
    day = date_ids[(selected//2) % len(date_ids)]
    pair = pair_ids[selected//(2*len(date_ids))]
    source, target = pairs["pairs"][pair].T
    source_c, target_c = weather[variable, source, day], weather[variable, target, day]
    keep = np.isfinite(source_c) & np.isfinite(target_c)
    if keep.sum() < count*.9:
        raise ValueError("Too many missing seNorge sampled temperatures")
    variable, day, pair, source, target, source_c, target_c = (
        value[keep] for value in (variable, day, pair, source, target, source_c, target_c))
    dates = plan["date"].to_numpy(dtype=str)[day]
    x = features(table, source, target, source_c, dates, variable)
    return dict(x=x, source_c=source_c, target_c=target_c, source=source.astype(str), target=target.astype(str),
                pair=pair, date=dates, variable=variable, block=pairs["block"][pair],
                missing_rows=int(count-keep.sum()))


def weights(data):
    frame = pd.DataFrame({key: data[key] for key in ("block", "target", "date")})
    groups = frame.groupby(["block", "target", "date"]).size()
    block_groups = groups.groupby(level=0).size()
    return np.asarray([1/(groups[(row.block, row.target, row.date)]*block_groups[row.block])
                       for row in frame.itertuples()])


def subset(data, mask):
    return {key: value[mask] if isinstance(value, np.ndarray) else value for key, value in data.items()}


def fit_mean(data, family, jobs):
    w = weights(data)
    delta = data["target_c"].astype(np.float64)-data["source_c"]
    if family == "offset":
        return dict(family=family, offset=float(np.average(delta, weights=w)))
    if family not in ("signed_linear", "terrain_tree"):
        raise ValueError("Unknown mean predictor family")
    mean = np.average(data["x"].astype(np.float64), axis=0, weights=w)
    std = np.sqrt(np.average((data["x"]-mean)**2, axis=0, weights=w))
    std[std < 1e-8] = 1
    normalized = (data["x"].astype(np.float64)-mean)/std
    linear = Ridge(alpha=10.)
    linear.fit(normalized, delta, sample_weight=w/w.mean())
    result = dict(family=family, linear=linear, mean=mean, std=std)
    if family == "terrain_tree":
        residual = delta-linear.predict(normalized)
        tree = ExtraTreesRegressor(n_estimators=64, max_depth=12, min_samples_leaf=24,
                                   random_state=2026, n_jobs=jobs)
        tree.fit(data["x"], residual, sample_weight=w/w.mean())
        result["tree"] = tree
    return result


def predict_mean(model, data):
    if model["family"] == "offset":
        delta = np.full(len(data["source_c"]), model["offset"])
    else:
        delta = model["linear"].predict((data["x"].astype(np.float64)-model["mean"])/model["std"])
        if model["family"] == "terrain_tree":
            delta += model["tree"].predict(data["x"])
    prediction = data["source_c"].astype(np.float64)+delta
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite signed temperature prediction")
    return prediction


def score(data, prediction):
    error = prediction-data["target_c"]
    return dict(rows=len(error), blocks=len(set(data["block"])), targets=len(set(data["target"])),
                dates=len(set(data["date"])), rmse_c=float(np.sqrt(np.mean(error**2))),
                balanced_rmse_c=float(np.sqrt(np.average(error**2, weights=weights(data)))),
                mae_c=float(np.abs(error).mean()), bias_c=float(error.mean()))


def error_inputs(data, prediction):
    return np.column_stack([data["x"], prediction]).astype(np.float32)


def rmse_predictions(model, data, prediction):
    mse = model.predict(error_inputs(data, prediction))
    if not np.isfinite(mse).all() or (mse < 0).any():
        raise ValueError("Invalid predicted squared error")
    # No positive scale floor: an exactly learned deterministic offset has zero error.
    return np.sqrt(mse)


def error_report(data, prediction, rmse):
    result = dict(observed_rmse_c=float(np.sqrt(np.mean((prediction-data["target_c"])**2))),
                  predicted_pooled_rmse_c=float(np.sqrt(np.mean(rmse**2))))
    dimensions = dict(variable=np.where(data["variable"] == 0, "daily_mean_tg", "daily_minimum_tn"),
                      block=data["block"],
                      season=np.array(["DJF" if int(day[5:7]) in (12, 1, 2) else
                                       "JJA" if int(day[5:7]) in (6, 7, 8) else "MAM_SON" for day in data["date"]]),
                      distance=np.where(data["x"][:, 6] <= 5, "0-5km",
                                        np.where(data["x"][:, 6] <= 15, "5-15km", "15-46km")))
    for dimension, labels in dimensions.items():
        result[dimension] = {}
        for label in np.unique(labels):
            mask = labels == label
            result[dimension][str(label)] = dict(
                rows=int(mask.sum()), observed_rmse_c=float(np.sqrt(np.mean((prediction[mask]-data["target_c"][mask])**2))),
                predicted_pooled_rmse_c=float(np.sqrt(np.mean(rmse[mask]**2))))
    order = np.argsort(rmse, kind="stable")
    result["predicted_rmse_bins"] = [
        dict(rows=len(ids), predicted_pooled_rmse_c=float(np.sqrt(np.mean(rmse[ids]**2))),
             observed_rmse_c=float(np.sqrt(np.mean((prediction[ids]-data["target_c"][ids])**2))))
        for ids in np.array_split(order, 5) if len(ids)]
    return result


def run(args):
    output = args.output
    if output.exists():
        raise FileExistsError("Choose a new pilot output; no overwrite or unverified resume")
    atlas = args.atlas
    manifest = json.loads((atlas / "footprint_manifest.json").read_text())
    for name in ("grid_table.npy", "references.npz", "selected_dates.csv"):
        if sha256(atlas / name) != manifest["checksums"][name]:
            raise ValueError(f"Frozen footprint input changed: {name}")
    progress = json.loads((atlas / "cache" / "weather_progress.json").read_text())
    if progress != dict(completed_years=30, years=30):
        raise ValueError("The full 30-year weather cache is required")
    table = np.load(atlas / "grid_table.npy", mmap_mode="r", allow_pickle=False)
    weather = np.load(atlas / "cache" / "sampled_weather.npy", mmap_mode="r", allow_pickle=False)
    try:
        plan = pd.read_csv(atlas / "selected_dates.csv")
        if weather.shape != (2, len(table), len(plan)) or len(plan) != 3600 or set(plan.year) != set(range(1996, 2026)):
            raise ValueError("Unexpected tg/tn weather atlas/date plan")
        with np.load(atlas / "references.npz", allow_pickle=False) as refs:
            pairs = geography_pairs(table, refs, tuple(args.references), args.targets)
        year = plan.year.to_numpy()
        definitions = {
            "train": (0, year <= 2015, args.train_rows),
            "selection": (1, (year >= 2016) & (year <= 2018), args.evaluation_rows),
            "calibration": (1, (year >= 2019) & (year <= 2020), args.evaluation_rows),
            "test": (2, year >= 2021, args.evaluation_rows),
        }
        datasets = {}
        for number, (name, (geography, dates, count)) in enumerate(definitions.items()):
            datasets[name] = sampled_rows(table, weather, plan, pairs, np.flatnonzero(pairs["split"] == geography),
                                          np.flatnonzero(dates), count, 2026+number)
        # Persist fingerprints of actual immutable source bytes,including raw signed temperatures.
        provenance = {name: sha256(atlas / name) for name in
                      ("footprint_manifest.json", "grid_table.npy", "references.npz", "selected_dates.csv",
                       "cache/sampled_weather.npy", "cache/source_config.json")}
    finally:
        table._mmap.close()
        weather._mmap.close()
    train, selection = datasets["train"], datasets["selection"]
    blocks = sorted(set(train["block"]))
    if len(blocks) < 5:
        raise ValueError("Need at least five distinct training blocks for unseen-geography cross-fitting")
    output.mkdir(parents=True)
    started = time.monotonic()
    models, validation = {}, {}
    for family in ("offset", "signed_linear", "terrain_tree"):
        print(f"Fitting signed mean predictor: {family}", flush=True)
        model = fit_mean(train, family, args.jobs)
        models[family] = model
        validation[family] = score(selection, predict_mean(model, selection))
    selected = min(validation, key=lambda name: (validation[name]["balanced_rmse_c"], name))
    print(f"Selection-only winner: {selected}", flush=True)
    oof = np.full(len(train["target_c"]), np.nan)
    fold_reports = []
    for fold in range(5):
        held = blocks[fold::5]
        evaluate = np.isin(train["block"], held)
        fit = ~evaluate
        if (set(train["source"][fit]) | set(train["target"][fit])) & (
                set(train["source"][evaluate]) | set(train["target"][evaluate])):
            raise ValueError("Cross-fit held endpoint leakage")
        model = fit_mean(subset(train, fit), selected, args.jobs)
        prediction = predict_mean(model, subset(train, evaluate))
        oof[evaluate] = prediction
        fold_reports.append(dict(fold=fold, held_blocks=held, error=score(subset(train, evaluate), prediction)))
        print(f"Unseen-geography OOF fold {fold+1}/5 complete", flush=True)
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete OOF temperature predictions; no error model fitted")
    squared_error = (oof-train["target_c"])**2
    error_model = ExtraTreesRegressor(n_estimators=64, max_depth=12, min_samples_leaf=32,
                                      random_state=2026, n_jobs=args.jobs)
    w = weights(train)
    error_model.fit(error_inputs(train, oof), squared_error, sample_weight=w/w.mean())
    joblib.dump(models[selected], output / "signed_temperature_model.joblib")
    joblib.dump(error_model, output / "squared_error_model.joblib")
    np.savez_compressed(output / "oof_temperature_errors.npz", **{key: train[key] for key in
                        ("x", "source_c", "target_c", "source", "target", "date", "variable", "block")},
                        prediction_c=oof, squared_error_c2=squared_error)
    reports = {}
    for name in ("calibration", "test"):
        data = datasets[name]
        prediction = predict_mean(models[selected], data)
        rmse = rmse_predictions(error_model, data, prediction)
        reports[name] = dict(temperature=score(data, prediction), error_model=error_report(data, prediction, rmse),
                             baselines={family: score(data, predict_mean(model, data)) for family, model in models.items()},
                             raw_difference=score(data, data["source_c"].astype(np.float64)))
        frame = pd.DataFrame({key: data[key] for key in ("source", "target", "date", "variable", "source_c", "target_c", "block")})
        frame["prediction_c"], frame["predicted_rmse_c"] = prediction, rmse
        frame["squared_error_c2"] = (prediction-data["target_c"])**2
        frame.to_csv(output / f"{name}_predictions.csv", index=False)
    report = dict(schema="senorge-corrected-transfer-v1", selected=selected, validation=validation,
                  folds=fold_reports, evaluation=reports, features=FEATURES,
                  rows={name: len(data["target_c"]) for name, data in datasets.items()},
                  missing_rows={name: data["missing_rows"] for name, data in datasets.items()},
                  provenance=provenance, settings={key: str(value) for key, value in vars(args).items()},
                  elapsed_seconds=time.monotonic()-started,
                  limitations=[
                      "Bounded national pilot sampled from 3600 archived dates,not every day/cell pair.",
                      "seNorge grid-to-grid reproduction,not independent weather-station accuracy or a future forecast.",
                      "Original geographic blocks retained; train1996-2015,selection2016-18,calibration2019-20,test2021-25.",
                      "Original footprint geography/period was already inspected; this is exploratory evaluation.",
                      "OOF mean residuals are held-geography predictions; MSE model is not scored on its fitting residuals.",
                      "No claim of zero error for unseen arbitrary pairs; perfect offset requires that the predictor learns it.",
                      "Predicted RMSE=sqrt(predicted mean squared error),not mean absolute error,interval radius or standard deviation.",
                      "No calibrated prediction intervals,no national app replacement or station contribution map.",
                  ], checksums={})
    for name in ("signed_temperature_model.joblib", "squared_error_model.joblib", "oof_temperature_errors.npz",
                 "calibration_predictions.csv", "test_predictions.csv"):
        report["checksums"][name] = sha256(output / name)
    write_json(output / "report.json", report)
    write_json(output / "status.json", dict(state="complete", selected=selected, station_accuracy_validated=False))
    print(json.dumps(report["evaluation"]["test"], indent=2), flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=ROOT / "outputs" / "senorge_footprint_expanded")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "senorge_corrected_transfer_pilot_v2")
    parser.add_argument("--references", nargs=3, type=int, default=[120, 30, 30])
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--train-rows", type=int, default=100000)
    parser.add_argument("--evaluation-rows", type=int, default=25000)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if min(*args.references, args.targets, args.train_rows, args.evaluation_rows, args.jobs) < 1:
        parser.error("Use positive bounded sampling settings")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists() and not (args.output / "report.json").exists():
            write_json(args.output / "status.json", dict(state="failed", error=str(error)))
        raise
