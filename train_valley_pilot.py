"""Compare terrain, weather-regime and graph-informed station transfer."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from prepare_valley_pilot import ROOT, dates, validate_day
from remote_senorge.train_remote import sha256, write_json
from valley_pilot import (
    GRAPH_FIELDS, HOURS, REGIME_FIELDS, TERRAIN_FIELDS, WINTERS,
    conformal_radius, select_observations, winter_year,
)


VARIANTS = ("terrain", "regime", "graph_regime")


def static_features(table, source, target):
    a, b = table[source], table[target]
    return [
        np.linalg.norm(a[:2]-b[:2])/1000, (b[2]-a[2])/1000, a[2]/1000, b[2]/1000,
        a[4]/1000, b[4]/1000, a[3]/1000, b[3]/1000, a[8]/1000, b[8]/1000,
        a[6]/1000, b[6]/1000,
    ]


def verify_inputs(folder):
    config = json.loads((folder / "config.json").read_text())
    if config["schema"] != 1:
        raise ValueError("Unsupported valley pilot schema")
    for name, checksum in config["terrain_checksums"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"Pilot terrain/catalogue checksum mismatch: {name}")
    stations = pd.read_csv(folder / "stations.csv")
    if stations["station_id"].duplicated().any() or not stations["group"].isin([0, 1, 2]).all():
        raise ValueError("Invalid station catalogue")
    if stations.loc[stations["station_id"] == "SN56960", "group"].tolist() != [2]:
        raise ValueError("Haukedal must be final holdout")
    if stations.loc[stations["station_id"] == "SN55820", "group"].tolist() != [1]:
        raise ValueError("Fjærland–Bremuseet must be validation holdout")
    receipts = {}
    expected = {day.isoformat() for day in dates()}
    for stage in ("regimes", "observations"):
        path = folder / f"{stage}_progress.json"
        if not path.exists():
            raise ValueError(f"Missing {stage}: run prepare_valley_pilot.py --stage {stage}")
        receipts[stage] = json.loads(path.read_text())
        if set(receipts[stage]) != expected:
            raise ValueError(f"Incomplete {stage} preparation: need all{len(expected)}winter nights")
        for day, checksum in receipts[stage].items():
            path = folder / stage / f"{day}.json"
            if sha256(path) != checksum:
                raise ValueError(f"Pilot checksum mismatch: {path}")
            validate_day(json.loads(path.read_text()), pd.Timestamp(day).date(), stations, stage)
    return config, stations, receipts


def build_samples(stations, table, graph, observations, regimes):
    if table.shape != (len(stations), 16) or graph.shape != (len(stations), len(stations), len(GRAPH_FIELDS)):
        raise ValueError("Terrain feature shapes differ from station catalogue")
    if not np.isfinite(table).all() or not np.isfinite(graph).all():
        raise ValueError("Nonfinite terrain features")
    if not set(observations["station_id"]) <= set(stations["station_id"]):
        raise ValueError("Unknown observation station")
    if observations.duplicated(["station_id", "time"]).any():
        raise ValueError("Ambiguous station/time observations")
    identifiers = stations["station_id"].tolist()
    groups = stations["group"].to_numpy()
    times = observations.pivot(index="time", columns="station_id", values="value").reindex(columns=identifiers)
    times = times.sort_index()
    training_available = set(observations.loc[
        observations["time"].map(winter_year) == WINTERS[0], "station_id"])
    train_stations = np.array([i for i in np.flatnonzero(groups == 0)
                               if identifiers[i] in training_available], dtype=int)
    distances = np.linalg.norm(table[:, None, :2]-table[None, :, :2], axis=2)
    sources = {
        target: [int(i) for i in sorted(train_stations, key=lambda i: (distances[target, i], identifiers[i]))
                 if i != target and distances[target, i] <= 45000 and graph[i, target, 0] == 1][:6]
        for target in range(len(stations))}
    if not all(sources[i] for i in np.flatnonzero(groups != 0)):
        raise ValueError("Held-out group has no connected training station within45km")
    rows, base, weather, connected, y = [], [], [], [], []
    for stamp, temperatures in times.iterrows():
        year = winter_year(stamp)
        if year not in WINTERS or stamp.hour not in HOURS or stamp.minute != 0:
            raise ValueError("Observation time outside declared winters/hours")
        day = stamp.date().isoformat()
        if day not in regimes:
            raise ValueError(f"Missing weather regime for {day}")
        forecast = regimes[day]
        run = pd.Timestamp(forecast["run_time"])
        if run.tzinfo is None or (stamp-run).total_seconds()/3600 not in (12, 15, 18):
            raise ValueError("Forecast must precede observation by12–18hours")
        if forecast["station_ids"] != identifiers:
            raise ValueError("Weather/terrain station order differs")
        hour = HOURS.index(stamp.hour)
        fields = np.array([forecast["fields"][field][hour] for field in REGIME_FIELDS], dtype=float).T
        bearing = np.asarray(forecast["bearing"], dtype=float)
        if fields.shape != (len(stations), 4) or bearing.shape != (len(stations), len(stations), 2):
            raise ValueError("Invalid regime/bearing shape")
        if not np.isfinite(fields).all() or not np.isfinite(bearing).all():
            raise ValueError("Nonfinite weather inputs")
        for target in range(len(stations)):
            split = int(groups[target])
            if year != WINTERS[split] or not np.isfinite(temperatures.iloc[target]):
                continue
            for source in sources[target]:
                source_temp = temperatures.iloc[source]
                if not np.isfinite(source_temp):
                    continue
                calendar = [np.sin(2*np.pi*stamp.dayofyear/365.25),
                            np.cos(2*np.pi*stamp.dayofyear/365.25), stamp.hour/6]
                base.append([source_temp, *calendar, *static_features(table, source, target)])
                a, b = fields[source], fields[target]
                delta = b[0]-a[0]
                source_speed, target_speed = np.linalg.norm(a[1:3]), np.linalg.norm(b[1:3])
                weather.append([*a, *b, delta, source_temp-a[0], source_speed, target_speed])
                along = float(np.dot(b[1:3], bearing[source, target]))
                barrier = graph[source, target]
                connected.append([*barrier, along, abs(along)*barrier[2],
                                  source_speed*barrier[1], target_speed*barrier[2],
                                  (1-b[3])*barrier[2], (source_temp-a[0])*barrier[1]])
                y.append(float(temperatures.iloc[target]-source_temp))
                rows.append(dict(source=identifiers[source], target=identifiers[target], time=stamp.isoformat(),
                                 night=day, split=split, source_c=float(source_temp),
                                 target_c=float(temperatures.iloc[target]), forecast_delta_c=float(delta),
                                 calm_clear=bool(max(source_speed, target_speed) <= 2 and max(a[3], b[3]) <= .3)))
    if not rows:
        raise ValueError("No matched station-transfer examples")
    samples = pd.DataFrame(rows)
    base, weather, connected = (np.asarray(array, np.float32) for array in (base, weather, connected))
    data = dict(terrain=base, regime=np.column_stack([base, weather]),
                graph_regime=np.column_stack([base, weather, connected]))
    validate_splits(samples)
    feature_names = dict(
        terrain=["source_temperature_c", "day_sin", "day_cos", "utc_hour_fraction", *TERRAIN_FIELDS],
        regime=["source_forecast_c", "source_meps_wind_x", "source_meps_wind_y", "source_cloud",
                "target_forecast_c", "target_meps_wind_x", "target_meps_wind_y", "target_cloud",
                "forecast_difference_c", "source_forecast_residual_c", "source_speed", "target_speed"],
        graph_regime=[*GRAPH_FIELDS, "target_along_wind", "along_wind_times_target_pass",
                      "source_speed_times_pass", "target_speed_times_pass", "clear_times_target_pass",
                      "source_residual_times_pass"])
    feature_names["regime"] = feature_names["terrain"]+feature_names["regime"]
    feature_names["graph_regime"] = feature_names["regime"]+feature_names["graph_regime"]
    return samples, data, np.asarray(y, np.float32), feature_names


def validate_splits(samples):
    endpoints = {}
    times = {}
    for split in (0, 1, 2):
        part = samples.loc[samples["split"] == split]
        if part.empty:
            raise ValueError(f"No accepted examples for split{split}; inspect Frost quality/sensor coverage")
        endpoints[split] = set(part["target"])
        times[split] = set(part["time"])
        years = {winter_year(pd.Timestamp(stamp)) for stamp in times[split]}
        if years != {WINTERS[split]}:
            raise ValueError("Winter split leakage")
    train_endpoints = endpoints[0] | set(samples.loc[samples["split"] == 0, "source"])
    if train_endpoints & (endpoints[1] | endpoints[2]) or endpoints[1] & endpoints[2]:
        raise ValueError("Spatial endpoint leakage")
    if set(samples["source"]) & (endpoints[1] | endpoints[2]):
        raise ValueError("Held-out station cannot act as an observed source")
    if any(times[a] & times[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Temporal leakage")


def scores(samples, prediction, radius=None):
    error = np.asarray(prediction)-samples["target_c"].to_numpy()
    if error.shape != (len(samples),) or not np.isfinite(error).all():
        raise ValueError("Invalid temperature predictions")
    report = dict(rows=len(samples), nights=samples["night"].nunique(),
                  target_stations=samples["target"].nunique(), rmse_c=float(np.sqrt(np.mean(error**2))),
                  mae_c=float(np.mean(np.abs(error))), bias_c=float(error.mean()))
    for name, mask in (("forecast_calm_clear", samples["calm_clear"].to_numpy()),
                       ("cold_below_minus5", samples["target_c"].to_numpy() <= -5)):
        report[name] = dict(rows=int(mask.sum()), rmse_c=float(np.sqrt(np.mean(error[mask]**2)))
                           if mask.any() else None)
    report["per_target"] = {}
    for station in samples["target"].unique():
        values = error[samples["target"].to_numpy() == station]
        report["per_target"][station] = dict(rows=len(values), rmse_c=float(np.sqrt(np.mean(values**2))))
    if radius is not None:
        report["interval_radius_c"] = float(radius)
        report["empirical_pair_coverage_90"] = float((np.abs(error) <= radius).mean())
        night_max = pd.Series(np.abs(error), index=samples.index).groupby(samples["night"]).max()
        report["empirical_night_simultaneous_coverage_90"] = float((night_max <= radius).mean())
    return report


def compare(samples, data, y, *, trees=128, jobs=4):
    validate_splits(samples)
    train = samples["split"].to_numpy() == 0
    validation_nights = sorted(samples.loc[samples["split"] == 1, "night"].unique())
    if len(validation_nights) < 30:
        raise ValueError("Need at least30validation nights for separate selection and calibration")
    middle = len(validation_nights)//2
    selection = samples["night"].isin(validation_nights[:middle]).to_numpy()
    calibration = samples["night"].isin(validation_nights[middle:]).to_numpy()
    test = samples["split"].to_numpy() == 2
    masks = dict(selection=selection, calibration=calibration, test=test)
    predictions, models, report = {}, {}, {}
    for name in ("forecast_transfer", *VARIANTS):
        if name == "forecast_transfer":
            predicted = samples["source_c"].to_numpy()+samples["forecast_delta_c"].to_numpy()
        else:
            model = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=16, max_depth=18,
                                        n_jobs=jobs, random_state=42, max_features=1.)
            model.fit(data[name][train], y[train])
            predicted = samples["source_c"].to_numpy()+model.predict(data[name])
            models[name] = model
        cal_errors = pd.Series(
            np.abs(predicted[calibration]-samples.loc[calibration, "target_c"].to_numpy()),
            index=samples.loc[calibration, "night"]).groupby(level=0).max().to_numpy()
        radius = conformal_radius(cal_errors)
        report[name] = {
            part: scores(samples.loc[mask], predicted[mask], radius if part == "test" else None)
            for part, mask in masks.items()}
        report[name]["calibration_nights"] = len(cal_errors)
        predictions[name] = predicted
    selected = min(report, key=lambda name: report[name]["selection"]["rmse_c"])
    return models, predictions, dict(
        selected_by_validation=selected, variants=report,
        interval_policy="90% absolute-error radius calibrated on maximum error per held-out validation night, disjoint from selection nights.",
        limitations=[
            "One winter per split and two geographic anchor groups: pilot, not national certification.",
            "Rows share stations and nights; row count is not independent sample size.",
            "Conformal nominal coverage is not guaranteed under different valleys/weather distributions; report empirical coverage.",
            "MEPS can assimilate past held-out-station readings;12–18h lead avoids same-time targets, not all historical dependence.",
            "Forecast calm-clear is an inversion-risk proxy, not an observed inversion classification.",
            "Graph is a1km minimax terrain feature, not a GNN or a learned covariance model.",
            "Three nighttime samples do not define observed daily minimum.",
        ])


def load_prepared_samples(folder):
    folder = Path(folder).resolve()
    config, stations, receipts = verify_inputs(folder)
    with np.load(folder / "terrain.npz", allow_pickle=False) as terrain:
        table, graph = terrain["table"].copy(), terrain["graph"].copy()
    raw_rows, regimes = [], {}
    for day in sorted(receipts["regimes"]):
        raw_rows.extend(json.loads((folder / "observations" / f"{day}.json").read_text())["records"])
        regimes[day] = json.loads((folder / "regimes" / f"{day}.json").read_text())
    raw = pd.DataFrame(raw_rows)
    observations, sensors = select_observations(raw)
    samples, features, labels, names = build_samples(stations, table, graph, observations, regimes)
    return config, samples, features, labels, names, sensors, raw


def run(args):
    folder = args.data.resolve()
    config, samples, features, labels, names, sensors, raw = load_prepared_samples(folder)
    print(f"Training pilot: {samples.groupby('split').size().to_dict()} matched pairs", flush=True)
    models, predictions, report = compare(samples, features, labels, trees=args.trees, jobs=args.jobs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sensors.to_csv(output / "sensor_choices.csv", index=False)
    samples.to_csv(output / "samples.csv", index=False)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib")
    test_predictions = samples.loc[samples["split"] == 2].copy()
    for name, values in predictions.items():
        test_predictions[name+"_prediction_c"] = values[samples["split"].to_numpy() == 2]
    test_predictions.to_csv(output / "test_predictions.csv", index=False)
    report.update(
        features=names, input_config=config, rows_by_split=samples.groupby("split").size().to_dict(),
        qc_rejections=dict(Counter(raw.loc[~raw["accepted"].eq(True), "rejection"])),
        input_checksums={name: sha256(folder / name) for name in
                         ("config.json", "stations.csv", "terrain.npz", "regimes_progress.json", "observations_progress.json")},
        model_checksums={name+".joblib": sha256(output / f"{name}.joblib") for name in models})
    write_json(output / "report.json", report)
    print(json.dumps({name: values["test"] for name, values in report["variants"].items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs" / "valley_regime_pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "valley_regime_pilot_results")
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.trees < 1 or not 1 <= args.jobs <= 8:
        parser.error("Positive trees and1–8jobs required")
    run(args)


if __name__ == "__main__":
    main()
