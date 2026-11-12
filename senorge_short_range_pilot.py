"""Test whether seNorge local virtual pairs improve Frost pair-error prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import joblib
import netCDF4
import numpy as np
import pandas as pd
import pyogrio
import pyproj
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_config import PAIR_FEATURES, RANDOM_STATE, WORK
from temperature_corridor_pilot import sample_pairs_by_distance, spatial_holdout_indices


DEFAULT_ARCHIVE = WORK / "senorge_norway_2021_2025" / "netcdf"
DEFAULT_GRID = WORK / "norway_grid_1000m.gpkg"
DEFAULT_FROST_PAIRS = WORK / "temperature_5y_prediction_rmse_pairs.csv"
DEFAULT_FROST_STATIONS = WORK / "temperature_5y_prediction_rmse_station_features.gpkg"
DEFAULT_OUTPUT = WORK / "senorge_short_range_temperature_pilot"
STATIONLAB_PREFIX = WORK / "temperature_5y_prediction_rmse_senorge"
MAX_STATIONLAB_FROST_PAIRS = 50_000
TERRAIN_FIELDS = [
    "relief_10km", "rugstd_10km", "tpi_10km",
    "relief_25km", "rugstd_25km", "tpi_25km",
]
GRID_FIELDS = [
    "x", "y", "elevation", "coast_dist_fjord", "coast_dist_ocean",
    "objtype", *TERRAIN_FIELDS,
]


def sample_virtual_pairs(grid, max_pairs=10_000, min_distance_m=1_000.0,
                         max_distance_m=50_000.0, distance_scale_m=15_000.0,
                         random_state=RANDOM_STATE):
    """Draw unique local grid-cell pairs, favoring short distances."""
    if max_pairs < 1 or min_distance_m < 0 or max_distance_m <= min_distance_m:
        raise ValueError("Invalid pair count or distance range")
    if distance_scale_m <= 0:
        raise ValueError("distance_scale_m must be positive")
    grid = grid.dropna(subset=GRID_FIELDS).reset_index(drop=True)
    coordinates = grid[["x", "y"]].to_numpy(dtype=np.float64)
    if len(grid) < 2:
        raise ValueError("Need at least two valid terrain-grid cells")
    tree = cKDTree(coordinates)
    rng = np.random.default_rng(random_state)
    selected = []
    seen = set()
    attempts = 0
    maximum_attempts = max_pairs * 20
    while len(selected) < max_pairs and attempts < maximum_attempts:
        source = int(rng.integers(len(grid)))
        neighbors = np.asarray(tree.query_ball_point(coordinates[source], max_distance_m), dtype=np.int64)
        neighbors = neighbors[neighbors != source]
        if not len(neighbors):
            attempts += 1
            continue
        distances = np.linalg.norm(coordinates[neighbors] - coordinates[source], axis=1)
        keep = (distances >= min_distance_m) & (distances <= max_distance_m)
        neighbors, distances = neighbors[keep], distances[keep]
        if not len(neighbors):
            attempts += 1
            continue
        weights = np.exp(-0.5 * np.square(distances / distance_scale_m))
        weights /= weights.sum()
        target = int(rng.choice(neighbors, p=weights))
        key = (min(source, target), max(source, target))
        if key not in seen:
            seen.add(key)
            selected.append(key)
        attempts += 1
    if len(selected) < max_pairs:
        raise ValueError(f"Only sampled {len(selected):,} of {max_pairs:,} unique local pairs")

    cells = np.asarray(selected, dtype=np.int64)
    first, second = cells[:, 0], cells[:, 1]
    x = grid.x.to_numpy(dtype=np.float64)
    y = grid.y.to_numpy(dtype=np.float64)
    elevation = grid.elevation.to_numpy(dtype=np.float64)
    fjord = grid.coast_dist_fjord.to_numpy(dtype=np.float64)
    ocean = grid.coast_dist_ocean.to_numpy(dtype=np.float64)
    objtype = grid.objtype.astype(str).to_numpy()
    pair = pd.DataFrame({
        "station_1": [f"GRID{index}" for index in first],
        "station_2": [f"GRID{index}" for index in second],
        "cell_1": first,
        "cell_2": second,
        "distance_m": np.hypot(x[first] - x[second], y[first] - y[second]),
        "elev_diff_m": np.abs(elevation[first] - elevation[second]),
        "coast_fjord_diff_m": np.abs(fjord[first] - fjord[second]),
        "coast_ocean_diff_m": np.abs(ocean[first] - ocean[second]),
        "same_objtype": (objtype[first] == objtype[second]).astype(np.int8),
        "x_diff": np.abs(x[first] - x[second]),
        "y_diff": np.abs(y[first] - y[second]),
    })
    for field in TERRAIN_FIELDS:
        pair[field.replace("_10km", "_diff_10km").replace("_25km", "_diff_25km")] = np.abs(
            grid[field].to_numpy(dtype=np.float64)[first]
            - grid[field].to_numpy(dtype=np.float64)[second]
        )
    return pair


def _netcdf_cell_indices(grid, dataset):
    x_values = np.asarray(dataset.variables["X"][:], dtype=np.float64)
    y_values = np.asarray(dataset.variables["Y"][:], dtype=np.float64)
    mapping = dataset.variables[dataset.variables["tg"].grid_mapping]
    target_crs = pyproj.CRS.from_proj4(mapping.proj4)
    transformer = pyproj.Transformer.from_crs(grid.crs, target_crs, always_xy=True)
    x, y = transformer.transform(grid.x.to_numpy(dtype=np.float64),
                                  grid.y.to_numpy(dtype=np.float64))
    ix = np.rint((x - x_values[0]) / (x_values[1] - x_values[0])).astype(np.int32)
    iy = np.rint((y - y_values[0]) / (y_values[1] - y_values[0])).astype(np.int32)
    inside = (ix >= 0) & (ix < len(x_values)) & (iy >= 0) & (iy < len(y_values))
    return ix, iy, inside


def _pair_regression_rmse(series, train_fraction):
    """Average directional validation RMSE using the station-pair OLS convention."""
    split = int(len(series) * train_fraction)
    if split < 2 or len(series) - split < 2:
        raise ValueError("Not enough sampled seNorge days for train/validation split")

    def directional(source_train, target_train, source_valid, target_valid):
        train_mask = np.isfinite(source_train) & np.isfinite(target_train)
        train_count = train_mask.sum(axis=0).astype(np.float64)
        safe_count = np.maximum(train_count, 1.0)
        x_train = np.where(train_mask, source_train, 0.0).astype(np.float64)
        y_train = np.where(train_mask, target_train, 0.0).astype(np.float64)
        mean_x = x_train.sum(axis=0) / safe_count
        mean_y = y_train.sum(axis=0) / safe_count
        variance_x = (x_train * x_train).sum(axis=0) - train_count * mean_x * mean_x
        covariance = (x_train * y_train).sum(axis=0) - train_count * mean_x * mean_y
        slope = np.divide(covariance, variance_x, out=np.zeros_like(covariance),
                          where=variance_x > 1e-10)
        intercept = mean_y - slope * mean_x

        valid_mask = np.isfinite(source_valid) & np.isfinite(target_valid)
        valid_count = valid_mask.sum(axis=0).astype(np.float64)
        x_valid = np.where(valid_mask, source_valid, 0.0).astype(np.float64)
        y_valid = np.where(valid_mask, target_valid, 0.0).astype(np.float64)
        residual = y_valid - (intercept[None, :] + slope[None, :] * x_valid)
        squared_error = np.where(valid_mask, residual * residual, 0.0).sum(axis=0)
        rmse = np.sqrt(squared_error / np.maximum(valid_count, 1.0))
        rmse[(train_count < 2) | (valid_count < 2)] = np.nan
        return rmse

    forward = directional(series[:split, :, 0], series[:split, :, 1],
                          series[split:, :, 0], series[split:, :, 1])
    reverse = directional(series[:split, :, 1], series[:split, :, 0],
                          series[split:, :, 1], series[split:, :, 0])
    return (forward + reverse) / 2.0, split


def sample_senorge_pair_errors(archive_dir, grid, pairs, years=(2021, 2022, 2023),
                               variable="tg", date_step_days=3, train_fraction=0.8):
    if date_step_days < 1:
        raise ValueError("date_step_days must be at least one")
    archive_dir = Path(archive_dir)
    archive_paths = [archive_dir / f"seNorge2018_{year}.nc" for year in years]
    missing = [path for path in archive_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing seNorge annual files: {missing}")

    with netCDF4.Dataset(archive_paths[0]) as first_dataset:
        ix, iy, inside = _netcdf_cell_indices(grid, first_dataset)
        units = str(first_dataset.variables[variable].units)
    pair_cells = pairs[["cell_1", "cell_2"]].to_numpy(dtype=np.int64)
    pair_inside = inside[pair_cells].all(axis=1)
    pairs = pairs.loc[pair_inside].reset_index(drop=True)
    pair_cells = pair_cells[pair_inside]
    if not len(pairs):
        raise ValueError("No virtual-pair terrain cells overlap the seNorge grid")
    source_x, target_x = ix[pair_cells[:, 0]], ix[pair_cells[:, 1]]
    source_y, target_y = iy[pair_cells[:, 0]], iy[pair_cells[:, 1]]

    records = []
    for archive_path in archive_paths:
        with netCDF4.Dataset(archive_path) as dataset:
            variable_data = dataset.variables[variable]
            for time_index in range(0, len(dataset.dimensions["time"]), date_step_days):
                records.append((archive_path, time_index))
    pair_series = np.full((len(records), len(pairs), 2), np.nan, dtype=np.float32)
    cursor = 0
    for archive_path in archive_paths:
        with netCDF4.Dataset(archive_path) as dataset:
            variable_data = dataset.variables[variable]
            for time_index in range(0, len(dataset.dimensions["time"]), date_step_days):
                slab = np.ma.asarray(variable_data[time_index], dtype=np.float32).filled(np.nan)
                pair_series[cursor, :, 0] = slab[source_y, source_x]
                pair_series[cursor, :, 1] = slab[target_y, target_x]
                cursor += 1

    labels, split = _pair_regression_rmse(pair_series, train_fraction)
    result = pairs.copy()
    result["target_error"] = labels
    result["n_overlap"] = len(records) - split
    result["source"] = "seNorge_2018"
    result["source_variable"] = variable
    result["source_units"] = units
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_error"]).reset_index(drop=True)
    if result.empty:
        raise ValueError("No valid seNorge virtual-pair labels were produced")
    return result, dict(
        years=list(years), sampled_days=len(records), train_days=split,
        validation_days=len(records) - split, variable=variable, units=units,
        date_step_days=date_step_days, valid_pairs=len(result),
    )


def _metrics(actual, prediction):
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)),
    }


def _distance_band_metrics(distances, actual, prediction):
    bands = {
        "1-10km": (distances >= 1_000) & (distances <= 10_000),
        "10-25km": (distances > 10_000) & (distances <= 25_000),
        "25-50km": (distances > 25_000) & (distances <= 50_000),
        "50-100km": (distances > 50_000) & (distances <= 100_000),
        "over-100km": distances > 100_000,
    }
    result = {}
    for name, mask in bands.items():
        if mask.any():
            result[name] = {"pairs": int(mask.sum()), **_metrics(actual[mask], prediction[mask])}
    return result


def _exclude_heldout_blocks(grid_pairs, grid, station_features, heldout_stations,
                            block_size_m=100_000):
    heldout = station_features.loc[list(heldout_stations)]
    heldout_blocks = set(zip(
        np.floor(heldout.x.to_numpy() / block_size_m).astype(np.int64),
        np.floor(heldout.y.to_numpy() / block_size_m).astype(np.int64),
    ))
    x_blocks = np.floor(grid.x.to_numpy() / block_size_m).astype(np.int64)
    y_blocks = np.floor(grid.y.to_numpy() / block_size_m).astype(np.int64)
    first = grid_pairs.cell_1.to_numpy(dtype=np.int64)
    second = grid_pairs.cell_2.to_numpy(dtype=np.int64)
    keep = np.ones(len(grid_pairs), dtype=bool)
    for block_x, block_y in heldout_blocks:
        first_in = (x_blocks[first] == block_x) & (y_blocks[first] == block_y)
        second_in = (x_blocks[second] == block_x) & (y_blocks[second] == block_y)
        keep &= ~(first_in | second_in)
    return grid_pairs.loc[keep].reset_index(drop=True), len(heldout_blocks)


def export_stationlab_profile(virtual_pairs, frost_pairs, frost_station_path,
                              prefix=STATIONLAB_PREFIX):
    """Write a separately selectable Station Lab model using the existing feature schema."""
    frost_train = sample_pairs_by_distance(
        frost_pairs, max_pairs=MAX_STATIONLAB_FROST_PAIRS, distance_scale_km=300.0,
    )
    virtual_train = virtual_pairs
    if len(virtual_train) > MAX_STATIONLAB_FROST_PAIRS:
        virtual_train = virtual_train.sample(
            n=MAX_STATIONLAB_FROST_PAIRS, random_state=RANDOM_STATE,
        )
    training_features = pd.concat(
        [frost_train[PAIR_FEATURES], virtual_train[PAIR_FEATURES]], ignore_index=True,
    )
    training_labels = np.concatenate([
        frost_train.target_error.to_numpy(dtype=np.float32),
        virtual_train.target_error.to_numpy(dtype=np.float32),
    ])
    virtual_weight = len(frost_train) / len(virtual_train)
    sample_weight = np.concatenate([
        np.ones(len(frost_train), dtype=np.float32),
        np.full(len(virtual_train), virtual_weight, dtype=np.float32),
    ])
    model = ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=8, max_features=1.0,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(training_features, training_labels, sample_weight=sample_weight)

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = prefix.with_name(prefix.name + "_model.joblib")
    features_path = prefix.with_name(prefix.name + "_station_features.gpkg")
    metadata_path = prefix.with_name(prefix.name + "_metadata.json")
    pending_model = model_path.with_name(model_path.stem + ".pending.joblib")
    pending_features = features_path.with_name(features_path.stem + ".pending.gpkg")
    pending_metadata = metadata_path.with_name(metadata_path.stem + ".pending.json")
    try:
        joblib.dump(model, pending_model)
        station_features = gpd.read_file(frost_station_path)
        pyogrio.write_dataframe(station_features, pending_features, driver="GPKG")
        metadata = {
            "target_type": "prediction_rmse",
            "features": PAIR_FEATURES,
            "train_fraction": 0.8,
            "min_valid_days": 20,
            "min_overlap_days": 60,
            "years": 5,
            "phenomenon": "temperature",
            "precip_wet_threshold_mm": None,
            "snow_present_threshold_cm": None,
            "temperature_class_limits": None,
            "profile_id": "temperature:senorge",
            "profile_label": "Temperatur (seNorge-pilot)",
            "training_sources": ["Frost station-pair errors", "seNorge_2018 1 km virtual pairs"],
            "senorge_teacher_years": [2021, 2022, 2023],
            "senorge_virtual_pairs": len(virtual_train),
            "frost_training_pairs": len(frost_train),
            "experimental": True,
        }
        pending_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        pending_model.replace(model_path)
        pending_features.replace(features_path)
        pending_metadata.replace(metadata_path)
    finally:
        for path in (pending_model, pending_features, pending_metadata):
            path.unlink(missing_ok=True)
    return dict(model=str(model_path), features=str(features_path), metadata=str(metadata_path),
                frost_training_pairs=len(frost_train), senorge_virtual_pairs=len(virtual_train))


def run_pilot(archive_dir=DEFAULT_ARCHIVE, grid_path=DEFAULT_GRID,
              frost_pairs_path=DEFAULT_FROST_PAIRS,
              frost_stations_path=DEFAULT_FROST_STATIONS, output_dir=DEFAULT_OUTPUT,
              max_grid_pairs=5_000, max_frost_pairs=5_000,
              min_distance_m=1_000.0, max_distance_m=50_000.0,
              distance_scale_m=15_000.0, date_step_days=12, years=(2021, 2022, 2023),
              spatial_repeats=3, export_stationlab=False):
    archive_dir, grid_path, frost_pairs_path, frost_stations_path, output_dir = map(
        Path, (archive_dir, grid_path, frost_pairs_path, frost_stations_path, output_dir)
    )
    grid = gpd.read_file(grid_path)
    if grid.crs is None or grid.crs.to_epsg() != 25833:
        raise ValueError("The terrain grid must use EPSG:25833")
    grid = grid.dropna(subset=GRID_FIELDS).reset_index(drop=True)
    grid_pairs = sample_virtual_pairs(
        grid, max_grid_pairs, min_distance_m, max_distance_m, distance_scale_m,
    )
    grid_pairs, teacher_meta = sample_senorge_pair_errors(
        archive_dir, grid, grid_pairs, years=years, date_step_days=date_step_days,
    )

    station_features = gpd.read_file(frost_stations_path)[["station_id", "x", "y"]]
    station_features["station_id"] = station_features.station_id.astype(str)
    station_features = station_features.drop_duplicates("station_id").set_index("station_id")
    frost_pairs = pd.read_csv(frost_pairs_path, dtype={"station_1": "string", "station_2": "string"})
    frost_pairs = frost_pairs.dropna(subset=[*PAIR_FEATURES, "target_error"])
    frost_pairs = frost_pairs.loc[
        frost_pairs.station_1.isin(station_features.index)
        & frost_pairs.station_2.isin(station_features.index)
    ]
    frost_pool_size = len(frost_pairs)
    all_frost_pairs = frost_pairs.copy()
    frost_pairs = sample_pairs_by_distance(
        frost_pairs, max_frost_pairs, distance_scale_km=300.0,
    )
    if spatial_repeats < 1:
        raise ValueError("spatial_repeats must be at least one")
    model_options = dict(n_estimators=300, min_samples_leaf=8, max_features=1.0,
                         n_jobs=-1, random_state=RANDOM_STATE)
    folds = []
    augmented_importance = []
    frost_test_records = []
    for repeat in range(spatial_repeats):
        train_ids, test_ids, heldout_stations, heldout_block_count = spatial_holdout_indices(
            frost_pairs, station_features, random_state=RANDOM_STATE + repeat,
        )
        real_train = frost_pairs.iloc[train_ids]
        test = frost_pairs.iloc[test_ids]
        teacher_train, excluded_blocks = _exclude_heldout_blocks(
            grid_pairs, grid, station_features, heldout_stations,
        )
        if teacher_train.empty:
            raise ValueError("Spatial holdout removed every seNorge virtual pair")
        teacher_train = teacher_train.sample(
            n=min(len(teacher_train), len(real_train)),
            random_state=RANDOM_STATE + repeat,
        )

        baseline = ExtraTreesRegressor(**model_options)
        baseline.fit(real_train[PAIR_FEATURES], real_train.target_error)
        baseline_prediction = np.maximum(baseline.predict(test[PAIR_FEATURES]), 0)

        augmented = ExtraTreesRegressor(**model_options)
        augmented_features = pd.concat([real_train[PAIR_FEATURES], teacher_train[PAIR_FEATURES]],
                                       ignore_index=True)
        augmented_labels = np.concatenate([
            real_train.target_error.to_numpy(dtype=np.float32),
            teacher_train.target_error.to_numpy(dtype=np.float32),
        ])
        synthetic_weight = len(real_train) / len(teacher_train)
        sample_weight = np.concatenate([
            np.ones(len(real_train), dtype=np.float32),
            np.full(len(teacher_train), synthetic_weight, dtype=np.float32),
        ])
        augmented.fit(augmented_features, augmented_labels, sample_weight=sample_weight)
        augmented_prediction = np.maximum(augmented.predict(test[PAIR_FEATURES]), 0)

        actual = test.target_error.to_numpy(dtype=np.float32)
        test_distances = test.distance_m.to_numpy(dtype=np.float64)
        baseline_metrics = _metrics(actual, baseline_prediction)
        augmented_metrics = _metrics(actual, augmented_prediction)
        folds.append({
            "seed": RANDOM_STATE + repeat,
            "heldout_station_count": len(heldout_stations),
            "heldout_block_count": heldout_block_count,
            "excluded_teacher_block_count": excluded_blocks,
            "frost_train_pairs": len(real_train),
            "frost_test_pairs": len(test),
            "teacher_pairs_used": len(teacher_train),
            "baseline": baseline_metrics,
            "with_senorge_teacher": augmented_metrics,
            "baseline_by_distance": _distance_band_metrics(
                test_distances, actual, baseline_prediction,
            ),
            "with_senorge_teacher_by_distance": _distance_band_metrics(
                test_distances, actual, augmented_prediction,
            ),
            "mae_reduction_fraction": float(
                (baseline_metrics["mae"] - augmented_metrics["mae"])
                / max(baseline_metrics["mae"], 1e-12)
            ),
            "rmse_reduction_fraction": float(
                (baseline_metrics["rmse"] - augmented_metrics["rmse"])
                / max(baseline_metrics["rmse"], 1e-12)
            ),
        })
        if repeat == 0:
            frost_test_records.append(test)
        augmented_importance.append(augmented.feature_importances_)

    def mean_metrics(key):
        return {name: float(np.mean([fold[key][name] for fold in folds]))
                for name in ("mae", "rmse", "r2")}

    def std_metrics(key):
        return {name: float(np.std([fold[key][name] for fold in folds]))
                for name in ("mae", "rmse", "r2")}

    summary = {
        "experiment": "seNorge short-range virtual-pair augmentation for Frost pair-error model",
        "interpretation": "seNorge is a gridded teacher, not independent ground truth; Frost remains the test target.",
        "archive_dir": str(archive_dir),
        "terrain_grid": str(grid_path),
        "frost_pair_source": str(frost_pairs_path),
        "teacher": teacher_meta,
        "spatial_repeats": spatial_repeats,
        "virtual_pair_distance_m": {
            "minimum": min_distance_m,
            "maximum": max_distance_m,
            "half_normal_scale": distance_scale_m,
            "median_sampled": float(grid_pairs.distance_m.median()),
        },
        "virtual_pairs_sampled": len(grid_pairs),
        "frost_pair_pool": frost_pool_size,
        "frost_pairs_sampled": len(frost_pairs),
        "validation": "Repeated Frost pair errors on held-out 100 km station blocks; seNorge teacher uses 2021-2023 only and excludes each held-out block.",
        "baseline": mean_metrics("baseline"),
        "baseline_std": std_metrics("baseline"),
        "with_senorge_teacher": mean_metrics("with_senorge_teacher"),
        "with_senorge_teacher_std": std_metrics("with_senorge_teacher"),
        "mae_reduction_fraction": float(np.mean([fold["mae_reduction_fraction"] for fold in folds])),
        "rmse_reduction_fraction": float(np.mean([fold["rmse_reduction_fraction"] for fold in folds])),
        "folds": folds,
        "teacher_feature_importance": dict(zip(
            PAIR_FEATURES,
            np.mean(augmented_importance, axis=0).astype(float).tolist(),
        )),
    }
    if export_stationlab:
        summary["stationlab_profile"] = export_stationlab_profile(
            grid_pairs, all_frost_pairs, frost_stations_path,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    grid_pairs.to_csv(output_dir / "senorge_virtual_pairs.csv", index=False)
    frost_test_records[0].to_csv(output_dir / "frost_spatial_test_pairs_seed42.csv", index=False)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--frost-pairs", type=Path, default=DEFAULT_FROST_PAIRS)
    parser.add_argument("--frost-stations", type=Path, default=DEFAULT_FROST_STATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-grid-pairs", type=int, default=5_000)
    parser.add_argument("--max-frost-pairs", type=int, default=5_000)
    parser.add_argument("--min-distance-km", type=float, default=1.0)
    parser.add_argument("--max-distance-km", type=float, default=50.0)
    parser.add_argument("--distance-scale-km", type=float, default=15.0)
    parser.add_argument("--date-step-days", type=int, default=12,
                        help="Use every Nth seNorge day to keep the pilot tractable")
    parser.add_argument("--spatial-repeats", type=int, default=3)
    parser.add_argument("--export-stationlab-profile", action="store_true",
                        help="Write a separate experimental temperature:senorge profile for Station Lab")
    args = parser.parse_args(argv)
    if (args.max_grid_pairs < 1 or args.max_frost_pairs < 20
            or args.min_distance_km < 0 or args.max_distance_km <= args.min_distance_km
            or args.distance_scale_km <= 0 or args.date_step_days < 1
            or args.spatial_repeats < 1):
        parser.error("Invalid pair count, distance range, or date sampling interval")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_pilot(
        args.archive_dir, args.grid, args.frost_pairs, args.frost_stations,
        args.output_dir, args.max_grid_pairs, args.max_frost_pairs,
        args.min_distance_km * 1000, args.max_distance_km * 1000,
        args.distance_scale_km * 1000, args.date_step_days,
        spatial_repeats=args.spatial_repeats,
        export_stationlab=args.export_stationlab_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())