"""Compare temperature pair-error models with and without corridor terrain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_config import PAIR_FEATURES, RANDOM_STATE, WORK


DEFAULT_PAIRS = WORK / "temperature_5y_prediction_rmse_pairs.csv"
DEFAULT_STATIONS = WORK / "temperature_5y_prediction_rmse_station_features.gpkg"
DEFAULT_TERRAIN = WORK / "norway_grid_1000m.gpkg"
DEFAULT_DTM = Path(__file__).resolve().parent / "data" / "DTM10"
OUTPUT_DIR = WORK / "temperature_corridor_pilot"
CORRIDOR_FEATURES = [
    "corridor_ridge_above_line_m",
    "corridor_valley_below_line_m",
    "corridor_mean_abs_deviation_m",
    "corridor_coverage_fraction",
]


class DtmTileSampler:
    """Sample aligned-coordinate DTM10 tiles without mosaicking or resampling them."""

    def __init__(self, tile_paths):
        tile_paths = sorted(Path(path).resolve() for path in tile_paths)
        if not tile_paths:
            raise FileNotFoundError("No DTM10 tiles were found")
        self.datasets = []
        bounds = []
        try:
            for path in tile_paths:
                source = rasterio.open(path)
                params = {} if source.crs is None else source.crs.to_dict()
                supported = (params.get("proj") == "utm" and params.get("zone") == 33
                             and params.get("ellps") == "GRS80" and params.get("units") == "m"
                             and not params.get("south", False))
                if (not supported or source.count != 1 or source.dtypes[0] != "float32"
                        or not np.allclose(source.res, (10.0, 10.0), atol=1e-6)):
                    source.close()
                    raise ValueError(f"Expected single-band 10 m float32 UTM33 DTM tile: {path}")
                self.datasets.append(source)
                bounds.append(source.bounds)
        except BaseException:
            self.close()
            raise

        self.bounds = np.asarray([
            [bound.left, bound.bottom, bound.right, bound.top] for bound in bounds
        ], dtype=np.float64)
        centers = np.column_stack([
            (self.bounds[:, 0] + self.bounds[:, 2]) / 2,
            (self.bounds[:, 1] + self.bounds[:, 3]) / 2,
        ])
        self.tree = cKDTree(centers)

    def close(self):
        for source in getattr(self, "datasets", []):
            source.close()
        self.datasets = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def sample(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        neighbor_count = min(8, len(self.datasets))
        _, candidates = self.tree.query(points, k=neighbor_count)
        if neighbor_count == 1:
            candidates = candidates[:, None]
        matches = np.full(len(points), -1, dtype=np.int32)
        for column in range(neighbor_count):
            candidate = candidates[:, column]
            candidate_bounds = self.bounds[candidate]
            contained = (
                (points[:, 0] >= candidate_bounds[:, 0])
                & (points[:, 0] <= candidate_bounds[:, 2])
                & (points[:, 1] >= candidate_bounds[:, 1])
                & (points[:, 1] <= candidate_bounds[:, 3])
            )
            assign = (matches < 0) & contained
            matches[assign] = candidate[assign]

        values = np.full(len(points), np.nan, dtype=np.float32)
        for tile_index in np.unique(matches[matches >= 0]):
            point_indices = np.flatnonzero(matches == tile_index)
            sampled = self.datasets[tile_index].sample(points[point_indices], masked=True)
            values[point_indices] = [
                np.nan if np.ma.is_masked(value[0]) else float(value[0])
                for value in sampled
            ]
        return values


def corridor_profile_features(start, end, start_elevation, end_elevation,
                               terrain_tree, terrain_elevation, sample_step_m=2000.0,
                               max_snap_m=750.0):
    """Summarize contour-grid elevation along a straight station-to-station transect."""
    if sample_step_m <= 0 or max_snap_m <= 0:
        raise ValueError("sample_step_m and max_snap_m must be positive")
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    distance = float(np.hypot(*(end - start)))
    sample_count = max(1, int(np.ceil(distance / sample_step_m)) - 1)
    fractions = np.arange(1, sample_count + 1, dtype=np.float64) / (sample_count + 1)
    points = start[None, :] + fractions[:, None] * (end - start)[None, :]
    distances, indices = terrain_tree.query(points, distance_upper_bound=max_snap_m)
    valid = np.isfinite(distances) & (indices < len(terrain_elevation))
    coverage = float(valid.mean())
    if not valid.any():
        return dict(
            corridor_ridge_above_line_m=0.0,
            corridor_valley_below_line_m=0.0,
            corridor_mean_abs_deviation_m=0.0,
            corridor_coverage_fraction=coverage,
        )

    profile = terrain_elevation[indices[valid]].astype(np.float64)
    endpoint_line = start_elevation + fractions[valid] * (end_elevation - start_elevation)
    residual = profile - endpoint_line
    return dict(
        corridor_ridge_above_line_m=float(max(residual.max(), 0.0)),
        corridor_valley_below_line_m=float(max(-residual.min(), 0.0)),
        corridor_mean_abs_deviation_m=float(np.mean(np.abs(residual))),
        corridor_coverage_fraction=coverage,
    )


def raster_corridor_profile_features(start, end, start_elevation, end_elevation,
                                      terrain_sampler, sample_step_m=1000.0):
    """Summarize DTM elevation sampled along a station-to-station transect."""
    if sample_step_m <= 0:
        raise ValueError("sample_step_m must be positive")
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    distance = float(np.hypot(*(end - start)))
    sample_count = max(1, int(np.ceil(distance / sample_step_m)) - 1)
    fractions = np.arange(1, sample_count + 1, dtype=np.float64) / (sample_count + 1)
    points = start[None, :] + fractions[:, None] * (end - start)[None, :]
    profile = terrain_sampler.sample(points).astype(np.float64)
    valid = np.isfinite(profile)
    coverage = float(valid.mean())
    if not valid.any():
        return dict(
            corridor_ridge_above_line_m=0.0,
            corridor_valley_below_line_m=0.0,
            corridor_mean_abs_deviation_m=0.0,
            corridor_coverage_fraction=coverage,
        )
    residual = profile[valid] - (
        start_elevation + fractions[valid] * (end_elevation - start_elevation)
    )
    return dict(
        corridor_ridge_above_line_m=float(max(residual.max(), 0.0)),
        corridor_valley_below_line_m=float(max(-residual.min(), 0.0)),
        corridor_mean_abs_deviation_m=float(np.mean(np.abs(residual))),
        corridor_coverage_fraction=coverage,
    )


def _metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def sample_pairs_by_distance(pairs, max_pairs=10_000, distance_scale_km=300.0,
                             random_state=RANDOM_STATE):
    """Randomly sample pairs with half-normal weights favoring short distances."""
    if max_pairs < 0 or distance_scale_km <= 0:
        raise ValueError("max_pairs must be nonnegative and distance_scale_km positive")
    if max_pairs == 0 or len(pairs) <= max_pairs:
        return pairs.reset_index(drop=True).copy()
    distances = pd.to_numeric(pairs["distance_m"], errors="coerce").to_numpy(dtype=np.float64)
    scale_m = distance_scale_km * 1000.0
    weights = np.exp(-0.5 * np.square(distances / scale_m))
    weights[~np.isfinite(weights)] = 0.0
    if weights.sum() <= 0:
        raise ValueError("No valid station-pair distances are available for sampling")
    selected = pairs.sample(n=max_pairs, weights=weights, random_state=random_state)
    return selected.reset_index(drop=True)


def spatial_holdout_indices(pairs, stations, test_fraction=0.2,
                            block_size_m=100_000, random_state=RANDOM_STATE):
    """Hold out geographic station blocks and every pair touching those stations."""
    if not 0 < test_fraction < 1 or block_size_m <= 0:
        raise ValueError("test_fraction must be between 0 and 1 and block_size_m must be positive")
    coordinates = stations[["x", "y"]].copy()
    coordinates["block_x"] = np.floor(coordinates["x"] / block_size_m).astype(np.int64)
    coordinates["block_y"] = np.floor(coordinates["y"] / block_size_m).astype(np.int64)
    coordinates["block"] = list(zip(coordinates["block_x"], coordinates["block_y"]))
    block_sizes = coordinates.groupby("block", sort=False).size()
    block_order = np.random.default_rng(random_state).permutation(len(block_sizes))
    target_station_count = max(1, int(round(len(coordinates) * test_fraction)))
    heldout_blocks = set()
    heldout_count = 0
    blocks = block_sizes.index.to_list()
    for block_index in block_order:
        block = blocks[block_index]
        heldout_blocks.add(block)
        heldout_count += int(block_sizes.iloc[block_index])
        if heldout_count >= target_station_count:
            break

    heldout_stations = set(coordinates.index[coordinates["block"].isin(heldout_blocks)])
    first_heldout = pairs["station_1"].isin(heldout_stations).to_numpy()
    second_heldout = pairs["station_2"].isin(heldout_stations).to_numpy()
    train_indices = np.flatnonzero(~first_heldout & ~second_heldout)
    test_indices = np.flatnonzero(first_heldout | second_heldout)
    if not len(train_indices) or not len(test_indices):
        raise ValueError("Spatial holdout produced an empty train or test partition")
    return train_indices, test_indices, heldout_stations, len(heldout_blocks)


def evaluate_spatial_repeats(pairs, stations, spatial_repeats=3):
    if spatial_repeats < 1:
        raise ValueError("spatial_repeats must be at least one")
    corridor_columns = [*PAIR_FEATURES, *CORRIDOR_FEATURES]
    model_options = dict(n_estimators=300, min_samples_leaf=8, max_features=1.0,
                         n_jobs=-1, random_state=RANDOM_STATE)
    folds = []
    baseline_importance = []
    corridor_importance = []

    for repeat in range(spatial_repeats):
        seed = RANDOM_STATE + repeat
        train_ids, test_ids, heldout_stations, heldout_blocks = spatial_holdout_indices(
            pairs, stations, random_state=seed,
        )
        train, test = pairs.iloc[train_ids], pairs.iloc[test_ids]
        actual = test["target_error"].to_numpy(dtype=np.float32)

        baseline_model = ExtraTreesRegressor(**model_options)
        baseline_model.fit(train[PAIR_FEATURES], train["target_error"])
        baseline_pred = np.maximum(baseline_model.predict(test[PAIR_FEATURES]), 0)
        baseline_metrics = _metrics(actual, baseline_pred)
        baseline_importance.append(baseline_model.feature_importances_)

        corridor_model = ExtraTreesRegressor(**model_options)
        corridor_model.fit(train[corridor_columns], train["target_error"])
        corridor_pred = np.maximum(corridor_model.predict(test[corridor_columns]), 0)
        corridor_metrics = _metrics(actual, corridor_pred)
        corridor_importance.append(corridor_model.feature_importances_[-len(CORRIDOR_FEATURES):])

        folds.append({
            "seed": seed,
            "train_pairs": len(train_ids),
            "test_pairs": len(test_ids),
            "heldout_station_count": len(heldout_stations),
            "heldout_block_count": heldout_blocks,
            "baseline": baseline_metrics,
            "with_corridor": corridor_metrics,
            "mae_reduction_fraction": float(
                (baseline_metrics["mae"] - corridor_metrics["mae"])
                / max(baseline_metrics["mae"], 1e-12)
            ),
            "rmse_reduction_fraction": float(
                (baseline_metrics["rmse"] - corridor_metrics["rmse"])
                / max(baseline_metrics["rmse"], 1e-12)
            ),
        })

    def metric_means(key):
        return {name: float(np.mean([fold[key][name] for fold in folds]))
                for name in ("mae", "rmse", "r2")}

    def metric_stds(key):
        return {name: float(np.std([fold[key][name] for fold in folds]))
                for name in ("mae", "rmse", "r2")}

    return {
        "spatial_repeats": spatial_repeats,
        "folds": folds,
        "baseline": metric_means("baseline"),
        "baseline_std": metric_stds("baseline"),
        "with_corridor": metric_means("with_corridor"),
        "with_corridor_std": metric_stds("with_corridor"),
        "mae_reduction_fraction": float(np.mean([fold["mae_reduction_fraction"] for fold in folds])),
        "rmse_reduction_fraction": float(np.mean([fold["rmse_reduction_fraction"] for fold in folds])),
        "corridor_feature_importance": dict(zip(
            CORRIDOR_FEATURES,
            np.mean(corridor_importance, axis=0).astype(float).tolist(),
        )),
        "baseline_feature_importance": dict(zip(
            PAIR_FEATURES,
            np.mean(baseline_importance, axis=0).astype(float).tolist(),
        )),
    }


def run_pilot(pairs_path=DEFAULT_PAIRS, stations_path=DEFAULT_STATIONS,
              terrain_path=DEFAULT_TERRAIN, output_dir=OUTPUT_DIR,
              max_pairs=10_000, sample_step_m=2_000.0, dtm_dir=None,
              distance_scale_km=300.0,
              spatial_repeats=3):
    pairs_path, stations_path, terrain_path, output_dir = map(
        Path, (pairs_path, stations_path, terrain_path, output_dir)
    )
    pair_ids = {"station_1": "string", "station_2": "string"}
    pairs = pd.read_csv(pairs_path, dtype=pair_ids)
    required = [*PAIR_FEATURES, "target_error", "station_1", "station_2"]
    missing = sorted(set(required) - set(pairs.columns))
    if missing:
        raise ValueError(f"Temperature pair table is missing columns: {missing}")
    pairs = pairs.dropna(subset=required).copy()

    stations = pyogrio.read_dataframe(
        stations_path, columns=["station_id", "x", "y", "elevation"], read_geometry=False,
    )
    stations["station_id"] = stations["station_id"].astype(str)
    stations = stations.drop_duplicates("station_id").set_index("station_id")
    pairs = pairs.loc[
        pairs.station_1.isin(stations.index) & pairs.station_2.isin(stations.index)
    ]
    pair_pool_size = len(pairs)
    did_sample_pairs = max_pairs > 0 and pair_pool_size > max_pairs
    pairs = sample_pairs_by_distance(pairs, max_pairs, distance_scale_km)
    if len(pairs) < 20:
        raise ValueError(f"Too few usable temperature pairs for a pilot: {len(pairs)}")

    station_xy = stations[["x", "y"]].to_dict("index")
    station_elevation = stations["elevation"].to_dict()
    if dtm_dir is None:
        terrain = pyogrio.read_dataframe(
            terrain_path, columns=["x", "y", "elevation"], read_geometry=False,
        ).dropna(subset=["x", "y", "elevation"])
        terrain_xy = terrain[["x", "y"]].to_numpy(dtype=np.float64)
        terrain_elevation = terrain["elevation"].to_numpy(dtype=np.float32)
        terrain_tree = cKDTree(terrain_xy)
        profile = lambda first, second, station_1, station_2: corridor_profile_features(
            (first["x"], first["y"]), (second["x"], second["y"]),
            float(station_elevation[station_1]), float(station_elevation[station_2]),
            terrain_tree, terrain_elevation, sample_step_m=sample_step_m,
        )
        terrain_source = terrain_path
        terrain_resolution_m = 1000
        terrain_note = "Existing contour-derived 1 km grid elevations; screening pilot, not DTM10."
        tile_count = None
        raster = None
    else:
        tile_paths = sorted(Path(dtm_dir).rglob("*.tif"))
        terrain_sampler = DtmTileSampler(tile_paths)
        tile_count = len(tile_paths)
        terrain_source = dtm_dir
        terrain_resolution_m = 10
        terrain_note = "10 m DTM10 tiles sampled in each tile's native grid along straight station-pair transects."

        def profile(first, second, station_1, station_2):
            return raster_corridor_profile_features(
                (first["x"], first["y"]), (second["x"], second["y"]),
                float(station_elevation[station_1]), float(station_elevation[station_2]),
                terrain_sampler, sample_step_m=sample_step_m,
            )

    corridor_values = []
    try:
        for station_1, station_2 in pairs[["station_1", "station_2"]].itertuples(index=False, name=None):
            corridor_values.append(profile(station_xy[station_1], station_xy[station_2],
                                           station_1, station_2))
    finally:
        if dtm_dir is not None:
            terrain_sampler.close()
    pairs = pd.concat([pairs, pd.DataFrame(corridor_values)], axis=1)

    evaluation = evaluate_spatial_repeats(pairs, stations, spatial_repeats)
    summary = {
        "experiment": "temperature station-pair RMSE with corridor terrain",
        "pairs_source": str(pairs_path),
        "terrain_source": str(terrain_source),
        "terrain_resolution_m": terrain_resolution_m,
        "terrain_elevation_note": terrain_note,
        "dtm_tile_count": tile_count,
        "sample_step_m": sample_step_m,
        "max_pairs": max_pairs,
        "pair_pool_size": pair_pool_size,
        "pair_sampling": ("random without replacement; half-normal distance weights"
                  if did_sample_pairs else "all valid pairs; no pair sampling"),
        "distance_scale_km": distance_scale_km,
        "selected_distance_quantiles_km": {
            str(percentile): float(value)
            for percentile, value in zip(
                (10, 25, 50, 75, 90),
                np.percentile(pairs["distance_m"].to_numpy(dtype=float) / 1000, (10, 25, 50, 75, 90)),
            )
        },
        "pairs_used": len(pairs),
        "spatial_repeats": evaluation["spatial_repeats"],
        "split": "Each fold holds out 20% of occupied 100 km station blocks; training excludes held-out stations and test pairs include at least one.",
        "baseline": evaluation["baseline"],
        "baseline_std": evaluation["baseline_std"],
        "with_corridor": evaluation["with_corridor"],
        "with_corridor_std": evaluation["with_corridor_std"],
        "mae_reduction_fraction": evaluation["mae_reduction_fraction"],
        "rmse_reduction_fraction": evaluation["rmse_reduction_fraction"],
        "corridor_feature_importance": evaluation["corridor_feature_importance"],
        "baseline_feature_importance": evaluation["baseline_feature_importance"],
        "folds": evaluation["folds"],
        "corridor_features": CORRIDOR_FEATURES,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pairs.to_csv(output_dir / "sampled_pairs.csv", index=False)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--terrain", type=Path, default=DEFAULT_TERRAIN)
    parser.add_argument("--dtm-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-pairs", type=int, default=10_000,
                        help="Deterministic random sample size; use 0 for all valid pairs")
    parser.add_argument("--distance-scale-km", type=float, default=300.0,
                        help="Half-normal distance scale; smaller values favor nearby pairs more strongly")
    parser.add_argument("--sample-step-m", type=float, default=2_000.0)
    parser.add_argument("--spatial-repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if (args.max_pairs < 0 or args.distance_scale_km <= 0 or args.sample_step_m <= 0
            or args.spatial_repeats < 1):
        parser.error("--max-pairs and --spatial-repeats must be valid; distance scale and sample step must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = OUTPUT_DIR / "dtm10" if args.dtm_dir is not None else OUTPUT_DIR
    run_pilot(args.pairs, args.stations, args.terrain, output_dir,
              args.max_pairs, args.sample_step_m, dtm_dir=args.dtm_dir,
              distance_scale_km=args.distance_scale_km,
              spatial_repeats=args.spatial_repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())