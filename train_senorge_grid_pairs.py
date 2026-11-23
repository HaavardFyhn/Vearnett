"""Stasjonsuavhengig seNorge-modell med potensielle terrengbasseng for begge ruter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import geopandas as gpd
import joblib
from rasterio.crs import CRS as RasterCRS
from scipy.spatial import cKDTree
from shapely import box
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from download_nve_grid import write_json
from pipeline_config import PAIR_FEATURES, RANDOM_STATE, WORK
from senorge_short_range_pilot import GRID_FIELDS, TERRAIN_FIELDS


LAKE_FIELDS = ["basin_present", "basin_count", "basin_max_depth_m", "basin_volume_m3"]
LAKE_PAIR_FEATURES = [f"{field}_{stat}" for field in LAKE_FIELDS for stat in ("min", "max", "diff")]
FEATURES = PAIR_FEATURES + LAKE_PAIR_FEATURES
DEFAULT_BASINS = WORK / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg"


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def attach_basins(grid: gpd.GeoDataFrame, basins: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Hele bassengvolumet summeres én gang per basseng med positivt arealoverlapp."""
    if grid.crs is None or basins.crs is None:
        raise ValueError("Rutenett og basseng må ha koordinatsystem")
    required = {"lake_id", "volume_m3", "max_depth_m"}
    if not required <= set(basins):
        raise ValueError("Basseng mangler lake_id, volume_m3 eller max_depth_m")
    if not basins.lake_id.is_unique or basins.empty or not basins.is_valid.all():
        raise ValueError("Basseng-ID må være unik og geometriene gyldige")
    if not np.isfinite(basins[["volume_m3", "max_depth_m"]]).all().all() or (
        basins[["volume_m3", "max_depth_m"]] < 0
    ).any().any():
        raise ValueError("Ugyldige bassengvolum/dybder")
    basins = basins.to_crs(grid.crs).reset_index(drop=True)
    cells = gpd.GeoDataFrame(
        geometry=box(grid.x.to_numpy() - 500, grid.y.to_numpy() - 500,
                     grid.x.to_numpy() + 500, grid.y.to_numpy() + 500), crs=grid.crs,
    )
    hits = gpd.sjoin(cells, basins[list(required) + ["geometry"]], how="inner", predicate="intersects")
    if len(hits):
        left = cells.geometry.iloc[hits.index.to_numpy()].reset_index(drop=True)
        right = basins.geometry.iloc[hits.index_right.to_numpy()].reset_index(drop=True)
        hits = hits.loc[(left.intersection(right).area > 0).to_numpy()]
    result = grid.copy()
    result["basin_count"] = 0
    result["basin_max_depth_m"] = 0.0
    result["basin_volume_m3"] = 0.0
    if len(hits):
        grouped = hits.groupby(level=0).agg(
            basin_count=("lake_id", "nunique"), basin_max_depth_m=("max_depth_m", "max"),
            basin_volume_m3=("volume_m3", "sum"),
        )
        result.loc[grouped.index, grouped.columns] = grouped
    result["basin_present"] = (result.basin_count > 0).astype("uint8")
    return result


def prepare_grid(grid_path: Path, basin_path: Path, archive: Path) -> gpd.GeoDataFrame:
    grid = gpd.read_file(grid_path)
    if grid.crs is None or grid.crs.to_epsg() != 25833:
        raise ValueError("Terrengrutenettet må bruke EPSG:25833")
    if not grid.grid_id.is_unique:
        raise ValueError("grid_id må være unik")
    original_count = len(grid)
    grid = grid.dropna(subset=GRID_FIELDS).reset_index(drop=True)
    with netCDF4.Dataset(archive) as ds:
        xs, ys = np.asarray(ds["X"][:]), np.asarray(ds["Y"][:])
        target = CRS.from_user_input(ds[ds["tg"].grid_mapping].proj4)
        x, y = Transformer.from_crs(grid.crs, target, always_xy=True).transform(grid.x.to_numpy(), grid.y.to_numpy())
        # Containing source cell, not bankers' rounding at half-pixel offsets.
        col = np.floor((x - (xs[0] - 500)) / 1000).astype(int)
        row = np.floor(((ys[0] + 500) - y) / 1000).astype(int)
        keep = (col >= 0) & (col < len(xs)) & (row >= 0) & (row < len(ys))
        keep[keep] &= ~np.ma.getmaskarray(np.ma.masked_invalid(ds["tg"][0]))[row[keep], col[keep]]
        grid["nc_row"], grid["nc_col"] = row, col
        grid = grid.loc[keep].drop_duplicates(["nc_row", "nc_col"]).reset_index(drop=True)
    print(f"Ruter: {len(grid):,}; utelatt {original_count-len(grid):,} med manglende data/utenfor/duplikat.", flush=True)
    basins = gpd.read_file(basin_path, columns=["lake_id", "volume_m3", "max_depth_m", "geometry"])
    grid = attach_basins(grid, basins)
    print(f"Bassengkobling: {int(grid.basin_present.sum()):,} ruter overlapper potensielle basseng.", flush=True)
    return grid


def sample_pairs(grid: pd.DataFrame, count: int, maximum: float = 50_000,
                 scale: float = 15_000, seed: int = RANDOM_STATE,
                 distribution: str = "gaussian-offset") -> np.ndarray:
    if count < 1 or len(grid) < 2 or maximum <= 1000 or scale <= 0:
        raise ValueError("Ugyldig antall rutepar eller avstandsgrenser")
    xy = grid[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(xy)
    if distribution not in ("gaussian-offset", "half-normal"):
        raise ValueError("Ukjent avstandsfordeling")
    rng = np.random.default_rng(seed)
    selected = np.empty((0, 2), dtype=np.int64)
    for _ in range(50):
        size = min(max((count - len(selected)) * 2, 1000), 2_000_000)
        first = rng.integers(len(grid), size=size)
        if distribution == "half-normal":
            radius = np.abs(rng.normal(0, scale, size=size))
            angle = rng.uniform(0, 2*np.pi, size=size)
            desired = xy[first] + radius[:, None]*np.column_stack([np.cos(angle), np.sin(angle)])
        else:
            desired = xy[first] + rng.normal(0, scale, size=(size, 2))
        snap_distance, second = tree.query(desired, workers=1)
        distance = np.linalg.norm(xy[first] - xy[second], axis=1)
        keep = (first != second) & (distance >= 1000 - 1e-5) & (distance <= maximum)
        if distribution == "half-normal":
            # Reject targets outside the covered 1 km grid instead of piling up on coastlines.
            keep &= snap_distance <= np.sqrt(2)*500 + 1e-5
        new = np.sort(np.column_stack([first[keep], second[keep]]), axis=1)
        selected = np.unique(np.concatenate([selected, new]), axis=0)
        if len(selected) >= count:
            return selected[rng.choice(len(selected), count, replace=False)]
    raise ValueError(f"Fant bare {len(selected):,} unike par; ønsket {count:,}")


def date_plan(years: list[int], date_step: int = 1, days_per_month: int = 0,
              temporal_split: str = "chronological", seed: int = RANDOM_STATE) -> pd.DataFrame:
    if not years or years != sorted(set(years)) or date_step < 1:
        raise ValueError("Ugyldige år eller tidssteg")
    if temporal_split not in ("chronological", "monthly-random"):
        raise ValueError("Ukjent tidsdeling")
    if days_per_month and (not 5 <= days_per_month <= 28 or date_step != 1):
        raise ValueError("Velg 5–28 dager per måned og tidssteg 1")
    if temporal_split == "monthly-random" and not days_per_month:
        raise ValueError("Tilfeldig månedsdeling krever dager per måned")
    records = []
    for year in years:
        calendar = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        for month in range(1, 13):
            indices = np.flatnonzero(calendar.month == month)
            rng = np.random.default_rng([seed, year, month])
            if days_per_month:
                indices = rng.choice(indices, days_per_month, replace=False)
            else:
                indices = indices[indices % date_step == 0]
            for j, index in enumerate(indices):
                records.append(dict(date=calendar[index].strftime("%Y-%m-%d"), year=year,
                                    day_index=int(index),
                                    partition=int(j >= int(len(indices)*.8))))
    plan = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if temporal_split == "chronological":
        plan["partition"] = (np.arange(len(plan)) >= int(len(plan)*.8)).astype(int)
    return plan


def pair_features(grid: pd.DataFrame, pairs: np.ndarray) -> pd.DataFrame:
    a, b = pairs.T
    x, y = grid.x.to_numpy(), grid.y.to_numpy()
    data = dict(
        distance_m=np.hypot(x[a]-x[b], y[a]-y[b]),
        elev_diff_m=np.abs(grid.elevation.to_numpy()[a]-grid.elevation.to_numpy()[b]),
        coast_fjord_diff_m=np.abs(grid.coast_dist_fjord.to_numpy()[a]-grid.coast_dist_fjord.to_numpy()[b]),
        coast_ocean_diff_m=np.abs(grid.coast_dist_ocean.to_numpy()[a]-grid.coast_dist_ocean.to_numpy()[b]),
        same_objtype=(grid.objtype.to_numpy()[a] == grid.objtype.to_numpy()[b]).astype(float),
        x_diff=np.abs(x[a]-x[b]), y_diff=np.abs(y[a]-y[b]),
    )
    for field in TERRAIN_FIELDS:
        values = grid[field].to_numpy(dtype=float)
        data[field.replace("_10km", "_diff_10km").replace("_25km", "_diff_25km")] = np.abs(values[a]-values[b])
    for field in LAKE_FIELDS:
        values = grid[field].to_numpy(dtype=float)
        data[field+"_min"] = np.minimum(values[a], values[b])
        data[field+"_max"] = np.maximum(values[a], values[b])
        data[field+"_diff"] = np.abs(values[a]-values[b])
    frame = pd.DataFrame(data)[FEATURES].astype("float32")
    if not np.isfinite(frame).all().all():
        raise ValueError("Ikke-endelige egenskaper i rutepar")
    return frame


def update_moments(stats: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(valid, x, 0).astype(float), np.where(valid, y, 0).astype(float)
    stats[0] += valid
    stats[1] += xx
    stats[2] += yy
    stats[3] += xx * xx
    stats[4] += yy * yy
    stats[5] += xx * yy


def ols_coefficients(stats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, sx, sy, sxx, _, sxy = stats
    safe = np.maximum(n, 1)
    variance = sxx-sx*sx/safe
    slope = np.divide(sxy-sx*sy/safe, variance, out=np.zeros_like(n), where=variance > 1e-10)
    return slope, sy/safe-slope*sx/safe


def regression_labels(train: np.ndarray, test: np.ndarray, minimum_train: int = 60,
                      minimum_test: int = 20) -> tuple[np.ndarray, np.ndarray]:
    def direction(t: np.ndarray, v: np.ndarray) -> np.ndarray:
        slope, intercept = ols_coefficients(t)
        m, vx, vy, vxx, vyy, vxy = v
        squared = vyy + slope*slope*vxx + m*intercept*intercept - 2*slope*vxy - 2*intercept*vy + 2*slope*intercept*vx
        if np.any(squared < -1e-5 * np.maximum(vyy, 1)):
            raise ValueError("Numerisk ugyldig OLS-feilsum")
        return np.sqrt(np.maximum(squared, 0)/np.maximum(m, 1))
    reverse = [0, 2, 1, 4, 3, 5]
    labels = (direction(train, test) + direction(train[reverse], test[reverse]))/2
    total = train + test
    n = np.maximum(total[0], 1)
    covariance = total[5] - total[1]*total[2]/n
    denominator = np.sqrt(np.maximum(total[3]-total[1]**2/n, 0)*np.maximum(total[4]-total[2]**2/n, 0))
    correlation = np.divide(covariance, denominator, out=np.full_like(n, np.nan), where=denominator > 1e-10)
    valid = (train[0] >= minimum_train) & (test[0] >= minimum_test)
    return np.where(valid, labels, np.nan), np.clip(correlation, -1, 1)


def symmetric_percentile_labels(forward_errors: np.ndarray, reverse_errors: np.ndarray,
                                quantile: float = .95) -> np.ndarray:
    forward_errors, reverse_errors = np.asarray(forward_errors), np.asarray(reverse_errors)
    if (forward_errors.ndim != 2 or reverse_errors.shape != forward_errors.shape or
            not 0 < quantile < 1 or np.isinf(forward_errors).any() or np.isinf(reverse_errors).any() or
            not np.isfinite(forward_errors).any(axis=0).all() or
            not np.isfinite(reverse_errors).any(axis=0).all()):
        raise ValueError("Invalid directional residuals for percentile labels")
    forward = np.nanquantile(np.abs(forward_errors), quantile, axis=0)
    reverse = np.nanquantile(np.abs(reverse_errors), quantile, axis=0)
    return (forward+reverse)/2


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def stream_percentile_labels(archives: list[Path], grid: pd.DataFrame, pairs: np.ndarray,
                             cache: Path, plan: pd.DataFrame, train_stats: np.ndarray,
                             usable: np.ndarray, variable: str, quantile: float = .95) -> np.ndarray:
    selected = plan.loc[plan.partition == 1]
    if selected.empty:
        raise ValueError("95th-percentile labels require held-out dates")
    label_cache = cache / "absolute_error_p95_v2"
    label_cache.mkdir(exist_ok=True)
    annual_cache = [cache / f"seNorge2018_{int(year)}.npz" for year in sorted(plan.year.unique())]
    identity = dict(schema=2, variable=variable, quantile=quantile,
                    pairs_sha256=array_sha256(pairs),
                    grid_indices_sha256=array_sha256(grid[["nc_row", "nc_col"]].to_numpy()),
                    plan_sha256=array_sha256(pd.util.hash_pandas_object(
                        plan[["date", "year", "day_index", "partition"]], index=False).to_numpy()),
                    archives=[fingerprint(path) for path in archives],
                    annual_moments=[fingerprint(path) for path in annual_cache])
    metadata_path = label_cache / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != identity:
            raise ValueError("95th-percentile label cache inputs changed; use a fresh label cache")
    else:
        write_json(metadata_path, identity)

    labels = np.full(len(pairs), np.nan, dtype=np.float64)
    eligible = np.flatnonzero(usable)
    rows, cols = grid.nc_row.to_numpy(), grid.nc_col.to_numpy()
    forward_slope, forward_intercept = ols_coefficients(train_stats)
    reverse_slope, reverse_intercept = ols_coefficients(train_stats[[0, 2, 1, 4, 3, 5]])
    pending_chunks = []
    missing_indices = []
    for start in range(0, len(eligible), 100_000):
        indices = eligible[start:start+100_000]
        stop = start+len(indices)
        chunk_path = label_cache / f"chunk_{start:07d}_{stop:07d}.npz"
        if chunk_path.exists():
            with np.load(chunk_path, allow_pickle=False) as saved:
                if (not np.array_equal(saved["rows"], indices) or
                        saved["quantile"].item() != quantile or
                        saved["target_error"].shape != (len(indices),)):
                    raise ValueError(f"Invalid cached percentile labels: {chunk_path.name}")
                chunk_labels = saved["target_error"]
            if not np.isfinite(chunk_labels).all() or (chunk_labels < 0).any():
                raise ValueError(f"Invalid cached percentile values: {chunk_path.name}")
            labels[indices] = chunk_labels
        else:
            column_start = sum(len(chunk) for chunk in missing_indices)
            missing_indices.append(indices)
            pending_chunks.append((chunk_path, indices, column_start))

    if not missing_indices:
        return labels

    missing = np.concatenate(missing_indices)
    forward_errors = np.full((len(selected), len(missing)), np.nan, dtype=np.float32)
    reverse_errors = np.full_like(forward_errors, np.nan)
    source_ids, target_ids = pairs[missing].T
    cursor = 0
    for path in sorted(archives, key=lambda item: int(item.stem.split("_")[-1])):
        year = int(path.stem.split("_")[-1])
        days = selected.loc[selected.year == year, "day_index"].to_numpy()
        with netCDF4.Dataset(path) as ds:
            for day in days:
                slab = np.ma.masked_invalid(ds[variable][day]).filled(np.nan)
                values = slab[rows, cols]
                source, target = values[source_ids], values[target_ids]
                valid = np.isfinite(source) & np.isfinite(target)
                forward = target-forward_slope[missing]*source-forward_intercept[missing]
                reverse = source-reverse_slope[missing]*target-reverse_intercept[missing]
                forward_errors[cursor] = np.where(valid, np.abs(forward), np.nan)
                reverse_errors[cursor] = np.where(valid, np.abs(reverse), np.nan)
                cursor += 1
    if cursor != len(selected):
        raise ValueError("Held-out residual dates do not match the date plan")

    for chunk_path, indices, column_start in pending_chunks:
        column_stop = column_start+len(indices)
        chunk_labels = symmetric_percentile_labels(
            forward_errors[:, column_start:column_stop],
            reverse_errors[:, column_start:column_stop], quantile)
        pending = chunk_path.with_suffix(".pending.npz")
        np.savez_compressed(pending, rows=indices, quantile=np.array(quantile),
                            target_error=chunk_labels)
        pending.replace(chunk_path)
        labels[indices] = chunk_labels
        print(f"95th-percentile residual labels: {min(column_stop, len(missing)):,}/{len(missing):,} pairs",
              flush=True)
    return labels


def stream_labels(archives: list[Path], grid: pd.DataFrame, pairs: np.ndarray, cache: Path,
                  date_step: int = 1, plan: pd.DataFrame | None = None,
                  variable: str = "tg", target_metric: str = "rmse") -> pd.DataFrame:
    if variable not in ("tg", "tn"):
        raise ValueError("Temperaturvariabel må være tg eller tn")
    if target_metric not in ("rmse", "absolute-error-p95"):
        raise ValueError("Ukjent mål for seNorge-par")
    dates = []
    if plan is None:
        plan = date_plan([int(p.stem.split("_")[-1]) for p in archives], date_step)
    plan_records = dict(dates=plan.to_dict("records"))
    if variable != "tg":
        plan_records["variable"] = variable
    plan_path = cache / "date_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan_records:
            raise ValueError("Datoutvalg er endret; bruk ny cache")
    elif any(cache.glob("seNorge2018_*.npz")):
        legacy = date_plan([int(p.stem.split("_")[-1]) for p in archives], date_step)
        if not plan.equals(legacy):
            raise ValueError("Eldre cache uten datoplan kan ikke brukes med nytt utvalg")
    write_json(plan_path, plan_records)
    reference_x = reference_y = None
    for path in archives:
        with netCDF4.Dataset(path) as ds:
            var = ds[variable]
            if var.units != "Celsius" or var.dimensions != ("time", "Y", "X"):
                raise ValueError(f"Uventet temperaturvariabel: {path}")
            if RasterCRS.from_string(ds[var.grid_mapping].proj4).to_epsg() != 32633:
                raise ValueError("Uventet projeksjon")
            xs, ys = np.asarray(ds["X"][:]), np.asarray(ds["Y"][:])
            if reference_x is None:
                reference_x, reference_y = xs, ys
            elif not (np.array_equal(xs, reference_x) and np.array_equal(ys, reference_y)):
                raise ValueError("Årsarkivene har forskjellige rutenett")
            t = ds["time"]
            decoded = netCDF4.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
            actual = pd.DatetimeIndex([pd.Timestamp(x.year, x.month, x.day) for x in decoded])
            year = int(path.stem.split("_")[-1])
            if not actual.equals(pd.date_range(f"{year}-01-01", f"{year}-12-31")):
                raise ValueError(f"Ufullstendig kalenderår: {path}")
            selected = plan.loc[plan.year == year]
            if not np.array_equal(actual[selected.day_index].strftime("%Y-%m-%d"), selected.date):
                raise ValueError("Datoplan stemmer ikke med arkivet")
            dates.extend(actual[selected.day_index])
    if dates != sorted(set(dates)):
        raise ValueError("Datoer må være unike og kronologiske")
    if len(dates) != len(plan):
        raise ValueError("Datoplanen inneholder år uten arkiv")
    split = int((plan.partition == 0).sum())
    totals = np.zeros((2, 6, len(pairs)), dtype=float)
    rows, cols = grid.nc_row.to_numpy(), grid.nc_col.to_numpy()
    a, b = pairs.T
    for path in archives:
        checkpoint = cache / f"{path.stem}.npz"
        with netCDF4.Dataset(path) as ds:
            selected = plan.loc[plan.year == int(path.stem.split("_")[-1])]
            indices = selected.day_index.to_numpy()
            if checkpoint.exists():
                with np.load(checkpoint) as stored:
                    yearly = stored["moments"]
                    if yearly.shape != totals.shape:
                        raise ValueError("Feil form på etikettcache")
                print(f"CACHE etiketter: {path.stem}", flush=True)
            else:
                yearly = np.zeros_like(totals)
                print(f"Etiketter: {path.stem}, {len(indices)} døgn, {len(pairs):,} par", flush=True)
                for day, partition in zip(indices, selected.partition):
                    slab = np.ma.masked_invalid(ds[variable][day]).filled(np.nan)
                    values = slab[rows, cols]
                    target = yearly[int(partition)]
                    for start in range(0, len(pairs), 100_000):
                        stop = start + 100_000
                        update_moments(target[:, start:stop], values[a[start:stop]], values[b[start:stop]])
                pending = checkpoint.with_suffix(".pending.npz")
                np.savez(pending, moments=yearly)
                pending.replace(checkpoint)
            totals += yearly
    labels, corr = regression_labels(totals[0], totals[1])
    required_train, required_test = np.ceil(split*.9), np.ceil((len(dates)-split)*.9)
    usable = (totals[0, 0] >= max(60, required_train)) & (totals[1, 0] >= max(20, required_test))
    if target_metric == "absolute-error-p95":
        labels = stream_percentile_labels(archives, grid, pairs, cache, plan, totals[0],
                                          usable, variable)
    return pd.DataFrame(dict(
        target_error=np.where(usable, labels, np.nan), correlation=corr,
        train_overlap=totals[0, 0].astype(int), test_overlap=totals[1, 0].astype(int),
    ))


def spatial_split(grid: pd.DataFrame, pairs: np.ndarray, seed: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.floor(grid[["x", "y"]].to_numpy()/100_000).astype(int)
    unique, inverse = np.unique(blocks, axis=0, return_inverse=True)
    if len(unique) < 5:
        raise ValueError("Trenger minst fem geografiske 100 km-blokker")
    heldout = np.random.default_rng(seed).choice(len(unique), max(1, int(np.ceil(len(unique)*.2))), replace=False)
    membership = np.isin(inverse, heldout)
    first, second = membership[pairs[:, 0]], membership[pairs[:, 1]]
    train, test = ~first & ~second, first & second
    if train.sum() < 20 or test.sum() < 20:
        raise ValueError("For få par i romlig trenings-/testutvalg")
    return train, test


def run(args: argparse.Namespace) -> dict:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    archives = []
    for year in args.years:
        path = args.archive_dir / f"seNorge2018_{year}.nc"
        reuse = getattr(args, "reuse_archive_dir", None)
        if not path.exists() and reuse is not None:
            path = reuse / path.name
        archives.append(path)
    plan = date_plan(args.years, args.date_step_days, getattr(args, "days_per_month", 0),
                     getattr(args, "temporal_split", "chronological"))
    maximum = getattr(args, "maximum_distance_m", 50_000)
    scale = getattr(args, "distance_scale_m", 15_000)
    distribution = getattr(args, "distance_distribution", "gaussian-offset")
    config = dict(schema=1, grid=fingerprint(args.grid), basins=fingerprint(args.basins),
                  archives=[fingerprint(p) for p in archives], pairs=args.pairs, years=args.years,
                  date_step=args.date_step_days, seed=RANDOM_STATE, maximum_distance_m=maximum,
                  distance_scale_m=scale, distance_distribution=distribution,
                  days_per_month=getattr(args, "days_per_month", 0),
                  temporal_split=getattr(args, "temporal_split", "chronological"),
                  features=FEATURES, target="symmetric OLS evaluation RMSE; " + getattr(args, "temporal_split", "chronological"),
                  lake_semantics="Individual lake_id basins with positive-area overlap: count, maximum whole-basin depth, sum of whole-basin volumes")
    config_path = output / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        comparison = config.copy()
        if "distance_distribution" not in existing and (
            distribution == "gaussian-offset" and scale == 15_000
            and not config["days_per_month"] and config["temporal_split"] == "chronological"
        ):
            for key in ("distance_scale_m", "distance_distribution", "days_per_month", "temporal_split"):
                comparison.pop(key)
            comparison["target"] = "symmetric chronological OLS validation RMSE"
        if existing != comparison:
            raise ValueError("Innstillinger/kilder er endret; bruk ny resultatmappe")
    if (output / "summary.json").exists():
        raise FileExistsError("Treningen er allerede fullført; eksisterende modell beholdes")
    write_json(config_path, config)
    grid_path = output / "grid_features.gpkg"
    if grid_path.exists():
        grid = gpd.read_file(grid_path)
    else:
        grid = prepare_grid(args.grid, args.basins, archives[0])
        pending = output / "grid_features.pending.gpkg"
        grid.to_file(pending, layer="grid_features", driver="GPKG")
        pending.replace(grid_path)
    pair_path = cache / "pairs.npy"
    if pair_path.exists():
        pairs = np.load(pair_path)
    else:
        print(f"Trekker {args.pairs:,} unike rutepar uten stasjonsutvalg.", flush=True)
        pairs = sample_pairs(grid, args.pairs, maximum=maximum, scale=scale, distribution=distribution)
        np.save(pair_path, pairs)
    plan.to_csv(output / "selected_dates.csv", index=False)
    distance = pair_features(grid, pairs)[["distance_m"]].to_numpy().ravel()
    counts, edges = np.histogram(distance, bins=[0, 1000, 5000, 10000, 20000, 30000,
                                               50000, 75000, 90000, 100000, np.inf])
    sampling = dict(pairs=len(pairs), distribution=distribution, scale_m=scale, maximum_m=maximum,
                    distance_quantiles_m=dict(zip(["min", "p50", "p90", "p95", "p99", "max"],
                                                 np.quantile(distance, [0, .5, .9, .95, .99, 1]).tolist())),
                    distance_bins=[dict(lower_m=float(lo), upper_m=None if np.isinf(hi) else float(hi),
                                        count=int(n)) for lo, hi, n in zip(edges[:-1], edges[1:], counts)],
                    sampled_days=len(plan), fitting_days=int((plan.partition == 0).sum()),
                    evaluation_days=int((plan.partition == 1).sum()), temporal_split=config["temporal_split"])
    write_json(output / "sampling.json", sampling)
    labels = stream_labels(archives, grid, pairs, cache, args.date_step_days, plan)
    accepted = labels.target_error.notna().to_numpy()
    print(f"Gyldige par: {accepted.sum():,} / {len(pairs):,}", flush=True)
    kept_pairs, labels = pairs[accepted], labels.loc[accepted].reset_index(drop=True)
    x = pair_features(grid, kept_pairs)
    y = labels.target_error.to_numpy()
    train, test = spatial_split(grid, kept_pairs)
    if getattr(args, "prepare_only", False):
        prepared = dict(status="prepared", accepted_pairs=len(kept_pairs), sampling=sampling,
                        spatial_train=int(train.sum()), spatial_test=int(test.sum()))
        write_json(output / "dataset_prepared.json", prepared)
        print(json.dumps(prepared, indent=2), flush=True)
        return prepared
    options = dict(n_estimators=args.trees, min_samples_leaf=8, max_features=1.0,
                   n_jobs=args.jobs, random_state=RANDOM_STATE)
    print(f"Romlig validering: {train.sum():,} treningspar / {test.sum():,} testpar.", flush=True)
    baseline = ExtraTreesRegressor(**options).fit(x.loc[train, PAIR_FEATURES].to_numpy(), y[train])
    full = ExtraTreesRegressor(**options).fit(x.loc[train].to_numpy(), y[train])
    def metrics(model, columns):
        predicted = model.predict(x.loc[test, columns].to_numpy())
        return dict(mae=float(mean_absolute_error(y[test], predicted)),
                    rmse=float(np.sqrt(mean_squared_error(y[test], predicted))),
                    r2=float(r2_score(y[test], predicted)))
    validation = dict(without_basins=metrics(baseline, PAIR_FEATURES), with_basins=metrics(full, FEATURES))
    heldout_records = labels.loc[test].copy()
    heldout_records["grid_id_1"] = grid.grid_id.to_numpy()[kept_pairs[test, 0]]
    heldout_records["grid_id_2"] = grid.grid_id.to_numpy()[kept_pairs[test, 1]]
    heldout_records["predicted_without_basins"] = baseline.predict(x.loc[test, PAIR_FEATURES].to_numpy())
    heldout_records["predicted_with_basins"] = full.predict(x.loc[test].to_numpy())
    heldout_records.to_csv(output / "spatial_holdout_predictions.csv", index=False)
    importance = dict(zip(FEATURES, full.feature_importances_.tolist()))
    del baseline, full
    print("Tilpasser sluttmodell til alle gyldige par.", flush=True)
    model = ExtraTreesRegressor(**options).fit(x.to_numpy(), y)
    pending_model = output / "model.pending.joblib"
    joblib.dump(model, pending_model)
    pending_model.replace(output / "model.joblib")
    records = labels.copy()
    records["grid_id_1"] = grid.grid_id.to_numpy()[kept_pairs[:, 0]]
    records["grid_id_2"] = grid.grid_id.to_numpy()[kept_pairs[:, 1]]
    records["spatial_partition"] = np.where(test, "test", np.where(train, "train", "excluded_cross_boundary"))
    records.to_csv(output / "pair_labels.csv", index=False)
    test_records = records.loc[test].copy()
    test_records["predicted_rmse_final_model"] = model.predict(x.loc[test].to_numpy())
    # Final refit includes these cells; its predictions are NOT holdout metrics.
    test_records.to_csv(output / "final_refit_example_predictions.csv", index=False)
    summary = dict(
        status="complete", training_source="seNorge_2018 only; no Frost training pairs or station sampling",
        features=FEATURES, requested_pairs=args.pairs, accepted_pairs=int(accepted.sum()),
        excluded_pairs=int((~accepted).sum()), years=args.years, variable="tg",
        spatial_train_pairs=int(train.sum()), spatial_test_pairs=int(test.sum()),
        excluded_cross_boundary_pairs=int((~train & ~test).sum()), validation=validation,
        feature_importance=importance, trees=args.trees, jobs=args.jobs,
        target=config["target"], sampling=sampling, maximum_distance_m=maximum,
        validation_method="20% of occupied 100 km blocks held out; neither endpoint of any training pair in a held-out block. Cross-boundary pairs excluded from validation, included in final refit.",
        basin_semantics=config["lake_semantics"],
        limitations=["Gridded interpolation similarity, not independent station accuracy.",
                     "Potential basins are terrain depressions, not observed water surfaces.",
                     "Whole basin volume can occur in multiple cells; never sum it over the grid as unique national volume.",
                     "Basin influence may be smoothed out by 1 km seNorge; added features do not prove a physical effect.",
                     f"Pair distances trained only from 1 to {maximum/1000:g} km; longer distances not supported.",
                     "Existing Station Lab models unchanged; use the companion prediction CLI for this separate model."],
    )
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=WORK / "norway_grid_1000m.gpkg")
    parser.add_argument("--basins", type=Path, default=DEFAULT_BASINS)
    parser.add_argument("--archive-dir", type=Path, default=WORK / "senorge_norway_2021_2025" / "netcdf")
    parser.add_argument("--output-dir", type=Path, default=WORK / "senorge_grid_pairs_1m_basins")
    parser.add_argument("--pairs", type=int, default=1_000_000)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2021, 2026)))
    parser.add_argument("--date-step-days", type=int, default=1)
    parser.add_argument("--reuse-archive-dir", type=Path)
    parser.add_argument("--days-per-month", type=int, default=0)
    parser.add_argument("--temporal-split", choices=["chronological", "monthly-random"], default="chronological")
    parser.add_argument("--distance-distribution", choices=["gaussian-offset", "half-normal"], default="gaussian-offset")
    parser.add_argument("--distance-scale-m", type=float, default=15_000)
    parser.add_argument("--maximum-distance-m", type=float, default=50_000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.pairs < 100 or args.date_step_days < 1 or args.trees < 1 or args.jobs < 1:
        parser.error("Minst 100 par; tidssteg, trær og arbeidere må være positive")
    if args.years != sorted(set(args.years)):
        parser.error("År må være unike og stigende")
    date_plan(args.years, args.date_step_days, args.days_per_month, args.temporal_split)
    if args.maximum_distance_m <= 1000 or args.maximum_distance_m > 100_000 or args.distance_scale_m <= 0:
        parser.error("Maksavstand må være over 1 km og høyst 100 km; skala må være positiv")
    from precompute_frost import computation_lock
    with computation_lock(args.output_dir):
        run(args)


if __name__ == "__main__":
    main()
