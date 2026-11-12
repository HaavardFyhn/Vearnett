"""Classify mainland Norway's 2021-2025 weather, preserving levels and timing."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import netCDF4
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from download_nve_grid import NODATA, write_json


ROOT = Path(__file__).resolve().parent
SEED = 1729
MONTH_METRICS = (
    "temperature_mean", "precipitation_total", "temperature_std", "wet_fraction",
    "heavy_fraction", "precipitation_p95", "precipitation_max", "frost_fraction",
    "freeze_thaw_fraction", "diurnal_range", "temperature_p05", "temperature_p95",
)
MI = {name: index for index, name in enumerate(MONTH_METRICS)}
BLOCK_NAMES = (
    "seasonal_temperature", "seasonal_precipitation", "variability_extremes",
    "interannual_departures", "temperature_timing", "precipitation_timing",
)


def describe_month(tg, rr, tn, tx) -> np.ndarray:
    """Return statistics without filling missing weather or normalizing each cell."""
    if not (tg.shape == rr.shape == tn.shape == tx.shape) or tg.ndim != 2:
        raise ValueError("Monthly weather must have matching (days, cells) shapes")
    return np.stack([
        tg.mean(axis=0), rr.sum(axis=0), tg.std(axis=0),
        (rr >= 1).mean(axis=0), (rr >= 20).mean(axis=0),
        np.percentile(rr, 95, axis=0), rr.max(axis=0),
        (tn < 0).mean(axis=0), ((tn < 0) & (tx > 0)).mean(axis=0),
        (tx - tn).mean(axis=0), np.percentile(tg, 5, axis=0),
        np.percentile(tg, 95, axis=0),
    ]).astype(np.float32)


def source_config(source: Path, polygon: Path, max_cells: int | None) -> tuple[dict, list[dict]]:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Source annual download manifest is not complete")
    records = manifest["archives"]
    if [record["year"] for record in records] != list(range(2021, 2026)):
        raise ValueError("This analysis requires complete 2021-2025 archives")
    files = []
    for record in records:
        path = Path(record["netcdf"])
        stat = path.stat()
        if stat.st_size != record["bytes"]:
            raise ValueError(f"Source size differs from download manifest: {path}")
        files.append(dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    stat = polygon.stat()
    return dict(schema=1, files=files, polygon=str(polygon.resolve()),
                polygon_bytes=stat.st_size, polygon_mtime_ns=stat.st_mtime_ns,
                max_cells=max_cells, seed=SEED, metrics=list(MONTH_METRICS)), records


def prepare_cache(source: Path, polygon: Path, output: Path, max_cells: int | None) -> dict:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    config, records = source_config(source, polygon, max_cells)
    config_path = cache / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Cached weather belongs to different inputs; use another output directory")
    else:
        write_json(config_path, config)
    grid_path = cache / "grid.json"
    if grid_path.exists():
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        cell_ids = np.load(cache / "cell_ids.npy")
    else:
        with netCDF4.Dataset(records[0]["netcdf"]) as ds:
            xs, ys = np.asarray(ds["X"][:]), np.asarray(ds["Y"][:])
            if not (np.allclose(np.diff(xs), 1000, rtol=0, atol=1e-6) and
                    np.allclose(np.diff(ys), -1000, rtol=0, atol=1e-6)):
                raise ValueError("Expected north-up seNorge native 1 km coordinate order")
            crs = rasterio.crs.CRS.from_string(ds[ds["tg"].grid_mapping].proj4)
            if crs.to_epsg() != 32633:
                raise ValueError("Expected seNorge UTM zone 33N")
            land = gpd.read_file(polygon)
            if land.crs is None or land.empty or not land.is_valid.all():
                raise ValueError("Norway polygon is empty, invalid or lacks a CRS")
            land = land.to_crs(crs)
            transform = from_origin(float(xs[0] - 500), float(ys[0] + 500), 1000, 1000)
            mask = rasterize(
                ((geometry, 1) for geometry in land.geometry if geometry is not None),
                out_shape=(len(ys), len(xs)), transform=transform, fill=0,
                dtype="uint8", all_touched=False,
            ).astype(bool)
            initial = mask.copy()
            for name in ("tg", "rr", "tn", "tx"):
                field = np.ma.masked_invalid(ds[name][0])
                initial &= ~np.ma.getmaskarray(field)
            cell_ids = np.flatnonzero(initial.ravel())
            if max_cells is not None and max_cells < len(cell_ids):
                cell_ids = np.sort(np.random.default_rng(SEED).choice(
                    cell_ids, max_cells, replace=False,
                ))
            if len(cell_ids) < 100:
                raise ValueError("Fewer than 100 Norway cells are available")
            grid = dict(
                width=len(xs), height=len(ys), west=float(xs[0] - 500),
                north=float(ys[0] + 500), cell_m=1000, crs="EPSG:32633",
                norway_land_centres=int(mask.sum()), initial_cells=len(cell_ids),
                initial_missing_land_cells=int(mask.sum() - initial.sum()),
                preview=max_cells is not None, centre_rule="Cell centre inside Norway land polygon",
            )
        np.save(cache / "cell_ids.npy", cell_ids)
        write_json(grid_path, grid)
    for record in records:
        year = record["year"]
        receipt = cache / f"{year}_complete.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if all((cache / filename).is_file() for filename in saved["files"]):
                print(f"CACHE {year}: extracted weather", flush=True)
                continue
        print(f"EXTRACT {year}: four variables, {len(cell_ids):,} Norway cells", flush=True)
        with netCDF4.Dataset(record["netcdf"]) as ds:
            if ds["tg"].units != "Celsius" or ds["rr"].units != "kg/m^2":
                raise ValueError(f"Unexpected source units in {year}")
            if ds["tn"].units != "Celsius" or ds["tx"].units != "Celsius":
                raise ValueError(f"Unexpected min/max temperature units in {year}")
            if (len(ds.dimensions["Y"]), len(ds.dimensions["X"])) != (
                grid["height"], grid["width"]
            ):
                raise ValueError("Annual grid dimensions differ")
            if not (np.allclose(np.asarray(ds["X"][:]),
                                grid["west"] + 500 + np.arange(grid["width"]) * 1000) and
                    np.allclose(np.asarray(ds["Y"][:]),
                                grid["north"] - 500 - np.arange(grid["height"]) * 1000)):
                raise ValueError("Annual grid coordinates differ")
            t = ds["time"]
            decoded = netCDF4.num2date(t[:], t.units)
            dates = [date(value.year, value.month, value.day) for value in decoded]
            expected = pd.date_range(f"{year}-01-01", f"{year}-12-31").date.tolist()
            if dates != expected:
                raise ValueError(f"Missing, duplicate or wrong dates in {year}")
            count = len(dates)
            paths = [f"tg_{year}.npy", f"rr_{year}.npy", f"months_{year}.npy",
                     f"valid_{year}.npy"]
            tg_out = np.lib.format.open_memmap(cache / paths[0], mode="w+",
                                             dtype="float32", shape=(count, len(cell_ids)))
            rr_out = np.lib.format.open_memmap(cache / paths[1], mode="w+",
                                             dtype="float32", shape=(count, len(cell_ids)))
            months = np.empty((12, len(MONTH_METRICS), len(cell_ids)), dtype=np.float32)
            valid = np.ones(len(cell_ids), dtype=bool)
            for month in range(1, 13):
                indices = [i for i, day in enumerate(dates) if day.month == month]
                arrays = {}
                for name in ("tg", "rr", "tn", "tx"):
                    values = np.empty((len(indices), len(cell_ids)), dtype=np.float32)
                    for j, index in enumerate(indices):
                        field = np.ma.masked_invalid(ds[name][index])
                        values[j] = field.filled(np.nan).ravel()[cell_ids]
                    valid &= np.all(np.isfinite(values), axis=0)
                    arrays[name] = values
                if np.any(arrays["rr"][np.isfinite(arrays["rr"])] < 0):
                    raise ValueError("Negative precipitation is not valid for this analysis")
                tg_out[indices] = arrays["tg"]
                rr_out[indices] = arrays["rr"]
                months[month - 1] = describe_month(**arrays)
                print(f"  {year}-{month:02}: extracted", flush=True)
            tg_out.flush()
            rr_out.flush()
            del tg_out, rr_out
            np.save(cache / paths[2], months)
            np.save(cache / paths[3], valid)
            write_json(receipt, dict(year=year, days=count, files=paths,
                                    complete_cells=int(valid.sum())))
    return grid


def load_weather(cache: Path):
    years = list(range(2021, 2026))
    valid = np.logical_and.reduce([np.load(cache / f"valid_{year}.npy") for year in years])
    ids = np.load(cache / "cell_ids.npy")[valid]
    if len(ids) < 100:
        raise ValueError("Too few cells with complete weather across all five years")
    monthly = np.stack([np.load(cache / f"months_{year}.npy")[:, :, valid] for year in years])
    daily = {}
    for name in ("tg", "rr"):
        daily[name] = np.concatenate([
            np.load(cache / f"{name}_{year}.npy", mmap_mode="r")[:, valid] for year in years
        ], axis=0).T
    if not np.all(np.isfinite(monthly)) or any(not np.all(np.isfinite(a)) for a in daily.values()):
        raise ValueError("Nonfinite weather survived the complete-cell mask")
    dates = pd.date_range("2021-01-01", "2025-12-31")
    print(f"COMPLETE: {len(ids):,} cells; excluded {int((~valid).sum()):,} incomplete cells",
          flush=True)
    return ids, monthly, daily, dates


def summary_blocks(monthly: np.ndarray, year_indices: list[int]) -> list[np.ndarray]:
    selected = monthly[year_indices]
    climatology = selected.mean(axis=0)
    # Matrix shapes are (cells, features); precipitation totals are log-transformed.
    seasonal_t = climatology[:, MI["temperature_mean"]].T
    seasonal_p = np.log1p(climatology[:, MI["precipitation_total"]].T)
    variability = []
    for name in MONTH_METRICS[2:]:
        values = climatology[:, MI[name]].T
        if name.startswith("precipitation"):
            values = np.log1p(values)
        variability.append(values)
    t = selected[:, :, MI["temperature_mean"]]
    p = np.log1p(selected[:, :, MI["precipitation_total"]])
    annual = np.concatenate([
        (t - t.mean(axis=0)).reshape(-1, t.shape[-1]).T,
        (p - p.mean(axis=0)).reshape(-1, p.shape[-1]).T,
    ], axis=1)
    return [seasonal_t, seasonal_p, np.concatenate(variability, axis=1), annual]


def daily_anomalies(values: np.ndarray, months: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    for month in np.unique(months):
        select = months == month
        result[:, select] -= result[:, select].mean(axis=1, keepdims=True)
    return result


def transform_timing(values, transform_ids, keep_time, months, pca, divisor, *, logarithmic=False):
    transformed = np.empty((len(transform_ids), pca.n_components_), dtype=np.float32)
    for start in range(0, len(transform_ids), 2048):
        ids = transform_ids[start:start + 2048]
        batch = values[ids][:, keep_time]
        if logarithmic:
            batch = np.log1p(batch)
        transformed[start:start + len(ids)] = pca.transform(
            daily_anomalies(batch, months),
        ) / divisor
    return transformed


def fit_features(monthly, daily, dates, fit_ids, year_indices, transform_ids=None):
    """Fit spatial scaling/PCA on training cells only; no per-cell variance scaling."""
    transform_ids = np.arange(monthly.shape[-1]) if transform_ids is None else transform_ids
    year_values = np.asarray([2021 + index for index in year_indices])
    keep_time = np.isin(dates.year, year_values)
    month_ids = np.asarray(dates.month)[keep_time]
    block_models = []
    blocks = []
    for block in summary_blocks(monthly, year_indices):
        scaler = StandardScaler().fit(block[fit_ids])
        # Equal expected squared contribution per feature family.
        divisor = np.sqrt(max(1, int(np.count_nonzero(scaler.var_ > 1e-12))) * len(BLOCK_NAMES))
        blocks.append((scaler.transform(block[transform_ids]) / divisor).astype(np.float32))
        block_models.append(dict(kind="standard", model=scaler, divisor=float(divisor)))
    timing_models = []
    for name in ("tg", "rr"):
        train = daily[name][fit_ids][:, keep_time]
        if name == "rr":
            train = np.log1p(train)
        train = daily_anomalies(train, month_ids)
        n_components = min(12, len(fit_ids) - 1, train.shape[1] - 1)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)
        scores = pca.fit_transform(train)
        divisor = float(np.sqrt(max(float(np.var(scores, axis=0).sum()), 1e-12) * len(BLOCK_NAMES)))
        transformed = transform_timing(daily[name], transform_ids, keep_time, month_ids,
                                       pca, divisor, logarithmic=name == "rr")
        blocks.append(transformed)
        timing_models.append(dict(kind="pca", variable=name, model=pca, divisor=divisor,
                                  explained_variance=float(pca.explained_variance_ratio_.sum())))
    features = np.concatenate(blocks, axis=1)
    if not np.all(np.isfinite(features)):
        raise ValueError("Nonfinite clustering features")
    models = dict(blocks=block_models + timing_models, names=list(BLOCK_NAMES),
                  equal_block_weights=[1 / len(BLOCK_NAMES)] * len(BLOCK_NAMES),
                  years=year_values.tolist())
    return features, models


def transform_features(monthly, daily, dates, models):
    years = models["years"]
    blocks = [
        (model["model"].transform(block) / model["divisor"]).astype(np.float32)
        for model, block in zip(models["blocks"][:4],
                                summary_blocks(monthly, [year - 2021 for year in years]))
    ]
    keep_time = np.isin(dates.year, years)
    months = np.asarray(dates.month)[keep_time]
    ids = np.arange(monthly.shape[-1])
    for model in models["blocks"][4:]:
        blocks.append(transform_timing(
            daily[model["variable"]], ids, keep_time, months,
            model["model"], model["divisor"], logarithmic=model["variable"] == "rr",
        ))
    features = np.concatenate(blocks, axis=1)
    if not np.all(np.isfinite(features)):
        raise ValueError("Nonfinite transformed weather features")
    return features


def fit_kmeans(features: np.ndarray, k: int) -> MiniBatchKMeans:
    if len(features) <= k:
        raise ValueError("More training cells than clusters are required")
    model = MiniBatchKMeans(
        n_clusters=k, n_init=10, random_state=SEED, batch_size=2048,
        max_iter=300, max_no_improvement=30, reassignment_ratio=0.005,
    ).fit(features)
    if len(np.unique(model.labels_)) != k:
        raise ValueError(f"K-means produced fewer than {k} populated groups")
    return model


def representative_ids(features, labels, centres) -> np.ndarray:
    result = []
    for label, centre in enumerate(centres):
        members = np.flatnonzero(labels == label)
        if not len(members):
            raise ValueError(f"Group {label + 1} has no members")
        distance = np.sum((features[members] - centre) ** 2, axis=1)
        result.append(members[int(distance.argmin())])
    return np.asarray(result)


def representation_errors(daily, monthly, cell_ids, rep_ids) -> dict[str, np.ndarray]:
    if len(cell_ids) != len(rep_ids):
        raise ValueError("Each evaluated cell needs one representative")
    result = {name: np.empty(len(cell_ids), dtype=np.float32) for name in (
        "temperature_rmse_c", "precipitation_rmse_mm", "wet_day_disagreement",
        "monthly_precipitation_mae_mm", "annual_precipitation_mae_mm",
        "temperature_extreme_mae_c", "heavy_day_frequency_error",
    )}
    monthly_p = monthly[:, :, MI["precipitation_total"]]
    extremes = monthly[:, :, [MI["temperature_p05"], MI["temperature_p95"]], :]
    for start in range(0, len(cell_ids), 1024):
        ids = cell_ids[start:start + 1024]
        reps = rep_ids[start:start + 1024]
        t, rt = daily["tg"][ids], daily["tg"][reps]
        p, rp = daily["rr"][ids], daily["rr"][reps]
        values = [
            np.sqrt(np.mean((t - rt) ** 2, axis=1)),
            np.sqrt(np.mean((p - rp) ** 2, axis=1)),
            np.mean((p >= 1) != (rp >= 1), axis=1),
        ]
        delta = np.take(monthly_p, ids, axis=2) - np.take(monthly_p, reps, axis=2)
        values += [np.mean(np.abs(delta), axis=(0, 1)),
                   np.mean(np.abs(delta.sum(axis=1)), axis=0)]
        values += [np.mean(np.abs(np.take(extremes, ids, axis=3) -
                                 np.take(extremes, reps, axis=3)), axis=(0, 1, 2)),
                   np.abs((p >= 20).mean(axis=1) - (rp >= 20).mean(axis=1))]
        for name, value in zip(result, values):
            result[name][start:start + len(ids)] = value
    return result


def select_candidate(table: pd.DataFrame) -> int:
    """Transparent elbow heuristic, not a claim of a uniquely optimal K."""
    errors = table["normalized_representative_error"].to_numpy()
    x = np.log(table["k"].to_numpy(dtype=float))
    x = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
    if errors[-1] >= errors[0]:
        raise ValueError("Largest K does not improve representative error; inspect candidate results")
    gain = (errors[0] - errors) / float(errors[0] - errors[-1])
    table["error_elbow"] = gain - x
    table["selection_score"] = (table["error_elbow"] + 0.2 * table["silhouette"] +
                                0.2 * table["year_omission_ari_mean"])
    return int(table.loc[table["selection_score"].idxmax(), "k"])


def compare_candidates(features, monthly, daily, dates, training, validation, candidates, output):
    primary = ("temperature_rmse_c", "precipitation_rmse_mm",
               "wet_day_disagreement", "monthly_precipitation_mae_mm")
    one_centre = features[training].mean(axis=0, keepdims=True)
    base_rep = training[representative_ids(features[training],
                                         np.zeros(len(training), dtype=int), one_centre)[0]]
    baseline = representation_errors(daily, monthly, validation,
                                     np.full(len(validation), base_rep))
    normalizers = {key: max(float(baseline[key].mean()), 1e-6) for key in primary}
    models, validation_labels, rows = {}, {}, []
    for k in candidates:
        print(f"COMPARE k={k}: fitting training cells and evaluating independent cells", flush=True)
        model = fit_kmeans(features[training], k)
        train_labels = model.predict(features[training])
        labels = model.predict(features[validation])
        reps = training[representative_ids(features[training], train_labels, model.cluster_centers_)]
        errors = representation_errors(daily, monthly, validation, reps[labels])
        separation = silhouette_score(
            features[validation], labels, sample_size=min(2000, len(validation)), random_state=SEED,
        )
        normalized = float(np.mean([errors[key].mean() / normalizers[key] for key in primary]))
        rows.append(dict(k=k, silhouette=float(separation),
                         normalized_representative_error=normalized,
                         **{key: float(value.mean()) for key, value in errors.items()}))
        models[k], validation_labels[k] = model, labels
    aris = {k: [] for k in candidates}
    evaluation_ids = np.concatenate([training, validation])
    for omitted in range(5):
        print(f"STABILITY: refitting features/PCA and groups without {2021 + omitted}", flush=True)
        fold, _ = fit_features(monthly, daily, dates, training,
                               [i for i in range(5) if i != omitted], evaluation_ids)
        for k in candidates:
            model = fit_kmeans(fold[:len(training)], k)
            score = adjusted_rand_score(validation_labels[k], model.predict(fold[len(training):]))
            aris[k].append(float(score))
    for row in rows:
        scores = aris[row["k"]]
        row.update(year_omission_ari_mean=float(np.mean(scores)),
                   year_omission_ari_min=float(min(scores)))
        row.update({f"ari_omit_{2021 + index}": value for index, value in enumerate(scores)})
    table = pd.DataFrame(rows)
    chosen = select_candidate(table)
    table["selected"] = table["k"] == chosen
    table.to_csv(output / "cluster_comparison.csv", index=False)
    write_json(output / "selection.json", dict(
        selected_k=chosen, training_cells=len(training), validation_cells=len(validation),
        baseline_representative_cell_index=int(base_rep), baseline_errors=normalizers,
        formula="score = elbow(normalized representative error vs log K) + "
                "0.2*silhouette + 0.2*mean year-omission ARI; choose maximum",
        representative_error="Equal mean of validation temperature RMSE, precipitation RMSE, "
                             "wet-day disagreement and monthly precipitation MAE, each divided "
                             "by its single-representative baseline.",
        limitations="Heuristic compromise within tested counts, not a uniquely optimal climate "
                    "classification. Spatial validation cells are random, not independent stations. "
                    "Year omission measures partition sensitivity, not future predictive skill.",
    ))
    return chosen, models[chosen], table


def write_raster(path: Path, grid: dict, cell_ids, values, units: str, *, categorical=False):
    import rasterio
    from rasterio.transform import from_origin

    nodata = 0 if categorical else NODATA
    dtype = "uint16" if categorical else "float32"
    array = np.full(grid["height"] * grid["width"], nodata, dtype=dtype)
    array[cell_ids] = values
    with rasterio.open(
        path, "w", driver="GTiff", width=grid["width"], height=grid["height"],
        count=1, dtype=dtype, nodata=nodata, crs=grid["crs"],
        transform=from_origin(grid["west"], grid["north"], 1000, 1000),
        compress="deflate", tiled=True,
    ) as target:
        target.write(array.reshape(grid["height"], grid["width"]), 1)
        target.update_tags(units=units, period="2021-2025", source="MET Norway seNorge_2018",
                           classification="Five-year weather regions, not 30-year climate normals")


def export_results(output, grid, cell_ids, features, model, monthly, daily, table):
    import geopandas as gpd
    import joblib

    labels = model.predict(features)
    centres = model.cluster_centers_
    representatives = representative_ids(features, labels, centres)
    # Stable, interpretable numbering: order groups from coldest to warmest.
    annual_t = daily["tg"].mean(axis=1)
    order = np.argsort([annual_t[labels == i].mean() for i in range(len(centres))])
    numbering = np.empty(len(order), dtype=int)
    numbering[order] = np.arange(1, len(order) + 1)
    groups = numbering[labels]
    errors = representation_errors(daily, monthly, np.arange(len(labels)), representatives[labels])
    errors["feature_distance"] = np.sqrt(np.sum((features - centres[labels]) ** 2, axis=1))
    write_raster(output / "weather_regions_1km.tif", grid, cell_ids, groups, "region_id",
                 categorical=True)
    units = {
        "temperature_rmse_c": "degC", "precipitation_rmse_mm": "mm/day",
        "wet_day_disagreement": "fraction", "monthly_precipitation_mae_mm": "mm/month",
        "annual_precipitation_mae_mm": "mm/year", "temperature_extreme_mae_c": "degC",
        "heavy_day_frequency_error": "fraction", "feature_distance": "weighted feature distance",
    }
    for name, values in errors.items():
        write_raster(output / f"{name}_1km.tif", grid, cell_ids, values, units[name])
    rows, profiles, annual_profiles = [], [], []
    for label in order:
        members = np.flatnonzero(labels == label)
        rep = representatives[label]
        flat = int(cell_ids[rep])
        row, column = divmod(flat, grid["width"])
        x, y = grid["west"] + (column + 0.5) * 1000, grid["north"] - (row + 0.5) * 1000
        region = int(numbering[label])
        summary = dict(
            region_id=region, cells=len(members), nominal_area_km2=len(members),
            representative_x=x, representative_y=y,
            mean_temperature_c=float(annual_t[members].mean()),
            mean_annual_precipitation_mm=float(
                monthly[:, :, MI["precipitation_total"], :][:, :, members].sum(axis=1).mean()),
        )
        for name, values in errors.items():
            summary[f"{name}_mean"] = float(values[members].mean())
            summary[f"{name}_p95"] = float(np.percentile(values[members], 95))
        rows.append(summary)
        for month in range(12):
            for metric in ("temperature_mean", "precipitation_total", "wet_fraction",
                           "frost_fraction", "freeze_thaw_fraction", "heavy_fraction"):
                values = monthly[:, month, MI[metric], :]
                profiles.append(dict(
                    region_id=region, month=month + 1, metric=metric,
                    region_mean=float(values[:, members].mean()),
                    representative=float(values[:, rep].mean()),
                ))
        for year in range(5):
            for month in range(12):
                annual_profiles.append(dict(
                    region_id=region, year=2021 + year, month=month + 1,
                    mean_temperature_c=float(monthly[year, month, MI["temperature_mean"], members].mean()),
                    precipitation_mm=float(monthly[year, month, MI["precipitation_total"], members].mean()),
                ))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "region_summary.csv", index=False)
    pd.DataFrame(profiles).to_csv(output / "seasonal_profiles.csv", index=False)
    pd.DataFrame(annual_profiles).to_csv(output / "year_month_profiles.csv", index=False)
    representatives_frame = gpd.GeoDataFrame(
        summary, geometry=gpd.points_from_xy(summary.representative_x, summary.representative_y),
        crs=grid["crs"],
    )
    representatives_frame.to_file(output / "representatives.gpkg", layer="representatives", driver="GPKG")
    joblib.dump(dict(model=model, group_numbering=numbering), output / "kmeans.joblib")
    plot_results(output, grid, cell_ids, groups, summary, pd.DataFrame(profiles), table, errors)
    return summary, errors


def plot_results(output, grid, cell_ids, groups, summary, profiles, comparison, errors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    extent = (grid["west"], grid["west"] + grid["width"] * 1000,
              grid["north"] - grid["height"] * 1000, grid["north"])
    raster = np.full(grid["height"] * grid["width"], np.nan)
    raster[cell_ids] = groups
    k = len(summary)
    colors = plt.get_cmap("tab20" if k <= 20 else "gist_ncar")(np.linspace(0, 1, k))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, k + 1.5), k)
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set_facecolor("#e6e6e6")
    image = ax.imshow(raster.reshape(grid["height"], grid["width"]), extent=extent,
                      cmap=cmap, norm=norm, interpolation="nearest")
    ax.scatter(summary.representative_x, summary.representative_y, s=22, c="black", marker="x")
    for item in summary.itertuples():
        ax.annotate(str(item.region_id), (item.representative_x, item.representative_y),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(title=f"Norway weather regions, 2021-2025: {k} groups",
           xlabel="UTM 33N easting (m)", ylabel="UTM 33N northing (m)")
    fig.colorbar(image, ax=ax, ticks=np.arange(1, k + 1), label="Region ID (cold to warm)")
    fig.tight_layout()
    fig.savefig(output / "weather_regions.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    for ax, name, title in zip(axes, ("temperature_rmse_c", "wet_day_disagreement"),
                               ("Daily temperature RMSE (degC)", "Wet-day disagreement (fraction)")):
        raster[:] = np.nan
        raster[cell_ids] = errors[name]
        image = ax.imshow(raster.reshape(grid["height"], grid["width"]), extent=extent,
                          cmap="magma", vmin=0, vmax=float(np.percentile(errors[name], 99)))
        ax.set_title(title + "\nversus the region's representative cell")
        fig.colorbar(image, ax=ax, shrink=0.65, extend="max")
    fig.tight_layout()
    fig.savefig(output / "representation_errors.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for region in range(1, k + 1):
        for ax, metric in zip(axes, ("temperature_mean", "precipitation_total")):
            values = profiles[(profiles.region_id == region) & (profiles.metric == metric)]
            ax.plot(values.month, values.region_mean, color=colors[region - 1], label=str(region))
    axes[0].set_ylabel("Monthly mean temperature (degC)")
    axes[1].set_ylabel("Mean monthly precipitation (mm)")
    axes[1].set_xlabel("Month")
    axes[0].legend(title="Region", ncol=min(k, 16), fontsize=8)
    for ax in axes:
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "seasonal_profiles.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(axes, ("normalized_representative_error", "silhouette",
                                      "year_omission_ari_mean"),
                                 ("Representative error (lower is better)", "Feature separation",
                                  "Year-omission stability (ARI)")):
        ax.plot(comparison.k, comparison[metric], "o-")
        ax.set(title=title, xlabel="Number of groups")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "cluster_comparison.png", dpi=140)
    plt.close(fig)


def export_additional(output: Path, counts: list[int]) -> None:
    import joblib

    original = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    table = pd.read_csv(output / "cluster_comparison.csv")
    if any(k not in table.k.tolist() for k in counts):
        raise ValueError("Additional group counts must have been included in the original comparison")
    for k in counts:
        if (output / f"k{k}").exists():
            raise FileExistsError(f"Additional result already exists: {output / f'k{k}'}")
    ids, monthly, daily, dates = load_weather(output / "cache")
    models = joblib.load(output / "feature_preprocessing.joblib")
    features = transform_features(monthly, daily, dates, models)
    for k in counts:
        target = output / f"k{k}"
        target.mkdir()
        print(f"ALTERNATIVE: refitting {k} groups on all {len(ids):,} cells", flush=True)
        model = fit_kmeans(features, k)
        _, errors = export_results(
            target, original["grid"], ids, features, model, monthly, daily, table,
        )
        metadata = dict(original)
        metadata.update(
            selected_k=k, selection="User-requested alternative from the tested candidates",
            overview_groups=original["selected_k"],
            preprocessing=str((output / "feature_preprocessing.joblib").resolve()),
            comparison=str((output / "cluster_comparison.csv").resolve()),
            mean_representation_errors={name: float(values.mean()) for name, values in errors.items()},
            topology_note="Independently fitted clusters, not nested subdivisions of the overview.",
        )
        write_json(target / "analysis.json", metadata)
        print(f"COMPLETE ALTERNATIVE: {target}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "senorge_norway_2021_2025")
    parser.add_argument("--polygon", type=Path, default=ROOT / "outputs" / "norway_polygon.gpkg")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    parser.add_argument("--training-cells", type=int, default=12000)
    parser.add_argument("--validation-cells", type=int, default=4000)
    parser.add_argument("--max-cells", type=int, help="Random-cell preview only; omit for national map")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--additional-k", type=int, nargs="+",
                        help="Export tested alternatives in k<K>/ using an already completed analysis")
    args = parser.parse_args(argv)
    candidates = sorted(set(args.candidates))
    if len(candidates) < 3 or min(candidates) < 2:
        parser.error("Provide at least three candidate counts, all >= 2")
    if min(args.training_cells, args.validation_cells) <= max(candidates) or args.threads < 1:
        parser.error("Training/validation counts must exceed maximum K; threads must be positive")
    if args.max_cells is not None and args.max_cells < 100:
        parser.error("--max-cells must be >= 100")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed = (output / "analysis.json").exists()
    if args.additional_k and not completed:
        parser.error("--additional-k requires a completed analysis in --output-dir")
    if completed and not args.additional_k:
        parser.error("Analysis is already complete; use --additional-k or a different output directory")
    with threadpool_limits(limits=args.threads):
        grid = prepare_cache(args.source, args.polygon, output, args.max_cells)
        if args.additional_k:
            export_additional(output, sorted(set(args.additional_k)))
            return 0
        cell_ids, monthly, daily, dates = load_weather(output / "cache")
        if args.training_cells + args.validation_cells > len(cell_ids):
            parser.error("Not enough complete cells for disjoint training/validation samples")
        sample = np.random.default_rng(SEED).permutation(len(cell_ids))
        training = sample[:args.training_cells]
        validation = sample[args.training_cells:args.training_cells + args.validation_cells]
        print("FEATURES: seasonal levels, extremes, year departures and daily anomaly PCA", flush=True)
        features, preprocessing = fit_features(monthly, daily, dates, training, list(range(5)))
        import joblib
        joblib.dump(preprocessing, output / "feature_preprocessing.joblib")
        chosen, model, table = compare_candidates(
            features, monthly, daily, dates, training, validation, candidates, output,
        )
        print(f"SELECTED: {chosen} groups; refitting K-means on all {len(cell_ids):,} cells", flush=True)
        model = fit_kmeans(features, chosen)
        summary, errors = export_results(
            output, grid, cell_ids, features, model, monthly, daily, table,
        )
        write_json(output / "analysis.json", dict(
            status="complete", period="2021-2025", selected_k=chosen, grid=grid,
            complete_cells=len(cell_ids), excluded_incomplete_cells=grid["initial_cells"] - len(cell_ids),
            candidates=candidates, seed=SEED, feature_blocks=list(BLOCK_NAMES),
            block_weights=preprocessing["equal_block_weights"],
            pca_explained_variance={model["variable"]: model["explained_variance"]
                                    for model in preprocessing["blocks"] if model["kind"] == "pca"},
            mean_representation_errors={name: float(values.mean()) for name, values in errors.items()},
            source="MET Norway seNorge_2018; source file provenance in cache/config.json",
            limitations=[
                "Five-year weather classification, not a 30-year climate normal or future forecast.",
                "No wind, humidity or measured snow features.",
                "Region numbering is categorical; disconnected areas may share a weather type.",
                "Regions are not constrained by geography; coordinates are not clustering features.",
                "Cell-centre land mask; coastal partial-cell area is not exact polygon area.",
                "Year-omission stability is not independent station validation.",
                "Representation errors are descriptive for 2021-2025, not out-of-period prediction errors.",
            ],
        ))
        print(summary[["region_id", "cells", "mean_temperature_c",
                       "mean_annual_precipitation_mm"]].to_string(index=False), flush=True)
        print(f"COMPLETE: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
