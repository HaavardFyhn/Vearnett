"""Stream 1996-2025 seNorge into reproducible, Norway-only climate regions."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import shutil
import time

import netCDF4
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import weather_regions as weather
from download_nve_grid import write_json


ROOT = Path(__file__).resolve().parent
YEARS = list(range(1996, 2026))
PERIOD = "1996-2025"
SKETCH_SIZE = 128
ANNUAL_NAMES = (
    "longest_wet_spell_days", "longest_dry_spell_days",
    "longest_warm_spell_tg_ge20_days", "longest_cold_spell_tg_lt0_days",
    "growing_degree_days_base5", "heating_degree_days_base17",
    "freezing_degree_days_base0", "days_tg_gt5",
    "max_3day_precipitation_mm", "max_5day_precipitation_mm",
)
FAMILIES = (
    "seasonal_temperature", "seasonal_precipitation", "variability_extremes",
    "spells_accumulation", "interannual_changes", "temperature_timing", "precipitation_timing",
)
SEASONAL_NAMES = (
    "summer_temperature_c", "winter_temperature_c",
    "summer_winter_temperature_amplitude_c", "frost_days_per_year",
)


def seasonal_indicators(monthly: np.ndarray, year: int) -> np.ndarray:
    """Day-weighted JJA minus Jan/Feb/Dec temperature, and days with tn < 0."""
    days = pd.date_range(f"{year}-01-01", periods=12, freq="MS").days_in_month.to_numpy()
    temperatures = monthly[:, weather.MI["temperature_mean"]]
    summer, winter = np.array([5, 6, 7]), np.array([0, 1, 11])
    summer_mean = np.average(temperatures[summer], axis=0, weights=days[summer])
    winter_mean = np.average(temperatures[winter], axis=0, weights=days[winter])
    # Recover integer counts from float32 monthly frequencies.
    frost_days = np.rint(monthly[:, weather.MI["frost_fraction"]] * days[:, None]).sum(axis=0)
    return np.asarray([summer_mean, winter_mean, summer_mean - winter_mean, frost_days],
                      dtype=np.float32)


def annual_indices(tg: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """Calendar-year spell maxima and within-year rolling precipitation totals."""
    if tg.shape != rr.shape or tg.ndim != 2:
        raise ValueError("Annual fields must have matching (days, cells) shapes")
    if len(tg) < 5:
        raise ValueError("At least five daily samples are required")
    maxima = []
    for condition in (rr >= 1, rr < 1, tg >= 20, tg < 0):
        run = np.zeros(tg.shape[1], dtype=np.int16)
        longest = run.copy()
        for day in condition:
            run = np.where(day, run + 1, 0)
            longest = np.maximum(longest, run)
        maxima.append(longest)
    maxima.extend([
        np.maximum(tg - 5, 0).sum(axis=0),
        np.maximum(17 - tg, 0).sum(axis=0),
        np.maximum(-tg, 0).sum(axis=0),
        (tg > 5).sum(axis=0),
    ])
    cumulative = np.concatenate([np.zeros((1, rr.shape[1])), np.cumsum(rr, axis=0, dtype=np.float64)])
    for window in (3, 5):
        maxima.append((cumulative[window:] - cumulative[:-window]).max(axis=0))
    return np.asarray(maxima, dtype=np.float32)


def temporal_projection(year: int, days: int) -> np.ndarray:
    return (np.random.default_rng(weather.SEED + year).choice(
        [-1.0, 1.0], size=(days, SKETCH_SIZE),
    ) / np.sqrt(SKETCH_SIZE)).astype(np.float32)


def timing_sketch(values: np.ndarray, months: np.ndarray, year: int) -> np.ndarray:
    anomalies = weather.daily_anomalies(values.T, months)
    return anomalies @ temporal_projection(year, len(values))


def load_records(source: Path) -> dict[int, dict]:
    path = source / "manifest.json"
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {entry["year"]: entry for entry in manifest["archives"]}


def make_grid(record: dict, polygon: Path, cache: Path, sample_count: int,
              max_cells: int | None) -> tuple[dict, np.ndarray, np.ndarray]:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    land = gpd.read_file(polygon)
    if land.crs is None or land.empty or not land.is_valid.all():
        raise ValueError("Invalid Norway polygon")
    with netCDF4.Dataset(record["netcdf"]) as ds:
        xs, ys = np.asarray(ds["X"][:]), np.asarray(ds["Y"][:])
        if not (np.allclose(np.diff(xs), 1000, atol=1e-6, rtol=0) and
                np.allclose(np.diff(ys), -1000, atol=1e-6, rtol=0)):
            raise ValueError("Expected north-up 1 km seNorge grid")
        crs = rasterio.crs.CRS.from_string(ds[ds["tg"].grid_mapping].proj4)
        if crs.to_epsg() != 32633:
            raise ValueError("Expected UTM 33N source projection")
        land = land.to_crs(crs)
        mask = rasterize(
            ((g, 1) for g in land.geometry), out_shape=(len(ys), len(xs)),
            transform=from_origin(xs[0] - 500, ys[0] + 500, 1000, 1000),
            fill=0, dtype="uint8", all_touched=False,
        ).astype(bool)
        ids = np.flatnonzero(mask.ravel())
        if max_cells is not None:
            ids = np.sort(np.random.default_rng(weather.SEED).choice(
                ids, min(max_cells, len(ids)), replace=False,
            ))
        if len(ids) < sample_count:
            raise ValueError("Fewer land cells than requested training/validation sample")
        grid = dict(
            width=len(xs), height=len(ys), west=float(xs[0] - 500), north=float(ys[0] + 500),
            crs="EPSG:32633", cell_m=1000, land_cells=int(mask.sum()), cells=len(ids),
            preview=max_cells is not None,
        )
    sample = np.random.default_rng(weather.SEED).permutation(len(ids))[:sample_count]
    np.save(cache / "cell_ids.npy", ids)
    np.save(cache / "sample.npy", sample)
    write_json(cache / "grid.json", grid)
    return grid, ids, sample


def extract_year(record: dict, grid: dict, ids: np.ndarray, sample: np.ndarray, cache: Path) -> None:
    year = record["year"]
    path = Path(record["netcdf"])
    stat = path.stat()
    fingerprint = dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if stat.st_size != record["bytes"]:
        raise ValueError(f"{year} source size changed")
    receipt = cache / f"{year}.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved["source"] != fingerprint:
            raise ValueError(f"{year} cached source fingerprint mismatch")
        if all((cache / name).exists() for name in saved["files"]):
            for name, shape in saved["shapes"].items():
                if list(np.load(cache / name, mmap_mode="r").shape) != shape:
                    raise ValueError(f"{year}: cached array shape changed: {name}")
            print(f"CACHE {year}: verified extraction", flush=True)
            return
    if shutil.disk_usage(cache).free < 5 * 1024**3:
        raise OSError("Less than 5 GiB free; extraction stopped without deleting originals")
    print(f"EXTRACT {year}: {len(ids):,} land cells", flush=True)
    import rasterio

    with netCDF4.Dataset(path) as ds:
        xs = grid["west"] + 500 + np.arange(grid["width"]) * 1000
        ys = grid["north"] - 500 - np.arange(grid["height"]) * 1000
        if not (np.allclose(ds["X"][:], xs, rtol=0, atol=1e-6) and
                np.allclose(ds["Y"][:], ys, rtol=0, atol=1e-6)):
            raise ValueError(f"{year}: source grid differs")
        times = ds["time"]
        decoded = netCDF4.num2date(times[:], times.units, getattr(times, "calendar", "standard"))
        dates = pd.DatetimeIndex([date(t.year, t.month, t.day) for t in decoded])
        if not dates.equals(pd.date_range(f"{year}-01-01", f"{year}-12-31")):
            raise ValueError(f"{year}: missing or duplicate days")
        for name in ("tg", "rr", "tn", "tx"):
            expected = "kg/m^2" if name == "rr" else "Celsius"
            if ds[name].units != expected or ds[name].dimensions != ("time", "Y", "X"):
                raise ValueError(f"{year}: unexpected units/dimensions for {name}")
            if rasterio.crs.CRS.from_string(ds[ds[name].grid_mapping].proj4).to_epsg() != 32633:
                raise ValueError(f"{year}: unexpected projection for {name}")
        if any((t.hour, t.minute, t.second) != (6, 0, 0) for t in decoded):
            raise ValueError(f"{year}: expected daily 06 UTC timestamps")
        day_count = len(dates)
        # One year in RAM, not thirty. Invalid cells are explicitly excluded after extraction.
        t_year = np.empty((day_count, len(ids)), dtype=np.float32)
        p_year = np.empty_like(t_year)
        monthly = np.empty((12, len(weather.MONTH_METRICS), len(ids)), dtype=np.float32)
        valid = np.ones(len(ids), dtype=bool)
        for month in range(1, 13):
            indices = np.flatnonzero(dates.month == month)
            fields = {}
            for name in ("tg", "rr", "tn", "tx"):
                values = np.empty((len(indices), len(ids)), dtype=np.float32)
                for j, index in enumerate(indices):
                    field = np.ma.masked_invalid(ds[name][index])
                    values[j] = field.filled(np.nan).ravel()[ids]
                valid &= np.all(np.isfinite(values), axis=0)
                fields[name] = values
            finite_p = fields["rr"][np.isfinite(fields["rr"])]
            if np.any(finite_p < 0):
                raise ValueError(f"{year}: negative precipitation")
            t_year[indices], p_year[indices] = fields["tg"], fields["rr"]
            monthly[month - 1] = weather.describe_month(**fields)
        annual = annual_indices(t_year, p_year)
        sketches = {}
        for name, values in (("tg", t_year), ("rr", np.log1p(p_year))):
            sketch = np.empty((len(ids), SKETCH_SIZE), dtype=np.float32)
            for start in range(0, len(ids), 4096):
                sketch[start:start + 4096] = timing_sketch(
                    values[:, start:start + 4096], np.asarray(dates.month), year,
                )
            sketches[name] = sketch
        files, shapes = [], {}
        for name, value in (
            ("months", monthly), ("annual", annual), ("valid", valid),
            ("sketch_tg", sketches["tg"]), ("sketch_rr", sketches["rr"]),
            ("sample_tg", t_year[:, sample].T), ("sample_rr", p_year[:, sample].T),
        ):
            filename = f"{name}_{year}.npy"
            np.save(cache / filename, value)
            files.append(filename)
            shapes[filename] = list(value.shape)
        write_json(receipt, dict(
            year=year, days=day_count, source=fingerprint, source_url=record["source"],
            valid_cells=int(valid.sum()), files=files, shapes=shapes, sketch_size=SKETCH_SIZE,
            time_reference="06 UTC daily timestamps; tg/rr previous 06-06 UTC, tn/tx 18-18 UTC",
            attribution=getattr(ds, "institution", "MET Norway"),
            license=getattr(ds, "license", "See original MET Norway archive metadata"),
        ))
        print(f"EXTRACTED {year}: {int(valid.sum()):,} complete cells", flush=True)


def extract_all(args) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    cache = args.output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    pstat = args.polygon.stat()
    config = dict(
        schema=2, years=YEARS, source=str(args.source.resolve()),
        polygon=str(args.polygon.resolve()), polygon_bytes=pstat.st_size,
        polygon_mtime_ns=pstat.st_mtime_ns, sketch_size=SKETCH_SIZE,
        training_cells=args.training_cells, validation_cells=args.validation_cells,
        max_cells=args.max_cells, seed=weather.SEED,
    )
    config_path = cache / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Cache configuration mismatch; use a different output directory")
    write_json(config_path, config)
    for year in YEARS:
        start = time.monotonic()
        records = load_records(args.source)
        while year not in records:
            if not args.wait_for_download:
                raise ValueError(f"No verified annual archive yet for {year}")
            if time.monotonic() - start > 7200:
                raise TimeoutError(f"Waited two hours without a verified {year} archive")
            print(f"WAIT: verified {year} archive not yet available", flush=True)
            time.sleep(60)
            records = load_records(args.source)
        if (cache / "grid.json").exists():
            grid = json.loads((cache / "grid.json").read_text(encoding="utf-8"))
            ids, sample = np.load(cache / "cell_ids.npy"), np.load(cache / "sample.npy")
        else:
            grid, ids, sample = make_grid(
                records[year], args.polygon, cache,
                args.training_cells + args.validation_cells, args.max_cells,
            )
        extract_year(records[year], grid, ids, sample, cache)
    valid = np.logical_and.reduce([np.load(cache / f"valid_{year}.npy") for year in YEARS])
    return grid, ids, sample, valid


def aggregate_blocks(cache: Path, valid: np.ndarray, years: list[int]) -> list[np.ndarray]:
    """Aggregate annual/monthly maps without constructing a 30-year daily cube."""
    n = int(valid.sum())
    mean = np.zeros((12, len(weather.MONTH_METRICS), n), dtype=np.float64)
    t_sum = np.zeros((12, n), dtype=np.float64)
    p_sum = np.zeros_like(t_sum)
    t_square, p_square = np.zeros_like(t_sum), np.zeros_like(t_sum)
    t_trend, p_trend = np.zeros_like(t_sum), np.zeros_like(t_sum)
    annual_mean = np.zeros((len(ANNUAL_NAMES), n), dtype=np.float64)
    annual_square = np.zeros_like(annual_mean)
    seasonal_sum = np.zeros((len(SEASONAL_NAMES), n), dtype=np.float64)
    frost_square = np.zeros(n, dtype=np.float64)
    sketches = [np.zeros((n, SKETCH_SIZE), dtype=np.float32) for _ in range(2)]
    centred_years = np.asarray(years) - np.mean(years)
    for year, centred in zip(years, centred_years):
        monthly = np.load(cache / f"months_{year}.npy", mmap_mode="r")[:, :, valid]
        mean += monthly
        t = monthly[:, weather.MI["temperature_mean"]]
        p = np.log1p(monthly[:, weather.MI["precipitation_total"]])
        t_sum += t
        p_sum += p
        t_square += t * t
        p_square += p * p
        t_trend += centred * t
        p_trend += centred * p
        annual = np.load(cache / f"annual_{year}.npy", mmap_mode="r")[:, valid]
        annual_mean += annual
        annual_square += annual * annual
        seasonal = seasonal_indicators(monthly, year)
        seasonal_sum += seasonal
        frost_square += seasonal[3].astype(np.float64)**2
        for i, name in enumerate(("tg", "rr")):
            sketches[i] += np.load(cache / f"sketch_{name}_{year}.npy", mmap_mode="r")[valid]
    mean /= len(years)
    annual_mean /= len(years)
    annual_std = np.sqrt(np.maximum(annual_square / len(years) - annual_mean**2, 0))
    t_std = np.sqrt(np.maximum(t_square / len(years) - (t_sum / len(years))**2, 0))
    p_std = np.sqrt(np.maximum(p_square / len(years) - (p_sum / len(years))**2, 0))
    trend_denominator = max(float(np.sum(centred_years**2)), 1)
    trend = np.concatenate([t_trend.T, p_trend.T], axis=1) * (10 / trend_denominator)
    seasonal_mean = seasonal_sum / len(years)
    frost_std = np.sqrt(np.maximum(frost_square / len(years) - seasonal_mean[3]**2, 0))
    variability = []
    for name in weather.MONTH_METRICS[2:]:
        values = mean[:, weather.MI[name]].T
        variability.append(np.log1p(values) if name.startswith("precipitation") else values)
    return [
        np.column_stack([mean[:, weather.MI["temperature_mean"]].T,
                         seasonal_mean[2]]).astype(np.float32),
        np.log1p(mean[:, weather.MI["precipitation_total"]].T).astype(np.float32),
        np.concatenate(variability, axis=1).astype(np.float32),
        np.column_stack([np.log1p(annual_mean.T), np.log1p(annual_std.T),
                         np.log1p(seasonal_mean[3]), np.log1p(frost_std)]).astype(np.float32),
        np.concatenate([t_std.T, p_std.T, trend], axis=1).astype(np.float32),
        sketches[0] / np.sqrt(len(years)), sketches[1] / np.sqrt(len(years)),
    ]


def encode_blocks(blocks, training, target_ids=None):
    ids = np.arange(len(blocks[0])) if target_ids is None else target_ids
    encoded, models = [], []
    for index, block in enumerate(blocks):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"Nonfinite values in {FAMILIES[index]}")
        if index < 5:
            model = StandardScaler().fit(block[training])
            divisor = np.sqrt(max(1, np.count_nonzero(model.var_ > 1e-12)) * len(FAMILIES))
            values = model.transform(block[ids]) / divisor
        else:
            model = PCA(n_components=12, svd_solver="randomized", random_state=weather.SEED)
            scores = model.fit_transform(block[training])
            divisor = np.sqrt(max(float(np.var(scores, axis=0).sum()), 1e-12) * len(FAMILIES))
            values = model.transform(block[ids]) / divisor
        encoded.append(values.astype(np.float32))
        models.append(dict(name=FAMILIES[index], model=model, divisor=float(divisor)))
    return np.concatenate(encoded, axis=1), models


def sample_weather(cache, years, sample_keep):
    return {
        name: np.concatenate([
            np.load(cache / f"sample_{name}_{year}.npy", mmap_mode="r")[sample_keep]
            for year in years
        ], axis=1) for name in ("tg", "rr")
    }


def pair_errors(daily, ids, reps) -> dict:
    t = daily["tg"][ids] - daily["tg"][reps]
    p, rp = daily["rr"][ids], daily["rr"][reps]
    return dict(
        temperature_rmse_c=float(np.sqrt(np.mean(t**2, axis=1)).mean()),
        precipitation_rmse_mm=float(np.sqrt(np.mean((p - rp)**2, axis=1)).mean()),
        wet_day_disagreement=float(np.mean((p >= 1) != (rp >= 1))),
    )


def timing_audit(cache, valid, complete_sample, sample_keep, models) -> dict:
    """Compare compressed timing distances with exact sampled daily anomalies."""
    rng = np.random.default_rng(weather.SEED)
    left = rng.integers(len(sample_keep), size=512)
    right = (left + rng.integers(1, len(sample_keep), size=512)) % len(sample_keep)
    raw_sample = np.flatnonzero(valid)[complete_sample]
    report = {}
    for name, fitted in zip(("tg", "rr"), models[5:]):
        exact_square = np.zeros(len(left), dtype=np.float64)
        projected = np.zeros((len(sample_keep), SKETCH_SIZE), dtype=np.float32)
        for year in YEARS:
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
            values = np.load(cache / f"sample_{name}_{year}.npy", mmap_mode="r")[sample_keep]
            if name == "rr":
                values = np.log1p(values)
            anomalies = weather.daily_anomalies(values, np.asarray(dates.month))
            exact_square += np.sum((anomalies[left] - anomalies[right])**2, axis=1)
            projected += np.load(cache / f"sketch_{name}_{year}.npy", mmap_mode="r")[raw_sample]
        exact = np.sqrt(exact_square)
        compressed = np.linalg.norm(projected[left] - projected[right], axis=1)
        scores = fitted["model"].transform(projected / np.sqrt(len(YEARS)))
        pca_distance = np.linalg.norm(scores[left] - scores[right], axis=1) * np.sqrt(len(YEARS))
        nonzero = exact > 1e-8
        if nonzero.sum() < 2:
            raise ValueError(f"Too few distinct {name} anomaly pairs for timing audit")
        report[name] = dict(
            sampled_pairs=int(nonzero.sum()),
            projection_median_relative_distance_error=float(
                np.median(np.abs(compressed[nonzero] / exact[nonzero] - 1))),
            projection_distance_correlation=float(np.corrcoef(exact, compressed)[0, 1]),
            pca_distance_correlation=float(np.corrcoef(exact, pca_distance)[0, 1]),
            pca_sketch_variance_retained=float(fitted["model"].explained_variance_ratio_.sum()),
            caveat="Sampled compression diagnostic, not independent weather accuracy.",
        )
    return report


def compare(cache, valid, sample, args):
    # Map original extraction indices to complete-cell indices, retaining disjoint samples.
    sample_keep = np.flatnonzero(valid[sample])
    lookup = np.full(len(valid), -1, dtype=int)
    lookup[valid] = np.arange(valid.sum())
    complete_sample = lookup[sample[sample_keep]]
    train_keep = np.flatnonzero(sample_keep < args.training_cells)
    validation_keep = np.flatnonzero(sample_keep >= args.training_cells)
    if min(len(train_keep), len(validation_keep)) <= max(args.candidates):
        raise ValueError("Too few complete sampled cells for candidate validation")
    training = complete_sample[train_keep]
    validation = complete_sample[validation_keep]
    blocks = aggregate_blocks(cache, valid, YEARS)
    features, models = encode_blocks(blocks, training)
    del blocks
    write_json(args.output_dir / "timing_compression_audit.json",
               timing_audit(cache, valid, complete_sample, sample_keep, models))
    sample_daily = sample_weather(cache, YEARS, sample_keep)
    candidates, labels, rows = {}, {}, []
    for k in args.candidates:
        print(f"COMPARE k={k}: thirty-year training and validation", flush=True)
        model = weather.fit_kmeans(features[training], k)
        tl = model.predict(features[training])
        vl = model.predict(features[validation])
        reps = train_keep[weather.representative_ids(features[training], tl, model.cluster_centers_)]
        error = pair_errors(sample_daily, validation_keep, reps[vl])
        rows.append(dict(k=k, **error, silhouette=float(silhouette_score(
            features[validation], vl, sample_size=min(2000, len(validation)), random_state=weather.SEED,
        ))))
        candidates[k], labels[k] = model, vl
    # Independent held-out five-year weather blocks; all preprocessing is refitted.
    for start in range(1996, 2026, 5):
        held = list(range(start, start + 5))
        retained = [year for year in YEARS if year not in held]
        print(f"BLOCK VALIDATION {held[0]}-{held[-1]}: refit without five held-out years", flush=True)
        fold_blocks = aggregate_blocks(cache, valid, retained)
        fold, _ = encode_blocks(fold_blocks, training, complete_sample)
        del fold_blocks
        held_daily = sample_weather(cache, held, sample_keep)
        for row in rows:
            k = row["k"]
            model = weather.fit_kmeans(fold[train_keep], k)
            flabels = model.predict(fold[validation_keep])
            reps = train_keep[weather.representative_ids(
                fold[train_keep], model.predict(fold[train_keep]), model.cluster_centers_,
            )]
            error = pair_errors(held_daily, validation_keep, reps[flabels])
            row[f"ari_omit_{start}_{start+4}"] = adjusted_rand_score(labels[k], flabels)
            for key, value in error.items():
                row[f"heldout_{start}_{key}"] = value
        del held_daily
    for row in rows:
        row["block_stability_ari_mean"] = float(np.mean([
            row[f"ari_omit_{start}_{start+4}"] for start in range(1996, 2026, 5)
        ]))
        for key in ("temperature_rmse_c", "precipitation_rmse_mm", "wet_day_disagreement"):
            row[f"heldout_mean_{key}"] = float(np.mean([
                row[f"heldout_{start}_{key}"] for start in range(1996, 2026, 5)
            ]))
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "cluster_comparison.csv", index=False)
    import joblib
    joblib.dump(models, args.output_dir / "preprocessing.joblib")
    return features, table, models


def write_map(path, grid, ids, values, units, categorical=False):
    weather.write_raster(path, grid, ids, values, units, categorical=categorical)
    import rasterio
    with rasterio.open(path, "r+") as target:
        target.update_tags(period=PERIOD, classification="30-year seNorge climate regions",
                           source="MET Norway seNorge_2018")


def final_groups(features, ids, grid, args):
    import joblib
    results = {}
    for k in (8, 32):
        target = args.output_dir / f"k{k}"
        target.mkdir(exist_ok=True)
        print(f"FIT FINAL {k} groups on {len(ids):,} complete land cells", flush=True)
        model = weather.fit_kmeans(features, k)
        labels = model.predict(features)
        representatives = weather.representative_ids(features, labels, model.cluster_centers_)
        results[k] = dict(model=model, labels=labels, representatives=representatives, target=target)
        joblib.dump(dict(model=model, representatives=representatives),
                    target / "model.joblib")
    return results


def export_results(results, ids, grid, valid, args):
    """Stream original daily fields once for both maps' exact representation errors."""
    import geopandas as gpd

    cache = args.output_dir / "cache"
    records = load_records(args.source)
    n = len(ids)
    statistics = {}
    annual_profiles = {k: [] for k in results}
    for k in results:
        statistics[k] = {
            name: np.zeros(n, dtype=np.float64) for name in (
                "temperature_square", "precipitation_square", "wet_disagreement",
                "monthly_precipitation_abs", "annual_precipitation_abs",
            )
        }
    mean_t = np.zeros(n, dtype=np.float64)
    mean_p = np.zeros(n, dtype=np.float64)
    seasonal_t = np.zeros((12, n), dtype=np.float64)
    seasonal_p = np.zeros_like(seasonal_t)
    indicator_sum = np.zeros((len(SEASONAL_NAMES), n), dtype=np.float64)
    days = 0
    for year in YEARS:
        print(f"EXACT REPRESENTATION ERRORS {year}: streaming daily fields", flush=True)
        monthly = np.load(cache / f"months_{year}.npy", mmap_mode="r")[:, :, valid]
        mt = monthly[:, weather.MI["temperature_mean"]]
        mp = monthly[:, weather.MI["precipitation_total"]]
        seasonal_t += mt
        seasonal_p += mp
        indicators = seasonal_indicators(monthly, year)
        indicator_sum += indicators
        mean_p += mp.sum(axis=0)
        indices = np.load(cache / f"annual_{year}.npy", mmap_mode="r")[:, valid]
        weights = pd.date_range(f"{year}-01-01", periods=12, freq="MS").days_in_month.to_numpy()
        annual_temperature = np.average(mt, axis=0, weights=weights)
        for k, result in results.items():
            reps = result["representatives"][result["labels"]]
            for label, rep in enumerate(result["representatives"]):
                members = result["labels"] == label
                profile = dict(
                    label=label, year=year, temperature_c=float(annual_temperature[members].mean()),
                    precipitation_mm=float(mp[:, members].sum(axis=0).mean()),
                    representative_temperature_c=float(annual_temperature[rep]),
                    representative_precipitation_mm=float(mp[:, rep].sum()),
                )
                for j, name in enumerate(ANNUAL_NAMES):
                    profile[name] = float(indices[j, members].mean())
                    profile[f"representative_{name}"] = float(indices[j, rep])
                for j, name in enumerate(SEASONAL_NAMES):
                    profile[name] = float(indicators[j, members].mean())
                    profile[f"representative_{name}"] = float(indicators[j, rep])
                annual_profiles[k].append(profile)
            statistics[k]["monthly_precipitation_abs"] += np.abs(mp - mp[:, reps]).sum(axis=0)
            statistics[k]["annual_precipitation_abs"] += np.abs(
                mp.sum(axis=0) - mp[:, reps].sum(axis=0),
            )
        path = Path(records[year]["netcdf"])
        receipt = json.loads((cache / f"{year}.json").read_text(encoding="utf-8"))
        stat = path.stat()
        if receipt["source"] != dict(path=str(path.resolve()), bytes=stat.st_size,
                                    mtime_ns=stat.st_mtime_ns):
            raise ValueError(f"{year}: original changed after feature extraction")
        with netCDF4.Dataset(path) as ds:
            for index in range(len(ds.dimensions["time"])):
                t = np.ma.masked_invalid(ds["tg"][index]).filled(np.nan).ravel()[ids]
                p = np.ma.masked_invalid(ds["rr"][index]).filled(np.nan).ravel()[ids]
                if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))):
                    raise ValueError("Invalid complete-cell value during error evaluation")
                mean_t += t
                for k, result in results.items():
                    reps = result["representatives"][result["labels"]]
                    statistics[k]["temperature_square"] += (t - t[reps])**2
                    statistics[k]["precipitation_square"] += (p - p[reps])**2
                    statistics[k]["wet_disagreement"] += (p >= 1) != (p[reps] >= 1)
                days += 1
    if days != len(pd.date_range("1996-01-01", "2025-12-31")):
        raise ValueError(f"Expected 10,958 daily samples; found {days}")
    mean_t /= days
    mean_p /= len(YEARS)
    seasonal_t /= len(YEARS)
    seasonal_p /= len(YEARS)
    indicator_mean = indicator_sum / len(YEARS)
    for name, value in zip(SEASONAL_NAMES, indicator_mean):
        write_map(args.output_dir / f"{name}_1km.tif", grid, ids, value,
                  "days/year" if name == "frost_days_per_year" else "degC")
    for k, result in results.items():
        target, labels, reps = result["target"], result["labels"], result["representatives"]
        order = np.argsort([mean_t[labels == label].mean() for label in range(k)])
        numbering = np.empty(k, dtype=int)
        numbering[order] = np.arange(1, k + 1)
        groups = numbering[labels]
        raw = statistics[k]
        errors = dict(
            temperature_rmse_c=np.sqrt(raw["temperature_square"] / days),
            precipitation_rmse_mm=np.sqrt(raw["precipitation_square"] / days),
            wet_day_disagreement=raw["wet_disagreement"] / days,
            monthly_precipitation_mae_mm=raw["monthly_precipitation_abs"] / (12 * len(YEARS)),
            annual_precipitation_mae_mm=raw["annual_precipitation_abs"] / len(YEARS),
        )
        write_map(target / "climate_regions_1km.tif", grid, ids, groups, "region_id", True)
        for name, value in errors.items():
            write_map(target / f"{name}_1km.tif", grid, ids, value,
                      "degC" if name == "temperature_rmse_c" else
                      "fraction" if name == "wet_day_disagreement" else "mm")
        rows, profiles = [], []
        for label in order:
            members = np.flatnonzero(labels == label)
            rep = reps[label]
            row, col = divmod(int(ids[rep]), grid["width"])
            summary = dict(
                region_id=int(numbering[label]), cells=len(members),
                representative_x=grid["west"] + (col + 0.5) * 1000,
                representative_y=grid["north"] - (row + 0.5) * 1000,
                mean_temperature_c=float(mean_t[members].mean()),
                mean_annual_precipitation_mm=float(mean_p[members].mean()),
            )
            for name, value in errors.items():
                summary[f"{name}_mean"] = float(value[members].mean())
                summary[f"{name}_p95"] = float(np.percentile(value[members], 95))
            for name, value in zip(SEASONAL_NAMES, indicator_mean):
                summary[name] = float(value[members].mean())
                summary[f"representative_{name}"] = float(value[rep])
            rows.append(summary)
            for month in range(12):
                profiles.append(dict(
                    region_id=int(numbering[label]), month=month + 1,
                    temperature_c=float(seasonal_t[month, members].mean()),
                    precipitation_mm=float(seasonal_p[month, members].mean()),
                    representative_temperature_c=float(seasonal_t[month, rep]),
                    representative_precipitation_mm=float(seasonal_p[month, rep]),
                ))
        frame = pd.DataFrame(rows)
        frame.to_csv(target / "region_summary.csv", index=False)
        annual_frame = pd.DataFrame(annual_profiles[k])
        annual_frame["region_id"] = numbering[annual_frame.pop("label").to_numpy()]
        annual_frame.to_csv(target / "annual_profiles.csv", index=False)
        annual_frame["decade_start"] = 1996 + ((annual_frame["year"] - 1996) // 10) * 10
        annual_frame.drop(columns="year").groupby(["region_id", "decade_start"]).mean().to_csv(
            target / "decade_profiles.csv")
        pd.DataFrame(profiles).to_csv(target / "seasonal_profiles.csv", index=False)
        gpd.GeoDataFrame(
            frame, geometry=gpd.points_from_xy(frame.representative_x, frame.representative_y),
            crs=grid["crs"],
        ).to_file(target / "representatives.gpkg", layer="representatives", driver="GPKG")
        write_json(target / "analysis.json", dict(
            period=PERIOD, status="complete", groups=k, complete_cells=n, days=days, grid=grid,
            mean_representation_errors={name: float(value.mean()) for name, value in errors.items()},
            limitations=["Gridded product similarity, not independent station accuracy.",
                         "Spells and rolling rainfall maxima are bounded to calendar years.",
                         "No wind, humidity or measured snow inputs.",
                         "Independent 8/32-group fits, not nested partitions.",
                         "Daily timing uses seeded temporal projection then PCA, not exact full-series PCA."],
        ))
        plot_map(target, grid, ids, groups, frame, errors, k)


def plot_map(target, grid, ids, groups, frame, errors, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    raster = np.full(grid["height"] * grid["width"], np.nan)
    raster[ids] = groups
    cmap = ListedColormap(plt.get_cmap("tab20" if k <= 20 else "gist_ncar")(np.linspace(0, 1, k)))
    extent = (grid["west"], grid["west"] + grid["width"] * 1000,
              grid["north"] - grid["height"] * 1000, grid["north"])
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set_facecolor("#e6e6e6")
    image = ax.imshow(raster.reshape(grid["height"], grid["width"]), extent=extent,
                      cmap=cmap, norm=BoundaryNorm(np.arange(0.5, k + 1.5), k))
    ax.scatter(frame.representative_x, frame.representative_y, marker="x", c="black", s=18)
    ax.set_title(f"Norway climate regions {PERIOD}: {k} groups")
    fig.colorbar(image, ax=ax, ticks=np.arange(1, k + 1), label="Region ID (cold to warm)")
    fig.tight_layout()
    fig.savefig(target / "climate_regions.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    for ax, name in zip(axes, ("temperature_rmse_c", "wet_day_disagreement")):
        raster[:] = np.nan
        raster[ids] = errors[name]
        image = ax.imshow(raster.reshape(grid["height"], grid["width"]), extent=extent,
                          cmap="magma", vmin=0, vmax=np.percentile(errors[name], 99))
        ax.set_title(name)
        fig.colorbar(image, ax=ax, shrink=0.65, extend="max")
    fig.tight_layout()
    fig.savefig(target / "representation_errors.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "senorge_norway_1996_2025")
    parser.add_argument("--polygon", type=Path, default=ROOT / "outputs" / "norway_polygon.gpkg")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--training-cells", type=int, default=8000)
    parser.add_argument("--validation-cells", type=int, default=2000)
    parser.add_argument("--candidates", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--wait-for-download", action="store_true")
    args = parser.parse_args(argv)
    args.candidates = sorted(set(args.candidates))
    if min(args.training_cells, args.validation_cells) <= max([32] + args.candidates) or args.threads < 1:
        parser.error("Training and validation samples must exceed K; threads must be positive")
    if any(k < 2 for k in args.candidates):
        parser.error("All candidate counts must be at least two")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "analysis.json").exists():
        parser.error("Analysis is already complete; use a different output directory")
    with threadpool_limits(limits=args.threads):
        grid, raw_ids, sample, valid = extract_all(args)
        if valid.sum() < args.training_cells + args.validation_cells:
            raise ValueError("Too few complete cells for requested samples")
        print(f"COMPLETE THIRTY-YEAR DATA: {valid.sum():,} of {len(valid):,} land cells", flush=True)
        features, table, models = compare(args.output_dir / "cache", valid, sample, args)
        results = final_groups(features, raw_ids[valid], grid, args)
        export_results(results, raw_ids[valid], grid, valid, args)
        write_json(args.output_dir / "analysis.json", dict(
            status="complete", period=PERIOD, grid=grid, complete_cells=int(valid.sum()),
            excluded_incomplete_cells=int((~valid).sum()), feature_families=list(FAMILIES),
            family_weights=[1 / len(FAMILIES)] * len(FAMILIES),
            annual_features=list(ANNUAL_NAMES) + ["frost_days_tn_lt0"],
            seasonal_indicators=list(SEASONAL_NAMES),
            summer_winter_amplitude="Annual day-weighted JJA mean minus Jan/Feb/Dec mean; "
                                    "30-year mean added explicitly to seasonal-temperature features.",
            frost_days="Annual count with tn < 0 C; mean and interannual standard deviation "
                       "added explicitly to annual-index features, alongside monthly frost frequency.",
            temporal_projection_dimensions=SKETCH_SIZE, groups=[8, 32],
            days=10958, normal_period_note="Recent 30-year period, not the standard 1991-2020 normal.",
            source="MET Norway seNorge_2018",
            validation="Six held-out five-year blocks with refitted preprocessing, PCA and clusters; "
                       "error evaluated on held-out daily weather. Not independent station validation.",
        ))
    print(f"COMPLETE: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
