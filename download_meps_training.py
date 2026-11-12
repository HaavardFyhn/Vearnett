"""Download small MEPS forecast subsets matched to local Frost observations."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
import sys
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time as time_module

import numpy as np
import pandas as pd
import pyproj
import requests
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from pipeline_config import DATABASE, ROOT, WORK


ARCHIVE = "https://thredds.met.no/thredds/ncss/grid/meps25epsarchive"
LEGACY_LAST_DAY = date(2023, 9, 30)
PHENOMENA = {
    "temperature": ("air_temperature", "air_temperature_2m", "degC"),
    "humidity": ("relative_humidity", "relative_humidity_2m", "percent"),
}
RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat,
                        help="First forecast-run date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, type=date.fromisoformat,
                        help="Last date, exclusive (YYYY-MM-DD)")
    parser.add_argument("--phenomena", nargs="+", choices=PHENOMENA,
                        default=["temperature", "humidity"])
    parser.add_argument("--cycles", nargs="+", type=int, choices=[0, 6, 12, 18], default=[0],
                        help="UTC forecast cycles; 00 only is a reasonable first pilot")
    parser.add_argument("--lead-hours", type=int, default=12,
                        help="Forecast lead to match with Frost observations (default: 12)")
    parser.add_argument("--bbox", type=float, nargs=4,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="Optional small-area bounds in lon/lat degrees")
    parser.add_argument("--observation-tolerance-minutes", type=int, default=60)
    parser.add_argument("--request-delay", type=float, default=1.0,
                        help="Minimum seconds between archive requests; do not run parallel sessions")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-response-mb", type=int, default=250,
                        help="Reject unexpectedly large responses instead of saving a full grid")
    parser.add_argument("--database", type=Path, default=None,
                        help="Output SQLite file; defaults to outputs/meps_training_<dates>.sqlite")
    parser.add_argument("--save-first-grid-dir", type=Path,
                        help="Save the first successful regional NetCDF subset and ArcGIS-ready GeoTIFF layers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.start >= args.end:
        parser.error("--end must be later than --start")
    if not 0 <= args.lead_hours <= 66:
        parser.error("MEPS lead time must be between 0 and 66 hours")
    if args.observation_tolerance_minutes < 0 or args.request_delay < 0 or args.timeout < 1:
        parser.error("Tolerance/delay must be nonnegative and timeout positive")
    if args.retries < 0 or args.max_response_mb < 1:
        parser.error("Retries must be nonnegative and maximum response size positive")
    if args.bbox is not None:
        west, south, east, north = args.bbox
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            parser.error("--bbox must be WEST SOUTH EAST NORTH with valid increasing lon/lat bounds")
    args.cycles = sorted(set(args.cycles))
    args.phenomena = list(dict.fromkeys(args.phenomena))
    if args.database is None:
        args.database = WORK / f"meps_training_{args.start}_{args.end}.sqlite"
    return args


def run_times(start, end, cycles):
    current = start
    while current < end:
        for cycle in cycles:
            yield datetime.combine(current, time(cycle), tzinfo=timezone.utc)
        current += timedelta(days=1)


def dataset_path(run_time):
    stamp = run_time.strftime("%Y%m%dT%HZ")
    folder = run_time.strftime("%Y/%m/%d")
    if run_time.date() <= LEGACY_LAST_DAY:
        filename = f"meps_det_2_5km_{stamp}.nc"
    else:
        filename = f"meps_det_sfc_{stamp}.ncml"
    return f"{folder}/{filename}"


def ncss_url(run_time):
    return f"{ARCHIVE}/{dataset_path(run_time)}"


def init_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("""CREATE TABLE IF NOT EXISTS training_samples (
            run_time TEXT NOT NULL,
            valid_time TEXT NOT NULL,
            lead_hours INTEGER NOT NULL,
            phenomenon TEXT NOT NULL,
            station_id TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            observation_time TEXT NOT NULL,
            observation REAL NOT NULL,
            forecast REAL NOT NULL,
            bias_observation_minus_forecast REAL NOT NULL,
            forecast_variable TEXT NOT NULL,
            forecast_units TEXT NOT NULL,
            PRIMARY KEY(run_time, phenomenon, station_id)
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS run_status (
            run_time TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            detail TEXT,
            updated TEXT NOT NULL
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")
        connection.commit()
    return path


def run_status(connection, run_time):
    row = connection.execute("SELECT status FROM run_status WHERE run_time=?",
                             (run_time.isoformat(),)).fetchone()
    return None if row is None else row[0]


def mark_status(connection, run_time, status, detail=None):
    connection.execute("INSERT OR REPLACE INTO run_status VALUES(?,?,?,?)",
        (run_time.isoformat(), status, detail, datetime.now(timezone.utc).isoformat()))
    connection.commit()


def observations_near(connection, run_time, phenomena, lead_hours, tolerance_minutes, bbox=None):
    valid_time = run_time + timedelta(hours=lead_hours)
    tolerance = timedelta(minutes=tolerance_minutes)
    start = (valid_time - tolerance).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end = (valid_time + tolerance).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    elements = [PHENOMENA[name][0] for name in phenomena]
    placeholders = ",".join("?" for _ in elements)
    spatial_filter = ""
    spatial_parameters = ()
    if bbox is not None:
        west, south, east, north = bbox
        spatial_filter = " AND s.lon>=? AND s.lat>=? AND s.lon<=? AND s.lat<=?"
        spatial_parameters = (west, south, east, north)
    query = f"""SELECT o.station_id,o.reference_time,o.element,o.value,s.lat,s.lon
        FROM observations o JOIN stations s ON s.station_id=o.station_id
        WHERE s.inside_norway=1 AND o.value IS NOT NULL
          AND o.element IN ({placeholders}) AND o.reference_time>=? AND o.reference_time<=?{spatial_filter}"""
    records = connection.execute(query, (*elements, start, end, *spatial_parameters)).fetchall()
    closest = {}
    for station_id, reference_time, element, value, latitude, longitude in records:
        timestamp = pd.to_datetime(reference_time, utc=True)
        offset = abs(timestamp - pd.Timestamp(valid_time))
        key = (station_id, element)
        if key not in closest or offset < closest[key]["offset"]:
            closest[key] = dict(station_id=station_id, observation_time=timestamp.isoformat(),
                element=element, observation=float(value), latitude=float(latitude),
                longitude=float(longitude), offset=offset)
    return valid_time, closest


def request_forecast(session, run_time, valid_time, observations, phenomena, args, bbox=None):
    stations = list({item["station_id"]: item for item in observations.values()}.values())
    margin = .15
    if bbox is None:
        west = max(-180., min(item["longitude"] for item in stations) - margin)
        south = max(-90., min(item["latitude"] for item in stations) - margin)
        east = min(180., max(item["longitude"] for item in stations) + margin)
        north = min(90., max(item["latitude"] for item in stations) + margin)
    else:
        west, south, east, north = bbox
    params = {
        "var": ",".join(PHENOMENA[name][1] for name in phenomena),
        "north": north,
        "south": south,
        "west": west,
        "east": east,
        "time": valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "accept": "netCDF",
    }
    url = ncss_url(run_time)
    max_bytes = args.max_response_mb * 1024 * 1024
    for attempt in range(args.retries + 1):
        response = None
        try:
            response = session.get(url, params=params, timeout=(20, args.timeout), stream=True)
        except requests.RequestException:
            if attempt >= args.retries:
                raise
            time_module.sleep(min(5 * 2 ** attempt, 60))
            continue
        try:
            if response.status_code == 404:
                return None, "not_found"
            if response.status_code in RETRY_STATUSES:
                if attempt >= args.retries:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(5 * 2 ** attempt, 60)
                time_module.sleep(delay)
                continue
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise ValueError(f"Subset response declares {int(declared)/1024/1024:.1f} MB; "
                                 f"limit is {args.max_response_mb} MB. Check the spatial/time subset.")
            chunks, received = [], 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                received += len(chunk)
                if received > max_bytes:
                    raise ValueError(f"Subset response exceeded {args.max_response_mb} MB. "
                                     "Stopped reading to avoid downloading a full grid.")
                chunks.append(chunk)
            return b"".join(chunks), "ok"
        except requests.HTTPError:
            raise
        except requests.RequestException:
            if attempt >= args.retries:
                raise
            time_module.sleep(min(5 * 2 ** attempt, 60))
        finally:
            if response is not None:
                response.close()
    raise RuntimeError("MEPS request retries exhausted")


def nearest_grid_values(content, observations, phenomena):
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as temporary:
        temp_path = Path(temporary.name)
        temporary.write(content)
    try:
        with xr.open_dataset(temp_path, engine="scipy", decode_times=True) as dataset:
            if not len(dataset.x) or not len(dataset.y):
                raise ValueError("MEPS subset returned an empty grid")
            if "projection_lambert" in dataset:
                projection = dataset["projection_lambert"].attrs
            else:
                mapping = next((dataset[PHENOMENA[name][1]].attrs.get("grid_mapping")
                                for name in phenomena if PHENOMENA[name][1] in dataset), None)
                if not mapping or mapping not in dataset:
                    raise ValueError("MEPS subset did not include its Lambert grid mapping")
                projection = dataset[mapping].attrs
            transformer = pyproj.Transformer.from_crs(4326, pyproj.CRS.from_cf(projection), always_xy=True)
            x_values, y_values = dataset.x.to_numpy(), dataset.y.to_numpy()
            indices = {}
            for item in observations.values():
                x, y = transformer.transform(item["longitude"], item["latitude"])
                ix, iy = int(np.abs(x_values-x).argmin()), int(np.abs(y_values-y).argmin())
                if abs(x_values[ix]-x) > 5000 or abs(y_values[iy]-y) > 5000:
                    continue
                indices[item["station_id"]] = (iy, ix)
            forecasts = {}
            for name in phenomena:
                variable_name = PHENOMENA[name][1]
                if variable_name not in dataset:
                    raise ValueError(f"MEPS variable {variable_name} is missing")
                variable = dataset[variable_name]
                if "time" in variable.dims and variable.sizes["time"] != 1:
                    target = pd.Timestamp(next(iter(observations.values()))["observation_time"])
                    times = pd.to_datetime(dataset.time.to_numpy(), utc=True)
                    variable = variable.isel(time=int(np.abs(times-target).argmin()))
                elif "time" in variable.dims:
                    variable = variable.isel(time=0)
                for dimension in tuple(variable.dims):
                    if dimension not in ("y", "x"):
                        if variable.sizes[dimension] != 1:
                            raise ValueError(f"MEPS variable {variable_name} has an unexpected {dimension} dimension")
                        variable = variable.isel({dimension: 0})
                if set(variable.dims) != {"y", "x"}:
                    raise ValueError(f"MEPS variable {variable_name} is not a horizontal grid")
                variable = variable.transpose("y", "x")
                units = str(variable.attrs.get("units", ""))
                values = variable.to_numpy()
                for station_id, (iy, ix) in indices.items():
                    value = float(values[iy, ix])
                    if not np.isfinite(value) or abs(value) > 1e10:
                        continue
                    if name == "temperature" and units.lower() in ("k", "kelvin"):
                        value -= 273.15
                    if name == "humidity" and units.lower() in ("1", "fraction", ""):
                        value *= 100
                    forecasts[(name, station_id)] = (value, units)
            return forecasts
    finally:
        temp_path.unlink(missing_ok=True)


def write_grid_geotiff(dataset, phenomenon, output_path, run_time, valid_time, lead_hours):
    variable_name = PHENOMENA[phenomenon][1]
    if variable_name not in dataset:
        raise ValueError(f"MEPS variable {variable_name} is missing")
    variable = dataset[variable_name]
    if "time" in variable.dims:
        if variable.sizes["time"] > 1:
            target = pd.Timestamp(valid_time)
            times = pd.to_datetime(dataset.time.to_numpy(), utc=True)
            variable = variable.isel(time=int(np.abs(times - target).argmin()))
        else:
            variable = variable.isel(time=0)
    for dimension in tuple(variable.dims):
        if dimension not in ("y", "x"):
            if variable.sizes[dimension] != 1:
                raise ValueError(f"MEPS variable {variable_name} has an unexpected {dimension} dimension")
            variable = variable.isel({dimension: 0})
    if set(variable.dims) != {"y", "x"}:
        raise ValueError(f"MEPS variable {variable_name} is not a horizontal grid")

    mapping = variable.attrs.get("grid_mapping")
    if "projection_lambert" in dataset:
        projection = dataset["projection_lambert"].attrs
    elif mapping and mapping in dataset:
        projection = dataset[mapping].attrs
    else:
        raise ValueError("MEPS subset did not include its Lambert grid mapping")
    crs = pyproj.CRS.from_cf(projection)

    x_values = np.asarray(dataset.x.to_numpy(), dtype=np.float64)
    y_values = np.asarray(dataset.y.to_numpy(), dtype=np.float64)
    if x_values.ndim != 1 or y_values.ndim != 1 or len(x_values) < 2 or len(y_values) < 2:
        raise ValueError("MEPS grid export requires one-dimensional x/y axes with at least two cells")
    dx_values, dy_values = np.diff(x_values), np.diff(y_values)
    if (not np.allclose(dx_values, dx_values[0], rtol=1e-5, atol=1e-3)
            or not np.allclose(dy_values, dy_values[0], rtol=1e-5, atol=1e-3)):
        raise ValueError("MEPS grid x/y axes are not regularly spaced")
    dx, dy = float(dx_values[0]), float(dy_values[0])

    values = np.asarray(variable.transpose("y", "x").to_numpy(), dtype=np.float32)
    if dx < 0:
        x_values = x_values[::-1]
        values = values[:, ::-1]
    if dy > 0:
        y_values = y_values[::-1]
        values = values[::-1, :]
    pixel_width, pixel_height = abs(dx), abs(dy)
    transform = from_origin(x_values[0] - pixel_width / 2,
                            y_values[0] + pixel_height / 2,
                            pixel_width, pixel_height)

    units = str(variable.attrs.get("units", ""))
    if phenomenon == "temperature" and units.lower() in ("k", "kelvin"):
        values -= 273.15
        units = "degC"
    elif phenomenon == "humidity" and units.lower() in ("1", "fraction", ""):
        values *= 100
        units = "percent"
    invalid = ~np.isfinite(values) | (np.abs(values) > 1e10)
    values[invalid] = -9999.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Grid GeoTIFF already exists: {output_path}")
    profile = {
        "driver": "GTiff", "width": len(x_values), "height": len(y_values),
        "count": 1, "dtype": "float32", "crs": crs, "transform": transform,
        "nodata": -9999.0, "compress": "deflate", "predictor": 3,
    }
    with rasterio.open(output_path, "w", **profile) as target:
        target.write(values, 1)
        target.update_tags(
            phenomenon=phenomenon,
            source_variable=variable_name,
            units=units,
            run_time=run_time.isoformat(),
            valid_time=valid_time.isoformat(),
            lead_hours=str(lead_hours),
            source="MET Norway MEPS archive NCSS subset",
        )
    return output_path


def save_grid_snapshot(content, phenomena, output_dir, run_time, valid_time, lead_hours):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = run_time.strftime("%Y%m%dT%HZ")
    netcdf_path = output_dir / f"meps_grid_{stamp}.nc"
    if netcdf_path.exists():
        raise FileExistsError(f"Grid NetCDF already exists: {netcdf_path}")
    pending = netcdf_path.with_name(netcdf_path.stem + ".pending.nc")
    pending.write_bytes(content)
    pending.replace(netcdf_path)

    raster_paths = []
    with xr.open_dataset(netcdf_path, engine="scipy", decode_times=True) as dataset:
        for phenomenon in phenomena:
            filename = output_dir / f"meps_{phenomenon}_{stamp}.tif"
            raster_paths.append(write_grid_geotiff(
                dataset, phenomenon, filename, run_time, valid_time, lead_hours,
            ))
    metadata = {
        "run_time": run_time.isoformat(),
        "valid_time": valid_time.isoformat(),
        "lead_hours": lead_hours,
        "phenomena": phenomena,
        "netcdf": str(netcdf_path),
        "geotiffs": [str(path) for path in raster_paths],
        "grid_crs": "MEPS native Lambert conformal conic",
    }
    netcdf_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return netcdf_path, raster_paths


def collect(args):
    output = init_output(args.database)
    selected = sorted(run_times(args.start, args.end, args.cycles))
    config = dict(start=args.start.isoformat(), end=args.end.isoformat(), cycles=args.cycles,
                  lead_hours=args.lead_hours, phenomena=args.phenomena,
                  tolerance_minutes=args.observation_tolerance_minutes,
                  bbox=args.bbox)
    with closing(sqlite3.connect(DATABASE.resolve().as_uri() + "?mode=ro", uri=True)) as frost, \
            closing(sqlite3.connect(output)) as destination, requests.Session() as session:
        session.headers.update({"User-Agent": "pipeline-v3-meps-training/1.0 (sequential archive subset)"})
        existing = destination.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
        if existing is not None and json.loads(existing[0]) != config:
            raise ValueError(f"Output database {output} already belongs to a different MEPS query configuration")
        destination.execute("INSERT OR IGNORE INTO metadata VALUES('config',?)", (json.dumps(config, sort_keys=True),))
        destination.commit()
        grid_saved = False
        for index, run_time in enumerate(selected, 1):
            if run_status(destination, run_time) in ("complete", "no_observations", "not_found"):
                print(f"SKIP {index}/{len(selected)} {run_time.isoformat()} already handled", flush=True)
                continue
            valid_time, observations = observations_near(
                frost, run_time, args.phenomena, args.lead_hours,
                args.observation_tolerance_minutes, args.bbox,
            )
            if not observations:
                mark_status(destination, run_time, "no_observations")
                print(f"SKIP {index}/{len(selected)} {run_time.isoformat()} no matching Frost observations", flush=True)
                continue
            try:
                content, status = request_forecast(session, run_time, valid_time, observations,
                                                   args.phenomena, args, args.bbox)
                if status == "not_found":
                    mark_status(destination, run_time, "not_found")
                    print(f"SKIP {index}/{len(selected)} {run_time.isoformat()} forecast unavailable", flush=True)
                    continue
                forecasts = nearest_grid_values(content, observations, args.phenomena)
                if args.save_first_grid_dir is not None and not grid_saved:
                    netcdf_path, raster_paths = save_grid_snapshot(
                        content, args.phenomena, args.save_first_grid_dir,
                        run_time, valid_time, args.lead_hours,
                    )
                    grid_saved = True
                    print(f"Saved MEPS grid snapshot: {netcdf_path}", flush=True)
                    for raster_path in raster_paths:
                        print(f"ArcGIS GeoTIFF: {raster_path}", flush=True)
                rows = []
                for (name, station_id), (forecast, units) in forecasts.items():
                    item = observations.get((station_id, PHENOMENA[name][0]))
                    if item is None:
                        continue
                    rows.append((run_time.isoformat(), valid_time.isoformat(), args.lead_hours,
                        name, station_id, item["latitude"], item["longitude"], item["observation_time"],
                        item["observation"], forecast, item["observation"]-forecast,
                        PHENOMENA[name][1], units))
                destination.executemany("INSERT OR REPLACE INTO training_samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                mark_status(destination, run_time, "complete", f"samples={len(rows)}")
                print(f"OK {index}/{len(selected)} {run_time.isoformat()} valid={valid_time.isoformat()} rows={len(rows)}", flush=True)
            except requests.HTTPError as error:
                code = error.response.status_code if error.response is not None else "HTTP"
                mark_status(destination, run_time, "error", f"HTTP {code}: {error}")
                raise
            except Exception as error:
                mark_status(destination, run_time, "error", f"{type(error).__name__}: {error}")
                raise
            if index < len(selected):
                time_module.sleep(args.request_delay)
    return output


def main(argv=None):
    args = parse_args(argv)
    runs = list(run_times(args.start, args.end, args.cycles))
    print(f"Forecast runs: {len(runs)} · phenomena: {', '.join(args.phenomena)} · "
          f"cycle(s): {', '.join(f'{cycle:02d}Z' for cycle in args.cycles)} · "
          f"lead: +{args.lead_hours} h · at most one sequential regional request per run")
    print(f"Frost observations: {DATABASE} · output: {args.database}")
    if args.dry_run:
        for run_time in runs[:3]:
            print(f"{run_time.isoformat()} -> {ncss_url(run_time)}")
        if len(runs) > 3:
            print(f"... and {len(runs)-3} more runs; no network requests made")
        return 0
    try:
        output = collect(args)
    except requests.RequestException as error:
        print(f"MEPS archive request failed after retries: {error}", file=sys.stderr, flush=True)
        print(f"Completed runs remain saved in {args.database}. Rerun the same command "
              "when the archive is available; the failed run will be retried.", file=sys.stderr, flush=True)
        return 1
    print(f"Training samples saved to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())