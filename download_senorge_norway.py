"""Download whole-country daily seNorge_2018 grids from MET Norway."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import re
import sys
import time

# Load network and NetCDF DLLs before Rasterio in the ArcGIS environment.
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import netCDF4
import numpy as np

from download_nve_grid import NODATA, write_json


BASE_URL = "https://thredds.met.no/thredds/fileServer/senorge/seNorge_2018/Latest"
ARCHIVE_URL = BASE_URL.rsplit("/", 1)[0] + "/Archive"
VARIABLES = {"temperature": "tg", "precipitation": "rr"}
MAX_BYTES = 100 * 1024 * 1024


def daily_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch_file(session: requests.Session, day: date, path: Path) -> str:
    url = f"{BASE_URL}/seNorge2018_{day:%Y%m%d}.nc"
    if path.exists():
        return url
    pending = path.with_suffix(".pending.nc")
    try:
        with session.get(url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            declared = int(response.headers.get("Content-Length", 0))
            if declared > MAX_BYTES:
                raise ValueError(f"Daily file exceeds {MAX_BYTES} bytes: {url}")
            received = 0
            with pending.open("wb") as target:
                for chunk in response.iter_content(1024 * 1024):
                    received += len(chunk)
                    if received > MAX_BYTES:
                        raise ValueError(f"Daily download exceeded {MAX_BYTES} bytes: {url}")
                    target.write(chunk)
            if declared and received != declared:
                raise ValueError(f"Incomplete NetCDF download: {received} of {declared} bytes")
        with netCDF4.Dataset(pending) as dataset:
            if not {"X", "Y", "time", "tg", "rr"}.issubset(dataset.variables):
                raise ValueError("Downloaded file is missing required seNorge variables")
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()
    return url


def export_day(path: Path, day: date, output: Path, source: str, *, time_index: int = 0) -> list[dict]:
    import rasterio
    from rasterio.transform import from_origin

    summaries = []
    with netCDF4.Dataset(path) as dataset:
        times = dataset.variables["time"]
        decoded = netCDF4.num2date(times[time_index], times.units,
                                  calendar=getattr(times, "calendar", "standard"))
        if (decoded.year, decoded.month, decoded.day) != (
            day.year, day.month, day.day
        ):
            raise ValueError(f"NetCDF time does not match requested day {day}")
        timestamp = decoded.isoformat()
        xs = np.asarray(dataset.variables["X"][:], dtype=np.float64)
        ys = np.asarray(dataset.variables["Y"][:], dtype=np.float64)
        if xs.ndim != 1 or ys.ndim != 1 or len(xs) < 2 or len(ys) < 2:
            raise ValueError("Expected one-dimensional projected grid coordinates")
        for axis in (xs, ys):
            if not np.all(np.isfinite(axis)) or not (
                np.allclose(np.diff(axis), 1000, rtol=0, atol=1e-6) or
                np.allclose(np.diff(axis), -1000, rtol=0, atol=1e-6)
            ):
                raise ValueError("seNorge coordinates do not form a regular native 1 km grid")
        transform = from_origin(float(xs.min() - 500), float(ys.max() + 500), 1000, 1000)
        for phenomenon, variable_name in VARIABLES.items():
            variable = dataset.variables[variable_name]
            if variable.dimensions != ("time", "Y", "X"):
                raise ValueError(f"Unexpected dimensions for {variable_name}: {variable.dimensions}")
            projection = dataset.variables[variable.grid_mapping]
            crs = rasterio.crs.CRS.from_string(projection.proj4)
            if crs.to_epsg() != 32633:
                raise ValueError(f"Unexpected source projection: {crs}")
            values = np.ma.asarray(variable[time_index], dtype=np.float32)
            if xs[0] > xs[-1]:
                values = values[:, ::-1]
            if ys[0] < ys[-1]:
                values = values[::-1, :]
            values = np.ma.masked_invalid(values)
            if not values.count():
                raise ValueError(f"All {variable_name} cells are missing for {day}")
            source_units = variable.units
            if phenomenon == "temperature":
                if source_units != "Celsius":
                    raise ValueError(f"Unexpected temperature units: {source_units}")
                units = "degC"
            else:
                if source_units != "kg/m^2":
                    raise ValueError(f"Unexpected precipitation units: {source_units}")
                # 1 kg of liquid water per square metre is 1 mm water depth.
                units = "mm"
            raster_path = output / f"senorge_{variable_name}_{day:%Y%m%d}_1km.tif"
            pending = raster_path.with_suffix(".pending.tif")
            try:
                with rasterio.open(
                    pending, "w", driver="GTiff", height=len(ys), width=len(xs),
                    count=1, dtype="float32", crs=crs, transform=transform,
                    nodata=NODATA, compress="deflate", predictor=3, tiled=True,
                ) as target:
                    target.write(values.filled(NODATA), 1)
                    target.set_band_description(1, f"{phenomenon} [{units}] {day}")
                    target.update_tags(
                        source=source, source_product="seNorge_2018",
                        source_variable=variable_name, source_units=source_units, units=units,
                        valid_time=timestamp, interval=variable.long_name,
                        institution=dataset.institution,
                        product_version=getattr(variable, "version", dataset.version),
                        license=dataset.license, attribution="MET Norway",
                    )
                pending.replace(raster_path)
            finally:
                if pending.exists():
                    pending.unlink()
            summaries.append(dict(
                date=day.isoformat(), valid_time=timestamp, phenomenon=phenomenon,
                variable=variable_name, units=units, source_units=source_units,
                minimum=float(values.min()), maximum=float(values.max()),
                valid_cells=int(values.count()), nodata_cells=int(values.size - values.count()),
                width=len(xs), height=len(ys), cell_m=1000, crs=crs.to_string(),
                bounds=[float(xs.min() - 500), float(ys.min() - 500),
                        float(xs.max() + 500), float(ys.max() + 500)],
                geotiff=str(raster_path.resolve()), netcdf=str(path.resolve()), source=source,
            ))
            print(f"{day} {phenomenon}: {values.count()} valid cells; "
                  f"{float(values.min()):.2f} to {float(values.max()):.2f} {units}; "
                  f"{raster_path}", flush=True)
    return summaries


def validate_archive(path: Path, year: int) -> dict:
    with netCDF4.Dataset(path) as dataset:
        expected = list(daily_dates(date(year, 1, 1), date(year, 12, 31)))
        times = dataset.variables["time"]
        decoded = netCDF4.num2date(times[:], times.units,
                                  calendar=getattr(times, "calendar", "standard"))
        actual = [date(value.year, value.month, value.day) for value in decoded]
        if actual != expected or any(
            right - left != timedelta(days=1) for left, right in zip(decoded, decoded[1:])
        ):
            raise ValueError(f"{year} archive does not contain every consecutive day of the year")
        for axis in ("X", "Y"):
            coordinates = np.asarray(dataset.variables[axis][:])
            if coordinates.ndim != 1 or len(coordinates) < 2 or not np.all(np.isfinite(coordinates)):
                raise ValueError(f"Invalid annual {axis} coordinates")
            delta = np.diff(coordinates)
            if not (np.allclose(delta, 1000, rtol=0, atol=1e-6) or
                    np.allclose(delta, -1000, rtol=0, atol=1e-6)):
                raise ValueError(f"Annual {axis} spacing is not 1 km")
        variables = {}
        for name, units in (("tg", "Celsius"), ("rr", "kg/m^2")):
            variable = dataset.variables[name]
            if variable.dimensions != ("time", "Y", "X") or variable.units != units:
                raise ValueError(f"Unexpected annual {name} dimensions/units")
            mapping = dataset.variables[variable.grid_mapping]
            if mapping.grid_mapping_name != "transverse_mercator" or mapping.utm_zone_number != 33:
                raise ValueError("Annual archive does not use UTM zone 33")
            # Read each compressed daily slab to detect corruption without loading the year.
            counts = []
            minimum, maximum = float("inf"), float("-inf")
            for index in range(len(expected)):
                values = np.ma.masked_invalid(variable[index])
                if not values.count():
                    raise ValueError(f"All {name} cells are missing on {expected[index]}")
                counts.append(int(values.count()))
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))
            variables[name] = dict(units=units, minimum=minimum, maximum=maximum,
                                   min_valid_cells=min(counts), max_valid_cells=max(counts))
        return dict(year=year, days=len(expected), first=decoded[0].isoformat(),
                    last=decoded[-1].isoformat(), width=len(dataset.dimensions["X"]),
                    height=len(dataset.dimensions["Y"]), cell_m=1000, variables=variables,
                    netcdf=str(path.resolve()), bytes=path.stat().st_size,
                    source=f"{ARCHIVE_URL}/seNorge2018_{year}.nc")


def fetch_archive(session: requests.Session, year: int, path: Path) -> None:
    if path.exists():
        return
    url = f"{ARCHIVE_URL}/seNorge2018_{year}.nc"
    pending = path.with_suffix(".pending.nc")
    receipt = path.with_suffix(".transfer.json")
    for attempt in range(4):
        state = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
        offset = pending.stat().st_size if pending.exists() else 0
        validator = state.get("validator") if state.get("url") == url else None
        if offset and offset == state.get("total") and validator:
            pending.replace(path)
            return
        headers = {"Accept-Encoding": "identity"}
        if offset and validator:
            headers.update({"Range": f"bytes={offset}-", "If-Range": validator})
        else:
            offset = 0
        try:
            with session.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                length = int(response.headers.get("Content-Length", 0))
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[2]) + 1 != int(match[3]):
                        raise ValueError("Invalid archive Content-Range; refusing to append")
                    total = int(match[3])
                    if length != total - offset:
                        raise ValueError("Archive range length does not match Content-Length")
                elif response.status_code == 200:
                    offset, total = 0, length
                else:
                    raise ValueError(f"Unexpected archive HTTP status: {response.status_code}")
                if not 0 < total <= 5 * 1024**3:
                    raise ValueError(f"Unexpected annual archive size: {total}")
                write_json(receipt, dict(url=url, total=total,
                    validator=response.headers.get("ETag") or response.headers.get("Last-Modified")))
                received = offset
                next_progress = received + 128 * 1024**2
                print(f"{year}: downloading from {offset / 1024**2:.0f} / "
                      f"{total / 1024**2:.0f} MiB", flush=True)
                with pending.open("ab" if offset else "wb") as target:
                    for chunk in response.iter_content(4 * 1024**2):
                        received += len(chunk)
                        if received > total:
                            raise ValueError("Annual response exceeded its declared size")
                        target.write(chunk)
                        if received >= next_progress:
                            print(f"{year}: {received / total:.1%} "
                                  f"({received / 1024**2:.0f} MiB)", flush=True)
                            next_progress = received + 128 * 1024**2
                if received != total:
                    raise requests.ConnectionError(f"Incomplete archive: {received} of {total} bytes")
            pending.replace(path)
            return
        except requests.RequestException as error:
            print(f"{year}: transfer attempt {attempt + 1} failed: {error}", file=sys.stderr, flush=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def collect_annual(start: date, end: date, output: Path, session: requests.Session,
                   export_geotiffs: bool, reuse_dir: Path | None = None) -> None:
    archives = output / "netcdf"
    archives.mkdir(exist_ok=True)
    records = []
    for year in range(start.year, end.year + 1):
        path = archives / f"seNorge2018_{year}.nc"
        if not path.exists() and reuse_dir is not None:
            existing = reuse_dir / path.name
            if existing.is_file():
                path = existing
                print(f"{year}: reusing existing original {path}", flush=True)
        fetch_archive(session, year, path)
        print(f"{year}: verifying all daily temperature and precipitation slabs", flush=True)
        record = validate_archive(path, year)
        records.append(record)
        write_json(output / "manifest.json", dict(
            status="complete" if year == end.year else "in_progress",
            requested_start=start.isoformat(), requested_end=end.isoformat(),
            archives=records, attribution="MET Norway",
            coverage="Mainland Norway and neighbouring areas; excludes Svalbard and Jan Mayen.",
            source_note="seNorge_2018 annual archive, not assumed identical to NVE GTS.",
        ))
        print(f"{year}: VERIFIED {record['days']} days, {record['width']} x "
              f"{record['height']} cells at 1 km", flush=True)
        if export_geotiffs:
            export_dir = output / "geotiffs" / str(year)
            export_dir.mkdir(parents=True, exist_ok=True)
            for index, day in enumerate(daily_dates(date(year, 1, 1), date(year, 12, 31))):
                if not start <= day <= end:
                    continue
                day_record = export_dir / f"{day:%Y%m%d}.json"
                if day_record.exists():
                    saved = json.loads(day_record.read_text(encoding="utf-8"))
                    if all(Path(item["geotiff"]).is_file() for item in saved["rasters"]):
                        continue
                rasters = export_day(path, day, export_dir, record["source"], time_index=index)
                write_json(day_record, dict(rasters=rasters))
    print(f"Annual collection complete: {len(records)} verified archives.", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat,
                        help="Last day, inclusive")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annual", action="store_true",
                        help="Download and verify full annual archives, retaining original NetCDFs")
    parser.add_argument("--export-geotiffs", action="store_true",
                        help="Also export requested days from annual archives (resumable per day)")
    parser.add_argument("--reuse-dir", type=Path,
                        help="Reuse annual originals in this folder without copying them")
    args = parser.parse_args(argv)
    if args.end < args.start:
        parser.error("--end must be on or after --start")
    if args.export_geotiffs and not args.annual:
        parser.error("--export-geotiffs is only needed with --annual")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(start=args.start.isoformat(), end=args.end.isoformat(),
                  source=ARCHIVE_URL if args.annual else BASE_URL, variables=VARIABLES)
    config_path = output / "query.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        parser.error("Output directory belongs to another query; use a different directory")
    write_json(config_path, config)
    originals = output / "netcdf"
    originals.mkdir(exist_ok=True)
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    results = []
    try:
        with requests.Session() as session:
            session.headers["User-Agent"] = "pipeline-v3-senorge-national/1.0 (sequential daily files)"
            session.mount("https://", HTTPAdapter(max_retries=retry))
            if args.annual:
                collect_annual(args.start, args.end, output, session, args.export_geotiffs,
                               args.reuse_dir)
                return 0
            for day in daily_dates(args.start, args.end):
                path = originals / f"seNorge2018_{day:%Y%m%d}.nc"
                print(f"Whole-country daily file: {day}", flush=True)
                source = fetch_file(session, day, path)
                results.extend(export_day(path, day, output, source))
        write_json(output / "manifest.json", dict(
            query=config, rasters=results, attribution="MET Norway",
            coverage="Full native seNorge domain: mainland Norway and neighbouring areas; "
                     "not Svalbard/Jan Mayen and not clipped to national borders.",
            source_note="MET Norway seNorge_2018 bulk product; not assumed identical to NVE GTS.",
        ))
    except (requests.RequestException, OSError, ValueError, KeyError) as error:
        print(f"National seNorge download failed: {error}\n"
              "Completed daily files remain cached. Rerun the same command.", file=sys.stderr)
        return 1
    print(f"Complete: {len(results)} national rasters; manifest: {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
