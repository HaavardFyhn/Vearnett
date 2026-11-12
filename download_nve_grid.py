"""Download native 1 km NVE weather cells as ArcGIS-ready GeoTIFF time series."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import sys
import time

# Load HTTPS/OpenSSL before Rasterio's bundled DLLs on Windows.
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import numpy as np


BASE_URL = "https://gts.nve.no/api/GridTimeSeries"
CELL_M = 1000
CRS = "EPSG:32633"
NODATA = -9999.0
THEMES = (
    "tm", "tm3h", "rr", "rr3h", "sd", "swe",
    "windSpeed10m24h06", "windSpeed10m3h",
    "windDirection10m24h06", "windDirection10m3h",
)


def api_date(value: str) -> str:
    for pattern in ("%Y-%m-%d", "%Y-%m-%dT%H"):
        try:
            return datetime.strptime(value, pattern).strftime(pattern)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("Use YYYY-MM-DD or YYYY-MM-DDTHH")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=api_date,
                        help="First date/hour, inclusive")
    parser.add_argument("--end", required=True, type=api_date,
                        help="Last date/hour, INCLUSIVE (unlike the MEPS downloader)")
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"),
                      help="One cell containing this UTM 33N coordinate in metres")
    area.add_argument("--bbox", nargs=4, type=float,
                      metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                      help="UTM 33N bounds in metres; expanded outward to whole 1 km cells")
    parser.add_argument("--themes", nargs="+", choices=THEMES, default=["tm", "rr"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-cells", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if datetime.fromisoformat(args.start) > datetime.fromisoformat(args.end):
        parser.error("--end must be on or after --start")
    if len(args.start) != len(args.end):
        parser.error("--start and --end must both use dates or both include hours")
    if not math.isfinite(args.request_delay) or args.request_delay < 0:
        parser.error("--request-delay must be finite and nonnegative")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_cells < 1:
        parser.error("--timeout and --max-cells must be positive")
    coordinates = args.point if args.point is not None else args.bbox
    if not all(math.isfinite(value) for value in coordinates):
        parser.error("Coordinates must be finite UTM 33N metres")
    if args.bbox is not None:
        west, south, east, north = args.bbox
        if west >= east or south >= north:
            parser.error("--bbox requires WEST < EAST and SOUTH < NORTH")
    args.themes = list(dict.fromkeys(args.themes))
    return args


def grid_centres(point, bbox, max_cells=10000) -> tuple[np.ndarray, np.ndarray]:
    if point is not None:
        west, south = (math.floor(value / CELL_M) * CELL_M for value in point)
        east, north = west + CELL_M, south + CELL_M
    else:
        west, south, east, north = bbox
        west, south = (math.floor(value / CELL_M) * CELL_M for value in (west, south))
        east, north = (math.ceil(value / CELL_M) * CELL_M for value in (east, north))
    count = ((east - west) // CELL_M) * ((north - south) // CELL_M)
    if count > max_cells:
        raise ValueError(f"{count} cells exceeds --max-cells={max_cells}; use a smaller area")
    # North-to-south rows match GeoTIFF's negative Y pixel size.
    xs = np.arange(west + CELL_M / 2, east, CELL_M, dtype=np.int64)
    ys = np.arange(north - CELL_M / 2, south, -CELL_M, dtype=np.int64)
    return xs, ys


def parse_series(payload: dict, theme: str, x: int, y: int) -> tuple[dict, np.ndarray]:
    required = {"Theme", "X", "Y", "StartDate", "EndDate", "TimeResolution",
                "Unit", "NoDataValue", "Data"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"NVE returned an invalid time series for {theme} at {x},{y}")
    if payload["Theme"] != theme or payload["X"] != x or payload["Y"] != y:
        raise ValueError("NVE response theme/coordinates do not match the requested cell")
    start = datetime.strptime(payload["StartDate"], "%d.%m.%Y %H:%M:%S")
    end = datetime.strptime(payload["EndDate"], "%d.%m.%Y %H:%M:%S")
    minutes = payload["TimeResolution"]
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("NVE returned an invalid TimeResolution")
    if not isinstance(payload["Unit"], str) or not payload["Unit"]:
        raise ValueError("NVE returned missing units")
    values = np.asarray(payload["Data"], dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("NVE returned an empty or non-vector Data field")
    if start + timedelta(minutes=minutes * (len(values) - 1)) != end:
        raise ValueError("NVE Data length does not match StartDate/EndDate/TimeResolution")
    missing = ~np.isfinite(values) | (values == float(payload["NoDataValue"]))
    if np.any(np.abs(values[~missing]) >= np.finfo(np.float32).max):
        raise ValueError("NVE values exceed GeoTIFF float32 range")
    if np.any(values[~missing] == NODATA):
        raise ValueError("NVE value conflicts with the output NoData sentinel")
    values[missing] = NODATA
    metadata = {
        "theme": theme, "name": payload.get("FullName", theme), "unit": payload["Unit"],
        "start": start.isoformat(), "end": end.isoformat(), "step_minutes": minutes,
        "count": len(values),
    }
    return metadata, values.astype(np.float32)


def write_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    pending.replace(path)


def is_missing_cell(payload: dict, theme: str, x: int, y: int) -> bool:
    return payload == {"Error": f"Tema: {theme}. No cell exists for coordinates {x}, {y}."}


def download(args) -> list[Path]:
    xs, ys = grid_centres(args.point, args.bbox, args.max_cells)
    cell_count = len(xs) * len(ys)
    print(f"{len(xs)} columns x {len(ys)} rows; {CELL_M} m cells; {CRS}; "
          f"at most {cell_count * len(args.themes)} sequential point requests", flush=True)
    if args.dry_run:
        return []
    import rasterio
    from rasterio.transform import from_origin

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(start=args.start, end=args.end, themes=args.themes,
                  west=int(xs[0] - CELL_M / 2), north=int(ys[0] + CELL_M / 2),
                  width=len(xs), height=len(ys), cell_m=CELL_M, crs=CRS, source=BASE_URL)
    config_path = output / "query.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Output directory belongs to another query; choose a new --output-dir")
    else:
        write_json(config_path, config)
    cache = output / "responses"
    cache.mkdir(exist_ok=True)
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[408, 429, 500, 502, 503, 504],
                  allowed_methods=["GET"], respect_retry_after_header=True)
    rasters = []
    summaries = []
    with requests.Session() as session:
        session.headers["User-Agent"] = "pipeline-v3-nve-grid/1.0 (sequential 1km pilot)"
        session.mount("https://", HTTPAdapter(max_retries=retry))
        last_request = None
        for theme in args.themes:
            metadata = None
            cube = None
            missing_cells = []
            for row, y in enumerate(ys):
                for column, x in enumerate(xs):
                    raw_path = cache / f"{theme}_{x}_{y}.json"
                    if raw_path.exists():
                        payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    else:
                        if last_request is not None:
                            time.sleep(max(0, args.request_delay - (time.monotonic() - last_request)))
                        url = f"{BASE_URL}/{x}/{y}/{args.start}/{args.end}/{theme}.json"
                        print(f"GET {theme} cell {row * len(xs) + column + 1}/{cell_count}: {x},{y}",
                              flush=True)
                        last_request = time.monotonic()
                        with session.get(url, timeout=(15, args.timeout)) as response:
                            if response.status_code == 400:
                                payload = response.json()
                                if not is_missing_cell(payload, theme, int(x), int(y)):
                                    response.raise_for_status()
                            else:
                                response.raise_for_status()
                                payload = response.json()
                    if is_missing_cell(payload, theme, int(x), int(y)):
                        print(f"NODATA {theme} at {x},{y}: NVE reports no grid cell", flush=True)
                        missing_cells.append([int(x), int(y)])
                        if not raw_path.exists():
                            write_json(raw_path, payload)
                        continue
                    current, values = parse_series(payload, theme, int(x), int(y))
                    for field, requested in (("start", args.start), ("end", args.end)):
                        actual = datetime.fromisoformat(current[field])
                        expected = datetime.fromisoformat(requested)
                        if (len(requested) == 10 and actual.date() != expected.date()) or (
                            len(requested) != 10 and actual != expected
                        ):
                            raise ValueError(f"NVE returned a different {field} than requested: {actual}")
                    if metadata is None:
                        metadata = current
                        if len(values) * cell_count > 10_000_000:
                            raise ValueError("Query exceeds 10 million cell/time values; split the date range")
                        cube = np.full((len(values), len(ys), len(xs)), NODATA, dtype=np.float32)
                    if current != metadata:
                        raise ValueError(f"Inconsistent time axis/units for {theme} at {x},{y}")
                    if not raw_path.exists():
                        write_json(raw_path, payload)
                    cube[:, row, column] = values
            if metadata is None or cube is None:
                raise ValueError("No grid cells selected")
            valid = cube != NODATA
            if not np.any(valid):
                raise ValueError(f"All {theme} cells are NoData; inspect the area/date/theme")
            start = datetime.fromisoformat(metadata["start"])
            for index, values in enumerate(cube):
                timestamp = start + timedelta(minutes=metadata["step_minutes"] * index)
                path = output / f"nve_{theme}_{timestamp:%Y%m%dT%H%M}_1km.tif"
                pending = path.with_suffix(".pending.tif")
                with rasterio.open(
                    pending, "w", driver="GTiff", width=len(xs), height=len(ys),
                    count=1, dtype="float32", crs=CRS, nodata=NODATA,
                    transform=from_origin(config["west"], config["north"], CELL_M, CELL_M),
                    compress="deflate", predictor=3,
                ) as target:
                    target.write(values, 1)
                    target.set_band_description(1, f"{theme} [{metadata['unit']}] {timestamp.isoformat()}")
                    target.update_tags(
                        source=BASE_URL, theme=theme, full_name=metadata["name"],
                        units=metadata["unit"], valid_time=timestamp.isoformat(),
                        time_resolution_minutes=str(metadata["step_minutes"]),
                        time_zone="unspecified by NVE API",
                        source_start=metadata["start"], source_end=metadata["end"],
                        cell_size_m=str(CELL_M),
                    )
                pending.replace(path)
                rasters.append(path)
                print(f"ArcGIS GeoTIFF: {path}", flush=True)
            summaries.append(dict(
                **metadata, valid_values=int(valid.sum()), nodata_values=int((~valid).sum()),
                missing_cells=missing_cells,
                minimum=float(cube[valid].min()), maximum=float(cube[valid].max()),
            ))
    write_json(output / "manifest.json", dict(
        query=config, series=summaries, geotiffs=[str(path) for path in rasters],
        time_note="Timestamps preserved exactly as returned by NVE; timezone not specified by API.",
    ))
    return rasters


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        paths = download(args)
    except (requests.RequestException, ValueError, OSError) as error:
        print(f"NVE download failed: {error}\nValidated responses remain cached; rerun the same command.",
              file=sys.stderr, flush=True)
        return 1
    if paths:
        print(f"Saved {len(paths)} raster(s). ArcGIS Pro: Map > Add Data > select the GeoTIFFs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
