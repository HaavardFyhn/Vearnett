"""Resumable, quality-preserving Frost/MEPS data for the western-valley pilot."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import requests

from remote_senorge.footprint_atlas import FEATURE_FIELDS
from remote_senorge.train_remote import sha256, write_json
from valley_pilot import HOURS, WINTERS, REGIME_FIELDS, frost_records, graph_features, parse_ascii, spatial_groups


ROOT = Path(__file__).resolve().parent
BBOX = (5.4, 60.9, 7.3, 61.9)
VARIABLES = {
    "air_temperature_2m": ("temperature_c", "K"),
    "x_wind_10m": ("wind_x_ms", "m/s"),
    "y_wind_10m": ("wind_y_ms", "m/s"),
    "cloud_area_fraction": ("cloud_fraction", "1"),
}
ARCHIVE = "https://thredds.met.no/thredds/dodsC/meps25epsarchive"
GROUP_DEFINITION = (
    "Nearest anchor within6km: Haukedal(SN56960) test, Fjærland–Bremuseet(SN55820) "
    "validation; remaining stations train. Geographic challenge groups, not verified valley boundaries."
)


def dates():
    for year in WINTERS:
        current, end = date(year, 12, 1), date(year+1, 3, 1)
        while current < end:
            yield current
            current += timedelta(days=1)


def http(session, url, params=None, *, auth=None, maximum=8*1024**2):
    for attempt in range(4):
        try:
            time.sleep(1.)
            with session.get(url, params=params, auth=auth, timeout=(15, 90), stream=True) as response:
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    print(f"HTTP{response.status_code}; retry {attempt+1}/3", flush=True)
                    time.sleep(min(2**attempt*2, 10))
                    continue
                response.raise_for_status()
                pieces, size = [], 0
                for piece in response.iter_content(65536):
                    size += len(piece)
                    if size > maximum:
                        raise ValueError("Pilot response exceeded bounded subset size")
                    pieces.append(piece)
                return b"".join(pieces).decode("utf-8")
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            print(f"Connection timeout; retry {attempt+1}/3", flush=True)
            time.sleep(2**attempt*2)
    raise RuntimeError("HTTP retries exhausted")


def prepare_terrain(root: Path, folder: Path):
    import pyogrio
    atlas = root / "outputs" / "senorge_footprint_expanded"
    manifest_path = atlas / "footprint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    table = np.load(atlas / "grid_table.npy", mmap_mode="r", allow_pickle=False)
    keys = {tuple(np.rint(xy/1000).astype(int)): i for i, xy in enumerate(table[:, :2])}
    with closing(sqlite3.connect((root / "frost_history.db").as_uri()+"?mode=ro", uri=True)) as connection:
        stations = pd.read_sql_query(
            "SELECT station_id,name,lat,lon,elevation FROM stations WHERE inside_norway=1 "
            "AND lon BETWEEN ? AND ? AND lat BETWEEN ? AND ? ORDER BY station_id",
            connection, params=(BBOX[0], BBOX[2], BBOX[1], BBOX[3]))
    features = pyogrio.read_dataframe(
        root / "outputs" / "temperature_5y_prediction_rmse_station_features.gpkg", read_geometry=False)
    stations = stations.merge(features[["station_id", "x", "y"]].drop_duplicates("station_id"), how="inner")
    refs = [keys.get(tuple(np.floor(xy/1000+.5).astype(int)), -1)
            for xy in stations[["x", "y"]].to_numpy()]
    stations["reference"] = refs
    excluded = stations.loc[stations["reference"] < 0, "station_id"].tolist()
    stations = stations.loc[stations["reference"] >= 0].reset_index(drop=True)
    if len(stations) < 12:
        raise ValueError("Too few supported stations for spatial holdouts")
    xy = stations[["x", "y"]].to_numpy()
    left, bottom = np.floor((xy.min(axis=0)-15000)/1000)*1000
    right, top = np.ceil((xy.max(axis=0)+15000)/1000)*1000
    shape = (int((top-bottom)/1000)+1, int((right-left)/1000)+1)
    height = np.zeros(shape, np.float32)
    valid = np.zeros(shape, bool)
    atlas_height = np.load(atlas / "terrain_height.npy", mmap_mode="r", allow_pickle=False)
    atlas_valid = np.load(atlas / "terrain_valid.npy", mmap_mode="r", allow_pickle=False)
    x0, y0 = manifest["atlas_origin"]
    indices = np.flatnonzero((table[:, 0] >= left) & (table[:, 0] <= right) &
                             (table[:, 1] >= bottom) & (table[:, 1] <= top))
    for index in indices:
        x, y = table[index, :2]
        r, c = int((top-y)/1000), int((x-left)/1000)
        ar, ac = int(round((y0-y)/1000))*10, int(round((x-x0)/1000))*10
        mask = atlas_valid[ar:ar+10, ac:ac+10].astype(bool)
        if mask.sum() < 90:
            continue
        # Low-terrain quantile is a drainage-path proxy, not a resolved10m valley floor.
        height[r, c] = np.quantile(atlas_height[ar:ar+10, ac:ac+10][mask], .25)*1000
        valid[r, c] = True
    references = stations["reference"].to_numpy(dtype=int)
    positions = np.column_stack(((top-table[references, 1])/1000,
                                 (table[references, 0]-left)/1000)).astype(int)
    keep = valid[positions[:, 0], positions[:, 1]]
    excluded.extend(stations.loc[~keep, "station_id"].tolist())
    stations = stations.loc[keep].reset_index(drop=True)
    positions, references = positions[keep], references[keep]
    stations["group"] = spatial_groups(stations)
    stations.to_csv(folder / "stations.csv", index=False)
    graph = graph_features(height, valid, positions)
    np.savez_compressed(folder / "terrain.npz", table=table[references], graph=graph,
                        height=height, valid=valid, positions=positions)
    configuration = dict(
        schema=1, bbox=BBOX, winters=WINTERS, utc_hours=HOURS, lead_hours=[12, 15, 18],
        variables=VARIABLES, excluded_stations=excluded, terrain_fields=FEATURE_FIELDS,
        atlas_manifest_sha256=sha256(manifest_path),
        graph="Four-neighbour minimax pass on1km land, height=25th-percentile100m DEM;15km regional halo.",
        groups=GROUP_DEFINITION,
        observations="Frost qualityCode0 only; one sensor chosen by training-period accepted metadata coverage.",
        terrain_checksums={name: sha256(folder / name) for name in ("stations.csv", "terrain.npz")},
        interpretation="Contemporaneous winter nighttime station transfer, not daily minimum or independent national uncertainty.",
        leakage_warning="Forecasts precede evaluated observations by12–18h, but may assimilate earlier observations from held-out stations. No claim of an observation-independent NWP background.")
    write_json(folder / "config.json", configuration)
    print(f"Terrain ready: {len(stations)} stations; groups {stations.groupby('group').size().to_dict()}", flush=True)


def forecast_url(day: date):
    run = datetime.combine(day, datetime.min.time(), timezone.utc)-timedelta(hours=12)
    filename = (f"meps_det_2_5km_{run:%Y%m%dT%HZ}.nc" if run.date() <= date(2023, 9, 30)
                else f"meps_det_sfc_{run:%Y%m%dT%HZ}.ncml")
    return f"{ARCHIVE}/{run:%Y/%m/%d}/{filename}", run


def update_groups(folder: Path):
    """Change only split labels; provider caches depend on station identity, not splits."""
    path = folder / "stations.csv"
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text())
    for name, checksum in config["terrain_checksums"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"Pilot terrain/catalogue checksum mismatch: {name}")
    stations = pd.read_csv(path)
    stations["group"] = spatial_groups(stations)
    pending = folder / "stations.pending.csv"
    stations.to_csv(pending, index=False)
    pending.replace(path)
    config["groups"] = GROUP_DEFINITION
    config["terrain_checksums"]["stations.csv"] = sha256(path)
    write_json(config_path, config)
    print(f"Updated geographic groups: {stations.groupby('group').size().to_dict()}; provider caches retained",
          flush=True)


def projection_from_das(text):
    match = re.search(r"\bprojection_lambert\s*\{([^}]+)", text)
    if match is None:
        raise ValueError("Missing MEPS projection metadata")
    attrs = {}
    for kind, name, value in re.findall(r"(String|Float64|Float32)\s+(\w+)\s+([^;]+);", match.group(1)):
        if kind == "String":
            attrs[name] = value.strip().strip('"')
        else:
            values = [float(item) for item in value.split(",")]
            attrs[name] = values[0] if len(values) == 1 else values
    return CRS.from_cf(attrs)


def download_regime(session, day, stations):
    url, run = forecast_url(day)
    metadata = http(session, url+".das")
    projection = projection_from_das(metadata)
    units = {}
    for variable, (_, expected) in VARIABLES.items():
        match = re.search(r"\b"+variable+r"\s*\{([^}]+)", metadata)
        unit = re.search(r'String units "([^"]+)"', match.group(1)) if match else None
        if unit is None or unit.group(1) != expected:
            raise ValueError(f"Unexpected MEPS units/variable: {variable}")
        units[variable] = unit.group(1)
    coords = http(session, url+".ascii?x,y,time[12:3:18]")
    x, y, times = (parse_ascii(coords, name) for name in ("x", "y", "time"))
    expected_times = np.array([(datetime.combine(day, datetime.min.time(), timezone.utc)+
                               timedelta(hours=hour)).timestamp() for hour in HOURS])
    if not np.array_equal(times, expected_times):
        raise ValueError("MEPS subset times differ from requested evaluation times")
    transformer = Transformer.from_crs(4326, projection, always_xy=True)
    sx, sy = transformer.transform(stations["lon"].to_numpy(), stations["lat"].to_numpy())
    ix = np.abs(sx[:, None]-x).argmin(axis=1)
    iy = np.abs(sy[:, None]-y).argmin(axis=1)
    if ((np.abs(x[ix]-sx) > 2500).any() or (np.abs(y[iy]-sy) > 2500).any()):
        raise ValueError("Pilot station outside MEPS grid")
    c0, c1, r0, r1 = int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())
    if (c1-c0+1)*(r1-r0+1) > 10000:
        raise ValueError("Refusing unexpectedly large regional forecast subset")
    selection = f"[12:3:18][0:1:0][{r0}:1:{r1}][{c0}:1:{c1}]"
    text = http(session, url+".ascii?"+",".join(name+selection for name in VARIABLES))
    fields = {}
    for variable, (field, _) in VARIABLES.items():
        values = parse_ascii(text, variable+"."+variable)
        if values.shape != (3, 1, r1-r0+1, c1-c0+1):
            raise ValueError(f"Unexpected MEPS subset shape: {variable}")
        fields[field] = values[:, 0, iy-r0, ix-c0]
    fields["temperature_c"] -= 273.15
    if ((fields["cloud_fraction"] < 0).any() or (fields["cloud_fraction"] > 1).any() or
            (np.abs(fields["wind_x_ms"]) > 100).any() or (np.abs(fields["wind_y_ms"]) > 100).any()):
        raise ValueError("MEPS regime values outside physical bounds")
    # Orientation is in the same Lambert grid as the model wind components.
    bearing = np.stack([sx[None]-sx[:, None], sy[None]-sy[:, None]], axis=-1)
    length = np.linalg.norm(bearing, axis=-1, keepdims=True)
    bearing = np.divide(bearing, length, out=np.zeros_like(bearing), where=length > 0)
    return dict(date=day.isoformat(), run_time=run.isoformat(), url=url, units=units,
                station_ids=stations["station_id"].tolist(), hours=list(HOURS),
                fields={key: value.tolist() for key, value in fields.items()},
                bearing=bearing.tolist())


def download_observations(session, day, stations, client, *, allow_empty=False):
    params = dict(
        sources=",".join(stations["station_id"]),
        referencetime=f"{day}T00:00:00Z/{day}T07:00:00Z",
        elements="air_temperature", timeoffsets="PT0H", timeresolutions="PT1H")
    url = "https://frost.met.no/observations/v0.jsonld"
    records, pages = [], set()
    while url:
        if not url.startswith("https://frost.met.no/") or url in pages or len(pages) >= 100:
            raise ValueError("Invalid/unbounded Frost pagination")
        pages.add(url)
        try:
            payload = json.loads(http(session, url, params, auth=(client, "")))
        except requests.HTTPError as error:
            response = error.response
            if (allow_empty and len(pages) == 1 and response is not None and
                    response.status_code in (404, 412) and
                    response.url.split("?", 1)[0] == "https://frost.met.no/observations/v0.jsonld"):
                print(f"Frost HTTP{response.status_code}: no matching observations/time series for{day},"
                      f"{len(stations)}stations; recorded as an explicit empty batch", flush=True)
                return dict(date=day.isoformat(), records=[], pages=1,
                            station_ids=stations["station_id"].tolist(),
                            provider_no_data_status=response.status_code)
            raise
        if "data" not in payload:
            raise ValueError("Frost response missing observation data")
        records.extend(frost_records(payload))
        url = payload.get("@nextLink")
        params = None
    return dict(date=day.isoformat(), records=records, pages=len(pages),
                station_ids=stations["station_id"].tolist())


def validate_day(payload, day, stations, stage):
    if payload.get("date") != day.isoformat():
        raise ValueError("Cached day does not match requested day")
    identifiers = set(stations["station_id"])
    if stage == "regimes":
        if payload.get("station_ids") != stations["station_id"].tolist() or payload.get("hours") != list(HOURS):
            raise ValueError("Cached MEPS station/time identity changed")
        if set(payload["fields"]) != set(REGIME_FIELDS):
            raise ValueError("Cached MEPS fields changed")
        for field in REGIME_FIELDS:
            values = np.asarray(payload["fields"][field])
            if values.shape != (3, len(stations)) or not np.isfinite(values).all():
                raise ValueError("Invalid cached MEPS values")
    else:
        if payload.get("station_ids") != stations["station_id"].tolist():
            raise ValueError("Cached Frost request station identity changed")
        if any(row["station_id"] not in identifiers for row in payload["records"]):
            raise ValueError("Frost returned unexpected stations")
        for row in payload["records"]:
            stamp = pd.Timestamp(row["time"])
            if stamp.date() != day or stamp.tzinfo is None:
                raise ValueError("Frost timestamp outside requested day")
            if row["accepted"] and (row["quality_code"] != 0 or row["unit"] != "degC" or
                                    row["time_offset"] != "PT0H" or row["time_resolution"] != "PT1H" or
                                    row["sensor_id"] is None):
                raise ValueError("Accepted Frost row has invalid QC/time/sensor metadata")


def run(args):
    root, folder = args.root.resolve(), args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    from precompute_frost import computation_lock
    if args.stage == "groups":
        with computation_lock(folder):
            update_groups(folder)
        return
    # Hold separate locks for independent providers; only terrain setup is shared.
    config_path = folder / "config.json"
    if not config_path.exists() or args.stage == "terrain":
        with computation_lock(folder):
            prepare_terrain(root, folder)
    configuration = json.loads(config_path.read_text())
    if configuration["schema"] != 1 or tuple(configuration["bbox"]) != BBOX:
        raise ValueError("Incompatible pilot configuration")
    for name, checksum in configuration["terrain_checksums"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"Pilot terrain/catalogue checksum mismatch: {name}")
    stations = pd.read_csv(folder / "stations.csv")
    requested = [date.fromisoformat(args.day)] if args.day else list(dates())
    if any(day not in set(dates()) for day in requested):
        raise ValueError("Day outside declared pilot winters")
    if args.stage == "terrain":
        return
    stages = ("regimes", "observations") if args.stage == "all" else (args.stage,)
    for stage in stages:
        destination = folder / stage
        destination.mkdir(exist_ok=True)
        with computation_lock(destination), requests.Session() as session:
            try:
                if stage == "observations":
                    from pipeline_config import FROST_CLIENT_ID_ENV
                    client = os.getenv(FROST_CLIENT_ID_ENV)
                    if not client:
                        raise RuntimeError(
                            f"Set ${'{'}env:{FROST_CLIENT_ID_ENV}{'}'} locally before running "
                            "--stage observations. Do not paste credentials into chat. "
                            "Original Frost data lack QC codes and will not be substituted.")
                else:
                    client = None
                receipt_path = folder / f"{stage}_progress.json"
                receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                for count, day in enumerate(requested, 1):
                    key = day.isoformat()
                    path = destination / f"{key}.json"
                    if key in receipts:
                        if not path.exists() or sha256(path) != receipts[key]:
                            raise ValueError(f"Pilot cache checksum mismatch: {path}")
                        validate_day(json.loads(path.read_text()), day, stations, stage)
                        continue
                    payload = (download_regime(session, day, stations) if stage == "regimes" else
                               download_observations(session, day, stations, client))
                    validate_day(payload, day, stations, stage)
                    write_json(path, payload)
                    receipts[key] = sha256(path)
                    write_json(receipt_path, receipts)
                    write_json(destination / "status.json", dict(state="running", stage=stage,
                                                                 completed=len(receipts), days=len(list(dates()))))
                    print(f"{stage}: {key} ({count}/{len(requested)})", flush=True)
                    time.sleep(args.delay)
                write_json(destination / "status.json", dict(
                    state="ready" if len(receipts) == len(list(dates())) else "partial",
                    completed=len(receipts), days=len(list(dates()))))
            except Exception as error:
                write_json(destination / "status.json", dict(state="failed", stage=stage, error=str(error)))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "valley_regime_pilot")
    parser.add_argument("--stage", choices=("terrain", "groups", "regimes", "observations", "all"), default="all")
    parser.add_argument("--day", help="Single-day real-data smoke; resumes full preparation later")
    parser.add_argument("--delay", type=float, default=1.)
    args = parser.parse_args()
    if args.delay < 1 or not np.isfinite(args.delay):
        parser.error("Request delay must be at least1second")
    try:
        run(args)
    except Exception as error:
        print(f"Pilot preparation failed ({args.stage}): {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
