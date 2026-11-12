"""Prepare a separate, resumable five-winter national station-transfer dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from pyproj import Transformer
import requests

from prepare_valley_pilot import (
    ROOT, VARIABLES, download_observations, forecast_url, http,
    projection_from_das, validate_day,
)
from precompute_frost import computation_lock
from remote_senorge.footprint_atlas import FEATURE_FIELDS
from remote_senorge.train_remote import sha256, write_json
from valley_pilot import GRAPH_FIELDS, HOURS, REGIME_FIELDS, minimax_heights, parse_ascii, winter_year
from national_temperature_features import temperature_features


WINTERS = (2020, 2021, 2022, 2023, 2024)
TRAIN_WINTERS = WINTERS[:3]
SCHEMA = "national-station-transfer-v1"
POLICY = dict(block_m=100000, buffer_m=10000, seed=2026,
              validation_blocks=2, test_blocks=2, source_radius_m=100000,
              source_shortlist=12, sources_per_target=6, graph_halo_m=15000,
              minimum_training_nights=30, minimum_holdout_nights=15)
FEATURES = [
    "source_temperature_c", "day_sin", "day_cos", "utc_hour_fraction",
    "distance_km", "signed_height_difference_km", "source_height_km", "target_height_km",
    "source_ocean_distance_km", "target_ocean_distance_km",
    "source_fjord_distance_km", "target_fjord_distance_km",
    "source_tpi10_km", "target_tpi10_km", "source_relief10_km", "target_relief10_km",
    "source_forecast_c", "source_meps_wind_x", "source_meps_wind_y", "source_cloud",
    "target_forecast_c", "target_meps_wind_x", "target_meps_wind_y", "target_cloud",
    "forecast_difference_c", "source_forecast_residual_c", "source_speed", "target_speed",
    *GRAPH_FIELDS, "target_along_wind", "along_wind_times_target_pass",
    "source_speed_times_pass", "target_speed_times_pass", "clear_times_target_pass",
    "source_residual_times_pass",
]


def dates():
    for year in WINTERS:
        current, end = date(year, 12, 1), date(year+1, 3, 1)
        while current < end:
            yield current
            current += timedelta(days=1)


def atomic_npz(path, **arrays):
    pending = path.with_name(path.stem+".pending.npz")
    np.savez_compressed(pending, **arrays)
    pending.replace(path)


def atomic_csv(path, frame):
    pending = path.with_suffix(".pending.csv")
    frame.to_csv(pending, index=False)
    pending.replace(path)


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def initialize(root, folder):
    if (folder / "config.json").exists():
        return load_config(folder)
    atlas = root / "outputs" / "senorge_footprint_expanded"
    manifest = json.loads((atlas / "footprint_manifest.json").read_text())
    if manifest["crs"] != "EPSG:25833" or manifest["context_resolution_m"] != 100:
        raise ValueError("National preparation requires the existing EPSG:25833 100m terrain atlas")
    table = np.load(atlas / "grid_table.npy", mmap_mode="r", allow_pickle=False)
    raw_height = np.load(atlas / "terrain_height.npy", mmap_mode="r", allow_pickle=False)
    raw_valid = np.load(atlas / "terrain_valid.npy", mmap_mode="r", allow_pickle=False)
    rows, cols = np.array(raw_valid.shape)//10
    height, valid = np.zeros((rows, cols), np.float32), np.zeros((rows, cols), bool)
    # Chunk quantiles to avoid materializing the full200-million-pixel atlas.
    for start in range(0, rows, 16):
        stop = min(start+16, rows)
        masks = raw_valid[start*10:stop*10, :cols*10].reshape(stop-start, 10, cols, 10)
        masks = masks.transpose(0, 2, 1, 3).reshape(stop-start, cols, 100).astype(bool)
        pixels = raw_height[start*10:stop*10, :cols*10].reshape(stop-start, 10, cols, 10)
        pixels = pixels.transpose(0, 2, 1, 3).reshape(stop-start, cols, 100)
        supported = masks.sum(axis=-1) >= 90
        values = pixels[supported].copy()
        if not np.isfinite(values[masks[supported]]).all():
            raise ValueError("Nonfinite supported terrain atlas pixels")
        values[~masks[supported]] = np.nan
        block = height[start:stop]
        block[supported] = np.nanquantile(values, .25, axis=-1)*1000
        valid[start:stop] = supported
    with closing(sqlite3.connect((root / "frost_history.db").as_uri()+"?mode=ro", uri=True)) as connection:
        stations = pd.read_sql_query(
            "SELECT station_id,name,lat,lon,elevation FROM stations WHERE inside_norway=1 "
            "AND lat BETWEEN 57.8 AND 71.3 AND lon BETWEEN 4 AND 32 ORDER BY station_id", connection)
    if stations.empty or stations["station_id"].duplicated().any():
        raise ValueError("Empty or duplicate national Frost catalogue")
    stations["x"], stations["y"] = Transformer.from_crs(4326, 25833, always_xy=True).transform(
        stations["lon"].to_numpy(), stations["lat"].to_numpy())
    keys = {tuple(np.rint(xy/1000).astype(int)): i for i, xy in enumerate(table[:, :2])}
    references = np.array([keys.get(tuple(np.floor(xy/1000+.5).astype(int)), -1)
                           for xy in stations[["x", "y"]].to_numpy()])
    origin = np.array(manifest["atlas_origin"])
    positions = np.rint((table[np.maximum(references, 0), :2]-origin)*[1, -1]/1000).astype(int)
    positions = positions[:, ::-1]
    keep = references >= 0
    in_bounds = ((positions >= 0) & (positions < [rows, cols])).all(axis=1)
    keep &= in_bounds
    supported = np.zeros(len(stations), bool)
    supported[in_bounds] = valid[positions[in_bounds, 0], positions[in_bounds, 1]]
    keep &= supported
    exclusions = stations.loc[~keep].copy()
    exclusions["reason"] = np.where(references[~keep] < 0, "outside_terrain_table", "unsupported_1km_land")
    stations = stations.loc[keep].copy().reset_index(drop=True)
    references, positions = references[keep], positions[keep]
    if len(stations) < 20:
        raise ValueError("Too few terrain-supported national stations")
    stations["reference"] = references
    stations["block_x"] = np.floor(stations["x"]/POLICY["block_m"]).astype(int)
    stations["block_y"] = np.floor(stations["y"]/POLICY["block_m"]).astype(int)
    atomic_csv(folder / "stations.csv", stations)
    atomic_csv(folder / "terrain_exclusions.csv", exclusions)
    atomic_npz(folder / "terrain.npz", table=table[references], positions=positions,
               height=height, valid=valid, origin=origin)
    config = dict(
        schema=SCHEMA, winters=WINTERS, train_winters=TRAIN_WINTERS, utc_hours=HOURS,
        policy=POLICY, terrain_fields=FEATURE_FIELDS, created_utc=datetime.now(timezone.utc).isoformat(),
        station_scope="Existing Frost inside_norway catalogue,57.8-71.3N/4-32E; coastal islands included if terrain supported; excludes Arctic territories.",
        observations="qualityCode0,degC,PT1H/PT0H,2m or unspecified height; sensor chosen using2020-23 training metadata coverage only.",
        graph="1km land cells;25th percentile100m DEM,>=90%valid; local source100km-radius square plus15km halo; not national-optimal paths.",
        holdouts="100km geographic blocks,not physical valley boundaries; selected deterministically using QC availability only;10km surrounding training buffer.",
        limitation="Contemporaneous DJF00/03/06UTC transfers,not daily minima. MEPS2.5km background may assimilate earlier held-out station readings. No accuracy claim at1km.",
        atlas_manifest_sha256=sha256(atlas / "footprint_manifest.json"),
        checksums={name: sha256(folder / name) for name in ("stations.csv", "terrain.npz", "terrain_exclusions.csv")})
    write_json(folder / "config.json", config)
    return config


def load_config(folder):
    config = json.loads((folder / "config.json").read_text())
    if config["schema"] != SCHEMA or config["policy"] != POLICY or config["winters"] != list(WINTERS):
        raise ValueError("Incompatible national configuration; use a separate new output")
    for name, checksum in config["checksums"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"National input checksum mismatch: {name}")
    return config


def cached(path, receipt, key):
    if key not in receipt:
        return None
    if sha256(path) != receipt[key]:
        raise ValueError(f"Cache checksum mismatch: {path}")
    return json.loads(path.read_text())


def save_cache(path, receipt_path, receipt, key, payload):
    write_json(path, payload)
    receipt[key] = sha256(path)
    write_json(receipt_path, receipt)


def observations(folder, stations, requested):
    client = os.getenv("FROST_CLIENT_ID")
    destination = folder / "observations"
    destination.mkdir(exist_ok=True)
    path = folder / "observations_progress.json"
    receipts = json.loads(path.read_text()) if path.exists() else {}
    with computation_lock(destination), requests.Session() as session:
        provenance = dict(
            station_ids=stations["station_id"].tolist(), winters=list(WINTERS), hours=list(HOURS),
            qc_rules_sha256=sha256(ROOT / "valley_pilot.py"),
            provider_code_sha256=sha256(ROOT / "prepare_valley_pilot.py"))
        provenance_path = folder / "observations_config.json"
        if provenance_path.exists():
            if json.loads(provenance_path.read_text()) != provenance:
                raise ValueError("Observation station/QC/provider configuration changed; use a new output")
        else:
            write_json(provenance_path, provenance)
        for number, day in enumerate(requested, 1):
            for batch, start in enumerate(range(0, len(stations), 64)):
                subset = stations.iloc[start:start+64]
                key = f"{day}_b{batch:03d}"
                target = destination / f"{key}.json"
                payload = cached(target, receipts, key)
                if payload is None:
                    if not client:
                        raise RuntimeError("Set $env:FROST_CLIENT_ID in your terminal; never paste credentials into chat.")
                    payload = download_observations(session, day, subset, client, allow_empty=True)
                    validate_day(payload, day, subset, "observations")
                    save_cache(target, path, receipts, key, payload)
                validate_day(payload, day, subset, "observations")
            print(f"observations: {day} ({number}/{len(requested)})", flush=True)


def verified_observation_batches(folder, stations):
    receipt_path = folder / "observations_progress.json"
    if not receipt_path.exists():
        raise ValueError("Run the complete observations stage before the QC audit")
    receipts = json.loads(receipt_path.read_text())
    expected = {f"{day}_b{batch:03d}" for day in dates()
                for batch in range((len(stations)+63)//64)}
    if set(receipts) != expected:
        raise ValueError(f"Incomplete observations: {len(set(receipts)&expected)}/{len(expected)}batches")
    for day in dates():
        for batch, start in enumerate(range(0, len(stations), 64)):
            key = f"{day}_b{batch:03d}"
            payload = cached(folder / "observations" / f"{key}.json", receipts, key)
            validate_day(payload, day, stations.iloc[start:start+64], "observations")
            yield payload


def audit(folder, stations):
    coverage, rejected, empty_batches = {}, Counter(), Counter()
    for payload in verified_observation_batches(folder, stations):
        if "provider_no_data_status" in payload:
            empty_batches[payload["provider_no_data_status"]] += 1
        for row in payload["records"]:
            if not row["accepted"]:
                rejected[row["rejection"]] += 1
                continue
            stamp = pd.Timestamp(row["time"])
            year = winter_year(stamp)
            sensor = (row["station_id"], row["source_id"], row["sensor_id"])
            entry = coverage.setdefault(sensor, {y: set() for y in WINTERS})
            if row["time"] in entry[year]:
                raise ValueError(f"Duplicate selected sensor/time: {sensor}")
            entry[year].add(row["time"])
    if not coverage:
        raise ValueError("No accepted qualityCode0 hourly nighttime observations")
    choices = []
    for station in stations["station_id"]:
        sensors = [key for key in coverage if key[0] == station]
        if not sensors:
            choices.append(dict(station_id=station, source_id="", sensor_id=-1, train_nights=0,
                                validation_nights=0, test_nights=0, eligible=False))
            continue
        selected = min(sensors, key=lambda key: (
            -sum(len(coverage[key][year]) for year in TRAIN_WINTERS), key[1], str(key[2])))
        counts = {year: len({stamp[:10] for stamp in coverage[selected][year]}) for year in WINTERS}
        train_nights = sum(counts[y] for y in TRAIN_WINTERS)
        choices.append(dict(
            station_id=station, source_id=selected[1], sensor_id=selected[2],
            train_nights=train_nights, validation_nights=counts[2023], test_nights=counts[2024],
            eligible=train_nights >= POLICY["minimum_training_nights"],
            **{f"nights_{year}": counts[year] for year in WINTERS}))
    frame = pd.DataFrame(choices)
    atomic_csv(folder / "qc_audit.csv", frame)
    write_json(folder / "qc_audit.json", dict(
        schema=SCHEMA, stations=len(frame), eligible=int(frame["eligible"].sum()),
        rejected=dict(rejected), provider_no_data_batches=dict(empty_batches),
        sensor_selection="Training winters only,accepted timestamp count with deterministic ties",
        qc_csv_sha256=sha256(folder / "qc_audit.csv"),
        observations_receipt_sha256=sha256(folder / "observations_progress.json")))
    print(f"QC audit: {int(frame['eligible'].sum())}/{len(frame)}stations eligible", flush=True)
    return frame


def choose_groups(stations, qc):
    frame = stations.merge(qc, on="station_id", validate="one_to_one")
    good = frame.loc[frame["eligible"] &
                     (frame["validation_nights"] >= POLICY["minimum_holdout_nights"]) &
                     (frame["test_nights"] >= POLICY["minimum_holdout_nights"])]
    blocks = good.groupby(["block_x", "block_y"]).size()
    candidates = [tuple(map(int, key)) for key in blocks.index if blocks[key] >= 2]
    pilot_blocks = set(map(tuple, frame.loc[
        frame["lon"].between(5.4, 7.3) & frame["lat"].between(60.9, 61.9),
        ["block_x", "block_y"]].to_numpy()))
    candidates = [key for key in candidates if key not in pilot_blocks]
    candidates.sort(key=lambda key: (key[1], key[0]))
    if len(candidates) < 8:
        raise ValueError("Need>=8geographic blocks outside the inspected pilot with>=2QC-supported stations; no automatic relaxed split")
    # Force northern/southern representation without inspecting temperature values.
    middle = len(candidates)//2
    chosen = {}
    for half in (candidates[:middle], candidates[middle:]):
        ranked = sorted(half, key=lambda key: identity([POLICY["seed"], *key]))
        chosen[ranked[0]], chosen[ranked[1]] = "test", "validation"
    frame["group"] = "train"
    frame.loc[~frame["eligible"], "group"] = "excluded_qc"
    for (bx, by), group in chosen.items():
        inside = (frame["block_x"] == bx) & (frame["block_y"] == by)
        frame.loc[inside & frame["eligible"], "group"] = group
        left, bottom = bx*POLICY["block_m"], by*POLICY["block_m"]
        dx = np.maximum(np.maximum(left-frame["x"], frame["x"]-(left+POLICY["block_m"])), 0)
        dy = np.maximum(np.maximum(bottom-frame["y"], frame["y"]-(bottom+POLICY["block_m"])), 0)
        buffered = np.hypot(dx, dy) < POLICY["buffer_m"]
        frame.loc[buffered & frame["group"].eq("train"), "group"] = "excluded_buffer"
    if (frame["group"].eq("train").sum() < max(8, frame["eligible"].sum()*.5) or
            any(frame["group"].eq(group).sum() < 2 for group in ("validation", "test"))):
        raise ValueError("Regional split leaves insufficient stations; explicit policy revision required")
    return frame, [{"block_x": x, "block_y": y, "group": group}
                   for (x, y), group in sorted(chosen.items())]


def freeze(folder, stations):
    path = folder / "splits.json"
    if path.exists():
        return load_splits(folder)
    qc = audit(folder, stations)
    frame, blocks = choose_groups(stations, qc)
    atomic_csv(folder / "station_splits.csv", frame)
    write_json(folder / "station_coverage.geojson", dict(
        type="FeatureCollection", features=[
            dict(type="Feature", geometry=dict(type="Point", coordinates=[float(row.lon), float(row.lat)]),
                 properties=dict(station_id=row.station_id, name=row.name, group=row.group,
                                 block_x=int(row.block_x), block_y=int(row.block_y),
                                 train_nights=int(row.train_nights), validation_nights=int(row.validation_nights),
                                 test_nights=int(row.test_nights)))
            for row in frame.itertuples()]))
    split = dict(
        schema=SCHEMA, policy=POLICY, blocks=blocks,
        group_counts=frame.groupby("group").size().to_dict(),
        train_winters=list(TRAIN_WINTERS), validation_winter=2023, final_test_winter=2024,
        selection="2023-24 validation-block nights,alternating complete calendar nights",
        calibration="Other2023-24 validation-block nights",
        excluded="Other group/winter combinations intentionally unused; no held-out station acts as source",
        test_warning="Holdout blocks overlapping the inspected western pilot are excluded.2024-25 weather period was already used there; do not claim entirely fresh independent conditions.",
        checksums={name: sha256(folder / name) for name in
                   ("config.json", "qc_audit.csv", "qc_audit.json", "station_splits.csv",
                    "station_coverage.geojson", "observations_progress.json", "observations_config.json")})
    write_json(path, split)
    print(f"Frozen regional splits: {split['group_counts']}", flush=True)
    return split


def load_splits(folder):
    split = json.loads((folder / "splits.json").read_text())
    if split["schema"] != SCHEMA or split["policy"] != POLICY:
        raise ValueError("Incompatible national split")
    for name, checksum in split["checksums"].items():
        if sha256(folder / name) != checksum:
            raise ValueError(f"Frozen split input changed: {name}")
    return split


def rectangles(ix, iy, max_cells=4096):
    """Adaptive station-only bounding boxes; never request a national dense grid."""
    result = []

    def partition(indices):
        x, y = ix[indices], iy[indices]
        if (x.max()-x.min()+1)*(y.max()-y.min()+1) <= max_cells:
            result.append((indices, int(x.min()), int(x.max()), int(y.min()), int(y.max())))
            return
        values = x if np.ptp(x) >= np.ptp(y) else y
        cut = np.unique(values)[len(np.unique(values))//2]
        lower = values < cut
        if not lower.any() or lower.all():
            raise ValueError("Unable to bound forecast subset")
        partition(indices[lower])
        partition(indices[~lower])

    if not len(ix) or len(ix) != len(iy):
        raise ValueError("Empty or mismatched forecast stations")
    partition(np.arange(len(ix)))
    return result


def regime_metadata(session, day, stations):
    import re
    url, run = forecast_url(day)
    metadata = http(session, url+".das")
    projection = projection_from_das(metadata)
    for variable, (_, expected) in VARIABLES.items():
        match = re.search(r"\b"+variable+r"\s*\{([^}]+)", metadata)
        unit = re.search(r'String units "([^"]+)"', match.group(1)) if match else None
        if unit is None or unit.group(1) != expected:
            raise ValueError(f"Unexpected national MEPS variable/units: {variable}")
    text = http(session, url+".ascii?x,y,time[12:3:18]")
    x, y, times = (parse_ascii(text, name) for name in ("x", "y", "time"))
    expected = np.array([(datetime.combine(day, datetime.min.time(), timezone.utc)+
                          timedelta(hours=hour)).timestamp() for hour in HOURS])
    if not np.array_equal(times, expected):
        raise ValueError("MEPS forecast times mismatch")
    sx, sy = Transformer.from_crs(4326, projection, always_xy=True).transform(
        stations["lon"].to_numpy(), stations["lat"].to_numpy())
    ix = np.concatenate([np.abs(sx[start:start+2048, None]-x).argmin(axis=1)
                         for start in range(0, len(sx), 2048)])
    iy = np.concatenate([np.abs(sy[start:start+2048, None]-y).argmin(axis=1)
                         for start in range(0, len(sy), 2048)])
    if (np.abs(x[ix]-sx) > 2500).any() or (np.abs(y[iy]-sy) > 2500).any():
        raise ValueError("National station outside MEPS archive grid")
    return dict(date=str(day), url=url, run_time=run.isoformat(), hours=list(HOURS),
                station_ids=stations["station_id"].tolist(), ix=ix.tolist(), iy=iy.tolist(),
                projection_xy=np.column_stack([sx, sy]).tolist())


def preflight(folder, stations):
    path = folder / "archive_preflight.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["stations_sha256"] != sha256(folder / "stations.csv"):
            raise ValueError("Archive preflight station catalogue changed")
        return saved
    checked = []
    with requests.Session() as session:
        for day in [date(year, 12, 15) for year in WINTERS]:
            metadata = regime_metadata(session, day, stations)
            checked.append(dict(date=str(day), url=metadata["url"], run_time=metadata["run_time"],
                                stations=len(stations),
                                chunks=len(rectangles(np.array(metadata["ix"]), np.array(metadata["iy"])))))
            print(f"Archive metadata preflight passed: {day}", flush=True)
    saved = dict(stations_sha256=sha256(folder / "stations.csv"), checked=checked,
                 limitation="Checks projection,variables,units,time/domain on five sample dates; does not prove every archive day exists.")
    write_json(path, saved)
    return saved


def regimes(folder, stations, requested):
    destination = folder / "regimes"
    destination.mkdir(exist_ok=True)
    receipt_path = folder / "regimes_progress.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    with computation_lock(destination), requests.Session() as session:
        for number, day in enumerate(requested, 1):
            day_key = str(day)
            complete_path = destination / f"{day}.json"
            complete = cached(complete_path, receipts, day_key)
            if complete is not None:
                validate_regime(complete, day, stations)
                continue
            metadata_key = f"{day}_metadata"
            metadata_path = destination / f"{metadata_key}.json"
            metadata = cached(metadata_path, receipts, metadata_key)
            if metadata is None:
                try:
                    metadata = regime_metadata(session, day, stations)
                except requests.HTTPError as error:
                    response = error.response
                    expected_url = forecast_url(day)[0]+".das"
                    if response is None or response.status_code != 404 or response.url != expected_url:
                        raise
                    excluded = confirmed_missing_run(session, day, stations)
                    save_cache(complete_path, receipt_path, receipts, day_key, excluded)
                    print(f"regimes: {day} EXCLUDED: run absent from official catalogue", flush=True)
                    continue
                save_cache(metadata_path, receipt_path, receipts, metadata_key, metadata)
            if metadata["station_ids"] != stations["station_id"].tolist():
                raise ValueError("Cached forecast metadata station order changed")
            ix, iy = np.array(metadata["ix"]), np.array(metadata["iy"])
            fields = {name: np.full((3, len(stations)), np.nan) for name in REGIME_FIELDS}
            subsets = rectangles(ix, iy)
            for chunk, (indices, c0, c1, r0, r1) in enumerate(subsets):
                key = f"{day}_c{chunk:03d}"
                path = destination / f"{key}.json"
                saved = cached(path, receipts, key)
                if saved is None:
                    selection = f"[12:3:18][0:1:0][{r0}:1:{r1}][{c0}:1:{c1}]"
                    text = http(session, metadata["url"]+".ascii?"+
                                ",".join(name+selection for name in VARIABLES))
                    values = {}
                    for variable, (name, _) in VARIABLES.items():
                        array = parse_ascii(text, variable+"."+variable)
                        if array.shape != (3, 1, r1-r0+1, c1-c0+1):
                            raise ValueError(f"Unexpected national forecast subset shape: {variable}")
                        values[name] = array[:, 0, iy[indices]-r0, ix[indices]-c0]
                    values["temperature_c"] -= 273.15
                    saved = dict(indices=indices.tolist(), fields={k: v.tolist() for k, v in values.items()})
                    save_cache(path, receipt_path, receipts, key, saved)
                if saved["indices"] != indices.tolist():
                    raise ValueError("Forecast chunk identity changed")
                for name in REGIME_FIELDS:
                    values = np.asarray(saved["fields"][name])
                    if values.shape != (3, len(indices)) or not np.isfinite(values).all():
                        raise ValueError("Invalid cached forecast chunk")
                    fields[name][:, indices] = values
            payload = {key: metadata[key] for key in
                       ("date", "url", "run_time", "hours", "station_ids", "projection_xy")}
            payload["fields"] = {key: values.tolist() for key, values in fields.items()}
            validate_regime(payload, day, stations)
            save_cache(complete_path, receipt_path, receipts, day_key, payload)
            print(f"regimes: {day} ({number}/{len(requested)}),{len(subsets)}bounded chunks", flush=True)


def validate_regime(payload, day, stations):
    if payload.get("state") == "excluded_missing_run":
        validate_exclusion(payload, day, stations)
        return
    validate_day(payload, day, stations, "regimes")
    expected_run = forecast_url(day)[1].isoformat()
    xy = np.asarray(payload["projection_xy"])
    if payload["run_time"] != expected_run or xy.shape != (len(stations), 2) or not np.isfinite(xy).all():
        raise ValueError("Invalid national forecast issue time/projection")
    fields = {key: np.asarray(value) for key, value in payload["fields"].items()}
    if ((fields["cloud_fraction"] < 0).any() or (fields["cloud_fraction"] > 1).any() or
            (np.abs(fields["wind_x_ms"]) > 100).any() or (np.abs(fields["wind_y_ms"]) > 100).any() or
            (fields["temperature_c"] < -80).any() or (fields["temperature_c"] > 60).any()):
        raise ValueError("National forecast outside physical bounds")


def confirmed_missing_run(session, day, stations):
    import xml.etree.ElementTree as ET
    url, run = forecast_url(day)
    catalogue = f"https://thredds.met.no/thredds/catalog/meps25epsarchive/{run:%Y/%m/%d}/catalog.xml"
    text = http(session, catalogue)
    tree = ET.fromstring(text)
    namespace = "{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}"
    if tree.tag != namespace+"catalog":
        raise ValueError("Unexpected MEPS catalogue format; cannot confirm missing run")
    paths = sorted({node.attrib["urlPath"] for node in tree.iter(namespace+"dataset")
                    if "urlPath" in node.attrib})
    expected = url.split("/thredds/dodsC/", 1)[1]
    if not paths or expected in paths:
        raise ValueError("MEPS404 not confirmed by a populated catalogue; retry later, not excluded")
    payload = dict(
        state="excluded_missing_run", date=str(day), url=url, run_time=run.isoformat(),
        station_ids=stations["station_id"].tolist(), hours=list(HOURS),
        reason="Expected deterministic12UTC run absent from populated official archive catalogue",
        catalogue_url=catalogue, catalogue_sha256=hashlib.sha256(text.encode()).hexdigest(),
        catalogue_paths=paths, policy="Exclude whole night; no substituted run or invented weather")
    validate_exclusion(payload, day, stations)
    return payload


def validate_exclusion(payload, day, stations):
    url, run = forecast_url(day)
    expected = url.split("/thredds/dodsC/", 1)[1]
    paths = payload.get("catalogue_paths")
    checksum = payload.get("catalogue_sha256", "")
    if (payload.get("date") != str(day) or payload.get("url") != url or
            payload.get("run_time") != run.isoformat() or
            payload.get("station_ids") != stations["station_id"].tolist() or
            payload.get("hours") != list(HOURS) or
            payload.get("catalogue_url") !=
            f"https://thredds.met.no/thredds/catalog/meps25epsarchive/{run:%Y/%m/%d}/catalog.xml" or
            not isinstance(paths, list) or not paths or expected in paths or
            len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum) or
            "fields" in payload):
        raise ValueError("Invalid confirmed missing-run exclusion")


def pair_graph(table, positions, height, valid, groups):
    train = np.flatnonzero(np.asarray(groups) == "train")
    pairs, barriers, reasons = [], [], []
    # One local search per observed source, reused for all candidate targets.
    pending = {}
    for target in range(len(table)):
        if groups[target] not in ("train", "validation", "test"):
            continue
        distance = np.linalg.norm(table[train, :2]-table[target, :2], axis=1)
        candidates = train[np.argsort(distance, kind="stable")]
        candidates = [int(i) for i in candidates if i != target and
                      np.linalg.norm(table[i, :2]-table[target, :2]) <= POLICY["source_radius_m"]]
        for source in candidates[:POLICY["source_shortlist"]]:
            pending.setdefault(source, []).append(target)
    for source, targets in sorted(pending.items()):
        radius = int((POLICY["source_radius_m"]+POLICY["graph_halo_m"])/1000)
        r, c = positions[source]
        r0, c0 = max(0, r-radius), max(0, c-radius)
        r1, c1 = min(height.shape[0], r+radius+1), min(height.shape[1], c+radius+1)
        local_height, local_valid = height[r0:r1, c0:c1], valid[r0:r1, c0:c1]
        peaks = minimax_heights(local_height, local_valid, (r-r0, c-c0))
        for target in targets:
            tr, tc = positions[target]
            peak = peaks[tr-r0, tc-c0]
            if not np.isfinite(peak):
                continue
            source_z, target_z = height[r, c], height[tr, tc]
            steps = int(np.max(np.abs(positions[target]-positions[source])))+1
            line = np.rint(np.linspace(positions[source], positions[target], steps)).astype(int)
            straight = bool(valid[line[:, 0], line[:, 1]].all())
            ridge = max(0., float(height[line[:, 0], line[:, 1]].max())-max(source_z, target_z)) if straight else 0.
            pairs.append((source, target))
            barriers.append([1., (peak-source_z)/1000, (peak-target_z)/1000, ridge/1000, float(straight)])
        if len(pairs) and source % 25 == 0:
            print(f"Graph source{source}/{len(table)}: {len(pairs)}connected candidates so far", flush=True)
    keep = []
    for target in range(len(table)):
        indices = [i for i, pair in enumerate(pairs) if pair[1] == target]
        indices.sort(key=lambda i: (np.linalg.norm(table[pairs[i][0], :2]-table[target, :2]), pairs[i][0]))
        keep.extend(indices[:POLICY["sources_per_target"]])
        if not indices and groups[target] in ("train", "validation", "test"):
            reasons.append(dict(target_index=target, reason="no_connected_training_source_within100km"))
    if not keep:
        raise ValueError("No connected national station pairs")
    return np.asarray(pairs, np.int32)[keep], np.asarray(barriers, np.float32)[keep], reasons


@lru_cache(maxsize=4096)
def split_for(day, group):
    year = winter_year(pd.Timestamp(day))
    if group == "train" and year in TRAIN_WINTERS:
        return "train"
    if group == "validation" and year == 2023:
        nights = [d for d in dates() if winter_year(pd.Timestamp(d)) == 2023]
        return "selection" if nights.index(day) % 2 == 0 else "calibration"
    if group == "test" and year == 2024:
        return "test"
    return None


def selected_temperatures(payloads, station_splits):
    choices = station_splits.set_index("station_id")
    temperatures = {}
    for payload in payloads:
        for row in payload["records"]:
            if not row["accepted"]:
                continue
            chosen = choices.loc[row["station_id"]]
            if row["source_id"] != chosen["source_id"] or str(row["sensor_id"]) != str(chosen["sensor_id"]):
                continue
            key = (row["station_id"], row["time"])
            if key in temperatures:
                raise ValueError("Ambiguous selected-sensor station/time")
            temperatures[key] = float(row["value"])
    return temperatures


def day_samples(day, stations, table, pairs, graph, temperatures, forecast):
    ids = stations["station_id"].tolist()
    fields = np.array([forecast["fields"][field] for field in REGIME_FIELDS]).transpose(1, 2, 0)
    xy = np.array(forecast["projection_xy"])
    rows, values = [], []
    for pair, barrier in zip(pairs, graph):
        source, target = map(int, pair)
        split = split_for(day, stations.iloc[target]["group"])
        if split is None:
            continue
        for hour_index, hour in enumerate(HOURS):
            stamp = pd.Timestamp(day, tz="UTC")+pd.Timedelta(hours=hour)
            time_key = stamp.isoformat()
            a_obs, b_obs = temperatures.get((ids[source], time_key)), temperatures.get((ids[target], time_key))
            if a_obs is None or b_obs is None:
                continue
            a, b = fields[hour_index, source], fields[hour_index, target]
            source_speed, target_speed = np.linalg.norm(a[1:3]), np.linalg.norm(b[1:3])
            feature, deltas = temperature_features(
                table, source, [target], np.asarray([barrier]), a_obs, stamp,
                fields[hour_index], xy)
            delta = float(deltas[0])
            values.append(feature[0])
            rows.append(dict(source=ids[source], target=ids[target], time=time_key, night=str(day),
                             split=split, source_c=a_obs, target_c=b_obs, forecast_delta_c=float(delta),
                             calm_clear=bool(max(source_speed, target_speed) <= 2 and max(a[3], b[3]) <= .3)))
    x = np.asarray(values, np.float32).reshape(-1, len(FEATURES))
    arrays = dict(x=x, y=np.array([r["target_c"]-r["source_c"] for r in rows], np.float32))
    for key in ("source", "target", "time", "night", "split"):
        arrays[key] = np.array([r[key] for r in rows], dtype=str)
    for key in ("source_c", "target_c", "forecast_delta_c"):
        arrays[key] = np.array([r[key] for r in rows], np.float32)
    arrays["calm_clear"] = np.array([r["calm_clear"] for r in rows], bool)
    return arrays


def validate_samples(arrays, stations):
    n = len(arrays["y"])
    if arrays["x"].shape != (n, len(FEATURES)) or not np.isfinite(arrays["x"]).all():
        raise ValueError("Invalid national feature output")
    for key in ("source", "target", "time", "night", "split", "source_c", "target_c", "forecast_delta_c", "calm_clear"):
        if arrays[key].shape != (n,):
            raise ValueError(f"Invalid national metadata shape: {key}")
    if not np.isfinite(arrays["y"]).all() or not np.allclose(
            arrays["y"], arrays["target_c"]-arrays["source_c"], atol=1e-5):
        raise ValueError("Invalid national labels")
    groups = stations.set_index("station_id")["group"].to_dict()
    keys = set()
    for i in range(n):
        source, target = arrays["source"][i], arrays["target"][i]
        stamp = pd.Timestamp(str(arrays["time"][i]))
        if (source == target or groups.get(source) != "train" or target not in groups or
                stamp.tzinfo is None or stamp.hour not in HOURS or stamp.minute or stamp.second or
                str(stamp.date()) != arrays["night"][i] or
                split_for(stamp.date(), groups[target]) != arrays["split"][i]):
            raise ValueError("National endpoint/time/split leakage")
        key = (source, target, arrays["time"][i])
        if key in keys:
            raise ValueError("Duplicate national paired examples")
        keys.add(key)


def build(folder, all_stations, requested):
    split = load_splits(folder)
    stations = pd.read_csv(folder / "station_splits.csv", keep_default_na=False)
    active = stations["group"].isin(["train", "validation", "test"]).to_numpy()
    forecast_stations = stations.loc[active].reset_index(drop=True)
    receipt_path = folder / "samples_progress.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    weather_receipt = json.loads((folder / "regimes_progress.json").read_text())
    if not {str(day) for day in dates()} <= set(weather_receipt):
        raise ValueError("Complete all forecast nights before building; partial downloads remain resumable")
    provenance = dict(split_sha256=sha256(folder / "splits.json"),
                      forecast_receipt_sha256=sha256(folder / "regimes_progress.json"),
                      builder_sha256=sha256(Path(__file__)),
                      qc_rules_sha256=sha256(ROOT / "valley_pilot.py"),
                      provider_code_sha256=sha256(ROOT / "prepare_valley_pilot.py"))
    destination = folder / "samples"
    destination.mkdir(exist_ok=True)
    setup_path = folder / "samples_config.json"
    if setup_path.exists():
        if json.loads(setup_path.read_text()) != provenance:
            raise ValueError("Sample inputs changed; complete forecasts before building")
    else:
        write_json(setup_path, provenance)
    pair_path = folder / "pairs.npz"
    pair_info = folder / "pairs.json"
    if pair_info.exists():
        info = json.loads(pair_info.read_text())
        if (info["split_sha256"] != provenance["split_sha256"] or sha256(pair_path) != info["sha256"] or
                sha256(folder / "pair_exclusions.csv") != info["exclusions_sha256"]):
            raise ValueError("National pair graph changed")
    else:
        with np.load(folder / "terrain.npz", allow_pickle=False) as terrain:
            pairs, graph, excluded = pair_graph(terrain["table"][active], terrain["positions"][active],
                                                terrain["height"], terrain["valid"], forecast_stations["group"].tolist())
        atomic_npz(pair_path, pairs=pairs, graph=graph)
        atomic_csv(folder / "pair_exclusions.csv", pd.DataFrame(
            [dict(station_id=forecast_stations.iloc[r["target_index"]]["station_id"], reason=r["reason"]) for r in excluded],
            columns=["station_id", "reason"]))
        write_json(pair_info, dict(split_sha256=provenance["split_sha256"], sha256=sha256(pair_path),
                                   exclusions_sha256=sha256(folder / "pair_exclusions.csv"), pairs=len(pairs)))
    with np.load(pair_path, allow_pickle=False) as saved:
        pairs, graph = saved["pairs"], saved["graph"]
    with np.load(folder / "terrain.npz", allow_pickle=False) as terrain:
        table = terrain["table"][active]
    obs_receipt = json.loads((folder / "observations_progress.json").read_text())
    weather_receipt = json.loads((folder / "regimes_progress.json").read_text())
    for number, day in enumerate(requested, 1):
        key = str(day)
        path = destination / f"{day}.npz"
        weather = cached(folder / "regimes" / f"{day}.json", weather_receipt, key)
        if weather is None:
            raise ValueError(f"Missing complete forecast: {day}")
        validate_regime(weather, day, forecast_stations)
        if key in receipt:
            if sha256(path) != receipt[key]:
                raise ValueError(f"Sample shard checksum mismatch: {day}")
        else:
            if weather.get("state") == "excluded_missing_run":
                arrays = dict(x=np.empty((0, len(FEATURES)), np.float32), y=np.empty(0, np.float32))
                for field in ("source", "target", "time", "night", "split"):
                    arrays[field] = np.array([], dtype=str)
                for field in ("source_c", "target_c", "forecast_delta_c"):
                    arrays[field] = np.empty(0, np.float32)
                arrays["calm_clear"] = np.empty(0, bool)
            else:
                payloads = []
                for batch, start in enumerate(range(0, len(all_stations), 64)):
                    obs_key = f"{day}_b{batch:03d}"
                    payload = cached(folder / "observations" / f"{obs_key}.json", obs_receipt, obs_key)
                    if payload is None:
                        raise ValueError(f"Missing observations: {obs_key}")
                    validate_day(payload, day, all_stations.iloc[start:start+64], "observations")
                    payloads.append(payload)
                temperatures = selected_temperatures(payloads, stations)
                arrays = day_samples(day, forecast_stations, table, pairs, graph, temperatures, weather)
            validate_samples(arrays, forecast_stations)
            atomic_npz(path, **arrays)
            receipt[key] = sha256(path)
            write_json(receipt_path, receipt)
        print(f"samples: {day} ({number}/{len(requested)})", flush=True)
    if set(receipt) != {str(day) for day in dates()}:
        print("Partial samples only; not marking dataset ready", flush=True)
        return
    totals = {group: dict(rows=0, nights=set(), targets=set()) for group in
              ("train", "selection", "calibration", "test")}
    block_coverage = {(item["block_x"], item["block_y"]): dict(nights=set(), targets=set(), rows=0)
                      for item in split["blocks"]}
    station_blocks = forecast_stations.set_index("station_id")[["block_x", "block_y"]].to_dict("index")
    excluded_nights = []
    for key, checksum in receipt.items():
        weather = cached(folder / "regimes" / f"{key}.json", weather_receipt, key)
        validate_regime(weather, date.fromisoformat(key), forecast_stations)
        path = destination / f"{key}.npz"
        if sha256(path) != checksum:
            raise ValueError(f"Sample checksum mismatch: {key}")
        with np.load(path, allow_pickle=False) as arrays:
            validate_samples(arrays, forecast_stations)
            if weather.get("state") == "excluded_missing_run":
                if len(arrays["y"]):
                    raise ValueError("Excluded forecast night contains training rows")
                excluded_nights.append(weather)
            for group, counts in totals.items():
                mask = arrays["split"] == group
                counts["rows"] += int(mask.sum())
                counts["nights"].update(arrays["night"][mask])
                counts["targets"].update(arrays["target"][mask])
            for target in np.unique(arrays["target"]):
                block = station_blocks[target]
                key = (block["block_x"], block["block_y"])
                if key in block_coverage:
                    mask = arrays["target"] == target
                    counts = block_coverage[key]
                    counts["rows"] += int(mask.sum())
                    counts["targets"].add(target)
                    counts["nights"].update(arrays["night"][mask])
    if any(counts["rows"] < 2 or len(counts["nights"]) < 15 or len(counts["targets"]) < 2
           for counts in totals.values()):
        raise ValueError("Insufficient final paired split coverage; inspect exclusions, not a ready dataset")
    if any(len(counts["targets"]) < 2 or len(counts["nights"]) < 15 for counts in block_coverage.values()):
        raise ValueError("A held-out geographic block has insufficient paired coverage; not a ready dataset")
    counts = {key: dict(rows=value["rows"], nights=len(value["nights"]), targets=len(value["targets"]))
              for key, value in totals.items()}
    manifest = dict(schema=SCHEMA, state="ready", features=FEATURES, terrain_count=16, regime_count=28,
                    counts=counts, shards=receipt, splits=split,
                    excluded_forecast_nights=excluded_nights,
                    forecast_nights=dict(requested=len(receipt), excluded=len(excluded_nights),
                                         available=len(receipt)-len(excluded_nights)),
                    heldout_block_coverage=[
                        dict(block_x=key[0], block_y=key[1], rows=value["rows"],
                             nights=len(value["nights"]), targets=len(value["targets"]))
                        for key, value in block_coverage.items()],
                    provenance=provenance, pairs_sha256=sha256(pair_path),
                    weighting="Pairs share targets/timestamps; balance target-night and geography during later fitting.",
                    compatibility="National daily shards; NOT the three-winter valley neural NPZ schema. Requires a national trainer.",
                    no_fit="No model fitted, no normalization computed, no test metrics inspected.")
    write_json(folder / "dataset_manifest.json", manifest)
    print(json.dumps(counts, indent=2), flush=True)


def run(args):
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with computation_lock(folder):
        stage = args.stage
        try:
            config = initialize(args.root.resolve(), folder)
            stations = pd.read_csv(folder / "stations.csv")
            print(f"National catalogue: {len(stations)}terrain-supported stations; stage={args.stage}", flush=True)
            requested = [date.fromisoformat(args.day)] if args.day else list(dates())
            if any(day not in set(dates()) for day in requested):
                raise ValueError("Day outside declared national winters")
            write_json(folder / "status.json", dict(state="running", stage=args.stage))
            stages = ("preflight", "observations", "freeze", "regimes", "build") if args.stage == "all" else (args.stage,)
            if args.stage == "all" and args.day:
                raise ValueError("Use individual provider stages for a single-day smoke, not --stage all")
            for stage in stages:
                write_json(folder / "status.json", dict(state="running", stage=stage))
                if stage == "preflight":
                    preflight(folder, stations)
                elif stage == "observations":
                    observations(folder, stations, requested)
                elif stage == "audit":
                    if (folder / "splits.json").exists():
                        load_splits(folder)
                    else:
                        audit(folder, stations)
                elif stage == "freeze":
                    freeze(folder, stations)
                elif stage == "regimes":
                    load_splits(folder)
                    split_stations = pd.read_csv(folder / "station_splits.csv")
                    active = split_stations.loc[split_stations["group"].isin(["train", "validation", "test"])]
                    regimes(folder, active.reset_index(drop=True), requested)
                elif stage == "build":
                    build(folder, stations, requested)
            ready = (folder / "dataset_manifest.json").exists()
            write_json(folder / "status.json", dict(state="ready" if ready else "stage_complete",
                                                    stage=args.stage, dataset_ready=ready,
                                                    winters=config["winters"], stations=len(stations)))
        except Exception as error:
            write_json(folder / "status.json", dict(state="failed", stage=stage, error=str(error)))
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "national_station_dataset")
    parser.add_argument("--stage", choices=("terrain", "preflight", "observations", "audit", "freeze", "regimes", "build", "all"),
                        default="terrain")
    parser.add_argument("--day", help="Bounded single-night smoke; never marks a full dataset ready")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
