"""Fetch Frost stations and store coordinates, elevation, and inside_norway.

Run directly:
    python fetch_stations.py
    python fetch_stations.py --force

Or from main.py:
    from fetch_stations import run as fetch_stations_run
    fetch_stations_run(force=a.force)
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd
import requests
from shapely.prepared import prep

from pipeline_config import (
    DATABASE,
    DOWNLOAD_TIMEOUT_SECONDS,
    FROST_BASE_URL,
    FROST_CLIENT_ID_ENV,
    NORWAY_POLYGON_FILE,
)

STATUS_STEP = "fetch_stations_v1"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _connect():
    cx = sqlite3.connect(DATABASE)
    cx.execute("PRAGMA journal_mode=WAL")
    cx.execute("PRAGMA synchronous=NORMAL")
    cx.execute("""
        CREATE TABLE IF NOT EXISTS stations(
            station_id TEXT PRIMARY KEY,
            name TEXT,
            lat REAL,
            lon REAL,
            elevation REAL,
            inside_norway INTEGER
        )
    """)

    columns = {row[1] for row in cx.execute("PRAGMA table_info(stations)")}
    required = {
        "name": "TEXT",
        "lat": "REAL",
        "lon": "REAL",
        "elevation": "REAL",
        "inside_norway": "INTEGER",
    }
    for column, sql_type in required.items():
        if column not in columns:
            cx.execute(f"ALTER TABLE stations ADD COLUMN {column} {sql_type}")

    cx.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_status(
            step TEXT PRIMARY KEY,
            completed_at TEXT NOT NULL,
            row_count INTEGER NOT NULL
        )
    """)
    cx.execute("""
        CREATE INDEX IF NOT EXISTS idx_stations_inside_norway
        ON stations(inside_norway)
    """)
    cx.commit()
    return cx

def _request_all(url, params, client_id):
    while url:
        response = None
        for attempt in range(6):
            try:
                response = requests.get(
                    url,
                    params=params,
                    auth=(client_id, ""),
                    timeout=DOWNLOAD_TIMEOUT_SECONDS,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == 5:
                    raise RuntimeError(
                        f"Frost connection failed after 6 attempts: {url}"
                    ) from exc
                wait = 5 * (2 ** attempt)
                print(f"Connection failed; retrying in {wait}s ({attempt + 1}/6)")
                time.sleep(wait)
                continue

            if response.status_code == 200:
                break
            if response.status_code in RETRYABLE_STATUS:
                if attempt == 5:
                    raise RuntimeError(
                        f"Frost HTTP {response.status_code} after 6 attempts: {response.url}"
                    )
                wait = 5 * (2 ** attempt)
                print(f"HTTP {response.status_code}; retrying in {wait}s ({attempt + 1}/6)")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Frost HTTP {response.status_code}\n"
                f"URL: {response.url}\n"
                f"Response: {response.text[:4000]}"
            )
        else:
            raise RuntimeError(f"Frost request failed after 6 attempts: {url}")

        payload = response.json()
        yield from payload.get("data", [])
        url = payload.get("@nextLink") or payload.get("nextLink")
        params = None


def run():
    client_id = os.getenv(FROST_CLIENT_ID_ENV)
    if not client_id:
        raise RuntimeError(
            f"Set environment variable {FROST_CLIENT_ID_ENV} to your Frost client ID"
        )
    if not NORWAY_POLYGON_FILE.exists():
        raise FileNotFoundError(f"Missing Norway polygon: {NORWAY_POLYGON_FILE}")

    with _connect() as cx:
        total = cx.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
        inside = cx.execute(
            "SELECT COUNT(*) FROM stations WHERE inside_norway=1"
        ).fetchone()[0]
        print(
            f"SKIP stations; already complete "
            f"({total:,} total, {inside:,} inside Norway)"
        )
        return total

    print("Fetching station metadata from Frost...")
    url = f"{FROST_BASE_URL}/sources/v0.jsonld"
    params = {
        "country": "NO",
        "types": "SensorSystem",
        "fields": "id,name,geometry,masl",
    }

    records = []
    for item in _request_all(url, params, client_id):
        station_id = str(item.get("id", "")).split(":")[0]
        coords = (item.get("geometry") or {}).get("coordinates") or [None, None]
        lon, lat = (list(coords) + [None, None])[:2]
        if station_id:
            records.append(
                (station_id, item.get("name"), lat, lon, item.get("masl"))
            )
    print(f"Fetched {len(records):,} stations from Frost")
    if not records:
        raise RuntimeError("Frost returned no stations")

    stations = pd.DataFrame(
        records,
        columns=["station_id", "name", "lat", "lon", "elevation"],
    ).drop_duplicates("station_id", keep="last")
    stations = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(
            stations.lon,
            stations.lat
        ),
        crs=4326
    ).to_crs(25833)
    stations["inside_norway"] = 0
    norway = gpd.read_file(
    NORWAY_POLYGON_FILE
    )
    norway_geom = norway.geometry.iloc[0]
    prepared_norway = prep(
        norway_geom
    )

    stations["inside_norway"] = [
        prepared_norway.contains(g)
        for g in stations.geometry
    ]

    rows = list(
        stations[
            ["station_id", "name", "lat", "lon", "elevation", "inside_norway"]
        ].itertuples(index=False, name=None)
    )

    with _connect() as cx:
        cx.executemany("""
            INSERT INTO stations(
                station_id, name, lat, lon, elevation, inside_norway
            ) VALUES(?,?,?,?,?,?)
            ON CONFLICT(station_id) DO UPDATE SET
                name=excluded.name,
                lat=excluded.lat,
                lon=excluded.lon,
                elevation=excluded.elevation,
                inside_norway=excluded.inside_norway
        """, rows)
        cx.execute("""
            INSERT INTO pipeline_status(step, completed_at, row_count)
            VALUES(?,?,?)
            ON CONFLICT(step) DO UPDATE SET
                completed_at=excluded.completed_at,
                row_count=excluded.row_count
        """, (
            STATUS_STEP,
            datetime.now(timezone.utc).isoformat(),
            len(rows),
        ))
        cx.commit()

    inside = int(stations["inside_norway"].sum())
    print(f"Stored {len(stations):,} stations; {inside:,} inside Norway")
    return len(stations)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Frost stations and classify them inside Norway"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
