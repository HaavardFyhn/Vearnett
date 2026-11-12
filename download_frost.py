"""Incremental Frost downloader used by main.py.
Requires internet and environment variable FROST_CLIENT_ID.
Existing element/month data are skipped unless force=True.
"""
import os, sqlite3, calendar
import time as time_module
import math
from contextlib import closing
import pandas as pd
from datetime import datetime, time, timezone, timedelta
from dateutil.relativedelta import relativedelta
import requests
from pipeline_config import (DATABASE, FROST_CLIENT_ID_ENV, FROST_SOURCE_BATCH_SIZE, PHENOMENA, FROST_BASE_URL, DOWNLOAD_TIMEOUT_SECONDS, NORWAY_POLYGON_FILE, element_update_marker)

PRECIP_DAILY = "sum(precipitation_amount P1D)"
PRECIP_TEN_MINUTES = "sum(precipitation_amount PT10M)"
PRECIP_MIN_DAILY_READINGS = math.ceil(144 * 0.90)
PRECIP_AGGREGATION_VERSION = "coverage90_zero_missing_06utc_v2"


def _touch_marker(element):
    marker = element_update_marker(DATABASE, element)
    marker.parent.mkdir(exist_ok=True)
    marker.touch()


def _connect():
    cx=sqlite3.connect(DATABASE)
    cx.execute("PRAGMA journal_mode=WAL")
    cx.execute("PRAGMA synchronous=NORMAL")
    cx.execute("""CREATE TABLE IF NOT EXISTS observations(
        station_id TEXT NOT NULL, reference_time TEXT NOT NULL,
        element TEXT NOT NULL, value REAL,
        PRIMARY KEY(station_id,reference_time,element))""")
    cx.execute("""
    CREATE TABLE IF NOT EXISTS download_status(
        element TEXT NOT NULL,
        month TEXT NOT NULL,
        completed INTEGER NOT NULL,
        PRIMARY KEY(element, month)
    )
    """)
    cx.execute("""
    CREATE TABLE IF NOT EXISTS batch_status(
        element TEXT NOT NULL,
        month TEXT NOT NULL,
        station_count INTEGER NOT NULL,
        completed INTEGER NOT NULL,
        PRIMARY KEY(
            element,
            month,
            station_count
        )
    )
    """)

    cx.execute("""CREATE TABLE IF NOT EXISTS precipitation_fallback_status(
        station_id TEXT NOT NULL, month TEXT NOT NULL, horizon TEXT NOT NULL,
        version TEXT NOT NULL, PRIMARY KEY(station_id, month))""")
    cx.execute("""CREATE TABLE IF NOT EXISTS precipitation_daily_derived(
        station_id TEXT NOT NULL, reference_time TEXT NOT NULL,
        PRIMARY KEY(station_id, reference_time))""")

    return cx


def _months(years, end=None):
    end=end or datetime.now(timezone.utc)
    start=end-relativedelta(years=years)
    cur=datetime(start.year,start.month,1,tzinfo=timezone.utc)
    stop=datetime(end.year,end.month,1,tzinfo=timezone.utc)
    while cur <= stop:
        nxt=cur+relativedelta(months=1)
        yield cur, min(nxt,end)
        cur=nxt


def _iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_all(
    url,
    params,
    client_id
):

    while url:
        response = None
        for attempt in range(6):
            try:
                response = requests.get(
                    url,
                    params=params,
                    auth=(client_id, ""),
                    timeout=DOWNLOAD_TIMEOUT_SECONDS
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ):
                wait = 5 * (2 ** attempt)

                print(
                    f"Connection failed. "
                    f"Retrying in {wait}s "
                    f"(attempt {attempt+1}/6)"
                )

                time_module.sleep(wait)

                continue

            reason = (response.json().get("error", {}).get("reason", "").lower()
                      if response.status_code == 412 else "")
            no_series = response.status_code == 412 and (
                "no matching time series" in reason or
                "no time series found for this combination of parameters" in reason)
            if response.status_code == 404 or no_series:

                payload = response.json()

                reason = (
                    payload.get("error", {})
                    .get(
                        "reason",
                        "No matching time series"
                    )
                )

                print(
                    f"SKIP batch: {reason}"
                )

                return
            if response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:
                wait = 5 * (2 ** attempt)

                print(
                    f"HTTP {response.status_code}. "
                    f"Retrying in {wait}s "
                    f"(attempt {attempt+1}/6)"
                )

                time_module.sleep(wait)
                continue
            
            elif response.status_code == 200:
                break

            raise RuntimeError(
                f"Frost HTTP {response.status_code}\n"
                f"URL:\n{response.url}\n\n"
                f"Response:\n{response.text[:4000]}"
            )

        else:
            raise RuntimeError("Frost request failed after 6 attempts; batch and month must remain incomplete")
        payload = response.json()

        for row in payload.get(
            "data",
            []
        ):
            yield row

        url = (
            payload.get("@nextLink")
            or
            payload.get("nextLink")
        )
        params = None


def month_complete(
    cx,
    element,
    start
):

    key = start.strftime("%Y-%m")

    row = cx.execute(
        """
        SELECT completed
        FROM download_status
        WHERE element=?
          AND month=?
        """,
        (element, key)
    ).fetchone()

    return row is not None and row[0] == 1


def mark_month_complete(cx, element, start):
    key = start.strftime("%Y-%m")
    cx.execute(
        """
        INSERT INTO download_status(element, month, completed)
        VALUES(?, ?, 1)
        ON CONFLICT(element, month) DO UPDATE SET
        completed=1
        """,
        (element, key)
    )

def batch_complete(
    cx,
    element,
    start,
    batch_id
):

    month = start.strftime("%Y-%m")

    row = cx.execute(
        """
        SELECT completed
        FROM batch_status
        WHERE element=?
          AND month=?
          AND batch_id=?
        """,
        (
            element,
            month,
            batch_id
        )
    ).fetchone()

    return (
        row is not None
        and
        row[0] == 1
    )

def mark_batch_complete(
    cx,
    element,
    start,
    station_count
):

    month = start.strftime("%Y-%m")

    cx.execute(
        """
        INSERT INTO batch_status(
            element,
            month,
            station_count,
            completed
        )
        VALUES(?,?,?,1)

        ON CONFLICT(
            element,
            month,
            station_count
        )
        DO UPDATE SET
            completed=1
        """,
        (
            element,
            month,
            station_count
        )
    )

def station_id_batches(batch_size=50):
    with closing(_connect()) as cx, cx:
        station_ids = pd.read_sql(
            """
            SELECT station_id
            FROM stations
            WHERE inside_norway = 1
            ORDER BY station_id
            """,
            cx
        )["station_id"].tolist()
    for start in range(
        0,
        len(station_ids),
        batch_size
    ):

        yield station_ids[
            start:start + batch_size
        ]


def fetch_batch(
    client_id,
    station_ids,
    element,
    start,
    end
):

    params = {
        "sources": ",".join(station_ids),
        "elements": element,
        "referencetime":
            f"{_iso(start)}/{_iso(end)}"
    }

    url = (
        f"{FROST_BASE_URL}"
        "/observations/v0.jsonld"
    )

    for item in _request_all(
        url,
        params,
        client_id
    ):
        yield item

def fetch_batch_recursive(
    client_id,
    station_ids,
    element,
    start,
    end
):

    try:

        yield from fetch_batch(
            client_id,
            station_ids,
            element,
            start,
            end
        )

    except RuntimeError as exc:

        msg = str(exc)

        if (
            "exceeds the maximum number allowed"
            in msg
        ):

            if len(station_ids) == 1:
                if end - start <= timedelta(days=1):
                    raise RuntimeError("Frost response is too large for one station/day; batch remains incomplete") from exc
                mid_time = start + (end - start) / 2
                yield from fetch_batch_recursive(client_id, station_ids, element, start, mid_time)
                yield from fetch_batch_recursive(client_id, station_ids, element, mid_time, end)
                return

            mid = len(station_ids) // 2

            left = station_ids[:mid]
            right = station_ids[mid:]

            print(
                f"Splitting batch "
                f"{len(station_ids)}"
                f" -> "
                f"{len(left)} + "
                f"{len(right)}"
            )

            yield from fetch_batch_recursive(
                client_id,
                left,
                element,
                start,
                end
            )

            yield from fetch_batch_recursive(
                client_id,
                right,
                element,
                start,
                end
            )

        else:

            raise


def _daily_mean_rows(sums, element):
    return [(sid, f"{day}T00:00:00Z", element, total / count) for (sid, day), (total, count) in sums.items()]


def download_month(client_id, element, start, end, station_batches, force=False, daily_mean=False):
    with closing(_connect()) as cx, cx:
        if month_complete( cx, element, start ) and not force:
            print(f"SKIP data {element}: {start:%Y-%m}"); return 0
        if force:
            cx.execute("DELETE FROM observations WHERE element=? AND reference_time>=? AND reference_time<?",
                       (element,_iso(start),_iso(end))); cx.commit()
    batches = station_batches


    url = f"{FROST_BASE_URL}/observations/v0.jsonld"

    rows = []
    count = 0

    with closing(_connect()) as cx, cx:
        print(
            f"DOWNLOAD {element} "
            f"{start:%Y-%m}: "
            f"{len(batches)} station batches"
        )

        for batch_number, station_ids in enumerate(
            batches,
            start=1
        ):
            
            print(
                f"Batch "
                f"{batch_number}/{len(batches)}"
            )
            
            stations_processed = ( batch_number * FROST_SOURCE_BATCH_SIZE )
            if not force:
                row = cx.execute( """ SELECT 1 FROM batch_status WHERE element=? AND month=? AND station_count>=? LIMIT 1 """, ( element, start.strftime("%Y-%m"), stations_processed ) ).fetchone()
                if row is not None:

                    print(
                        f"SKIP batch "
                        f"{batch_number}"
                    )

                    continue


            params = {
                "sources":
                    ",".join(station_ids),

                "elements":
                    element,

                "referencetime":
                    f"{_iso(start)}/{_iso(end)}"
            }

            sums = {}
            for item in fetch_batch_recursive(client_id,station_ids,element,start,end):
                sid=str(item.get("sourceId","")).split(":")[0]; ref=item.get("referenceTime")
                for ob in item.get("observations",[]):
                    if ob.get("value") is not None:
                        if daily_mean:
                            key = (sid, str(ref)[:10])
                            total, n = sums.get(key, (0.0, 0))
                            sums[key] = (total + float(ob["value"]), n + 1)
                        else:
                            rows.append((sid,ref,ob.get("elementId",element),float(ob["value"])))
                if len(rows)>=10000 and element != PRECIP_DAILY:
                    cx.executemany("INSERT OR REPLACE INTO observations VALUES(?,?,?,?)",rows)
                    count+=len(rows); rows.clear(); cx.commit()
            if sums:
                rows.extend(_daily_mean_rows(sums, element))
            if rows:
                if element == PRECIP_DAILY:
                    for sid, ref, _, _ in rows:
                        derived = cx.execute(
                            "SELECT reference_time FROM precipitation_daily_derived "
                            "WHERE station_id=? AND substr(reference_time,1,10)=?",
                            (sid, str(ref)[:10])).fetchall()
                        for (old_ref,) in derived:
                            cx.execute("DELETE FROM observations WHERE station_id=? AND reference_time=? AND element=?",
                                       (sid, old_ref, element))
                            cx.execute("DELETE FROM precipitation_daily_derived WHERE station_id=? AND reference_time=?",
                                       (sid, old_ref))
                cx.executemany("INSERT OR REPLACE INTO observations VALUES(?,?,?,?)", rows)
                count += len(rows)
                rows.clear()
            mark_batch_complete(
                cx,
                element,
                start,
                stations_processed
            )

            cx.commit()
            _touch_marker(element)

        if rows:

            cx.executemany(
                "INSERT OR REPLACE INTO observations VALUES(?,?,?,?)",
                rows
            )

            count += len(rows)
        
        mark_month_complete( cx, element, start )
        cx.commit()
        _touch_marker(element)
        

    print(
        f"DONE data {element} "
        f"{start:%Y-%m}: "
        f"{count:,} values"
    )

    return count


def precipitation_fallback_sources(client_id, station_batches, start, end):
    daily_sources = set()
    ten_minute_sources = {}
    for station_ids in station_batches:
        for element in (PRECIP_DAILY, PRECIP_TEN_MINUTES):
            params = {"sources": ",".join(station_ids), "elements": element,
                      "referencetime": f"{_iso(start)}/{_iso(end)}"}
            for series in _request_all(f"{FROST_BASE_URL}/observations/availableTimeSeries/v0.jsonld",
                                       params, client_id):
                source = str(series["sourceId"])
                if element == PRECIP_DAILY:
                    daily_sources.add(source.split(":")[0])
                elif (source.endswith(":0") and series.get("timeSeriesId") == 0
                      and series.get("timeResolution") == "PT10M" and not series.get("level")):
                    valid_from = datetime.fromisoformat(series["validFrom"].replace("Z", "+00:00"))
                    valid_to = (datetime.fromisoformat(series["validTo"].replace("Z", "+00:00"))
                                if series.get("validTo") else end)
                    ten_minute_sources.setdefault(source.split(":")[0], []).append((valid_from, valid_to))
    fallback = {sid: ten_minute_sources[sid] for sid in sorted(ten_minute_sources.keys() - daily_sources)}
    print(f"Precipitation series: {len(daily_sources)} daily stations; "
          f"{len(fallback)} ten-minute-only stations", flush=True)
    return fallback


def precipitation_daily_rows(items, start, end):
    slots = {}
    rejected = 0
    for item in items:
        source = str(item.get("sourceId", ""))
        if not source.endswith(":0"):
            continue
        sid = source.split(":")[0]
        stamp = datetime.fromisoformat(item["referenceTime"].replace("Z", "+00:00")).astimezone(timezone.utc)
        if stamp.second or stamp.microsecond or stamp.minute % 10:
            rejected += 1
            continue
        # Ten-minute sums end at referenceTime. Daily Frost rain ends at 06 UTC.
        day = (stamp - timedelta(hours=6, microseconds=1)).date() + timedelta(days=1)
        ref = datetime.combine(day, time(), tzinfo=timezone.utc)
        if not start <= ref < end or ref + timedelta(hours=6) > end:
            continue
        for ob in item.get("observations", []):
            if (ob.get("elementId") != PRECIP_TEN_MINUTES or ob.get("timeSeriesId") != 0
                    or ob.get("timeResolution") != "PT10M" or ob.get("level")):
                continue
            value = ob.get("value")
            quality = ob.get("qualityCode")
            if value is None or not math.isfinite(float(value)) or float(value) < 0 or quality not in (0, 1, 2, 4):
                rejected += 1
                continue
            readings = slots.setdefault((sid, ref), {})
            value = float(value)
            if stamp in readings and readings[stamp] != value:
                raise ValueError(f"Conflicting precipitation readings for {sid} at {_iso(stamp)}")
            readings[stamp] = value
    rows = []
    incomplete = 0
    zero_filled = 0
    for (sid, ref), readings in slots.items():
        if len(readings) < PRECIP_MIN_DAILY_READINGS:
            incomplete += 1
            continue
        if len(readings) < 144:
            zero_filled += 1
        rows.append((sid, _iso(ref), PRECIP_DAILY, math.fsum(readings.values())))
    print(f"Ten-minute aggregation: {len(rows)} accepted days (at least 90% coverage); "
          f"{zero_filled} days with missing intervals counted as zero; "
          f"{incomplete} days below 90%; {rejected} invalid readings", flush=True)
    return rows


def download_precipitation_fallback(client_id, station_ids, start, end, force=False):
    month = start.strftime("%Y-%m")
    horizon = _iso(end)
    count = 0
    with closing(_connect()) as cx:
        pending = [sid for sid in station_ids if force or cx.execute(
            "SELECT 1 FROM precipitation_fallback_status "
            "WHERE station_id=? AND month=? AND horizon=? AND version=?",
            (sid, month, horizon, PRECIP_AGGREGATION_VERSION)).fetchone() is None]
        cx.commit()
        batch_size = min(FROST_SOURCE_BATCH_SIZE, 8)
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            print(f"DOWNLOAD ten-minute rain {month}: stations {offset + 1}-{offset + len(batch)}/{len(pending)}",
                  flush=True)
            request_start = start - timedelta(days=1) + timedelta(hours=6)
            rows = precipitation_daily_rows(
                fetch_batch_recursive(client_id, batch, PRECIP_TEN_MINUTES, request_start, end), start, end)
            with cx:
                for sid, ref, element, value in rows:
                    next_day = (datetime.fromisoformat(ref.replace("Z", "+00:00")) + timedelta(days=1)).date().isoformat()
                    existing = cx.execute(
                        "SELECT reference_time FROM observations WHERE station_id=? "
                        "AND reference_time>=? AND reference_time<? AND element=?",
                        (sid, ref[:10], next_day, element)).fetchall()
                    if existing and not all(cx.execute(
                        "SELECT 1 FROM precipitation_daily_derived WHERE station_id=? AND reference_time=?",
                        (sid, old_ref)).fetchone() for (old_ref,) in existing):
                        continue
                    cx.execute("INSERT OR REPLACE INTO observations VALUES(?,?,?,?)", (sid, ref, element, value))
                    cx.execute("INSERT OR REPLACE INTO precipitation_daily_derived VALUES(?,?)", (sid, ref))
                    count += 1
                cx.executemany("INSERT OR REPLACE INTO precipitation_fallback_status VALUES(?,?,?,?)",
                               [(sid, month, horizon, PRECIP_AGGREGATION_VERSION) for sid in batch])
            if rows:
                _touch_marker(PRECIP_DAILY)
    return count


def run(phenomenon, years, force=False):
    client_id=os.getenv(FROST_CLIENT_ID_ENV)
    if not client_id:
        raise RuntimeError(f"Set environment variable {FROST_CLIENT_ID_ENV} to your Frost client ID")
    element=PHENOMENA[phenomenon]["element"]
    daily_mean=PHENOMENA[phenomenon].get("download_daily_mean", False)
    total=0

    station_batches = list( station_id_batches(batch_size=FROST_SOURCE_BATCH_SIZE) )

    months = list(_months(years))
    fallback = (precipitation_fallback_sources(client_id, station_batches, months[0][0], months[-1][1])
                if phenomenon == "precipitation" else {})
    for start,end in months:
        total += download_month(client_id, element,start,end,station_batches, force, daily_mean)
        active = [sid for sid, periods in fallback.items()
                  if any(valid_from < end and valid_to > start - timedelta(days=1) + timedelta(hours=6)
                         for valid_from, valid_to in periods)]
        if active:
            total += download_precipitation_fallback(client_id, active, start, end, force)
    print(f"Frost update complete: {total:,} downloaded values")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--phenomenon",choices=PHENOMENA,required=True)
    p.add_argument("--years",type=int,default=1); p.add_argument("--force",action="store_true")
    a=p.parse_args(); run(a.phenomenon,a.years,a.force)
