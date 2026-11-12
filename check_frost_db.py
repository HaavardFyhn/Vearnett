"""Inspect the local Frost SQLite database without changing it.
Usage: python check_frost_db.py [path_to_frost_history.db]
Writes CSV summaries to frost_db_report/.
"""
from pathlib import Path
import sqlite3, sys
import pandas as pd

DB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "frost_history.db"
REPORT = DB.parent / "frost_db_report"
REPORT.mkdir(exist_ok=True)

if not DB.exists():
    raise FileNotFoundError(f"Database not found: {DB}")

def q(cx, sql, params=()):
    return pd.read_sql_query(sql, cx, params=params)

def write(df, name):
    df.to_csv(REPORT / name, index=False, encoding="utf-8-sig")
    print(f"\n{name}\n{df.to_string(index=False)}")

with sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True) as cx:
    tables=q(cx,"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    write(tables,"tables.csv")
    if "observations" not in set(tables["name"]):
        raise RuntimeError("No observations table found")
    cols=q(cx,"PRAGMA table_info(observations)")
    write(cols,"observation_schema.csv")
    names=set(cols["name"])
    required={"station_id","reference_time","element","value"}
    missing=required-names
    if missing: raise RuntimeError(f"observations is missing columns: {sorted(missing)}")

    overview=q(cx,"""
      SELECT element,
             COUNT(*) AS rows,
             COUNT(DISTINCT station_id) AS stations,
             MIN(reference_time) AS first_time,
             MAX(reference_time) AS last_time,
             SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS null_values
      FROM observations GROUP BY element ORDER BY element
    """)
    write(overview,"element_overview.csv")

    yearly=q(cx,"""
      SELECT element, substr(reference_time,1,4) AS year,
             COUNT(*) AS rows, COUNT(DISTINCT station_id) AS stations,
             COUNT(DISTINCT substr(reference_time,1,10)) AS dates
      FROM observations
      GROUP BY element, substr(reference_time,1,4)
      ORDER BY element, year
    """)
    write(yearly,"element_by_year.csv")

    station_element=q(cx,"""
      SELECT element, station_id,
             COUNT(*) AS rows,
             COUNT(DISTINCT substr(reference_time,1,10)) AS observed_days,
             MIN(reference_time) AS first_time,
             MAX(reference_time) AS last_time,
             SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) AS null_values
      FROM observations
      GROUP BY element, station_id
      ORDER BY element, observed_days DESC
    """)
    write(station_element.head(100),"station_element_top100_preview.csv")
    station_element.to_csv(REPORT/"station_element_all.csv",index=False,encoding="utf-8-sig")

    count_bands=q(cx,"""
      WITH x AS (
        SELECT element, station_id,
               COUNT(DISTINCT substr(reference_time,1,10)) AS days
        FROM observations GROUP BY element, station_id
      )
      SELECT element,
        SUM(CASE WHEN days>=100 THEN 1 ELSE 0 END) AS stations_ge_100_days,
        SUM(CASE WHEN days>=365 THEN 1 ELSE 0 END) AS stations_ge_365_days,
        SUM(CASE WHEN days>=730 THEN 1 ELSE 0 END) AS stations_ge_730_days,
        SUM(CASE WHEN days>=1095 THEN 1 ELSE 0 END) AS stations_ge_1095_days,
        MAX(days) AS max_days
      FROM x GROUP BY element ORDER BY element
    """)
    write(count_bands,"station_day_coverage.csv")

    if "stations" in set(tables["name"]):
        station_cols=q(cx,"PRAGMA table_info(stations)")
        write(station_cols,"station_schema.csv")
        station_count=q(cx,"SELECT COUNT(*) AS station_rows, COUNT(DISTINCT station_id) AS distinct_stations FROM stations")
        write(station_count,"station_overview.csv")

print(f"\nDatabase: {DB}")
print(f"Size: {DB.stat().st_size/1024**2:,.1f} MB")
print(f"Reports saved in: {REPORT}")
