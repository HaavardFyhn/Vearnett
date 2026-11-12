import sqlite3
import pandas as pd
import geopandas as gpd
from shapely.prepared import prep
import time

from pathlib import Path

DATABASE = "frost_history.db"
NORWAY_POLYGON_FILE = "outputs/norway_polygon.gpkg"
t0_total = time.perf_counter()
# Load Norway polygon
norway = gpd.read_file(
    NORWAY_POLYGON_FILE
)

norway_geom = norway.geometry.iloc[0]

with sqlite3.connect(DATABASE) as cx:

    # Ensure column exists
    columns = {
        row[1]
        for row in cx.execute(
            "PRAGMA table_info(stations)"
        )
    }

    if "inside_norway" not in columns:

        cx.execute(
            """
            ALTER TABLE stations
            ADD COLUMN inside_norway INTEGER
            """
        )

        cx.commit()

    t0 = time.perf_counter()

    stations = pd.read_sql(
        """
        SELECT
            station_id,
            lat,
            lon
        FROM stations
        """,
        cx
    )

    print(
        f"Read stations: "
        f"{time.perf_counter()-t0:.2f}s"
    )


t0 = time.perf_counter()

stations = gpd.GeoDataFrame(
    stations,
    geometry=gpd.points_from_xy(
        stations.lon,
        stations.lat
    ),
    crs=4326
).to_crs(25833)

print(
    f"Create GeoDataFrame: "
    f"{time.perf_counter()-t0:.2f}s"
)

print(
    f"Stations in database: "
    f"{len(stations):,}"
)
t0 = time.perf_counter()

prepared_norway = prep(
    norway_geom
)
print(
    f"Prepared Norway polygon: "
    f"{time.perf_counter()-t0:.2f}s"
)
t0 = time.perf_counter()

stations["inside_norway"] = [
    prepared_norway.contains(g)
    for g in stations.geometry
]


print(
    f"Polygon test: "
    f"{time.perf_counter()-t0:.2f}s"
)


print(
    stations["inside_norway"]
    .value_counts()
)
t0 = time.perf_counter()
with sqlite3.connect(DATABASE) as cx:

    cx.executemany(
        """
        UPDATE stations
        SET inside_norway = ?
        WHERE station_id = ?
        """,
        [
            (
                int(row.inside_norway),
                row.station_id
            )
            for _, row
            in stations.iterrows()
        ]
    )

    cx.commit()

print(
    f"Updated {len(stations):,} stations: "
    f"{time.perf_counter()-t0:.2f}s"
)


print(
    stations["inside_norway"]
    .value_counts()
)
print(
    f"Total time: "
    f"{time.perf_counter()-t0_total:.2f}s"
)