"""Polygonize lake IDs in bounded blocks and join lake CSV attributes."""
from __future__ import annotations

import argparse
from contextlib import closing
from itertools import groupby
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from shapely import from_wkb, union_all
from shapely.geometry import MultiPolygon, shape


DEFAULT_FOLDER = (Path(__file__).resolve().parent / "outputs"
                  / "potential_lakes_norway_DTM10_sill20_tiled")


def export_lakes(raster_path, csv_path, output_path, block_size=2048):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    table = pd.read_csv(csv_path).set_index("lake_id", verify_integrity=True)
    if table.empty or (table.index <= 0).any():
        raise ValueError("CSV must contain positive, unique lake IDs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path.with_name(output_path.stem + ".pending.gpkg")
    if pending.exists():
        raise FileExistsError(f"Pending export already exists: {pending}")
    written = 0
    seen = set()
    with tempfile.TemporaryDirectory(prefix="lake_polygonize_") as scratch:
        with closing(sqlite3.connect(Path(scratch) / "parts.sqlite")) as connection:
            connection.execute("CREATE TABLE parts (lake_id INTEGER NOT NULL, geometry BLOB NOT NULL)")
            with rasterio.Env(GDAL_CACHEMAX=256 * 1024**2), rasterio.open(raster_path) as source:
                if source.crs is None or source.crs.to_epsg() != 25833:
                    raise ValueError("Lake ID raster must be EPSG:25833")
                crs = source.crs
                blocks = ((source.height + block_size - 1) // block_size
                          * ((source.width + block_size - 1) // block_size))
                completed = 0
                for row in range(0, source.height, block_size):
                    for col in range(0, source.width, block_size):
                        window = Window(col, row, min(block_size, source.width - col),
                                        min(block_size, source.height - row))
                        values = source.read(1, window=window)
                        mask = values > 0
                        if mask.any():
                            records = [
                                (int(value), shape(geometry).wkb)
                                for geometry, value in shapes(
                                    values, mask=mask, connectivity=4,
                                    transform=source.window_transform(window),
                                )
                            ]
                            connection.executemany("INSERT INTO parts VALUES (?, ?)", records)
                        completed += 1
                        if completed % 100 == 0 or completed == blocks:
                            connection.commit()
                            print(f"Polygonizing raster: {completed}/{blocks} blocks "
                                  f"({100 * completed / blocks:.1f}%)", flush=True)
            connection.commit()
            print("Merging block boundaries and joining CSV attributes...", flush=True)
            connection.execute("CREATE INDEX parts_id ON parts(lake_id)")
            cursor = connection.execute("SELECT lake_id, geometry FROM parts ORDER BY lake_id")
            batch = []

            def write_batch(records):
                fields = ["lake_id", *table.columns]
                pyogrio.raw.write(
                    pending,
                    geometry=np.asarray([record["geometry"].wkb for record in records], dtype=object),
                    field_data=[np.asarray([record[field] for record in records]) for field in fields],
                    fields=fields, layer="potential_lakes", driver="GPKG",
                    geometry_type="MultiPolygon", crs=crs.to_wkt(), append=pending.exists(),
                )

            for lake_id, parts in groupby(cursor, key=lambda record: record[0]):
                if lake_id not in table.index:
                    raise ValueError(f"Raster lake ID {lake_id} has no CSV row")
                geometry = union_all([from_wkb(record[1]) for record in parts])
                if geometry.geom_type == "Polygon":
                    geometry = MultiPolygon([geometry])
                if geometry.geom_type != "MultiPolygon" or not geometry.is_valid:
                    raise ValueError(f"Invalid lake polygon for ID {lake_id}")
                attributes = table.loc[lake_id].to_dict()
                if not np.isclose(geometry.area, attributes["area_m2"], rtol=1e-8, atol=0.01):
                    raise ValueError(f"Polygon area does not match CSV for lake {lake_id}")
                batch.append(dict(lake_id=lake_id, **attributes, geometry=geometry))
                seen.add(lake_id)
                written += 1
                if len(batch) >= 500:
                    write_batch(batch)
                    batch.clear()
                    if written % 5000 == 0:
                        print(f"Joined {written:,}/{len(table):,} lakes", flush=True)
            if batch:
                write_batch(batch)
    if seen != set(table.index):
        raise ValueError(f"{len(set(table.index) - seen)} CSV lakes missing from raster polygons")
    info = pyogrio.read_info(pending, layer="potential_lakes")
    if info["features"] != len(table):
        raise ValueError("Exported feature count does not match CSV")
    if info["crs"] != "EPSG:25833":
        raise ValueError(f"Exported CRS is incorrect: {info['crs']}")
    pending.replace(output_path)
    print(f"Complete: {written:,} lakes with CSV attributes: {output_path}", flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    args = parser.parse_args()
    export_lakes(args.folder / "potential_lake_id.tif",
                 args.folder / "potential_lakes.csv",
                 args.folder / "potential_lakes.gpkg")


if __name__ == "__main__":
    main()