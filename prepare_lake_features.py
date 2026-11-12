"""Aggregate nationwide 10 m potential-lake depth to a station-model grid."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from pipeline_config import ROOT, WORK


DEFAULT_GRID = WORK / "norway_grid_1000m.gpkg"
DEFAULT_LAKE_DEPTH = WORK / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lake_depth.tif"
DEFAULT_OUTPUT = WORK / "potential_lake_features_1000m"
FEATURE_FIELDS = [
    "potential_lake_mean_depth_1km",
    "potential_lake_max_depth_1km",
    "potential_lake_present_1km",
]
NODATA = -9999.0


def _grid_spacing(values):
    unique = np.unique(np.round(np.asarray(values, dtype=np.float64), 5))
    differences = np.diff(unique)
    differences = differences[differences > 0]
    if not len(differences):
        raise ValueError("The target grid has no measurable spacing")
    spacing = float(np.min(differences))
    if not np.allclose(differences / spacing, np.rint(differences / spacing), atol=1e-5, rtol=0):
        raise ValueError("The target grid is not a regular point lattice")
    return spacing


def _file_sha256(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_lake_features(grid_path=DEFAULT_GRID, lake_depth_path=DEFAULT_LAKE_DEPTH,
                          output_dir=DEFAULT_OUTPUT, force=False):
    grid_path, lake_depth_path, output_dir = map(Path, (grid_path, lake_depth_path, output_dir))
    raster_path = output_dir.with_suffix(".tif")
    gpkg_path = output_dir.with_suffix(".gpkg")
    manifest_path = output_dir.with_suffix(".json")
    outputs = (raster_path, gpkg_path, manifest_path)
    if all(path.is_file() for path in outputs) and not force:
        return dict(raster=str(raster_path), grid=str(gpkg_path), manifest=str(manifest_path), reused=True)
    if not grid_path.is_file() or not lake_depth_path.is_file():
        raise FileNotFoundError(f"Missing terrain grid or lake raster: {grid_path}, {lake_depth_path}")

    grid = pyogrio.read_dataframe(grid_path)
    if grid.crs is None or grid.crs.to_epsg() != 25833:
        raise ValueError("The station-model grid must use EPSG:25833")
    for field in ("grid_id", "x", "y", "geometry"):
        if field not in grid:
            raise ValueError(f"The station-model grid is missing {field}")
    spacing_x = _grid_spacing(grid.x.to_numpy())
    spacing_y = _grid_spacing(grid.y.to_numpy())
    if not np.isclose(spacing_x, spacing_y) or not np.isclose(spacing_x, round(spacing_x)):
        raise ValueError("The station-model grid must have square integer-metre cells")
    cell_m = int(round(spacing_x))
    x_min, y_min, x_max, y_max = map(float, grid[["x", "y"]].agg({"x": ["min", "max"], "y": ["min", "max"]}).to_numpy().ravel())
    width = int(round((x_max - x_min) / cell_m)) + 1
    height = int(round((y_max - y_min) / cell_m)) + 1
    column = np.rint((grid.x.to_numpy(dtype=np.float64) - x_min) / cell_m).astype(np.int32)
    row = np.rint((y_max - grid.y.to_numpy(dtype=np.float64)) / cell_m).astype(np.int32)
    if (column < 0).any() or (column >= width).any() or (row < 0).any() or (row >= height).any():
        raise ValueError("Target grid centres do not fit the derived raster extent")

    mean_depth = np.full((height, width), NODATA, dtype=np.float32)
    max_depth = np.full((height, width), NODATA, dtype=np.float32)
    transform = from_origin(x_min - cell_m / 2, y_max + cell_m / 2, cell_m, cell_m)
    with rasterio.open(lake_depth_path) as source:
        if source.crs is None or source.crs.to_epsg() != 25833 or source.count != 1:
            raise ValueError("Potential-lake depth must be a single-band EPSG:25833 raster")
        if source.res[0] >= cell_m or source.res[1] >= cell_m:
            raise ValueError("Potential-lake raster must be finer than the station-model grid")
        reproject(
            source=rasterio.band(source, 1), destination=mean_depth,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=transform, dst_crs="EPSG:25833", dst_nodata=NODATA,
            resampling=Resampling.average, init_dest_nodata=True,
        )
        reproject(
            source=rasterio.band(source, 1), destination=max_depth,
            src_transform=source.transform, src_crs=source.crs, src_nodata=source.nodata,
            dst_transform=transform, dst_crs="EPSG:25833", dst_nodata=NODATA,
            resampling=Resampling.max, init_dest_nodata=True,
        )

    point_mean = mean_depth[row, column]
    point_max = max_depth[row, column]
    valid = np.isfinite(point_mean) & np.isfinite(point_max) & (point_mean != NODATA) & (point_max != NODATA)
    if not valid.all():
        raise ValueError(f"{int((~valid).sum()):,} station-grid cells have no lake-raster coverage")
    grid[FEATURE_FIELDS[0]] = point_mean
    grid[FEATURE_FIELDS[1]] = point_max
    grid[FEATURE_FIELDS[2]] = (point_max > 0).astype(np.uint8)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    raster_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_raster = raster_path.with_name(raster_path.stem + ".pending.tif")
    temporary_gpkg = gpkg_path.with_name(gpkg_path.stem + ".pending.gpkg")
    temporary_manifest = manifest_path.with_name(manifest_path.stem + ".pending.json")
    try:
        profile = {
            "driver": "GTiff", "width": width, "height": height, "count": 3,
            "dtype": "float32", "crs": "EPSG:25833", "transform": transform,
            "nodata": NODATA, "compress": "deflate", "predictor": 3,
            "tiled": True, "blockxsize": 512, "blockysize": 512,
        }
        with rasterio.open(temporary_raster, "w", **profile) as target:
            for index, (values, name, units) in enumerate((
                (mean_depth, FEATURE_FIELDS[0], "m"),
                (max_depth, FEATURE_FIELDS[1], "m"),
                ((max_depth > 0).astype(np.float32), FEATURE_FIELDS[2], "boolean"),
            ), 1):
                target.write(values, index)
                target.set_band_description(index, name)
                target.update_tags(index, feature=name, units=units)
            target.update_tags(source="potential_lake_depth.tif", aggregation="10 m to station grid")
        pyogrio.write_dataframe(grid, temporary_gpkg, layer="lake_features", driver="GPKG")
        manifest = {
            "grid": str(grid_path.resolve()),
            "source_lake_depth": str(lake_depth_path.resolve()),
            "source_sha256": _file_sha256(lake_depth_path),
            "features": FEATURE_FIELDS,
            "cell_m": cell_m,
            "width": width,
            "height": height,
            "crs": "EPSG:25833",
            "valid_cells": len(grid),
            "raster": str(raster_path.resolve()),
            "geopackage": str(gpkg_path.resolve()),
            "aggregation": "mean and max potential depth in each station-model cell; present if max depth > 0",
        }
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for pending, final in zip((temporary_raster, temporary_gpkg, temporary_manifest), outputs):
            pending.replace(final)
    finally:
        for pending in (temporary_raster, temporary_gpkg, temporary_manifest):
            pending.unlink(missing_ok=True)
    return dict(raster=str(raster_path), grid=str(gpkg_path), manifest=str(manifest_path), reused=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--lake-depth", type=Path, default=DEFAULT_LAKE_DEPTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    result = prepare_lake_features(args.grid, args.lake_depth, args.output_dir, args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
