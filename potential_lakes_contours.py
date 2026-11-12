"""Screen for potential lake basins using N50 height centerlines, not DTM elevations.

The N50 contour elevations (hoyde) are rasterized onto a metric grid and the
spaces between them are interpolated with GDAL/Rasterio fillnodata. The resulting
surface drains through cells outside the Norway polygon, which are approximated
as ocean (including neighboring land). This is a screening model: its vertical
detail cannot exceed the 10/20 m contour intervals, and interpolated terrain is
not a surveyed DEM.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd  # Load before rasterio to avoid PROJ/GDAL DLL conflicts.
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.fill import fillnodata
from rasterio.transform import Affine, from_origin
from scipy import ndimage
from shapely.geometry import shape

from potential_lakes import basin_stats, polygons, priority_flood, relabel, write_raster

ROOT = Path(__file__).resolve().parent
DEFAULT_N50 = ROOT / "data" / "Basisdata_0000_Norge_25833_N50Kartdata_FGDB.gdb"
DEFAULT_NORWAY = ROOT / "outputs" / "norway_polygon.gpkg"
DEFAULT_OUT = ROOT / "outputs" / "potential_lakes_contours"
HEIGHT_LAYER = "N50_Høyde_senterlinje"
HEIGHT_TYPES = ("Høydekurve", "Hjelpekurve", "Forsenkningskurve")
NODATA = -32767.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n50", type=Path, default=DEFAULT_N50,
                        help="N50 geodatabase containing elevation centerlines")
    parser.add_argument("--norway", type=Path, default=DEFAULT_NORWAY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cell-m", type=float, default=50.0,
                        help="Contour surface grid spacing in metres; larger cells run faster")
    parser.add_argument("--tile-cells", type=int, default=1000,
                        help="Core width of each processing tile in grid cells")
    parser.add_argument("--max-gap-m", type=float, default=15_000.0,
                        help="Maximum interpolation distance from contour lines, in metres")
    parser.add_argument("--buffer-m", type=float, default=50_000.0,
                        help="Processing margin outside Norway, in metres")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="Optional EPSG:25833 test extent; its edges are not drainage outlets")
    parser.add_argument("--min-depth", type=float, default=1.0,
                        help="Keep basins whose maximum interpolated depth is at least this (m)")
    parser.add_argument("--min-area", type=float, default=10_000.0,
                        help="Keep basins with at least this surface area (m²)")
    parser.add_argument("--no-polygons", action="store_true", help="Skip GeoPackage polygon output")
    args = parser.parse_args(argv)
    if args.cell_m <= 0 or args.tile_cells <= 0 or args.max_gap_m <= 0 or args.buffer_m < 0:
        parser.error("--cell-m, --tile-cells, and --max-gap-m must be positive; --buffer-m must be nonnegative")
    if args.min_depth < 0 or args.min_area < 0:
        parser.error("--min-depth and --min-area must be nonnegative")
    return args


def format_duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def report_tile_progress(done, total, started, completed_before=0):
    elapsed = time.perf_counter() - started
    rate = elapsed / max(done - completed_before, 1)
    remaining = rate * max(total - done, 0)
    eta = datetime.now() + timedelta(seconds=remaining)
    print(f"Contour interpolation: {done:,}/{total:,} tiles ({100 * done / total:.1f}%), "
          f"elapsed {format_duration(elapsed)}, ETA {format_duration(remaining)} "
          f"(about {eta:%H:%M})", flush=True)


def path_signature(path):
    path = Path(path).resolve()
    if path.is_file():
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.is_dir():
        entries = []
        for item in sorted(path.rglob("*")):
            if item.is_file():
                stat = item.stat()
                entries.append((str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns))
        return {"path": str(path), "files": entries}
    return {"path": str(path), "missing": True}


def save_checkpoint_manifest(path, metadata, completed):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"metadata": metadata, "completed_tiles": sorted(completed)},
                                    separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def read_norway(path):
    frame = pyogrio.read_dataframe(path)
    if frame.empty or frame.crs is None:
        raise ValueError(f"Could not read a projected Norway boundary from {path}")
    if frame.crs.to_epsg() != 25833:
        raise ValueError(f"Norway boundary must be EPSG:25833, got {frame.crs}")
    return frame


def make_grid(bounds, cell_m):
    xmin, ymin, xmax, ymax = bounds
    xmin = math.floor(xmin / cell_m) * cell_m
    ymin = math.floor(ymin / cell_m) * cell_m
    xmax = math.ceil(xmax / cell_m) * cell_m
    ymax = math.ceil(ymax / cell_m) * cell_m
    width = int(round((xmax - xmin) / cell_m))
    height = int(round((ymax - ymin) / cell_m))
    return (height, width), from_origin(xmin, ymax, cell_m, cell_m)


def assumed_sea_mask(norway_mask):
    """Approximate every cell outside Norway as an ocean drainage outlet."""
    return ~np.asarray(norway_mask, dtype=bool)


def interpolate_contour_tile(n50, bounds, shape_, transform, max_gap_cells):
    names = ",".join("'" + name + "'" for name in HEIGHT_TYPES)
    contours = pyogrio.read_dataframe(
        n50, layer=HEIGHT_LAYER, columns=["objtype", "hoyde"],
        where=f"objtype IN ({names})", bbox=tuple(bounds),
    )
    if contours.empty:
        return np.full(shape_, NODATA, dtype=np.float32)

    elevations = pd.to_numeric(contours["hoyde"], errors="coerce").to_numpy(dtype=np.float32)
    return interpolate_contour_geometries(
        contours.geometry.to_numpy(), elevations, shape_, transform, max_gap_cells,
    )


def interpolate_contour_geometries(geometries, elevations, shape_, transform, max_gap_cells):
    records = ((geom, float(elev)) for geom, elev in zip(geometries, elevations)
               if geom is not None and not geom.is_empty and np.isfinite(elev))
    known = rasterize(records, out_shape=shape_, transform=transform,
                      fill=NODATA, dtype="float32", all_touched=True)
    valid = (known != NODATA) & np.isfinite(known)
    if not valid.any():
        return known
    interpolated = fillnodata(known, mask=valid.astype(np.uint8),
                              max_search_distance=max_gap_cells, smoothing_iterations=0)
    interpolated[~np.isfinite(interpolated)] = NODATA
    return interpolated


def build_contour_surface(args, grid_shape, transform, bounds):
    height, width = grid_shape
    cell_m = args.cell_m
    tile_cells = args.tile_cells
    halo = int(math.ceil(args.max_gap_m / cell_m))
    row_starts = list(range(0, height, tile_cells))
    col_starts = list(range(0, width, tile_cells))
    total = len(row_starts) * len(col_starts)
    checkpoint_dir = args.out_dir / ".contour_surface_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    surface_path = checkpoint_dir / "surface.npy"
    manifest_path = checkpoint_dir / "manifest.json"
    metadata = {
        "version": 1,
        "n50": path_signature(args.n50),
        "norway": path_signature(args.norway),
        "bounds": list(bounds),
        "grid_shape": list(grid_shape),
        "transform": list(transform),
        "cell_m": cell_m,
        "tile_cells": tile_cells,
        "max_gap_m": args.max_gap_m,
        "height_layer": HEIGHT_LAYER,
        "height_types": list(HEIGHT_TYPES),
    }
    completed = set()
    surface = None
    if manifest_path.exists() and surface_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("metadata") == metadata:
                cached = np.load(surface_path, mmap_mode="r+")
                if cached.shape == tuple(grid_shape) and cached.dtype == np.float32:
                    surface = cached
                    completed = {int(tile) for tile in manifest.get("completed_tiles", [])
                                 if 1 <= int(tile) <= total}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            surface = None
            completed = set()
    if surface is None:
        surface = np.lib.format.open_memmap(surface_path, mode="w+", dtype=np.float32, shape=grid_shape)
        surface[:] = NODATA
        surface.flush()
        save_checkpoint_manifest(manifest_path, metadata, completed)
        print("Created contour interpolation checkpoint", flush=True)
    else:
        print(f"Resuming contour interpolation: {len(completed):,}/{total:,} tiles already complete", flush=True)
    started = time.perf_counter()
    done = len(completed)
    completed_before = done
    interval = max(1, total // 100)
    tile_index = 0

    for row0 in row_starts:
        row1 = min(row0 + tile_cells, height)
        r0 = max(0, row0 - halo)
        r1 = min(height, row1 + halo)
        for col0 in col_starts:
            col1 = min(col0 + tile_cells, width)
            c0 = max(0, col0 - halo)
            c1 = min(width, col1 + halo)
            tile_index += 1
            if tile_index in completed:
                continue
            tile_transform = transform * Affine.translation(c0, r0)
            tile_height, tile_width = r1 - r0, c1 - c0
            tile_xmin = tile_transform.c
            tile_ymax = tile_transform.f
            tile_bounds = (tile_xmin, tile_ymax - tile_height * cell_m,
                           tile_xmin + tile_width * cell_m, tile_ymax)
            tile = interpolate_contour_tile(
                args.n50, tile_bounds, (tile_height, tile_width), tile_transform, halo,
            )
            core = tile[row0 - r0:row1 - r0, col0 - c0:col1 - c0]
            surface[row0:row1, col0:col1] = core
            surface.flush()
            completed.add(tile_index)
            save_checkpoint_manifest(manifest_path, metadata, completed)
            done += 1
            if done - completed_before == 1 or done == total or (done - completed_before) % interval == 0:
                report_tile_progress(done, total, started, completed_before)
            del tile, core

    elapsed = time.perf_counter() - started
    covered = int(np.count_nonzero(surface != NODATA))
    print(f"Contour surface complete: {covered:,}/{surface.size:,} grid cells covered "
          f"({100 * covered / surface.size:.1f}%) in {format_duration(elapsed)}", flush=True)
    return surface


def monitor_flood(counter, expected, started, stop):
    last = -1
    while not stop.wait(10):
        processed = int(counter[0])
        if processed == last:
            continue
        last = processed
        elapsed = time.perf_counter() - started
        rate = processed / max(elapsed, 1.0)
        remaining = max(expected - processed, 0) / max(rate, 1e-9)
        eta = datetime.now() + timedelta(seconds=remaining)
        percent = min(processed / max(expected, 1) * 100, 100.0)
        print(f"Priority-Flood: {processed:,}/{expected:,} cells ({percent:.1f}%), "
              f"elapsed {format_duration(elapsed)}, ETA {format_duration(remaining)} "
              f"(about {eta:%H:%M}; estimate assumes all contour cells connect to sea)", flush=True)


def find_potential_lakes(surface, valid, sea, transform, min_depth, min_area, report):
    cell_area = abs(transform.a * transform.e)
    counter = np.zeros(1, dtype=np.int64)
    stop = threading.Event()
    started = time.perf_counter()
    expected = int(valid.sum())
    monitor = threading.Thread(target=monitor_flood, args=(counter, expected, started, stop), daemon=True)
    monitor.start()
    filled = np.where(valid, surface, NODATA).astype(np.float32)
    try:
        processed = priority_flood(filled, valid, sea, counter)
    finally:
        stop.set()
        monitor.join()
    print(f"Priority-Flood complete: {processed:,} cells in {format_duration(time.perf_counter() - started)}", flush=True)

    depth = filled
    depth -= surface
    depth[~valid | sea | (depth < 1e-3)] = 0.0
    if report is not None:
        depth[~report] = 0.0

    started = time.perf_counter()
    labels, count = ndimage.label(depth > 0, structure=np.ones((3, 3), dtype=bool))
    labels = labels.astype(np.int32, copy=False)
    print(f"Basin labeling: {count:,} components in {format_duration(time.perf_counter() - started)}", flush=True)
    cells, volume, deepest, deepest_index, level = basin_stats(labels, depth, surface, count)
    area = cells * cell_area
    keep = (deepest >= min_depth) & (area >= min_area)
    keep[0] = False
    mapping = np.zeros(count + 1, np.int32)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    relabel(labels, depth, mapping)
    rows, cols = np.divmod(deepest_index[keep], surface.shape[1])
    x, y = rasterio.transform.xy(transform, rows, cols)
    table = pd.DataFrame({
        "lake_id": np.arange(1, int(keep.sum()) + 1),
        "area_m2": area[keep],
        "volume_m3": volume[keep] * cell_area,
        "max_depth_m": deepest[keep].astype(float),
        "mean_depth_m": volume[keep] / np.maximum(cells[keep], 1),
        "spill_level_m": level[keep],
        "deepest_x": np.asarray(x, dtype=float),
        "deepest_y": np.asarray(y, dtype=float),
    })
    return depth, labels, table.sort_values("volume_m3", ascending=False).reset_index(drop=True)


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    norway = read_norway(args.norway)
    if args.bbox is None:
        xmin, ymin, xmax, ymax = norway.total_bounds
        padding = args.buffer_m
        bounds = (xmin - padding, ymin - padding, xmax + padding, ymax + padding)
    else:
        bounds = tuple(args.bbox)
    grid_shape, transform = make_grid(bounds, args.cell_m)
    print(f"Contour-only terrain: N50 {HEIGHT_LAYER}, field hoyde; no DTM elevations", flush=True)
    print(f"Grid: {grid_shape[1]:,} x {grid_shape[0]:,} cells at {args.cell_m:g} m; "
          f"extent {bounds}; tile core {args.tile_cells:,} cells", flush=True)

    report = rasterize(
        ((geom, 1) for geom in norway.geometry if geom is not None),
        out_shape=grid_shape, transform=transform, fill=0, dtype="uint8",
    ).astype(bool)
    sea = assumed_sea_mask(report)
    print(f"Assumed ocean outlets outside Norway: {sea.sum():,} cells "
          "(neighboring land is intentionally treated as ocean)", flush=True)
    surface = build_contour_surface(args, grid_shape, transform, bounds)
    valid = (surface != NODATA) & np.isfinite(surface)
    surface[sea] = 0.0
    valid |= sea
    uncovered_norway = report & ~valid
    if uncovered_norway.any():
        print(f"WARNING: contour interpolation leaves {uncovered_norway.sum():,} Norway cells "
              f"uncovered ({100 * uncovered_norway.sum() / max(report.sum(), 1):.2f}%); these are not treated as outlets",
              flush=True)
    else:
        print("Contour interpolation covers every Norway grid cell", flush=True)

    depth, labels, table = find_potential_lakes(
        surface, valid, sea, transform, args.min_depth, args.min_area, report,
    )
    depth[~valid] = -9999.0
    surface[~valid] = -9999.0
    output_started = time.perf_counter()
    write_raster(args.out_dir / "contour_surface.tif", surface, transform, "EPSG:25833", -9999.0)
    print(f"Wrote contour_surface.tif in {format_duration(time.perf_counter() - output_started)}", flush=True)
    output_started = time.perf_counter()
    write_raster(args.out_dir / "potential_lake_depth.tif", depth, transform, "EPSG:25833", -9999.0)
    print(f"Wrote potential_lake_depth.tif in {format_duration(time.perf_counter() - output_started)}", flush=True)
    output_started = time.perf_counter()
    write_raster(args.out_dir / "potential_lake_id.tif", labels, transform, "EPSG:25833", 0)
    print(f"Wrote potential_lake_id.tif in {format_duration(time.perf_counter() - output_started)}", flush=True)
    table.to_csv(args.out_dir / "potential_lakes.csv", index=False)
    if not args.no_polygons and len(table):
        print("Polygonizing basin raster (this stage can take several minutes)...", flush=True)
        polygons(labels, transform, "EPSG:25833", table).to_file(
            args.out_dir / "potential_lakes.gpkg", layer="potential_lakes", driver="GPKG",
        )
    summary = {
        "method": "N50_Høyde_senterlinje contour interpolation; outside Norway assumed ocean; no DTM elevations",
        "height_field": "hoyde",
        "cell_m": args.cell_m,
        "contour_max_interpolation_distance_m": args.max_gap_m,
        "contour_surface_coverage_fraction": float(np.count_nonzero(valid & report) / max(np.count_nonzero(report), 1)),
        "lakes": len(table),
        "total_area_km2": float(table.area_m2.sum() / 1e6),
        "total_volume_km3": float(table.volume_m3.sum() / 1e9),
        "min_depth_m": args.min_depth,
        "min_area_m2": args.min_area,
        "bbox": bounds,
        "seconds": round(time.perf_counter() - run_started, 1),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if len(table):
        print(table.head(10).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
