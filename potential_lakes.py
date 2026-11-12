"""Screen DTM terrain for depressions that could hold additional water.

The DTM is the elevation surface. Cells outside the Norway polygon are treated
as 0 m ocean outlets; lakes and rivers are not masked. DTM NoData inside Norway
is not an outlet. Depth is filled elevation minus DTM elevation, so results are
screening estimates whose quality is limited by the chosen DTM resolution.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd  # noqa: F401  Load before rasterio to avoid PROJ/GDAL DLL conflicts.
import numba
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.merge import merge
from rasterio.warp import transform as transform_coordinates
from scipy import ndimage
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parent
DEFAULT_DTM = ROOT.parent / "dtm50" / "data"
DEFAULT_OUT = ROOT / "outputs" / "potential_lakes_dtm"
DEFAULT_NORWAY = ROOT / "outputs" / "norway_polygon.gpkg"
NODATA = -32767.0


def configure_proj_data():
    proj_data = Path(rasterio.__file__).resolve().parent / "proj_data"
    if not (proj_data / "proj.db").is_file():
        raise FileNotFoundError(f"Rasterio's bundled PROJ database is missing: {proj_data / 'proj.db'}")
    os.environ["PROJ_DATA"] = str(proj_data)
    os.environ["PROJ_LIB"] = str(proj_data)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtm-dir", type=Path, default=DEFAULT_DTM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--norway", type=Path, default=DEFAULT_NORWAY,
                        help="Land boundary; all cells outside it are assumed to be ocean")
    parser.add_argument("--buffer-m", type=float, default=50_000.0,
                        help="DTM margin outside Norway, in metres, for the assumed ocean outlet")
    parser.add_argument("--center-xy", type=float, nargs=2, metavar=("X", "Y"),
                        help="Local test center in --center-crs; report pools inside --radius-m")
    parser.add_argument("--center-crs", default="EPSG:25832",
                        help="CRS for --center-xy (default: EPSG:25832 / UTM zone 32N)")
    parser.add_argument("--radius-m", type=float,
                        help="Report radius around --center-xy, in metres")
    parser.add_argument("--context-m", type=float, default=20_000.0,
                        help="Extra DTM context around --radius-m; context-window edges are open boundaries")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="EPSG:25833 test area")
    parser.add_argument("--min-depth", type=float, default=1.0,
                        help="Keep basins whose maximum depth is at least this (m)")
    parser.add_argument("--min-area", type=float, default=10_000,
                        help="Keep basins with at least this surface area (m²)")
    parser.add_argument("--merge-sill-m", type=float, default=20.0,
                        help="Group basins separated by a one-cell saddle below the lower spill level plus this tolerance; 0 disables merging")
    parser.add_argument("--no-polygons", action="store_true", help="Skip the GeoPackage polygons")
    args = parser.parse_args(argv)
    if (args.min_depth < 0 or args.min_area < 0 or args.buffer_m < 0 or args.context_m < 0
            or args.merge_sill_m < 0):
        parser.error("depth, area, buffer, context, and merge-sill values must be nonnegative")
    if (args.center_xy is None) != (args.radius_m is None):
        parser.error("--center-xy and --radius-m must be supplied together")
    if args.radius_m is not None and args.radius_m <= 0:
        parser.error("--radius-m must be positive")
    if args.center_xy is not None and args.bbox is not None:
        parser.error("Use either --center-xy/--radius-m or --bbox, not both")
    return args


def load_dtm(dtm_dir, bbox=None):
    configure_proj_data()
    tiles = sorted(glob.glob(str(Path(dtm_dir) / "**" / "*.tif"), recursive=True))
    if not tiles:
        raise FileNotFoundError(f"No DTM tiles found in {dtm_dir}")
    all_sources = [rasterio.open(path) for path in tiles]
    try:
        sources = all_sources
        if bbox is not None:
            sources = [s for s in sources if not (s.bounds.right <= bbox[0] or s.bounds.left >= bbox[2]
                                                  or s.bounds.top <= bbox[1] or s.bounds.bottom >= bbox[3])]
            if not sources:
                raise ValueError("The bounding box does not overlap the DTM")
        crs = sources[0].crs
        if crs is None or crs.to_epsg() != 25833:
            raise ValueError(f"DTM tiles must use EPSG:25833; got {crs}")
        array, transform = merge(sources, bounds=bbox, nodata=NODATA, dtype="float32", mem_limit=4096)
    finally:
        for source in all_sources:
            source.close()
    dem = array[0]
    valid = np.isfinite(dem) & (dem > NODATA + 1)
    return dem, valid, transform, crs


def read_norway(path):
    norway = pyogrio.read_dataframe(path)
    if norway.empty or norway.crs is None:
        raise ValueError(f"Could not read Norway polygon from {path}")
    if norway.crs.to_epsg() != 25833:
        norway = norway.to_crs("EPSG:25833")
    return norway


def norway_mask(norway, shape_, transform):
    return rasterize(((geometry, 1) for geometry in norway.geometry if geometry is not None),
                     out_shape=shape_, transform=transform, fill=0, dtype="uint8").astype(bool)


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
              f"elapsed {int(elapsed)}s, ETA {int(remaining)}s (about {eta:%H:%M})", flush=True)


def prepare_ocean_outlets(dem, valid, norway):
    """Set outside-Norway terrain to ocean level; use its boundary ring as outlets."""
    outside = ~norway
    outlet = outside & ndimage.binary_dilation(norway, structure=np.ones((3, 3), dtype=bool))
    dem[outside] = 0.0
    valid &= norway
    valid[outlet] = True
    return outlet


def window_boundary_mask(shape_):
    boundary = np.zeros(shape_, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    return boundary


def transform_center(center_xy, source_crs):
    configure_proj_data()
    x, y = transform_coordinates(source_crs, "EPSG:25833", [center_xy[0]], [center_xy[1]])
    return float(x[0]), float(y[0])


@numba.njit(cache=True, nogil=True)
def priority_flood(z, valid, outlet, progress=None):
    """Barnes et al. (2014) improved Priority-Flood; raises z in place to its spill level."""
    rows, cols = z.shape
    closed = np.zeros((rows, cols), np.uint8)
    capacity = 1 << 22
    heap_z = np.empty(capacity, np.float32)
    heap_i = np.empty(capacity, np.int64)
    size = 0
    pit_capacity = 1 << 22
    pit = np.empty(pit_capacity, np.int64)
    pits = 0

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                closed[r, c] = 1
                continue
            seed = outlet[r, c]
            if seed:
                closed[r, c] = 1
                if size == capacity:
                    capacity *= 2
                    grown_z = np.empty(capacity, np.float32)
                    grown_i = np.empty(capacity, np.int64)
                    grown_z[:size] = heap_z[:size]
                    grown_i[:size] = heap_i[:size]
                    heap_z, heap_i = grown_z, grown_i
                value, index, k = z[r, c], r * cols + c, size
                size += 1
                while k > 0:
                    parent = (k - 1) >> 1
                    if heap_z[parent] <= value:
                        break
                    heap_z[k], heap_i[k] = heap_z[parent], heap_i[parent]
                    k = parent
                heap_z[k], heap_i[k] = value, index

    processed = 0
    while size > 0 or pits > 0:
        if pits > 0:
            pits -= 1
            current = pit[pits]
        else:
            current = heap_i[0]
            size -= 1
            if size > 0:
                last_z, last_i, k = heap_z[size], heap_i[size], 0
                while True:
                    child = 2 * k + 1
                    if child >= size:
                        break
                    if child + 1 < size and heap_z[child + 1] < heap_z[child]:
                        child += 1
                    if heap_z[child] >= last_z:
                        break
                    heap_z[k], heap_i[k] = heap_z[child], heap_i[child]
                    k = child
                heap_z[k], heap_i[k] = last_z, last_i
        r, c = current // cols, current % cols
        level = z[r, c]
        processed += 1
        if progress is not None and processed % 1_000_000 == 0:
            progress[0] = processed
        elif progress is None and processed % 50_000_000 == 0:
            print("  flooded cells:", processed)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if (dr == 0 and dc == 0) or nr < 0 or nc < 0 or nr >= rows or nc >= cols or closed[nr, nc]:
                    continue
                closed[nr, nc] = 1
                if z[nr, nc] <= level:
                    z[nr, nc] = level
                    if pits == pit_capacity:
                        pit_capacity *= 2
                        grown = np.empty(pit_capacity, np.int64)
                        grown[:pits] = pit[:pits]
                        pit = grown
                    pit[pits] = nr * cols + nc
                    pits += 1
                else:
                    if size == capacity:
                        capacity *= 2
                        grown_z = np.empty(capacity, np.float32)
                        grown_i = np.empty(capacity, np.int64)
                        grown_z[:size] = heap_z[:size]
                        grown_i[:size] = heap_i[:size]
                        heap_z, heap_i = grown_z, grown_i
                    value, index, k = z[nr, nc], nr * cols + nc, size
                    size += 1
                    while k > 0:
                        parent = (k - 1) >> 1
                        if heap_z[parent] <= value:
                            break
                        heap_z[k], heap_i[k] = heap_z[parent], heap_i[parent]
                        k = parent
                    heap_z[k], heap_i[k] = value, index
    if progress is not None:
        progress[0] = processed
    return processed


@numba.njit(cache=True)
def basin_stats(labels, depth, dem, count):
    cells = np.zeros(count + 1, np.int64)
    volume = np.zeros(count + 1, np.float64)
    deepest = np.zeros(count + 1, np.float32)
    deepest_index = np.full(count + 1, -1, np.int64)
    level = np.full(count + 1, -np.inf, np.float64)
    rows, cols = labels.shape
    for r in range(rows):
        for c in range(cols):
            label = labels[r, c]
            if label == 0:
                continue
            d = depth[r, c]
            cells[label] += 1
            volume[label] += d
            if d > deepest[label]:
                deepest[label] = d
                deepest_index[label] = r * cols + c
            surface = dem[r, c] + d
            if surface > level[label]:
                level[label] = surface
    return cells, volume, deepest, deepest_index, level


@numba.njit(cache=True)
def relabel(labels, depth, mapping):
    rows, cols = labels.shape
    for r in range(rows):
        for c in range(cols):
            new = mapping[labels[r, c]]
            labels[r, c] = new
            if new == 0:
                depth[r, c] = 0.0


@numba.njit(cache=True)
def _find_union_root(parent, value):
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


@numba.njit(cache=True, nogil=True)
def _saddle_union_parents(labels, dem, report, spill_levels, sill_tolerance_m, row_min, row_max,
                          col_min, col_max):
    count = len(spill_levels) - 1
    parent = np.arange(count + 1, dtype=np.int32)
    neighbors = np.empty(8, dtype=np.int32)
    rows, cols = labels.shape
    scanned = 0
    for r in range(row_min, row_max):
        for c in range(col_min, col_max):
            if not report[r, c] or labels[r, c] != 0:
                continue
            scanned += 1
            if scanned % 50_000_000 == 0:
                print("  checked dry saddle cells:", scanned)
            neighbor_count = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if (dr == 0 and dc == 0) or nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                        continue
                    basin_id = labels[nr, nc]
                    if basin_id <= 0:
                        continue
                    found = False
                    for i in range(neighbor_count):
                        if neighbors[i] == basin_id:
                            found = True
                            break
                    if not found:
                        neighbors[neighbor_count] = basin_id
                        neighbor_count += 1
            if neighbor_count < 2:
                continue
            sill = dem[r, c]
            for i in range(neighbor_count):
                first = neighbors[i]
                for j in range(i + 1, neighbor_count):
                    second = neighbors[j]
                    lower_spill = min(spill_levels[first], spill_levels[second])
                    if sill >= lower_spill + sill_tolerance_m:
                        continue
                    first_root = _find_union_root(parent, first)
                    second_root = _find_union_root(parent, second)
                    if first_root != second_root:
                        parent[second_root] = first_root
    return parent


def group_basins_by_sill(labels, dem, report, table, sill_tolerance_m):
    if table.empty:
        return np.zeros(1, dtype=np.int32), table.copy(), pd.DataFrame(), 0
    basin_count = int(table.lake_id.max())
    spill_levels = np.full(basin_count + 1, np.inf, dtype=np.float32)
    spill_levels[table.lake_id.to_numpy(dtype=np.int64)] = table.spill_level_m.to_numpy(dtype=np.float32)
    if sill_tolerance_m > 0:
        rows = np.flatnonzero(report.any(axis=1))
        cols = np.flatnonzero(report.any(axis=0))
        parent = _saddle_union_parents(
            labels, dem, report, spill_levels, sill_tolerance_m,
            int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1),
        )
    else:
        parent = np.arange(basin_count + 1, dtype=np.int32)
    roots = np.empty(basin_count, dtype=np.int32)
    for index in range(1, basin_count + 1):
        roots[index - 1] = _find_union_root.py_func(parent, index)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    mapping = np.zeros(basin_count + 1, dtype=np.int32)
    mapping[1:] = inverse + 1

    grouped = table.copy()
    grouped["group_id"] = mapping[grouped.lake_id.to_numpy(dtype=np.int64)]
    groups = grouped.groupby("group_id", as_index=False).agg(
        member_basin_count=("lake_id", "size"),
        group_area_m2=("area_m2", "sum"),
        group_volume_sum_m3=("volume_m3", "sum"),
        group_max_depth_m=("max_depth_m", "max"),
        group_max_spill_level_m=("spill_level_m", "max"),
    )
    group_lookup = groups.set_index("group_id")
    for column in ("member_basin_count", "group_area_m2", "group_volume_sum_m3",
                   "group_max_depth_m", "group_max_spill_level_m"):
        grouped[column] = grouped.group_id.map(group_lookup[column])
    merged_basins = basin_count - len(unique_roots)
    grouped["merge_sill_tolerance_m"] = sill_tolerance_m
    return mapping, grouped, groups, merged_basins


def find_potential_lakes(dem, valid, sea, transform, min_depth=1.0, min_area=10_000.0, report=None):
    cell_area = abs(transform.a * transform.e)
    filled = np.where(valid, dem, NODATA).astype(np.float32)
    started = time.perf_counter()
    counter = np.zeros(1, dtype=np.int64)
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_flood,
                               args=(counter, int(valid.sum()), started, stop), daemon=True)
    monitor.start()
    try:
        processed = priority_flood(filled, valid, sea, counter)
    finally:
        stop.set()
        monitor.join()
    print(f"Priority-Flood complete: {processed:,} cells in {time.perf_counter() - started:.1f} s")
    depth = filled
    depth -= dem
    depth[~valid | sea | (depth < 1e-3)] = 0.0
    if report is not None:
        depth[~report] = 0.0
    labels, count = ndimage.label(depth > 0, structure=np.ones((3, 3), dtype=bool))
    labels = labels.astype(np.int32, copy=False)
    cells, volume, deepest, deepest_index, level = basin_stats(labels, depth, dem, count)
    area = cells * cell_area
    keep = (deepest >= min_depth) & (area >= min_area)
    keep[0] = False
    mapping = np.zeros(count + 1, np.int32)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    relabel(labels, depth, mapping)
    rows, cols = np.divmod(deepest_index[keep], dem.shape[1])
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


def write_raster(path, array, transform, crs, nodata):
    profile = dict(driver="GTiff", height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype,
                   crs=crs, transform=transform, nodata=nodata, tiled=True, blockxsize=512, blockysize=512,
                   compress="deflate", predictor=2 if array.dtype.kind == "i" else 3, BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **profile) as target:
        target.write(array, 1)


def polygons(labels, transform, crs, table):
    records = [(shape(geometry), int(value)) for geometry, value in
               shapes(labels, mask=labels > 0, transform=transform, connectivity=8)]
    frame = gpd.GeoDataFrame({"lake_id": [value for _, value in records]},
                             geometry=[geometry for geometry, _ in records], crs=crs)
    frame = frame.dissolve(by="lake_id", as_index=False)
    return frame.merge(table, on="lake_id")


def main(argv=None):
    args = parse_args(argv)
    configure_proj_data()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    norway = read_norway(args.norway)
    center_25833 = None
    report_geometry = None
    local_test = args.center_xy is not None or args.bbox is not None
    if args.center_xy is not None:
        center_25833 = transform_center(args.center_xy, args.center_crs)
        report_geometry = Point(*center_25833).buffer(args.radius_m, quad_segs=64)
        extent_radius = args.radius_m + args.context_m
        bounds = (center_25833[0] - extent_radius, center_25833[1] - extent_radius,
                  center_25833[0] + extent_radius, center_25833[1] + extent_radius)
    elif args.bbox is None:
        xmin, ymin, xmax, ymax = norway.total_bounds
        bounds = (xmin - args.buffer_m, ymin - args.buffer_m,
                  xmax + args.buffer_m, ymax + args.buffer_m)
    else:
        bounds = tuple(args.bbox)
    dem, valid, transform, crs = load_dtm(args.dtm_dir, bounds)
    print(f"DTM {dem.shape[1]} x {dem.shape[0]} cells ({valid.sum():,} valid), "
          f"{abs(transform.a):g} m cells")
    norway_land = norway_mask(norway, dem.shape, transform)
    if not norway_land.any():
        raise ValueError("The DTM extent does not overlap the Norway polygon")
    sea = prepare_ocean_outlets(dem, valid, norway_land)
    if local_test:
        window_edge = window_boundary_mask(dem.shape) & valid
        sea |= window_edge
        print(f"Local open-boundary outlet cells: {window_edge.sum():,}", flush=True)
    report = norway_land
    if report_geometry is not None:
        radius_mask = rasterize(((report_geometry, 1),), out_shape=dem.shape,
                                transform=transform, fill=0, dtype="uint8").astype(bool)
        report = norway_land & radius_mask
        if not report.any():
            raise ValueError("The requested radius does not overlap Norway in the selected DTM extent")
    if not sea.any():
        raise ValueError("The DTM extent needs at least one raster cell outside Norway for the ocean outlet")
    print(f"Report cells: {report.sum():,}; outside-Norway ocean outlet ring: {sea.sum():,} cells")
    depth, labels, table = find_potential_lakes(dem, valid, sea, transform, args.min_depth,
                                                args.min_area, report)
    group_mapping, table, groups, merged_basins = group_basins_by_sill(
        labels, dem, report, table, args.merge_sill_m,
    )
    if report_geometry is not None and len(table):
        report_edge = report & ~ndimage.binary_erosion(report, structure=np.ones((3, 3), dtype=bool))
        edge_ids = np.unique(labels[report_edge])
        edge_ids = edge_ids[edge_ids > 0]
        table["touches_report_boundary"] = table.lake_id.isin(edge_ids)
    if report_geometry is not None:
        dem[~report] = -9999.0
        write_raster(args.out_dir / "dtm_elevation.tif", dem, transform, crs, -9999.0)
    del dem
    depth[~report | ~valid] = -9999.0
    write_raster(args.out_dir / "potential_lake_depth.tif", depth, transform, crs, -9999.0)
    write_raster(args.out_dir / "potential_lake_id.tif", labels, transform, crs, 0)
    table.to_csv(args.out_dir / "potential_lakes.csv", index=False)
    if not args.no_polygons and len(table):
        pyogrio.write_dataframe(polygons(labels, transform, crs, table),
                                args.out_dir / "potential_lakes.gpkg",
                                layer="potential_lakes", driver="GPKG")
    groups.to_csv(args.out_dir / "potential_lake_groups.csv", index=False)
    if args.merge_sill_m > 0:
        relabel(labels, depth, group_mapping)
        write_raster(args.out_dir / "potential_lake_group_id.tif", labels, transform, crs, 0)
        if not args.no_polygons and len(groups):
            group_table = groups.rename(columns={"group_id": "lake_id"})
            group_frame = polygons(labels, transform, crs, group_table).rename(columns={"lake_id": "group_id"})
            pyogrio.write_dataframe(group_frame, args.out_dir / "potential_lake_groups.gpkg",
                                    layer="groups", driver="GPKG")
    summary = dict(lakes=len(table), total_area_km2=float(table.area_m2.sum() / 1e6),
                   total_volume_km3=float(table.volume_m3.sum() / 1e9),
                   min_depth_m=args.min_depth, min_area_m2=args.min_area, bbox=bounds,
                   dtm_dir=str(args.dtm_dir), resolution_m=abs(transform.a),
                   outlet_assumption="cells outside Norway polygon are 0 m ocean",
                   report_radius_m=args.radius_m, center_input=args.center_xy,
                   center_input_crs=args.center_crs if args.center_xy is not None else None,
                   center_epsg25833=center_25833,
                   context_m=args.context_m if local_test and args.center_xy is not None else None,
                   local_boundary_outlet=local_test, merge_sill_m=args.merge_sill_m,
                   merged_basins=merged_basins, basin_groups=len(groups),
                   seconds=round(time.perf_counter() - started, 1))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if len(table):
        print(table.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
