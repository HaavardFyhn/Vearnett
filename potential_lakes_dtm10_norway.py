"""Disk-backed nationwide DTM10 potential-lake analysis for Linux servers.

This computes one global Priority-Flood with bounded-memory local tile queues
and iterative spill-level exchange across tile boundaries. Large working arrays
are memory-mapped in --work-dir; polygons are intentionally not generated at
national 10 m scale. Outputs are ArcGIS-ready rasters and CSV tables.
"""
from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numba
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from scipy import ndimage
from shapely.geometry import box
from shapely.strtree import STRtree

import potential_lakes as base


ROOT = Path(__file__).resolve().parent
DEFAULT_DTM = ROOT / "data" / "DTM10"
DEFAULT_OUT = ROOT / "outputs" / "potential_lakes_norway_DTM10_sill20"
DEFAULT_NORWAY = ROOT / "outputs" / "norway_polygon.gpkg"
BLOCK_SIZE = 2048
FLOOD_CHUNK_SIZE = 512
INVALID = 1
OUTLET = 2
LAND_CLOSED = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtm-dir", type=Path, default=DEFAULT_DTM)
    parser.add_argument("--mosaic", type=Path,
                        help="Reuse an existing single-band 10 m UTM33 mosaic instead of remosaicking tiles")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work-dir", type=Path,
                        help="Scratch directory on a large local disk; must be empty or absent")
    parser.add_argument("--norway", type=Path, default=DEFAULT_NORWAY)
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="Optional EPSG:25833 crop for testing; crop edges become open outlets")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help="Window size for mosaic, mask, and output I/O")
    parser.add_argument("--flood-chunk-size", type=int, default=FLOOD_CHUNK_SIZE,
                        help="Maximum local Priority-Flood tile dimension; smaller values use less RAM")
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--min-area", type=float, default=10_000.0)
    parser.add_argument("--merge-sill-m", type=float, default=20.0)
    parser.add_argument("--keep-work", action="store_true",
                        help="Keep large memory-mapped scratch arrays after a successful run")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    args = parser.parse_args(argv)
    if args.min_depth < 0 or args.min_area < 0 or args.merge_sill_m < 0:
        parser.error("depth, area, and sill thresholds must be nonnegative")
    if args.block_size < 128:
        parser.error("--block-size must be at least 128")
    if not 32 <= args.flood_chunk_size <= 1024:
        parser.error("--flood-chunk-size must be between 32 and 1024")
    if args.bounds is not None:
        xmin, ymin, xmax, ymax = args.bounds
        if xmin >= xmax or ymin >= ymax:
            parser.error("--bounds must have increasing minimum and maximum coordinates")
        if args.mosaic is not None:
            parser.error("--mosaic cannot be combined with --bounds")
    if args.work_dir is None:
        args.work_dir = args.out_dir.with_name(args.out_dir.name + "_work")
    return args


def windows(width, height, block_size):
    for row in range(0, height, block_size):
        for col in range(0, width, block_size):
            yield Window(col, row, min(block_size, width - col), min(block_size, height - row))


def is_supported_dtm_crs(crs):
    if crs is None:
        return False
    parameters = crs.to_dict()
    return (
        parameters.get("proj") == "utm"
        and parameters.get("zone") == 33
        and parameters.get("ellps") == "GRS80"
        and parameters.get("units") == "m"
        and not parameters.get("south", False)
    )


def source_tiles(dtm_dir, requested_bounds):
    paths = sorted(Path(dtm_dir).rglob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No DTM tiles found in {dtm_dir}")
    selected = []
    left = bottom = float("inf")
    right = top = float("-inf")
    for path in paths:
        with rasterio.open(path) as source:
            if not is_supported_dtm_crs(source.crs):
                raise ValueError(f"DTM tiles must use EPSG:25833 or its axis-order variant; "
                                 f"got {source.crs} for {path}")
            if abs(source.res[0] - 10.0) > 1e-6 or abs(source.res[1] - 10.0) > 1e-6:
                raise ValueError(f"Expected 10 m tiles; got {source.res} for {path}")
            tile_bounds = source.bounds
            left = min(left, tile_bounds.left)
            bottom = min(bottom, tile_bounds.bottom)
            right = max(right, tile_bounds.right)
            top = max(top, tile_bounds.top)
            if requested_bounds is None or not (
                    tile_bounds.right <= requested_bounds[0] or tile_bounds.left >= requested_bounds[2]
                    or tile_bounds.top <= requested_bounds[1] or tile_bounds.bottom >= requested_bounds[3]):
                selected.append(path)
    union = (left, bottom, right, top)
    return selected, union


def prepare_directories(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        "potential_lake_depth.tif", "potential_lake_id.tif", "potential_lake_group_id.tif",
        "potential_lakes.csv", "potential_lake_groups.csv", "summary.json",
    ]
    existing = [args.out_dir / name for name in outputs if (args.out_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist; use --overwrite to replace them: {existing[0]}")
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise FileExistsError(f"Work directory is not empty: {args.work_dir}; choose a fresh --work-dir")
    args.work_dir.mkdir(parents=True, exist_ok=True)


def check_scratch_space(args, shape_):
    cells = shape_[0] * shape_[1]
    required = cells * 24
    free = shutil.disk_usage(args.work_dir).free
    if free < required:
        raise OSError(f"Need about {required / 1024**3:.1f} GiB free scratch space; "
                      f"only {free / 1024**3:.1f} GiB is available")


def polygon_index(norway_path):
    norway = base.read_norway(norway_path)
    geometries = []
    for geometry in norway.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "Polygon":
            geometries.append(geometry)
        elif hasattr(geometry, "geoms"):
            geometries.extend(part for part in geometry.geoms if part.geom_type == "Polygon")
    if not geometries:
        raise ValueError(f"No polygon geometry found in {norway_path}")
    return norway.total_bounds, geometries, STRtree(geometries)


def rasterize_window(tree, geometries, source, window, halo=False):
    if halo:
        col0 = max(int(window.col_off) - 1, 0)
        row0 = max(int(window.row_off) - 1, 0)
        col1 = min(int(window.col_off + window.width) + 1, source.width)
        row1 = min(int(window.row_off + window.height) + 1, source.height)
        expanded = Window(col0, row0, col1 - col0, row1 - row0)
    else:
        expanded = window
    tile_box = box(*window_bounds(expanded, source.transform))
    indices = tree.query(tile_box)
    shapes = [(geometries[int(index)], 1) for index in indices]
    land = rasterize(shapes, out_shape=(int(expanded.height), int(expanded.width)),
                     transform=window_transform(expanded, source.transform), fill=0,
                     dtype="uint8").astype(bool)
    if not halo:
        return land
    near_land = ndimage.binary_dilation(land, structure=np.ones((3, 3), dtype=bool))
    row_offset = int(window.row_off) - row0
    col_offset = int(window.col_off) - col0
    height, width = int(window.height), int(window.width)
    land = land[row_offset:row_offset + height, col_offset:col_offset + width]
    near_land = near_land[row_offset:row_offset + height, col_offset:col_offset + width]
    return land, near_land


def create_mosaic(paths, bounds, mosaic_path):
    print(f"Mosaicking {len(paths)} DTM10 tiles to {mosaic_path}", flush=True)
    merge(paths, bounds=bounds, res=10.0, nodata=base.NODATA, dtype="float32",
          method="first", mem_limit=512, dst_path=mosaic_path,
          dst_kwds={"compress": "deflate", "predictor": 3, "tiled": True,
                    "blockxsize": 512, "blockysize": 512, "BIGTIFF": "YES"})


def inspect_mosaic(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DTM10 mosaic not found: {path}")
    with rasterio.open(path) as source:
        if not is_supported_dtm_crs(source.crs) or source.count != 1:
            raise ValueError(f"Expected a single-band UTM33 DTM10 mosaic; got {source.crs}")
        if abs(source.res[0] - 10.0) > 1e-6 or abs(source.res[1] - 10.0) > 1e-6:
            raise ValueError(f"Expected a 10 m mosaic; got {source.res}")
        return path, tuple(source.bounds), (source.height, source.width)


def initialize_arrays(mosaic_path, work_dir, tree, geometries, local_boundary, block_size):
    with rasterio.open(mosaic_path) as source:
        shape_ = (source.height, source.width)
        transform = source.transform
        if not is_supported_dtm_crs(source.crs):
            raise ValueError(f"Expected DTM10 in ETRS89 / UTM zone 33N; got {source.crs}")
        crs = rasterio.crs.CRS.from_epsg(25833)
        elevation = np.memmap(work_dir / "elevation.f32", mode="w+", dtype=np.float32, shape=shape_)
        filled = np.memmap(work_dir / "filled.f32", mode="w+", dtype=np.float32, shape=shape_)
        state = np.memmap(work_dir / "state.u8", mode="w+", dtype=np.uint8, shape=shape_)
        expected = 0
        for window in windows(source.width, source.height, block_size):
            dem = source.read(1, window=window, out_dtype="float32")
            land, near_land = rasterize_window(tree, geometries, source, window, halo=True)
            outlet = ~land & near_land
            valid_land = land & np.isfinite(dem) & (dem > base.NODATA + 1)
            dem[~land] = 0.0
            dem[outlet] = 0.0
            row0, col0 = int(window.row_off), int(window.col_off)
            height, width = int(window.height), int(window.width)
            state_block = state[row0:row0 + height, col0:col0 + width]
            state_block.fill(INVALID)
            state_block[valid_land] = 0
            state_block[outlet] = OUTLET
            elevation[row0:row0 + height, col0:col0 + width] = dem
            filled_block = filled[row0:row0 + height, col0:col0 + width]
            filled_block.fill(np.inf)
            expected += int(valid_land.sum()) + int(outlet.sum())
        if local_boundary:
            boundary = np.zeros(shape_, dtype=bool)
            boundary[0, :] = True
            boundary[-1, :] = True
            boundary[:, 0] = True
            boundary[:, -1] = True
            open_land = boundary & (state == 0)
            state[open_land] = OUTLET
        elevation.flush()
        filled.flush()
        state.flush()
    print(f"DTM {shape_[1]:,} x {shape_[0]:,} cells ({expected:,} flood cells), 10 m cells",
          flush=True)
    return elevation, filled, state, shape_, transform, crs, expected


@numba.njit(cache=True, nogil=True)
def priority_flood_disk(z, state, progress=None):
    """Global Priority-Flood using state codes instead of a second closed mask."""
    rows, cols = z.shape
    capacity = 1 << 22
    heap_z = np.empty(capacity, np.float32)
    heap_i = np.empty(capacity, np.int64)
    size = 0
    pit_capacity = 1 << 22
    pit = np.empty(pit_capacity, np.int64)
    pits = 0
    scanned = 0

    for r in range(rows):
        for c in range(cols):
            scanned += 1
            if progress is not None and scanned % 100_000_000 == 0:
                progress[0] = -scanned
            if state[r, c] != OUTLET:
                continue
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

    if progress is not None:
        progress[0] = 0
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
        row, col = current // cols, current % cols
        level = z[row, col]
        processed += 1
        if progress is not None and processed % 1_000_000 == 0:
            progress[0] = processed
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if (dr == 0 and dc == 0) or nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                    continue
                if state[nr, nc] != 0:
                    continue
                state[nr, nc] = LAND_CLOSED
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
                        heap_z[k], heap_i[k] = heap_z[parent], heap_i[k]
                        k = parent
                    heap_z[k], heap_i[k] = value, index
    if progress is not None:
        progress[0] = processed
    return processed


@numba.njit(cache=True, nogil=True)
def _heap_push(heap_z, heap_i, size, value, index):
    k = size
    while k > 0:
        parent = (k - 1) >> 1
        if heap_z[parent] <= value:
            break
        heap_z[k], heap_i[k] = heap_z[parent], heap_i[parent]
        k = parent
    heap_z[k], heap_i[k] = value, index
    return size + 1


@numba.njit(cache=True, nogil=True)
def priority_flood_chunk(elevation, state, global_filled, row_offset, col_offset):
    """Solve one tile from its outlets and currently known neighboring spill levels."""
    rows, cols = elevation.shape
    global_rows, global_cols = global_filled.shape
    best = np.full((rows, cols), np.inf, np.float32)
    closed = np.zeros((rows, cols), np.uint8)
    capacity = 1 << 14
    heap_z = np.empty(capacity, np.float32)
    heap_i = np.empty(capacity, np.int64)
    size = 0

    for row in range(rows):
        for col in range(cols):
            if state[row, col] != OUTLET:
                continue
            value = elevation[row, col]
            if value >= best[row, col]:
                continue
            best[row, col] = value
            if size >= capacity:
                capacity *= 2
                grown_z = np.empty(capacity, np.float32)
                grown_i = np.empty(capacity, np.int64)
                grown_z[:size] = heap_z[:size]
                grown_i[:size] = heap_i[:size]
                heap_z, heap_i = grown_z, grown_i
            size = _heap_push(heap_z, heap_i, size, value, row * cols + col)

    for row in range(rows):
        for col in range(cols):
            if state[row, col] == INVALID:
                continue
            if row != 0 and row != rows - 1 and col != 0 and col != cols - 1:
                continue
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    local_row, local_col = row + dr, col + dc
                    if 0 <= local_row < rows and 0 <= local_col < cols:
                        continue
                    outside_row = row_offset + local_row
                    outside_col = col_offset + local_col
                    if (outside_row < 0 or outside_col < 0
                            or outside_row >= global_rows or outside_col >= global_cols):
                        continue
                    outside_level = global_filled[outside_row, outside_col]
                    if not np.isfinite(outside_level):
                        continue
                    value = max(outside_level, elevation[row, col])
                    if value >= best[row, col]:
                        continue
                    best[row, col] = value
                    if size >= capacity:
                        capacity *= 2
                        grown_z = np.empty(capacity, np.float32)
                        grown_i = np.empty(capacity, np.int64)
                        grown_z[:size] = heap_z[:size]
                        grown_i[:size] = heap_i[:size]
                        heap_z, heap_i = grown_z, grown_i
                    size = _heap_push(heap_z, heap_i, size, value, row * cols + col)

    processed = 0
    while size > 0:
        value = heap_z[0]
        current = heap_i[0]
        size -= 1
        if size > 0:
            last_z, last_i, index = heap_z[size], heap_i[size], 0
            while True:
                child = 2 * index + 1
                if child >= size:
                    break
                if child + 1 < size and heap_z[child + 1] < heap_z[child]:
                    child += 1
                if heap_z[child] >= last_z:
                    break
                heap_z[index], heap_i[index] = heap_z[child], heap_i[child]
                index = child
            heap_z[index], heap_i[index] = last_z, last_i

        row, col = current // cols, current % cols
        if closed[row, col] != 0 or value != best[row, col]:
            continue
        closed[row, col] = 1
        processed += 1
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                next_row, next_col = row + dr, col + dc
                if (next_row < 0 or next_col < 0 or next_row >= rows or next_col >= cols
                        or state[next_row, next_col] == INVALID
                        or closed[next_row, next_col] != 0):
                    continue
                next_value = max(value, elevation[next_row, next_col])
                if next_value >= best[next_row, next_col]:
                    continue
                best[next_row, next_col] = next_value
                if size >= capacity:
                    capacity *= 2
                    grown_z = np.empty(capacity, np.float32)
                    grown_i = np.empty(capacity, np.int64)
                    grown_z[:size] = heap_z[:size]
                    grown_i[:size] = heap_i[:size]
                    heap_z, heap_i = grown_z, grown_i
                size = _heap_push(heap_z, heap_i, size, next_value,
                                  next_row * cols + next_col)
    return best, processed


def priority_flood_tiled(elevation, filled, state, chunk_size, progress=None):
    """Exact tiled minimax flood, revisiting tiles when neighboring spills improve."""
    rows, cols = elevation.shape
    tile_rows = (rows + chunk_size - 1) // chunk_size
    tile_cols = (cols + chunk_size - 1) // chunk_size
    queued = np.zeros((tile_rows, tile_cols), dtype=bool)
    work = deque()
    resolved = 0

    for tile_row in range(tile_rows):
        row0 = tile_row * chunk_size
        row1 = min(row0 + chunk_size, rows)
        for tile_col in range(tile_cols):
            col0 = tile_col * chunk_size
            col1 = min(col0 + chunk_size, cols)
            state_block = state[row0:row1, col0:col1]
            outlet_count = int(np.count_nonzero(state_block == OUTLET))
            if outlet_count:
                queued[tile_row, tile_col] = True
                work.append((tile_row, tile_col))
    if progress is not None:
        progress[0] = resolved
        progress[1] = 0
        progress[2] = len(work)
    if not work:
        raise ValueError("Priority-Flood has no outlet cells")

    tile_runs = 0
    while work:
        tile_row, tile_col = work.popleft()
        queued[tile_row, tile_col] = False
        row0, col0 = tile_row * chunk_size, tile_col * chunk_size
        row1, col1 = min(row0 + chunk_size, rows), min(col0 + chunk_size, cols)
        slices = (slice(row0, row1), slice(col0, col1))
        elevation_block = elevation[slices]
        state_block = state[slices]
        filled_block = filled[slices]
        old_filled = np.array(filled_block, copy=True)
        best, _ = priority_flood_chunk(
            elevation_block, state_block, filled, row0, col0,
        )
        improved = np.isfinite(best) & (best < old_filled)
        newly_resolved = improved & ~np.isfinite(old_filled) & (state_block != INVALID)
        resolved += int(newly_resolved.sum())
        filled_block[improved] = best[improved]
        tile_runs += 1

        boundary_improved = (
            np.any(improved[0, :]) or np.any(improved[-1, :])
            or np.any(improved[:, 0]) or np.any(improved[:, -1])
        )
        if boundary_improved:
            for row_delta in (-1, 0, 1):
                for col_delta in (-1, 0, 1):
                    if row_delta == 0 and col_delta == 0:
                        continue
                    neighbor_row = tile_row + row_delta
                    neighbor_col = tile_col + col_delta
                    if (0 <= neighbor_row < tile_rows and 0 <= neighbor_col < tile_cols
                            and not queued[neighbor_row, neighbor_col]):
                        queued[neighbor_row, neighbor_col] = True
                        work.append((neighbor_row, neighbor_col))

        if progress is not None:
            progress[0] = resolved
            progress[1] = tile_runs
            progress[2] = len(work)

    unresolved = 0
    for row0 in range(0, rows, chunk_size):
        row1 = min(row0 + chunk_size, rows)
        for col0 in range(0, cols, chunk_size):
            col1 = min(col0 + chunk_size, cols)
            state_block = state[row0:row1, col0:col1]
            filled_block = filled[row0:row1, col0:col1]
            land = state_block == 0
            unresolved += int(np.count_nonzero(land & ~np.isfinite(filled_block)))
            state_block[land & np.isfinite(filled_block)] = LAND_CLOSED
    if unresolved:
        raise ValueError(f"{unresolved:,} valid land cells are not connected to any outlet")
    return resolved, tile_runs


def monitor_flood(counter, expected, started, stop):
    last = None
    while not stop.wait(10):
        resolved = int(counter[0])
        tile_runs = int(counter[1])
        pending_tiles = int(counter[2])
        current = (resolved, tile_runs, pending_tiles)
        if current == last:
            continue
        last = current
        elapsed = time.perf_counter() - started
        rate = resolved / max(elapsed, 1.0)
        remaining = max(expected - resolved, 0) / max(rate, 1e-9)
        percent = min(resolved / max(expected, 1) * 100, 100.0)
        if not np.isfinite(remaining) or remaining > 365 * 24 * 60 * 60:
            eta_text = "ETA stabilizing"
        else:
            eta = datetime.now() + timedelta(seconds=remaining)
            eta_text = f"ETA {int(remaining)}s (about {eta:%H:%M})"
        print(f"Priority-Flood chunks: {resolved:,}/{expected:,} cells ({percent:.1f}%), "
              f"tile solves {tile_runs:,}, pending {pending_tiles:,}, "
              f"elapsed {int(elapsed)}s, {eta_text}", flush=True)


@numba.njit(cache=True)
def basin_stats_from_filled(labels, depth, filled, count):
    cells = np.zeros(count + 1, np.int64)
    volume = np.zeros(count + 1, np.float64)
    deepest = np.zeros(count + 1, np.float32)
    deepest_index = np.full(count + 1, -1, np.int64)
    spill = np.full(count + 1, -np.inf, np.float64)
    rows, cols = labels.shape
    for row in range(rows):
        for col in range(cols):
            label = labels[row, col]
            if label == 0:
                continue
            value = depth[row, col]
            cells[label] += 1
            volume[label] += value
            if value > deepest[label]:
                deepest[label] = value
                deepest_index[label] = row * cols + col
            if filled[row, col] > spill[label]:
                spill[label] = filled[row, col]
    return cells, volume, deepest, deepest_index, spill


@numba.njit(cache=True, nogil=True)
def saddle_union_disk(labels, sill_dem, state, spill_levels, tolerance):
    count = len(spill_levels) - 1
    parent = np.arange(count + 1, dtype=np.int32)
    neighbors = np.empty(8, dtype=np.int32)
    rows, cols = labels.shape
    scanned = 0
    for row in range(rows):
        for col in range(cols):
            if state[row, col] != LAND_CLOSED or labels[row, col] != 0:
                continue
            scanned += 1
            if scanned % 50_000_000 == 0:
                print("  checked dry saddle cells:", scanned)
            neighbor_count = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = row + dr, col + dc
                    if (dr == 0 and dc == 0) or nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                        continue
                    basin_id = labels[nr, nc]
                    if basin_id <= 0:
                        continue
                    found = False
                    for index in range(neighbor_count):
                        if neighbors[index] == basin_id:
                            found = True
                            break
                    if not found:
                        neighbors[neighbor_count] = basin_id
                        neighbor_count += 1
            if neighbor_count < 2:
                continue
            sill = sill_dem[row, col]
            for first_index in range(neighbor_count):
                first = neighbors[first_index]
                for second_index in range(first_index + 1, neighbor_count):
                    second = neighbors[second_index]
                    lower_spill = min(spill_levels[first], spill_levels[second])
                    if sill >= lower_spill + tolerance:
                        continue
                    first_root = base._find_union_root(parent, first)
                    second_root = base._find_union_root(parent, second)
                    if first_root != second_root:
                        parent[second_root] = first_root
    return parent


def group_basins(labels, sill_dem, state, table, tolerance):
    if table.empty:
        return np.zeros(1, dtype=np.int32), table.copy(), pd.DataFrame(), 0
    basin_count = int(table.lake_id.max())
    spill_levels = np.full(basin_count + 1, np.inf, dtype=np.float32)
    spill_levels[table.lake_id.to_numpy(dtype=np.int64)] = table.spill_level_m.to_numpy(dtype=np.float32)
    if tolerance > 0:
        parent = saddle_union_disk(labels, sill_dem, state, spill_levels, tolerance)
    else:
        parent = np.arange(basin_count + 1, dtype=np.int32)
    roots = np.empty(basin_count, dtype=np.int32)
    for index in range(1, basin_count + 1):
        roots[index - 1] = base._find_union_root.py_func(parent, index)
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
    lookup = groups.set_index("group_id")
    for column in ("member_basin_count", "group_area_m2", "group_volume_sum_m3",
                   "group_max_depth_m", "group_max_spill_level_m"):
        grouped[column] = grouped.group_id.map(lookup[column])
    grouped["merge_sill_tolerance_m"] = tolerance
    return mapping, grouped, groups, basin_count - len(unique_roots)


def make_table(labels, depth, filled, count, transform, min_depth, min_area):
    cells, volume, deepest, deepest_index, spill = basin_stats_from_filled(labels, depth, filled, count)
    cell_area = abs(transform.a * transform.e)
    area = cells * cell_area
    keep = (deepest >= min_depth) & (area >= min_area)
    keep[0] = False
    mapping = np.zeros(count + 1, dtype=np.int32)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    base.relabel(labels, depth, mapping)
    rows, cols = np.divmod(deepest_index[keep], labels.shape[1])
    x, y = rasterio.transform.xy(transform, rows, cols)
    table = pd.DataFrame({
        "lake_id": np.arange(1, int(keep.sum()) + 1),
        "area_m2": area[keep],
        "volume_m3": volume[keep] * cell_area,
        "max_depth_m": deepest[keep].astype(float),
        "mean_depth_m": volume[keep] / np.maximum(cells[keep], 1),
        "spill_level_m": spill[keep],
        "deepest_x": np.asarray(x, dtype=float),
        "deepest_y": np.asarray(y, dtype=float),
    })
    return table.sort_values("volume_m3", ascending=False).reset_index(drop=True)


def iter_windows(width, height, block_size):
    yield from windows(width, height, block_size)


def write_raster(path, array, transform, crs, dtype, nodata, block_size, state=None):
    height, width = array.shape
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": dtype, "crs": crs, "transform": transform, "nodata": nodata,
        "tiled": True, "blockxsize": 512, "blockysize": 512,
        "compress": "deflate", "predictor": 3 if dtype == "float32" else 2, "BIGTIFF": "YES",
    }
    with rasterio.open(path, "w", **profile) as target:
        for window in iter_windows(width, height, block_size):
            row0, col0 = int(window.row_off), int(window.col_off)
            row1, col1 = row0 + int(window.height), col0 + int(window.width)
            data = np.array(array[row0:row1, col0:col1], copy=True)
            if state is not None:
                data[state[row0:row1, col0:col1] != LAND_CLOSED] = nodata
            target.write(data, 1, window=window)


def run(args):
    base.configure_proj_data()
    run_started = time.perf_counter()
    prepare_directories(args)
    norway_bounds, geometries, tree = polygon_index(args.norway)
    if args.mosaic is not None:
        mosaic_path, bounds, _ = inspect_mosaic(args.mosaic)
        print(f"Reusing DTM10 mosaic: {mosaic_path}", flush=True)
    else:
        paths, tile_bounds = source_tiles(args.dtm_dir, args.bounds)
        bounds = tuple(args.bounds) if args.bounds is not None else tile_bounds
        if args.bounds is None and (tile_bounds[0] > norway_bounds[0] + 10
                                    or tile_bounds[1] > norway_bounds[1] + 10
                                    or tile_bounds[2] < norway_bounds[2] - 10
                                    or tile_bounds[3] < norway_bounds[3] - 10):
            print("WARNING: DTM10 tile bounds do not contain the full Norway polygon bounds; "
                  "areas outside tile coverage will not be analyzed.", flush=True)
        if not paths:
            raise ValueError("No DTM10 tile overlaps the requested bounds")
        print(f"Selected {len(paths)} of {len(list(Path(args.dtm_dir).rglob('*.tif')))} DTM10 tiles", flush=True)

        estimated_shape = (
            round((bounds[3] - bounds[1]) / 10.0),
            round((bounds[2] - bounds[0]) / 10.0),
        )
        check_scratch_space(args, estimated_shape)
        mosaic_path = args.work_dir / "dtm10_mosaic.tif"
        if mosaic_path.exists():
            raise FileExistsError(f"Mosaic already exists in work directory: {mosaic_path}")
        create_mosaic(paths, bounds, mosaic_path)
    with rasterio.open(mosaic_path) as mosaic:
        shape_ = (mosaic.height, mosaic.width)
    check_scratch_space(args, shape_)
    local_boundary = args.bounds is not None
    elevation, filled, state, shape_, transform, crs, expected = initialize_arrays(
        mosaic_path, args.work_dir, tree, geometries, local_boundary, args.block_size,
    )

    started = time.perf_counter()
    counter = np.zeros(3, dtype=np.int64)
    stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_flood,
        args=(counter, expected, started, stop),
        daemon=True,
    )
    monitor.start()
    try:
        resolved, tile_runs = priority_flood_tiled(
            elevation, filled, state, args.flood_chunk_size, counter,
        )
    finally:
        stop.set()
        monitor.join()
    print(f"Priority-Flood complete: resolved {resolved:,} cells in {tile_runs:,} tile solves; "
          f"{time.perf_counter() - started:.1f} s", flush=True)
    filled.flush()
    state.flush()

    height, width = shape_
    depth = np.memmap(args.work_dir / "depth.f32", mode="w+", dtype=np.float32, shape=shape_)
    wet = np.memmap(args.work_dir / "wet.u8", mode="w+", dtype=np.uint8, shape=shape_)
    for window in windows(width, height, args.block_size):
        row0, col0 = int(window.row_off), int(window.col_off)
        row1, col1 = row0 + int(window.height), col0 + int(window.width)
        depth_block = filled[row0:row1, col0:col1] - elevation[row0:row1, col0:col1]
        state_block = state[row0:row1, col0:col1]
        depth_block[(state_block != LAND_CLOSED) | (depth_block < 1e-3)] = 0.0
        depth[row0:row1, col0:col1] = depth_block
        wet[row0:row1, col0:col1] = (depth_block > 0).astype(np.uint8)
    depth.flush()
    wet.flush()

    labels = np.memmap(args.work_dir / "labels.i32", mode="w+", dtype=np.int32, shape=shape_)
    count = ndimage.label(wet, structure=np.ones((3, 3), dtype=bool), output=labels)
    if count >= np.iinfo(np.int32).max:
        raise OverflowError(f"Too many connected basins for int32 labels: {count}")
    labels.flush()
    del wet
    (args.work_dir / "wet.u8").unlink()
    print(f"Labeled {count:,} raw depressions", flush=True)

    table = make_table(labels, depth, filled, count, transform, args.min_depth, args.min_area)
    mapping, table, groups, merged = group_basins(
        labels, elevation, state, table, args.merge_sill_m,
    )
    print(f"Kept {len(table):,} basins in {len(groups):,} groups; merged {merged:,}", flush=True)

    write_raster(args.out_dir / "potential_lake_depth.tif", depth, transform, crs,
                 "float32", -9999.0, args.block_size, state)
    write_raster(args.out_dir / "potential_lake_id.tif", labels, transform, crs,
                 "int32", 0, args.block_size)
    table.to_csv(args.out_dir / "potential_lakes.csv", index=False)
    groups.to_csv(args.out_dir / "potential_lake_groups.csv", index=False)
    base.relabel(labels, depth, mapping)
    labels.flush()
    depth.flush()
    write_raster(args.out_dir / "potential_lake_group_id.tif", labels, transform, crs,
                 "int32", 0, args.block_size)

    summary = {
        "lakes": len(table),
        "total_area_km2": float(table.area_m2.sum() / 1e6),
        "total_volume_km3": float(table.volume_m3.sum() / 1e9),
        "min_depth_m": args.min_depth,
        "min_area_m2": args.min_area,
        "bbox": list(bounds),
        "dtm_dir": str(args.dtm_dir),
        "mosaic_path": str(mosaic_path),
        "resolution_m": 10.0,
        "crs": "EPSG:25833",
        "outlet_assumption": "cells outside Norway polygon are 0 m ocean",
        "local_boundary_outlet": local_boundary,
        "merge_sill_m": args.merge_sill_m,
        "merged_basins": merged,
        "basin_groups": len(groups),
        "polygons_written": False,
        "work_dir": str(args.work_dir),
        "flood_chunk_size": args.flood_chunk_size,
        "flood_tile_solves": tile_runs,
        "seconds": round(time.perf_counter() - run_started, 1),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if len(table):
        print(table.head(10).to_string(index=False))

    for array in (elevation, filled, state, depth, labels):
        array.flush()
        array._mmap.close()
    if not args.keep_work:
        try:
            shutil.rmtree(args.work_dir)
        except PermissionError:
            if any(args.work_dir.iterdir()):
                raise
            print(f"Scratch files removed; could not remove empty work directory {args.work_dir}",
                  flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())