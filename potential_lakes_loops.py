"""Find mapped depression basins directly from N50 contour loops.

This vector-only workflow uses N50_Høyde_senterlinje height, auxiliary, and
depression contours. It joins line segments at each elevation, polygonizes
closed loops, and groups nested contours. Ordinary contours are considered
basins only when lower-elevation loops are nested inside higher ones. No DTM,
raster grid, interpolation, or tiles are used. Depth and volume are
contour-based lower bounds; single-contour loops have unknown depth.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd  # Load before pyogrio/Shapely to avoid PROJ/GDAL DLL conflicts.
import numpy as np
import pandas as pd
import pyogrio
from shapely.ops import polygonize, unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parent
DEFAULT_N50 = ROOT / "data" / "Basisdata_0000_Norge_25833_N50Kartdata_FGDB.gdb"
DEFAULT_NORWAY = ROOT / "outputs" / "norway_polygon.gpkg"
DEFAULT_OUT = ROOT / "outputs" / "potential_lakes_loops"
LAYER = "N50_Høyde_senterlinje"
DEPRESSION_TYPE = "Forsenkningskurve"
CONTOUR_TYPES = ("Høydekurve", "Hjelpekurve", DEPRESSION_TYPE)
CRS = "EPSG:25833"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n50", type=Path, default=DEFAULT_N50)
    parser.add_argument("--norway", type=Path, default=DEFAULT_NORWAY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-area-m2", type=float, default=0.0,
                        help="Minimum clipped outer-loop area to report")
    parser.add_argument("--min-depth-span-m", type=float, default=0.0,
                        help="Optional minimum observed contour span; single-loop depth is unknown")
    parser.add_argument("--floor-step-m", type=float, default=20.0,
                        help="Assumed depth below the lowest mapped contour for step-based max-depth/volume estimates")
    parser.add_argument("--min-loop-area-m2", type=float, default=100.0,
                        help="Discard polygonized linework slivers below this area")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="Optional EPSG:25833 test extent")
    args = parser.parse_args(argv)
    if min(args.min_area_m2, args.min_depth_span_m, args.min_loop_area_m2, args.floor_step_m) < 0:
        parser.error("area and depth thresholds must be nonnegative")
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


def progress(label, done, total, started):
    elapsed = time.perf_counter() - started
    remaining = elapsed / max(done, 1) * max(total - done, 0)
    eta = datetime.now() + timedelta(seconds=remaining)
    print(f"{label}: {done:,}/{total:,} elevations ({100 * done / max(total, 1):.1f}%), "
          f"elapsed {format_duration(elapsed)}, ETA {format_duration(remaining)} "
          f"(about {eta:%H:%M})", flush=True)


def read_norway(path):
    frame = pyogrio.read_dataframe(path)
    if frame.empty or frame.crs is None:
        raise ValueError(f"Could not read Norway polygon from {path}")
    if frame.crs.to_epsg() != 25833:
        frame = frame.to_crs(CRS)
    geometries = [geom for geom in frame.geometry if geom is not None and not geom.is_empty]
    if len(geometries) == 1:
        return geometries[0]
    return unary_union(geometries)


def line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [line for part in geometry.geoms for line in line_parts(part)]
    return []


def polygonize_contours(frame, min_loop_area_m2=100.0):
    """Join same-elevation, same-class linework and return polygonized loops."""
    if "objtype" not in frame:
        frame = frame.assign(objtype="")
    groups = list(frame.groupby(["hoyde", "objtype"], sort=True))
    total = len(groups)
    started = time.perf_counter()
    loops = []
    report_interval = max(1, total // 50)

    for index, ((elevation, objtype), group) in enumerate(groups, 1):
        lines = [line for geom in group.geometry for line in line_parts(geom)]
        if lines:
            noded = unary_union(lines)
            for polygon in polygonize(noded):
                if polygon.area >= min_loop_area_m2:
                    loops.append({"elevation_m": float(elevation),
                                  "source_types": [str(objtype)] if objtype else [],
                                  "geometry": polygon})
        if index == 1 or index == total or index % report_interval == 0:
            progress("Polygonizing contour groups", index, total, started)
    return loops


def nest_contour_loops(loops):
    """Assign each lower loop to its smallest enclosing higher contour."""
    geometries = [loop["geometry"] for loop in loops]
    elevations = [loop["elevation_m"] for loop in loops]
    source_types = [tuple(loop.get("source_types", [])) for loop in loops]
    tree = STRtree(geometries)
    parent = [-1] * len(loops)

    for child_index, child in enumerate(loops):
        child_geom = geometries[child_index]
        candidates = []
        for candidate_index in tree.query(child_geom):
            candidate_index = int(candidate_index)
            if candidate_index == child_index:
                continue
            if source_types[candidate_index] != source_types[child_index]:
                continue
            candidate_geom = geometries[candidate_index]
            if (elevations[candidate_index] > elevations[child_index]
                    and candidate_geom.area > child_geom.area
                    and candidate_geom.covers(child_geom)):
                candidates.append(candidate_index)
        if candidates:
            parent[child_index] = min(candidates, key=lambda i: geometries[i].area)

    families = {}
    for index in range(len(loops)):
        root = index
        seen = set()
        while parent[root] >= 0 and root not in seen:
            seen.add(root)
            root = parent[root]
        families.setdefault(root, []).append(index)
    return families


def basin_records(loops, families, norway, min_area_m2, min_depth_span_m, floor_step_m=20.0):
    records = []
    loop_records = []
    prepared_norway = prep(norway)
    for root, members in families.items():
        clipped_geometries = {}
        for index in members:
            geometry = loops[index]["geometry"]
            if prepared_norway.covers(geometry):
                clipped = geometry
            elif prepared_norway.intersects(geometry):
                clipped = geometry.intersection(norway)
            else:
                continue
            if not clipped.is_empty and clipped.area > 0:
                clipped_geometries[index] = clipped
        if root not in clipped_geometries:
            continue
        members = list(clipped_geometries)
        outer_geom = clipped_geometries[root]
        if outer_geom.area < min_area_m2:
            continue

        levels = sorted({loops[index]["elevation_m"] for index in members}, reverse=True)
        explicit_depression = any(
            DEPRESSION_TYPE in loops[index].get("source_types", []) for index in members
        )
        nested_lower_contours = len(levels) > 1
        if not explicit_depression and not nested_lower_contours:
            continue
        rim_level = levels[0]
        lowest_level = levels[-1]
        depth_span = rim_level - lowest_level if len(levels) > 1 else np.nan
        if np.isfinite(depth_span) and depth_span < min_depth_span_m:
            continue

        areas_by_level = {}
        clipped_by_level = {}
        for level in levels:
            clipped = [clipped_geometries[index] for index in members
                       if loops[index]["elevation_m"] == level]
            clipped_by_level[level] = clipped
            areas_by_level[level] = unary_union(clipped).area if clipped else 0.0

        volume_lower_bound = np.nan
        volume_floor_step = areas_by_level[lowest_level] * floor_step_m
        if len(levels) > 1:
            volume_lower_bound = sum(
                0.5 * (areas_by_level[high] + areas_by_level[low]) * (high - low)
                for high, low in zip(levels, levels[1:])
            )
            volume_floor_step += sum(
                areas_by_level[low] * (high - low)
                for high, low in zip(levels, levels[1:])
            )

        low_geometries = clipped_by_level[lowest_level]
        deepest_contour = max(low_geometries, key=lambda geom: geom.area).representative_point()
        basin_id = len(records) + 1
        records.append({
            "basin_id": basin_id,
            "area_m2": float(outer_geom.area),
            "rim_contour_elevation_m": float(rim_level),
            "lowest_contour_elevation_m": float(lowest_level),
            "depth_span_lower_bound_m": float(depth_span) if np.isfinite(depth_span) else np.nan,
            "volume_above_lowest_contour_m3": float(volume_lower_bound),
            "max_depth_assumed_m": float((depth_span if np.isfinite(depth_span) else 0.0) + floor_step_m),
            "volume_floor_step_assumed_m3": float(volume_floor_step),
            "floor_step_assumption_m": float(floor_step_m),
            "contour_level_count": len(levels),
            "contour_loop_count": len(members),
            "detection_basis": "nested lower contours" if nested_lower_contours else "explicit depression contour",
            "depth_status": "contour span plus assumed floor step" if nested_lower_contours
                            else "assumed one floor step below single contour",
            "deepest_contour_x": float(deepest_contour.x),
            "deepest_contour_y": float(deepest_contour.y),
            "geometry": outer_geom,
        })
        for index in members:
            loop_geom = clipped_geometries[index]
            loop_records.append({
                "basin_id": basin_id,
                "contour_elevation_m": loops[index]["elevation_m"],
                "source_types": ",".join(loops[index].get("source_types", [])),
                "area_m2": float(loop_geom.area),
                "is_outer_contour": index == root,
                "geometry": loop_geom,
            })
    return records, loop_records


def write_geopackage(frame, path, layer):
    pyogrio.write_dataframe(frame, path, layer=layer, driver="GPKG")


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    norway = read_norway(args.norway)

    print(f"Reading N50 height, auxiliary, and depression contours from {LAYER}...", flush=True)
    read_started = time.perf_counter()
    read_options = {} if args.bbox is None else {"bbox": tuple(args.bbox)}
    contour_types = ",".join("'" + name + "'" for name in CONTOUR_TYPES)
    contours = pyogrio.read_dataframe(
        args.n50, layer=LAYER, columns=["objtype", "hoyde"],
        where=f"objtype IN ({contour_types})", **read_options,
    )
    contours["hoyde"] = pd.to_numeric(contours["hoyde"], errors="coerce")
    contours = contours.dropna(subset=["hoyde", "geometry"])
    contour_feature_count = len(contours)
    print(f"Read {contour_feature_count:,} features in {format_duration(time.perf_counter() - read_started)}; "
          f"{contours.hoyde.nunique():,} distinct elevations", flush=True)

    loops = polygonize_contours(contours, args.min_loop_area_m2)
    closed_loop_count = len(loops)
    del contours
    print(f"Closed contour polygons: {closed_loop_count:,}", flush=True)
    families = nest_contour_loops(loops)
    print(f"Nested contour families: {len(families):,}", flush=True)
    records, loop_records = basin_records(
        loops, families, norway, args.min_area_m2, args.min_depth_span_m, args.floor_step_m,
    )
    del loops
    records.sort(key=lambda record: (
        -np.nan_to_num(record["volume_floor_step_assumed_m3"], nan=-1.0),
        -record["area_m2"],
    ))
    remap = {record["basin_id"]: new_id for new_id, record in enumerate(records, 1)}
    for record in records:
        record["basin_id"] = remap[record["basin_id"]]
    for record in loop_records:
        record["basin_id"] = remap[record["basin_id"]]

    table = pd.DataFrame([{key: value for key, value in record.items() if key != "geometry"}
                          for record in records])
    table.to_csv(args.out_dir / "potential_lakes.csv", index=False)
    if records:
        basins = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS)
        write_geopackage(basins, args.out_dir / "potential_lakes.gpkg", "basins")
    if loop_records:
        loops_frame = gpd.GeoDataFrame(loop_records, geometry="geometry", crs=CRS)
        write_geopackage(loops_frame, args.out_dir / "potential_lake_contours.gpkg", "contours")

    summary = {
        "method": "N50 height/auxiliary/depression contour polygonization; no DTM, raster, or interpolation",
        "contour_types": list(CONTOUR_TYPES),
        "contour_features_read": contour_feature_count,
        "closed_contour_polygons": closed_loop_count,
        "basins": len(records),
        "basins_with_depth_span": int(table.depth_span_lower_bound_m.notna().sum()) if len(table) else 0,
        "total_area_km2": float(table.area_m2.sum() / 1e6) if len(table) else 0.0,
        "volume_above_mapped_contours_km3": float(table.volume_above_lowest_contour_m3.sum() / 1e9) if len(table) else 0.0,
        "volume_floor_step_assumed_km3": float(table.volume_floor_step_assumed_m3.sum() / 1e9) if len(table) else 0.0,
        "floor_step_assumption_m": args.floor_step_m,
        "min_area_m2": args.min_area_m2,
        "min_depth_span_m": args.min_depth_span_m,
        "min_polygonized_loop_area_m2": args.min_loop_area_m2,
        "seconds": round(time.perf_counter() - run_started, 1),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if records:
        print(table.head(10).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
