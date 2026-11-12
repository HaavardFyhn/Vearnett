"""Build the complete Norway training grid from N100 geodatabases."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from contour_terrain_metrics import add_contour_terrain_metrics

CRS = "EPSG:25833"
ROOT = Path(__file__).resolve().parent
N100_GDB = ROOT / "data" / "Basisdata_0000_Norge_25833_N100Kartdata_FGDB.gdb"
MARITIME_GDB = ROOT / "data" / "Basisdata_0000_Norge_25833_NorgesMaritimeGrenser_FGDB.gdb"
OUTPUT = ROOT / "outputs"
AREA_CANDIDATES = ("N100_Arealdekke_omrade", "N_100_Arealdekke", "N100_Arealdekke")
CONTOUR_CANDIDATES = ("N100_Høyde_senterlinje", "N100_Hoyde_senterlinje")
POINT_CANDIDATES = ("N100_Høyde_posisjon", "N100_Hoyde_posisjon")
BASELINE_CANDIDATES = ("grunnlinje", "Grunnlinje")
HEIGHT_FIELDS = ("høyde", "hoyde", "elevation", "elev", "z", "meterOverHavet", "vertikalnivå")
RADII = (10_000, 25_000)


def layers(path):
    try:
        import pyogrio
        return pyogrio.list_layers(path)[:, 0].tolist()
    except Exception:
        import fiona
        return list(fiona.listlayers(path))


def resolve_layer(path, requested, candidates):
    available = layers(path)
    if requested:
        if requested not in available:
            raise ValueError(f"Layer {requested!r} not found. Available: {available}")
        return requested
    folded = {x.casefold(): x for x in available}
    for x in candidates:
        if x.casefold() in folded:
            return folded[x.casefold()]
    raise ValueError(f"None of {candidates} found in {path}. Available: {available}")


def projected(g, label):
    if g.crs is None:
        raise ValueError(f"{label} has no CRS")
    return g.to_crs(CRS) if g.crs.to_epsg() != 25833 else g


def find_field(columns, requested, candidates, label):
    cols = list(columns)
    if requested:
        if requested not in cols:
            raise ValueError(f"{requested!r} not found in {label}. Available: {cols}")
        return requested
    folded = {str(c).casefold(): c for c in cols}
    for x in candidates:
        if x.casefold() in folded:
            return folded[x.casefold()]
    for c in cols:
        f = str(c).casefold()
        if "høyde" in f or "hoyde" in f or f == "elevation":
            return c
    raise ValueError(f"Cannot detect value field in {label}. Available: {cols}")


def nearest(points, source, value_field, out_value, out_distance):
    cols = ["geometry"] + ([value_field] if value_field else [])
    j = gpd.sjoin_nearest(points[["grid_id", "geometry"]], source[cols], how="left", distance_col=out_distance)
    j = j.sort_values(["grid_id", out_distance]).drop_duplicates("grid_id")
    keep = ["grid_id", out_distance] + ([value_field] if value_field else [])
    j = pd.DataFrame(j[keep])
    if value_field:
        j = j.rename(columns={value_field: out_value})
    return j


def terrain_metrics(grid, elevation_col):
    xy = grid[["x", "y"]].to_numpy(np.float64)
    z = grid[elevation_col].to_numpy(np.float64)
    tree = cKDTree(xy)
    for radius in RADII:
        suffix = f"{radius // 1000}km"
        groups = tree.query_ball_point(xy, r=radius)
        relief = np.full(len(grid), np.nan, np.float32)
        rugged = np.full(len(grid), np.nan, np.float32)
        tpi = np.full(len(grid), np.nan, np.float32)
        for i, ids in enumerate(groups):
            ids = np.asarray(ids, dtype=np.int64)
            values = z[ids]
            values = values[np.isfinite(values)]
            if len(values) >= 3 and np.isfinite(z[i]):
                relief[i] = values.max() - values.min()
                rugged[i] = values.std()
                tpi[i] = z[i] - values.mean()
        grid[f"relief_{suffix}"] = relief
        grid[f"rugstd_{suffix}"] = rugged
        grid[f"tpi_{suffix}"] = tpi
        print(f"Terrain metrics complete: {suffix}")
    return grid


def build(output=OUTPUT, spacing_m=5000.0, force=False, n100_gdb=N100_GDB,
          maritime_gdb=MARITIME_GDB, area_layer=None, contour_layer=None,
          point_layer=None, baseline_layer=None, contour_field=None, point_field=None):
    outputdir = ( output / f"norway_grid_{int(spacing_m)}m.gpkg" )
    outputdir, n100_gdb, maritime_gdb = map(Path, (outputdir, n100_gdb, maritime_gdb))
    if outputdir.exists() and not force:
        print(f"SKIP trainmap; exists: {outputdir}")
        return outputdir
    started = time.time()
    area_layer = resolve_layer(n100_gdb, area_layer, AREA_CANDIDATES)
    contour_layer = resolve_layer(n100_gdb, contour_layer, CONTOUR_CANDIDATES)
    point_layer = resolve_layer(n100_gdb, point_layer, POINT_CANDIDATES)
    baseline_layer = resolve_layer(maritime_gdb, baseline_layer, BASELINE_CANDIDATES)

    areas = projected(gpd.read_file(n100_gdb, layer=area_layer), area_layer)
    obj_field = next((c for c in areas.columns if str(c).casefold() == "objtype"), None)
    if obj_field is None:
        raise ValueError(f"objtype not found. Available: {list(areas.columns)}")
    areas = areas.loc[areas.geometry.notna() & ~areas.geometry.is_empty, [obj_field, "geometry"]].copy()
    ocean_mask = areas[obj_field].astype(str).str.strip().str.casefold().eq("havflate")
    ocean = areas.loc[ocean_mask, ["geometry"]]
    land = areas.loc[~ocean_mask, [obj_field, "geometry"]]

    minx, miny, maxx, maxy = land.total_bounds
    xs = np.arange(np.floor(minx/spacing_m)*spacing_m, np.ceil(maxx/spacing_m)*spacing_m + spacing_m/2, spacing_m)
    ys = np.arange(np.floor(miny/spacing_m)*spacing_m, np.ceil(maxy/spacing_m)*spacing_m + spacing_m/2, spacing_m)
    xx, yy = np.meshgrid(xs, ys)
    lattice = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=CRS)
    grid = gpd.sjoin(lattice, land, how="inner", predicate="within")
    grid = grid.loc[~grid.index.duplicated()].drop(columns="index_right", errors="ignore").reset_index(drop=True)
    grid = grid.rename(columns={obj_field: "objtype"})
    grid.insert(0, "grid_id", np.arange(len(grid), dtype=np.int32))
    grid["x"], grid["y"] = grid.geometry.x, grid.geometry.y
    print(f"Grid points retained: {len(grid):,}; ocean points excluded")

    contours = projected(gpd.read_file(n100_gdb, layer=contour_layer), contour_layer)
    contour_field = find_field(contours.columns, contour_field, HEIGHT_FIELDS, contour_layer)
    contours[contour_field] = pd.to_numeric(contours[contour_field], errors="coerce")
    contours = contours.dropna(subset=[contour_field, "geometry"])
    grid = grid.merge(nearest(grid, contours, contour_field, "elevation_contour", "elevation_contour_dist"), on="grid_id", how="left")

    points = projected(gpd.read_file(n100_gdb, layer=point_layer), point_layer)
    point_field = find_field(points.columns, point_field, HEIGHT_FIELDS, point_layer)
    points[point_field] = pd.to_numeric(points[point_field], errors="coerce")
    points = points.dropna(subset=[point_field, "geometry"])
    grid = grid.merge(nearest(grid, points, point_field, "elevation_point", "elevation_point_dist"), on="grid_id", how="left")

    grid = gpd.GeoDataFrame(grid, geometry="geometry", crs=CRS)
    grid["elevation_source_diff"] = (grid["elevation_contour"] - grid["elevation_point"]).abs()
    # Compatibility with the existing pipeline: terrain metrics use contour elevation.
    grid["elevation"] = grid["elevation_contour"]

    coast_lines = gpd.GeoDataFrame(geometry=ocean.geometry.boundary, crs=CRS)
    grid = grid.merge(nearest(grid, coast_lines, None, None, "coast_dist_fjord"), on="grid_id", how="left")
    baseline = projected(gpd.read_file(maritime_gdb, layer=baseline_layer), baseline_layer)
    baseline = baseline.loc[baseline.geometry.notna() & ~baseline.geometry.is_empty, ["geometry"]]
    grid = grid.merge(nearest(grid, baseline, None, None, "coast_dist_ocean"), on="grid_id", how="left")
    grid = gpd.GeoDataFrame(grid, geometry="geometry", crs=CRS)
    # Compatibility with current models.
    grid["coast_dist"] = grid["coast_dist_fjord"]


    t0 = time.time()
    grid = add_contour_terrain_metrics(
        grid=grid,
        contours=contours,
        elevation_field=contour_field,
        radii_m=(10_000, 25_000),
        contour_sample_step_m=1000.0,
        terrain_resolution_m=float(spacing_m),
    )
    print(f"Terrain metrics: {time.time()-t0:.1f} s")

    order = ["grid_id", "x", "y", "objtype", "elevation", "elevation_contour", "elevation_point",
             "elevation_source_diff", "elevation_contour_dist", "elevation_point_dist",
             "coast_dist", "coast_dist_fjord", "coast_dist_ocean",
             "relief_10km", "rugstd_10km", "tpi_10km",
             "relief_25km", "rugstd_25km", "tpi_25km", "geometry"]
    grid = grid[order]
    outputdir.parent.mkdir(parents=True, exist_ok=True)
    if outputdir.exists(): outputdir.unlink()
    t0 = time.time()
    import pyogrio
    pyogrio.write_dataframe(grid, outputdir, layer="trainmap", driver="GPKG")
    print(f"Write: {time.time()-t0:.1f} s")
    print(f"Saved {len(grid):,} points: {outputdir}")
    print(f"Elapsed: {(time.time()-started)/60:.1f} min")
    return outputdir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spacing-m", type=float, default=5000)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--n100-gdb", type=Path, default=N100_GDB)
    p.add_argument("--maritime-gdb", type=Path, default=MARITIME_GDB)
    p.add_argument("--area-layer"); p.add_argument("--contour-layer"); p.add_argument("--point-layer"); p.add_argument("--baseline-layer")
    p.add_argument("--contour-field"); p.add_argument("--point-field")
    p.add_argument("--force", action="store_true")
    p.add_argument("--list-layers", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.list_layers:
        print("N100 layers:\n" + "\n".join(layers(a.n100_gdb)))
        print("\nMaritime layers:\n" + "\n".join(layers(a.maritime_gdb)))
        return
    build(**{k:v for k,v in vars(a).items() if k != "list_layers"})

if __name__ == "__main__": main()
