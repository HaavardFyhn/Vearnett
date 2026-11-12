"""Contour-derived terrain metrics for a point grid.

Builds a continuous elevation field from regularly sampled vertices along elevation
contours, then computes neighbourhood relief, ruggedness (standard deviation), and
TPI. This avoids assigning every grid point the elevation of its nearest contour.

Dependencies: geopandas, numpy, pandas, scipy, shapely.
All geometries must be in a metric projected CRS (the pipeline uses EPSG:25833).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree


def sample_contours_regularly(
    contours: gpd.GeoDataFrame,
    elevation_field: str,
    sample_step_m: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return regularly spaced XY samples and elevations from contour lines.

    Regular spacing prevents long/densely digitized contours from dominating the
    interpolation merely because they contain more vertices.
    """
    if sample_step_m <= 0:
        raise ValueError("sample_step_m must be > 0")
    if contours.crs is None or not contours.crs.is_projected:
        raise ValueError("Contours must use a projected metric CRS")
    if elevation_field not in contours.columns:
        raise ValueError(f"Missing contour elevation field: {elevation_field}")

    xy_parts, z_parts = [], []
    source = contours[[elevation_field, "geometry"]].copy()
    source[elevation_field] = pd.to_numeric(source[elevation_field], errors="coerce")
    source = source.dropna(subset=[elevation_field, "geometry"])

    for z, geom in source.itertuples(index=False, name=None):
        if geom is None or geom.is_empty:
            continue
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            if line.length == 0:
                continue
            distances = np.arange(0.0, line.length, sample_step_m)
            # Always include the endpoint.
            if distances.size == 0 or distances[-1] < line.length:
                distances = np.append(distances, line.length)
            pts = [line.interpolate(float(d)) for d in distances]
            xy_parts.append(np.asarray([(p.x, p.y) for p in pts], dtype=np.float64))
            z_parts.append(np.full(len(pts), float(z), dtype=np.float64))

    if not xy_parts:
        raise ValueError("No usable contour-line samples were produced")

    xy = np.vstack(xy_parts)
    z = np.concatenate(z_parts)

    # Collapse coincident XY samples; average only if overlapping contours disagree.
    samples = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "z": z})
    samples = samples.groupby(["x", "y"], as_index=False, sort=False)["z"].mean()
    return samples[["x", "y"]].to_numpy(np.float64), samples["z"].to_numpy(np.float64)


def interpolate_contour_elevation(
    query_xy: np.ndarray,
    contour_xy: np.ndarray,
    contour_z: np.ndarray,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """Interpolate a continuous contour-derived elevation field.

    Linear interpolation is used inside the convex hull. Nearest-sample fallback is
    used only outside the hull or where linear interpolation cannot return a value.
    """
    query_xy = np.asarray(query_xy, dtype=np.float64)
    linear = LinearNDInterpolator(contour_xy, contour_z, fill_value=np.nan)
    nearest = NearestNDInterpolator(contour_xy, contour_z)
    out = np.empty(len(query_xy), dtype=np.float32)

    for start in range(0, len(query_xy), chunk_size):
        stop = min(start + chunk_size, len(query_xy))
        q = query_xy[start:stop]
        values = np.asarray(linear(q), dtype=np.float64)
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = nearest(q[missing])
        out[start:stop] = values.astype(np.float32)
    return out


def add_contour_terrain_metrics(
    grid: gpd.GeoDataFrame,
    contours: gpd.GeoDataFrame,
    elevation_field: str,
    radii_m: tuple[int, ...] = (10_000, 25_000),
    contour_sample_step_m: float = 1000.0,
    terrain_resolution_m: float | None = None,
    min_neighbors: int = 3,
) -> gpd.GeoDataFrame:
    """Add contour DEM elevation, relief, ruggedness, and TPI to a point grid.

    A regular terrain lattice is interpolated from contour lines. Neighbourhood
    metrics are calculated from that lattice, not from nearest-contour elevations.

    Definitions:
      relief_R = max(z within R) - min(z within R)
      rugstd_R = population standard deviation of z within R
      tpi_R    = grid-point interpolated z - mean(z within R)
    """
    if grid.crs is None or not grid.crs.is_projected:
        raise ValueError("Grid must use a projected metric CRS")
    if contours.crs != grid.crs:
        contours = contours.to_crs(grid.crs)
    if terrain_resolution_m is None:
        # Use grid spacing when detectable; otherwise a conservative 1 km DEM.
        xs = np.unique(np.round(grid.geometry.x.to_numpy(), 6))
        ys = np.unique(np.round(grid.geometry.y.to_numpy(), 6))
        dx = np.diff(xs); dy = np.diff(ys)
        steps = np.concatenate([dx[dx > 0], dy[dy > 0]])
        terrain_resolution_m = float(np.min(steps)) if len(steps) else 1000.0
    if terrain_resolution_m <= 0:
        raise ValueError("terrain_resolution_m must be > 0")

    contour_xy, contour_z = sample_contours_regularly(
        contours, elevation_field, contour_sample_step_m
    )

    minx, miny, maxx, maxy = grid.total_bounds
    margin = max(radii_m)
    xs = np.arange(minx - margin, maxx + margin + terrain_resolution_m * 0.5,
                   terrain_resolution_m)
    ys = np.arange(miny - margin, maxy + margin + terrain_resolution_m * 0.5,
                   terrain_resolution_m)
    xx, yy = np.meshgrid(xs, ys)
    terrain_xy = np.column_stack([xx.ravel(), yy.ravel()])
    terrain_z = interpolate_contour_elevation(terrain_xy, contour_xy, contour_z)

    grid_xy = np.column_stack([grid.geometry.x.to_numpy(), grid.geometry.y.to_numpy()])
    grid_z = interpolate_contour_elevation(grid_xy, contour_xy, contour_z)
    out = grid.copy()
    out["elevation_contour_surface"] = grid_z

    # Neighbourhood statistics are computed on the lattice with window filters,
    # then sampled bilinearly at the grid points. This replaces a per-point loop.
    shape = (len(ys), len(xs))
    z2d = terrain_z.reshape(shape).astype(np.float64)
    valid = np.isfinite(z2d)
    z0 = np.where(valid, z2d, 0.0)
    vf = valid.astype(np.float64)
    coords = np.vstack([
        (grid_xy[:, 1] - ys[0]) / terrain_resolution_m,
        (grid_xy[:, 0] - xs[0]) / terrain_resolution_m,
    ])

    def sample(arr2d: np.ndarray) -> np.ndarray:
        return ndimage.map_coordinates(arr2d, coords, order=1, mode="nearest")

    for radius in radii_m:
        suffix = f"{int(radius // 1000)}km"
        half = int(np.ceil(radius / terrain_resolution_m))
        r_idx = np.arange(-half, half + 1) * terrain_resolution_m
        disk = (r_idx[:, None] ** 2 + r_idx[None, :] ** 2) <= float(radius) ** 2

        zmax = ndimage.maximum_filter(np.where(valid, z2d, -np.inf), footprint=disk, mode="nearest")
        zmin = ndimage.minimum_filter(np.where(valid, z2d, np.inf), footprint=disk, mode="nearest")
        count2d = ndimage.correlate(vf, disk.astype(np.float64), mode="constant")
        s1 = ndimage.correlate(z0, disk.astype(np.float64), mode="constant")
        s2 = ndimage.correlate(z0 * z0, disk.astype(np.float64), mode="constant")
        with np.errstate(divide="ignore", invalid="ignore"):
            mean2d = s1 / count2d
            var2d = np.maximum(s2 / count2d - mean2d * mean2d, 0.0)
        std2d = np.sqrt(var2d)
        enough = count2d >= min_neighbors
        relief2d = np.where(enough, zmax - zmin, np.nan)
        std2d = np.where(enough, std2d, np.nan)
        mean2d = np.where(enough, mean2d, np.nan)

        grid_ok = np.isfinite(grid_z)
        relief = np.where(grid_ok, sample(relief2d), np.nan)
        rugged = np.where(grid_ok, sample(std2d), np.nan)
        tpi = np.where(grid_ok, grid_z - sample(mean2d), np.nan)
        count = np.rint(sample(count2d)).astype(np.int32)

        out[f"relief_{suffix}"] = relief.astype(np.float32)
        out[f"rugstd_{suffix}"] = rugged.astype(np.float32)
        out[f"tpi_{suffix}"] = tpi.astype(np.float32)
        out[f"nterr_{suffix}"] = count

    return out


# Integration example for build_trainmap_complete.py:
# contours = projected(gpd.read_file(n100_gdb, layer=contour_layer), contour_layer)
# contour_field = find_field(contours.columns, contour_field, HEIGHT_FIELDS, contour_layer)
# grid = add_contour_terrain_metrics(
#     grid=grid,
#     contours=contours,
#     elevation_field=contour_field,
#     radii_m=(10_000, 25_000),
#     contour_sample_step_m=1000.0,
#     terrain_resolution_m=1000.0,
# )
# grid["elevation"] = grid["elevation_contour_surface"]
