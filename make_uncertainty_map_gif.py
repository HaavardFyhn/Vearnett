"""Map and animate the model-predicted uncertainty for the greedy station network.

Re-evaluates the selected stations on a configurable target grid. Station locations
and source terrain features stay on the original selection grid. Affected area is
the number of cells improved beyond the threshold times the target cell area;
boundary cells count as full squares, not clipped land area.

Outputs (in outputs/, suffixed with the target resolution):
    uncertainty_map_final_1000m.png   final uncertainty map
    uncertainty_growth_1000m.gif      uncertainty and newly affected cells per station
    station_affected_area_1000m.csv   affected grid area per station in square metres
    station_affected_area_1000m.png   affected area versus station rank

Disk-backed replay frames use a unique temporary directory. Completed predictions
are copied to a uniquely named .npy file in outputs/ before plotting, for reuse.
"""
from __future__ import annotations
import sys
import time
import argparse
import tempfile
import shutil
from contextlib import ExitStack
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm  # noqa: F401 (used by make_norm)
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from greedy_400_stations_pruned import (  # noqa: E402
    make_cache, predict, subset_cache)
import joblib  # noqa: E402
import geopandas as gpd  # noqa: E402
import pyogrio  # noqa: E402
from greedy_400_stations_pruned import TERRAIN  # noqa: E402

OUT = HERE / "outputs"
MODEL = OUT / "temperature_5y_prediction_rmse_model.joblib"
SOURCE_GRID = OUT / "norway_grid_5000m.gpkg"
CKPT = OUT / "greedy_400_stations_checkpoint.npz"
STATIONS = OUT / "greedy_400_stations_stations.csv"
MAP_PNG = OUT / "uncertainty_map_final.png"
GIF = OUT / "uncertainty_growth.gif"
CELL_M = 1000
# Model target is RMSE of daily temperature (air_temperature, degrees Celsius).
UNIT_LABEL = "Usikkerhet (RMSE, °C)"
CMAP = "RdYlBu_r"  # blue = low uncertainty, red = high


def read_grid(path):
    """Read the grid with pyogrio, since Fiona on this machine drops the terrain columns."""
    g = pyogrio.read_dataframe(path)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=g.crs)
    if g.crs.to_epsg() != 25833:
        g = g.to_crs(25833)
    return g


def load_grid_all_fields(path):
    """Same preprocessing as load_grid in greedy_400_stations_pruned, using read_grid."""
    g = read_grid(path)
    required = ["elevation", "coast_dist_fjord", "coast_dist_ocean",
                "objtype", *TERRAIN, "geometry"]
    missing = sorted(set(required) - set(g.columns))
    if missing:
        raise ValueError(f"Grid missing fields: {missing}")
    g = g.dropna(subset=required).copy().reset_index(drop=True)
    g["x"] = g.geometry.x.astype(np.float32)
    g["y"] = g.geometry.y.astype(np.float32)
    g["grid_id"] = np.arange(len(g), dtype=np.int32)
    g["objtype_code"] = g["objtype"].astype("category").cat.codes.astype(np.int16)
    return g


def replay_surfaces(model, grid, cache, selected, source_cache=None,
                    surface_path=None, batch_size=50000, cleanup_stack=None):
    """Return the uncertainty surface after each station (shape: n_stations x n_cells)."""
    if source_cache is None:
        source_cache = cache
    shape = (len(selected), len(grid))
    surfaces = (np.empty(shape, dtype=np.float32) if surface_path is None else
                np.lib.format.open_memmap(surface_path, mode="w+", dtype=np.float32,
                                         shape=shape))
    if surface_path is not None and cleanup_stack is not None:
        cleanup_stack.callback(surfaces._mmap.close)
    current = None
    t0 = time.perf_counter()
    for k, idx in enumerate(selected):
        source = subset_cache(source_cache, np.array([idx]))
        pred = np.empty(len(grid), dtype=np.float32)
        for start in range(0, len(grid), batch_size):
            target_ids = np.arange(start, min(start + batch_size, len(grid)))
            pred[target_ids] = predict(model, source, cache, target_ids)[0]
        current = pred if current is None else np.minimum(current, pred)
        surfaces[k] = current
        if (k + 1) % 25 == 0 or k + 1 == len(selected):
            print(f"  replayed {k + 1}/{len(selected)} stations "
                  f"({time.perf_counter() - t0:.1f} s)")
    return surfaces


def affected_cells(surfaces, index, min_improvement):
    if index == 0:
        return np.ones(surfaces.shape[1], dtype=bool)
    return surfaces[index - 1] - surfaces[index] > min_improvement


def affected_areas(surfaces, cell_m, min_improvement):
    counts = [np.count_nonzero(affected_cells(surfaces, index, min_improvement))
              for index in range(len(surfaces))]
    return pd.DataFrame({
        "rank": np.arange(1, len(surfaces) + 1),
        "baseline": np.arange(len(surfaces)) == 0,
        "affected_cells": counts,
        "affected_area_m2": np.asarray(counts, dtype=np.int64) * cell_m ** 2,
        "cell_size_m": cell_m,
        "min_improvement_c": min_improvement,
    })


def plot_affected_areas(areas, path):
    incremental = areas.loc[~areas["baseline"].astype(bool)]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(incremental["rank"], incremental["affected_area_m2"])
    if (incremental["affected_area_m2"] > 0).all() and len(incremental):
        ax.set_yscale("log")
    else:
        cell_area = float(areas["cell_size_m"].iloc[0]) ** 2
        ax.set_yscale("symlog", linthresh=cell_area)
    ax.set_xlabel("Station added (rank)")
    ax.set_ylabel("Affected area (m², log scale)")
    threshold = float(areas["min_improvement_c"].iloc[0])
    cell_m = int(areas["cell_size_m"].iloc[0])
    ax.set_title(f"Area improved by > {threshold:g} °C RMSE ({cell_m} m grid)")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_surfaces(path, shape, resources, allow_prefix=False):
    surfaces = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(surfaces, np.memmap):
        surfaces.close()
        raise ValueError("Saved surfaces must be a .npy array")
    resources.callback(surfaces._mmap.close)
    if surfaces.shape != shape:
        if not (allow_prefix and surfaces.ndim == 2 and
                surfaces.shape[0] >= shape[0] and surfaces.shape[1] == shape[1]):
            raise ValueError(f"Saved surface shape {surfaces.shape} does not match {shape}")
    surfaces = surfaces[:shape[0]]
    for index, surface in enumerate(surfaces):
        if not np.isfinite(surface).all() or (surface < 0).any():
            raise ValueError(f"Invalid saved uncertainty values at station {index + 1}")
        if index and (surface > surfaces[index - 1] + 1e-5).any():
            raise ValueError(f"Saved uncertainty increases at station {index + 1}")
    return surfaces


def export_arcgis(grid, surfaces, areas, cell_m, threshold, prefix, rank):
    import rasterio
    from rasterio.transform import from_origin

    xs = grid.geometry.x.to_numpy()
    ys = grid.geometry.y.to_numpy()
    column_offsets = (xs - xs.min()) / cell_m
    row_offsets = (ys.max() - ys) / cell_m
    if not (np.allclose(column_offsets, np.rint(column_offsets), atol=1e-5, rtol=0) and
            np.allclose(row_offsets, np.rint(row_offsets), atol=1e-5, rtol=0)):
        raise ValueError("Grid centres do not align with the requested raster cell size")
    columns = np.rint(column_offsets).astype(int)
    rows = np.rint(row_offsets).astype(int)
    height, width = int(rows.max()) + 1, int(columns.max()) + 1
    if len(np.unique(rows * width + columns)) != len(grid):
        raise ValueError("Multiple target points occupy the same raster cell")
    transform = from_origin(xs.min() - cell_m / 2, ys.max() + cell_m / 2,
                            cell_m, cell_m)
    index = rank - 1
    reduction = (np.zeros(len(grid), dtype=np.float32) if index == 0 else
                 surfaces[index - 1] - surfaces[index])
    layers = {
        "uncertainty": (surfaces[index], "float32", -9999.0, "RMSE degrees C"),
        "rmse_reduction": (reduction, "float32", -9999.0, "RMSE reduction degrees C"),
        "affected": ((reduction > threshold).astype(np.uint8), "uint8", 255,
                     "1 = reduction above threshold; first station is baseline"),
    }
    for name, (values, dtype, nodata, description) in layers.items():
        raster = np.full((height, width), nodata, dtype=dtype)
        raster[rows, columns] = values
        path = prefix.with_name(f"{prefix.name}_rank{rank}_{name}.tif")
        with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                           count=1, dtype=dtype, crs=grid.crs, transform=transform,
                           nodata=nodata, compress="deflate", tiled=True) as dataset:
            dataset.write(raster, 1)
            dataset.set_band_description(1, description)
            dataset.update_tags(station_rank=rank, min_improvement_c=threshold,
                                cell_size_m=cell_m, baseline=(index == 0))
        print(f"Saved {path}")
    stations = gpd.GeoDataFrame(areas.copy(),
                               geometry=gpd.points_from_xy(areas["x"], areas["y"]),
                               crs=grid.crs)
    path = prefix.with_suffix(".gpkg")
    pyogrio.write_dataframe(stations, path, layer="stations", driver="GPKG")
    print(f"Saved {path} (stations with rank and affected_area_m2)")


def make_norm(final_surface, first_surface):
    """Log colour scale from the low end of the final map to the high end of the first map."""
    lo = max(float(np.percentile(final_surface, 0.5)), 1e-3)
    hi = float(np.percentile(first_surface, 99.5))
    return LogNorm(vmin=lo, vmax=hi), lo, hi


def set_log_ticks(cb, lo, hi):
    """Label the colourbar with plain numbers in degrees, not powers of ten."""
    candidates = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8]
    ticks = [t for t in candidates if lo <= t <= hi]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    cb.minorticks_off()


def cell_squares(grid, cell_m=CELL_M):
    """Square polygon for each cell, centred on the grid point."""
    half = cell_m / 2
    xs = grid.geometry.x.to_numpy()
    ys = grid.geometry.y.to_numpy()
    return [np.array([[x - half, y - half], [x + half, y - half],
                      [x + half, y + half], [x - half, y + half]])
            for x, y in zip(xs, ys)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-m", type=int, choices=[1000, 2500, 5000], default=CELL_M)
    parser.add_argument("--min-improvement", type=float, default=0.01,
                        help="Minimum RMSE reduction in degrees C to count a cell")
    parser.add_argument("--max-stations", type=int,
                        help="Replay only this many stations for a quick preview")
    parser.add_argument("--replot-only", action="store_true",
                        help="Redraw the log-area chart from the existing CSV; no prediction")
    parser.add_argument("--reuse-surfaces", type=Path,
                        help="Read completed .npy frames from the same grid/station run; no prediction")
    parser.add_argument("--skip-gif", action="store_true", help="Do not render the GIF")
    parser.add_argument("--export-arcgis", action="store_true",
                        help="Export GeoTIFF rasters and a station GeoPackage")
    parser.add_argument("--export-rank", type=int,
                        help="Station rank for ArcGIS rasters (default: final station)")
    args = parser.parse_args()
    if not np.isfinite(args.min_improvement) or args.min_improvement < 0:
        parser.error("--min-improvement must be finite and nonnegative")
    if args.max_stations is not None and args.max_stations < 1:
        parser.error("--max-stations must be positive")
    if args.export_rank is not None and args.export_rank < 1:
        parser.error("--export-rank must be positive")
    if args.replot_only and (args.reuse_surfaces or args.export_arcgis):
        parser.error("--replot-only uses only the CSV; use --reuse-surfaces for map exports")
    with ExitStack() as resources:
        scratch_dir = resources.enter_context(
            tempfile.TemporaryDirectory(prefix="uncertainty_replay_"))
        run(args, Path(scratch_dir), resources)


def run(args, scratch_dir, resources):
    grid_path = OUT / f"norway_grid_{args.cell_m}m.gpkg"
    suffix = f"_{args.cell_m}m"
    if args.max_stations is not None:
        suffix += f"_preview{args.max_stations}"
    map_png = MAP_PNG.with_stem(MAP_PNG.stem + suffix)
    gif = GIF.with_stem(GIF.stem + suffix)
    area_csv = OUT / f"station_affected_area{suffix}.csv"
    area_png = OUT / f"station_affected_area{suffix}.png"
    if args.replot_only:
        plot_affected_areas(pd.read_csv(area_csv), area_png)
        print(f"Saved {area_png} from {area_csv}; no predictions were run")
        return
    surface_path = scratch_dir / f"uncertainty_surfaces{suffix}.npy"
    print(f"Temporary replay cache: {surface_path}")
    timings = {}

    def tick(name, start):
        timings[name] = time.perf_counter() - start
        print(f"[{name}] {timings[name]:.1f} s")

    t = time.perf_counter()
    source_grid = load_grid_all_fields(SOURCE_GRID)
    grid = source_grid.copy() if grid_path == SOURCE_GRID else load_grid_all_fields(grid_path)
    categories = pd.Index(source_grid["objtype"].unique()).union(grid["objtype"].unique())
    for frame in (source_grid, grid):
        frame["objtype_code"] = pd.Categorical(frame["objtype"], categories=categories).codes
    z = np.load(CKPT)
    selected = z["selected"].astype(int).tolist()
    saved_final = z["current_error"].astype(np.float32)
    stations = pd.read_csv(STATIONS).sort_values("rank").reset_index(drop=True)
    if not selected or len(saved_final) != len(source_grid):
        raise ValueError("Checkpoint is empty or does not match the original selection grid")
    if len(stations) != len(selected) or not np.array_equal(
            source_grid.iloc[selected][["x", "y"]].to_numpy(),
            stations[["x", "y"]].to_numpy()):
        raise ValueError("Station CSV does not match the checkpoint and source grid")
    if args.max_stations is not None:
        selected = selected[:args.max_stations]
        stations = stations.iloc[:len(selected)].copy()
    if args.export_rank is not None and args.export_rank > len(selected):
        raise ValueError(f"--export-rank must not exceed {len(selected)}")
    tick("load", t)

    t = time.perf_counter()
    if args.reuse_surfaces:
        surfaces = load_surfaces(args.reuse_surfaces, (len(selected), len(grid)), resources,
                                 allow_prefix=args.max_stations is not None)
        print(f"Loaded {args.reuse_surfaces}; no model predictions were run")
    else:
        print(f"Replaying {len(selected)} stations over {len(grid):,} cells ...")
        source_cache = make_cache(source_grid)
        cache = make_cache(grid)
        model = joblib.load(MODEL)
        surfaces = replay_surfaces(model, grid, cache, selected, source_cache, surface_path,
                                   cleanup_stack=resources)
        surfaces.flush()
        with tempfile.NamedTemporaryFile(prefix=f"uncertainty_surfaces{suffix}_",
                                         suffix=".npy", dir=OUT, delete=False) as saved:
            saved_path = Path(saved.name)
        shutil.copyfile(surface_path, saved_path)
        print(f"Saved reusable predictions: {saved_path}")
    tick("replay", t)
    if grid_path == SOURCE_GRID and len(selected) == len(z["selected"]):
        max_diff = float(np.max(np.abs(surfaces[-1] - saved_final)))
        print(f"Replay check vs saved surface: max abs diff = {max_diff:.6f}")
        if max_diff > 1e-3:
            raise RuntimeError("Replayed surface does not match the checkpoint; aborting")
    else:
        print("New evaluation grid or preview: saved final surface is not directly comparable")
    areas = affected_areas(surfaces, args.cell_m, args.min_improvement)
    areas.insert(1, "x", stations["x"].to_numpy())
    areas.insert(2, "y", stations["y"].to_numpy())
    areas.to_csv(area_csv, index=False)
    plot_affected_areas(areas, area_png)
    print(f"Saved {area_csv} and {area_png}; first station is the initial baseline")
    if args.export_arcgis:
        export_arcgis(grid, surfaces, areas, args.cell_m, args.min_improvement,
                      OUT / f"station_network{suffix}", args.export_rank or len(selected))

    t = time.perf_counter()
    polys = cell_squares(grid, args.cell_m)
    xs = stations["x"].to_numpy()
    ys = stations["y"].to_numpy()
    # One colour scale for all frames, so the GIF shows the drop in uncertainty.
    norm, lo, hi = make_norm(surfaces[-1], surfaces[0])
    print(f"Colour scale: {lo:.2f} to {hi:.2f} °C (log)")
    cmap = plt.get_cmap(CMAP)
    marker = dict(s=18, c="black", edgecolors="white", linewidths=0.5, zorder=5)

    def draw_base(ax):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(grid.geometry.x.min() - 5000, grid.geometry.x.max() + 5000)
        ax.set_ylim(grid.geometry.y.min() - 5000, grid.geometry.y.max() + 5000)

    # Static map: final uncertainty with all completed stations.
    fig, ax = plt.subplots(figsize=(7, 10), dpi=150)
    pc = PolyCollection(polys, array=np.clip(surfaces[-1], lo, hi), cmap=cmap,
                        edgecolors="none", norm=norm)
    ax.add_collection(pc)
    ax.scatter(xs, ys, label=f"Stasjoner ({len(selected)})", **marker)
    draw_base(ax)
    ax.legend(loc="lower left", fontsize=8)
    cb = fig.colorbar(pc, ax=ax, shrink=0.6, label=UNIT_LABEL)
    set_log_ticks(cb, lo, hi)
    ax.set_title(f"Temperaturusikkerhet med {len(selected)} stasjoner")
    fig.savefig(map_png, bbox_inches="tight")
    plt.close(fig)
    tick("static_map", t)
    print(f"Saved {map_png}")
    if args.skip_gif:
        print("GIF rendering skipped; saved predictions can be reused later")
        return

    # Animation: one frame per station added.
    t = time.perf_counter()
    fig, ax = plt.subplots(figsize=(7, 10), dpi=80)
    pc = PolyCollection(polys, array=np.clip(surfaces[0], lo, hi), cmap=cmap,
                        edgecolors="none", norm=norm)
    ax.add_collection(pc)
    highlight = PolyCollection(polys, edgecolors="none", zorder=3)
    ax.add_collection(highlight)
    highlight_colors = np.zeros((len(grid), 4), dtype=np.float32)
    highlight_colors[:, :3] = [0.1, 0.85, 0.35]
    dots = ax.scatter([], [], **marker)
    new_dot = ax.scatter([], [], s=65, c="yellow", edgecolors="black",
                         marker="*", linewidths=0.6, zorder=6)
    cb = fig.colorbar(pc, ax=ax, shrink=0.6, label=UNIT_LABEL)
    set_log_ticks(cb, lo, hi)
    draw_base(ax)
    title = ax.set_title("")

    def update(k):
        pc.set_array(np.clip(surfaces[k], lo, hi))
        highlight_colors[:, 3] = affected_cells(surfaces, k, args.min_improvement) * 0.45
        if k == 0:
            highlight_colors[:, 3] = 0
        highlight.set_facecolors(highlight_colors)
        dots.set_offsets(np.column_stack([xs[: k + 1], ys[: k + 1]]))
        new_dot.set_offsets([[xs[k], ys[k]]])
        area = int(areas.iloc[k]["affected_area_m2"])
        label = "Initial coverage" if k == 0 else f"Improved > {args.min_improvement:g} °C (green)"
        title.set_text(f"Temperaturusikkerhet etter {k + 1} stasjoner\n"
                       f"{label}: {area:,} m²")
        return pc, highlight, dots, new_dot, title

    anim = FuncAnimation(fig, update, frames=len(selected), blit=False)
    anim.save(gif, writer=PillowWriter(fps=10))
    plt.close(fig)
    tick("gif", t)
    print(f"Saved {gif} ({len(selected)} frames)")
    print("Total: " + ", ".join(f"{k} {v:.1f} s" for k, v in timings.items())
          + f", sum {sum(timings.values()):.1f} s")


if __name__ == "__main__":
    main()
