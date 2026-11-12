"""Prepare independent seasonal targets and multiscale terrain for GPU comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET

import train_senorge_grid_pairs as training
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from terrain_patches import patch

from remote_senorge.train_remote import sha256, write_json


ROOT = Path(__file__).resolve().parent
TARGETS = [("winter_tg", (12, 1, 2), "tg"), ("summer_tg", (6, 7, 8), "tg"),
           ("winter_tn", (12, 1, 2), "tn"), ("summer_tn", (6, 7, 8), "tn")]
SCALES = [50, 100, 500]


def buffered_split(xy, pairs, seed=42, radius=16500):
    blocks, inverse = np.unique(np.floor(xy/100000).astype(np.int64), axis=0, return_inverse=True)
    if len(blocks) < 8:
        raise ValueError("Need at least eight occupied 100 km blocks")
    rng = np.random.default_rng(seed)
    test = rng.choice(len(blocks), max(1, int(np.ceil(len(blocks)*.2))), replace=False)
    remaining = np.setdiff1d(np.arange(len(blocks)), test)
    validation = rng.choice(remaining, max(1, int(np.ceil(len(remaining)*.2))), replace=False)
    group = np.zeros(len(blocks), dtype=np.int8)
    group[validation], group[test] = 1, 2
    position = xy % 100000
    interior = ((position >= radius) & (position <= 100000-radius)).all(axis=1)
    cell_group = group[inverse]
    keep = interior[pairs].all(axis=1) & (cell_group[pairs[:, 0]] == cell_group[pairs[:, 1]])
    result = np.full(len(pairs), -1, dtype=np.int8)
    result[keep] = cell_group[pairs[keep, 0]]
    if any((result == i).sum() < 2 for i in range(3)):
        raise ValueError("Buffered splits are too small")
    return result


def terrain_rasters(dtm_dir, cache):
    from osgeo import gdal
    gdal.UseExceptions()
    paths = sorted(dtm_dir.glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No DEM tiles in {dtm_dir}")
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs.to_epsg() != 25833 or src.count != 1 or max(src.res) > 50:
                raise ValueError(f"Expected single-band DEM at 50 m or finer in EPSG:25833: {path}")
    mosaic = cache / "terrain.vrt"
    if not mosaic.exists():
        # All tiles were checked as EPSG:25833 above; only their WKT spelling differs.
        dataset = gdal.BuildVRT(str(mosaic), [str(p.resolve()) for p in paths],
                                options=gdal.BuildVRTOptions(
                                    allowProjectionDifference=True, outputSRS="EPSG:25833", strict=True))
        if dataset is None:
            raise RuntimeError("GDAL failed to build terrain VRT")
        dataset = None
    included = set()
    for node in ET.parse(mosaic).findall(".//SourceFilename"):
        path = Path(node.text)
        if node.get("relativeToVRT") == "1":
            path = mosaic.parent / path
        included.add(path.resolve())
    if included != {path.resolve() for path in paths}:
        raise ValueError("Terrain VRT omitted DEM tiles; remove the incomplete raster cache before resuming")
    print(f"Verified all {len(paths)} source DEM tiles in mosaic", flush=True)
    rasters = [mosaic]
    for resolution in SCALES[1:]:
        path = cache / f"dem_{resolution}m.tif"
        if not path.exists():
            pending = path.with_name(path.stem + ".pending.tif")
            print(f"Preparing DEM at {resolution} m", flush=True)
            ds = gdal.Warp(str(pending), str(mosaic), xRes=resolution, yRes=resolution,
                           resampleAlg="average", format="GTiff", dstNodata=-9999,
                           creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
                           warpMemoryLimit=256)
            if ds is None:
                raise RuntimeError("GDAL failed to resample terrain")
            ds = None
            pending.replace(path)
        rasters.append(path)
    return rasters


def prepare_patches(grid, rasters, folder, cache):
    shape = (len(grid), 3, 64, 64)
    progress = cache / "patch_progress.json"
    completed = json.loads(progress.read_text())["completed"] if progress.exists() else 0
    mode = "r+" if completed else "w+"
    height = np.lib.format.open_memmap(folder / "terrain.npy", mode=mode, dtype=np.float16, shape=shape)
    valid = np.lib.format.open_memmap(folder / "terrain_valid.npy", mode=mode, dtype=np.uint8, shape=shape)
    sources = []
    try:
        sources = [rasterio.open(path) for path in rasters]
        for start in range(completed, len(grid), 512):
            stop = min(start+512, len(grid))
            for i in range(start, stop):
                row = grid.iloc[i]
                for scale, (source, resolution) in enumerate(zip(sources, SCALES)):
                    h, mask = patch(source, row.x, row.y, resolution, row.elevation)
                    height[i, scale], valid[i, scale] = h, mask
            height.flush()
            valid.flush()
            write_json(progress, dict(completed=stop, cells=len(grid)))
            if start % 8192 == 0 or stop == len(grid):
                print(f"Terrain patches: {stop:,}/{len(grid):,} cells", flush=True)
    finally:
        for source in sources:
            source.close()
        del height, valid


def additional_patch_bytes(folder, cells):
    remaining = 0
    for name, itemsize in (("terrain.npy", 2), ("terrain_valid.npy", 1)):
        expected = cells*3*64*64*itemsize + 128
        path = folder / name
        allocated = 0
        if path.exists():
            stat = path.stat()
            allocated = stat.st_blocks*512 if hasattr(stat, "st_blocks") else stat.st_size
        remaining += max(0, expected-allocated)
    return remaining


def prepare(args):
    folder, cache = args.output, args.output / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    grid = gpd.read_file(args.source / "grid_features.gpkg")
    pairs = np.load(args.source / "cache" / "pairs.npy", allow_pickle=False)
    original_config = json.loads((args.source / "config.json").read_text())
    plan = pd.read_csv(args.source / "selected_dates.csv")
    if set(plan.year) != set(range(1996, 2026)):
        raise ValueError("Expected 1996–2025 date plan")
    config = dict(schema=1, source=training.fingerprint(args.source / "config.json"),
                  grid=training.fingerprint(args.source / "grid_features.gpkg"),
                  pairs=training.fingerprint(args.source / "cache" / "pairs.npy"),
                  dates=plan.to_dict("records"), scales=SCALES, targets=TARGETS,
                  dem=[training.fingerprint(p) for p in sorted(args.dtm_dir.glob("*.tif"))],
                  archives=original_config["archives"], patch_radius_m=16500)
    config = json.loads(json.dumps(config))
    config_path = cache / "source_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Seasonal sources/settings changed; choose another output")
    if not config["dem"]:
        raise FileNotFoundError("No terrain tiles found")
    write_json(config_path, config)
    package = args.output.with_suffix(".zip")
    if package.exists():
        raise FileExistsError("Seasonal ZIP already exists; choose another output")
    # 3 scales * 64^2 * (float16 height + uint8 mask), plus caches, archive and reserve.
    remaining_bytes = additional_patch_bytes(folder, len(grid))
    if shutil.disk_usage(folder).free < remaining_bytes + 18*1024**3:
        raise OSError("Insufficient disk: terrain allocation plus at least 18 GiB for labels, package and reserve required")
    archives = [Path(record["path"]) for record in original_config["archives"]]
    for record, path in zip(original_config["archives"], archives):
        if training.fingerprint(path) != record:
            raise ValueError(f"Source archive changed: {path.name}")
    labels = []
    date_counts = {}
    for name, months, variable in TARGETS:
        selected = plan.loc[pd.to_datetime(plan.date).dt.month.isin(months)].reset_index(drop=True)
        partition = selected.groupby(["year", pd.to_datetime(selected.date).dt.month, "partition"]).size()
        if len(selected) != 900 or len(partition) != 180:
            raise ValueError(f"Unexpected seasonal date selection: {name}")
        target_cache = cache / name
        target_cache.mkdir(exist_ok=True)
        target = training.stream_labels(archives, grid, pairs, target_cache, plan=selected, variable=variable)
        labels.append(target.target_error.to_numpy(dtype=np.float32))
        date_counts[name] = dict(fitting=int((selected.partition == 0).sum()),
                                 evaluation=int((selected.partition == 1).sum()), months=list(months))
    y = np.column_stack(labels)
    good = np.isfinite(y).all(axis=1)
    pairs, y = pairs[good], y[good]
    x = training.pair_features(grid, pairs).to_numpy()
    xy = grid[["x", "y"]].to_numpy(dtype=np.float64)
    split = buffered_split(xy, pairs)
    np.savez_compressed(folder / "seasonal_training.npz", x=x, y=y, pairs=pairs.astype(np.int32),
                        xy=xy, split=split, grid_ids=grid.grid_id.to_numpy())
    print(f"Seasonal labels ready: {len(pairs):,} complete pairs", flush=True)
    print("Buffered train/validation/test: " + str([int((split == group).sum()) for group in range(3)]), flush=True)
    rasters = terrain_rasters(args.dtm_dir, cache)
    prepare_patches(grid, rasters, folder, cache)
    valid_patches = np.load(folder / "terrain_valid.npy", mmap_mode="r")
    coverage = dict(mean_fraction=[float(valid_patches[:, scale].mean()) for scale in range(3)],
                    fully_missing_cells=[int((valid_patches[:, scale].sum(axis=(1, 2)) == 0).sum())
                                         for scale in range(3)])
    valid_patches._mmap.close()
    plan.to_csv(folder / "selected_dates.csv", index=False)
    files = ["terrain.npy", "terrain_valid.npy", "seasonal_training.npz", "selected_dates.csv"]
    for name in ("train_seasonal_terrain.py", "train_remote.py", "requirements.txt", "requirements-seasonal.txt",
                 "SEASONAL_TERRAIN_README.md", "test_seasonal_terrain.py"):
        shutil.copyfile(ROOT / "remote_senorge" / name, folder / name)
        files.append(name)
    manifest = dict(schema=1, targets=[t[0] for t in TARGETS], features=training.FEATURES,
                    patches=dict(pixels=64, resolutions_m=SCALES, north_up=True,
                                 values="(DEM elevation minus endpoint elevation)/1000, float16",
                                 mask="uint8 validity; missing elevation encoded0 with mask0, NOT sea level",
                                 reference="endpoint terrain-grid elevation; no global normalization"),
                    patch_radius_m=16500, cells=len(grid), pairs=len(pairs),
                    terrain_coverage=coverage,
                    date_counts=date_counts, years=list(range(1996, 2026)),
                    seasons="DJF/JJA by calendar month, not contiguous winter-year sequences",
                    spatial_counts={name: int((split == group).sum()) for name, group in
                                    (("train", 0), ("validation", 1), ("test", 2), ("excluded", -1))},
                    spatial_method="100km blocks, seed42 test20%, validation20% of remaining. Endpoints of every evaluated pair share partition; 16.5km interior buffer applied to ALL partitions (includes resampling margin).",
                    checksums={name: sha256(folder / name) for name in files},
                    limitations=["seNorge interpolated1km similarity, not independent station accuracy.",
                                 "50m/100m/500m DEM is a resampled representation, not exact or native10m terrain.",
                                 "Random within-month temporal evaluation describes historical similarity, not future performance.",
                                 "No temporal extrema/event-duration guarantee from sampled dates.",
                                 "Test pairs with overlapping geographic terrain support are excluded by full-block interior buffers.",
                                 "Terrain-only ablation includes distance and absolute x/y differences but no other tabular features.",
                                 "No Station Lab integration until trained seasonal models are returned and evaluated."])
    write_json(folder / "seasonal_manifest.json", manifest)
    files.append("seasonal_manifest.json")
    from remote_senorge.train_seasonal_terrain import load_dataset
    _, checked_height, checked_valid, _ = load_dataset(folder)
    checked_height._mmap.close()
    checked_valid._mmap.close()
    pending_package = package.with_suffix(".pending.zip")
    with zipfile.ZipFile(pending_package, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name in files:
            archive.write(folder / name, name)
    with zipfile.ZipFile(pending_package) as archive:
        if set(archive.namelist()) != set(files) or archive.testzip() is not None:
            raise ValueError("Seasonal archive integrity/allowlist check failed")
    digest = sha256(pending_package)
    pending_package.replace(package)
    write_json(cache / "complete.json", dict(status="complete", zip_sha256=digest,
                                           zip_bytes=package.stat().st_size, spatial_counts=manifest["spatial_counts"]))
    print(f"COMPLETE {package}: {package.stat().st_size/1024**3:.2f} GiB", flush=True)
    print(json.dumps(manifest["spatial_counts"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=training.WORK / "senorge_grid_pairs_1m_1996_2025_random")
    parser.add_argument("--output", type=Path, default=training.WORK / "senorge_seasonal_terrain")
    parser.add_argument("--dtm-dir", type=Path, default=ROOT.parent / "dtm50" / "data")
    args = parser.parse_args()
    from precompute_frost import computation_lock
    with computation_lock(args.output):
        if args.output.with_suffix(".zip").exists():
            raise FileExistsError("Seasonal ZIP already exists; choose another output")
        try:
            write_json(args.output / "status.json", dict(state="running"))
            prepare(args)
            write_json(args.output / "status.json", dict(state="complete"))
        except Exception as error:
            if not (args.output / "cache" / "complete.json").exists():
                write_json(args.output / "status.json", dict(state="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
