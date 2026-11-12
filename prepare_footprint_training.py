"""Prepare dense six-target reference maps without modifying earlier experiments."""
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

import train_senorge_grid_pairs as training
import netCDF4
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from remote_senorge.footprint_model import TARGETS
from remote_senorge.train_remote import sha256, write_json
from precompute_frost import computation_lock


ROOT = Path(__file__).resolve().parent
FILES = ["scenes.npy", "local10.npy", "features.npy", "labels.npy", "label_mask.npy",
         "references.npz", "selected_dates.csv", "footprint_model.py", "train_footprint.py",
         "train_remote.py", "requirements.txt", "requirements-seasonal.txt",
         "FOOTPRINT_README.md", "test_footprint.py"]


def geometry(grid, counts=(256, 64, 64), seed=42, pixels=64):
    xy = grid[["x", "y"]].to_numpy(dtype=float)
    if pixels != 64:
        raise ValueError("Preparation uses 64 x 64 output pixels")
    blocks, inverse = np.unique(np.floor(xy/100000).astype(int), axis=0, return_inverse=True)
    if len(blocks) < 8:
        raise ValueError("Need at least eight geographic blocks")
    rng = np.random.default_rng(seed)
    group = np.zeros(len(blocks), dtype=np.int8)
    held = rng.choice(len(blocks), int(np.ceil(len(blocks)*.2)), replace=False)
    group[held] = 2
    remaining = np.setdiff1d(np.arange(len(blocks)), held)
    group[rng.choice(remaining, int(np.ceil(len(remaining)*.2)), replace=False)] = 1
    # 64 km output + 8 km halo each side; farthest input edge is 40.5 km.
    position = xy % 100000
    interior = ((position >= 41000) & (position <= 59000)).all(axis=1)
    refs, split = [], []
    for part, count in enumerate(counts):
        choices = np.flatnonzero(interior & (group[inverse] == part))
        if count < 2 or len(choices) < count:
            raise ValueError(f"Insufficient reference cells in partition {part}: {len(choices)}")
        # Round-robin blocks, random within blocks: avoid concentrating all scenes in one valley.
        buckets = [list(rng.permutation(choices[inverse[choices] == block]))
                   for block in np.unique(inverse[choices])]
        order = rng.permutation(len(buckets))
        chosen = []
        while len(chosen) < count:
            for bucket in order:
                if buckets[bucket] and len(chosen) < count:
                    chosen.append(buckets[bucket].pop())
        refs.extend(chosen)
        split.extend([part]*count)
    refs, split = np.asarray(refs, np.int32), np.asarray(split, np.int8)
    keys = np.round(xy/1000).astype(np.int64)
    if not np.allclose(xy, keys*1000):
        raise ValueError("Expected integer-kilometre terrain centres")
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    if len(lookup) != len(grid):
        raise ValueError("Duplicate terrain cells")
    mapping = np.full((len(refs), pixels, pixels), -1, dtype=np.int32)
    pairs, anchors, rows, columns = [], [], [], []
    for a, ref in enumerate(refs):
        cx, cy = keys[ref]
        for row in range(pixels):
            for col in range(pixels):
                target = lookup.get((cx+col-pixels//2, cy+pixels//2-row))
                if target is not None:
                    mapping[a, row, col] = target
                    pairs.append((ref, target))
                    anchors.append(a)
                    rows.append(row)
                    columns.append(col)
    return dict(refs=refs, split=split, mapping=mapping, pairs=np.asarray(pairs, np.int32),
                anchor=np.asarray(anchors, np.int32), row=np.asarray(rows, np.int16),
                col=np.asarray(columns, np.int16), xy=xy[refs])


def terrain_window(source, x, y, resolution, pixels, elevation):
    half = pixels*resolution/2
    raw = source.read(1, window=from_bounds(x-half, y-half, x+half, y+half, source.transform),
                      out_shape=(pixels, pixels), boundless=True, masked=True,
                      out_dtype="float32", resampling=Resampling.average)
    mask = ~np.ma.getmaskarray(raw) & np.isfinite(raw.data)
    height = np.where(mask, (raw.data.astype(float)-elevation)/1000, 0).astype(np.float16)
    if not np.isfinite(height).all():
        raise ValueError("Nonfinite encoded terrain")
    return np.stack([height, mask.astype(np.float16)])


def fine_mosaic(directory, cache):
    from osgeo import gdal
    gdal.UseExceptions()
    paths = sorted(directory.glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No native 10 m tiles in {directory}")
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs.to_epsg() != 25833 or src.count != 1 or max(src.res) > 10:
                raise ValueError(f"Not a native 10 m EPSG:25833 DEM: {path}")
    vrt = cache / "dtm10.vrt"
    if not vrt.exists():
        ds = gdal.BuildVRT(str(vrt), [str(p.resolve()) for p in paths],
                          options=gdal.BuildVRTOptions(
                              allowProjectionDifference=True, outputSRS="EPSG:25833", strict=True))
        if ds is None:
            raise RuntimeError("Could not build native 10 m mosaic")
        ds = None
    included = set()
    for node in ET.parse(vrt).findall(".//SourceFilename"):
        path = Path(node.text)
        included.add((vrt.parent / path if node.get("relativeToVRT") == "1" else path).resolve())
    if included != {path.resolve() for path in paths}:
        raise ValueError("Native 10 m mosaic omitted tiles")
    return vrt


def label_maps(archives, plan, grid, geometry_data, cache, labels, mask):
    pairs = geometry_data["pairs"]
    shape = (6, 2, 6, len(pairs))
    state_path = cache / "moments.npz"
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as stored:
            totals, completed = stored["moments"], int(stored["completed"])
        if totals.shape != shape:
            raise ValueError("Invalid footprint moment checkpoint")
    else:
        totals, completed = np.zeros(shape, dtype=np.float64), 0
    a, b = pairs.T
    rows, cols = grid.nc_row.to_numpy(), grid.nc_col.to_numpy()
    for year_index, path in enumerate(archives):
        if year_index < completed:
            continue
        year = int(path.stem.split("_")[-1])
        selected = plan.loc[plan.year == year]
        print(f"Footprint labels: {year}, {len(pairs):,} supervised cells", flush=True)
        with netCDF4.Dataset(path) as ds:
            dates = netCDF4.num2date(ds["time"][:], ds["time"].units,
                                    getattr(ds["time"], "calendar", "standard"))
            expected = pd.to_datetime(selected.date)
            actual = [f"{dates[i].year:04}-{dates[i].month:02}-{dates[i].day:02}"
                      for i in selected.day_index]
            if actual != expected.dt.strftime("%Y-%m-%d").tolist():
                raise ValueError("Date selection does not match NetCDF")
            xs, ys = np.asarray(ds["X"][:]), np.asarray(ds["Y"][:])
            if not np.allclose(np.diff(xs), 1000) or not np.allclose(abs(np.diff(ys)), 1000):
                raise ValueError("Unexpected seNorge resolution")
            if ((cols < 0) | (cols >= len(xs)) | (rows < 0) | (rows >= len(ys))).any():
                raise ValueError("Weather mapping outside source grid")
            # Same containing-cell convention as original preparation.
            gx, gy = grid.x.to_numpy(), grid.y.to_numpy()
            if (np.abs(xs[cols]-gx) > 1000).any() or (np.abs(ys[rows]-gy) > 1000).any():
                raise ValueError("Weather mapping changed")
            for day_number, record in enumerate(selected.itertuples(), start=1):
                month = pd.Timestamp(record.date).month
                seasonal = 2 if month in (12, 1, 2) else 4 if month in (6, 7, 8) else None
                for variable_index, variable in enumerate(("tg", "tn")):
                    var = ds[variable]
                    if var.units != "Celsius" or var.dimensions != ("time", "Y", "X"):
                        raise ValueError("Unexpected weather variable")
                    slab = np.ma.masked_invalid(var[record.day_index]).filled(np.nan)
                    values = slab[rows, cols]
                    targets = [variable_index] + ([] if seasonal is None else [seasonal+variable_index])
                    for target in targets:
                        for start in range(0, len(pairs), 100000):
                            stop = start+100000
                            training.update_moments(totals[target, int(record.partition), :, start:stop],
                                                    values[a[start:stop]], values[b[start:stop]])
                if day_number % 30 == 0:
                    print(f"Footprint labels: {year}, {day_number}/{len(selected)} dates", flush=True)
        pending = state_path.with_suffix(".pending.npz")
        np.savez(pending, moments=totals, completed=year_index+1)
        pending.replace(state_path)
        write_json(cache / "label_progress.json", dict(completed_years=year_index+1, years=len(archives)))
    for target in range(6):
        result, _ = training.regression_labels(totals[target, 0], totals[target, 1])
        months = pd.to_datetime(plan.date).dt.month
        selected = plan if target < 2 else plan.loc[months.isin((12, 1, 2) if target < 4 else (6, 7, 8))]
        fit_count, eval_count = int((selected.partition == 0).sum()), int((selected.partition == 1).sum())
        good = (totals[target, 0, 0] >= fit_count*.9) & (totals[target, 1, 0] >= eval_count*.9)
        good &= np.isfinite(result)
        ai, rr, cc = (geometry_data[name] for name in ("anchor", "row", "col"))
        labels[ai[good], target, rr[good], cc[good]] = result[good]
        mask[ai[good], target, rr[good], cc[good]] = True
    labels.flush()
    mask.flush()


def prepare(args):
    folder, cache = args.output, args.output / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    if args.output.with_suffix(".zip").exists():
        raise FileExistsError("Footprint ZIP already exists; choose a different output")
    grid = pyogrio.read_dataframe(args.source / "grid_features.gpkg")
    plan = pd.read_csv(args.source / "selected_dates.csv")
    if set(plan.year) != set(range(1996, 2026)) or len(plan) != 3600:
        raise ValueError("Expected the original 3600-date 1996-2025 plan")
    strata = plan.groupby(["year", pd.to_datetime(plan.date).dt.month, "partition"]).size()
    if len(strata) != 720 or not (strata.xs(0, level=2) == 8).all() or not (strata.xs(1, level=2) == 2).all():
        raise ValueError("Expected 8 fit / 2 evaluation dates per month")
    config = json.loads((args.source / "config.json").read_text())
    archives = [Path(record["path"]) for record in config["archives"]]
    for path, record in zip(archives, config["archives"]):
        if training.fingerprint(path) != record:
            raise ValueError(f"Weather source changed: {path}")
    dem100 = args.terrain / "cache" / "dem_100m.tif"
    settings = dict(schema=1, counts=args.counts, seed=42, targets=TARGETS,
                    grid=training.fingerprint(args.source / "grid_features.gpkg"),
                    dates=training.fingerprint(args.source / "selected_dates.csv"),
                    archives=config["archives"], dem100=training.fingerprint(dem100),
                    fine=[training.fingerprint(p) for p in sorted(args.dtm10.glob("*.tif"))])
    settings = json.loads(json.dumps(settings))
    settings_path = cache / "source_config.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Footprint sources/settings changed; choose another output")
    write_json(settings_path, settings)
    geo = geometry(grid, args.counts)
    n = len(geo["refs"])
    # Arrays, atomic moment checkpoint, final archive and reserve. No deletion of old outputs.
    required = n*2*800*800*2 + len(geo["pairs"])*6*2*6*8*2 + 4*1024**3
    allocated = sum(p.stat().st_size for p in folder.glob("*.npy"))
    allocated += (cache / "moments.npz").stat().st_size if (cache / "moments.npz").exists() else 0
    if shutil.disk_usage(folder).free < max(0, required-allocated):
        raise OSError("Insufficient free disk for footprint arrays, atomic checkpoints and archive")
    fine = fine_mosaic(args.dtm10, cache)
    np.savez_compressed(folder / "references.npz", **geo)
    specs = {"scenes.npy": (np.float16, (n, 2, 800, 800)),
             "local10.npy": (np.float16, (n, 2, 128, 128)),
             "features.npy": (np.float32, (n, 25, 64, 64)),
             "labels.npy": (np.float32, (n, 6, 64, 64)),
             "label_mask.npy": (np.bool_, (n, 6, 64, 64))}
    with ExitStack() as stack:
        arrays = {}
        for name, (dtype, shape) in specs.items():
            path = folder / name
            exists = path.exists()
            data = np.lib.format.open_memmap(path, mode="r+" if exists else "w+", dtype=dtype, shape=shape)
            stack.callback(data._mmap.close)
            if data.shape != shape or data.dtype != dtype:
                raise ValueError(f"Wrong footprint array: {name}")
            if not exists:
                data[:] = 0
            arrays[name] = data
        progress = cache / "terrain_progress.json"
        completed = json.loads(progress.read_text())["completed"] if progress.exists() else 0
        with rasterio.open(dem100) as context, rasterio.open(fine) as local:
            for i in range(completed, n):
                ref = grid.iloc[int(geo["refs"][i])]
                # Output reference is pixel (32,32); an even grid's scene centre is offset 500 m.
                arrays["scenes.npy"][i] = terrain_window(context, ref.x-500, ref.y+500, 100, 800, ref.elevation)
                arrays["local10.npy"][i] = terrain_window(local, ref.x, ref.y, 10, 128, ref.elevation)
                keep = geo["anchor"] == i
                features = training.pair_features(grid, geo["pairs"][keep]).to_numpy()
                arrays["features.npy"][i, :, geo["row"][keep], geo["col"][keep]] = features
                for name in ("scenes.npy", "local10.npy", "features.npy"):
                    arrays[name].flush()
                write_json(progress, dict(completed=i+1, references=n))
                if i % 16 == 0 or i+1 == n:
                    print(f"Footprint terrain: {i+1}/{n} references", flush=True)
        label_maps(archives, plan, grid, geo, cache, arrays["labels.npy"], arrays["label_mask.npy"])
        coverage = dict(context_fraction=float(arrays["scenes.npy"][:, 1].mean(dtype=np.float64)),
                        fine10_fraction=float(arrays["local10.npy"][:, 1].mean(dtype=np.float64)),
                        fully_missing_fine10=int((arrays["local10.npy"][:, 1].sum(axis=(1, 2)) == 0).sum()))
        coverage["by_partition"] = {
            name: dict(fine10_fraction=float(arrays["local10.npy"][geo["split"] == part, 1].mean(dtype=np.float64)))
            for part, name in enumerate(("train", "validation", "test"))}
        if coverage["fully_missing_fine10"] == n:
            raise ValueError("No selected reference has native 10 m terrain coverage")
        counts = {TARGETS[i]: int(arrays["label_mask.npy"][:, i].sum()) for i in range(6)}
        if min(counts.values()) == 0:
            raise ValueError("A target has no valid labels")
    plan.to_csv(folder / "selected_dates.csv", index=False)
    for name in FILES[7:]:
        shutil.copyfile(ROOT / "remote_senorge" / name, folder / name)
    manifest = dict(schema=1, targets=TARGETS, features=training.FEATURES, pixels=64, halo=8,
                    context_resolution_m=100, fine_resolution_m=10, fine_pixels=128,
                    source_period="1996–2025", selected_dates=3600,
                    reference_pixel=[32, 32], crs="EPSG:25833", north_up=True,
                    context_values="float16 kilometres relative to reference grid elevation, validity channel",
                    splits={name: int((geo["split"] == i).sum()) for i, name in enumerate(("train", "validation", "test"))},
                    split_method="100km blocks seed42, test20%, validation20% remaining; all reference centres 41km inside block. Entire 80km input scenes stay inside one partition block.",
                    supervised_cells=counts, terrain_coverage=coverage,
                    limitations=[
                        "Historical seNorge representation error, not temperature forecast or independent station accuracy.",
                        "Annual uses all months; DJF/JJA use calendar month, not contiguous winter-year.",
                        "10m input covers 1.28km around reference only; the spatial scene uses 100m terrain.",
                        "Output is 1km, 64km square (approximately 45.3km maximum corner distance), not a 100km-radius model.",
                        "Some scenes may have incomplete native10m coverage; masks are explicit and coverage reported.",
                        "Prior experiments have inspected these geographic block groups; this is not a new untouched final test.",
                        "Output not constrained symmetric between swapped references; labels are symmetric OLS RMSE.",
                        "No automatic replacement of Station Lab model. Requires evaluated results and a map-inference adapter."],
                    checksums={name: sha256(folder / name) for name in FILES})
    write_json(folder / "footprint_manifest.json", manifest)
    from remote_senorge.train_footprint import load_data
    checked, _, _ = load_data(folder)
    for array in checked.values():
        array._mmap.close()
    package = args.output.with_suffix(".zip")
    pending = package.with_suffix(".pending.zip")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name in FILES + ["footprint_manifest.json"]:
            archive.write(folder / name, name)
    with zipfile.ZipFile(pending) as archive:
        if set(archive.namelist()) != set(FILES + ["footprint_manifest.json"]) or archive.testzip() is not None:
            raise ValueError("Footprint ZIP failed allowlist/CRC verification")
    digest = sha256(pending)
    pending.replace(package)
    write_json(cache / "complete.json", dict(status="complete", sha256=digest, bytes=package.stat().st_size))
    print(f"COMPLETE footprint package: {package}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "senorge_grid_pairs_1m_1996_2025_random")
    parser.add_argument("--terrain", type=Path, default=ROOT / "outputs" / "senorge_seasonal_terrain")
    parser.add_argument("--dtm10", type=Path, default=ROOT / "data" / "DTM10")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "senorge_footprint_training")
    parser.add_argument("--counts", nargs=3, type=int, default=[256, 64, 64], metavar=("TRAIN", "VALIDATION", "TEST"))
    args = parser.parse_args()
    with computation_lock(args.output):
        if args.output.with_suffix(".zip").exists():
            raise FileExistsError("Completed footprint ZIP exists; choose a different output")
        try:
            write_json(args.output / "status.json", dict(state="running"))
            prepare(args)
            write_json(args.output / "status.json", dict(state="complete"))
        except Exception as error:
            write_json(args.output / "status.json", dict(state="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
