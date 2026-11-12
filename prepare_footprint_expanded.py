"""Prepare many unique reference scenes using shared terrain and streamed weather."""
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import train_senorge_grid_pairs as training
import netCDF4
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from prepare_footprint_training import fine_mosaic
from precompute_frost import computation_lock
from remote_senorge.footprint_atlas import ABSOLUTE_FIELDS, FEATURE_FIELDS, SPATIAL_FIELDS
from remote_senorge.footprint_model import TARGETS
from remote_senorge.train_remote import write_json, sha256


ROOT = Path(__file__).resolve().parent
FILES = ["grid_table.npy", "references.npz", "terrain_height.npy", "terrain_valid.npy",
         "spatial10.npy", "labels.npy", "label_mask.npy", "selected_dates.csv",
         "footprint_atlas.py", "footprint_model.py", "train_footprint.py", "train_remote.py",
         "requirements.txt", "requirements-seasonal.txt", "FOOTPRINT_EXPANDED_README.md",
         "test_footprint.py", "test_footprint_atlas.py"]


def expanded_geometry(grid, counts=(10000, 1000, 1000), seed=314159, block_size=150000):
    xy = grid[["x", "y"]].to_numpy(dtype=float)
    if not np.allclose(xy, np.round(xy/1000)*1000):
        raise ValueError("Expected integer-kilometre terrain centres")
    keys = np.round(xy/1000).astype(np.int64)
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    if len(lookup) != len(grid):
        raise ValueError("Duplicate terrain centres")
    blocks, inverse = np.unique(np.floor(xy/block_size).astype(np.int64), axis=0, return_inverse=True)
    position = xy % block_size
    interior = ((position >= 41000) & (position <= block_size-41000)).all(axis=1)
    capacity = np.bincount(inverse[interior], minlength=len(blocks))
    eligible = np.flatnonzero(capacity >= 2)
    if len(eligible) < 12:
        raise ValueError("Need at least twelve occupied scene-safe regional blocks")
    rng = np.random.default_rng(seed)
    # Interleave regional latitude strata so each split samples more than one end of Norway.
    ordered = eligible[np.lexsort((blocks[eligible, 0], blocks[eligible, 1]))]
    group = np.full(len(blocks), -1, dtype=np.int8)
    for band in np.array_split(ordered, max(1, len(ordered)//6)):
        perm = rng.permutation(band)
        if len(perm) < 3:
            raise ValueError("Insufficient regions within latitude stratum")
        nheld = max(1, int(round(len(perm)*.2)))
        group[perm[:nheld]] = 2
        group[perm[nheld:2*nheld]] = 1
        group[perm[2*nheld:]] = 0
    refs, split = [], []
    for part, requested in enumerate(counts):
        choices = np.flatnonzero(interior & (group[inverse] == part))
        if requested < 2 or len(choices) < requested:
            raise ValueError(f"Only {len(choices)} unique eligible references for split {part}; requested {requested}")
        # 5km spatial strata first; use remaining cells only after all strata had a turn.
        strata = np.floor(xy[choices]/5000).astype(np.int64)
        _, sub = np.unique(strata, axis=0, return_inverse=True)
        buckets = [list(rng.permutation(choices[sub == j])) for j in np.unique(sub)]
        order = rng.permutation(len(buckets))
        chosen = []
        while len(chosen) < requested:
            for j in order:
                if buckets[j] and len(chosen) < requested:
                    chosen.append(buckets[j].pop())
        refs.extend(chosen)
        split.extend([part]*requested)
    refs, split = np.asarray(refs, np.int32), np.asarray(split, np.int8)
    mapping = np.full((len(refs), 64, 64), -1, np.int32)
    offsets = [(r, c, c-32, 32-r) for r in range(64) for c in range(64)]
    for i, ref in enumerate(refs):
        x, y = keys[ref]
        for r, c, dx, dy in offsets:
            mapping[i, r, c] = lookup.get((x+dx, y+dy), -1)
    report = {}
    for part, name in enumerate(("train", "validation", "test")):
        selected = refs[split == part]
        mapped = mapping[split == part]
        targets, times = np.unique(mapped[mapped >= 0], return_counts=True)
        nearest = training.cKDTree(xy[selected]).query(xy[selected], k=2)[0][:, 1]
        report[name] = dict(references=len(selected), eligible_references=int(
            (interior & (group[inverse] == part)).sum()),
            regional_blocks=len(np.unique(inverse[selected])),
            occupied_5km_strata=len(np.unique(np.floor(xy[selected]/5000).astype(int), axis=0)),
            minimum_reference_spacing_m=float(nearest.min()),
            median_nearest_reference_spacing_m=float(np.median(nearest)),
            unique_output_cells=len(targets), mean_output_cell_reuse=float(times.mean()),
            supervised_pairs=int(times.sum()),
            region_coordinates=blocks[np.unique(inverse[selected])].tolist())
    return dict(refs=refs, split=split, mapping=mapping, xy=xy[refs]), report


def fine_descriptors(raw):
    """Physical 10m summaries in a 1km cell; missing cells never imply sea level."""
    data = np.asarray(raw.data, dtype=float)
    valid = ~np.ma.getmaskarray(raw) & np.isfinite(data)
    result = np.zeros(len(SPATIAL_FIELDS), np.float32)
    result[0] = valid.mean()
    if not valid.any():
        return result
    values = data[valid]
    mean = values.mean()
    result[1:3] = [values.std()/1000, (values.max()-values.min())/1000]
    east_valid = valid[:, 1:] & valid[:, :-1]
    north_valid = valid[1:] & valid[:-1]
    east = np.diff(data, axis=1)/10
    north = np.diff(data, axis=0)/10
    result[4] = np.abs(east[east_valid]).mean() if east_valid.any() else 0
    result[5] = np.abs(north[north_valid]).mean() if north_valid.any() else 0
    joint = east_valid[:-1] & north_valid[:, :-1]
    slope = np.hypot(east[:-1], north[:, :-1])
    result[3] = slope[joint].mean() if joint.any() else 0
    h, w = data.shape
    for i, (rs, cs) in enumerate(((slice(0, h//2), slice(0, w//2)),
                                 (slice(0, h//2), slice(w//2, w)),
                                 (slice(h//2, h), slice(0, w//2)),
                                 (slice(h//2, h), slice(w//2, w)))):
        keep = valid[rs, cs]
        if keep.any():
            result[6+i] = (data[rs, cs][keep].mean()-mean)/1000
    result[10] = (values < mean-10).mean()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite native10m descriptor")
    return result


def open_array(stack, path, dtype, shape):
    exists = path.exists()
    if not exists and shutil.disk_usage(path.parent).free < np.prod(shape, dtype=np.int64)*np.dtype(dtype).itemsize+4*1024**3:
        raise OSError(f"Insufficient free space for {path.name} plus 4 GiB reserve")
    value = np.lib.format.open_memmap(path, mode="r+" if exists else "w+", dtype=dtype, shape=shape)
    stack.callback(value._mmap.close)
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"Unexpected checkpoint array {path.name}")
    if not exists:
        value[:] = 0
        value.flush()
    return value


def build_terrain(args, grid, folder, cache):
    xy = grid[["x", "y"]].to_numpy()
    x0 = int(xy[:, 0].min())-41000
    y0 = int(xy[:, 1].max())+41000
    width = int((xy[:, 0].max()-x0)/1000)+42
    height = int((y0-xy[:, 1].min())/1000)+42
    origin = [x0, y0]
    with ExitStack() as stack:
        terrain = open_array(stack, folder / "terrain_height.npy", np.float32, (height*10, width*10))
        valid = open_array(stack, folder / "terrain_valid.npy", np.uint8, terrain.shape)
        spatial = open_array(stack, folder / "spatial10.npy", np.float16, (len(SPATIAL_FIELDS), height, width))
        state = cache / "atlas_complete.json"
        if not state.exists():
            with rasterio.open(args.terrain / "cache" / "dem_100m.tif") as src:
                for start in range(0, terrain.shape[0], 256):
                    stop = min(start+256, terrain.shape[0])
                    left, right = x0-500, x0-500+width*1000
                    top, bottom = y0+500-start*100, y0+500-stop*100
                    raw = src.read(1, window=from_bounds(left, bottom, right, top, src.transform),
                                   out_shape=(stop-start, width*10), boundless=True, masked=True,
                                   out_dtype="float32", resampling=Resampling.average)
                    mask = ~np.ma.getmaskarray(raw) & np.isfinite(raw.data)
                    terrain[start:stop] = np.where(mask, raw.data/1000, 0)
                    valid[start:stop] = mask
            terrain.flush()
            valid.flush()
            write_json(state, dict(complete=True, origin=origin))
        fine = fine_mosaic(args.dtm10, cache)
        progress = cache / "fine_progress.json"
        completed = json.loads(progress.read_text())["completed"] if progress.exists() else 0
        with rasterio.open(fine) as src:
            for start in range(completed, len(grid), 256):
                stop = min(start+256, len(grid))
                for i in range(start, stop):
                    x, y = xy[i]
                    raw = src.read(1, window=from_bounds(x-500, y-500, x+500, y+500, src.transform),
                                   out_shape=(100, 100), boundless=True, masked=True, out_dtype="float32",
                                   resampling=Resampling.average)
                    row, col = int(round((y0-y)/1000)), int(round((x-x0)/1000))
                    spatial[:, row, col] = fine_descriptors(raw)
                spatial.flush()
                write_json(progress, dict(completed=stop, cells=len(grid)))
                if start % 4096 == 0 or stop == len(grid):
                    print(f"Native10m spatial descriptors: {stop:,}/{len(grid):,} cells", flush=True)
        stats = dict(mean_native10m_valid_fraction=float(spatial[0].sum(dtype=float)/len(grid)),
                     complete_grid_cells=int((spatial[0] == 1).sum()),
                     partial_grid_cells=int(((spatial[0] > 0) & (spatial[0] < 1)).sum()))
        if stats["mean_native10m_valid_fraction"] == 0:
            raise ValueError("No land-grid cell has native10m terrain coverage")
    return origin, stats


def build_weather(grid, plan, archives, cache):
    path = cache / "sampled_weather.npy"
    with ExitStack() as stack:
        weather = open_array(stack, path, np.float32, (2, len(grid), len(plan)))
        progress = cache / "weather_progress.json"
        completed = json.loads(progress.read_text())["completed_years"] if progress.exists() else 0
        rows, cols = grid.nc_row.to_numpy(), grid.nc_col.to_numpy()
        for index, archive in enumerate(archives):
            if index < completed:
                continue
            year = int(archive.stem.split("_")[-1])
            selected = plan.loc[plan.year == year]
            indices = selected.index.to_numpy()
            print(f"Sampled weather atlas: {year}, {len(selected)} dates, both tg/tn", flush=True)
            with netCDF4.Dataset(archive) as ds:
                dates = netCDF4.num2date(ds["time"][:], ds["time"].units,
                                        getattr(ds["time"], "calendar", "standard"))
                for record in selected.itertuples():
                    decoded = dates[record.day_index]
                    if f"{decoded.year:04}-{decoded.month:02}-{decoded.day:02}" != record.date:
                        raise ValueError("Date selection changed")
                xs, ys = np.asarray(ds["X"][:]), np.asarray(ds["Y"][:])
                if not np.allclose(np.diff(xs), 1000) or not np.allclose(abs(np.diff(ys)), 1000):
                    raise ValueError("Unexpected seNorge resolution")
                if ((cols < 0) | (cols >= len(xs)) | (rows < 0) | (rows >= len(ys))).any():
                    raise ValueError("Weather mapping outside source grid")
                if (np.abs(xs[cols]-grid.x.to_numpy()) > 1000).any() or (
                        np.abs(ys[rows]-grid.y.to_numpy()) > 1000).any():
                    raise ValueError("Weather mapping changed")
                for variable_index, variable in enumerate(("tg", "tn")):
                    var = ds[variable]
                    if var.units != "Celsius" or var.dimensions != ("time", "Y", "X"):
                        raise ValueError("Unexpected seNorge variable")
                    values = np.empty((len(grid), len(selected)), np.float32)
                    for j, day in enumerate(selected.day_index):
                        slab = np.ma.masked_invalid(var[day]).filled(np.nan)
                        values[:, j] = slab[rows, cols]
                    weather[variable_index][:, indices] = values
            weather.flush()
            write_json(progress, dict(completed_years=index+1, years=len(archives)))
    return path


def moments(reference, targets):
    valid = np.isfinite(targets) & np.isfinite(reference)[None]
    a = np.where(valid, reference[None], 0).astype(np.float64)
    b = np.where(valid, targets, 0).astype(np.float64)
    return np.array([valid.sum(axis=1), a.sum(axis=1), b.sum(axis=1),
                     (a*a).sum(axis=1), (b*b).sum(axis=1), (a*b).sum(axis=1)])


def label_chunk(weather, reference, targets, plan, output, mask):
    months = pd.to_datetime(plan.date).dt.month.to_numpy()
    for target in range(6):
        variable = target % 2
        season = np.ones(len(plan), bool) if target < 2 else np.isin(
            months, [12, 1, 2] if target < 4 else [6, 7, 8])
        fit = np.flatnonzero(season & (plan.partition.to_numpy() == 0))
        evaluation = np.flatnonzero(season & (plan.partition.to_numpy() == 1))
        for start in range(0, len(targets), 256):
            ids = targets[start:start+256]
            a = weather[variable, reference]
            b = weather[variable, ids]
            tr = moments(a[fit], b[:, fit])
            te = moments(a[evaluation], b[:, evaluation])
            result, _ = training.regression_labels(tr, te)
            good = np.isfinite(result) & (tr[0] >= max(60, len(fit)*.9)) & (
                te[0] >= max(20, len(evaluation)*.9))
            output[target, start:start+len(ids)] = np.where(good, result, 0)
            mask[target, start:start+len(ids)] = good


def build_labels(geo, plan, weather_path, folder, cache):
    with ExitStack() as stack:
        shape = (len(geo["refs"]), 6, 64, 64)
        labels = open_array(stack, folder / "labels.npy", np.float32, shape)
        masks = open_array(stack, folder / "label_mask.npy", np.bool_, shape)
        weather = np.load(weather_path, mmap_mode="r", allow_pickle=False)
        stack.callback(weather._mmap.close)
        progress = cache / "label_progress.json"
        completed = json.loads(progress.read_text())["completed"] if progress.exists() else 0
        for i in range(completed, len(geo["refs"])):
            keep = geo["mapping"][i] >= 0
            targets = geo["mapping"][i][keep]
            output, mask = np.zeros((6, len(targets)), np.float32), np.zeros((6, len(targets)), bool)
            label_chunk(weather, int(geo["refs"][i]), targets, plan, output, mask)
            labels[i][:, keep], masks[i][:, keep] = output, mask
            labels.flush()
            masks.flush()
            write_json(progress, dict(completed=i+1, references=len(geo["refs"])))
            if i % 16 == 0 or i+1 == len(geo["refs"]):
                print(f"Dense six-target label maps: {i+1:,}/{len(geo['refs']):,}", flush=True)
        counts = {name: int(masks[:, i].sum()) for i, name in enumerate(TARGETS)}
    return counts


def prepare(args):
    folder, cache = args.output, args.output / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    grid = pyogrio.read_dataframe(args.source / "grid_features.gpkg")
    plan = pd.read_csv(args.source / "selected_dates.csv")
    if len(plan) != 3600 or set(plan.year) != set(range(1996, 2026)):
        raise ValueError("Expected the original 3600-date plan")
    strata = plan.groupby(["year", pd.to_datetime(plan.date).dt.month, "partition"]).size()
    if len(strata) != 720 or not (strata.xs(0, level=2) == 8).all() or not (strata.xs(1, level=2) == 2).all():
        raise ValueError("Date plan must retain the original 8/2 monthly partition")
    config = json.loads((args.source / "config.json").read_text())
    archives = [Path(record["path"]) for record in config["archives"]]
    settings = dict(schema=2, seed=314159, block_size_m=150000, counts=args.counts,
                    preparer_sha256=sha256(Path(__file__)),
                    spatial_fields=SPATIAL_FIELDS, endpoint_fields=FEATURE_FIELDS,
                    grid=training.fingerprint(args.source / "grid_features.gpkg"),
                    dates=training.fingerprint(args.source / "selected_dates.csv"),
                    dem=training.fingerprint(args.terrain / "cache" / "dem_100m.tif"),
                    fine=[training.fingerprint(path) for path in sorted(args.dtm10.glob("*.tif"))],
                    archives=config["archives"])
    settings = json.loads(json.dumps(settings))
    if not settings["fine"]:
        raise FileNotFoundError("No native10m terrain")
    for path, record in zip(archives, config["archives"]):
        if training.fingerprint(path) != record:
            raise ValueError(f"Weather source changed: {path}")
    settings_path = cache / "source_config.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Expanded footprint inputs/settings changed; choose another output")
    write_json(settings_path, settings)
    if shutil.disk_usage(folder).free < 20*1024**3:
        raise OSError("At least 20 GiB free required for weather atlas, labels and verified package")
    geo, report = expanded_geometry(grid, args.counts)
    print(json.dumps(report, indent=2), flush=True)
    np.savez_compressed(folder / "references.npz", **geo)
    table = grid.copy()
    if table.objtype.isna().any():
        raise ValueError("Missing grid object type")
    table["objtype_code"] = pd.Categorical(table.objtype).codes
    np.save(folder / "grid_table.npy", table[FEATURE_FIELDS].to_numpy(dtype=np.float64))
    write_json(folder / "geography_report.json", report)
    write_json(folder / "status.json", dict(state="running", stage="terrain_atlases"))
    origin, terrain_stats = build_terrain(args, grid, folder, cache)
    write_json(folder / "status.json", dict(state="running", stage="sampled_weather_atlas"))
    weather = build_weather(grid, plan, archives, cache)
    write_json(folder / "status.json", dict(state="running", stage="dense_six_target_labels"))
    labels = build_labels(geo, plan, weather, folder, cache)
    plan.to_csv(folder / "selected_dates.csv", index=False)
    manifest = dict(schema=2, architecture="reference-resunet-spatial10-absolute-v2", targets=TARGETS,
                    features=training.FEATURES+ABSOLUTE_FIELDS, absolute_features=ABSOLUTE_FIELDS,
                    spatial_fields=SPATIAL_FIELDS,
                    pixels=64, halo=8, atlas_origin=origin, block_size_m=150000,
                    context_resolution_m=100, native_descriptor_resolution_m=10,
                    source_period="1996-2025", selected_dates=3600,
                    crs="EPSG:25833", north_up=True, reference_pixel=[32, 32],
                    geography=report, native10m_coverage=terrain_stats, supervised_cells=labels,
                    limitations=[
                        "Historical seNorge1km representation RMSE, not temperature forecast or independent station accuracy.",
                        "Reference10m branch replaced by spatial descriptors of native10m terrain across all land cells in scene.",
                        "Descriptors are deterministic physical summaries, not a learned raw10m encoder.",
                        "Zero coverage channel means missing descriptors, not flat sea-level terrain.",
                        "64km output square,80km context; not a100km-radius model.",
                        "Redesigned150km region split seed314159; not directly comparable to previous test results.",
                        "Geography used in earlier experiments overlaps this new split; not a newly untouched final test.",
                        "Overlapping scenes within partitions are correlated; evaluate variability by regional block.",
                        "Fine descriptors computed on1km grid cells; do not imply10m weather predictions.",
                        "No automatic Station Lab replacement; require evaluated returned results."])
    package_dataset(folder, manifest)


def package_dataset(folder, manifest):
    cache = folder / "cache"
    manifest["feature_values"] = "31 channels in manifest order; elevation and coast distances in metres; training-only normalization."
    for name in FILES[8:]:
        shutil.copyfile(ROOT / "remote_senorge" / name, folder / name)
    manifest["checksums"] = {name: sha256(folder / name) for name in FILES}
    write_json(folder / "footprint_manifest.json", manifest)
    write_json(folder / "status.json", dict(state="running", stage="validating_and_packaging"))
    from remote_senorge.footprint_atlas import AtlasDataset
    checked = AtlasDataset(folder)
    checked.close()
    package = folder.with_suffix(".zip")
    pending = package.with_suffix(".pending.zip")
    if shutil.disk_usage(folder).free < sum((folder / name).stat().st_size for name in FILES)+4*1024**3:
        raise OSError("Insufficient free space for verified ZIP plus 4 GiB reserve")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name in FILES + ["geography_report.json", "footprint_manifest.json"]:
            archive.write(folder / name, name)
    with zipfile.ZipFile(pending) as archive:
        if set(archive.namelist()) != set(FILES + ["geography_report.json", "footprint_manifest.json"]) or archive.testzip() is not None:
            raise ValueError("Expanded footprint archive failed integrity/allowlist check")
    digest = sha256(pending)
    pending.replace(package)
    write_json(cache / "complete.json", dict(status="complete", sha256=digest, bytes=package.stat().st_size))
    print(f"COMPLETE expanded footprint package: {package}", flush=True)


def repackage(folder):
    """Expose stored absolute values without recomputing any scientific arrays."""
    receipt = folder / "cache" / "complete.json"
    if not receipt.exists():
        raise FileNotFoundError("Repackaging requires a completed dataset receipt")
    completed = json.loads(receipt.read_text())
    if sha256(folder.with_suffix(".zip")) != completed["sha256"]:
        raise ValueError("Existing completed ZIP checksum mismatch")
    from remote_senorge.footprint_atlas import AtlasDataset
    checked = AtlasDataset(folder)
    manifest = checked.manifest
    checked.close()
    if manifest.get("features") not in (training.FEATURES, training.FEATURES+ABSOLUTE_FIELDS):
        raise ValueError("Existing endpoint feature order does not match the supported dataset")
    manifest["architecture"] = "reference-resunet-spatial10-absolute-v2"
    manifest["features"] = training.FEATURES+ABSOLUTE_FIELDS
    manifest["absolute_features"] = ABSOLUTE_FIELDS
    package_dataset(folder, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "senorge_grid_pairs_1m_1996_2025_random")
    parser.add_argument("--terrain", type=Path, default=ROOT / "outputs" / "senorge_seasonal_terrain")
    parser.add_argument("--dtm10", type=Path, default=ROOT / "data" / "DTM10")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "senorge_footprint_expanded")
    parser.add_argument("--counts", nargs=3, type=int, default=[10000, 1000, 1000])
    parser.add_argument("--repackage", action="store_true",
                        help="Verify and update completed package to include absolute inputs; reuse all scientific arrays")
    args = parser.parse_args()
    with computation_lock(args.output):
        if args.output.with_suffix(".zip").exists() and not args.repackage:
            raise FileExistsError("Expanded footprint ZIP already complete")
        try:
            write_json(args.output / "status.json", dict(state="running"))
            if args.repackage:
                repackage(args.output)
            else:
                prepare(args)
            write_json(args.output / "status.json", dict(state="complete"))
        except Exception as error:
            write_json(args.output / "status.json", dict(state="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
