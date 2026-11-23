"""Calculate temperature placement gain over the national 200 m road corridor."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from station_pross import EMPTY_REFERENCE_RMSE, PLACEMENT_SUFFIXES


DATASET = "temperature:senorge30"
CELL_M = 500
ROAD_DISTANCE_M = 200
BUFFER_QUAD_SEGS = 64
TARGET_BATCH = 50000
CHUNK_SIZE = 100


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    pending.replace(path)


def corridor_areas(grid, road_network, mask, tile_m=20000):
    """Area in km² of each grid cell intersecting the union of buffered roads."""
    import pyogrio
    import shapely

    xs = grid["x"].to_numpy(dtype=np.float64)
    ys = grid["y"].to_numpy(dtype=np.float64)
    indices = np.flatnonzero(mask)
    areas = np.zeros(len(xs), dtype=np.float64)
    if not len(indices):
        raise ValueError("Ingen terrengruter berører vegkorridoren")
    tiles = np.floor(np.column_stack((xs[indices], ys[indices])) / tile_m).astype(np.int64)
    _, groups = np.unique(tiles, axis=0, return_inverse=True)
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.diff(groups[order])) + 1
    batches = np.split(indices[order], boundaries)
    half = CELL_M / 2
    for number, selected in enumerate(batches, 1):
        xmin, xmax = xs[selected].min() - half, xs[selected].max() + half
        ymin, ymax = ys[selected].min() - half, ys[selected].max() + half
        bbox = (xmin - ROAD_DISTANCE_M, ymin - ROAD_DISTANCE_M,
                xmax + ROAD_DISTANCE_M, ymax + ROAD_DISTANCE_M)
        roads = pyogrio.read_dataframe(road_network.path, layer=road_network.table,
                                       columns=[], bbox=bbox)
        geometry = roads.geometry
        geometry = geometry[geometry.notna() & ~geometry.is_empty &
                            geometry.geom_type.isin(["LineString", "MultiLineString"])].to_numpy()
        if len(geometry):
            buffered = shapely.union_all(shapely.buffer(geometry, ROAD_DISTANCE_M,
                                                        quad_segs=BUFFER_QUAD_SEGS))
            squares = shapely.box(xs[selected] - half, ys[selected] - half,
                                   xs[selected] + half, ys[selected] + half)
            areas[selected] = shapely.area(shapely.intersection(squares, buffered)) / 1e6
        if number % 100 == 0 or number == len(batches):
            print(f"Road area: {number}/{len(batches)} tiles", flush=True)
    if not np.isfinite(areas).all() or (areas < 0).any() or (areas > .250001).any():
        raise ValueError("Ugyldig vegkorridorareal")
    if not np.any(areas > 0):
        raise ValueError("Vegkorridoren har ikke positivt areal")
    return areas


def gain_from_predictions(baseline, predictions, area_km2):
    baseline = np.asarray(baseline, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    area_km2 = np.asarray(area_km2, dtype=np.float64)
    if (baseline.shape != predictions.shape or baseline.shape != area_km2.shape or
            not np.isfinite(baseline).all() or not np.isfinite(predictions).all() or
            not np.isfinite(area_km2).all() or (baseline < 0).any() or
            (predictions < 0).any() or (area_km2 < 0).any()):
        raise ValueError("Ugyldige verdier for beregning av plasseringsgevinst")
    return float(np.dot(np.maximum(baseline - predictions, 0), area_km2))


def basin_keys(x, y):
    cells = np.floor(np.column_stack((x, y)) / 1000 + .5).astype(np.int64)
    if ((cells[:, 1] < 0) | (cells[:, 1] >= 100000)).any():
        raise ValueError("Terrengruten ligger utenfor det støttede området for bassengindeksen")
    return cells[:, 0] * 100000 + cells[:, 1]


class PreparedBasins:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    @staticmethod
    def cell_keys(x, y):
        return np.floor(np.column_stack((x, y)) / 1000 + .5).astype(np.int64)

    def lookup(self, x, y):
        keys = basin_keys(x, y)
        indexes = np.searchsorted(self.keys, keys)
        if (indexes >= len(self.keys)).any() or not np.array_equal(self.keys[indexes], keys):
            raise ValueError("Bassengegenskaper mangler for en terrengrute")
        return self.values[indexes]


WORKER = {}


def initialize_worker(directory, model_path, radius_m=100000):
    os.environ["WEATHER_CPU_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    from station_senorge import SeNorgePairModel
    from scipy.spatial import cKDTree

    directory = Path(directory)
    manifest = json.loads((directory / "arrays.json").read_text(encoding="utf-8"))
    cache = {key: np.load(directory / name, mmap_mode="r") for key, name in manifest["cache"].items()}
    basins = PreparedBasins(np.load(directory / "basin_keys.npy", mmap_mode="r"),
                            np.load(directory / "basin_values.npy", mmap_mode="r"))
    target_ids = np.load(directory / "target_ids.npy", mmap_mode="r")
    xs = cache.get("senorge_x", cache["x"])
    ys = cache.get("senorge_y", cache["y"])
    WORKER.update(cache=cache, model=SeNorgePairModel(model_path, basins),
                  target_ids=target_ids, radius_m=radius_m,
                  tree=cKDTree(np.column_stack((xs[target_ids], ys[target_ids]))),
                  baseline=np.load(directory / "baseline.npy", mmap_mode="r"),
                  area=np.load(directory / "area.npy", mmap_mode="r"))


def calculate_chunk(start, stop):
    from greedy_400_stations_pruned import predict, subset_cache

    cache, model = WORKER["cache"], WORKER["model"]
    target_ids, baseline, area = WORKER["target_ids"], WORKER["baseline"], WORKER["area"]
    gains = np.zeros((stop - start, *baseline.shape[:-1]), dtype=np.float64)
    for offset, index in enumerate(range(start, stop)):
        source = subset_cache(cache, np.array([index]))
        if WORKER["radius_m"]:
            x = cache.get("senorge_x", cache["x"])[index]
            y = cache.get("senorge_y", cache["y"])[index]
            nearby = np.asarray(WORKER["tree"].query_ball_point(
                [x, y], WORKER["radius_m"], return_sorted=True), dtype=np.int64)
        else:
            nearby = np.arange(len(target_ids))
        for first in range(0, len(nearby), TARGET_BATCH):
            selected = nearby[first:first + TARGET_BATCH]
            ids = target_ids[selected]
            prediction = predict(model, source, cache, ids)[0]
            if baseline.ndim == 1:
                gains[offset] += gain_from_predictions(baseline[selected], prediction, area[selected])
            else:
                for column, reference in enumerate(baseline):
                    gains[offset, column] += gain_from_predictions(reference[selected], prediction, area[selected])
    return start, gains


def save_arrays(directory, engine, profile, areas):
    directory.mkdir(parents=True, exist_ok=True)
    target_ids = np.flatnonzero(areas > 0)
    baseline = np.asarray(profile["baseline"], dtype=np.float32)[..., target_ids]
    if not np.isfinite(baseline).all() or (baseline < 0).any():
        raise ValueError("Grunnflaten har ugyldige verdier i vegkorridoren")
    cache = engine.cache
    xs = cache.get("senorge_x", cache["x"])
    ys = cache.get("senorge_y", cache["y"])
    keys, first = np.unique(basin_keys(xs, ys), return_index=True)
    basin_values = []
    for start in range(0, len(first), 50000):
        ids = first[start:start + 50000]
        basin_values.append(profile["model"].basins.lookup(xs[ids], ys[ids]))
    arrays = {"target_ids": target_ids, "baseline": baseline, "area": areas[target_ids],
              "basin_keys": keys, "basin_values": np.concatenate(basin_values)}
    for name, array in arrays.items():
        np.save(directory / f"{name}.npy", array, allow_pickle=False)
    files = {}
    for index, (key, array) in enumerate(cache.items()):
        name = f"cache_{index}.npy"
        np.save(directory / name, array, allow_pickle=False)
        files[key] = name
    (directory / "arrays.json").write_text(json.dumps({"cache": files}), encoding="utf-8")


def export_geotiff(path, engine, gains, radius_m, station_network="svv", empty_reference_rmse=None, *, dataset=DATASET):
    import rasterio

    pending = path.with_name(path.name + ".pending")
    values = np.full(engine.raster_shape, -9999, dtype=np.float32)
    values[engine.rows, engine.columns] = gains
    with rasterio.open(pending, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float32", crs="EPSG:25833", transform=engine.transform,
                       nodata=-9999, compress="deflate", tiled=True) as image:
        image.write(values, 1)
        baseline = {"svv": "SVV", "svv-met": "SVV+MET.NO", "all": "All temperature stations",
                    "empty": "Empty network (assumed reference)"}[station_network]
        image.update_tags(dataset=dataset, baseline=baseline,
                          station_network=station_network, road_distance_m=str(ROAD_DISTANCE_M),
                          unit="degree_Celsius_km2", buffer_quad_segs=str(BUFFER_QUAD_SEGS),
                          station_radius_m=str(radius_m))
        if empty_reference_rmse is not None:
            image.update_tags(empty_reference_rmse=str(empty_reference_rmse))
    pending.replace(path)


def placement_profile(engine, station_network, empty_reference_rmse: float | None = EMPTY_REFERENCE_RMSE, *, dataset=DATASET):
    if station_network == "svv":
        return engine.weather.svv_profile(dataset)
    if station_network not in PLACEMENT_SUFFIXES:
        raise ValueError("Ukjent stasjonsnett for plasseringsgevinst")
    from station_pross import svv_met_excluded
    from station_weather import restricted_profile

    profile = engine.weather.profile(dataset)
    if station_network == "all":
        return dict(profile, network_excluded=frozenset())
    if station_network == "empty":
        if empty_reference_rmse is None or not np.isfinite(empty_reference_rmse) or empty_reference_rmse <= 0:
            raise ValueError("Referansefeilen for tomt nett må være endelig og større enn null")
        return dict(profile, baseline=np.full(len(engine.cache["x"]), empty_reference_rmse, dtype=np.float32),
                    network_excluded=frozenset(range(len(profile["stations"]))), cache_path=None,
                    empty_reference_rmse=float(empty_reference_rmse))
    manifest = getattr(engine.weather, "portable_manifest", None)
    if manifest is not None and manifest["profiles"][dataset].get("svv_met"):
        return engine.weather.portable_reference(dataset, "svv_met")
    excluded = svv_met_excluded(profile["stations"], engine.station_owner_catalog())
    print(f"Preparing svv-met baseline: {len(profile['stations']) - len(excluded)} temperature stations",
          flush=True)
    result = restricted_profile(profile, engine.cache, excluded, build=True)
    if result is None:
        raise ValueError("Ingen temperaturstasjoner i Vegværstasjoner + MET-stasjoner")
    return result


def calculate(args):
    import multiprocessing
    import station_app
    from precompute_frost import computation_lock
    from station_roads import RoadNetwork
    from station_weather import DIRECT_P95

    for name in ("WEATHER_CPU_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if args.station_network == "svv-met":
        os.environ["WEATHER_CPU_THREADS"] = str(args.workers)
    root = args.data_root.resolve()
    dataset = getattr(args, "dataset", DATASET)
    if args.station_network in ("all-networks", "existing-networks"):
        networks = tuple(network for network in PLACEMENT_SUFFIXES
                         if args.station_network == "all-networks" or network != "empty")
    else:
        networks = (args.station_network,)
    shared = len(networks) > 1
    stem = "placement_gain_500m_road200" + ("_direct_p95" if dataset == DIRECT_P95 else "")
    outputs = [root / "outputs" / f"{stem}{PLACEMENT_SUFFIXES[network]}.npz" for network in networks]
    if not args.benchmark and not args.overwrite and any(
            path.exists() or path.with_suffix(".tif").exists() for path in outputs):
        raise FileExistsError("Resultatfil finnes allerede; bruk --overwrite for å erstatte den")
    manifest_path = root / "outputs" / "portable_weather.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["cell_m"] != CELL_M:
        raise ValueError("Datamappen må inneholde 500 m rutenett")
    entry = manifest["profiles"][dataset]
    model_path = root / entry["model"]
    station_app.HERE, station_app.OUT = root, root / "outputs"
    engine = station_app.StationEngine(root / "outputs" / "unused.npy", cell_m=CELL_M)
    try:
        engine.ensure_candidates()
        reference = args.empty_reference_rmse if "empty" in networks else None
        profiles = [placement_profile(engine, network, empty_reference_rmse=reference, dataset=dataset)
                    for network in networks]
        profile = profiles[0]
        from station_pross import pross_excluded
        counts = []
        for network, item in zip(networks, profiles):
            excluded = item.get("network_excluded", pross_excluded(item["stations"]))
            counts.append(len(item["stations"]) - len(excluded))
            print(f"Baseline: {network}; {counts[-1]} temperature stations", flush=True)
        if reference is not None:
            print(f"Empty-network reference: {reference:g} degrees C {profile.get('metric', 'RMSE')} (assumed, not measured)", flush=True)
        if profile.get("cache_path") is not None:
            print(f"Baseline cache: {profile['cache_path']}", flush=True)
        road_source = root / manifest["roads"]["indexed_source"]
        roads = RoadNetwork(road_source, cache_dir=root / "outputs" / "station_app_cache" / "roads",
                            portable_cache=True)
        if not roads.metadata["available"]:
            raise FileNotFoundError(f"Vegnett mangler: {road_source}")
        source_paths = [manifest_path, root / "outputs" / "norway_grid_500m.gpkg",
                        root / entry["profile" if args.station_network in ("all", "empty") else "svv"],
                        model_path, root / entry["basin_features"],
                        root / entry["basins"], roads.path, Path(__file__).resolve(),
                        Path(station_app.__file__).resolve()]
        for module in ("station_senorge.py", "station_weather.py", "station_roads.py",
                       "greedy_400_stations_pruned.py", "station_pross.py", "station_owners.py"):
            source_paths.append(Path(__file__).resolve().with_name(module))
        if shared:
            source_paths.extend(Path(item["cache_path"]) for item in profiles if item.get("cache_path"))
        if "svv-met" in networks and not entry.get("svv_met"):
            from station_owners import OWNER_FILE
            owner_path = next((base / OWNER_FILE for base in (station_app.ASSET_ROOT, root)
                               if (base / OWNER_FILE).is_file()), None)
            if owner_path is None:
                raise FileNotFoundError("Stasjonseiere mangler for Vegværstasjoner + MET-stasjoner")
            source_paths.extend([root / entry["profile"], owner_path, Path(profile["cache_path"])])
        signature = {str(path.resolve()): digest_file(path) for path in source_paths}
        signature.update(cell_m=CELL_M, distance_m=ROAD_DISTANCE_M, quad_segs=BUFFER_QUAD_SEGS,
                         dataset=dataset, station_network=args.station_network,
                         baseline_station_count=counts if shared else counts[0],
                         empty_reference_rmse=reference,
                         chunk_size=CHUNK_SIZE, radius_m=args.radius_km * 1000,
                         python=sys.version,
                         versions={name: importlib.metadata.version(name) for name in
                                   ("numpy", "pandas", "shapely", "pyogrio", "xgboost", "rasterio")})
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
        with computation_lock(work):
            metadata = work / "inputs.json"
            if metadata.exists():
                if json.loads(metadata.read_text(encoding="utf-8")) != signature:
                    raise ValueError("Beregningsgrunnlaget er endret; bruk en ny --work-dir")
            else:
                metadata.write_text(json.dumps(signature, indent=2), encoding="utf-8")
            arrays = work / "arrays"
            if not (arrays / "arrays.json").is_file():
                print("Calculating exact road corridor areas...", flush=True)
                mask = roads.national_corridor_mask(engine, ROAD_DISTANCE_M)
                areas = corridor_areas(engine.grid, roads, mask)
                prepared = dict(profile, baseline=np.stack([item["baseline"] for item in profiles])) if shared else profile
                save_arrays(arrays, engine, prepared, areas)
            count = len(engine.grid)
            if args.benchmark:
                indices = np.linspace(0, count - 1, min(args.benchmark, count), dtype=np.int64)
                print(f"Benchmarking {len(indices)} candidates with {args.workers} workers", flush=True)
                start_time = time.monotonic()
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=initialize_worker,
                                         initargs=(arrays, model_path, args.radius_km * 1000)) as pool:
                    results = list(pool.map(calculate_chunk, indices.tolist(), (indices + 1).tolist()))
                seconds = time.monotonic() - start_time
                estimate = seconds * count / len(results) / 3600
                print(f"Benchmark: {seconds:.1f} s; extrapolated national computation: {estimate:.1f} h "
                      "(approximate; excludes preparation and export)", flush=True)
                return
            chunks = work / "chunks"
            chunks.mkdir(exist_ok=True)
            jobs = []
            gain_shape = (len(networks),) if shared else ()
            gains = np.empty((count, *gain_shape), dtype=np.float64)
            for start in range(0, count, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, count)
                path = chunks / f"{start:09d}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as stored:
                        values = stored["gains"]
                    if values.shape != (stop - start, *gain_shape) or not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError(f"Ugyldig delresultat: {path}")
                    gains[start:stop] = values
                else:
                    jobs.append((start, stop))
            print(f"Candidates: {count}; remaining: {sum(stop-start for start, stop in jobs)}; "
                  f"workers: {args.workers}", flush=True)
            start_time = time.monotonic()
            done = 0
            remaining_total = sum(stop-start for start, stop in jobs)
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize_worker,
                                     initargs=(arrays, model_path, args.radius_km * 1000)) as pool:
                pending = {}
                iterator = iter(jobs)
                def submit_next():
                    job = next(iterator, None)
                    if job is not None:
                        pending[pool.submit(calculate_chunk, *job)] = job
                for _ in range(min(len(jobs), args.workers * 2)):
                    submit_next()
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        pending.pop(future)
                        start, values = future.result()
                        gains[start:start + len(values)] = values
                        atomic_npz(chunks / f"{start:09d}.npz", gains=values)
                        done += len(values)
                        elapsed = time.monotonic() - start_time
                        remaining = remaining_total - done
                        eta = elapsed * remaining / done / 3600
                        print(f"Calculated {done} candidates; {remaining} remaining; ETA {eta:.1f} h",
                              flush=True)
                        submit_next()
            for index, (network, output) in enumerate(zip(networks, outputs)):
                extra = dict(empty_reference_rmse=reference) if network == "empty" else {}
                values = gains[:, index] if shared else gains
                geotiff = output.with_suffix(".tif")
                export_geotiff(geotiff, engine, values, args.radius_km * 1000, network, dataset=dataset, **extra)
                provenance = dict(signature, station_network=network, baseline_station_count=counts[index],
                                  empty_reference_rmse=extra.get("empty_reference_rmse"))
                atomic_npz(output, x=engine.grid["x"].to_numpy(dtype=np.float64),
                           y=engine.grid["y"].to_numpy(dtype=np.float64), gain=values,
                           dataset=np.array(dataset), cell_m=np.array(CELL_M),
                           road_distance_m=np.array(ROAD_DISTANCE_M),
                           station_radius_m=np.array(args.radius_km * 1000),
                           station_network=np.array(network), **extra,
                           provenance=np.array(json.dumps(provenance, sort_keys=True)))
                print(f"Finished: {output}; GIS raster: {geotiff}", flush=True)
    finally:
        engine.resources.close()


def main(argv=None):
    from station_weather import DIRECT_P95

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--dataset", choices=[DATASET, DIRECT_P95], default=DATASET)
    parser.add_argument("--station-network", choices=[*PLACEMENT_SUFFIXES, "all-networks", "existing-networks"],
                        default="svv", help="Use one baseline, or share predictions across all networks or only non-empty networks")
    parser.add_argument("--empty-reference-rmse", type=float, default=EMPTY_REFERENCE_RMSE,
                        help="Assumed reference error in degrees C for an empty network (default: 0.5); not measured")
    parser.add_argument("--radius-km", type=float, default=100,
                        help="Candidate influence radius in km; 0 evaluates all road cells")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Benchmark this many evenly spaced candidates; do not write final results")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing final result files")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.station_network in ("all-networks", "existing-networks") and args.dataset != DIRECT_P95:
        parser.error(f"--station-network {args.station_network} requires prepared Direct P95 baselines")
    if not np.isfinite(args.empty_reference_rmse) or args.empty_reference_rmse <= 0:
        parser.error("--empty-reference-rmse must be finite and greater than zero")
    if args.benchmark < 0:
        parser.error("--benchmark must be non-negative")
    if not np.isfinite(args.radius_km) or args.radius_km < 0:
        parser.error("--radius-km must be finite and non-negative")
    calculate(args)


if __name__ == "__main__":
    main()
