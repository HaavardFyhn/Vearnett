import tempfile
import unittest
import json
from unittest.mock import Mock, patch
from concurrent.futures import Future, ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import LineString

from placement_gain import (BUFFER_QUAD_SEGS, PreparedBasins, basin_keys, calculate_chunk,
                            corridor_areas, gain_from_predictions, initialize_worker, save_arrays)
from placement_gain import placement_profile, export_geotiff, calculate


class PlacementGainTests(unittest.TestCase):
    def test_all_and_empty_baselines_reuse_full_model_without_mutating_the_profile(self):
        profile = dict(stations=pd.DataFrame(dict(name=["E6 Road", "Other"])),
                       baseline=np.array([1., 2., 3.]), cache_path=Path("full.joblib"), model=object())
        engine = SimpleNamespace(weather=SimpleNamespace(profile=Mock(return_value=profile)),
                                 cache={"x": np.arange(3)})
        full = placement_profile(engine, "all")
        self.assertIs(full["baseline"], profile["baseline"])
        self.assertEqual(full["network_excluded"], frozenset())
        empty = placement_profile(engine, "empty")
        np.testing.assert_array_equal(empty["baseline"], [.5, .5, .5])
        self.assertEqual(empty["network_excluded"], frozenset([0, 1]))
        self.assertEqual(empty["empty_reference_rmse"], .5)
        self.assertIsNone(empty["cache_path"])
        self.assertIs(empty["model"], profile["model"])
        np.testing.assert_array_equal(profile["baseline"], [1., 2., 3.])
        self.assertAlmostEqual(gain_from_predictions(empty["baseline"], [.1, .6, .4], [.1, .2, .3]), .07)
        for value in (0, -1, np.inf, np.nan, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                placement_profile(engine, "empty", empty_reference_rmse=value)

    def test_combined_job_keeps_original_output_and_rejects_changed_owner_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "outputs"
            out.mkdir()
            entry = dict(model="model.ubj", svv="svv.npz", profile="profile.joblib",
                         basin_features="features.npz", basins="basins.gpkg")
            manifest = dict(cell_m=500, profiles={"temperature:senorge30": entry,
                                                 "temperature:direct_p95": dict(entry, svv_met="baseline.npz")},
                            roads=dict(indexed_source="roads.gpkg"))
            (out / "portable_weather.json").write_text(json.dumps(manifest))
            for filename in [*entry.values(), "roads.gpkg", "baseline.npz"]:
                (root / filename).write_bytes(b"input")
            (out / "norway_grid_500m.gpkg").write_bytes(b"input")
            owners = root / "data" / "station_owners.json"
            owners.parent.mkdir()
            owners.write_text(json.dumps(dict(stations={"SN2": dict(owners=["MET.NO"])})))
            original = out / "placement_gain_500m_road200.npz"
            original.write_bytes(b"original result")
            stations = pd.DataFrame(dict(name=["E6 Road", "MET station", "Private station"]))
            profile = dict(stations=stations, network_excluded=frozenset([2]),
                           cache_path=root / "baseline.npz")
            engine = SimpleNamespace(grid=pd.DataFrame(dict(x=[0., 500., 1000.], y=[7000000.] * 3)),
                                     ensure_candidates=Mock(), resources=SimpleNamespace(close=Mock()))
            roads = SimpleNamespace(path=root / "roads.gpkg", metadata=dict(available=True),
                                    national_corridor_mask=Mock(return_value=np.ones(3, dtype=bool)))
            pool = Mock()
            def submit(function, start, stop):
                result = Future()
                result.set_result((start, np.arange(start, stop, dtype=np.float64)))
                return result
            pool.submit.side_effect = submit
            def prepare_arrays(path, *unused):
                path.mkdir()
                (path / "arrays.json").write_text("{}")
            args = SimpleNamespace(data_root=root, work_dir=root / "work", workers=2,
                                   station_network="svv-met", radius_km=100, benchmark=0, overwrite=True)
            with patch.dict("os.environ"), patch("station_app.HERE", root), \
                    patch("station_app.OUT", out), patch("station_app.ASSET_ROOT", root), \
                    patch("station_app.StationEngine", return_value=engine), \
                    patch("station_roads.RoadNetwork", return_value=roads), \
                    patch("placement_gain.placement_profile", return_value=profile), \
                    patch("placement_gain.corridor_areas", return_value=np.ones(3)), \
                    patch("placement_gain.save_arrays", side_effect=prepare_arrays) as arrays, \
                    patch("placement_gain.ProcessPoolExecutor") as executor, \
                    patch("placement_gain.export_geotiff"):
                executor.return_value.__enter__.return_value = pool
                calculate(args)
                output = out / "placement_gain_500m_road200_svv_met.npz"
                with np.load(output, allow_pickle=False) as saved:
                    self.assertEqual(str(saved["station_network"]), "svv-met")
                    self.assertEqual(float(saved["station_radius_m"]), 100000)
                    np.testing.assert_array_equal(saved["gain"], [0., 1., 2.])
                    provenance = json.loads(str(saved["provenance"]))
                    self.assertEqual(provenance["baseline_station_count"], 2)
                calculate(args)
                arrays.assert_called_once()
                self.assertEqual(pool.submit.call_count, 1)
                owners.write_text(json.dumps(dict(stations={"SN2": dict(owners=["OTHER"])})))
                with self.assertRaisesRegex(ValueError, "Beregningsgrunnlaget er endret"):
                    calculate(args)
                for name, count in (("empty", 0), ("all", 3)):
                    args.station_network = name
                    args.empty_reference_rmse = 5.
                    args.work_dir = root / ("work_" + name)
                    profile["network_excluded"] = (frozenset(range(3)) if name == "empty" else frozenset())
                    calculate(args)
                    with np.load(out / f"placement_gain_500m_road200_{name}.npz", allow_pickle=False) as saved:
                        self.assertEqual(str(saved["station_network"]), name)
                        self.assertEqual(json.loads(str(saved["provenance"]))["baseline_station_count"], count)
                        if name == "empty":
                            self.assertEqual(float(saved["empty_reference_rmse"]), 5.)
                    if name == "empty":
                        args.empty_reference_rmse = 6.
                        with self.assertRaisesRegex(ValueError, "Beregningsgrunnlaget er endret"):
                            calculate(args)
                stem = "placement_gain_500m_road200_direct_p95"
                empty_results = [(out / f"{stem}_empty.{suffix}") for suffix in ("npz", "tif")]
                for path in empty_results:
                    path.write_bytes(b"completed empty-network result")
                args.dataset = "temperature:direct_p95"
                args.station_network = "existing-networks"
                args.work_dir = root / "work_existing"
                pool.submit.reset_mock()
                def submit_existing(function, start, stop):
                    result = Future()
                    result.set_result((start, np.arange(start, stop)[:, None] + [10., 20., 30.]))
                    return result
                pool.submit.side_effect = submit_existing
                with patch("placement_gain.placement_profile", return_value=dict(profile, baseline=np.ones(3))) as profiles:
                    calculate(args)
                    self.assertEqual([call.args[1] for call in profiles.call_args_list], ["svv", "svv-met", "all"])
                    calculate(args)
                pool.submit.assert_called_once()
                for suffix, network, gain in (("", "svv", 10.), ("_svv_met", "svv-met", 20.), ("_all", "all", 30.)):
                    with np.load(out / f"{stem}{suffix}.npz", allow_pickle=False) as saved:
                        self.assertEqual(str(saved["station_network"]), network)
                        np.testing.assert_array_equal(saved["gain"], [gain, gain + 1, gain + 2])
                        self.assertNotIn("empty_reference_rmse", saved)
                for path in empty_results:
                    self.assertEqual(path.read_bytes(), b"completed empty-network result")
            self.assertEqual(original.read_bytes(), b"original result")

    def test_combined_baseline_matches_exact_subset_minimum_and_reuses_checkpoint(self):
        from station_weather import restricted_profile
        from test_station_app import FakeEngine
        stations = FakeEngine.grid.copy()
        stations["station_id"] = ["SN1", "SN2", "SN3"]
        stations["name"] = ["E6 Road", "MET station", "Private station"]
        model = SimpleNamespace(predict_pairs=lambda source, target, sources, targets:
            np.array([[3., 4.], [2., 5.], [1., 1.]], dtype=np.float32)[sources, targets])
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "profile.joblib"
            source_path.write_bytes(b"baseline")
            profile = dict(id="temperature:senorge30", phenomenon="temperature", kind="senorge30",
                cell_m=500, cache_path=source_path, stations=stations, model=model,
                baseline=np.array([1., 1.]), second=np.array([2., 4.]), winner=np.array([2, 2]))
            engine = SimpleNamespace(weather=SimpleNamespace(profile=lambda dataset: profile),
                cache={"x": np.array([0., 500.])},
                station_owner_catalog=lambda: dict(stations={"SN2": dict(owners=["MET.NO"])}))
            with patch("station_weather.make_cache", side_effect=lambda frame: {"x": frame.x.to_numpy()}):
                result = placement_profile(engine, "svv-met")
                np.testing.assert_array_equal(result["baseline"], [2., 4.])
                np.testing.assert_array_equal(result["second"], [3., 5.])
                self.assertEqual(result["network_excluded"], frozenset([2]))
                with patch("station_weather.exact_network", side_effect=AssertionError("Must reuse prepared baseline")):
                    reused = placement_profile(engine, "svv-met")
                np.testing.assert_array_equal(reused["baseline"], result["baseline"])

    def test_original_network_still_uses_saved_svv_baseline(self):
        profile = {"baseline": np.array([3.])}
        engine = SimpleNamespace(weather=SimpleNamespace(svv_profile=Mock(return_value=profile)))
        self.assertIs(placement_profile(engine, "svv"), profile)
        engine.weather.svv_profile.assert_called_once_with("temperature:senorge30")

    def test_gis_export_identifies_combined_baseline_and_preserves_radius(self):
        import rasterio
        from rasterio.transform import from_origin
        engine = SimpleNamespace(raster_shape=(1, 2), rows=np.array([0, 0]), columns=np.array([0, 1]),
                                 transform=from_origin(250000, 6600000, 500, 500))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gain.tif"
            export_geotiff(path, engine, np.array([1., 2.]), 100000, "svv-met")
            with rasterio.open(path) as image:
                self.assertEqual(image.tags()["baseline"], "SVV+MET.NO")
                self.assertEqual(image.tags()["station_network"], "svv-met")
                self.assertEqual(image.tags()["station_radius_m"], "100000")
                np.testing.assert_array_equal(image.read(1), [[1., 2.]])
            export_geotiff(path, engine, np.array([3., 4.]), 100000, "empty", empty_reference_rmse=5.)
            with rasterio.open(path) as image:
                self.assertEqual(image.tags()["empty_reference_rmse"], "5.0")
                self.assertEqual(image.tags()["station_network"], "empty")
                self.assertIn("assumed", image.tags()["baseline"])
            export_geotiff(path, engine, np.array([1., 2.]), 100000, "all")
            with rasterio.open(path) as image:
                self.assertEqual(image.tags()["station_network"], "all")
                self.assertNotIn("empty_reference_rmse", image.tags())

    def test_gain_weights_only_positive_reductions_by_actual_road_area(self):
        baseline = np.array([3., 2., 1.])
        predicted = np.array([1., 3., .5])
        areas = np.array([.1, .2, .05])
        self.assertAlmostEqual(gain_from_predictions(baseline, predicted, areas), .225)
        with self.assertRaises(ValueError):
            gain_from_predictions(baseline, [1., np.nan, 0.], areas)

    def test_corridor_intersection_unions_overlapping_roads_across_tiles(self):
        grid = pd.DataFrame({"x": [0., 500., 1000.], "y": [7000000.] * 3})
        lines = [LineString([(0, 6999700), (0, 7000300)]),
                 LineString([(0, 6999700), (0, 7000300)]),
                 LineString([(600, 6999700), (600, 7000300)])]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roads.gpkg"
            gpd.GeoDataFrame(geometry=lines, crs=25833).to_file(path, layer="roads", driver="GPKG")
            roads = SimpleNamespace(path=path, table="roads")
            actual = corridor_areas(grid, roads, np.ones(3, dtype=bool), tile_m=500)
        buffer = shapely.union_all(shapely.buffer(np.array(lines, dtype=object), 200,
                                                  quad_segs=BUFFER_QUAD_SEGS))
        squares = shapely.box(grid.x - 250, grid.y - 250, grid.x + 250, grid.y + 250)
        expected = shapely.area(shapely.intersection(squares, buffer)) / 1e6
        np.testing.assert_allclose(actual, expected)
        self.assertGreater(actual[0], 0)
        self.assertLess(actual[0], .25)
        self.assertAlmostEqual(actual[2], .025)

    def test_prepared_basins_match_grid_cell_lookup(self):
        x = np.array([0., 500., 1500.])
        y = np.array([7000000.] * 3)
        keys = basin_keys(x, y)
        order = np.argsort(keys)
        values = np.arange(12).reshape(3, 4)
        lookup = PreparedBasins(keys[order], values[order])
        np.testing.assert_array_equal(lookup.lookup(x, y), values)
        with self.assertRaises(ValueError):
            lookup.lookup(np.array([9000.]), np.array([7000000.]))

    def test_spawned_workers_match_original_pair_model(self):
        from greedy_400_stations_pruned import CACHE_KEYS, make_cache, predict, subset_cache
        from station_senorge import PAIR_FEATURES
        from xgboost import XGBRegressor

        grid = pd.DataFrame({key: np.array([1., 2., 3.]) for key in CACHE_KEYS})
        grid["x"] = [0., 500., 1000.]
        grid["y"] = [7000000.] * 3
        grid["objtype_code"] = [0, 0, 1]
        cache = make_cache(grid)
        class Basins:
            def lookup(self, x, y):
                return np.column_stack((x * 0, x * 0, x * 0, x * 0))
        profile = {"baseline": np.array([4., 5., 6.]), "model": SimpleNamespace(basins=Basins())}
        model = XGBRegressor(n_estimators=2, max_depth=2, n_jobs=1)
        model.fit(np.zeros((3, len(PAIR_FEATURES)), dtype=np.float32), np.array([1., 2., 3.]))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / "model.ubj"
            model.save_model(path)
            save_arrays(folder, SimpleNamespace(cache=cache), profile, np.array([.1, 0., .2]))
            initialize_worker(folder, path)
            from placement_gain import WORKER
            expected = []
            for index in range(3):
                prediction = predict(WORKER["model"], subset_cache(cache, np.array([index])),
                                     cache, np.array([0, 2]))[0]
                expected.append(gain_from_predictions([4., 6.], prediction, [.1, .2]))
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize_worker, initargs=(folder, path)) as pool:
                results = list(pool.map(calculate_chunk, [0, 2], [2, 3]))
            actual = np.concatenate([gains for _, gains in results])
            np.testing.assert_allclose(actual, expected)
            WORKER.clear()

            baselines = np.array([[4., 5., 6.], [3., 4., 5.], [2., 3., 4.], [.5, .5, .5]])
            save_arrays(folder, SimpleNamespace(cache=cache), dict(profile, baseline=baselines),
                        np.array([.1, 0., .2]))
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize_worker, initargs=(folder, path)) as pool:
                results = list(pool.map(calculate_chunk, [0, 2], [2, 3]))
            shared = np.concatenate([gains for _, gains in results])
            initialize_worker(folder, path)
            for index in range(3):
                prediction = predict(WORKER["model"], subset_cache(cache, np.array([index])),
                                     cache, np.array([0, 2]))[0]
                np.testing.assert_allclose(shared[index], [
                    gain_from_predictions(baseline[[0, 2]], prediction, [.1, .2]) for baseline in baselines
                ])
            WORKER.clear()
            save_arrays(folder, SimpleNamespace(cache=cache), profile, np.array([.1, 0., .2]))

            initialize_worker(folder, path, radius_m=500)
            actual = calculate_chunk(0, 3)[1]
            expected_cutoff = []
            for index in range(3):
                ids = np.array([0, 2])
                ids = ids[np.abs(grid.x.to_numpy()[ids] - grid.x.iloc[index]) <= 500]
                prediction = predict(WORKER["model"], subset_cache(cache, np.array([index])),
                                     cache, ids)[0]
                expected_cutoff.append(gain_from_predictions(
                    profile["baseline"][ids], prediction, np.array([.1, 0., .2])[ids]))
            np.testing.assert_allclose(actual, expected_cutoff)
            initialize_worker(folder, path, radius_m=1)
            self.assertEqual(calculate_chunk(1, 2)[1][0], 0)
            WORKER.clear()


if __name__ == "__main__":
    unittest.main()
