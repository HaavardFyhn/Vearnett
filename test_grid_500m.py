import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from grid_resolution import source_spacing, cell_policy, training_spacing, weather_folder
from station_senorge import station_cell_features, PAIR_FEATURES
from station_weather import WeatherRegistry
from greedy_400_stations_pruned import make_cache, TERRAIN
import prepare_remote_500m
import run_500m


class Grid500Tests(unittest.TestCase):
    def test_plain_startup_selects_500m_stage_and_explicit_1000_overrides(self):
        import station_app
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            stage = project / "outputs" / "portable_build" / "Remote_500m" / "data"
            (stage / "outputs").mkdir(parents=True)
            manifest = stage / "outputs" / "portable_weather.json"
            manifest.write_text(json.dumps(dict(cell_m=500)))
            for arguments, expected_root, spacing in (
                    ([], stage, 500), (["--cell-m", "500"], stage, 500),
                    (["--cell-m", "1000"], project, 1000)):
                engine = Mock()
                engine.weather.available = [dict(id="temperature:senorge30")]
                with patch.object(station_app, "ASSET_ROOT", project), \
                        patch.object(station_app, "HERE", project), \
                        patch.object(station_app, "OUT", project / "outputs"), \
                        patch("sys.argv", ["station_app.py", *arguments]), \
                        patch.object(station_app, "StationEngine", return_value=engine) as constructor, \
                        patch.object(station_app, "create_app", return_value=Mock(config={})), \
                        patch.object(station_app, "KuldehullLayer"), \
                        patch.object(station_app.socket, "socket") as sockets:
                    sockets.return_value.__enter__.return_value.connect_ex.return_value = 1
                    station_app.main()
                    self.assertEqual(station_app.HERE, expected_root)
                    self.assertEqual(constructor.call_args.kwargs["cell_m"], spacing)
                    engine.warm_weather_maps.assert_called_once_with(resolutions=(spacing,))

    def test_source_app_uses_returned_stage_with_current_assets(self):
        import station_app
        engine = Mock()
        engine.weather.available = [dict(id="temperature:senorge30")]
        app = Mock(config={})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "outputs").mkdir()
            (root / "outputs" / "portable_weather.json").write_text(json.dumps(dict(cell_m=500)))
            with patch.object(station_app, "HERE"), patch.object(station_app, "OUT"), \
                    patch("sys.argv", ["station_app.py", "--data-root", str(root)]), \
                    patch.object(station_app, "StationEngine", return_value=engine) as constructor, \
                    patch.object(station_app, "create_app", return_value=app), \
                    patch.object(station_app, "KuldehullLayer") as basins, \
                    patch.object(station_app.socket, "socket") as sockets:
                sockets.return_value.__enter__.return_value.connect_ex.return_value = 1
                station_app.main()
                self.assertEqual(station_app.HERE, root.resolve())
                self.assertEqual(station_app.OUT, root.resolve() / "outputs")
                self.assertNotEqual(station_app.ASSET_ROOT, root.resolve())
                self.assertEqual(constructor.call_args.kwargs["cell_m"], 500)
                engine.warm_weather_maps.assert_called_once_with(resolutions=(500,))
                basins.assert_called_once_with(root.resolve() / "outputs" /
                    "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg")
            with patch("sys.argv", ["station_app.py", "--data-root", str(root), "--cell-m", "1000"]), \
                    self.assertRaises(SystemExit):
                station_app.main()

    def test_runtime_hashing_does_not_import_remote_training_package(self):
        import builtins
        import hashlib
        from station_weather_maps import WeatherMapCache
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == "remote_senorge" or name.startswith("remote_senorge."):
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            surface = root / "surface.joblib"
            surface.write_bytes(b"500m surface")
            profile = dict(status=dict(state="ready"), baseline=np.array([1.]), cache_path=surface,
                           id="temperature:senorge30")
            with patch("builtins.__import__", side_effect=guarded_import):
                from file_hash import sha256
                self.assertEqual(sha256(surface), hashlib.sha256(surface.read_bytes()).hexdigest())
                self.assertIsNone(WeatherMapCache(root, renderer_version="test").get(
                    SimpleNamespace(cell_m=500), profile, 400, (0, 1, 2)))
            for name in ("station_weather.py", "station_weather_maps.py", "station_roads.py"):
                self.assertNotIn("from remote_senorge", (Path(__file__).parent / name).read_text(encoding="utf-8"))

    def test_existing_bundle_is_rejected_before_hashing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "code.zip"
            output.write_bytes(b"existing")
            with patch("prepare_remote_500m.sha256", side_effect=AssertionError("Must fail early")), \
                    self.assertRaisesRegex(FileExistsError, "--force"):
                prepare_remote_500m.prepare(output)
            self.assertEqual(output.read_bytes(), b"existing")

    def test_verify_reports_all_missing_size_and_hash_failures(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "size").write_bytes(b"x")
            (root / "hash").write_bytes(b"y")
            (root / "good").write_bytes(b"ok")
            manifest = dict(files=[
                dict(path="missing", bytes=1, sha256="unused"),
                dict(path="size", bytes=10, sha256="unused"),
                dict(path="hash", bytes=1, sha256="wrong"),
                dict(path="good", bytes=2, sha256=hashlib.sha256(b"ok").hexdigest())])
            (root / "remote_500m_inputs.json").write_text(json.dumps(manifest))
            with patch("run_500m.ROOT", root), self.assertRaises(ValueError) as failure:
                run_500m.verify_inputs()
            message = str(failure.exception)
            self.assertIn("missing: missing file", message)
            self.assertIn("size: size mismatch", message)
            self.assertIn("hash: SHA256 mismatch", message)
            self.assertNotIn("good:", message)

    def test_real_small_area_terrain_is_calculated_at_500m(self):
        import geopandas as gpd
        import pyogrio
        from shapely.geometry import box, LineString, Point
        from build_trainmap import build

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "n100.gdb"
            maritime = root / "maritime.gdb"
            land = box(0, 0, 2000, 2000)
            areas = gpd.GeoDataFrame(dict(objtype=["land", "Havflate"]),
                geometry=[land, box(-1000, -1000, 3000, 3000).difference(land)], crs=25833)
            contours = gpd.GeoDataFrame(dict(elevation=[0., 200.]),
                geometry=[LineString([(0, 0), (2000, 0)]), LineString([(0, 2000), (2000, 2000)])],
                crs=25833)
            heights = gpd.GeoDataFrame(dict(elevation=[50.]), geometry=[Point(1000, 1000)], crs=25833)
            baseline = gpd.GeoDataFrame(geometry=[LineString([(-1000, -1000), (-1000, 3000)])], crs=25833)
            for frame, path, layer in [(areas, source, "areas"), (contours, source, "contours"),
                                        (heights, source, "points"), (baseline, maritime, "baseline")]:
                pyogrio.write_dataframe(frame, path, layer=layer, driver="OpenFileGDB")
            output = build(output=root / "outputs", spacing_m=500, n100_gdb=source,
                maritime_gdb=maritime, area_layer="areas", contour_layer="contours",
                point_layer="points", baseline_layer="baseline")
            actual = pyogrio.read_dataframe(output)
            self.assertEqual(len(actual), 9)
            np.testing.assert_array_equal(np.diff(np.unique(actual.x)), [500., 500.])
            self.assertTrue(np.isfinite(actual[["elevation", "coast_dist_fjord", "coast_dist_ocean", *TERRAIN]]).all().all())
            self.assertGreater(actual.tpi_10km.nunique(), 1)

    def test_policies_keep_existing_one_km_behavior(self):
        self.assertEqual([source_spacing(value) for value in (500, 1000, 2500, 5000)],
                         [500, 1000, 1000, 1000])
        self.assertEqual(training_spacing(Path("norway_grid_500m.gpkg")), 500)
        self.assertEqual(cell_policy(500), "cell-500m-v1")
        self.assertEqual(weather_folder(Path("root"), 500).name, "weather_500m")
        for value in (250, 750):
            with self.assertRaises(ValueError):
                source_spacing(value)

    @staticmethod
    def grid():
        return pd.DataFrame(dict(x=[0., 500., 1000.], y=[0., 0., 0.],
            elevation=[100., 250., 500.], objtype=["land"] * 3, objtype_code=[0] * 3,
            coast_dist_fjord=[100.] * 3, coast_dist_ocean=[1000.] * 3,
            **{field: [1., 2., 3.] for field in TERRAIN}))

    def test_station_uses_new_half_km_cell_instead_of_one_km_cell(self):
        grid = self.grid()
        source = grid.iloc[[0]].assign(x=510., station_id="SN1", lon=10., lat=60.)
        actual = station_cell_features(source, grid)
        self.assertEqual(actual.x.iloc[0], 500.)
        self.assertEqual(actual.elevation.iloc[0], 250.)
        self.assertEqual(actual.lon.iloc[0], 10.)
        self.assertEqual(source.x.iloc[0], 510.)

    def test_registry_requires_separate_half_km_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for spacing in (500, 1000):
                folder = weather_folder(root, spacing) / "wind"
                folder.mkdir(parents=True)
                for suffix in ("_model.joblib", "_metadata.json", "_station_features.gpkg"):
                    (folder / ("wind_5y_prediction_rmse" + suffix)).touch()
                if spacing == 1000:
                    (folder / "ready.json").write_text("{}")
            self.assertFalse(WeatherRegistry(root, None, None, None, cell_m=500).available)
            with patch.dict(os.environ, {"WEATHER_MODEL_STAGING": "wind"}):
                registry = WeatherRegistry(root, None, None, None, cell_m=500)
            self.assertEqual(registry.available[0]["model_grid_m"], 500)
            self.assertIn("weather_500m", registry.available[0]["prefix"])
            (weather_folder(root, 500) / "wind" / "ready.json").write_text("{}")
            self.assertEqual(WeatherRegistry(root, None, None, None, cell_m=500).available[0]["model_grid_m"], 500)
            self.assertEqual(WeatherRegistry(root, None, None, None, cell_m=1000).available[0]["model_grid_m"], 1000)

    def test_temperature_uses_500m_terrain_and_transfer_stable_cache_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "outputs"
            prefix = output / "senorge_30yr_station_model"
            prefix.mkdir(parents=True)
            model_path = prefix / "xgboost_with_basins.ubj"
            feature_path = output / "temperature_5y_prediction_rmse_station_features.gpkg"
            basin_path = output / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg"
            basin_path.parent.mkdir()
            for path in (model_path, feature_path, basin_path, prefix / "grid_features.csv.gz",
                         output / "norway_grid_500m.gpkg"):
                path.touch()
            (prefix / "metadata.json").write_text(json.dumps(dict(
                features=PAIR_FEATURES, model_sha256="valid", basin_grid_sha256="valid")))
            grid = self.grid()
            features = grid.iloc[[0]].assign(station_id="SN1", x=510.)
            observations = pd.DataFrame([dict(station_id="SN1", name="E6 Test", lat=60., lon=10.,
                                               first_time="2021-01-01", last_time="2026-01-01")])
            def load():
                registry = WeatherRegistry(root, grid, make_cache(grid), pd.Index(["land"]), cell_m=500)
                registry.models[str(model_path)] = Mock(n_features_in_=len(PAIR_FEATURES))
                return registry.profile("temperature:senorge30")
            with patch("pyogrio.read_dataframe", return_value=features), \
                    patch("make_uncertainty_map_gif.load_grid_all_fields", return_value=grid) as reader, \
                    patch("file_hash.sha256", return_value="valid"), \
                    patch("station_weather.observation_stations",
                          return_value=(observations, "2021-01-01", "2026-01-01")):
                first = load()
                os.utime(model_path, (100, 100))
                os.utime(feature_path, (100, 100))
                os.utime(basin_path, (100, 100))
                second = load()
            self.assertEqual(first["source_grid_m"], 500)
            self.assertEqual(first["stations"].x.iloc[0], 500.)
            self.assertEqual(first["cache_path"], second["cache_path"])
            self.assertEqual(reader.call_args.args[0].name, "norway_grid_500m.gpkg")

    def test_parallel_commands_are_isolated_and_never_retrain_temperature(self):
        for phenomenon, metric in run_500m.MODELS:
            train, maps = run_500m.commands(phenomenon, metric)
            self.assertNotEqual(phenomenon, "temperature")
            self.assertIn("--skip-download", train)
            self.assertIn("--station-cell-inputs", train)
            self.assertEqual(train[train.index("--grid-spacing-m") + 1], "500")
            self.assertIn("weather_500m", train[train.index("--output-dir") + 1])
            self.assertEqual(maps[maps.index("--cell-m") + 1], "500")
        with patch("run_500m.os.cpu_count", return_value=4), self.assertRaises(SystemExit):
            run_500m.main(["--jobs", "2", "--threads", "4", "--dry-run"])

    def test_runner_orders_phases_and_resumes_without_repeating_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []
            def execute(command, title, threads, staging=None):
                calls.append((title, staging))
            args = SimpleNamespace(phase="all", jobs=2, threads=2, dry_run=False, stage_name="Remote_500m")
            with patch("run_500m.ROOT", root), patch("run_500m.verify_inputs"), \
                    patch("run_500m.execute", side_effect=execute), \
                    patch("pyogrio.read_dataframe", return_value=pd.DataFrame(dict(x=[0., 500.], y=[0., 0.]))):
                run_500m.run(args)
                self.assertEqual(calls[0][0], "terrain")
                titles = [item[0] for item in calls]
                self.assertLess(max(titles.index(name + "_train") for name, _ in run_500m.MODELS),
                                titles.index("temperature_maps"))
                self.assertEqual(titles[-2:], ["roads", "portable_stage"])
                self.assertEqual([staging for title, staging in calls if title.endswith("_maps") and
                                  title != "temperature_maps"], [name for name, _ in run_500m.MODELS])
                calls.clear()
                run_500m.run(args)
                self.assertEqual(calls, [])

    def test_source_bundle_closure_and_sanitized_config(self):
        root = Path(__file__).resolve().parent
        names = {path.relative_to(root).as_posix() for path in prepare_remote_500m.source_files(root)}
        for name in ("run_500m.py", "station_app.py", "station_weather.py", "information_gain.py",
                     "grid_resolution.py", "contour_terrain_metrics.py", "remote_senorge/train_remote.py"):
            self.assertIn(name, names)
        self.assertFalse(any(name.startswith("test_") for name in names))
        tree = ast.parse(prepare_remote_500m.sanitized_config(root / "pipeline_config.py"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in (
                    "FROST_CLIENT_ID", "FROST_CLIENT_ID_ENV", "FROST_CLIENT_SECRET"):
                self.assertIsNone(ast.literal_eval(node.value))


if __name__ == "__main__":
    unittest.main()
