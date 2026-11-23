from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import train_senorge_grid_pairs as training
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import box

from greedy_400_stations_pruned import FEATURES, TERRAIN, make_cache, predict
from station_senorge import (
    BasinLookup, SeNorgePairModel, PAIR_FEATURES, WARNING, p95_reference_values, station_cell_features,
)
from station_weather import paired_predictions, WeatherRegistry, exact_network
from test_station_app import FakeWeather, FakeWeatherEngine
from station_app import Scenario, create_app


class StationSeNorgeTests(unittest.TestCase):
    def frame(self):
        frame = pd.DataFrame(dict(x=[1000., 2000., 3000.], y=[1000.]*3,
                                  elevation=[1., 2., 3.], coast_dist_fjord=[10.]*3,
                                  coast_dist_ocean=[99999.1245123, 100328.122651, 100499.85231], objtype_code=[0]*3,
                                  basin_present=[0, 1, 1], basin_count=[0, 1, 2],
                                  basin_max_depth_m=[0., 4.123456789, 8.234567891],
                                  basin_volume_m3=[0., 200000.123456789, 600000.234567891]))
        for field in TERRAIN:
            frame[field] = [1., 3., 6.]
        return frame

    def test_p95_scores_rmse_winners_and_masks_unsupported_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            grid = self.frame()
            grid.to_csv(root / "grid.csv.gz", index=False)
            adapter = SeNorgePairModel.__new__(SeNorgePairModel)
            adapter.basins = BasinLookup(root / "grid.csv.gz", root / "missing.gpkg")
            adapter.model = Mock()
            adapter.model.predict.side_effect = lambda x: 2 + x[:, 0] / 1000
            cache = make_cache(grid)
            winners = np.array([0, 0, 0, -1])
            values = p95_reference_values(adapter, cache, cache, winners, 1500)
            np.testing.assert_array_equal(values, [0, 3, np.nan, np.nan])
            np.testing.assert_array_equal(adapter.model.predict.call_args.args[0][:, 0], [0, 1000])

    def test_p95_is_optional_when_no_reference_engine_is_installed(self):
        client = create_app(FakeWeatherEngine()).test_client()
        self.assertEqual(client.get("/api/p95/available").json, {"available": False})
        self.assertEqual(client.get("/api/p95/p95.bin").status_code, 404)

    def test_feature_parity_all_prediction_routes_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            grid = self.frame()
            path = Path(temp) / "grid.csv.gz"
            grid.to_csv(path, index=False)
            adapter = SeNorgePairModel.__new__(SeNorgePairModel)
            adapter.basins = BasinLookup(path, Path(temp) / "missing.gpkg")
            model = Mock()
            model.predict.side_effect = lambda x: x.sum(axis=1)
            adapter.model = model
            cache = make_cache(grid)
            first, second = np.array([0, 1]), np.array([1, 2])
            actual = adapter.predict_pairs(cache, cache, first, second)
            grid["objtype"] = "land"
            expected = training.pair_features(grid, np.column_stack([first, second])).to_numpy()
            np.testing.assert_array_equal(model.predict.call_args.args[0], expected)
            np.testing.assert_array_equal(actual, expected.sum(axis=1))
            np.testing.assert_array_equal(paired_predictions(adapter, cache, cache, first, second, False), actual)
            cartesian = predict(adapter, cache, cache, np.arange(3))
            self.assertEqual(cartesian.shape, (3, 3))
            np.testing.assert_array_equal(np.diag(cartesian), 0)
            np.testing.assert_array_equal(cartesian, cartesian.T)
            best, _, _ = exact_network(adapter, cache, cache, False)
            np.testing.assert_array_equal(best, 0)
            self.assertEqual(adapter.n_features_in_, len(PAIR_FEATURES))
            self.assertEqual(len(FEATURES), 13)

    def test_missing_cells_use_positive_area_whole_basin_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.frame().to_csv(root / "grid.csv.gz", index=False)
            basins = gpd.GeoDataFrame(dict(lake_id=[1, 2], max_depth_m=[4., 8.], volume_m3=[20., 60.]),
                                     geometry=[box(9500, 800, 10500, 1200),
                                               box(10500, 800, 11000, 1200)], crs=25833)
            basins.to_file(root / "basins.gpkg", driver="GPKG")
            lookup = BasinLookup(root / "grid.csv.gz", root / "basins.gpkg")
            values = lookup.lookup(np.array([10000., 11000., 12000.]), np.array([1000.]*3))
            np.testing.assert_array_equal(values, [[1, 1, 4, 20], [1, 1, 8, 60], [0, 0, 0, 0]])
            np.testing.assert_array_equal(values, lookup.lookup(np.array([10000., 11000., 12000.]),
                                                              np.array([1000.]*3)))

    def test_existing_station_uses_exactly_the_same_features_as_placement(self):
        grid = self.frame().assign(objtype="land")
        stations = grid.iloc[[0]].copy()
        stations["station_id"], stations["lon"], stations["lat"] = "SN13110", 10.41067, 61.16967
        stations["x"], stations["y"] = 1248.647, 1350.369
        stations["elevation"], stations["tpi_10km"], stations["tpi_25km"] = 205., 19., -96.
        original = stations.copy(deep=True)
        actual = station_cell_features(stations, grid)
        pd.testing.assert_frame_equal(stations, original)
        fields = ["x", "y", "elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype", *TERRAIN]
        pd.testing.assert_frame_equal(actual[fields], grid.iloc[[0]][fields])
        self.assertEqual(actual["station_id"].iloc[0], "SN13110")
        self.assertEqual(actual["lon"].iloc[0], 10.41067)
        self.assertEqual(actual["lat"].iloc[0], 61.16967)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "grid.csv.gz"
            grid.to_csv(path, index=False)
            adapter = SeNorgePairModel.__new__(SeNorgePairModel)
            adapter.basins = BasinLookup(path, Path(temp) / "unused.gpkg")
            adapter.model = Mock()
            adapter.model.predict.side_effect = lambda x: x.sum(axis=1)
            actual_predictions = predict(adapter, make_cache(actual), make_cache(grid), np.arange(3))
            placed_predictions = predict(adapter, make_cache(grid.iloc[[0]]), make_cache(grid), np.arange(3))
            np.testing.assert_array_equal(actual_predictions, placed_predictions)
            self.assertEqual(actual_predictions[0, 0], 0)

    def test_registry_recalculates_sources_and_does_not_reuse_legacy_surface(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = root / "outputs"
            prefix = outputs / "senorge_30yr_station_model"
            prefix.mkdir(parents=True)
            feature_path = outputs / "temperature_5y_prediction_rmse_station_features.gpkg"
            model_path = prefix / "xgboost_with_basins.ubj"
            basin_path = outputs / "potential_lakes_norway_DTM10_sill20_tiled" / "potential_lakes.gpkg"
            basin_path.parent.mkdir()
            for path in (feature_path, model_path, basin_path, prefix / "grid_features.csv.gz",
                         outputs / "norway_grid_1000m.gpkg"):
                path.touch()
            metadata = dict(features=PAIR_FEATURES, model_sha256="valid", basin_grid_sha256="valid")
            (prefix / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            grid = self.frame().assign(objtype="land")
            features = grid.iloc[[0]].assign(station_id="SN13110", x=1248., y=1350., tpi_10km=19.)
            observations = pd.DataFrame([dict(station_id="SN13110", name="E6 FABERG",
                lat=61.16967, lon=10.41067, first_time="2021-01-01", last_time="2026-01-01")])
            registry = WeatherRegistry(root, grid, make_cache(grid), pd.Index(["land"]))
            config = next(item for item in registry.available if item["id"] == "temperature:senorge30")
            registry.models[str(model_path)] = Mock(n_features_in_=len(PAIR_FEATURES))
            fields = ["x", "y", "elevation", "objtype", "coast_dist_fjord", "coast_dist_ocean", *TERRAIN]
            grid_hash = hashlib.sha256(pd.util.hash_pandas_object(
                grid[fields], index=False).to_numpy().tobytes()).hexdigest()
            signature = dict(schema="exact-best-two-v1", mode=config["id"], stations=["SN13110"],
                model=(model_path.stat().st_size, model_path.stat().st_mtime_ns),
                features=(feature_path.stat().st_size, feature_path.stat().st_mtime_ns),
                grid_size=len(grid), grid_hash=grid_hash, start="2021-01-01", end="2026-01-01",
                portable=dict(schema="basin-cell-identity-extrapolation-float64-v2", metadata=metadata,
                    basins=(basin_path.stat().st_size, basin_path.stat().st_mtime_ns)))
            digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
            cache_folder = outputs / "station_app_cache"
            cache_folder.mkdir()
            legacy = cache_folder / f"frost_exact_{digest}.npz"
            np.savez(legacy, baseline=[0., 0., 0.], second=[1., 1., 1.], winner=[0, 0, 0], completed=1)
            with patch("pyogrio.read_dataframe", return_value=features), \
                    patch("make_uncertainty_map_gif.load_grid_all_fields", return_value=grid), \
                    patch("file_hash.sha256", return_value="valid"), \
                    patch("station_weather.observation_stations",
                          return_value=(observations, "2021-01-01", "2026-01-01")):
                profile = registry.profile(config["id"])
            self.assertEqual(profile["stations"]["x"].iloc[0], 1000)
            self.assertEqual(profile["stations"]["tpi_10km"].iloc[0], 1)
            self.assertEqual(profile["source_grid_m"], 1000)
            self.assertNotEqual(profile["cache_path"], legacy)
            self.assertIsNone(profile["baseline"])
            self.assertEqual(profile["status"]["state"], "missing")

    def test_calculation_point_does_not_move_real_station_marker(self):
        class Weather(FakeWeather):
            def profile(self, identifier):
                result = super().profile(identifier)
                result["source_grid_m"] = 1000
                result["stations"]["lon"] = [10.41067, 11.]
                result["stations"]["lat"] = [61.16967, 62.]
                return result

        engine = FakeWeatherEngine()
        engine.weather = Weather()
        scenario = Scenario(engine)
        marker = scenario.state()["fixed_stations"][0]
        self.assertEqual((marker["lon"], marker["lat"]), (10.41067, 61.16967))
        lon, lat = engine.to_lonlat.transform(0, 0)
        self.assertEqual(marker["calculation_positions"][0], dict(
            dataset=scenario.dataset, lon=lon, lat=lat, cell_m=1000))
        scenario.apply(dict(action="move_frost", index=0, lon=12, lat=63))
        marker = scenario.state()["fixed_stations"][0]
        self.assertEqual(marker["original_position"], dict(lon=10.41067, lat=61.16967))
        self.assertEqual((marker["lon"], marker["lat"]), (12, 63))
        lon, lat = engine.to_lonlat.transform(1000, 0)
        self.assertEqual(marker["calculation_positions"][0]["lon"], lon)
        self.assertEqual(marker["calculation_positions"][0]["lat"], lat)
        scenario.apply(dict(action="restore_frost", station_id="SN1"))
        self.assertEqual(scenario.state()["fixed_stations"][0]["lon"], 10.41067)

    def test_debug_points_include_secondary_layer_and_manual_calculation_positions(self):
        class Weather(FakeWeather):
            def profile(self, identifier):
                result = super().profile(identifier)
                if identifier == "temperature:continuous":
                    result["source_grid_m"] = 1000
                return result

        engine = FakeWeatherEngine()
        engine.weather = Weather()
        scenario = Scenario(engine)
        scenario.apply(dict(action="dataset", value="snow:class"))
        scenario.layers = ["snow:class", "temperature:continuous"]
        marker = scenario.state()["fixed_stations"][0]
        self.assertEqual(marker["calculation_positions"][0]["dataset"], "temperature:continuous")
        scenario.apply(dict(action="add", lon=12, lat=63))
        manual = scenario.state()["stations"][0]
        self.assertEqual((manual["lon"], manual["lat"]), (12, 63))
        self.assertEqual(manual["calculation_positions"], [dict(lon=5, lat=60, cell_m=1000)])

    def test_discovery_requires_complete_install_and_original_station_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = root / "outputs"
            folder = outputs / "senorge_30yr_station_model"
            folder.mkdir(parents=True)
            for name in ("xgboost_with_basins.ubj", "metadata.json", "grid_features.csv.gz"):
                (folder / name).touch()
            self.assertNotIn("temperature:senorge30", [x["id"] for x in WeatherRegistry(root, None, None, None).available])
            (outputs / "temperature_5y_prediction_rmse_station_features.gpkg").touch()
            self.assertIn("temperature:senorge30", [x["id"] for x in WeatherRegistry(root, None, None, None).available])

    def test_api_profile_warning_survives_scenario_edits(self):
        class Weather(FakeWeather):
            available = FakeWeather.available + [dict(id="temperature:senorge30", phenomenon="temperature",
                kind="senorge30", label="seNorge 30 år", metric="RMSE", unit="°C")]

            def profile(self, identifier):
                result = super().profile(identifier)
                if identifier == "temperature:senorge30":
                    result.update(model_warning=WARNING, training_period="1996–2025")
                return result

        engine = FakeWeatherEngine()
        engine.weather = Weather()
        client = create_app(engine).test_client()
        response = client.post("/api/action", json=dict(action="dataset", value="temperature:senorge30"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["model_warning"], WARNING)
        self.assertEqual(response.get_json()["training_period"], "1996–2025")
        response = client.post("/api/action", json=dict(action="add", lon=5, lat=60))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["model_warning"], WARNING)
        self.assertEqual(len(response.get_json()["stations"]), 1)


if __name__ == "__main__":
    unittest.main()
