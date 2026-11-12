import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
import geopandas as gpd

from test_station_app import FakeWeather, FakeWeatherEngine
from station_app import Scenario, create_app
from station_weather import WeatherRegistry
from greedy_400_stations_pruned import TERRAIN
from calculation_progress import report, read_progress
import information_gain
import retrain_weather_1km


class DifferentSensors(FakeWeather):
    def profile(self, identifier):
        profile = super().profile(identifier)
        if identifier == "snow:class":
            profile["stations"] = profile["stations"].iloc[[0]].copy()
            profile["stations"]["station_id"] = "SN3"
            profile["stations"]["name"] = "E6 Snow"
            profile["stations"]["x"] = 3000.
            profile["winner"] = np.zeros(3, dtype=np.int32)
        return profile


class SensorEngine(FakeWeatherEngine):
    weather = DifferentSensors()

    def sensor_catalog(self):
        return {"SN1": ["temperature"], "SN2": ["temperature"], "SN3": ["snow"]}


class SensorProposalTests(unittest.TestCase):
    def test_move_observation_carries_added_sensors_and_preserves_original_data(self):
        engine = SensorEngine()
        scenario = Scenario(engine)
        originals = engine.weather.profile("temperature:continuous")["stations"].copy()
        scenario.apply(dict(action="add_sensor", station_id="SN1", sensors=["snow"]))
        with patch.object(engine, "snap", return_value=7), \
                patch.object(engine, "station", return_value=dict(
                    id=7, x=7000, y=0, lon=13, lat=64, elevation=15)):
            scenario.apply(dict(action="move_frost", station_id="SN1", lon=13, lat=64))
            self.assertEqual(scenario.attached_proposal("SN1"), 7)
            self.assertEqual(scenario.added, [7])
            self.assertEqual(scenario.added_sensors, {7: ("snow",)})
            self.assertEqual(scenario.added_positions, {7: dict(lon=13, lat=64)})
            self.assertEqual(scenario.station_edits["SN1"]["candidate_index"], 7)
            with scenario.as_dataset("snow:class"):
                self.assertIn(7, scenario.active_added())
            with patch.object(engine, "snap", return_value=1):
                scenario.apply(dict(action="restore_frost", station_id="SN1"))
            self.assertEqual(scenario.added, [1])
            self.assertEqual(scenario.attached_proposal("SN1"), 1)
            self.assertEqual(scenario.station_edits, {})
        pd.testing.assert_frame_equal(engine.weather.profile("temperature:continuous")["stations"], originals)

    def test_move_collision_keeps_station_and_proposal_unchanged(self):
        engine = SensorEngine()
        scenario = Scenario(engine)
        scenario.apply(dict(action="add_sensor", station_id="SN1", sensors=["snow"]))
        with patch.object(engine, "snap", return_value=7), \
                patch.object(engine, "station", return_value=dict(
                    id=7, x=7000, y=0, lon=13, lat=64, elevation=15)):
            scenario.apply(dict(action="add", lon=13, lat=64, sensors=["snow"]))
            revision = scenario.revision
            with self.assertRaisesRegex(ValueError, "eget forslag"):
                scenario.apply(dict(action="move_frost", station_id="SN1", lon=13, lat=64))
        self.assertEqual(scenario.revision, revision)
        self.assertEqual(scenario.added, [1, 7])
        self.assertEqual(scenario.station_edits, {})

    def test_remove_sensor_api_updates_marker_and_surface(self):
        client = create_app(SensorEngine()).test_client()
        response = client.post("/api/action", json=dict(
            action="remove_sensor", station_id="SN2", sensor="temperature"))
        self.assertEqual(response.status_code, 200)
        marker = next(item for item in response.json["fixed_stations"] if item["station_id"] == "SN2")
        self.assertEqual(marker["sensors"], [])
        self.assertFalse(marker["removed"])
        self.assertTrue(marker["sensors_changed"])
        self.assertEqual(client.post("/api/action", json=dict(
            action="remove_sensor", station_id="SN2", sensor="temperature")).status_code, 400)

    def test_remove_native_sensor_only_excludes_its_phenomenon_and_can_restore(self):
        scenario = Scenario(SensorEngine())
        scenario.apply(dict(action="remove_sensor", station_id="SN2", sensor="temperature"))
        self.assertEqual(scenario.frost_edits, {1: dict(removed=True)})
        marker = next(item for item in scenario.state()["fixed_stations"] if item["station_id"] == "SN2")
        self.assertFalse(marker["removed"])
        self.assertNotIn("temperature", marker["sensors"])
        self.assertTrue(marker["sensors_changed"])
        with scenario.as_dataset("snow:class"):
            self.assertEqual(scenario.frost_edits, {})
        np.testing.assert_array_equal(scenario.surface(), scenario.profile()["second"])
        scenario.apply(dict(action="add_sensor", station_id="SN2", sensors=["temperature"]))
        self.assertEqual(scenario.frost_edits, {})
        self.assertEqual(scenario.added, [])
        self.assertIn("temperature", next(item for item in scenario.state()["fixed_stations"]
                                        if item["station_id"] == "SN2")["sensors"])

    def test_remove_added_and_manual_sensors_including_last_sensor(self):
        scenario = Scenario(SensorEngine())
        scenario.apply(dict(action="add_sensor", station_id="SN1", sensors=["snow", "wind"]))
        scenario.apply(dict(action="remove_sensor", station_id="SN1", sensor="snow"))
        self.assertEqual(scenario.added_sensors[1], ("wind",))
        scenario.apply(dict(action="remove_sensor", station_id="SN1", sensor="wind"))
        self.assertEqual(scenario.added, [])
        self.assertEqual(scenario.station_sensor_removals, {})
        scenario.apply(dict(action="add", lon=12, lat=63, sensors=["snow"]))
        scenario.apply(dict(action="remove_sensor", id=1, sensor="snow"))
        self.assertEqual(scenario.added, [])
        self.assertEqual(scenario.added_sensors, {})
        self.assertIsNone(scenario.selected)

    def test_invalid_sensor_removal_is_atomic_and_restore_clears_removals(self):
        scenario = Scenario(SensorEngine())
        for data in [dict(station_id="SN1", sensor="snow"),
                     dict(station_id="unknown", sensor="temperature"),
                     dict(station_id="SN1", sensor="invalid")]:
            with self.assertRaises(ValueError):
                scenario.apply(dict(action="remove_sensor", **data))
        self.assertEqual(scenario.revision, 0)
        scenario.apply(dict(action="remove_sensor", station_id="SN1", sensor="temperature"))
        scenario.apply(dict(action="restore_frost", station_id="SN1"))
        self.assertEqual(scenario.station_sensor_removals, {})
        self.assertEqual(scenario.frost_edits, {})

    def test_move_proposal_preserves_sensors_order_and_changes_selected_cell(self):
        engine = SensorEngine()
        scenario = Scenario(engine)
        scenario.apply(dict(action="add", lon=12, lat=63, sensors=["snow", "wind"]))
        with patch.object(engine, "snap", return_value=7), \
                patch.object(engine, "station", return_value=dict(
                    id=7, x=7000, y=0, lon=13, lat=64, elevation=15)):
            scenario.apply(dict(action="move_manual", id=1, lon=13.1, lat=64.1))
        self.assertEqual(scenario.added, [7])
        self.assertEqual(scenario.added_sensors[7], ("snow", "wind"))
        self.assertEqual(scenario.added_positions[7], dict(lon=13.1, lat=64.1))
        self.assertNotIn(1, scenario.added_positions)
        self.assertNotIn(1, scenario.added_sensors)
        self.assertEqual(scenario.selected, dict(kind="manual", index=7))
        self.assertEqual(scenario.station_edits, {})

    def test_move_within_same_cell_updates_display_position(self):
        scenario = Scenario(SensorEngine())
        scenario.apply(dict(action="add", lon=12, lat=63, sensors=["snow"]))
        scenario.apply(dict(action="move_manual", id=1, lon=12.1, lat=63.1))
        self.assertEqual(scenario.added, [1])
        self.assertEqual(scenario.added_sensors[1], ("snow",))
        self.assertEqual(scenario.added_positions[1], dict(lon=12.1, lat=63.1))

    def test_failed_proposal_move_does_not_change_original(self):
        engine = SensorEngine()
        scenario = Scenario(engine)
        scenario.apply(dict(action="add", lon=12, lat=63, sensors=["snow"]))
        revision = scenario.revision
        with self.assertRaises(ValueError):
            scenario.apply(dict(action="move_manual", id=999, lon=12, lat=63))
        with patch.object(engine, "snap", return_value=0), \
                patch.object(engine, "station", return_value=dict(id=0, x=0, y=0)), \
                self.assertRaisesRegex(ValueError, "grunnettet"):
            scenario.apply(dict(action="move_manual", id=1, lon=12, lat=63))
        with patch.object(engine, "snap", return_value=7), \
                patch.object(engine, "station", return_value=dict(id=7, x=7000, y=0)), \
                patch.object(engine, "prediction", side_effect=ValueError("prediction failed")), \
                self.assertRaisesRegex(ValueError, "prediction failed"):
            scenario.apply(dict(action="move_manual", id=1, lon=12, lat=63))
        self.assertEqual(scenario.revision, revision)
        self.assertEqual(scenario.added, [1])
        self.assertEqual(scenario.added_positions[1], dict(lon=12, lat=63))

    def test_station_without_selected_sensor_remains_selectable_and_removable(self):
        scenario = Scenario(SensorEngine())
        before = scenario.surface().copy()
        markers = scenario.state()["fixed_stations"]
        self.assertEqual({marker["station_id"] for marker in markers}, {"SN1", "SN2", "SN3"})
        snow = next(marker for marker in markers if marker["station_id"] == "SN3")
        self.assertEqual(snow["layers"], [])
        self.assertEqual(snow["sensors"], ["snow"])
        self.assertIsNone(snow["index"])
        scenario.apply(dict(action="select", kind="frost", station_id="SN3"))
        self.assertEqual(scenario.state()["selected_station_label"], "SN3 · E6 Snow")
        np.testing.assert_array_equal(scenario.surface(), before)
        scenario.apply(dict(action="remove_frost", station_id="SN3"))
        self.assertTrue(next(marker for marker in scenario.state()["fixed_stations"]
                             if marker["station_id"] == "SN3")["removed"])
        np.testing.assert_array_equal(scenario.surface(), before)

    def test_sensor_proposal_is_a_manual_station_and_changes_only_its_sensor(self):
        scenario = Scenario(SensorEngine())
        old_snow = scenario.engine.weather.profile("snow:class")["stations"].copy()
        before = scenario.surface().copy()
        scenario.apply(dict(action="add_sensor", station_id="SN1", sensors=["snow"]))
        self.assertEqual(scenario.added, [1])
        self.assertEqual(scenario.added_sensors[1], ("snow",))
        np.testing.assert_array_equal(scenario.surface(), before)
        self.assertEqual(scenario.selected, dict(kind="manual", index=1))
        with scenario.as_dataset("snow:class"):
            expected = np.minimum(scenario.base(), scenario.engine.prediction(1, "snow:class"))
            np.testing.assert_array_equal(scenario.surface(), expected)
        pd.testing.assert_frame_equal(scenario.engine.weather.profile("snow:class")["stations"], old_snow)
        scenario.apply(dict(action="remove", id=1))
        self.assertEqual(scenario.added, [])

    def test_network_filter_still_applies_to_inactive_stations(self):
        scenario = Scenario(SensorEngine())
        scenario.pross_only = True
        self.assertEqual([marker["station_id"] for marker in scenario.state()["fixed_stations"]], ["SN3"])
        scenario.layers = []
        self.assertEqual([marker["station_id"] for marker in scenario.state()["fixed_stations"]], ["SN3"])

    def test_invalid_sensor_does_not_change_scenario(self):
        scenario = Scenario(SensorEngine())
        for data in (dict(station_id="SN1", sensors=[]),
                     dict(station_id="SN1", sensors=["temperature"]),
                     dict(station_id="unknown", sensors=["snow"]),
                     dict(station_id="SN1", sensors=["invalid"])):
            with self.assertRaises(ValueError):
                scenario.apply(dict(action="add_sensor", **data))
        self.assertEqual(scenario.revision, 0)
        self.assertEqual(scenario.added, [])


class OneKmTrainingTests(unittest.TestCase):
    def test_progress_has_measured_phase_eta_and_can_be_read_by_app(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "outputs" / "weather_calculation_progress.json"
            with patch.dict(os.environ, {"WEATHER_CALCULATION_PROGRESS": str(path)}), \
                    patch("calculation_progress.time.monotonic", return_value=140):
                result = report("Stasjonspar", 25, 100, started=100)
            self.assertEqual(result["remaining_seconds"], 120)
            self.assertEqual(read_progress(temp)["completed"], 25)
            self.assertFalse(read_progress(temp)["stale"])
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_queue_has_no_download_and_only_publishes_after_both_steps(self):
        for phenomenon, metric in retrain_weather_1km.MODELS:
            train, maps = retrain_weather_1km.commands(phenomenon, metric)
            self.assertIn("--station-cell-inputs", train)
            self.assertIn("--skip-download", train)
            self.assertEqual(train[train.index("--grid-spacing-m") + 1], "1000")
            self.assertEqual(maps[maps.index("--station-network") + 1], "both")
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(retrain_weather_1km, "ROOT", Path(temp)), \
                patch.object(retrain_weather_1km, "MODELS", [("wind", "prediction_rmse")]), \
                patch.dict(os.environ, {}, clear=False):
            marker = Path(temp) / "outputs" / "weather_1km" / "wind" / "ready.json"
            def execute(*args, **kwargs):
                self.assertFalse(marker.exists())
                return type("Result", (), {"returncode": 0})()
            with patch.object(retrain_weather_1km.subprocess, "run", side_effect=execute):
                self.assertEqual(retrain_weather_1km.main([]), 0)
            self.assertTrue(marker.exists())

    def test_registry_does_not_select_unpublished_models(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            folder = Path(temp) / "outputs" / "weather_1km" / "wind"
            folder.mkdir(parents=True)
            for suffix in ("_model.joblib", "_metadata.json", "_station_features.gpkg"):
                (folder / f"wind_5y_prediction_rmse{suffix}").touch()
            self.assertNotIn("wind:continuous", [item["id"] for item in WeatherRegistry(temp, None, None, None).available])
            (folder / "ready.json").write_text("{}")
            config = next(item for item in WeatherRegistry(temp, None, None, None).available
                          if item["id"] == "wind:continuous")
            self.assertEqual(config["model_grid_m"], 1000)
            self.assertEqual(Path(config["prefix"]).parent, folder)

    def test_failed_map_calculation_never_publishes_model(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(retrain_weather_1km, "ROOT", Path(temp)), \
                patch.object(retrain_weather_1km, "MODELS", [("wind", "prediction_rmse")]), \
                patch.dict(os.environ, {}, clear=False), \
                patch.object(retrain_weather_1km.subprocess, "run",
                             side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)]):
            self.assertEqual(retrain_weather_1km.main([]), 1)
            self.assertFalse((Path(temp) / "outputs" / "weather_1km" / "wind" / "ready.json").exists())
            self.assertEqual(read_progress(temp)["state"], "failed")

    def test_training_uses_cell_coordinates_elevation_and_terrain_but_retains_real_position(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            database = root / "stations.db"
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("CREATE TABLE stations (station_id TEXT,lat REAL,lon REAL,elevation REAL)")
                connection.execute("INSERT INTO stations VALUES ('SN1',61.16967,10.41067,205)")
                connection.execute("INSERT INTO stations VALUES ('SNmissing',NULL,NULL,205)")
            grid = gpd.GeoDataFrame(dict(x=[253000.], y=[6790000.], elevation=[150.],
                coast_dist_fjord=[100.], coast_dist_ocean=[200.], objtype=["land"],
                **{field: [-326.] for field in TERRAIN}),
                geometry=gpd.points_from_xy([253000], [6790000]), crs=25833)
            paths = information_gain.paths("wind", 5, "prediction_rmse", output_dir=root)
            with patch.object(information_gain, "DATABASE", database):
                result = information_gain.station_features(grid, paths, force=True, cell_inputs=True)
            self.assertEqual(result["station_id"].tolist(), ["SN1"])
            self.assertEqual(float(result.iloc[0].x), 253000)
            self.assertEqual(float(result.iloc[0].y), 6790000)
            self.assertEqual(float(result.iloc[0].elevation), 150)
            self.assertEqual(float(result.iloc[0].tpi_10km), -326)
            self.assertEqual(float(result.iloc[0].lon), 10.41067)
            self.assertEqual(float(result.geometry.iloc[0].x), 253000)


if __name__ == "__main__":
    unittest.main()
