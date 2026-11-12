"""Focused scenario/API tests; no production model or terrain files are loaded."""
from io import BytesIO
from contextlib import closing
from collections import OrderedDict
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pyproj import Transformer

from station_app import GreedySurfaces, Scenario, StationEngine, TERRAIN_LAYERS, create_app
from station_weather import WeatherRegistry, exact_network, observation_stations, paired_predictions, save_exact_cache
from station_roads import RoadNetwork, normalize_reference, road_categories
from greedy_400_stations_pruned import CACHE_KEYS, FEATURES, TERRAIN, feature_matrix, make_cache


class FakeEngine:
    cell_m = 1000
    minimum, maximum = 0.5, 6
    bounds = [[58, 5], [71, 31]]
    frames = np.array([[4, 5, 6], [3, 4, 3]], dtype=np.float32)
    grid = pd.DataFrame({"x": [0, 1000, 2000], "y": [0, 0, 0]})
    stations = pd.DataFrame({"x": [0, 2000], "y": [0, 0]})
    to_lonlat = Transformer.from_crs(25833, 4326, always_xy=True)

    def prediction(self, index):
        return np.array([2, 0.5, 2], dtype=np.float32)

    def snap(self, lon, lat):
        return 1

    def station(self, index):
        return dict(id=index, x=1000, y=0, lon=5, lat=60, elevation=10)

    def render(self, values, mode, *limits, cell_mask=None, display_size=None):
        from PIL import Image

        buffer = BytesIO()
        Image.new("RGBA", (2, 2)).save(buffer, format="PNG")
        buffer.seek(0)
        return buffer


class FakeWeather:
    available = [dict(id="temperature:continuous", phenomenon="temperature", kind="continuous",
                      label="Temperature", metric="RMSE", unit="\u00b0C"),
              dict(id="temperature:senorge", phenomenon="temperature", kind="senorge",
                  label="Temperature (seNorge pilot)", metric="RMSE", unit="\u00b0C"),
                 dict(id="snow:class", phenomenon="snow", kind="class",
                      label="Snow", metric="Brier score", unit="score")]

    def svv_profile(self, identifier):
        return self.profile(identifier)

    def profile(self, identifier):
        config = next((item for item in self.available if item["id"] == identifier), None)
        if config is None:
            raise ValueError("Unknown weather mode")
        stations = FakeEngine.stations.copy()
        stations["station_id"] = ["SN1", "SN2"]
        stations["name"] = ["First", "Second"]
        stations["first_time"] = "2021-10-14"
        stations["last_time"] = "2026-10-14"
        return dict(**config, baseline=FakeEngine.frames[-1] / (20 if config["kind"] == "class" else 1),
                    second=FakeEngine.frames[0] / (20 if config["kind"] == "class" else 1),
                    winner=np.ones(3, dtype=np.int32), status=dict(state="ready", percent=100),
                    stations=stations, minimum=0.01, maximum=1 if config["kind"] == "class" else 6,
                    start="2021-10-14", end="2026-10-15", missing_features=0,
                    method="Eksakt minimum over alle Frost-stasjoner", classes=1)


class FakeWeatherEngine(FakeEngine):
    weather = FakeWeather()

    def prediction(self, index, dataset=None):
        values = super().prediction(index)
        return values / 20 if dataset == "snow:class" else values

    def frames_for(self, dataset):
        return self.frames / 20 if dataset == "snow:class" else self.frames

    def greedy_prediction(self, index, dataset="temperature:continuous"):
        return self.frames_for(dataset)[index]

    def greedy_prediction_at(self, index, target_ids, dataset="temperature:continuous"):
        return self.greedy_prediction(index, dataset)[target_ids]


class TestHostingTests(unittest.TestCase):
    def test_waitress_loopback_http_state_action_and_origin_guard(self):
        import http.client
        try:
            from waitress import create_server
        except ImportError:
            self.skipTest("Install requirements-hosting.txt for the real Waitress smoke test")

        server = create_server(create_app(FakeEngine()), host="127.0.0.1", port=0, threads=2)
        worker = threading.Thread(target=server.run, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.effective_port, timeout=5)
        try:
            connection.request("GET", "/api/state")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            cookie = response.getheader("Set-Cookie").split(";", 1)[0]
            response.read()
            origin = f"http://127.0.0.1:{server.effective_port}"
            headers = {"Content-Type": "application/json", "Origin": origin, "Cookie": cookie}
            action = json.dumps(dict(action="add", lon=5, lat=60))
            connection.request("POST", "/api/action", body=action, headers=headers)
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(len(json.loads(response.read())["stations"]), 1)
            connection.request("POST", "/api/action", body=action, headers=dict(headers, Origin="http://other-site"))
            response = connection.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
        finally:
            connection.close()
            server.task_dispatcher.shutdown()
            server.close()
            worker.join(timeout=5)

    def test_test_hosting_defaults_are_loopback_and_use_separate_port(self):
        from host_station_app import parse_args

        args = parse_args([])
        self.assertEqual((args.host, args.port, args.threads), ("127.0.0.1", 8766, 4))
        self.assertEqual(parse_args(["--host", "0.0.0.0"]).host, "0.0.0.0")
        for values in (["--port", "0"], ["--threads", "0"], ["--host", "bad-address"]):
            with self.assertRaises(SystemExit):
                parse_args(values)

    def test_test_hosting_closes_resources_when_server_stops(self):
        from unittest.mock import Mock
        import host_station_app

        app, engine = Mock(), Mock()
        with patch.dict("sys.modules", {"waitress": Mock()}), \
                patch.object(host_station_app, "create_application", return_value=(app, engine)), \
                patch.object(host_station_app, "serve_application", side_effect=KeyboardInterrupt) as serve:
            self.assertEqual(host_station_app.main([]), 0)
        self.assertEqual(serve.call_args.args[1].host, "127.0.0.1")
        engine.resources.close.assert_called_once()

    def test_hosting_preserves_same_origin_protection_for_network_address(self):
        client = create_app(FakeEngine()).test_client()
        origin = "http://test-pc.internal:8766"
        action = dict(action="add", lon=5, lat=60)
        self.assertEqual(client.post("/api/action", base_url=origin, headers={"Origin": origin}, json=action).status_code, 200)
        self.assertEqual(client.post("/api/action", base_url=origin, headers={"Origin": "http://other-site"}, json=action).status_code, 403)


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.scenario = Scenario(self.engine)

    def add(self):
        self.scenario.apply(dict(action="add", lon=5, lat=60))

    def test_empty_weather_selection_preserves_proposals_and_can_be_reselected(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="add", lon=5, lat=60, sensors=["temperature"]))
        dataset = scenario.dataset
        scenario.apply(dict(action="layers", value=[]))
        self.assertEqual(scenario.layers, [])
        self.assertEqual(scenario.dataset, dataset)
        self.assertEqual(scenario.added, [1])
        self.assertEqual(scenario.state()["layers"], [])
        with patch.dict("station_app.PHENOMENON_DATASETS", {"temperature": dataset}, clear=True):
            scenario.apply(dict(action="layers", value=["temperature"]))
        self.assertEqual(scenario.layers, [dataset])
        self.assertEqual(scenario.added_sensors[1], ("temperature",))

    def test_selected_station_comparison_is_default(self):
        scenario = Scenario(FakeWeatherEngine())
        self.assertEqual(scenario.comparison_scope, "selected")
        self.assertEqual(scenario.state()["comparison_scope"], "selected")
        scenario.apply(dict(action="comparison", value="baseline"))
        scenario.apply(dict(action="reset_defaults"))
        self.assertEqual(scenario.comparison_scope, "selected")

    def test_edit_proposal_sensors_preserves_location_and_updates_predictions(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="add", lon=5.123, lat=60.456, sensors=["temperature"]))
        before = scenario.surface().copy()
        scenario.apply(dict(action="sensors", id=1, sensors=["snow"]))
        self.assertEqual(scenario.added, [1])
        self.assertEqual(scenario.added_positions[1], dict(lon=5.123, lat=60.456))
        self.assertEqual(scenario.added_sensors[1], ("snow",))
        np.testing.assert_array_equal(scenario.surface(), scenario.base())
        self.assertFalse(np.array_equal(before, scenario.surface()))
        with scenario.as_dataset("snow:class"):
            self.assertFalse(np.array_equal(scenario.surface(), scenario.base()))
        revision = scenario.revision
        for sensors in ([], ["unknown"]):
            with self.assertRaises(ValueError):
                scenario.apply(dict(action="sensors", id=1, sensors=sensors))
        self.assertEqual(scenario.added_sensors[1], ("snow",))
        self.assertEqual(scenario.revision, revision)
        with self.assertRaises(ValueError):
            scenario.apply(dict(action="sensors", id=999, sensors=["temperature"]))

    def test_restore_one_station_preserves_other_station_edits(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="move_frost", station_id="SN1", lon=6, lat=61))
        scenario.apply(dict(action="remove_frost", station_id="SN2"))
        scenario.apply(dict(action="restore_frost", station_id="SN1"))
        self.assertEqual(scenario.station_edits, {"SN2": dict(removed=True)})
        self.assertNotIn(0, scenario.frost_edits)
        scenario.apply(dict(action="restore_frost", station_id="SN2"))
        self.assertEqual(scenario.station_edits, {})
        np.testing.assert_array_equal(scenario.surface(), scenario.profile()["baseline"])

    def test_reset_defaults_clears_edits_and_preserves_session_lock(self):
        scenario = Scenario(FakeWeatherEngine())
        initial_dataset, lock = scenario.dataset, scenario.lock
        scenario.apply(dict(action="remove_frost", station_id="SN2"))
        scenario.apply(dict(action="dataset", value="snow:class"))
        scenario.apply(dict(action="network", value="empty"))
        scenario.apply(dict(action="add", lon=5, lat=60, sensors=["snow"]))
        scenario.counties = ["03"]
        previous_revision = scenario.revision
        scenario.apply(dict(action="reset_defaults"))
        self.assertEqual((scenario.dataset, scenario.layers, scenario.network),
                         (initial_dataset, [initial_dataset], "frost"))
        self.assertEqual((scenario.added, scenario.station_edits, scenario.added_positions), ([], {}, {}))
        self.assertIsNone(scenario.counties)
        self.assertIs(scenario.lock, lock)
        self.assertGreater(scenario.revision, previous_revision)

    def test_clicked_position_is_displayed_but_predictions_stay_snapped(self):
        self.scenario.apply(dict(action="add", lon=5.678, lat=60.123))
        self.assertEqual(self.scenario.added, [1])
        station = self.scenario.state()["stations"][0]
        self.assertEqual((station["lon"], station["lat"]), (5.678, 60.123))
        self.assertEqual((station["x"], station["y"]), (1000, 0))
        np.testing.assert_array_equal(self.scenario.surface(), self.engine.prediction(1))

    def test_add_and_undo(self):
        self.add()
        np.testing.assert_array_equal(self.scenario.surface(), [2, 0.5, 2])
        self.assertEqual(self.scenario.state()["affected_area_m2"], 3000000)
        self.scenario.apply(dict(action="undo"))
        np.testing.assert_array_equal(self.scenario.surface(), self.engine.frames[-1])

    def test_senorge_profile_is_opt_in_and_preserves_station_edits(self):
        client = create_app(FakeWeatherEngine()).test_client()
        initial = client.get("/api/state").get_json()
        self.assertEqual(initial["dataset"], "temperature:continuous")
        self.assertIn("temperature:senorge", [mode["id"] for mode in initial["weather_modes"]])

        selected = client.post("/api/action", json={
            "action": "dataset", "value": "temperature:senorge",
        })
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(selected.get_json()["dataset"], "temperature:senorge")
        self.assertEqual(selected.get_json()["threshold"], 0.01)

        added = client.post("/api/action", json={"action": "add", "lon": 5, "lat": 60})
        self.assertEqual(added.status_code, 200)
        self.assertEqual(added.get_json()["dataset"], "temperature:senorge")
        self.assertEqual(len(added.get_json()["stations"]), 1)

    def test_senorge_profile_can_be_selected_for_station_edit_scenario(self):
        client = create_app(FakeWeatherEngine()).test_client()
        initial = client.get("/api/state").get_json()
        self.assertIn("temperature:senorge", [mode["id"] for mode in initial["weather_modes"]])

        selected = client.post("/api/action", json={
            "action": "dataset", "value": "temperature:senorge",
        })
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(selected.get_json()["dataset"], "temperature:senorge")
        self.assertEqual(selected.get_json()["threshold"], 0.01)

        added = client.post("/api/action", json={"action": "add", "lon": 5, "lat": 60})
        self.assertEqual(added.status_code, 200)
        self.assertEqual(added.get_json()["dataset"], "temperature:senorge")
        self.assertEqual(len(added.get_json()["stations"]), 1)

    def test_first_station_establishes_baseline(self):
        self.scenario.apply(dict(action="baseline", count=0))
        self.assertIsNone(self.scenario.surface())
        self.add()
        self.assertIsNone(self.scenario.state()["mean_reduction"])
        self.assertIsNone(self.scenario.state()["affected_area_m2"])

    def test_duplicate_is_not_added(self):
        self.add()
        with self.assertRaises(ValueError):
            self.add()
        self.assertEqual(len(self.scenario.added), 1)

    def test_remove_and_reset_restore_saved_surface(self):
        self.add()
        self.scenario.apply(dict(action="remove", id=1))
        np.testing.assert_array_equal(self.scenario.surface(), self.engine.frames[-1])
        self.add()
        self.scenario.apply(dict(action="reset"))
        self.assertEqual(self.scenario.added, [])

    def test_threshold_and_baseline_validation(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                self.scenario.apply(dict(action="threshold", value=value))
        for count in [-1, 3]:
            with self.assertRaises(ValueError):
                self.scenario.apply(dict(action="baseline", count=count))
        self.assertEqual(self.scenario.threshold, 0.01)
        self.assertEqual(self.scenario.baseline, 2)

    def test_browser_sessions_are_independent(self):
        app = create_app(self.engine)
        first, second = app.test_client(), app.test_client()
        response = first.post("/api/action", json=dict(action="add", lon=5, lat=60))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(first.get("/api/state").json["stations"]), 1)
        self.assertEqual(second.get("/api/state").json["stations"], [])

    def test_cross_origin_and_invalid_actions(self):
        client = create_app(self.engine).test_client()
        response = client.post("/api/action", json=dict(action="reset"),
                               headers={"Origin": "https://example.org"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(client.post("/api/action", json=dict(action="unknown")).status_code, 400)

    def test_arcgis_export_mask_metadata(self):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin

        self.engine.raster_shape = (1, 3)
        self.engine.rows = np.array([0, 0, 0])
        self.engine.columns = np.array([0, 1, 2])
        self.engine.transform = from_origin(-500, 500, 1000, 1000)
        self.add()
        with ZipFile(self.scenario.export()) as archive:
            self.assertIn("stations.gpkg", archive.namelist())
            with MemoryFile(archive.read("selected_affected.tif")) as memory:
                with memory.open() as raster:
                    self.assertEqual(raster.crs.to_epsg(), 25833)
                    self.assertEqual(raster.tags()["units"], "binary 0/1")
                    np.testing.assert_array_equal(raster.read(1), [[1, 1, 1]])

    def test_affected_render_has_opaque_contrasting_fill(self):
        from PIL import Image

        engine = StationEngine.__new__(StationEngine)
        engine.grid = self.engine.grid
        engine.cell_m = 1000
        engine.minimum, engine.maximum = self.engine.minimum, self.engine.maximum
        engine._prepare_raster()
        with Image.open(engine.render(np.array([1, 1, 0], dtype=np.float32), "affected")) as image:
            pixels = np.asarray(image)
            fill = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 50) & (pixels[:, :, 2] > 80)
            self.assertTrue(fill.any())
            self.assertTrue((pixels[:, :, 3][fill] == 255).all())

    def test_missing_uncertainty_is_red_for_empty_and_nonfinite_surfaces(self):
        from PIL import Image

        engine = StationEngine.__new__(StationEngine)
        engine.grid = self.engine.grid
        engine.cell_m = 1000
        engine.minimum, engine.maximum = 1, 4
        engine._prepare_raster()
        for mode in ("uncertainty", "baseline", "combined"):
            for values in (None, np.array([np.nan, np.inf, -np.inf], dtype=np.float32)):
                with self.subTest(mode=mode, values=values), Image.open(engine.render(values, mode)) as image:
                    pixels = np.asarray(image)
                    visible = pixels[:, :, 3] > 0
                    self.assertTrue(visible.any())
                    self.assertTrue((pixels[:, :, 0][visible] > pixels[:, :, 2][visible]).all())

    def test_missing_uncertainty_stays_transparent_outside_polygon(self):
        from PIL import Image
        from shapely.geometry import box
        from rasterio.transform import array_bounds
        from rasterio.features import geometry_mask

        engine = StationEngine.__new__(StationEngine)
        engine.grid = self.engine.grid
        engine.cell_m = 1000
        engine.minimum, engine.maximum = 1, 4
        engine._prepare_raster()
        west, south, east, north = array_bounds(engine.web_height, engine.web_width, engine.web_transform)
        engine.norway_path = Path("unused.gpkg")
        engine._norway_shapes = [box(west, south, (west + east) / 2, north)]
        inside = geometry_mask(engine._norway_shapes, out_shape=(engine.web_height, engine.web_width),
                               transform=engine.web_transform, invert=True)
        with Image.open(engine.render(None, "uncertainty")) as image:
            pixels = np.asarray(image)
            self.assertTrue((pixels[:, :, 3][inside] > 0).any())
            self.assertTrue((pixels[:, :, 3][~inside] == 0).all())

    def test_rmse_render_has_blue_low_values_and_red_high_values(self):
        from PIL import Image

        engine = StationEngine.__new__(StationEngine)
        engine.grid = self.engine.grid
        engine.cell_m = 1000
        engine.minimum, engine.maximum = 1, 4
        engine._prepare_raster()
        with Image.open(engine.render(np.array([1, 2, 4], dtype=np.float32), "uncertainty")) as image:
            pixels = np.asarray(image)
            visible = pixels[:, :, 3] > 0
            self.assertTrue(((pixels[:, :, 2] > pixels[:, :, 0]) & visible).any())
            self.assertTrue(((pixels[:, :, 0] > pixels[:, :, 2]) & visible).any())

    def test_signed_changes_render_increases_red_and_reductions_blue(self):
        from PIL import Image

        engine = StationEngine.__new__(StationEngine)
        engine.grid = self.engine.grid
        engine.cell_m = 1000
        engine.minimum, engine.maximum = 1, 4
        engine._prepare_raster()
        for mode in ("change", "affected_change"):
            with self.subTest(mode=mode), Image.open(engine.render(
                    np.array([-1, 0, 1], dtype=np.float32), mode)) as image:
                pixels = np.asarray(image)
                red = (pixels[:, :, 0] > pixels[:, :, 2]) & (pixels[:, :, 3] > 0)
                blue = (pixels[:, :, 2] > pixels[:, :, 0]) & (pixels[:, :, 3] > 0)
                self.assertTrue(red.any())
                self.assertTrue(blue.any())
                if mode == "affected_change":
                    self.assertTrue((pixels[:, :, 3][red | blue] == 255).all())

    def test_combined_modes_render_visible_pixels(self):
        from PIL import Image

        engine = StationEngine.__new__(StationEngine)
        engine.grid = self.engine.grid
        engine.cell_m = 1000
        engine.minimum, engine.maximum = 1, 4
        engine._prepare_raster()
        for mode, values in (("combined", [0, 1, 2]), ("combined_change", [-1, 0, 2]), ("combined_reduction", [0, 1, 2])):
            with self.subTest(mode=mode), Image.open(engine.render(np.array(values, dtype=np.float32), mode, 0, 2, 1)) as image:
                self.assertTrue((np.asarray(image)[:, :, 3] > 0).any())


class PersistentWeatherMapTests(unittest.TestCase):
    def test_combined_layers_sum_normalised_phenomena_for_every_map_mode(self):
        mapping = {"temperature": "temperature:continuous", "snow": "snow:class"}
        with patch.dict("station_app.PHENOMENON_DATASETS", mapping, clear=True):
            client = create_app(FakeWeatherEngine()).test_client()
            state = client.post("/api/action", json={"action": "layers", "value": ["snow", "temperature"]}).get_json()
            self.assertEqual(state["layers"], ["temperature:continuous", "snow:class"])
            self.assertEqual(state["dataset"], "temperature:continuous")
            self.assertTrue(all(option["available"] for option in state["layer_options"]))
            client.post("/api/action", json={"action": "add", "lon": 5, "lat": 60})
            for mode in ("uncertainty", "baseline", "reduction", "affected"):
                with self.subTest(mode=mode):
                    self.assertEqual(client.get(f"/api/map.png?mode={mode}").status_code, 200)
            unknown = client.post("/api/action", json={"action": "layers", "value": ["wind"]})
            self.assertEqual(unknown.status_code, 400)
            single = client.post("/api/action", json={"action": "layers", "value": ["snow"]}).get_json()
            self.assertEqual((single["dataset"], single["layers"]), ("snow:class", ["snow:class"]))
            self.assertEqual(len(single["stations"]), 1)

    def test_own_station_only_affects_layers_for_its_sensors(self):
        mapping = {"temperature": "temperature:continuous", "snow": "snow:class"}
        with patch.dict("station_app.PHENOMENON_DATASETS", mapping, clear=True):
            scenario = Scenario(FakeWeatherEngine())
            scenario.apply(dict(action="layers", value=["temperature", "snow"]))
            base = scenario.base().copy()
            scenario.apply(dict(action="add", lon=5, lat=60, sensors=["snow"]))
            np.testing.assert_array_equal(scenario.surface(), base)
            with scenario.as_dataset("snow:class"):
                self.assertFalse(np.array_equal(scenario.surface(), scenario.base()))
            state = scenario.state()
            self.assertEqual(state["stations"][0]["sensors"], ["snow"])
            self.assertTrue(all("sensors" in marker for marker in state["fixed_stations"]))
            with self.assertRaises(ValueError):
                scenario.apply(dict(action="add", lon=5, lat=60, sensors=[]))

    def test_pross_mask_removes_unmatched_stations_from_calculation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ProssStasjon.gpkg"
            path.write_bytes(b"")
            with patch("station_app.pross_excluded", return_value=frozenset([1])):
                scenario = Scenario(FakeWeatherEngine())
                scenario.apply(dict(action="pross_mask", value=True))
                self.assertEqual(scenario.fixed()["station_id"].tolist(), ["SN1"])
                profile = scenario.profile()
                np.testing.assert_array_equal(scenario.base(), profile["second"])
                state = scenario.state()
                self.assertTrue(state["pross_only"])
                self.assertEqual(state["masked_count"], 1)
                with self.assertRaises(ValueError):
                    scenario.apply(dict(action="select", kind="frost", index=1))

    def test_station_edits_follow_station_id_through_every_layer(self):
        mapping = {"temperature": "temperature:continuous", "snow": "snow:class"}
        with patch.dict("station_app.PHENOMENON_DATASETS", mapping, clear=True):
            scenario = Scenario(FakeWeatherEngine())
            scenario.apply(dict(action="layers", value=["temperature", "snow"]))
            scenario.apply(dict(action="remove_frost", station_id="SN2"))
            self.assertEqual(scenario.frost_edits, {1: dict(removed=True)})
            with scenario.as_dataset("snow:class"):
                self.assertEqual(scenario.frost_edits, {1: dict(removed=True)})
            state = scenario.state()
            self.assertEqual([edit["station_id"] for edit in state["frost_edits"]], ["SN2"])
            self.assertTrue(next(m for m in state["fixed_stations"] if m["station_id"] == "SN2")["removed"])
            scenario.apply(dict(action="layers", value=["snow"]))
            self.assertEqual(scenario.frost_edits, {1: dict(removed=True)})
            scenario.apply(dict(action="restore_frost"))
            self.assertEqual((scenario.frost_edits, scenario.station_edits), ({}, {}))
            with patch.object(scenario.engine, "snap", return_value=7), \
                    patch.object(scenario.engine, "station", return_value=dict(id=7, x=500, y=0, lon=5, lat=60, elevation=1)):
                scenario.apply(dict(action="move_frost", station_id="SN1", lon=5, lat=60))
            self.assertEqual(scenario.station_edits["SN1"]["candidate_index"], 7)
            self.assertEqual(scenario.selected["station_id"], "SN1")

    def test_frost_station_can_be_selected_by_station_id_in_every_layer(self):
        mapping = {"temperature": "temperature:continuous", "snow": "snow:class"}
        with patch.dict("station_app.PHENOMENON_DATASETS", mapping, clear=True):
            scenario = Scenario(FakeWeatherEngine())
            scenario.apply(dict(action="layers", value=["temperature", "snow"]))
            scenario.apply(dict(action="select", kind="frost", station_id="SN2"))
            self.assertEqual(scenario.selected, dict(kind="frost", index=1, station_id="SN2"))
            with scenario.as_dataset("snow:class"):
                self.assertEqual(scenario.selected["index"], 1)
            self.assertIn("SN2", scenario.state()["selected_station_label"])
            with self.assertRaises(ValueError):
                scenario.apply(dict(action="select", kind="frost", station_id="SN9"))

    def test_combined_colour_cutoff_masks_low_values_but_keeps_missing(self):
        mapping = {"temperature": "temperature:continuous", "snow": "snow:class"}
        engine = FakeWeatherEngine()
        with patch.dict("station_app.PHENOMENON_DATASETS", mapping, clear=True):
            client = create_app(engine).test_client()
            response = client.post("/api/action", json={"action": "layers", "value": ["temperature", "snow"]})
            self.assertEqual(response.status_code, 200)
            values = np.array([0.2, 0.6, np.nan], dtype=np.float32)
            original = values.copy()
            with patch.object(Scenario, "combined", return_value=(values, "combined")), \
                    patch.object(engine, "render", wraps=engine.render) as render:
                response = client.get("/api/map.png?mode=uncertainty&rmse_min=0.4")
                self.assertEqual(response.status_code, 200)
                np.testing.assert_array_equal(render.call_args.kwargs["cell_mask"], [False, True, True])
                np.testing.assert_array_equal(values, original)

    def test_combined_values_are_each_scaled_to_unit_range(self):
        mapping = {"temperature": "temperature:continuous", "snow": "snow:class"}
        with patch.dict("station_app.PHENOMENON_DATASETS", mapping, clear=True):
            scenario = Scenario(FakeWeatherEngine())
            scenario.apply(dict(action="layers", value=["temperature", "snow"]))
            scenario.apply(dict(action="add", lon=5, lat=60))
            for mode in ("uncertainty", "reduction", "affected"):
                values, kind = scenario.combined(mode)
                self.assertTrue(kind.startswith("combined"))
                self.assertTrue(np.all(np.abs(values[np.isfinite(values)]) <= 2))
            self.assertEqual(scenario.dataset, "temperature:continuous")
            single = Scenario(FakeWeatherEngine())
            single.apply(dict(action="add", lon=5, lat=60))
            np.testing.assert_array_equal(scenario.surface(), single.surface())

    def test_root_sum_square_and_actual_range(self):
        from station_app import combine_layers, combined_range
        parts = [np.array([1, 0, np.nan, -1], np.float32), np.array([1, 0.5, np.nan, 1], np.float32)]
        np.testing.assert_allclose(combine_layers(parts, "sum"), [2, 0.5, np.nan, 0])
        np.testing.assert_allclose(combine_layers(parts, "rss"), [np.sqrt(2), 0.5, np.nan, 0])
        self.assertEqual(combined_range(np.array([0.2, 0.9, np.nan], np.float32), "combined"), (np.float32(0.2).item(), np.float32(0.9).item()))
        self.assertEqual(combined_range(np.array([-0.5, 0.25], np.float32), "combined_change"), (-0.5, 0.5))
        mapping = {"temperature": "temperature:continuous", "snow": "snow:class"}
        with patch.dict("station_app.PHENOMENON_DATASETS", mapping, clear=True):
            client = create_app(FakeWeatherEngine()).test_client()
            client.post("/api/action", json={"action": "layers", "value": ["temperature", "snow"]})
            summed = client.get("/api/combined-range?mode=uncertainty").get_json()
            state = client.post("/api/action", json={"action": "combine", "value": "rss"}).get_json()
            self.assertEqual(state["combine"], "rss")
            rss = client.get("/api/combined-range?mode=uncertainty").get_json()
            self.assertLessEqual(rss["maximum"], summed["maximum"])
            self.assertEqual(client.post("/api/action", json={"action": "combine", "value": "max"}).status_code, 400)

    def test_api_serves_prepared_baselines_but_renders_station_edits_live(self):
        from unittest.mock import Mock

        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        engine.weather = Mock(available=FakeWeather.available, profile=Mock(return_value=profile),
                              svv_profile=Mock(return_value=profile))
        prepared = engine.render(profile["baseline"], "uncertainty").getvalue()
        engine.saved_weather_map = Mock(return_value=prepared)
        engine.render = Mock(wraps=engine.render)
        client = create_app(engine).test_client()
        for mode in ("baseline", "uncertainty"):
            response = client.get(f"/api/map.png?mode={mode}&size=400")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, prepared)
            self.assertEqual(response.headers["X-Map-Cache"], "HIT")
        engine.render.assert_not_called()
        client.post("/api/action", json=dict(action="add", lon=5, lat=60))
        response = client.get("/api/map.png?mode=uncertainty&size=400")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Map-Cache"], "MISS")
        engine.render.assert_called_once()

    def test_prepared_images_reload_without_rendering_and_invalidate_with_source(self):
        from station_weather_maps import WeatherMapCache
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            surface = folder / "surface.npz"
            surface.write_bytes(b"first surface")
            profile = FakeWeather().profile("temperature:continuous")
            profile["cache_path"] = surface
            engine = FakeWeatherEngine()
            engine.render = Mock(wraps=engine.render)
            cache = WeatherMapCache(folder / "images", renderer_version="test-v1")
            limits = (0.01, 6., 2)
            self.assertIsNone(cache.get(engine, profile, 400, limits))
            first = cache.get(engine, profile, 400, limits, build=True)
            self.assertEqual(engine.render.call_count, 1)
            reader = WeatherMapCache(folder / "images", renderer_version="test-v1")
            engine.render.side_effect = AssertionError("Prepared map must not render")
            self.assertEqual(reader.get(engine, profile, 400, limits), first)
            self.assertIsNone(reader.get(engine, profile, 800, limits))
            self.assertIsNone(reader.get(engine, profile, 400, (0.02, 6., 2)))
            other_renderer = WeatherMapCache(folder / "images", renderer_version="test-v2")
            self.assertIsNone(other_renderer.get(engine, profile, 400, limits))
            surface.write_bytes(b"changed prediction surface")
            self.assertIsNone(reader.get(engine, profile, 400, limits))


class WeatherTests(unittest.TestCase):
    def test_portable_smoke_test_urls_exist_in_application(self):
        import ast
        import inspect
        from urllib.parse import urlsplit
        from portable_launcher import smoke_test

        routes = {rule.rule for rule in create_app(FakeWeatherEngine()).url_map.iter_rules()}
        for node in ast.walk(ast.parse(inspect.getsource(smoke_test))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("/"):
                self.assertIn(urlsplit(node.value).path, routes)

    def test_portable_scenario_reuses_bundled_svv_surface_without_recomputation(self):
        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        profile["cache_path"] = Path("bundled_profile.joblib")
        scenario = Scenario(engine)
        scenario.pross_only = True
        with patch.object(engine.weather, "portable_manifest", {}, create=True), \
                patch.object(engine.weather, "profile", return_value=profile), \
                patch.object(engine.weather, "svv_profile", return_value=profile, create=True) as saved, \
                patch("station_weather.restricted_profile", side_effect=AssertionError("Unexpected recomputation")):
            self.assertIs(scenario.profile(), profile)
            saved.assert_called_once_with("temperature:continuous")

    def test_portable_build_removes_embedded_frost_credentials(self):
        import ast
        from build_portable import sanitize_config

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sanitize_config(folder)
            tree = ast.parse((folder / "pipeline_config.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
                        ("FROST_CLIENT_ID", "FROST_CLIENT_SECRET") for target in node.targets):
                    self.assertIsNone(ast.literal_eval(node.value))

    def test_portable_weather_snapshot_loads_without_frost_database(self):
        import joblib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            output.mkdir()
            profile = FakeWeather().profile("temperature:continuous")
            config = FakeWeather.available[0]
            joblib.dump(profile, output / "profile.joblib")
            joblib.dump({"portable_test_model": True}, output / "model.joblib")
            manifest = dict(cell_m=1000, available=[config], profiles={config["id"]: dict(
                profile="outputs/profile.joblib", model="outputs/model.joblib", svv="outputs/profile.joblib")})
            (output / "portable_weather.json").write_text(json.dumps(manifest), encoding="utf-8")
            registry = WeatherRegistry(root, None, None, None)
            with patch("station_weather.sqlite3.connect", side_effect=AssertionError("Unexpected database access")):
                loaded = registry.profile(config["id"])
                self.assertEqual(loaded["model"], {"portable_test_model": True})
                np.testing.assert_array_equal(loaded["baseline"], profile["baseline"])
                np.testing.assert_array_equal(registry.svv_profile(config["id"])["baseline"], profile["baseline"])

    def test_precompute_all_selects_only_station_lab_models(self):
        from types import SimpleNamespace
        from precompute_frost import selected_modes

        with patch.dict("station_weather.PHENOMENON_DATASETS", {"temperature": "temperature:senorge"}, clear=True):
            modes = selected_modes(FakeWeather(), SimpleNamespace(phenomenon=["temperature"], kind="all"))
            self.assertEqual([mode["id"] for mode in modes], ["temperature:senorge"])
            everything = selected_modes(FakeWeather(), SimpleNamespace(
                phenomenon=["temperature"], kind="all", include_experimental=True))
            self.assertEqual(len(everything), 2)
            self.assertEqual(selected_modes(FakeWeather(), SimpleNamespace(phenomenon=["wind"], kind="all")), [])

    def test_senorge_temperature_profile_is_discovered_only_when_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = WeatherRegistry(root, None, None, None)
            self.assertNotIn("temperature:senorge", [item["id"] for item in registry.available])

            outputs = root / "outputs"
            outputs.mkdir()
            prefix = outputs / "temperature_5y_prediction_rmse_senorge"
            for suffix in ("_model.joblib", "_metadata.json", "_station_features.gpkg"):
                prefix.with_name(prefix.name + suffix).touch()

            registry = WeatherRegistry(root, None, None, None)

        profile = next(item for item in registry.available if item["id"] == "temperature:senorge")
        self.assertEqual(profile["kind"], "senorge")
        self.assertEqual(profile["model_grid_m"], 1000)

    def test_large_road_response_is_compressed_when_browser_accepts_gzip(self):
        import gzip
        from unittest.mock import Mock

        roads = Mock()
        roads.view.return_value = dict(type="FeatureCollection", features=[], padding="x" * 70000)
        client = create_app(FakeWeatherEngine(), road_network=roads).test_client()
        response = client.post("/api/roads/view", json=dict(bounds=[5, 58, 12, 64]),
                               headers={"Accept-Encoding": "gzip"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Encoding"], "gzip")
        self.assertEqual(len(json.loads(gzip.decompress(response.data))["padding"]), 70000)

    def test_saved_frost_map_never_predicts_or_calculates_contributions_and_reuses_image(self):
        from unittest.mock import Mock

        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        engine.weather = Mock(available=FakeWeather.available, profile=Mock(return_value=profile),
                              svv_profile=Mock(return_value=profile))
        engine.render = Mock(wraps=engine.render)
        client = create_app(engine).test_client()
        with patch.object(Scenario, "change", side_effect=AssertionError("Map must not calculate contributions")), \
                patch.object(engine, "prediction", side_effect=AssertionError("Saved map must not predict")):
            first = client.get("/api/map.png?mode=uncertainty&size=400")
            second = client.get("/api/map.png?mode=uncertainty&size=400")
            baseline = client.get("/api/map.png?mode=baseline&size=400")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.headers["X-Map-Cache"], "HIT")
        self.assertEqual(baseline.data, first.data)
        self.assertEqual(engine.render.call_count, 1)
        self.assertEqual(engine.render.call_args.kwargs["display_size"], 400)

    def test_moved_frost_station_bypasses_saved_map_cache(self):
        from unittest.mock import Mock

        engine = FakeWeatherEngine()
        engine.render = Mock(wraps=engine.render)
        client = create_app(engine).test_client()
        original = client.get("/api/map.png?mode=uncertainty&size=400")
        self.assertEqual(original.status_code, 200)
        moved = client.post("/api/action", json=dict(action="move_frost", index=1, lon=6, lat=61))
        self.assertEqual(moved.status_code, 200)
        updated = client.get("/api/map.png?mode=uncertainty&size=400")
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.headers["X-Map-Cache"], "MISS")
        self.assertEqual(engine.render.call_count, 2)
        np.testing.assert_array_equal(engine.render.call_args_list[0].args[0], [3, 4, 3])
        np.testing.assert_array_equal(engine.render.call_args_list[1].args[0], [2, 0.5, 2])

    def test_grunnett_stays_original_while_uncertainty_uses_moved_frost_station(self):
        from unittest.mock import Mock

        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        engine.weather = Mock(available=FakeWeather.available, profile=Mock(return_value=profile),
                              svv_profile=Mock(return_value=profile))
        engine.render = Mock(wraps=engine.render)
        client = create_app(engine).test_client()
        original = client.get("/api/map.png?mode=baseline&size=400")
        self.assertEqual(original.status_code, 200)
        moved = client.post("/api/action", json=dict(action="move_frost", index=1, lon=6, lat=61))
        self.assertEqual(moved.status_code, 200)
        baseline = client.get("/api/map.png?mode=baseline&size=400")
        uncertainty = client.get("/api/map.png?mode=uncertainty&size=400")
        self.assertEqual(baseline.data, original.data)
        self.assertEqual(baseline.headers["X-Map-Cache"], "HIT")
        self.assertEqual(uncertainty.headers["X-Map-Cache"], "MISS")
        np.testing.assert_array_equal(engine.render.call_args_list[0].args[0], [3, 4, 3])
        np.testing.assert_array_equal(engine.render.call_args_list[-1].args[0], [2, 0.5, 2])

    def test_corridor_raster_and_road_legend_share_baseline_colour_limits(self):
        from unittest.mock import Mock

        engine = FakeWeatherEngine()
        engine.render = Mock(wraps=engine.render)
        roads = Mock()
        roads.corridor_mask.return_value = np.array([True, False, True])
        roads.view.return_value = {}
        client = create_app(engine, road_network=roads).test_client()
        settings = dict(corridor=True, bounds=[5, 58, 12, 64])
        response = client.get("/api/map.png", query_string=dict(mode="uncertainty", roads=json.dumps(settings)))
        self.assertEqual(response.status_code, 200)
        np.testing.assert_allclose(engine.render.call_args.args[2:4], (.01, 6.))
        result = client.post("/api/roads/view", json=settings)
        self.assertEqual(result.status_code, 200)
        self.assertEqual((result.json["color_min"], result.json["color_max"]), engine.render.call_args.args[2:4])
        roads.corridor_mask.assert_called_once()

    def test_rmse_display_filter_strict_threshold_and_road_intersection(self):
        from unittest.mock import Mock

        engine = FakeWeatherEngine()
        engine.render = Mock(wraps=engine.render)
        roads = Mock()
        roads.corridor_mask.return_value = np.array([True, False, True])
        client = create_app(engine, road_network=roads).test_client()
        response = client.get("/api/map.png?mode=uncertainty&rmse_min=3")
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(engine.render.call_args.kwargs["cell_mask"], [False, True, False])
        settings = dict(corridor=True, bounds=[5, 58, 12, 64])
        response = client.get("/api/map.png", query_string=dict(
            mode="uncertainty", rmse_min=3, roads=json.dumps(settings)))
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(engine.render.call_args.kwargs["cell_mask"], [False, False, False])
        response = client.get("/api/map.png", query_string=dict(
            mode="baseline", rmse_min=2, roads=json.dumps(settings)))
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(engine.render.call_args.kwargs["cell_mask"], [True, False, True])
        response = client.get("/api/map.png?rmse_min=0")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("cell_mask", engine.render.call_args.kwargs)

    def test_rmse_slider_range_uses_actual_surface_and_road_mask(self):
        from unittest.mock import Mock

        roads = Mock()
        roads.corridor_mask.return_value = np.array([True, False, True])
        client = create_app(FakeWeatherEngine(), road_network=roads).test_client()
        result = client.post("/api/rmse-range", json={})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json["maximum"], 4)
        result = client.post("/api/rmse-range", json=dict(
            roads=dict(corridor=True, bounds=[5, 58, 12, 64])))
        self.assertEqual(result.json["maximum"], 3)
        self.assertEqual(result.json["cells"], 2)
        client.post("/api/action", json=dict(action="add", lon=5, lat=60))
        self.assertEqual(client.post("/api/rmse-range", json={}).json["maximum"], 2)
        self.assertEqual(client.post("/api/rmse-range", json=dict(mode="baseline")).json["maximum"], 4)
        roads.corridor_mask.return_value = np.zeros(3, dtype=bool)
        result = client.post("/api/rmse-range", json=dict(
            roads=dict(corridor=True, bounds=[5, 58, 12, 64])))
        self.assertEqual(result.json["maximum"], 0)
        self.assertEqual(result.json["cells"], 0)

    def test_rmse_display_filter_validation_and_preview(self):
        from unittest.mock import Mock

        engine = FakeWeatherEngine()
        engine.render = Mock(wraps=engine.render)
        client = create_app(engine).test_client()
        for value in ("nan", "inf", "-1", "bad"):
            self.assertEqual(client.get("/api/map.png", query_string=dict(rmse_min=value)).status_code, 400)
            self.assertEqual(client.post("/api/preview", json=dict(
                lon=5, lat=60, rmse_min=value)).status_code, 400)
        response = client.post("/api/preview", json=dict(
            lon=5, lat=60, mode="uncertainty", rmse_min=1))
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(engine.render.call_args.kwargs["cell_mask"], [True, False, True])
        roads = Mock()
        roads.corridor_mask.return_value = np.array([True, False, True])
        engine.prediction_at = lambda index, targets, dataset: engine.prediction(index, dataset)[targets]
        client = create_app(engine, road_network=roads).test_client()
        response = client.post("/api/preview", json=dict(
            lon=5, lat=60, mode="uncertainty", rmse_min=2,
            roads=dict(corridor=True, bounds=[5, 58, 12, 64])))
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(engine.render.call_args.kwargs["cell_mask"], [False, False, False])

    def test_switch_keeps_proposed_locations_and_changes_units(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="add", lon=5, lat=60))
        scenario.apply(dict(action="dataset", value="snow:class"))
        self.assertEqual(scenario.added, [1])
        self.assertEqual(scenario.state()["unit"], "score")
        self.assertEqual(scenario.state()["metric"], "Brier score")
        self.assertEqual(scenario.state()["fixed_stations"][0]["kind"], "observation")
        self.assertTrue((scenario.surface() <= 1).all())

    def test_hover_never_commits_a_station(self):
        client = create_app(FakeWeatherEngine()).test_client()
        before = client.get("/api/state").json
        response = client.post("/api/preview", json=dict(lon=5, lat=60, revision=before["revision"]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        after = client.get("/api/state").json
        self.assertEqual(after["stations"], before["stations"])
        self.assertEqual(after["revision"], before["revision"])
        self.assertGreater(int(response.headers["X-Affected-Area-M2"]), 0)
        stale = client.post("/api/preview", json=dict(lon=5, lat=60, revision=-1))
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(client.post("/api/preview", json=[]).status_code, 400)

    def test_unknown_model_does_not_change_scenario(self):
        scenario = Scenario(FakeWeatherEngine())
        with self.assertRaises(ValueError):
            scenario.apply(dict(action="dataset", value="missing:class"))
        self.assertEqual(scenario.dataset, "temperature:continuous")

    def test_database_availability_filters_values_dates_and_country(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "frost.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE stations(station_id TEXT,name TEXT,lat REAL,lon REAL,elevation REAL,inside_norway INTEGER)")
                connection.execute("CREATE TABLE observations(station_id TEXT,reference_time TEXT,element TEXT,value REAL)")
                connection.executemany("INSERT INTO stations VALUES(?,?,?,?,?,?)", [
                    ("valid", "A", 60, 10, 100, 1), ("null", "B", 60, 10, 100, 1),
                    ("old", "C", 60, 10, 100, 1), ("outside", "D", 60, 10, 100, 0)])
                connection.executemany("INSERT INTO observations VALUES(?,?,?,?)", [
                    ("valid", "2026-10-14T00:00:00.000Z", "air_temperature", 1),
                    ("null", "2026-10-14T00:00:00.000Z", "air_temperature", None),
                    ("old", "2019-01-01T00:00:00.000Z", "air_temperature", 1),
                    ("outside", "2026-10-14T00:00:00.000Z", "air_temperature", 1)])
                connection.commit()
            filtered, _, _ = observation_stations(database, root / "absent.csv", "air_temperature",
                                                   station_ids=["valid", "outside"])
            self.assertEqual(filtered["station_id"].tolist(), ["valid"])
            records, start, _ = observation_stations(database, root / "absent.csv", "air_temperature")
            self.assertEqual(records["station_id"].tolist(), ["valid"])
            self.assertTrue(start.startswith("2021-10-14"))

    def test_availability_accepts_native_and_derived_iso_timestamp_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "frost.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE stations(station_id TEXT,name TEXT,lat REAL,lon REAL,elevation REAL,inside_norway INTEGER)")
                connection.execute("CREATE TABLE observations(station_id TEXT,reference_time TEXT,element TEXT,value REAL)")
                connection.executemany("INSERT INTO stations VALUES(?,?,?,?,?,?)",
                                       [("native", "A", 60, 10, 100, 1), ("derived", "E18 B", 60, 10, 100, 1)])
                connection.executemany("INSERT INTO observations VALUES(?,?,?,?)", [
                    ("native", "2026-11-06T00:00:00.000Z", "sum(precipitation_amount P1D)", 1),
                    ("derived", "2026-11-05T00:00:00Z", "sum(precipitation_amount P1D)", 2)])
                connection.commit()
            records, start, end = observation_stations(
                database, root / "absent.csv", "sum(precipitation_amount P1D)")
            self.assertEqual(set(records.station_id), {"native", "derived"})
            self.assertEqual(start, "2021-11-06T00:00:00.000Z")
            self.assertEqual(end, "2026-11-07T00:00:00.000Z")

    def test_paired_features_match_greedy_predictor(self):
        class SumModel:
            def predict(self, features):
                return features.sum(axis=1)

        source = {key: np.array([1, 2], dtype=np.float32) for key in CACHE_KEYS}
        target = {key: np.array([3, 4], dtype=np.float32) for key in CACHE_KEYS}
        actual = paired_predictions(SumModel(), source, target, np.array([0, 1]), np.array([0, 1]), False)
        expected = [SumModel().predict(feature_matrix(
            {key: values[[index]] for key, values in source.items()}, target, np.array([index])))[0]
            for index in range(2)]
        np.testing.assert_allclose(actual, expected)


class StationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.values = [np.array([12 - index, index + 1, 5, index % 3 + 2], dtype=np.float32)
                       for index in range(12)]
        self.engine.grid = pd.DataFrame({"x": [0, 1000, 2000, 3000], "y": [0, 0, 0, 0]})
        self.engine.stations = pd.DataFrame({"x": np.arange(12) * 1000, "y": np.zeros(12)})
        self.engine.frames = np.minimum.accumulate(np.stack(self.values), axis=0)
        self.engine.greedy_prediction = lambda index: self.values[index]
        self.engine.greedy_prediction_at = lambda index, targets: self.values[index][targets]
        self.scenario = Scenario(self.engine, baseline=12)

    def test_twelve_stations_selects_twelfth(self):
        state = self.scenario.state()
        self.assertEqual(state["selected_station"], dict(kind="greedy", index=11))
        self.assertEqual(state["selected_station_label"], "Foreslått stasjon 12")
        self.assertEqual(state["affected_area_km2"], 1)
        np.testing.assert_array_equal(self.scenario.change()[2], [1, 0, 0, 0])

    def test_selected_first_station_compares_current_network(self):
        self.scenario.apply(dict(action="select", kind="greedy", index=0))
        np.testing.assert_array_equal(self.scenario.change()[2], [0, 1, 0, 0])
        self.assertEqual(self.scenario.state()["affected_area_km2"], 1)

    def test_optimized_contribution_matches_full_recalculation(self):
        for index in range(12):
            self.scenario.apply(dict(action="select", kind="greedy", index=index))
            current, before, reduction = self.scenario.change()
            expected = np.minimum.reduce([values for rank, values in enumerate(self.values) if rank != index])
            np.testing.assert_array_equal(before, expected)
            np.testing.assert_array_equal(reduction, expected - current)

    def test_area_uses_target_resolution_in_square_kilometres(self):
        self.engine.cell_m = 2500
        self.assertEqual(self.scenario.state()["affected_area_km2"], 6.25)
        self.engine.cell_m = 5000
        self.assertEqual(self.scenario.state()["affected_area_km2"], 25)

    def test_single_station_has_no_without_network_comparison(self):
        self.scenario.apply(dict(action="baseline", count=1))
        self.assertIsNone(self.scenario.state()["affected_area_km2"])

    def test_invalid_selection_does_not_change_network(self):
        for kind, index in [("greedy", 12), ("manual", 99), ("observation", 0)]:
            with self.assertRaises(ValueError):
                self.scenario.apply(dict(action="select", kind=kind, index=index))
        self.assertEqual(self.scenario.baseline, 12)

    def test_manual_selection_removes_only_selected_station(self):
        self.scenario.network = "empty"
        self.scenario.added = [1, 2]
        predictions = {1: np.array([3, 2, 7, 4], dtype=np.float32),
                       2: np.array([4, 1, 8, 3], dtype=np.float32)}
        self.engine.prediction = lambda index: predictions[index]
        self.scenario.apply(dict(action="select", kind="manual", index=1))
        np.testing.assert_array_equal(self.scenario.change()[2], [1, 0, 1, 0])
        self.assertEqual(self.scenario.state()["affected_area_km2"], 2)

    def test_coarse_replay_only_computes_requested_prefix(self):
        calls = []

        def prediction(index):
            calls.append(index)
            return self.values[index]

        self.engine.greedy_prediction = prediction
        replay = GreedySurfaces(self.engine)
        np.testing.assert_array_equal(replay[2], self.engine.frames[2])
        replay[1]
        self.assertEqual(calls, [0, 1, 2])


class ExactNetworkTests(unittest.TestCase):
    class DistanceModel:
        def predict(self, features):
            return features[:, 0]

    def setUp(self):
        self.source = self.cache([0, 10, 20])
        self.target = self.cache([0, 5, 10, 19])
        self.expected = np.abs(self.source["x"][:, None] - self.target["x"][None, :])

    @staticmethod
    def cache(xs):
        result = {key: np.zeros(len(xs), dtype=np.float32) for key in CACHE_KEYS}
        result["x"] = np.array(xs, dtype=np.float32)
        return result

    def test_exact_best_and_runner_up_include_all_sources(self):
        best, second, winner = exact_network(self.DistanceModel(), self.source, self.target, False,
                                             source_batch=2, pair_batch=3)
        np.testing.assert_array_equal(best, self.expected.min(axis=0))
        np.testing.assert_array_equal(second, np.sort(self.expected, axis=0)[1])
        for index in range(3):
            without = np.where(winner == index, second, best)
            np.testing.assert_array_equal(without, np.delete(self.expected, index, axis=0).min(axis=0))

    def test_equal_best_station_has_zero_unique_contribution(self):
        best, second, _ = exact_network(self.DistanceModel(), self.source, self.target, False)
        self.assertEqual(best[1], second[1])

    def test_checkpoint_resumes_without_losing_runner_up(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.npz"

            def stop(completed, total):
                if completed == 2:
                    raise RuntimeError("avbrudd")

            with self.assertRaises(RuntimeError):
                exact_network(self.DistanceModel(), self.source, self.target, False,
                              path, stop, source_batch=1)
            with np.load(path) as checkpoint:
                self.assertEqual(int(checkpoint["completed"]), 2)
            best, second, winner = exact_network(self.DistanceModel(), self.source, self.target, False,
                                                 path, source_batch=1)
            np.testing.assert_array_equal(best, self.expected.min(axis=0))
            np.testing.assert_array_equal(second, np.sort(self.expected, axis=0)[1])
            np.testing.assert_array_equal(winner, self.expected.argmin(axis=0))

    def test_nearest_station_approximation_is_not_supported(self):
        with self.assertRaises(ValueError):
            WeatherRegistry(".", None, None, None, neighbors=12)
        with self.assertRaises(ValueError):
            StationEngine(Path("unused.npy"), frost_neighbors=12)

    def test_select_frost_station_compares_network_with_and_without(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="comparison", value="selected"))
        scenario.apply(dict(action="select", kind="frost", index=1))
        np.testing.assert_array_equal(scenario.change()[2], [1, 1, 3])
        self.assertEqual(scenario.state()["affected_area_km2"], 3)
        self.assertEqual(scenario.state()["selected_station_label"], "SN2 · Second")

    def test_api_can_remove_and_restore_a_frost_station(self):
        client = create_app(FakeWeatherEngine()).test_client()
        self.assertEqual(client.get("/api/state").status_code, 200)
        removed = client.post("/api/action", json=dict(action="remove_frost", index=1))
        self.assertEqual(removed.status_code, 200)
        self.assertEqual(removed.json["fixed_count"], 1)
        self.assertTrue(removed.json["frost_edits"][0]["removed"])
        removed_again = client.post("/api/action", json=dict(action="remove_frost", index=0))
        self.assertEqual(removed_again.status_code, 200)
        self.assertEqual(removed_again.json["fixed_count"], 0)
        self.assertEqual(len(removed_again.json["frost_edits"]), 2)
        restored = client.post("/api/action", json=dict(action="restore_frost"))
        self.assertEqual(restored.status_code, 200)
        self.assertEqual(restored.json["fixed_count"], 2)
        self.assertEqual(restored.json["frost_edits"], [])

    def test_move_frost_station_replaces_its_exact_baseline_contribution(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="move_frost", index=1, lon=6, lat=61))
        np.testing.assert_array_equal(scenario.base(), [2, 0.5, 2])
        state = scenario.state()
        self.assertEqual(state["frost_edits"], [dict(index=1, removed=False, station_id="SN2", name="Second")])
        self.assertEqual(state["fixed_stations"][1]["index"], 1)
        original_lon, original_lat = scenario.engine.to_lonlat.transform(2000, 0)
        self.assertEqual(state["fixed_stations"][1]["original_position"], dict(lon=original_lon, lat=original_lat))
        self.assertNotIn("original_position", state["fixed_stations"][0])
        self.assertEqual((state["fixed_stations"][1]["lon"], state["fixed_stations"][1]["lat"]), (6, 61))
        self.assertEqual(state["selected_station_label"], "SN2 · Second")

    def test_moved_frost_comparison_preserves_increases_against_original_baseline(self):
        engine = FakeWeatherEngine()
        with patch.object(engine, "prediction", return_value=np.array([2, 5, 4], dtype=np.float32)):
            scenario = Scenario(engine)
            scenario.apply(dict(action="comparison", value="baseline"))
            scenario.apply(dict(action="move_frost", index=1, lon=6, lat=61))
            current, before, reduction = scenario.change()
            np.testing.assert_array_equal(current, [2, 5, 4])
            np.testing.assert_array_equal(before, [3, 4, 3])
            np.testing.assert_array_equal(reduction, [1, -1, -1])
            state = scenario.state()
            self.assertEqual(state["comparison"], "baseline")
            self.assertAlmostEqual(state["mean_reduction"], -1 / 3)
            self.assertEqual(state["affected_area_km2"], 3)
            scenario.apply(dict(action="threshold", value=1))
            self.assertEqual(scenario.state()["affected_area_km2"], 0)

    def test_removed_frost_remains_selectable_and_reports_increased_uncertainty(self):
        client = create_app(FakeWeatherEngine()).test_client()
        client.post("/api/action", json=dict(action="comparison", value="baseline"))
        removed = client.post("/api/action", json=dict(action="remove_frost", index=1))
        self.assertEqual(removed.status_code, 200)
        state = removed.json
        self.assertEqual(state["selected_station"], dict(kind="frost", index=1, station_id="SN2"))
        self.assertEqual(state["fixed_count"], 1)
        self.assertEqual(len(state["fixed_stations"]), 2)
        self.assertTrue(state["fixed_stations"][1]["removed"])
        self.assertAlmostEqual(state["mean_reduction"], -5 / 3)
        self.assertEqual(state["affected_area_km2"], 3)
        selected = client.post("/api/action", json=dict(action="select", kind="frost", index=1))
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(selected.json["selected_station_label"], "SN2 · Second")
        restored = client.post("/api/action", json=dict(action="restore_frost"))
        self.assertEqual(restored.json["comparison"], "baseline")
        self.assertFalse(any(station["removed"] for station in restored.json["fixed_stations"]))

    def test_removed_moved_station_returns_as_gray_marker_at_original_location(self):
        scenario = Scenario(FakeWeatherEngine())
        original = scenario.state()["fixed_stations"][1]
        scenario.apply(dict(action="move_frost", index=1, lon=6, lat=61))
        scenario.apply(dict(action="remove_frost", index=1))
        marker = scenario.state()["fixed_stations"][1]
        self.assertTrue(marker["removed"])
        self.assertNotIn("original_position", marker)
        self.assertEqual((marker["lon"], marker["lat"]), (original["lon"], original["lat"]))

    def test_signed_change_map_and_preview_use_original_baseline(self):
        engine = FakeWeatherEngine()
        client = create_app(engine).test_client()
        client.post("/api/action", json=dict(action="comparison", value="baseline"))
        client.post("/api/action", json=dict(action="remove_frost", index=1))
        with patch.object(engine, "render", wraps=engine.render) as render:
            response = client.get("/api/map.png?mode=reduction")
            self.assertEqual(response.status_code, 200)
            np.testing.assert_array_equal(render.call_args.args[0], [-1, -1, -3])
            self.assertEqual(render.call_args.args[1], "change")
            response = client.get("/api/map.png?mode=affected")
            np.testing.assert_array_equal(render.call_args.args[0], [-1, -1, -1])
            self.assertEqual(render.call_args.args[1], "affected_change")
            preview = client.post("/api/preview", json=dict(lon=6, lat=61, mode="reduction"))
            self.assertEqual(preview.status_code, 200)
            np.testing.assert_array_equal(render.call_args.args[0], [1, 3.5, 1])
            self.assertEqual(render.call_args.args[1], "change")
            self.assertEqual(float(preview.headers["X-Affected-Area-Km2"]), 3)

    def test_signed_frost_export_preserves_negative_changes_and_counts_both_directions(self):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin

        engine = FakeWeatherEngine()
        engine.raster_shape = (1, 3)
        engine.rows = np.array([0, 0, 0])
        engine.columns = np.array([0, 1, 2])
        engine.transform = from_origin(-500, 500, 1000, 1000)
        scenario = Scenario(engine)
        scenario.apply(dict(action="comparison", value="baseline"))
        with patch.object(engine, "prediction", return_value=np.array([2, 5, 4], dtype=np.float32)):
            scenario.apply(dict(action="move_frost", index=1, lon=6, lat=61))
            with ZipFile(scenario.export()) as archive:
                with MemoryFile(archive.read("selected_reduction.tif")) as memory:
                    with memory.open() as raster:
                        np.testing.assert_array_equal(raster.read(1), [[1, -1, -1]])
                        self.assertEqual(raster.tags()["comparison"], "baseline")
                with MemoryFile(archive.read("selected_affected.tif")) as memory:
                    with memory.open() as raster:
                        np.testing.assert_array_equal(raster.read(1), [[1, 1, 1]])
                metadata = json.loads(archive.read("scenario.json"))
                self.assertEqual(metadata["affected_area_km2"], 3)
                self.assertIn("grunnnett", metadata["contribution_method"])

    def test_edited_frost_corridor_preview_keeps_increases_and_original_reference(self):
        from unittest.mock import Mock

        engine = FakeWeatherEngine()
        engine.prediction_at = Mock(return_value=np.array([5, 4], dtype=np.float32))
        roads = Mock()
        roads.corridor_mask.return_value = np.array([False, True, True])
        client = create_app(engine, road_network=roads).test_client()
        client.post("/api/action", json=dict(action="comparison", value="baseline"))
        client.post("/api/action", json=dict(action="remove_frost", index=1))
        with patch.object(engine, "render", wraps=engine.render) as render:
            response = client.post("/api/preview", json=dict(
                lon=6, lat=61, mode="affected", roads=dict(bounds=[5, 58, 12, 64], corridor=True)))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["X-Preview-Scope"], "roads")
            self.assertEqual(float(response.headers["X-Mean-Reduction"]), -1)
            self.assertEqual(float(response.headers["X-Affected-Area-Km2"]), 2)
            np.testing.assert_array_equal(render.call_args.args[0], [0, -1, -1])
            self.assertEqual(render.call_args.args[1], "affected_change")
            np.testing.assert_array_equal(render.call_args.kwargs["cell_mask"], [False, True, True])

    def test_remove_and_restore_frost_station_preserves_stable_indices(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="remove_frost", index=1))
        np.testing.assert_array_equal(scenario.base(), [4, 5, 6])
        state = scenario.state()
        self.assertEqual(state["fixed_count"], 1)
        self.assertEqual([station["index"] for station in state["fixed_stations"]], [0, 1])
        self.assertEqual([station["removed"] for station in state["fixed_stations"]], [False, True])
        self.assertTrue(state["frost_edits"][0]["removed"])
        scenario.apply(dict(action="restore_frost"))
        self.assertEqual(scenario.state()["fixed_count"], 2)
        self.assertEqual(scenario.state()["frost_edits"], [])

    def test_removing_all_frost_stations_produces_an_empty_network(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="remove_frost", index=0))
        scenario.apply(dict(action="remove_frost", index=1))
        self.assertIsNone(scenario.base())
        state = scenario.state()
        self.assertEqual(state["fixed_count"], 0)
        self.assertIsNone(state["mean_rmse"])
        self.assertIsNone(state["mean_reduction"])
        self.assertEqual(len(state["fixed_stations"]), 2)
        self.assertTrue(all(station["removed"] for station in state["fixed_stations"]))

    def test_multiple_frost_moves_recompute_affected_cells_from_remaining_stations(self):
        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        stations = profile["stations"].copy()
        third = stations.iloc[[0]].copy()
        third["station_id"], third["name"], third["x"] = "SN3", "Third", 4000
        stations = pd.concat([stations, third], ignore_index=True)
        for key in CACHE_KEYS:
            if key not in stations:
                stations[key] = 0
        stations["objtype_code"] = 0
        profile["stations"] = stations
        profile["winner"] = np.array([0, 1, 0], dtype=np.int32)
        profile["model"] = object()
        engine.cache = {key: np.zeros(3, dtype=np.int16 if key == "objtype_code" else np.float32)
                        for key in CACHE_KEYS}
        locations = {
            3: dict(id=3, x=3000.0, y=0.0, lon=6.0, lat=60.0, elevation=15.0),
            4: dict(id=4, x=5000.0, y=0.0, lon=7.0, lat=61.0, elevation=20.0),
        }
        scenario = Scenario(engine)
        with patch.object(engine.weather, "profile", return_value=profile), \
            patch.object(engine, "snap", side_effect=[3, 4]), \
                patch.object(engine, "station", side_effect=lambda index: locations.get(index, FakeEngine.station(engine, index))), \
                patch("station_app.exact_network", return_value=(np.array([8, 9, 10], dtype=np.float32), None, None)) as recompute:
            scenario.apply(dict(action="move_frost", index=0, lon=6, lat=60))
            scenario.apply(dict(action="move_frost", index=1, lon=7, lat=61))
            np.testing.assert_array_equal(scenario.base(), [2, 0.5, 2])
        self.assertEqual(len(scenario.state()["frost_edits"]), 2)
        recompute.assert_called_once()

    def test_frost_contribution_respects_added_stations(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="comparison", value="selected"))
        scenario.apply(dict(action="add", lon=5, lat=60))
        scenario.apply(dict(action="select", kind="frost", index=1))
        np.testing.assert_array_equal(scenario.change()[2], [0, 0, 0])

    def test_proposed_network_is_retained_for_class_model(self):
        scenario = Scenario(FakeWeatherEngine())
        scenario.apply(dict(action="baseline", count=2))
        scenario.apply(dict(action="dataset", value="snow:class"))
        self.assertEqual(scenario.network, "greedy")
        self.assertEqual(scenario.baseline, 2)
        np.testing.assert_allclose(scenario.surface(), FakeEngine.frames[-1] / 20)
        scenario.apply(dict(action="select", kind="greedy", index=0))
        self.assertEqual(scenario.state()["unit"], "score")

    def test_pending_exact_surface_is_not_replaced_by_candidate_prediction(self):
        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        profile.update(baseline=None, status=dict(state="running", percent=25))

        class PendingWeather:
            available = FakeWeather.available

            def profile(self, identifier):
                return profile

        engine.weather = PendingWeather()
        client = create_app(engine).test_client()
        state = client.get("/api/state").json
        self.assertIsNone(state["mean_rmse"])
        self.assertFalse(state["frost_ready"])
        self.assertEqual(client.post("/api/preview", json=dict(lon=5, lat=60)).status_code, 409)
        self.assertEqual(client.get("/api/export").status_code, 400)


class SavedPlacementTests(unittest.TestCase):
    def setUp(self):
        from types import MethodType
        from unittest.mock import Mock

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.prefix = Path(directory.name) / "temperature_5y_prediction_rmse"
        self.path = self.prefix.with_name(self.prefix.name + "_error_reduction.gpkg")
        self.path.touch()
        self.engine = FakeWeatherEngine()
        self.engine.grid = pd.DataFrame(dict(x=[0., 5000., 10000.], y=[0., 0., 0.]))
        self.engine.cell_m = 5000
        self.engine.weather = FakeWeather()
        self.engine.weather.available = [dict(FakeWeather.available[0], prefix=str(self.prefix))]
        self.engine.at_resolution = Mock(return_value=self.engine)
        self.engine.placement_layer = MethodType(StationEngine.placement_layer, self.engine)
        patcher = patch("station_app.pyogrio.read_dataframe", return_value=pd.DataFrame(dict(
            x=[10000., 0., 5000.], y=[0., 0., 0.], total_error_reduction=[30., 10., 20.])))
        self.reader = patcher.start()
        self.addCleanup(patcher.stop)

    def test_saved_values_align_by_coordinates_and_load_once(self):
        first = self.engine.placement_layer("temperature:continuous")
        np.testing.assert_array_equal(first["values"], [10, 20, 30])
        self.assertEqual(first["cell_m"], 5000)
        self.assertEqual(first["maximum"], 30)
        self.assertIs(first, self.engine.placement_layer("temperature:continuous"))
        self.assertEqual(self.reader.call_count, 1)

    def test_placement_endpoints_do_not_compute_scenario_or_predict(self):
        from unittest.mock import Mock

        self.engine.render = Mock(wraps=self.engine.render)
        client = create_app(self.engine).test_client()
        with patch.object(Scenario, "change", side_effect=AssertionError("Saved map must not calculate scenario")), \
                patch.object(self.engine, "prediction", side_effect=AssertionError("Saved map must not predict")), \
                patch.object(self.engine.weather, "profile", side_effect=AssertionError("Saved map must not load Frost baseline")):
            metadata = client.get("/api/placement")
            first = client.get("/api/placement.png?size=400")
            second = client.get("/api/placement.png?size=400")
        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(metadata.json["field"], "total_error_reduction")
        self.assertNotIn("values", metadata.json)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.headers["X-Map-Cache"], "HIT")
        self.assertEqual(first.data, second.data)
        self.assertEqual(self.engine.render.call_count, 1)
        self.assertEqual(client.get("/api/placement.png?size=1").status_code, 400)

    def test_finer_live_grid_delegates_to_fixed_5000m_source(self):
        from types import SimpleNamespace

        finer = SimpleNamespace(at_resolution=lambda spacing: self.engine)
        layer = StationEngine.placement_layer(finer, "temperature:continuous")
        self.assertEqual(layer["cell_m"], 5000)
        np.testing.assert_array_equal(layer["values"], [10, 20, 30])

    def test_missing_saved_map_is_reported_without_recalculation(self):
        self.path.unlink()
        response = create_app(self.engine).test_client().get("/api/placement")
        self.assertEqual(response.status_code, 400)
        self.assertIn("mangler", response.json["error"])
        self.reader.assert_not_called()

    def test_wrong_coordinates_and_negative_gains_are_rejected(self):
        self.reader.return_value.loc[0, "x"] = 15000
        with self.assertRaisesRegex(ValueError, "terrengrutenettet"):
            self.engine.placement_layer("temperature:continuous")
        self.reader.return_value.loc[0, "x"] = 10000
        self.reader.return_value.loc[0, "total_error_reduction"] = -1
        with self.assertRaisesRegex(ValueError, "ugyldig"):
            self.engine.placement_layer("temperature:continuous")


class RoadTests(unittest.TestCase):
    def test_snapshot_includes_offscreen_roads_without_display_limit(self):
        import base64
        import gzip
        from station_snapshot import road_chunks

        self.network.limit = 1
        settings = dict(categories=["E", "R", "F"], classes=self.network.metadata["classes"])
        settings["bounds"] = [0, 0, 1, 1]
        expected, truncated = self.network.read(self.bounds, settings)
        self.assertTrue(truncated)
        exported = road_chunks(self.network, settings)
        features = [feature for chunk in exported["chunks"]
                    for feature in json.loads(gzip.decompress(base64.b64decode(chunk["data"])))["features"]]
        self.assertGreater(len(features), len(expected))
        self.assertEqual(len(features), len(self.network.matching_rows(self.network.geographic_bounds, settings)))
        self.assertEqual(exported["count"], len(features))
        self.assertTrue(all(len(feature["bbox"]) == 4 for feature in features))
        subset = road_chunks(self.network, dict(categories=["R"], classes=["3"], reference="Rv3"))
        self.assertEqual(subset["count"], 1)

    def test_prepared_overview_uses_saved_geometry_samples_and_fixed_colour_bands(self):
        settings = dict(colour="error", colour_min=.5, colour_max=8.)
        self.network.prepare_display(self.engine)
        network = RoadNetwork(self.path)
        rows = network.matching_rows(self.bounds, settings)
        with patch("station_roads.pyogrio.read_dataframe", side_effect=AssertionError("No GeoPackage reads in prepared view")), \
                patch("station_roads.shapely.simplify", side_effect=AssertionError("No per-request simplification")), \
                patch.object(network, "grid_values", side_effect=AssertionError("No per-request spatial sampling")):
            first = network.prepared_overview(self.engine, np.array([2., 4., 9.]), self.bounds, settings, rows)
            second = network.prepared_overview(self.engine, np.array([2., 4., 900.]), self.bounds, settings, rows)
        values = [feature["properties"] for feature in first["features"] if feature["properties"]["mean_error"] is not None]
        changed = [feature["properties"] for feature in second["features"] if feature["properties"]["mean_error"] is not None]
        self.assertEqual(first["summary"]["links"], 4)
        self.assertAlmostEqual(values[0]["mean_error"], 3.)
        self.assertEqual([value["colour_value"] for value in values], [value["colour_value"] for value in changed])

    def test_prepared_overview_preserves_exact_boundary_lengths(self):
        settings = dict(categories=["R"], classes=["3"])
        bounds = list(self.network.to_lonlat.transform_bounds(
            self.origin[0]+600, self.origin[1]-200, self.origin[0]+1200, self.origin[1]+200))
        roads, _ = self.network.read(bounds, settings)
        expected = float(roads.geometry.length.sum()/1000)
        self.network.prepare_display(self.engine)
        result = self.network.prepared_overview(self.engine, None, bounds, settings,
                                               self.network.matching_rows(bounds, settings))
        self.assertAlmostEqual(result["summary"]["displayed_km"], expected)

    def test_road_grid_precompute_dry_run_does_not_read_or_write_grid_caches(self):
        from precompute_road_grid import main

        root = Path(self.directory.name)
        (root / "data").mkdir()
        (root / "data" / "ProssExport.gdb").mkdir()
        (root / "outputs").mkdir()
        (root / "outputs" / "norway_grid_5000m.gpkg").touch()
        before = set(root.rglob("*"))
        with patch("precompute_road_grid.HERE", root), patch("precompute_road_grid.OUT", root / "outputs"), \
                patch("precompute_road_grid.RoadNetwork", side_effect=AssertionError("Dry run must not build caches")), \
                patch("precompute_road_grid.load_grid_all_fields", side_effect=AssertionError("Dry run must not load grids")), \
                patch("builtins.print"):
            self.assertEqual(main(["--cell-m", "5000", "--dry-run"]), 0)
        self.assertEqual(set(root.rglob("*")), before)

    def test_precomputed_corridor_matches_spatial_filter_without_reading_road_geometry(self):
        settings = dict(categories=["R"], classes=["3"], reference="Rv3")
        expected = self.network.corridor_mask(self.engine, self.bounds, settings, 100).copy()
        path = self.network.prepare_grid(self.engine, 500)
        self.assertTrue(path.exists())
        network = RoadNetwork(self.path)
        with patch.object(network, "read", side_effect=AssertionError("Interior roads must use saved grid links")):
            actual = network.corridor_mask(self.engine, self.bounds, settings, 100)
            excluded = network.corridor_mask(self.engine, self.bounds, dict(categories=["E"]), 100)
            recoloured = network.corridor_mask(self.engine, self.bounds, dict(settings, colour="error", zoom=3), 100)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.any())
        self.assertFalse(excluded.any())
        self.assertIs(actual, recoloured)

    def test_precomputed_corridor_keeps_exact_viewport_edge_clipping(self):
        settings = dict(categories=["R"], classes=["3"])
        bounds = list(self.network.to_lonlat.transform_bounds(
            self.origin[0]+600, self.origin[1]-200, self.origin[0]+1200, self.origin[1]+200))
        expected = self.network.corridor_mask(self.engine, bounds, settings, 0).copy()
        self.network.prepare_grid(self.engine, 500)
        network = RoadNetwork(self.path)
        with patch.object(network, "read", wraps=network.read) as read:
            actual = network.corridor_mask(self.engine, bounds, settings, 0)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(actual[0])
        self.assertTrue(actual[1])
        self.assertIsNotNone(read.call_args.kwargs["fid_filter"])

    def test_precomputed_corridor_is_grid_specific_and_does_not_cover_larger_distances(self):
        self.network.prepare_grid(self.engine, 500)
        self.assertIsNotNone(self.network.load_grid_links(self.engine, 500))
        self.assertIsNone(self.network.load_grid_links(self.engine, 501))
        self.engine.grid = self.engine.grid.copy()
        self.engine.grid["x"] += 100
        self.assertIsNone(self.network.load_grid_links(self.engine, 100))

    def test_uncertainty_colouring_is_available_in_overview_and_changes_with_surface(self):
        roads, _ = self.network.read(self.bounds, {})
        roads = pd.concat([roads] * 600, ignore_index=True)
        with patch.object(self.network, "read", return_value=(roads, False)):
            first = self.network.view(self.engine, np.array([2, 4, 9], dtype=np.float32),
                                      None, .1, self.bounds, dict(colour="error"), zoom=5)
            second = self.network.view(self.engine, np.array([1, 2, 9], dtype=np.float32),
                                       None, .1, self.bounds, dict(colour="error"), zoom=5)
        first_values = [feature["properties"]["mean_error"] for feature in first["features"]
                        if feature["properties"]["mean_error"] is not None]
        second_values = [feature["properties"]["mean_error"] for feature in second["features"]
                         if feature["properties"]["mean_error"] is not None]
        self.assertTrue(first_values)
        self.assertTrue(second_values)
        self.assertGreater(max(first_values), max(second_values))
        self.assertEqual(sum(feature["properties"]["links"] for feature in first["features"]), 2400)

    def setUp(self):
        import geopandas as gpd
        import pyogrio
        from shapely.geometry import LineString

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "roads.gpkg"
        self.origin = Transformer.from_crs(4326, 25833, always_xy=True).transform(10.75, 59.91)
        features = gpd.GeoDataFrame({
            "edge_id": [1, 2, 3, 4, 5], "roadclass": ["3", "0", "3", "1", "7"],
            "roadtype": ["Enkel bilveg"] * 5, "roadid": ["{Rv3}", "{E6}", "{F17}", '{E6,"Rv3"}', None],
            "length_m": [1000.] * 5, "streetname": ["Testveg", None, "Fylkesveg", "Felles", None],
            "municipality": [301] * 5, "isferry": [0] * 5, "isbridge": [0] * 5, "istunnel": [0] * 5,
        }, geometry=[LineString([(self.origin[0], self.origin[1] + index * 1000, 100),
                                 (self.origin[0] + 1000, self.origin[1] + index * 1000, 101)])
                     for index in range(5)], crs=25833)
        pyogrio.write_dataframe(features, self.path, driver="GPKG")
        self.network = RoadNetwork(self.path)
        west, south, east, north = self.network.geographic_bounds
        self.bounds = [west - 0.01, south - 0.01, east + 0.01, north + 0.01]
        self.engine = FakeEngine()
        self.engine.grid = pd.DataFrame({"x": [self.origin[0] + 250, self.origin[0] + 750, self.origin[0] + 10000],
                                         "y": [self.origin[1]] * 3})
        self.engine.cell_m = 500

    def test_geometry_only_road_api_never_calculates_weather_or_samples_grid(self):
        client = create_app(self.engine, road_network=self.network).test_client()
        settings = dict(bounds=self.bounds, categories=["R"], classes=["3"],
                        colour="score", geometry_only=True, zoom=10)
        with patch.object(Scenario, "change", side_effect=AssertionError("Unexpected weather calculation")), \
                patch.object(Scenario, "profile", side_effect=AssertionError("Unexpected weather profile")), \
                patch.object(self.network, "grid_values", side_effect=AssertionError("Unexpected grid sampling")):
            response = client.post("/api/roads/view", json=settings)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json["features"]), 1)
            self.assertIsNone(response.json["features"][0]["properties"]["mean_error"])
            with patch.object(self.network, "read", side_effect=AssertionError("Expected cached geometry")):
                cached = client.post("/api/roads/view", json=settings)
                self.assertEqual(cached.status_code, 200)
                self.assertEqual(cached.json["features"], response.json["features"])

    def test_categories_and_reference_normalization(self):
        self.assertEqual(road_categories("{Rv3}"), {"R"})
        self.assertEqual(road_categories('{E6,"Rv3"}'), {"E", "R"})
        self.assertEqual(road_categories(None), {"U"})
        self.assertEqual(road_categories(float("nan")), {"U"})
        self.assertEqual(normalize_reference("Rv 003"), "R3")

    def test_default_classes_prioritize_zero_through_four_but_others_remain_available(self):
        self.assertEqual(self.network.filters({})[1], ["0", "1", "3"])
        self.assertIn("7", self.network.metadata["classes"])
        self.assertEqual(self.network.filters(dict(classes=["7"]))[1], ["7"])

    def test_cached_attributes_survive_restart_and_road_numbers_are_not_reparsed(self):
        with patch("station_roads.pd.read_sql_query", side_effect=AssertionError("Attributes must come from cache")), \
                patch("station_roads.pyogrio.read_bounds", side_effect=AssertionError("Bounds must come from cache")), \
                patch("station_roads.road_categories", side_effect=AssertionError("Stored categories must be reused")), \
                patch("station_roads.road_references", side_effect=AssertionError("Stored road numbers must be reused")):
            network = RoadNetwork(self.path)
            roads, _ = network.read(self.bounds, {})
            self.assertEqual(roads["edge_id"].tolist(), [2, 4, 1, 3])
            selected, _ = network.read(self.bounds, dict(reference="Rv3"))
            self.assertEqual(selected["edge_id"].tolist(), [4, 1])
            result = network.view(self.engine, np.array([2, 4, 9], dtype=np.float32),
                                  None, .1, self.bounds, dict(reference="Rv3"), zoom=10)
            self.assertEqual(result["summary"]["links"], 2)
        self.assertEqual(network.index_path, self.network.index_path)
        self.assertEqual(network.metadata["class_counts"], {"3": 2, "0": 1, "1": 1, "7": 1})

    def test_changed_source_gets_a_new_attribute_cache(self):
        import os

        original = self.network.index_path
        stat = self.path.stat()
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with patch("station_roads.pyogrio.read_bounds", wraps=__import__("pyogrio").read_bounds) as read:
            network = RoadNetwork(self.path)
        self.assertNotEqual(network.index_path, original)
        self.assertTrue(original.exists())
        self.assertTrue(network.index_path.exists())
        self.assertEqual(read.call_count, 1)

    def test_invalid_attribute_cache_is_rebuilt(self):
        self.network.index_path.write_bytes(b"interrupted cache write")
        network = RoadNetwork(self.path)
        roads, _ = network.read(self.bounds, {})
        self.assertEqual(roads["edge_id"].tolist(), [2, 4, 1, 3])

    def test_default_view_includes_all_european_national_and_county_roads(self):
        roads, truncated = self.network.read(self.bounds, {})
        self.assertEqual(set(roads["edge_id"]), {1, 2, 3, 4})
        self.assertFalse(truncated)
        self.assertIsNone(self.network.limit)

    def test_zoomed_out_roads_keep_every_link_without_detailed_weather_sampling(self):
        import shapely

        roads, _ = self.network.read(self.bounds, {})
        roads = pd.concat([roads] * 600, ignore_index=True)
        with patch.object(self.network, "read", return_value=(roads, False)), \
                patch.object(self.network, "grid_values", side_effect=AssertionError("Overview must not sample weather")):
            result = self.network.view(self.engine, None, None, .1, self.bounds, {}, zoom=5)
        self.assertTrue(result["summary"]["overview"])
        self.assertFalse(result["summary"]["truncated"])
        self.assertEqual(result["summary"]["links"], 2400)
        self.assertEqual(sum(feature["properties"]["links"] for feature in result["features"]), 2400)
        self.assertEqual(sum(shapely.get_num_geometries(shapely.geometry.shape(feature["geometry"]))
                             for feature in result["features"]), 2400)
        self.assertEqual({feature["properties"]["categories"][0] for feature in result["features"]}, {"E", "R", "F"})
        with patch.object(self.network, "read", side_effect=AssertionError("Finished overview must be reused")):
            reused = self.network.view(self.engine, None, None, .5, self.bounds, {}, zoom=5)
        self.assertEqual(reused["summary"], result["summary"])
        reused["revision"] = 9
        self.assertNotIn("revision", next(iter(self.network.overviews.values())))

    def test_roads_inside_view_do_not_need_geometry_intersection(self):
        with patch("station_roads.shapely.intersection", side_effect=AssertionError("Contained roads must not be clipped")):
            roads, truncated = self.network.read(self.bounds, {})
        self.assertEqual(set(roads["edge_id"]), {1, 2, 3, 4})
        self.assertFalse(truncated)

    def test_category_class_and_route_filters_are_combined(self):
        roads, truncated = self.network.read(self.bounds, {"categories": ["R"], "classes": ["3"], "reference": "Rv3"})
        self.assertEqual(roads["edge_id"].tolist(), [1])
        self.assertFalse(truncated)
        self.assertFalse(roads.geometry.has_z.any())

    def test_combined_reference_is_visible_for_either_category(self):
        for category in ["E", "R"]:
            roads, _ = self.network.read(self.bounds, {"categories": [category], "classes": ["1"]})
            self.assertEqual(roads["edge_id"].tolist(), [4])

    def test_invalid_filters_and_empty_selection(self):
        with self.assertRaises(ValueError):
            self.network.read(self.bounds, {"categories": ["';DROP TABLE roads;"]})
        with self.assertRaises(ValueError):
            self.network.read(self.bounds, {"classes": ["999"]})
        roads, _ = self.network.read(self.bounds, {"categories": []})
        self.assertTrue(roads.empty)

    def test_length_weighted_error_and_improved_road_length(self):
        result = self.network.view(self.engine, np.array([2, 4, 9], dtype=np.float32),
                                   np.array([0, 1, 0], dtype=np.float32), 0.1, self.bounds,
                                   {"categories": ["R"], "classes": ["3"]})
        summary = result["summary"]
        self.assertAlmostEqual(summary["displayed_km"], 1, places=6)
        self.assertAlmostEqual(summary["mapped_km"], 1, places=6)
        self.assertAlmostEqual(summary["mean_error"], 3)
        self.assertEqual(summary["p90_error"], 4)
        self.assertAlmostEqual(summary["improved_km"], 0.5, places=6)

    def test_grid_sampling_does_not_extrapolate(self):
        coordinates = np.array([[self.origin[0] + 250, self.origin[1]],
                                [self.origin[0] + 250, self.origin[1] + 1000]])
        values = self.network.grid_values(self.engine, coordinates, np.array([2, 4, 9]))
        self.assertEqual(values[0], 2)
        self.assertTrue(np.isnan(values[1]))

    def test_corridor_selects_intersecting_cells_not_only_centres(self):
        self.engine.grid = pd.DataFrame({"x": [self.origin[0] + 500] * 2,
                                         "y": [self.origin[1] + 300, self.origin[1] + 600]})
        mask = self.network.corridor_mask(self.engine, self.bounds, {"categories": ["R"], "classes": ["3"]}, 60)
        self.assertEqual(mask.tolist(), [True, False])

    def test_display_limit_is_explicit(self):
        network = RoadNetwork(self.path, limit=1)
        roads, truncated = network.read(self.bounds, {"categories": ["E", "R"]})
        self.assertEqual(len(roads), 1)
        self.assertTrue(truncated)

    def test_source_is_read_only_and_road_api_is_available(self):
        modified = self.path.stat().st_mtime_ns
        client = create_app(self.engine, self.network).test_client()
        self.assertEqual(client.get("/api/roads/options").json["features"], 5)
        response = client.post("/api/roads/view", json={"bounds": self.bounds, "categories": ["R"], "classes": ["3"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["summary"]["links"], 1)
        self.assertEqual(self.path.stat().st_mtime_ns, modified)


class TerrainTests(unittest.TestCase):
    def setUp(self):
        self.engine = StationEngine.__new__(StationEngine)
        self.engine.cell_m = 1000
        self.engine.root = self.engine
        self.engine.engines = {1000: self.engine}
        self.engine.minimum, self.engine.maximum = 0.5, 6
        self.engine.lock = threading.RLock()
        self.engine.weather = None
        self.engine.grid = pd.DataFrame({
            "x": [250000, 251000, 250000], "y": [6650000, 6650000, 6651000],
            "elevation": [100, 300, 800],
            **{field: [-20, 0, 40] if field.startswith("tpi_") else [100, 200, 500]
               for field in TERRAIN_LAYERS if field != "elevation"}})
        self.engine.stations = pd.DataFrame({"x": [250000], "y": [6650000]})
        self.engine.frames = np.ones((1, 3), dtype=np.float32)
        self.engine.model_frames = {"temperature:continuous": self.engine.frames}
        self.engine.to_xy = Transformer.from_crs(4326, 25833, always_xy=True)
        self.engine.to_lonlat = Transformer.from_crs(25833, 4326, always_xy=True)
        self.engine._prepare_raster()
        self.client = create_app(self.engine).test_client()

    def test_catalog_has_all_nine_terrain_layers(self):
        response = self.client.get("/api/terrain/options")
        self.assertEqual(response.status_code, 200)
        self.assertEqual({layer["field"] for layer in response.json["layers"]}, set(TERRAIN_LAYERS))
        self.assertEqual(response.json["cell_m"], 1000)

    def test_tpi_colour_scale_is_symmetric_about_zero(self):
        for field in ("tpi_10km", "tpi_25km"):
            layer = self.engine.terrain_layer(field)
            self.assertEqual(layer["minimum"], -40)
            self.assertEqual(layer["maximum"], 40)

    def test_flat_layer_has_nonzero_scale(self):
        self.engine.grid["elevation"] = 100
        layer = self.engine.terrain_layer("elevation")
        self.assertGreater(layer["maximum"], layer["minimum"])

    def test_all_layers_render_without_weather_computation(self):
        from PIL import Image

        with patch.object(Scenario, "change", side_effect=AssertionError("Terrain must not compute weather")):
            for field in TERRAIN_LAYERS:
                response = self.client.get("/api/map.png", query_string={"mode": "terrain:" + field})
                self.assertEqual(response.status_code, 200)
                with Image.open(BytesIO(response.data)) as image:
                    self.assertGreater(np.asarray(image)[:, :, 3].max(), 0)

    def test_point_inspection_preserves_scenario_and_reads_actual_cell(self):
        before = self.client.get("/api/state").json
        lon, lat = self.engine.to_lonlat.transform(250000, 6650000)
        response = self.client.get("/api/terrain/point", query_string=dict(lon=lon, lat=lat))
        self.assertEqual(response.status_code, 200)
        values = {item["field"]: item for item in response.json["values"]}
        self.assertEqual(values["elevation"]["value"], 100)
        self.assertEqual(values["tpi_10km"]["value"], -20)
        after = self.client.get("/api/state").json
        self.assertEqual(before["revision"], after["revision"])
        self.assertEqual(before["stations"], after["stations"])

    def test_coast_distances_are_displayed_in_kilometres(self):
        layer = self.engine.terrain_layer("coast_dist_fjord")
        self.assertEqual(layer["unit"], "km")
        np.testing.assert_allclose(layer["values"], [0.1, 0.2, 0.5])

    def test_invalid_layers_and_outside_points_are_rejected(self):
        self.assertEqual(self.client.get("/api/map.png?mode=terrain:missing").status_code, 400)
        self.assertEqual(self.client.get("/api/terrain/point?lon=0&lat=0").status_code, 400)
        self.assertEqual(self.client.get("/api/terrain/point?lon=nan&lat=60").status_code, 400)

    def test_terrain_respects_optional_corridor_mask(self):
        from PIL import Image

        layer = self.engine.terrain_layer("elevation")
        buffer = self.engine.render(layer["values"], "terrain:elevation", cell_mask=np.zeros(3, dtype=bool))
        with Image.open(buffer) as image:
            self.assertEqual(np.count_nonzero(np.asarray(image)[:, :, 3]), 0)


class SelectedStationComparisonTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import Mock

        self.engine = FakeWeatherEngine()
        source = ExactNetworkTests.cache([0, 10, 20])
        target = ExactNetworkTests.cache([0, 5, 10, 19])
        model = ExactNetworkTests.DistanceModel()
        best, second, winner = exact_network(model, source, target, False)
        profile = self.engine.weather.profile("temperature:continuous")
        stations = pd.DataFrame(source)
        stations["station_id"] = ["SN1", "SN2", "SN3"]
        stations["name"] = ["First", "Second", "Third"]
        stations["first_time"], stations["last_time"] = "2021-10-14", "2026-10-14"
        stations["elevation"] = 0
        profile.update(stations=stations, baseline=best, second=second, winner=winner, model=model)
        self.engine.weather = Mock(available=FakeWeather.available, profile=Mock(return_value=profile),
                                   svv_profile=Mock(return_value=profile))
        self.engine.cache = target
        self.engine.grid = pd.DataFrame({"x": target["x"], "y": target["y"]})
        self.engine.candidate_grid = pd.DataFrame(ExactNetworkTests.cache(np.arange(40)))
        self.engine.to_xy = Transformer.from_crs(4326, 25833, always_xy=True)
        self.engine.target_index = lambda lon, lat: StationEngine.target_index(self.engine, lon, lat)
        self.engine.snap = lambda lon, lat: int(lon)
        self.engine.station = lambda index: dict(id=index, x=float(index), y=0., lon=5., lat=60., elevation=0.)
        self.engine.prediction = lambda index, dataset=None: np.abs(target["x"] - index)
        self.engine.prediction_at = lambda index, targets, dataset=None: np.abs(target["x"][targets] - index)
        self.scenario = Scenario(self.engine)
        self.scenario.apply(dict(action="comparison", value="selected"))

    def remove_and_move(self):
        self.scenario.apply(dict(action="remove_frost", index=0))
        self.scenario.apply(dict(action="move_frost", index=1, lon=30, lat=60))

    def test_repeated_moves_preserve_original_marker_until_restored(self):
        original = self.scenario.state()["fixed_stations"][1]
        self.scenario.apply(dict(action="move_frost", index=1, lon=30, lat=60))
        self.scenario.apply(dict(action="move_frost", index=1, lon=35, lat=60))
        state = self.scenario.state()
        moved = state["fixed_stations"][1]
        self.assertEqual(moved["original_position"], dict(lon=original["lon"], lat=original["lat"]))
        self.assertEqual((moved["lon"], moved["lat"]), (35, 60))
        self.assertEqual(state["fixed_count"], 3)
        self.scenario.apply(dict(action="restore_frost"))
        self.assertTrue(all("original_position" not in marker for marker in self.scenario.state()["fixed_stations"]))

    def test_point_inspection_finds_best_and_runner_up_without_changing_scenario(self):
        lon, lat = self.engine.to_lonlat.transform(10, 0)
        before = (self.scenario.revision, dict(self.scenario.frost_edits), self.scenario.selected)
        with patch.object(self.engine, "prediction", side_effect=AssertionError("No full-grid predictions")):
            result = self.scenario.best_station(lon, lat)
        self.assertEqual(result["best"]["label"], "SN2 · Second")
        self.assertEqual(result["best"]["error"], 0)
        self.assertEqual(result["runner_up"]["error"], 10)
        self.assertIsNotNone(result["third"])
        self.assertGreaterEqual(result["third"]["error"], result["runner_up"]["error"])
        self.assertEqual(result["removal_increase"], 10)
        self.assertEqual(result["station_count"], 3)
        self.assertEqual((self.scenario.revision, self.scenario.frost_edits, self.scenario.selected), before)

    def test_point_inspection_uses_moves_removals_and_manual_stations(self):
        lon, lat = self.engine.to_lonlat.transform(10, 0)
        self.remove_and_move()
        result = self.scenario.best_station(lon, lat)
        self.assertEqual(result["best"]["label"], "SN3 · Third")
        self.assertEqual(result["best"]["error"], 10)
        self.assertEqual(result["runner_up"]["index"], 1)
        self.assertIn("flyttet", result["runner_up"]["label"])
        self.scenario.apply(dict(action="add", lon=9, lat=60))
        result = self.scenario.best_station(lon, lat)
        self.assertEqual(result["best"]["kind"], "manual")
        self.assertEqual(result["best"]["error"], 1)
        self.assertEqual(result["removal_increase"], 9)

    def test_point_inspection_terrain_compares_place_and_current_station_location(self):
        lon, lat = self.engine.to_lonlat.transform(10, 0)
        self.engine.grid["elevation"] = [100, 200, 300, 400]
        self.engine.grid["relief_10km"] = 150
        self.engine.weather.profile.return_value["stations"]["elevation"] = [20, 250, 90]
        result = self.scenario.best_station(lon, lat)
        elevation = next(item for item in result["terrain"] if item["field"] == "elevation")
        self.assertEqual(elevation["place"], 300)
        self.assertEqual(elevation["station"], 250)
        self.assertEqual(elevation["runner_up"], 20)
        self.assertEqual(result["distance_km"], 0)
        self.remove_and_move()
        self.scenario.apply(dict(action="add", lon=9, lat=60))
        self.engine.candidate_grid.loc[9, "elevation"] = 275
        result = self.scenario.best_station(lon, lat)
        elevation = next(item for item in result["terrain"] if item["field"] == "elevation")
        self.assertEqual(elevation["station"], 275)
        self.assertEqual(result["distance_km"], 0.001)
        self.engine.grid.loc[2, "relief_10km"] = np.nan
        result = self.scenario.best_station(lon, lat)
        self.assertIsNone(next(item for item in result["terrain"] if item["field"] == "relief_10km")["place"])

    def test_point_inspection_moved_station_terrain_uses_new_location_and_kilometres(self):
        lon, lat = self.engine.to_lonlat.transform(10, 0)
        self.scenario.apply(dict(action="move_frost", index=1, lon=9, lat=60))
        self.engine.grid["coast_dist_fjord"] = 12000
        self.engine.candidate_grid.loc[9, "coast_dist_fjord"] = 24000
        self.engine.candidate_grid.loc[9, "elevation"] = 700
        result = self.scenario.best_station(lon, lat)
        self.assertIn("flyttet", result["best"]["label"])
        self.assertEqual(next(item for item in result["terrain"] if item["field"] == "elevation")["station"], 700)
        coast = next(item for item in result["terrain"] if item["field"] == "coast_dist_fjord")
        self.assertEqual((coast["place"], coast["station"], coast["unit"]), (12, 24, "km"))
        self.assertEqual(result["distance_km"], 0.001)

    def test_point_inspection_reports_ties_and_single_station(self):
        lon, lat = self.engine.to_lonlat.transform(5, 0)
        result = self.scenario.best_station(lon, lat)
        self.assertEqual(result["tied_count"], 2)
        self.assertEqual(result["removal_increase"], 0)
        self.scenario.apply(dict(action="remove_frost", index=0))
        self.scenario.apply(dict(action="remove_frost", index=1))
        result = self.scenario.best_station(lon, lat)
        self.assertEqual(result["station_count"], 1)
        self.assertEqual(result["best"]["index"], 2)
        self.assertIsNone(result["runner_up"])
        self.assertIsNone(result["removal_increase"])
        self.assertTrue(all(item["runner_up"] is None for item in result["terrain"]))

    def test_runner_up_terrain_uses_moved_and_manual_positions(self):
        lon, lat = self.engine.to_lonlat.transform(10, 0)
        self.engine.grid["elevation"] = 300
        self.engine.candidate_grid.loc[30, "elevation"] = 700
        self.remove_and_move()
        result = self.scenario.best_station(lon, lat)
        self.assertIn("flyttet", result["runner_up"]["label"])
        self.assertEqual(next(item for item in result["terrain"] if item["field"] == "elevation")["runner_up"], 700)
        self.scenario.apply(dict(action="add", lon=39, lat=60))
        self.engine.candidate_grid.loc[39, "elevation"] = 900
        self.scenario.apply(dict(action="remove_frost", index=1))
        result = self.scenario.best_station(lon, lat)
        self.assertEqual(result["runner_up"]["kind"], "manual")
        self.assertEqual(next(item for item in result["terrain"] if item["field"] == "elevation")["runner_up"], 900)

    def test_point_inspection_api_validation_and_no_scenario_mutation(self):
        client = create_app(self.engine).test_client()
        lon, lat = self.engine.to_lonlat.transform(10, 0)
        before = client.get("/api/state").json
        response = client.get("/api/stations/best", query_string=dict(lon=lon, lat=lat))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["best"]["index"], 1)
        self.assertEqual(client.get("/api/state").json["revision"], before["revision"])
        for longitude, latitude in [(float("nan"), lat), (181, lat), (0, 0)]:
            self.assertEqual(client.get("/api/stations/best", query_string=dict(
                lon=longitude, lat=latitude)).status_code, 400)
        client.post("/api/action", json=dict(action="network", value="empty"))
        self.assertEqual(client.get("/api/stations/best", query_string=dict(lon=lon, lat=lat)).status_code, 400)

    def test_point_inspection_greedy_network_uses_selected_weather_model(self):
        lon, lat = self.engine.to_lonlat.transform(10, 0)
        self.scenario.apply(dict(action="network", value="greedy"))
        self.engine.greedy_prediction_at = lambda index, targets, dataset="temperature:continuous": np.array([4 - index], dtype=np.float32)
        result = self.scenario.best_station(lon, lat)
        self.assertEqual(result["best"]["kind"], "greedy")
        self.assertEqual(result["best"]["index"], 1)
        self.assertEqual(result["best"]["error"], 3)
        self.assertEqual(result["removal_increase"], 1)

    def test_point_inspection_class_errors_are_bounded_and_pending_baseline_is_rejected(self):
        lon, lat = self.engine.to_lonlat.transform(5, 0)
        profile = self.engine.weather.profile.return_value
        profile.update(kind="class", unit="score", metric="Brier score")
        result = self.scenario.best_station(lon, lat)
        self.assertEqual(result["best"]["error"], 1)
        self.assertEqual(result["runner_up"]["error"], 1)
        self.assertEqual(result["tied_count"], 3)
        self.assertEqual(result["unit"], "score")
        profile.update(baseline=None, status=dict(state="missing"))
        with self.assertRaisesRegex(ValueError, "grunnflate mangler"):
            self.scenario.best_station(lon, lat)

    def test_point_inspection_rejects_invalid_model_predictions(self):
        lon, lat = self.engine.to_lonlat.transform(10, 0)
        profile = self.engine.weather.profile.return_value
        with patch.object(profile["model"], "predict", return_value=np.full(3, np.nan)):
            with self.assertRaisesRegex(ValueError, "ugyldig prediksjonsfeil"):
                self.scenario.best_station(lon, lat)

    def test_selected_move_isolated_from_removal_and_switching_station_clears_red(self):
        self.remove_and_move()
        np.testing.assert_array_equal(self.scenario.change()[0], [20, 15, 10, 1])
        np.testing.assert_array_equal(self.scenario.change()[1], [10, 5, 0, 1])
        np.testing.assert_array_equal(self.scenario.change()[2], [-10, -10, -10, 0])
        self.assertEqual(self.scenario.state()["comparison"], "move")
        self.scenario.apply(dict(action="select", kind="frost", index=2))
        np.testing.assert_array_equal(self.scenario.change()[2], [10, 10, 10, 10])
        self.assertEqual(self.scenario.state()["comparison"], "contribution")
        self.scenario.apply(dict(action="select", kind="frost", index=1))
        np.testing.assert_array_equal(self.scenario.change()[2], [-10, -10, -10, 0])

    def test_gray_station_shows_only_removal_with_other_move_preserved(self):
        self.remove_and_move()
        self.scenario.apply(dict(action="select", kind="frost", index=0))
        current, restored, reduction = self.scenario.change()
        np.testing.assert_array_equal(current, [20, 15, 10, 1])
        np.testing.assert_array_equal(restored, [0, 5, 10, 1])
        np.testing.assert_array_equal(reduction, [-20, -10, 0, 0])
        self.assertEqual(self.scenario.state()["comparison"], "removal")
        self.assertLess(self.scenario.state()["mean_reduction"], 0)
        self.assertEqual(self.scenario.state()["affected_area_km2"], 2)
        self.assertEqual(self.scenario.frost_edits, {0: dict(removed=True),
                 1: dict(candidate_index=30, removed=False, display_lon=30., display_lat=60.)})

    def test_removed_station_renders_red_without_blue_in_both_change_layers(self):
        from PIL import Image

        self.remove_and_move()
        self.scenario.apply(dict(action="select", kind="frost", index=0))
        reduction = self.scenario.change()[2]
        self.assertTrue((reduction <= 0).all())
        engine = StationEngine.__new__(StationEngine)
        engine.grid = self.engine.grid.copy()
        engine.grid["x"] = np.arange(len(reduction)) * 1000
        engine.cell_m = 1000
        engine.minimum, engine.maximum = 1, 4
        engine._prepare_raster()
        for mode in ("reduction", "affected"):
            with self.subTest(mode=mode):
                values, render_mode = self.scenario.map_change(reduction, mode)
                with Image.open(engine.render(values, render_mode)) as image:
                    pixels = np.asarray(image)
                    visible = pixels[:, :, 3] > 0
                    self.assertTrue(((pixels[:, :, 0] > pixels[:, :, 2]) & visible).any())
                    self.assertFalse(((pixels[:, :, 2] > pixels[:, :, 0]) & visible).any())

    def test_switching_tabs_preserves_edits_and_selection_but_changes_reference(self):
        self.remove_and_move()
        self.scenario.apply(dict(action="comparison", value="baseline"))
        np.testing.assert_array_equal(self.scenario.change()[2], [-20, -10, -10, 0])
        self.scenario.apply(dict(action="select", kind="frost", index=2))
        np.testing.assert_array_equal(self.scenario.change()[2], [-20, -10, -10, 0])
        self.scenario.apply(dict(action="comparison", value="selected"))
        np.testing.assert_array_equal(self.scenario.change()[2], [10, 10, 10, 10])
        self.assertEqual(self.scenario.selected, dict(kind="frost", index=2, station_id="SN3"))
        self.assertEqual(len(self.scenario.frost_edits), 2)

    def test_all_changes_includes_manual_stations_and_is_zero_before_edits(self):
        self.scenario.apply(dict(action="comparison", value="baseline"))
        self.scenario.apply(dict(action="select", kind="frost", index=1))
        np.testing.assert_array_equal(self.scenario.change()[2], [0, 0, 0, 0])
        self.scenario.apply(dict(action="add", lon=5, lat=60))
        np.testing.assert_array_equal(self.scenario.change()[2], [0, 5, 0, 0])

    def test_isolated_move_and_manual_contribution_respect_other_changes(self):
        self.remove_and_move()
        self.scenario.apply(dict(action="add", lon=5, lat=60))
        np.testing.assert_array_equal(self.scenario.change()[2], [15, 15, 5, 0])
        self.scenario.apply(dict(action="select", kind="frost", index=1))
        np.testing.assert_array_equal(self.scenario.change()[2], [0, 0, -5, 0])

    def test_multiple_moves_restore_only_selected_original_position(self):
        self.scenario.apply(dict(action="move_frost", index=0, lon=5, lat=60))
        self.scenario.apply(dict(action="move_frost", index=1, lon=30, lat=60))
        np.testing.assert_array_equal(self.scenario.change()[2], [0, 0, -5, 0])
        self.scenario.apply(dict(action="select", kind="frost", index=0))
        np.testing.assert_array_equal(self.scenario.change()[2], [-5, 5, 5, 0])

    def test_api_comparison_validation_and_independent_sessions(self):
        app = create_app(self.engine)
        client, other = app.test_client(), app.test_client()
        response = client.post("/api/action", json=dict(action="comparison", value="baseline"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["comparison_scope"], "baseline")
        self.assertEqual(other.get("/api/state").json["comparison_scope"], "selected")
        self.assertEqual(client.post("/api/action", json=dict(action="comparison", value="invalid")).status_code, 400)
        self.assertEqual(client.get("/api/state").json["comparison_scope"], "baseline")

    def test_maps_and_previews_switch_between_isolated_move_and_contribution(self):
        from unittest.mock import Mock

        client = create_app(self.engine).test_client()
        client.post("/api/action", json=dict(action="comparison", value="selected"))
        client.post("/api/action", json=dict(action="remove_frost", index=0))
        client.post("/api/action", json=dict(action="move_frost", index=1, lon=30, lat=60))
        self.engine.render = Mock(wraps=self.engine.render)
        self.assertEqual(client.get("/api/map.png?mode=reduction").status_code, 200)
        np.testing.assert_array_equal(self.engine.render.call_args.args[0], [-10, -10, -10, 0])
        self.assertEqual(self.engine.render.call_args.args[1], "change")
        preview = client.post("/api/preview", json=dict(lon=5, lat=60, mode="reduction"))
        self.assertEqual(preview.status_code, 200)
        np.testing.assert_array_equal(self.engine.render.call_args.args[0], [15, 15, 5, 0])
        client.post("/api/action", json=dict(action="select", kind="frost", index=2))
        self.assertEqual(client.get("/api/map.png?mode=reduction").status_code, 200)
        np.testing.assert_array_equal(self.engine.render.call_args.args[0], [10, 10, 10, 10])
        self.assertEqual(self.engine.render.call_args.args[1], "reduction")
        client.post("/api/action", json=dict(action="select", kind="frost", index=0))
        self.assertEqual(client.get("/api/map.png?mode=reduction").status_code, 200)
        np.testing.assert_array_equal(self.engine.render.call_args.args[0], [-20, -10, 0, 0])
        self.assertEqual(self.engine.render.call_args.args[1], "change")
        self.assertEqual(client.get("/api/map.png?mode=affected").status_code, 200)
        np.testing.assert_array_equal(self.engine.render.call_args.args[0], [-1, -1, 0, 0])
        self.assertEqual(self.engine.render.call_args.args[1], "affected_change")

    def test_selected_move_affected_area_uses_absolute_change_and_strict_threshold(self):
        self.scenario.apply(dict(action="move_frost", index=0, lon=5, lat=60))
        state = self.scenario.state()
        np.testing.assert_array_equal(self.scenario.change()[2], [-5, 5, 0, 0])
        self.assertEqual(state["affected_area_km2"], 2)
        values, mode = self.scenario.map_change(self.scenario.change()[2], "affected")
        self.assertEqual(mode, "affected_change")
        np.testing.assert_array_equal(values, [-1, 1, 0, 0])
        self.scenario.apply(dict(action="threshold", value=5))
        self.assertEqual(self.scenario.state()["affected_area_km2"], 0)

    def test_selected_export_uses_active_move_or_removal_not_all_changes(self):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin

        self.engine.raster_shape = (1, 4)
        self.engine.rows = np.array([0, 0, 0, 0])
        self.engine.columns = np.arange(4)
        self.engine.transform = from_origin(-500, 500, 1000, 1000)
        self.remove_and_move()
        for index, kind, expected in [(1, "move", [-10, -10, -10, 0]),
                                      (0, "removal", [-20, -10, 0, 0])]:
            with self.subTest(kind=kind):
                self.scenario.apply(dict(action="select", kind="frost", index=index))
                with ZipFile(self.scenario.export()) as archive:
                    with MemoryFile(archive.read("selected_reduction.tif")) as memory:
                        with memory.open() as raster:
                            np.testing.assert_array_equal(raster.read(1), [expected])
                            self.assertEqual(raster.tags()["comparison"], kind)
                            self.assertEqual(raster.tags()["comparison_scope"], "selected")
                    metadata = json.loads(archive.read("scenario.json"))
                    self.assertEqual(metadata["comparison_scope"], "selected")
                    self.assertEqual(metadata["selected_station"]["index"], index)

    def test_road_view_receives_selected_station_change_after_other_edits(self):
        from unittest.mock import Mock

        roads = Mock()
        roads.view.return_value = dict(type="FeatureCollection", features=[])
        client = create_app(self.engine, road_network=roads).test_client()
        client.post("/api/action", json=dict(action="comparison", value="selected"))
        client.post("/api/action", json=dict(action="remove_frost", index=0))
        client.post("/api/action", json=dict(action="move_frost", index=1, lon=30, lat=60))
        self.assertEqual(client.post("/api/roads/view", json=dict(bounds=[5, 58, 12, 64])).status_code, 200)
        np.testing.assert_array_equal(roads.view.call_args.args[2], [-10, -10, -10, 0])
        client.post("/api/action", json=dict(action="select", kind="frost", index=2))
        self.assertEqual(client.post("/api/roads/view", json=dict(bounds=[5, 58, 12, 64])).status_code, 200)
        np.testing.assert_array_equal(roads.view.call_args.args[2], [10, 10, 10, 10])


class RoadHoverTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeWeatherEngine()
        self.targets = []

        def partial(index, target_ids, dataset=None):
            self.targets.append(target_ids.copy())
            return np.array([2, 0.5, 2], dtype=np.float32)[target_ids]

        self.engine.prediction_at = partial
        self.mask = np.array([True, False, True])
        owner = self

        class Roads:
            def corridor_mask(self, engine, bounds, settings, distance):
                return owner.mask

        self.client = create_app(self.engine, Roads()).test_client()
        self.client.post("/api/action", json=dict(action="comparison", value="selected"))
        self.before = self.client.get("/api/state").json
        self.request = dict(lon=5, lat=60, revision=self.before["revision"],
                            roads=dict(bounds=[5, 59, 11, 62], corridor=False))

    def test_visible_roads_only_predict_selected_cells(self):
        with patch.object(Scenario, "surface", side_effect=AssertionError("Full surface forbidden")), \
                patch.object(self.engine, "prediction", side_effect=AssertionError("Full prediction forbidden")):
            response = self.client.post("/api/preview", json=self.request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.targets), 1)
        np.testing.assert_array_equal(self.targets[0], [0, 2])
        self.assertEqual(response.headers["X-Preview-Scope"], "roads")
        self.assertEqual(response.headers["X-Evaluated-Cells"], "2")
        self.assertEqual(float(response.headers["X-Mean-Rmse"]), 2)
        self.assertEqual(float(response.headers["X-Affected-Area-Km2"]), 2)
        after = self.client.get("/api/state").json
        self.assertEqual(self.before["revision"], after["revision"])
        self.assertEqual(after["stations"], [])

    def test_empty_corridor_never_calls_predictor(self):
        self.mask[:] = False
        with patch.object(self.engine, "prediction_at", side_effect=AssertionError("No cells to predict")):
            response = self.client.post("/api/preview", json=self.request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Evaluated-Cells"], "0")
        self.assertEqual(response.headers["X-Mean-Rmse"], "")

    def test_hidden_roads_keep_full_network_preview(self):
        request = {**self.request, "roads": None}
        with patch.object(self.engine, "prediction_at", side_effect=AssertionError("Full preview expected")):
            response = self.client.post("/api/preview", json=request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Preview-Scope"], "network")
        self.assertEqual(response.headers["X-Evaluated-Cells"], "3")

    def test_existing_manual_station_is_also_evaluated_only_locally(self):
        self.client.post("/api/action", json=dict(action="add", lon=5, lat=60))
        request = {**self.request, "revision": self.client.get("/api/state").json["revision"]}
        with patch.object(Scenario, "surface", side_effect=AssertionError("Full surface forbidden")):
            response = self.client.post("/api/preview", json=request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.targets), 2)
        self.assertEqual(float(response.headers["X-Affected-Area-Km2"]), 0)


class PartialPredictionTests(unittest.TestCase):
    def setUp(self):
        owner = self
        self.calls = []

        class DistanceModel:
            def predict(self, features):
                owner.calls.append(len(features))
                return features[:, 0]

        self.engine = StationEngine.__new__(StationEngine)
        self.engine.grid = FakeEngine.grid
        self.engine.lock = threading.RLock()
        self.engine.weather = None
        self.engine.model = DistanceModel()
        self.engine.predictions = OrderedDict()
        self.engine.partial_predictions = OrderedDict()
        self.engine.cache = {key: np.zeros(3, dtype=np.float32) for key in CACHE_KEYS}
        self.engine.cache["x"] = np.array([0, 1000, 2000], dtype=np.float32)
        self.engine.candidate_cache = self.engine.cache

    def test_partial_cache_matches_requested_cells_and_filter_changes(self):
        first = self.engine.prediction_at(1, np.array([0, 2]))
        np.testing.assert_array_equal(first, [1000, 1000])
        self.engine.prediction_at(1, np.array([0, 2]))
        second = self.engine.prediction_at(1, np.array([1]))
        np.testing.assert_array_equal(second, [0])
        self.assertEqual(self.calls, [2, 1])

    def test_partial_result_is_not_reused_as_full_prediction(self):
        self.engine.prediction_at(1, np.array([0, 2]))
        full = self.engine.prediction(1)
        np.testing.assert_array_equal(full, [1000, 0, 1000])
        self.assertEqual(self.calls, [2, 3])

    def test_full_cached_prediction_can_supply_subset(self):
        self.engine.predictions[(None, 1)] = np.array([1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(self.engine.prediction_at(1, [0, 2]), [1, 3])
        self.assertEqual(self.calls, [])

    def test_invalid_indices_rejected_and_empty_subset_is_free(self):
        for indices in [[-1], [3], [[0]]]:
            with self.assertRaises(ValueError):
                self.engine.prediction_at(1, indices)
        self.assertEqual(len(self.engine.prediction_at(1, [])), 0)
        self.assertEqual(self.calls, [])


class HumidityMapTests(unittest.TestCase):
    def test_initial_humidity_scenario_uses_humidity_threshold(self):
        scenario = Scenario(FakeEngine(), dataset="humidity:continuous")
        self.assertEqual(scenario.dataset, "humidity:continuous")
        self.assertEqual(scenario.threshold, 0.1)

    @staticmethod
    def bundle(root, spacing):
        folder = root / "outputs" / f"humidity_5y_{spacing}m"
        folder.mkdir(parents=True)
        prefix = folder / "humidity_5y_prediction_rmse"
        for suffix in ("_model.joblib", "_metadata.json", "_station_features.gpkg"):
            prefix.with_name(prefix.name + suffix).touch()
        return prefix

    def test_humidity_is_continuous_only_with_percentage_point_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.bundle(root, 5000)
            registry = WeatherRegistry(root, None, None, None, cell_m=1000)
            self.assertEqual([item["id"] for item in registry.available], ["humidity:continuous"])
            self.assertEqual(registry.available[0]["unit"], "prosentpoeng")
            self.assertEqual(registry.available[0]["model_grid_m"], 5000)
            self.assertEqual(registry.available[0]["element"], "relative_humidity")

    def test_matching_resolution_preferred_and_other_grids_fall_back_to_5000(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coarse = self.bundle(root, 5000)
            matching = self.bundle(root, 2500)
            for spacing, expected in [(1000, coarse), (2500, matching), (5000, coarse)]:
                registry = WeatherRegistry(root, None, None, None, cell_m=spacing)
                self.assertEqual(registry.available[0]["prefix"], str(expected))

    def test_incomplete_matching_bundle_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coarse = self.bundle(root, 5000)
            folder = root / "outputs/humidity_5y_1000m"
            folder.mkdir()
            (folder / "humidity_5y_prediction_rmse_model.joblib").touch()
            registry = WeatherRegistry(root, None, None, None, cell_m=1000)
            self.assertEqual(registry.available[0]["prefix"], str(coarse))

    def test_humidity_scenario_units_limits_and_proposed_station_support(self):
        class HumidityWeather(FakeWeather):
            available = FakeWeather.available + [dict(id="humidity:continuous", phenomenon="humidity",
                kind="continuous", label="Luftfuktighet", metric="RMSE", unit="prosentpoeng")]

            def profile(self, identifier):
                if identifier != "humidity:continuous":
                    return super().profile(identifier)
                result = super().profile("temperature:continuous")
                result.update(self.available[-1], model_grid_m=5000, classes=None)
                return result

        engine = FakeWeatherEngine()
        engine.weather = HumidityWeather()
        scenario = Scenario(engine)
        scenario.apply(dict(action="dataset", value="humidity:continuous"))
        state = scenario.state()
        self.assertEqual(state["unit"], "prosentpoeng")
        self.assertEqual(state["model_grid_m"], 5000)
        self.assertIsNone(state["classes"])
        self.assertEqual(scenario.limits()[2], 10)
        scenario.apply(dict(action="baseline", count=2))
        self.assertEqual(scenario.network, "greedy")
        self.assertGreater(scenario.state()["mean_rmse"], 1)

    def test_existing_wind_bundle_is_available_as_5000m_rmse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "outputs" / "wind_5y_5000m"
            folder.mkdir(parents=True)
            prefix = folder / "wind_5y_prediction_rmse"
            for suffix in ("_model.joblib", "_metadata.json", "_station_features.gpkg"):
                prefix.with_name(prefix.name + suffix).touch()
            registry = WeatherRegistry(root, None, None, None, cell_m=1000)
            wind = next(item for item in registry.available if item["id"] == "wind:continuous")
            self.assertEqual(wind["prefix"], str(prefix))
            self.assertEqual(wind["model_grid_m"], 5000)
            self.assertEqual(wind["unit"], "m/s")
            self.assertEqual(wind["element"], "wind_speed")

    def test_wind_mode_uses_wind_units_threshold_and_reduction_scale(self):
        class WindWeather(FakeWeather):
            available = FakeWeather.available + [dict(id="wind:continuous", phenomenon="wind", kind="continuous",
                                                       label="Vindhastighet", metric="RMSE", unit="m/s")]

            def profile(self, identifier):
                result = super().profile("temperature:continuous")
                result.update(self.available[-1], maximum=20, model_grid_m=5000, classes=None)
                return result

        engine = FakeWeatherEngine()
        engine.weather = WindWeather()
        scenario = Scenario(engine, dataset="wind:continuous")
        self.assertEqual(scenario.state()["unit"], "m/s")
        self.assertEqual(scenario.threshold, 0.1)
        self.assertEqual(scenario.limits()[2], 5)

    def test_availability_cache_invalidates_after_database_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "frost.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE stations(station_id TEXT,name TEXT,lat REAL,lon REAL,elevation REAL,inside_norway INTEGER)")
                connection.execute("CREATE TABLE observations(station_id TEXT,reference_time TEXT,element TEXT,value REAL,PRIMARY KEY(station_id,reference_time,element))")
                connection.executemany("INSERT INTO stations VALUES(?,?,?,?,?,?)", [("SN1", "A", 60, 10, 100, 1), ("SN2", "B", 61, 11, 200, 1)])
                connection.execute("INSERT INTO observations VALUES('SN1','2026-10-19T00:00:00.000Z','relative_humidity',80)")
                connection.commit()
            first, _, _ = observation_stations(database, root / "absent.csv", "relative_humidity")
            self.assertEqual(first["station_id"].tolist(), ["SN1"])
            first.loc[0, "name"] = "Mutated"
            from station_weather import _observation_stations
            _observation_stations.cache_clear()
            with patch("station_weather.sqlite3.connect", side_effect=AssertionError("Expected cached lookup")):
                cached, _, _ = observation_stations(database, root / "absent.csv", "relative_humidity")
            self.assertEqual(cached.loc[0, "name"], "A")
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("INSERT INTO observations VALUES('SN2','2026-10-19T00:00:00.000Z','relative_humidity',70)")
                connection.commit()
            import os
            stat = database.stat()
            os.utime(database, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            updated, _, _ = observation_stations(database, root / "absent.csv", "relative_humidity")
            self.assertEqual(set(updated["station_id"]), {"SN1", "SN2"})

    def test_availability_cache_ignores_downloads_of_other_elements(self):
        from pipeline_config import element_update_marker
        from station_weather import _observation_stations
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "frost.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE stations(station_id TEXT,name TEXT,lat REAL,lon REAL,elevation REAL,inside_norway INTEGER)")
                connection.execute("CREATE TABLE observations(station_id TEXT,reference_time TEXT,element TEXT,value REAL,PRIMARY KEY(station_id,reference_time,element))")
                connection.executemany("INSERT INTO stations VALUES(?,?,?,?,?,?)", [("SN1", "A", 60, 10, 100, 1), ("SN2", "B", 61, 11, 200, 1)])
                connection.execute("INSERT INTO observations VALUES('SN1','2026-10-19T00:00:00.000Z','wind_speed',3)")
                connection.commit()
            element_update_marker(database, "wind_speed").parent.mkdir()
            observation_stations(database, root / "absent.csv", "wind_speed")
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("INSERT INTO observations VALUES('SN2','2026-10-19T00:00:00.000Z','road',1)")
                connection.commit()
            _observation_stations.cache_clear()
            with patch("station_weather.sqlite3.connect", side_effect=AssertionError("Expected cached lookup")):
                cached, _, _ = observation_stations(database, root / "absent.csv", "wind_speed")
            self.assertEqual(cached["station_id"].tolist(), ["SN1"])
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("INSERT INTO observations VALUES('SN2','2026-10-19T00:00:00.000Z','wind_speed',4)")
                connection.commit()
            element_update_marker(database, "wind_speed").touch()
            updated, _, _ = observation_stations(database, root / "absent.csv", "wind_speed")
            self.assertEqual(set(updated["station_id"]), {"SN1", "SN2"})


class PerformanceCacheTests(unittest.TestCase):
    def test_repeated_state_uses_same_contribution(self):
        scenario = Scenario(FakeEngine())
        with patch.object(scenario, "_calculate_change", wraps=scenario._calculate_change) as compute:
            first = scenario.change()
            scenario.state()
            self.assertIs(scenario.change(), first)
            self.assertEqual(compute.call_count, 1)
            scenario.apply(dict(action="threshold", value=0.5))
            scenario.state()
            self.assertEqual(compute.call_count, 1)
            scenario.apply(dict(action="baseline", count=1))
            scenario.change()
            self.assertEqual(compute.call_count, 2)

    def test_frost_completion_invalidates_pending_result(self):
        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        baseline = profile["baseline"].copy()
        profile.update(baseline=None, status=dict(state="missing"))

        class Weather:
            available = FakeWeather.available

            def profile(self, identifier):
                return profile

        engine.weather = Weather()
        scenario = Scenario(engine)
        self.assertEqual(scenario.change(), (None, None, None))
        profile.update(baseline=baseline, status=dict(state="ready"))
        np.testing.assert_array_equal(scenario.change()[0], baseline)

    def test_candidate_source_is_not_loaded_when_cache_already_exists(self):
        engine = StationEngine.__new__(StationEngine)
        engine.candidate_cache = {"x": np.array([1])}
        with patch("station_app.load_grid_all_fields", side_effect=AssertionError("Unexpected grid read")):
            engine.ensure_candidates()

    def test_default_startup_uses_5000m_without_full_precompute(self):
        from station_app import main as start
        from unittest.mock import Mock

        engine = Mock()
        engine.weather.available = [dict(id="temperature:senorge30")]
        app = Mock(config={})
        with patch("sys.argv", ["station_app.py"]), \
                patch("station_app.StationEngine", return_value=engine) as constructor, \
            patch("station_app.create_app", return_value=app) as factory, \
            patch("station_app.KuldehullLayer") as cold_basins, \
                patch("station_app.socket.socket") as socket_factory, \
                patch("station_app.threading.Thread", side_effect=AssertionError("App must not start a worker")), \
                patch("station_weather.exact_network", side_effect=AssertionError("App must not compute Frost")):
            socket_factory.return_value.__enter__.return_value.connect_ex.return_value = 1
            start()
        self.assertEqual(constructor.call_args.kwargs["cell_m"], 5000)
        engine.queue_precompute.assert_not_called()
        engine.warm_weather_maps.assert_called_once()
        cold_basins.return_value.prepare.assert_called_once()
        self.assertIs(factory.call_args.kwargs["kuldehull_layer"], cold_basins.return_value)
        self.assertEqual(factory.call_args.kwargs["initial_dataset"], "temperature:senorge30")
        app.run.assert_called_once()


class OfflineFrostTests(unittest.TestCase):
    def test_checkpoint_retries_transient_file_locks_without_rewriting_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frost.partial.npz"
            values = np.array([1., 2.], dtype=np.float32)
            replace = Path.replace
            attempts = []

            def locked_replace(staged, target):
                attempts.append(staged)
                if len(attempts) < 3:
                    raise PermissionError("Simulated Windows file lock")
                return replace(staged, target)

            with patch.object(Path, "replace", autospec=True, side_effect=locked_replace), \
                    patch("station_weather.time.sleep") as pause, \
                    patch("station_weather.np.savez_compressed", wraps=np.savez_compressed) as write:
                save_exact_cache(path, values, values + 1, np.zeros(2, dtype=np.int32), 8)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(pause.call_count, 2)
            self.assertEqual(write.call_count, 1)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(int(saved["completed"]), 8)
                np.testing.assert_array_equal(saved["baseline"], values)
            self.assertEqual(list(path.parent.glob("*.pending_*.npz")), [])

    def test_checkpoint_permanent_lock_keeps_previous_and_new_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frost.partial.npz"
            values = np.array([1., 2.], dtype=np.float32)
            save_exact_cache(path, values, values + 1, np.zeros(2, dtype=np.int32), 8)
            with patch.object(Path, "replace", side_effect=PermissionError("Locked")), \
                    patch("station_weather.time.sleep"), self.assertRaisesRegex(PermissionError, "bevart"):
                save_exact_cache(path, values / 2, values, np.zeros(2, dtype=np.int32), 16)
            with np.load(path, allow_pickle=False) as previous:
                self.assertEqual(int(previous["completed"]), 8)
            pending = list(path.parent.glob("*.pending_*.npz"))
            self.assertEqual(len(pending), 1)
            with np.load(pending[0], allow_pickle=False) as saved:
                self.assertEqual(int(saved["completed"]), 16)
                np.testing.assert_array_equal(saved["baseline"], values / 2)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.folder = self.root / "outputs"
        self.folder.mkdir()
        prefix = self.folder / "temperature_5y_prediction_rmse"
        for suffix in ("_model.joblib", "_station_features.gpkg"):
            prefix.with_name(prefix.name + suffix).touch()
        prefix.with_name(prefix.name + "_metadata.json").write_text(json.dumps({"features": FEATURES}))
        self.grid = pd.DataFrame(dict(x=[0., 1000.], y=[0., 0.], elevation=[0., 0.],
            coast_dist_fjord=[0., 0.], coast_dist_ocean=[0., 0.], objtype=["Skog", "Skog"],
            objtype_code=[0, 0], **{field: [0., 0.] for field in TERRAIN}))
        sources = self.grid.copy()
        sources["station_id"] = ["SN1", "SN2"]
        observed = pd.DataFrame(dict(station_id=["SN1", "SN2"], name=["A", "B"],
            lat=[60., 61.], lon=[10., 11.], first_time=["2021-10-20"] * 2, last_time=["2026-10-20"] * 2))

        class DistanceModel:
            n_features_in_ = len(FEATURES)

            def predict(self, features):
                return np.maximum(features[:, 0], 1)

        for target, options in [
            ("station_weather.observation_stations", dict(return_value=(observed, "2021-10-20", "2026-10-21"))),
            ("station_weather.pyogrio.read_dataframe", dict(return_value=sources)),
            ("station_weather.joblib.load", dict(return_value=DistanceModel()))
        ]:
            mock = patch(target, **options)
            mock.start()
            self.addCleanup(mock.stop)

    def registry(self, compute=False):
        return WeatherRegistry(self.root, self.grid, make_cache(self.grid), pd.Index(["Skog"]),
                               cell_m=5000, allow_precompute=compute)

    def test_app_uses_completed_snapshot_while_download_changes_database(self):
        with patch("builtins.print"):
            computed = self.registry(compute=True).profile("temperature:continuous")
        observed = pd.DataFrame(dict(
            station_id=["SN1", "SN2"], name=["A", "B"], lat=[60., 61.], lon=[10., 11.],
            first_time=["2021-10-20"] * 2, last_time=["2026-10-20"] * 2))
        snapshot = computed["cache_path"].parent / "availability_test.json"
        snapshot.write_text(json.dumps(dict(stations=observed.to_json(orient="split"),
                                            start="2021-10-20", end="2026-10-21",
                                            element="air_temperature")), encoding="utf-8")
        with patch("station_weather.observation_stations",
                   side_effect=AssertionError("Interactive app must not scan the downloading database")), \
                patch("station_weather.exact_network",
                      side_effect=AssertionError("Interactive app must not recompute maps")):
            loaded = self.registry().profile("temperature:continuous")
        np.testing.assert_array_equal(loaded["baseline"], computed["baseline"])
        self.assertEqual(loaded["cache_path"], computed["cache_path"])
        self.assertEqual(loaded["stations"]["station_id"].tolist(), ["SN1", "SN2"])
        with patch("station_weather.observation_stations",
                   side_effect=RuntimeError("Precompute must read the updated observations")):
            with self.assertRaisesRegex(RuntimeError, "updated observations"):
                self.registry(compute=True).profile("temperature:continuous")

    def test_missing_surface_is_read_only_and_creates_no_thread_or_checkpoint(self):
        registry = self.registry()
        before = {thread.ident for thread in threading.enumerate()}
        with patch("station_weather.exact_network", side_effect=AssertionError("App must never compute")):
            profile = registry.profile("temperature:continuous")
        self.assertIsNone(profile["baseline"])
        self.assertEqual(profile["status"]["state"], "missing")
        self.assertIn("--phenomenon temperature --cell-m 5000 --kind continuous", profile["status"]["error"])
        self.assertEqual({thread.ident for thread in threading.enumerate()}, before)
        self.assertEqual(list(self.folder.glob("**/*.npz")), [])

    def test_offline_compute_persists_best_two_and_app_reloads_without_prediction(self):
        reader = self.registry()
        pending = reader.profile("temperature:continuous")
        with patch("builtins.print"):
            computed = self.registry(compute=True).profile("temperature:continuous")
        self.assertTrue(computed["cache_path"].exists())
        self.assertEqual(computed["cache_path"], pending["cache_path"])
        self.assertFalse(computed["cache_path"].with_suffix(".partial.npz").exists())
        with patch("station_weather.exact_network", side_effect=AssertionError("Saved surface must be reused")):
            loaded = reader.profile("temperature:continuous")
            reused = self.registry(compute=True).profile("temperature:continuous")
        for profile in (computed, loaded, reused):
            self.assertEqual(profile["status"]["state"], "ready")
            self.assertEqual(profile["status"]["percent"], 100)
            np.testing.assert_array_equal(profile["baseline"], [1, 1])
            np.testing.assert_array_equal(profile["second"], [1000, 1000])
            np.testing.assert_array_equal(profile["winner"], [0, 1])

    def test_offline_resumes_existing_checkpoint(self):
        pending = self.registry().profile("temperature:continuous")
        checkpoint = pending["cache_path"].with_suffix(".partial.npz")
        save_exact_cache(checkpoint, np.array([1., 1000.], dtype=np.float32),
                         np.full(2, np.inf, dtype=np.float32), np.zeros(2, dtype=np.int32), 1)
        with patch("station_weather.paired_predictions", wraps=paired_predictions) as predict, patch("builtins.print"):
            computed = self.registry(compute=True).profile("temperature:continuous")
        self.assertEqual(predict.call_count, 1)
        np.testing.assert_array_equal(computed["baseline"], [1, 1])
        np.testing.assert_array_equal(computed["second"], [1000, 1000])
        self.assertFalse(checkpoint.exists())

    def test_offline_resumes_newer_recovery_file_after_persistent_lock(self):
        pending = self.registry().profile("temperature:continuous")
        checkpoint = pending["cache_path"].with_suffix(".partial.npz")
        save_exact_cache(checkpoint, np.array([1., 1000.], dtype=np.float32),
                         np.full(2, np.inf, dtype=np.float32), np.zeros(2, dtype=np.int32), 1)
        with patch.object(Path, "replace", side_effect=PermissionError("Locked")), \
                patch("station_weather.time.sleep"), self.assertRaises(PermissionError):
            save_exact_cache(checkpoint, np.ones(2, dtype=np.float32),
                             np.full(2, 1000., dtype=np.float32), np.array([0, 1], dtype=np.int32), 2)
        with patch("builtins.print"), \
                patch("station_weather.paired_predictions", side_effect=AssertionError("Saved progress must be reused")):
            computed = self.registry(compute=True).profile("temperature:continuous")
        self.assertEqual(computed["status"]["state"], "ready")
        np.testing.assert_array_equal(computed["baseline"], [1, 1])
        np.testing.assert_array_equal(computed["second"], [1000, 1000])
        np.testing.assert_array_equal(computed["winner"], [0, 1])

    def test_incomplete_recovery_file_does_not_replace_valid_checkpoint(self):
        pending = self.registry().profile("temperature:continuous")
        checkpoint = pending["cache_path"].with_suffix(".partial.npz")
        save_exact_cache(checkpoint, np.array([1., 1000.], dtype=np.float32),
                         np.full(2, np.inf, dtype=np.float32), np.zeros(2, dtype=np.int32), 1)
        checkpoint.with_name(f"{checkpoint.stem}.pending_interrupted.npz").touch()
        with patch("builtins.print"), patch("station_weather.paired_predictions", wraps=paired_predictions) as predict:
            computed = self.registry(compute=True).profile("temperature:continuous")
        self.assertEqual(predict.call_count, 1)
        np.testing.assert_array_equal(computed["baseline"], [1, 1])

    def test_frost_colour_range_matches_saved_baseline_and_stays_fixed_for_proposals(self):
        self.grid["x"] = [200., 700.]
        with patch("builtins.print"):
            self.registry(compute=True).profile("temperature:continuous")
        engine = FakeWeatherEngine()
        engine.weather = self.registry()
        station_mask = patch("station_pross.pross_excluded", return_value=frozenset())
        station_mask.start()
        self.addCleanup(station_mask.stop)
        scenario = Scenario(engine)
        self.assertEqual(scenario.network, "frost")
        lower, upper, _ = scenario.limits()
        self.assertAlmostEqual(lower, 200.5)
        self.assertAlmostEqual(upper, 299.5)
        scenario.added = [1]
        self.assertEqual(scenario.limits()[:2], (lower, upper))
        self.assertEqual(scenario.limits(np.array([True, False]))[:2], (lower, upper))
        self.assertEqual(scenario.limits(np.array([False, False]))[:2], (lower, upper))
        self.assertEqual(scenario.limits()[:2], (lower, upper))
        scenario.network = "greedy"
        self.assertEqual(scenario.limits()[:2], (lower, upper))

    def test_constant_frost_colour_range_is_positive_and_nonzero(self):
        with patch("builtins.print"):
            profile = self.registry(compute=True).profile("temperature:continuous")
        self.assertGreater(profile["minimum"], 0)
        self.assertGreater(profile["frost_maximum"], profile["minimum"])

    def test_zero_frost_errors_do_not_break_logarithmic_colours(self):
        with patch("builtins.print"), patch("station_weather.paired_predictions", return_value=np.zeros(4, dtype=np.float32)):
            profile = self.registry(compute=True).profile("temperature:continuous")
        self.assertGreater(profile["minimum"], 0)
        self.assertGreater(profile["frost_maximum"], profile["minimum"])
        np.testing.assert_array_equal(profile["baseline"], [0, 0])

    def test_missing_frost_api_reports_missing_and_blocks_preview_and_export(self):
        engine = FakeWeatherEngine()
        ready = engine.weather.profile("temperature:continuous")
        registry = self.registry()
        engine.weather = registry
        engine.frost_statuses = lambda: [dict(profile["status"]) for profile in registry.loaded.values()]
        original = registry._load

        def load(config):
            profile = original(config)
            profile.update({key: value for key, value in ready.items()
                            if key in ("stations", "start", "end", "maximum", "minimum")})
            return profile

        with patch.object(registry, "_load", side_effect=load), \
                patch("station_weather.exact_network", side_effect=AssertionError("No API may compute Frost")):
            client = create_app(engine).test_client()
            state = client.get("/api/state").json
            self.assertFalse(state["frost_ready"])
            self.assertEqual(client.get("/api/precompute").json["jobs"][0]["state"], "missing")
            self.assertEqual(client.post("/api/preview", json=dict(lon=5, lat=60, revision=state["revision"])).status_code, 409)
            self.assertEqual(client.get("/api/export").status_code, 400)

    def test_dry_run_is_side_effect_free_and_missing_mode_is_rejected(self):
        from precompute_frost import main
        before = sorted(self.root.rglob("*"))
        with patch("precompute_frost.HERE", self.root), patch("builtins.print"), \
                patch("precompute_frost.load_grid_all_fields", side_effect=AssertionError("Dry run must not read grids")):
            self.assertEqual(main(["--dry-run", "--phenomenon", "temperature", "--cell-m", "5000"]), 0)
            with self.assertRaises(ValueError):
                main(["--dry-run", "--phenomenon", "humidity", "--cell-m", "5000"])
        self.assertEqual(sorted(self.root.rglob("*")), before)

    def test_separate_computations_cannot_run_concurrently_and_lock_is_released(self):
        from precompute_frost import computation_lock
        with computation_lock(self.folder):
            with self.assertRaisesRegex(RuntimeError, "allerede"):
                with computation_lock(self.folder):
                    self.fail("Concurrent computation must be blocked")
        with computation_lock(self.folder):
            pass


if __name__ == "__main__":
    unittest.main()