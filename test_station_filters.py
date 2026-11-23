from io import BytesIO
import gzip
from pathlib import Path
import tempfile
import threading
import unittest
from types import MethodType
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
import pyogrio
from shapely.geometry import box

from station_app import Scenario, create_app
from station_counties import CountyMask
from test_station_app import FakeWeatherEngine
from station_weather import WeatherRegistry, restricted_profile, save_exact_cache
from station_weather_maps import frost_limits


def county_frame():
    return gpd.GeoDataFrame(dict(fylkesnavn=["West", "East"]),
        geometry=[box(-500, -500, 500, 500), box(500, -500, 2500, 500)],
        crs=25833, index=pd.Index(["01", "02"], name="code"))


class StationFilterTests(unittest.TestCase):
    def test_all_network_modes_and_phenomena_use_unedited_svv_colour_reference(self):
        for phenomenon in ("temperature", "wind", "precipitation", "snow", "humidity"):
            for kind in ("continuous", "class"):
                with self.subTest(phenomenon=phenomenon, kind=kind):
                    engine = FakeWeatherEngine()
                    profile = engine.weather.profile("temperature:continuous")
                    reference = dict(profile, baseline=np.array([0.2, 0.4, 0.8]),
                                     minimum=0.202, frost_maximum=0.796)
                    profile.update(phenomenon=phenomenon, kind=kind)
                    reference.update(phenomenon=phenomenon, kind=kind)
                    engine.weather = Mock()
                    engine.weather.profile.return_value = profile
                    engine.weather.svv_profile.return_value = reference
                    scenario = Scenario(engine)
                    for network, svv_only in (("frost", True), ("frost", False), ("empty", False)):
                        scenario.network, scenario.pross_only = network, svv_only
                        scenario.station_since = "2026-01-01"
                        scenario.counties = ["01"]
                        scenario.frost_edits = {0: dict(removed=True)}
                        scenario.added = [1]
                        self.assertEqual(scenario.limits(), frost_limits(reference))
                    engine.weather.svv_profile.assert_called_with(scenario.dataset)

    def test_road_temperature_uses_temperature_limits_without_changing_temperature(self):
        from station_app import PHENOMENON_DATASETS
        engine = FakeWeatherEngine()
        template = engine.weather.profile("temperature:continuous")
        temp_id = PHENOMENON_DATASETS["temperature"]
        road_id = PHENOMENON_DATASETS["road_temperature"]
        temperature = dict(template, id=temp_id, phenomenon="temperature", minimum=0.1, frost_maximum=4.)
        road = dict(template, id=road_id, phenomenon="road_temperature", minimum=0.3, frost_maximum=8.)
        temp_reference = dict(temperature, minimum=0.2, frost_maximum=5.)
        road_reference = dict(road, minimum=0.4, frost_maximum=9.)
        engine.weather = Mock()
        engine.weather.profile.side_effect = {temp_id: temperature, road_id: road}.__getitem__
        engine.weather.svv_profile.side_effect = {temp_id: temp_reference, road_id: road_reference}.__getitem__
        air = Scenario(engine, dataset=temp_id)
        pavement = Scenario(engine, dataset=road_id)
        for network, svv_only in (("frost", True), ("frost", False), ("empty", False)):
            air.network = pavement.network = network
            air.pross_only = pavement.pross_only = svv_only
            air.station_since = pavement.station_since = "2026-01-01"
            air.counties = pavement.counties = ["01"]
            self.assertEqual(air.limits(), (0.2, 5., 2))
            self.assertEqual(pavement.limits(), air.limits())
        self.assertNotEqual(pavement.limits(), frost_limits(road_reference))

    def test_temperature_road_calibration_is_fixed_and_configurable(self):
        from station_app import StationEngine
        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        reference = dict(profile, baseline=np.array([1., 20., 4.]), minimum=.2, frost_maximum=25.)
        engine.weather = Mock()
        engine.weather.profile.return_value = profile
        engine.weather.svv_profile.return_value = reference
        network = Mock()
        network.national_corridor_mask.return_value = np.array([True, False, True])
        engine.root = engine
        engine.road_calibration_lock = threading.Lock()
        engine.road_calibration_limits = {}
        engine.road_network_provider = lambda: network
        engine.road_colour_limits = MethodType(StationEngine.road_colour_limits, engine)
        scenario = Scenario(engine)
        with patch("station_app.ROAD_COLOUR_CALIBRATION_ENABLED", True), \
                patch("station_app.ROAD_COLOUR_CALIBRATION_DISTANCE_M", 200):
            expected = tuple(np.percentile([1., 4.], [0.5, 99.5])) + (2,)
            self.assertEqual(scenario.limits(), expected)
            scenario.network = "empty"
            scenario.counties = ["01"]
            scenario.added = [1]
            self.assertEqual(scenario.limits(), expected)
            network.national_corridor_mask.assert_called_once_with(engine, 200)
        with patch("station_app.ROAD_COLOUR_CALIBRATION_ENABLED", False):
            self.assertEqual(scenario.limits(), frost_limits(reference))
        reference["baseline"] = np.array([0., 4., 8.])
        network.national_corridor_mask.return_value = np.array([True, True, True])
        with patch("station_app.ROAD_COLOUR_CALIBRATION_ENABLED", True):
            expected = tuple(np.percentile([4., 8.], [0.5, 99.5])) + (2,)
            self.assertEqual(scenario.limits(), expected)

    def test_svv_reference_uses_its_own_national_baseline_and_requires_precalculation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "all.npz"
            baseline = np.array([0.01, 0.02, 0.03], dtype=np.float32)
            save_exact_cache(path, baseline, baseline + 1, np.zeros(3, dtype=np.int32), 2)
            profile = FakeWeatherEngine().weather.profile("temperature:continuous")
            profile.update(cache_path=path, baseline=baseline, cell_m=1000, model=Mock())
            registry = WeatherRegistry(root, None, None, None)
            with patch.object(registry, "profile", return_value=profile), \
                    patch("station_pross.pross_excluded", return_value=frozenset([0])):
                with self.assertRaisesRegex(ValueError, "precompute_frost.py"):
                    registry.svv_profile(profile["id"])
                best = np.array([0.1, 0.5, 0.9], dtype=np.float32)
                with patch("station_weather.make_cache", return_value={}), \
                        patch("station_weather.exact_network",
                              return_value=(best, np.full(3, np.inf), np.zeros(3, dtype=np.int32))):
                    reference = registry.svv_profile(profile["id"], build=True)
                expected = np.percentile(best, [0.5, 99.5])
                np.testing.assert_allclose(frost_limits(reference)[:2], expected)
                np.testing.assert_array_equal(profile["baseline"], baseline)
                self.assertEqual(profile["minimum"], 0.01)
                self.assertIs(registry.svv_profile(profile["id"]), reference)
                with patch("station_pross.pross_excluded", return_value=frozenset([0, 1])):
                    with self.assertRaisesRegex(ValueError, "SVV"):
                        registry.svv_profile(profile["id"])
                    profile["phenomenon"] = "precipitation"
                    empty_reference = registry.svv_profile(profile["id"])
                    np.testing.assert_array_equal(empty_reference["baseline"], profile["baseline"])
                    self.assertIn("tomt", empty_reference["colour_reference_warning"])

    def test_county_borders_are_unfiltered_cached_lines_above_overlays(self):
        from shapely.geometry import shape
        engine = FakeWeatherEngine()
        with patch.object(CountyMask, "load", return_value=county_frame()):
            client = create_app(engine).test_client()
            response = client.get("/api/counties/borders")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["type"], "FeatureCollection")
            lines = shape(response.json["features"][0]["geometry"])
            self.assertIn(lines.geom_type, ("LineString", "MultiLineString"))
            # All original county edges remain represented, including the shared border.
            from pyproj import Transformer
            from shapely.ops import transform
            native = transform(Transformer.from_crs(4326, 25833, always_xy=True).transform, lines)
            self.assertAlmostEqual(native.length, 9000, places=3)
            client.post("/api/action", json=dict(action="counties", value=[]))
            self.assertEqual(client.get("/api/counties/borders").json, response.json)
            client.post("/api/action", json=dict(action="counties", value=["01"]))
            self.assertEqual(client.get("/api/counties/borders").json, response.json)
            compressed = client.get("/api/counties/borders", headers={"Accept-Encoding": "gzip"})
            self.assertEqual(compressed.headers["Content-Encoding"], "gzip")
            self.assertEqual(gzip.decompress(compressed.data), response.data)
            self.assertIn("max-age=", compressed.headers["Cache-Control"])

    def test_simplified_controls_keep_filters_and_remove_distractions(self):
        client = create_app(FakeWeatherEngine()).test_client()
        with client.get("/") as response:
            html = response.get_data(as_text=True)
        for removed in ('id="threshold"', 'id="weather-period"', 'id="mean"',
                        'data-mode="affected"', 'data-mode="placement"',
                        'id="road-length"', 'id="road-error"', 'id="road-p90"',
                        'id="road-improved"', 'id="road-score-legend"',
                        'seNorge 30 år + basseng', 'Nettets usikkerhet',
                        'Vegnettet er skjult', 'Vis antall berørte stasjoner først.',
                        'id="pross-only"', 'Om beregningene', 'id="model-warning"',
                        'Shift + klikk i kartet', 'Kartlagets opasitet'):
            self.assertNotIn(removed, html)
        for retained in ('data-mode="uncertainty"', 'data-mode="baseline"', 'data-mode="reduction"',
                         'id="station-date-apply"', 'id="road-corridor"', 'id="rmse-min"',
                         'id="county-options"', 'id="selected-station"', 'id="remove-frost"',
                         'id="kuldehull-picker"', 'id="road-picker"', 'id="station-picker"',
                         'id="road-show"', 'id="road-score-4"', 'id="road-score-5"', 'Vis vegnett',
                         'Vis vegscoreklasse 4', 'Vis vegscoreklasse 5',
                         'Alle værstasjoner', 'Vegværstasjoner', 'Værstasjoner', 'gjennomsiktighet'):
            self.assertIn(retained, html)
        self.assertNotIn('id="road-classes"', html)
        self.assertNotIn('id="road-score-filters"', html)

    def test_station_network_dropdown_switches_svv_all_and_empty_atomically(self):
        client = create_app(FakeWeatherEngine()).test_client()
        with patch("station_app.pross_excluded", return_value=frozenset([0])):
            response = client.post("/api/action", json=dict(action="network", value="svv"))
            self.assertEqual(response.status_code, 200, response.json)
            self.assertTrue(response.json["pross_only"])
            self.assertEqual(response.json["network"], "frost")
            self.assertEqual(response.json["fixed_count"], 1)
            response = client.post("/api/action", json=dict(action="network", value="frost", pross_only=False))
            self.assertFalse(response.json["pross_only"])
            self.assertEqual(response.json["fixed_count"], 2)
            response = client.post("/api/action", json=dict(action="network", value="empty", pross_only=False))
            self.assertEqual(response.json["fixed_count"], 0)
            self.assertIsNone(response.json["mean_rmse"])
            self.assertEqual(client.post("/api/action", json=dict(
                action="network", value="frost", pross_only=False)).json["fixed_count"], 2)
        with patch("station_app.pross_excluded", return_value=frozenset([0])):
            response = client.post("/api/action", json=dict(action="network", value="svv"))
            self.assertEqual(response.status_code, 200)
            state = client.get("/api/state").json
            self.assertTrue(state["pross_only"])
            self.assertEqual(state["fixed_count"], 1)

    def test_multiple_date_exclusions_serialize_and_cache_across_endpoints(self):
        from greedy_400_stations_pruned import CACHE_KEYS, make_cache
        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        stations = pd.concat([profile["stations"], profile["stations"]], ignore_index=True)
        stations["station_id"] = ["SN0", "SN1", "SN2", "SN3"]
        stations["last_time"] = ["2026-10-14", "2020-01-01", "", "2026-10-14"]
        for key in CACHE_KEYS:
            if key not in stations:
                stations[key] = 0.0
        engine.cache = make_cache(stations.iloc[:3])
        engine.weather = Mock(wraps=engine.weather)
        engine.weather.available = FakeWeatherEngine.weather.available
        engine.weather.profile.return_value = profile
        profile.update(stations=stations, model=Mock())
        with tempfile.TemporaryDirectory() as folder:
            profile["cache_path"] = Path(folder) / "all.npz"
            fallback = np.array([4, 5, 6], dtype=np.float32)
            with patch("station_app.exact_network", return_value=(fallback, fallback, np.zeros(3))) as predict:
                client = create_app(engine).test_client()
                response = client.post("/api/action", json=dict(action="station_since", value="2026-01-01"))
                self.assertEqual(response.status_code, 200, response.json)
                self.assertEqual(response.json["fixed_count"], 2)
                self.assertEqual(client.get("/api/state").status_code, 200)
                self.assertEqual(client.post("/api/rmse-range", json={}).status_code, 200)
                self.assertEqual(client.get("/api/map.png?size=400").status_code, 200)
                self.assertEqual(predict.call_count, 1)
            self.assertEqual(len(list(Path(folder).glob("frost_masked_*.npy"))), 1)
            with patch("station_app.exact_network", side_effect=AssertionError("Must reuse disk cache")):
                scenario = Scenario(engine)
                scenario.station_since = "2026-01-01"
                np.testing.assert_array_equal(scenario.surface(), fallback)
                self.assertTrue(all(type(index) is int for index in scenario.masked()))

    def test_date_preview_counts_without_predictions_or_scenario_changes(self):
        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        profile["stations"]["last_time"] = ["2026-10-14", None]
        engine.weather = Mock(wraps=engine.weather)
        engine.weather.available = FakeWeatherEngine.weather.available
        engine.weather.profile.return_value = profile
        client = create_app(engine).test_client()
        before = client.get("/api/state").json
        with patch("station_app.exact_network", side_effect=AssertionError("Must not predict")), \
                patch.object(Scenario, "surface", side_effect=AssertionError("Must not compute a surface")):
            response = client.post("/api/station-date-preview", json=dict(value="2026-10-14"))
        self.assertEqual(response.status_code, 200, response.json)
        layer = response.json["layers"][0]
        self.assertEqual((layer["total"], layer["removed"], layer["remaining"], layer["unknown"]), (2, 1, 1, 1))
        after = client.get("/api/state").json
        self.assertEqual(after["station_since"], "")
        self.assertEqual(after["revision"], before["revision"])
        self.assertEqual(after["fixed_count"], 2)
        response = client.post("/api/action", json=dict(action="station_since", value="2026-10-14",
                                                       expected_revision=before["revision"]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["fixed_count"], 1)
        restored = client.post("/api/station-date-preview", json=dict(value="")).json["layers"][0]
        self.assertEqual((restored["removed"], restored["restored"]), (0, 1))
        stale = client.post("/api/action", json=dict(action="station_since", value="",
                                                    expected_revision=before["revision"]))
        self.assertEqual(stale.status_code, 400)
        self.assertEqual(client.get("/api/state").json["station_since"], "2026-10-14")
        for value in (None, "2026-02-31", "20261014"):
            self.assertEqual(client.post("/api/station-date-preview", json=dict(value=value)).status_code, 400)

    def test_date_preview_respects_svv_and_checked_layers(self):
        engine = FakeWeatherEngine()
        scenario = Scenario(engine)
        scenario.pross_only = True
        scenario.layers = ["temperature:continuous", "snow:class"]
        with patch("station_app.pross_excluded", return_value=frozenset([0])):
            preview = scenario.station_date_preview("2027-01-01")
        self.assertEqual(len(preview["layers"]), 2)
        self.assertTrue(all(layer["total"] == layer["removed"] == 1 for layer in preview["layers"]))
        self.assertEqual(scenario.station_since, "")
        self.assertEqual(scenario.revision, 0)

    def test_svv_cached_network_uses_original_indices_and_second_best_without_prediction(self):
        from greedy_400_stations_pruned import CACHE_KEYS, make_cache
        from station_weather import exact_network, save_exact_cache
        engine = FakeWeatherEngine()
        engine.default_pross_only = True
        profile = engine.weather.profile("temperature:continuous")
        stations = pd.concat([profile["stations"], profile["stations"].iloc[:1]], ignore_index=True)
        for key in CACHE_KEYS:
            stations[key] = 0.0
        stations["x"] = [0, 1000, 2000]
        stations["station_id"] = ["SN0", "SN1", "SN2"]
        target = stations.copy()
        model = Mock(spec=["predict"])
        model.predict.side_effect = lambda values: values[:, 5] / 1000 + 1
        engine.cache = make_cache(target)
        baseline, second, winner = exact_network(model, make_cache(stations), engine.cache, False)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "all.npz"
            save_exact_cache(path, baseline, second, winner, 3)
            profile.update(stations=stations, baseline=baseline, second=second, winner=winner,
                           model=model, cache_path=path, cell_m=1000)
            self.assertIsNone(restricted_profile(profile, engine.cache, [1]))
            cached = restricted_profile(profile, engine.cache, np.array([1], dtype=np.int64), build=True)
            np.testing.assert_array_equal(cached["baseline"], [1, 2, 1])
            np.testing.assert_array_equal(cached["winner"], [0, 0, 2])
            engine.weather = Mock(wraps=engine.weather)
            engine.weather.available = FakeWeatherEngine.weather.available
            engine.weather.profile.return_value = profile
            with patch("station_app.pross_excluded", return_value=frozenset([1])), \
                    patch("station_weather.exact_network", side_effect=AssertionError("Cache must be used")), \
                    patch("station_app.exact_network", side_effect=AssertionError("Second best must be used")):
                scenario = Scenario(engine)
                np.testing.assert_array_equal(scenario.surface(), [1, 2, 1])
                self.assertEqual(scenario.fixed().station_id.tolist(), ["SN0", "SN2"])
                np.testing.assert_array_equal(scenario.frost_surface(omit_index=0), [3, 2, 1])
                np.testing.assert_array_equal(scenario.reference(), cached["baseline"])
                scenario.pross_only = False
                np.testing.assert_array_equal(scenario.surface(), [1, 1, 1])
            self.assertIsNone(restricted_profile(profile, engine.cache, [0, 1, 2], build=True))
            self.assertIs(restricted_profile(profile, engine.cache, []), profile)
            from precompute_frost import main
            (Path(folder) / "norway_grid_1000m.gpkg").touch()
            grid = stations.assign(objtype="Land")
            registry = Mock(available=[FakeWeatherEngine.weather.available[0]], target_cache=engine.cache)
            registry.profile.return_value = profile
            images = Mock()
            with patch("precompute_frost.OUT", Path(folder)), \
                    patch("precompute_frost.pyogrio.read_dataframe", return_value=grid), \
                    patch("precompute_frost.load_grid_all_fields", return_value=grid), \
                    patch("precompute_frost.WeatherRegistry", return_value=registry), \
                    patch("precompute_frost.WeatherMapCache", return_value=images), \
                    patch("station_app.pross_excluded", return_value=frozenset([1])), \
                    patch("station_app.StationEngine._prepare_raster"), \
                    patch("station_weather.exact_network", side_effect=AssertionError("Must reuse SVV cache")):
                self.assertEqual(main(["--phenomenon", "temperature", "--kind", "continuous",
                                       "--cell-m", "1000", "--prepare-maps"]), 0)
            self.assertEqual(images.get.call_count, 6)
            all_map = images.get.call_args_list[0].args[1]
            svv_map = images.get.call_args_list[3].args[1]
            self.assertEqual(all_map["cache_path"], profile["cache_path"])
            self.assertNotEqual(svv_map["cache_path"], profile["cache_path"])
            np.testing.assert_array_equal(svv_map["baseline"], [1, 2, 1])
            expected_limits = frost_limits(svv_map)
            for call in images.get.call_args_list:
                self.assertEqual(call.args[3], expected_limits)
            self.assertNotEqual(expected_limits, frost_limits(profile))

    def test_precompute_network_selection_defaults_to_both(self):
        from precompute_frost import parse_args
        self.assertEqual(parse_args([]).station_network, "both")
        self.assertEqual(parse_args(["--station-network", "svv"]).station_network, "svv")
        self.assertEqual(parse_args(["--station-network", "all"]).station_network, "all")

    def test_date_excludes_old_and_unknown_dates_and_clear_restores(self):
        engine = FakeWeatherEngine()
        profile = engine.weather.profile("temperature:continuous")
        profile["stations"]["last_time"] = ["2026-10-14", "2020-01-01"]
        engine.weather = Mock(wraps=engine.weather)
        engine.weather.available = FakeWeatherEngine.weather.available
        engine.weather.profile.return_value = profile
        scenario = Scenario(engine)
        before = scenario.surface().copy()
        scenario.apply(dict(action="station_since", value="2026-10-14"))
        self.assertEqual(scenario.masked(), frozenset([1]))
        np.testing.assert_array_equal(scenario.surface(), profile["second"])
        self.assertEqual(scenario.fixed().station_id.tolist(), ["SN1"])
        scenario.apply(dict(action="station_since", value=""))
        np.testing.assert_array_equal(scenario.surface(), before)
        profile["stations"]["last_time"] = ["", None]
        scenario.apply(dict(action="station_since", value="2026-01-01"))
        self.assertEqual(scenario.masked(), frozenset([0, 1]))
        self.assertIsNone(scenario.surface())
        self.assertIsNone(scenario.state()["mean_rmse"])
        with self.assertRaises(ValueError):
            scenario.apply(dict(action="station_since", value="2026-02-31"))

    def test_production_default_svv_and_date_intersection(self):
        engine = FakeWeatherEngine()
        engine.default_pross_only = True
        with patch("station_app.pross_excluded", return_value=frozenset([0])):
            scenario = Scenario(engine)
            self.assertTrue(scenario.pross_only)
            self.assertEqual(scenario.masked(), frozenset([0]))
            scenario.apply(dict(action="station_since", value="2027-01-01"))
            self.assertEqual(scenario.masked(), frozenset([0, 1]))

    def test_counties_do_not_remove_stations_or_change_network_values(self):
        engine = FakeWeatherEngine()
        engine.render = Mock(wraps=engine.render)
        roads = Mock()
        roads.corridor_mask.return_value = np.array([True, True, False])
        with patch.object(CountyMask, "load", return_value=county_frame()):
            client = create_app(engine, road_network=roads).test_client()
            before = client.get("/api/state").json
            result = client.post("/api/action", json=dict(action="counties", value=["01"]))
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json["fixed_stations"], before["fixed_stations"])
            self.assertEqual(result.json["mean_rmse"], 3.5)
            bounds = [5, 58, 12, 64]
            response = client.get("/api/map.png")
            self.assertEqual(response.status_code, 200, response.get_json(silent=True))
            np.testing.assert_array_equal(engine.render.call_args.kwargs["cell_mask"], [True, True, False])
            result = client.post("/api/rmse-range", json=dict(
                roads=dict(corridor=True, bounds=bounds)))
            self.assertEqual(result.json["cells"], 2)
            engine.prediction_at = lambda index, targets, dataset: engine.prediction(index, dataset)[targets]
            result = client.post("/api/preview", json=dict(lon=5, lat=60,
                mode="uncertainty", revision=before["revision"]+1))
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.headers["X-Preview-Scope"], "counties")
            np.testing.assert_array_equal(engine.render.call_args.kwargs["cell_mask"], [True, True, False])
            with tempfile.TemporaryDirectory() as folder:
                response = client.post("/api/export.gpkg", json=dict(revision=before["revision"]+1))
                self.assertEqual(response.status_code, 200)
                path = Path(folder)/"result.gpkg"
                path.write_bytes(response.data)
                full = pyogrio.read_dataframe(path, layer="rmse_norway")
                visible = pyogrio.read_dataframe(path, layer="rmse_visible")
                self.assertEqual(len(full), 3)
                self.assertEqual(visible.cell_id.tolist(), [0])
            result = client.post("/api/action", json=dict(action="counties", value=[]))
            self.assertEqual(result.status_code, 200)
            result = client.post("/api/rmse-range", json={})
            self.assertEqual(result.json["maximum"], 0)
            result = client.post("/api/action", json=dict(action="counties", value=["01", "02"]))
            self.assertIsNone(result.json["counties"])
            self.assertEqual(client.post("/api/action", json=dict(
                action="counties", value=["unknown"])).status_code, 400)
            self.assertEqual(client.post("/api/action", json=dict(
                action="station_since", value=None)).status_code, 400)

    def test_county_cells_multiselect_and_precise_image_alpha(self):
        with patch.object(CountyMask, "load", return_value=county_frame()):
            counties = CountyMask(Path("unused"))
            engine = FakeWeatherEngine()
            np.testing.assert_array_equal(counties.cells(engine, []), [False]*3)
            self.assertIsNone(counties.cells(engine, ["01", "02"]))
            self.assertEqual([x["name"] for x in counties.options()], ["East", "West"])
        # A simple geographic county verifies alpha clipping independently of the app.
        frame = gpd.GeoDataFrame(dict(fylkesnavn=["West", "East"]),
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326,
            index=pd.Index(["01", "02"], name="code")).to_crs(25833)
        with patch.object(CountyMask, "load", return_value=frame):
            counties = CountyMask(Path("unused"))
            buffer = BytesIO()
            Image.new("RGBA", (100, 50), (255, 0, 0, 210)).save(buffer, format="PNG")
            buffer.seek(0)
            image = counties.clip_image(buffer, [0, 0, 2, 1], ["01"])
            pixels = np.asarray(Image.open(image))
            self.assertTrue((pixels[:, :45, 3] == 210).all())
            self.assertTrue((pixels[:, 55:, 3] == 0).all())


if __name__ == "__main__":
    unittest.main()
