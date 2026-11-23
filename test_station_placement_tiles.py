from io import BytesIO
import gzip
import json
from pathlib import Path
import sys
import time
import tempfile
import unittest
from types import SimpleNamespace
from types import MethodType
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer

from station_app import StationEngine, create_app, legend_values
from station_counties import CountyMask
from station_placement_tiles import PlacementTiles, tile_bounds
from test_station_app import FakeWeatherEngine
from test_station_filters import county_frame


class FrozenRendererTests(unittest.TestCase):
    def test_caches_hash_the_frozen_executable_without_loose_python_sources(self):
        from station_weather_maps import WeatherMapCache

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / "Vaernett.exe"
            executable.write_bytes(b"first bundled renderer")
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "executable", str(executable)), \
                    patch("station_placement_tiles.__file__", str(root / "station_placement_tiles.pyc")), \
                    patch("station_weather_maps.__file__", str(root / "station_weather_maps.pyc")):
                placement = PlacementTiles().renderer
                weather = WeatherMapCache(root / "maps").renderer_version
                self.assertEqual(PlacementTiles().renderer, placement)
                self.assertEqual(WeatherMapCache(root / "maps").renderer_version, weather)
                executable.write_bytes(b"updated bundled renderer")
                self.assertNotEqual(PlacementTiles().renderer, placement)
                self.assertNotEqual(WeatherMapCache(root / "maps").renderer_version, weather)


class PlacementTileTests(unittest.TestCase):
    def setUp(self):
        x, y = Transformer.from_crs(4326, 25833, always_xy=True).transform(10.75, 59.91)
        xs, ys = np.meshgrid(np.arange(20) * 500 + x, np.arange(20) * 500 + y)
        self.active = StationEngine.__new__(StationEngine)
        self.active.cell_m = 500
        self.active.minimum, self.active.maximum = 0, 400
        self.active.grid = pd.DataFrame(dict(x=xs.ravel(), y=ys.ravel()))
        self.active._prepare_raster()
        self.layer = dict(values=np.arange(400, dtype=np.float32), cell_m=500, minimum=0,
                          maximum=400, source="saved.npz", source_version=[10, 20],
                          dataset="temperature:senorge30", station_network="svv", bounds=self.active.bounds)
        self.cache = PlacementTiles()
        self.version = self.cache.version(self.active, self.layer)
        self.z = 10
        self.x = int((10.78 + 180) / 360 * 2 ** self.z)
        lat = np.radians(59.94)
        self.y = int((1 - np.arcsinh(np.tan(lat)) / np.pi) / 2 * 2 ** self.z)

    def test_startup_overview_tiles_survive_restart_without_rendering(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = PlacementTiles(folder)
            ready = cache.warm(self.active, self.layer)
            self.assertGreater(ready, 0)
            self.assertEqual(len(list(Path(folder).rglob("*.bin.gz"))), ready)
            restarted = PlacementTiles(folder)
            with patch.object(self.active, "_render_pixels", side_effect=AssertionError("Must load startup tiles from disk")):
                self.assertEqual(restarted.warm(self.active, self.layer), ready)
            self.assertIsNone(restarted.source)

    def test_numeric_tiles_preserve_exact_values_colours_and_no_data(self):
        bounds = tile_bounds(self.z, self.x, self.y)
        raw, _ = self.cache.image(self.active, self.layer, self.version, self.z, self.x, self.y,
                                  0, None, None, None, lambda image, bounds: image, numeric=True)
        linear_raw, _ = self.cache.image(self.active, self.layer, self.version, self.z, self.x, self.y,
                                         0, None, None, None, lambda image, bounds: image, numeric=True,
                                         map_scale="linear")
        data = gzip.decompress(raw)
        linear_data = gzip.decompress(linear_raw)
        self.assertEqual(len(data), 8+256*256*8)
        np.testing.assert_array_equal(np.frombuffer(data, dtype="<u4", count=2), [256, 256])
        values = np.frombuffer(data, dtype="<f4", offset=8, count=256*256).reshape(256, 256)
        colours = np.frombuffer(data, dtype=np.uint8, offset=8+256*256*4).reshape(256, 256, 4)
        linear_colours = np.frombuffer(linear_data, dtype=np.uint8, offset=8+256*256*4).reshape(256, 256, 4)
        self.assertFalse(np.array_equal(colours, linear_colours))
        with Image.open(self.active.render(self.layer["values"], "placement", 0, 400,
                                          display_size=256, viewport=bounds)) as expected:
            np.testing.assert_array_equal(colours, np.asarray(expected))
        self.assertFalse(np.isfinite(values[colours[:, :, 3] == 0]).any())
        visible = colours[:, :, 3] > 0
        self.assertTrue(visible.any())
        self.assertTrue(np.isin(values[visible], self.layer["values"]).all())
        for cutoff in (0, 100, 150, 399):
            filtered = colours.copy()
            if cutoff:
                filtered[:, :, 3] = np.where(values > cutoff, colours[:, :, 3], 0)
            with Image.open(self.active.render(self.layer["values"], "placement", 0, 400,
                    cell_mask=self.layer["values"] > cutoff if cutoff else None,
                    display_size=256, viewport=bounds)) as expected:
                actual = np.asarray(expected)
                np.testing.assert_array_equal(filtered[:, :, 3], actual[:, :, 3])
                np.testing.assert_array_equal(filtered[actual[:, :, 3] > 0], actual[actual[:, :, 3] > 0])

    def test_uncertainty_tiles_keep_native_cell_colours(self):
        values = np.where(np.arange(400) % 2, 400, 0).astype(np.float32)
        layer = dict(values=values, minimum=0.1, maximum=400, reduction_max=400)
        version = self.cache.version(self.active, dict(layer, source="uncertainty", source_version=[1]))
        z = 12
        x = int((10.78 + 180) / 360 * 2 ** z)
        latitude = np.radians(59.94)
        y = int((1 - np.arcsinh(np.tan(latitude)) / np.pi) / 2 * 2 ** z)
        for render_mode, raster_values in (("uncertainty", values),
                                           ("reduction", values),
                                           ("change", np.where(values > 0, values, -400))):
            tiled_layer = dict(layer, values=raster_values)
            version = self.cache.version(self.active, dict(tiled_layer, source=render_mode, source_version=[1]))
            raw, _ = self.cache.image(self.active, tiled_layer, version, z, x, y,
                                      0, None, None, None, lambda image, bounds: image, numeric=True,
                                      render_mode=render_mode)
            data = gzip.decompress(raw)
            pixels = np.frombuffer(data, dtype=np.uint8, offset=8 + 256*256*4).reshape(256, 256, 4)
            visible = pixels[pixels[:, :, 3] > 0]
            self.assertGreater(len(visible), 0)
            self.assertLessEqual(len(np.unique(visible, axis=0)), 2)

    def test_numeric_endpoint_is_compressed_and_rejects_server_cutoffs(self):
        engine = FakeWeatherEngine()
        engine.placement_layer = Mock(return_value=self.layer)
        engine.at_resolution = Mock(return_value=self.active)
        client = create_app(engine).test_client()
        url = f"/api/placement/tiles/{self.z}/{self.x}/{self.y}.bin"
        query = dict(dataset=self.layer["dataset"], network="svv", version=self.version)
        response = client.get(url, query_string=query)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Encoding"], "gzip")
        self.assertEqual(len(gzip.decompress(response.data)), 8+256*256*8)
        self.assertEqual(client.get(url, query_string=query).headers["X-Map-Cache"], "HIT")
        self.assertEqual(client.get(url, query_string=dict(query, rmse_min=100)).status_code, 400)

    def weather_engine(self):
        from test_station_app import FakeWeather

        engine = FakeWeatherEngine()
        engine.weather = FakeWeather()
        engine.grid = self.active.grid.iloc[:3].copy()
        engine.cell_m = 500
        StationEngine._prepare_raster(engine)
        engine.render = MethodType(StationEngine.render, engine)
        engine._render_pixels = MethodType(StationEngine._render_pixels, engine)
        return engine

    def decode_raster(self, response):
        self.assertEqual(response.status_code, 200, response.data[:200])
        self.assertEqual(response.headers["Content-Encoding"], "gzip")
        data = gzip.decompress(response.data)
        width, height = np.frombuffer(data, dtype="<u4", count=2).tolist()
        count = width*height
        self.assertEqual(len(data), 8+count*8)
        values = np.frombuffer(data, dtype="<f4", count=count, offset=8).reshape(height, width)
        pixels = np.frombuffer(data, dtype=np.uint8, offset=8+count*4).reshape(height, width, 4)
        return values, pixels

    def test_numeric_weather_matches_png_and_tracks_station_changes(self):
        engine = self.weather_engine()
        client = create_app(engine).test_client()
        query = dict(mode="uncertainty", size=400)
        before, _ = self.decode_raster(client.get("/api/map.bin", query_string=query))
        linear_values, linear_pixels = self.decode_raster(
            client.get("/api/map.bin", query_string=dict(query, map_scale="linear")))
        np.testing.assert_array_equal(before, linear_values)
        self.assertFalse(np.array_equal(linear_pixels, self.decode_raster(
            client.get("/api/map.bin", query_string=query))[1]))
        with Image.open(BytesIO(client.get("/api/map.png", query_string=dict(query, map_scale="linear")).data)) as linear_png:
            np.testing.assert_array_equal(linear_pixels, np.asarray(linear_png))
        for mode in ("uncertainty", "baseline"):
            query["mode"] = mode
            values, pixels = self.decode_raster(client.get("/api/map.bin", query_string=query))
            with Image.open(BytesIO(client.get("/api/map.png", query_string=query).data)) as expected:
                np.testing.assert_array_equal(pixels, np.asarray(expected))
            cutoff = 3
            with Image.open(BytesIO(client.get("/api/map.png", query_string=dict(query, rmse_min=cutoff)).data)) as filtered:
                np.testing.assert_array_equal(np.where(values > cutoff, pixels[:, :, 3], 0), np.asarray(filtered)[:, :, 3])
        self.assertEqual(client.post("/api/action", json=dict(action="add", lon=5, lat=60)).status_code, 200)
        after, _ = self.decode_raster(client.get("/api/map.bin?mode=uncertainty&size=400"))
        self.assertFalse(np.array_equal(before, after, equal_nan=True))
        baseline, _ = self.decode_raster(client.get("/api/map.bin?mode=baseline&size=400"))
        np.testing.assert_array_equal(before, baseline)
        for invalid in ("mode=reduction", "mode=unknown", "size=1", "rmse_min=1", "rmse_min=nan",
                        "map_scale=invalid", "mode=reduction&map_scale=linear"):
            self.assertEqual(client.get("/api/map.bin?"+invalid).status_code, 400)

    def test_uncertainty_tiles_use_current_scenario_and_reject_stale_requests(self):
        engine = self.weather_engine()
        client = create_app(engine).test_client()
        query = dict(revision=0, dataset="temperature:continuous", mode="uncertainty")
        url = f"/api/map/tiles/{self.z}/{self.x}/{self.y}.bin"
        response = client.get(url, query_string=query)
        values, pixels = self.decode_raster(response)
        self.assertEqual(values.shape, (256, 256))
        self.assertTrue((pixels[:, :, 3] > 0).any())
        self.assertEqual(client.get(url, query_string=query).headers["X-Map-Cache"], "HIT")
        stale = client.get(url, query_string=dict(query, revision=1))
        self.assertEqual(stale.status_code, 409)
        reduction = client.get(url, query_string=dict(query, mode="reduction"))
        self.decode_raster(reduction)
        self.assertEqual(reduction.status_code, 200)
        with patch.dict("station_app.PHENOMENON_DATASETS",
                        temperature="temperature:continuous", snow="snow:class"):
            combined = client.post("/api/action", json=dict(action="layers", value=["temperature", "snow"]))
        self.assertEqual(combined.status_code, 200)
        combined_state = combined.get_json()
        combined_tiles = client.get(url, query_string=dict(
            revision=combined_state["revision"], dataset=combined_state["dataset"], mode="reduction"))
        self.decode_raster(combined_tiles)
        self.assertEqual(combined_tiles.status_code, 200)

    def test_linear_map_scale_renders_signed_change_differently(self):
        values = np.linspace(-2, 2, len(self.active.grid), dtype=np.float32)
        with Image.open(self.active.render(values, "change", -2, 2, 2)) as logarithmic:
            log_pixels = np.asarray(logarithmic)
        with Image.open(self.active.render(values, "change", -2, 2, 2, map_scale="linear")) as linear:
            linear_pixels = np.asarray(linear)
        self.assertFalse(np.array_equal(log_pixels, linear_pixels))

    def test_numeric_weather_preserves_road_county_and_viewport_masks(self):
        from station_kuldehull import KuldehullLayer

        engine = self.weather_engine()
        roads = Mock()
        roads.corridor_mask.return_value = np.array([True, False, True])
        lake = KuldehullLayer(Path("unused.gpkg"))
        lake.mask = lambda bounds, width, height, depth: np.ones((height, width), dtype=np.uint8)
        client = create_app(engine, road_network=roads, kuldehull_layer=lake).test_client()
        viewport = dict(only=True, depth=0, bounds=[
            engine.bounds[0][1], engine.bounds[0][0], engine.bounds[1][1], engine.bounds[1][0]],
            width=80, height=60)
        for extra in ({}, dict(kuldehull=json.dumps(viewport))):
            query = dict(size=400, roads=json.dumps(dict(corridor=True, bounds=viewport["bounds"])), **extra)
            values, pixels = self.decode_raster(client.get("/api/map.bin", query_string=query))
            with Image.open(BytesIO(client.get("/api/map.png", query_string=query).data)) as reference:
                np.testing.assert_array_equal(pixels, np.asarray(reference))
            self.assertTrue(np.isnan(values[pixels[:, :, 3] == 0]).all())
            self.assertTrue(np.isfinite(values[pixels[:, :, 3] > 0]).all())
            if extra:
                self.assertEqual(values.shape, (60, 80))
        with patch.object(CountyMask, "load", return_value=county_frame()):
            self.assertEqual(client.post("/api/action", json=dict(action="counties", value=[])).status_code, 200)
            values, pixels = self.decode_raster(client.get("/api/map.bin?size=400"))
            self.assertFalse(pixels[:, :, 3].any())
            self.assertTrue(np.isnan(values).all())

    def test_numeric_combined_and_missing_weather_preserve_colours(self):
        engine = self.weather_engine()
        client = create_app(engine).test_client()
        with patch.dict("station_app.PHENOMENON_DATASETS", temperature="temperature:continuous", snow="snow:class"):
            result = client.post("/api/action", json=dict(action="layers", value=["temperature", "snow"]))
            self.assertEqual(result.status_code, 200, result.json)
            _, pixels = self.decode_raster(client.get("/api/map.bin?size=400"))
            with Image.open(BytesIO(client.get("/api/map.png?size=400").data)) as reference:
                np.testing.assert_array_equal(pixels, np.asarray(reference))
        engine = self.weather_engine()
        profile = engine.weather.profile("temperature:continuous")
        profile["baseline"] = np.array([np.inf, np.nan, 3], dtype=np.float32)
        engine.weather.profile = Mock(return_value=profile)
        client = create_app(engine).test_client()
        values, pixels = self.decode_raster(client.get("/api/map.bin?size=400"))
        self.assertTrue(np.isposinf(values[pixels[:, :, 3] > 0]).any())
        with Image.open(BytesIO(client.get("/api/map.png?size=400").data)) as reference:
            np.testing.assert_array_equal(pixels, np.asarray(reference))

    def test_legend_numbers_follow_actual_colour_scales_and_modes(self):
        self.assertAlmostEqual(legend_values("uncertainty", .1, 10)[2], 1)
        self.assertAlmostEqual(legend_values("uncertainty", .1, 10, legend_scale="linear")[2], 5.05)
        self.assertAlmostEqual(legend_values("placement", 0, 99)[2], 9)
        self.assertAlmostEqual(legend_values("placement", 0, 99, legend_scale="linear")[2], 49.5)
        self.assertEqual(legend_values("terrain:tpi_1km", -2, 2), [-2, -1, 0, 1, 2])
        change = legend_values("change", .1, 6, 2)
        np.testing.assert_allclose([change[0], change[2], change[4]], [-2, 0, 2])
        np.testing.assert_allclose(legend_values("change", .1, 6, 2, legend_scale="linear"),
                                   [-2, -1, 0, 1, 2])
        self.assertAlmostEqual(legend_values("reduction", .1, 6, .5)[-1], .5)
        np.testing.assert_allclose(legend_values("reduction", .1, 6, .5, legend_scale="linear"),
                                   [0, .125, .25, .375, .5])
        engine = self.weather_engine()
        client = create_app(engine).test_client()
        for mode in ("uncertainty", "baseline", "reduction"):
            result = client.get("/api/legend", query_string=dict(mode=mode))
            self.assertEqual(result.status_code, 200)
            self.assertEqual(len(result.json["values"]), 5)
            self.assertEqual(result.json["unit"], "°C")
            self.assertEqual(result.json["revision"], client.get("/api/state").json["revision"])
        linear = client.get("/api/legend", query_string=dict(mode="uncertainty", legend_scale="linear"))
        self.assertEqual(linear.status_code, 200)
        values = linear.json["values"]
        self.assertAlmostEqual(values[2], (values[0] + values[-1]) / 2)
        linear_reduction = client.get("/api/legend?mode=reduction&legend_scale=linear")
        self.assertEqual(linear_reduction.status_code, 200)
        engine.placement_layer = Mock(return_value=dict(self.layer, unit="°C·km²"))
        result = client.get("/api/legend?mode=placement").json
        np.testing.assert_allclose(result["values"], legend_values("placement", 0, 400))
        self.assertEqual(result["unit"], "°C·km²")
        linear_placement = client.get("/api/legend?mode=placement&legend_scale=linear").json
        self.assertAlmostEqual(linear_placement["values"][2],
                               (linear_placement["values"][0] + linear_placement["values"][-1]) / 2)
        self.assertEqual(client.get("/api/legend?mode=unknown").status_code, 400)

    def test_legend_label_counts_preserve_endpoints_and_logarithmic_spacing(self):
        client = create_app(self.weather_engine()).test_client()
        default = client.get("/api/legend").json["values"]
        for count in (2, 3, 4, 5):
            response = client.get("/api/legend", query_string=dict(count=count))
            self.assertEqual(response.status_code, 200)
            values = response.json["values"]
            self.assertEqual(len(values), count)
            self.assertEqual([values[0], values[-1]], [default[0], default[-1]])
            steps = np.diff(np.log(values))
            np.testing.assert_allclose(steps, steps[0])
        self.assertEqual(legend_values("terrain:tpi_1km", -2, 2, count=2), [-2, 2])
        for invalid in ("1", "6", "2.5", "bad", ""):
            self.assertEqual(client.get("/api/legend", query_string=dict(count=invalid)).status_code, 400)

    def test_numeric_weather_cache_survives_restart_separately_from_png(self):
        from station_weather_maps import WeatherMapCache

        engine = self.weather_engine()
        profile = engine.weather.profile("temperature:continuous")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "weather.npz"
            np.savez(source, baseline=profile["baseline"])
            profile["cache_path"] = str(source)
            cache = WeatherMapCache(path / "maps", renderer_version="test")
            limits = (.01, 6, 2)
            data = cache.get(engine, profile, 400, limits, build=True, numeric=True)
            self.assertEqual(len(list((path / "maps").glob("*.bin.gz"))), 1)
            png = cache.get(engine, profile, 400, limits, build=True)
            self.assertEqual(len(list((path / "maps").glob("*.png"))), 1)
            values, pixels = self.decode_raster(SimpleNamespace(
                status_code=200, data=data, headers={"Content-Encoding": "gzip"}))
            with Image.open(BytesIO(png)) as image:
                np.testing.assert_array_equal(pixels, np.asarray(image))
            self.assertTrue(np.isfinite(values[pixels[:, :, 3] > 0]).all())
            restarted = WeatherMapCache(path / "maps", renderer_version="test")
            with patch.object(engine, "_render_pixels", side_effect=AssertionError("Must use persisted raster")):
                self.assertEqual(restarted.get(engine, profile, 400, limits, build=True, numeric=True), data)

    def test_engine_warms_only_available_networks_and_shares_cache(self):
        engine = SimpleNamespace(root=SimpleNamespace(), placement_layer=Mock(return_value=self.layer),
                                 at_resolution=Mock(return_value=self.active))
        with tempfile.TemporaryDirectory() as folder, patch("station_app.OUT", Path(folder)), \
                patch("station_placement_tiles.PlacementTiles") as factory:
            (Path(folder) / "placement_gain_500m_road200.npz").touch()
            (Path(folder) / "placement_gain_500m_road200_empty.npz").touch()
            factory.return_value.warm.return_value = 7
            StationEngine.warm_placement_maps(engine)
            self.assertIs(engine.root.placement_tiles, factory.return_value)
            self.assertEqual([call.kwargs["station_network"] for call in engine.placement_layer.call_args_list],
                             ["svv", "empty"])
            self.assertEqual(factory.return_value.warm.call_count, 2)

    def test_launchers_prepare_placement_before_creating_app(self):
        import station_app
        import host_station_app

        calls = []
        engine = Mock()
        engine.weather.available = [dict(id="temperature:senorge30")]
        engine.warm_weather_maps.side_effect = lambda **kwargs: calls.append("weather")
        engine.warm_placement_maps.side_effect = lambda: calls.append("placement")
        app = Mock(config={})
        with patch("sys.argv", ["station_app.py", "--cell-m", "1000"]), \
                patch("station_app.StationEngine", return_value=engine), \
                patch("station_app.create_app", side_effect=lambda *args, **kwargs: (calls.append("app"), app)[1]), \
                patch("station_app.KuldehullLayer"), patch("station_app.socket.socket") as socket_factory:
            socket_factory.return_value.__enter__.return_value.connect_ex.return_value = 1
            station_app.main()
        self.assertEqual(calls, ["weather", "placement", "app"])
        calls.clear()
        with patch("station_app.StationEngine", return_value=engine), \
                patch("station_app.create_app", side_effect=lambda *args, **kwargs: (calls.append("app"), app)[1]), \
                patch("station_app.KuldehullLayer"):
            host_station_app.create_application(host_station_app.parse_args([]))
        self.assertEqual(calls, ["weather", "placement", "app"])

    def test_tiles_match_full_source_render_and_cutoff_without_rebuilding_source(self):
        for z, x, y in ((self.z, self.x, self.y), (7, self.x // 8, self.y // 8), (0, 0, 0)):
            for cutoff in (0, 150):
                with self.subTest(z=z, cutoff=cutoff):
                    image, hit = self.cache.image(self.active, self.layer, self.version, z, x, y,
                        cutoff, None, None, None, lambda image, bounds: image)
                    self.assertFalse(hit)
                    bounds = tile_bounds(z, x, y)
                    expected = self.active.render(self.layer["values"], "placement", 0, 400,
                        cell_mask=self.layer["values"] > cutoff if cutoff else None,
                        display_size=256, viewport=bounds)
                    with Image.open(BytesIO(image)) as actual, Image.open(expected) as reference:
                        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        source = self.cache.source
        _, hit = self.cache.image(self.active, self.layer, self.version, self.z, self.x, self.y,
                                  0, None, None, None, lambda image, bounds: image)
        self.assertTrue(hit)
        self.assertIs(self.cache.source, source)

    def test_tile_filters_and_cache_separate_road_distance_counties_and_source(self):
        roads = Mock(return_value=np.arange(400) % 2 == 0)
        clip = Mock(side_effect=lambda image, bounds: image)
        settings = dict(corridor=True, distance_m=500)
        args = (self.active, self.layer, self.version, self.z, self.x, self.y, 100)
        image, hit = self.cache.image(*args, settings, ("01",), roads, clip)
        self.assertFalse(hit)
        expected = self.active.render(self.layer["values"], "placement", 0, 400,
            cell_mask=(self.layer["values"] > 100) & roads.return_value, display_size=256,
            viewport=tile_bounds(self.z, self.x, self.y))
        with Image.open(BytesIO(image)) as actual, Image.open(expected) as reference:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        road_bounds = roads.call_args.args[1]["bounds"]
        self.assertGreaterEqual(road_bounds[0], self.active.bounds[0][1])
        self.assertLessEqual(road_bounds[2], self.active.bounds[1][1])
        self.assertTrue(self.cache.image(*args, settings, ("01",), roads, clip)[1])
        self.assertFalse(self.cache.image(*args, dict(settings, distance_m=1000), ("01",), roads, clip)[1])
        self.assertFalse(self.cache.image(*args, settings, ("02",), roads, clip)[1])
        changed = dict(self.layer, source_version=[10, 21])
        self.assertNotEqual(self.cache.version(self.active, changed), self.version)

    def test_adjacent_tiles_have_no_crop_seams_and_outside_tiles_are_transparent(self):
        mosaic = np.zeros((512, 512, 4), dtype=np.uint8)
        for dy in range(2):
            for dx in range(2):
                image, _ = self.cache.image(self.active, self.layer, self.version, self.z,
                    self.x + dx, self.y + dy, 0, None, None, None, lambda image, bounds: image)
                with Image.open(BytesIO(image)) as tile:
                    mosaic[dy*256:(dy+1)*256, dx*256:(dx+1)*256] = np.asarray(tile)
        top_left = tile_bounds(self.z, self.x, self.y)
        bottom_right = tile_bounds(self.z, self.x + 1, self.y + 1)
        bounds = [top_left[0], bottom_right[1], bottom_right[2], top_left[3]]
        with Image.open(self.active.render(self.layer["values"], "placement", 0, 400,
                                          display_size=512, viewport=bounds)) as expected:
            np.testing.assert_array_equal(mosaic, np.asarray(expected))
        image, _ = self.cache.image(self.active, self.layer, self.version, 10, 0, 0, 0,
                                     None, None, None, lambda image, bounds: image)
        with Image.open(BytesIO(image)) as tile:
            self.assertFalse(np.asarray(tile)[:, :, 3].any())

    def test_endpoint_is_versioned_cached_and_independent_of_scenario_edits(self):
        engine = FakeWeatherEngine()
        engine.placement_layer = Mock(return_value=self.layer)
        engine.at_resolution = Mock(return_value=self.active)
        client = create_app(engine).test_client()
        metadata = client.get("/api/placement").json
        query = dict(dataset=self.layer["dataset"], network="svv", version=metadata["tile_version"])
        url = f"/api/placement/tiles/{self.z}/{self.x}/{self.y}.png"
        with patch.object(engine, "prediction", side_effect=AssertionError("Must not predict")):
            start = time.perf_counter()
            first = client.get(url, query_string=query)
            cold = time.perf_counter() - start
            start = time.perf_counter()
            second = client.get(url, query_string=query)
            warm = time.perf_counter() - start
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.headers["X-Map-Cache"], "HIT")
        self.assertEqual(first.data, second.data)
        self.assertIn("private", first.headers["Cache-Control"])
        print(f"Placement tile fixture: cold={cold*1000:.1f} ms, cached={warm*1000:.1f} ms")
        self.assertEqual(client.get(url, query_string=dict(query, version="old")).status_code, 409)
        self.assertEqual(client.get(url, query_string=dict(query, rmse_min="nan")).status_code, 400)
        self.assertEqual(client.get("/api/placement/tiles/19/0/0.png", query_string=query).status_code, 400)
        with patch.object(CountyMask, "load", return_value=county_frame()):
            empty = client.get(url, query_string=dict(query, counties=json.dumps([])))
            self.assertEqual(empty.status_code, 200)
            with Image.open(BytesIO(empty.data)) as image:
                self.assertFalse(np.asarray(image)[:, :, 3].any())

    def test_endpoint_road_filter_and_network_have_distinct_tiles(self):
        engine = FakeWeatherEngine()
        layers = {"svv": self.layer, "empty": dict(self.layer, source="empty.npz",
                  station_network="empty", values=self.layer["values"] / 2)}
        engine.placement_layer = lambda dataset, station_network: layers[station_network]
        engine.at_resolution = Mock(return_value=self.active)
        roads = Mock()
        roads.corridor_mask.return_value = np.arange(400) % 2 == 0
        client = create_app(engine, road_network=roads).test_client()
        query = dict(dataset=self.layer["dataset"], network="svv", version=self.version,
                     roads=json.dumps(dict(corridor=True, distance_m=500)), rmse_min=100)
        url = f"/api/placement/tiles/{self.z}/{self.x}/{self.y}.png"
        image = client.get(url, query_string=query)
        self.assertEqual(image.status_code, 200)
        expected = self.active.render(self.layer["values"], "placement", 0, 400,
            cell_mask=(self.layer["values"] > 100) & roads.corridor_mask.return_value,
            display_size=256, viewport=tile_bounds(self.z, self.x, self.y))
        with Image.open(BytesIO(image.data)) as actual, Image.open(expected) as reference:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        invalid = dict(query, roads=json.dumps(dict(corridor=True, distance_m=-1)))
        self.assertEqual(client.get(url, query_string=invalid).status_code, 400)
        disabled = client.get(url, query_string=dict(query, roads=json.dumps(dict(corridor=False))))
        self.assertEqual(disabled.status_code, 200)
        expected = self.active.render(self.layer["values"], "placement", 0, 400,
            cell_mask=self.layer["values"] > 100, display_size=256,
            viewport=tile_bounds(self.z, self.x, self.y))
        with Image.open(BytesIO(disabled.data)) as actual, Image.open(expected) as reference:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        other = client.get(url, query_string=dict(query, network="empty",
                           version=self.cache.version(self.active, layers["empty"])))
        self.assertEqual(other.status_code, 200)
        self.assertEqual(other.headers["X-Map-Cache"], "MISS")
        self.assertNotEqual(other.data, image.data)

    def test_coastal_tiles_preserve_full_renderer_pixels(self):
        from shapely.geometry import Polygon

        bounds = Transformer.from_crs(25833, 3857, always_xy=True).transform_bounds(
            self.active.grid.x.min(), self.active.grid.y.min(),
            self.active.grid.x.max(), self.active.grid.y.max())
        west, south, east, north = bounds
        self.active.norway_path = Path(__file__)
        self.active._norway_shapes = [Polygon([(west, south), (east, north), (west, north)])]
        image, _ = self.cache.image(self.active, self.layer, self.version, self.z, self.x, self.y,
                                   0, None, None, None, lambda image, bounds: image)
        expected = self.active.render(self.layer["values"], "placement", 0, 400, display_size=256,
                                       viewport=tile_bounds(self.z, self.x, self.y))
        with Image.open(BytesIO(image)) as actual, Image.open(expected) as reference:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))


if __name__ == "__main__":
    unittest.main()
