import base64
import gzip
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from test_station_app import FakeWeather, FakeWeatherEngine
from station_app import PHENOMENON_DATASETS, Scenario, create_app
from station_snapshot import basin_chunks, coverage_chunks, weather_maps, placement_map


class FullWeather(FakeWeather):
    available = [dict(id=dataset, phenomenon=phenomenon, kind=dataset.split(":")[1],
                      label=phenomenon, metric="RMSE", unit="test")
                 for phenomenon, dataset in PHENOMENON_DATASETS.items()]
    available += [FakeWeather.available[0]]


class FullEngine(FakeWeatherEngine):
    weather = FullWeather()

    def sensor_catalog(self):
        return {"SN1": ["temperature"], "SN2": ["temperature"]}


class SnapshotTests(unittest.TestCase):
    def test_placement_export_uses_selected_network_full_extent_and_county_clip(self):
        from io import BytesIO
        engine = FullEngine()
        scenario = Scenario(engine)
        scenario.pross_only = True
        scenario.include_met = True
        active = SimpleNamespace(render=Mock(return_value=BytesIO(b"image")))
        layer = dict(values=np.array([1., 2., 3.]), cell_m=500, minimum=0, maximum=9,
                     bounds=[[58, 5], [71, 31]], station_network="svv-met")
        engine.placement_layer = Mock(return_value=layer)
        engine.at_resolution = Mock(return_value=active)
        clip = Mock(side_effect=lambda current, image, active: image)
        mask = np.array([True, False, True])
        road_mask = Mock(return_value=mask)
        settings = dict(bounds=[5, 58, 30, 71], distance_m=500, corridor=False)
        result = placement_map(scenario, clip, road_mask, settings)
        engine.placement_layer.assert_called_once_with(PHENOMENON_DATASETS["temperature"], station_network="svv-met")
        engine.at_resolution.assert_called_once_with(500)
        self.assertEqual(active.render.call_count, 2)
        self.assertEqual(active.render.call_args.args[1:], ("placement", 0, 9))
        np.testing.assert_array_equal(active.render.call_args.kwargs["cell_mask"], mask)
        self.assertEqual(active.render.call_args.kwargs["display_size"], 1600)
        self.assertEqual(road_mask.call_args.args[1], dict(settings, corridor=True))
        self.assertIs(road_mask.call_args.args[2], active)
        self.assertEqual(clip.call_count, 2)
        self.assertEqual(base64.b64decode(result["url"].split(",")[1]), b"image")
        self.assertEqual(base64.b64decode(result["roads_url"].split(",")[1]), b"image")
        self.assertEqual(result["station_network"], "svv-met")
        for invalid in (dict(layer, station_network="svv"), dict(layer, cell_m=5000)):
            engine.placement_layer.return_value = invalid
            with self.assertRaisesRegex(ValueError, "mangler for dette stasjonsnettet"):
                placement_map(scenario, clip, road_mask, settings)

    def test_black_road_export_uses_background_source_without_score_class_filters(self):
        score_network = SimpleNamespace(metadata=dict(available=True), filters=Mock(),
                                        geographic_bounds=[5, 58, 30, 71])
        overview = dict(type="FeatureCollection", features=[])
        black_network = SimpleNamespace(metadata=dict(available=True, classes=["0", "6", "7"]),
            geographic_bounds=[5, 58, 30, 71], filters=Mock(), view=Mock(return_value=overview))
        client = create_app(FullEngine(), road_network=score_network).test_client()
        revision = client.get("/api/state").json["revision"]
        vectors = dict(chunks=[], count=0)
        with patch("station_app.RoadNetwork", return_value=black_network), \
                patch("station_snapshot.weather_maps", return_value=[]) as weather, \
                patch("station_snapshot.road_chunks", return_value=vectors) as exported_roads, \
                patch("station_snapshot.basin_chunks", return_value=vectors), \
                patch("station_snapshot.placement_map", return_value=dict(url="placement")) as placement:
            response = client.post("/api/snapshot-data", json=dict(revision=revision, road_scores=False,
                roads=dict(categories=["R"], classes=["1"], reference="Rv3"),
                weather_roads=dict(categories=["E", "R", "F"], classes=["1", "2"]),
                include_placement=True))
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(response.json["road_overview"], overview)
            self.assertIs(exported_roads.call_args.args[0], black_network)
            self.assertEqual(exported_roads.call_args.args[1]["reference"], "R3")
            self.assertEqual(exported_roads.call_args.args[1]["classes"], ["0", "6", "7"])
            self.assertEqual(weather.call_args.args[2]["classes"], ["1", "2"])
            self.assertEqual(response.json["placement"]["url"], "placement")
            placement.assert_called_once()
            black_network.view.assert_called_once()
            score_response = client.post("/api/snapshot-data", json=dict(revision=revision, road_scores=True,
                roads_visible=True, roads=dict(categories=["R"], classes=["4", "5"], reference="Rv3"),
                weather_roads=dict(categories=["E", "R", "F"], classes=["1", "2"])))
            self.assertEqual(score_response.status_code, 200, score_response.json)
            self.assertEqual(score_response.json["road_overview"], overview)
            self.assertIs(exported_roads.call_args.args[0], score_network)
            self.assertEqual(black_network.view.call_count, 2)
            invalid = client.post("/api/snapshot-data", json=dict(
                revision=revision, road_scores="false", roads={}))
            self.assertEqual(invalid.status_code, 400)
            invalid = client.post("/api/snapshot-data", json=dict(
                revision=revision, road_scores=True, roads_visible="false", roads={}))
            self.assertEqual(invalid.status_code, 400)

    def test_coverage_exports_offscreen_cells_filters_categories_and_clips_counties(self):
        from rasterio.transform import from_origin
        from shapely import union_all
        values = np.array([[5, 0, 4], [0, 0, 5]], dtype="uint8")
        layer = SimpleNamespace(classes=lambda classes: classes,
            source=lambda: (values, from_origin(250000, 6600500, 250, 250), "EPSG:25833"))

        def unpack(result):
            features = [feature for chunk in result["chunks"]
                        for feature in json.loads(gzip.decompress(base64.b64decode(chunk["data"])))["features"]]
            return gpd.GeoDataFrame.from_features(features, crs=4326).to_crs(25833)

        full = unpack(coverage_chunks(layer, [5]))
        self.assertEqual(len(full), 2)
        self.assertEqual(set(full.value), {5})
        self.assertAlmostEqual(full.geometry.area.sum(), 2 * 250**2, delta=5)
        county = box(250100, 6600000, 250750, 6600500)
        clipped = unpack(coverage_chunks(layer, [5], county))
        expected = union_all([box(250000, 6600250, 250250, 6600500),
                              box(250500, 6600000, 250750, 6600250)]).intersection(county)
        self.assertLess(union_all(clipped.geometry).symmetric_difference(expected).area, 5)
        self.assertEqual(coverage_chunks(layer, [], county)["count"], 0)
        self.assertEqual(coverage_chunks(layer, [5], box(0, 0, 1, 1))["count"], 0)

    def test_moving_road_station_preserves_identity_and_status_after_selection(self):
        engine = FullEngine()
        original_profile = engine.weather.profile

        def road_profile(dataset):
            profile = original_profile(dataset)
            profile["stations"].loc[0, "name"] = "E6 Test station"
            return profile

        with patch.object(engine.weather, "profile", side_effect=road_profile):
            client = create_app(engine).test_client()
            with patch.object(engine, "snap", return_value=7), patch.object(engine, "station", return_value=dict(
                    id=7, x=7000, y=0, lon=13, lat=64, elevation=15)):
                moved = client.post("/api/action", json=dict(action="move_frost", station_id="SN1", lon=13, lat=64))
                self.assertEqual(moved.status_code, 200, moved.json)
                selected = client.post("/api/action", json=dict(action="select", kind="frost", station_id="SN1"))
                marker = next(item for item in selected.json["fixed_stations"] if item["station_id"] == "SN1")
                self.assertTrue(marker["svv"])
                self.assertEqual(marker["name"], "E6 Test station")
                self.assertEqual((marker["lon"], marker["lat"]), (13, 64))
                self.assertIn("original_position", marker)

    def test_temperature_map_preserves_active_scenario_and_applies_sensor_edits(self):
        engine = FullEngine()
        client = create_app(engine).test_client()
        client.post("/api/action", json=dict(action="remove_sensor", station_id="SN2", sensor="temperature"))
        before = client.get("/api/scenario").json
        revision = client.get("/api/state").json["revision"]
        rendered, render_limits, render_masks = [], [], []
        original_render = engine.render
        original_profile = engine.weather.profile

        def capture(values, *args, **kwargs):
            rendered.append(values.copy())
            render_limits.append(args[1:4])
            render_masks.append(kwargs.get("cell_mask"))
            return original_render(values, *args, **kwargs)

        def distinct_limits(dataset):
            profile = original_profile(dataset)
            if profile["phenomenon"] != "temperature":
                raise AssertionError("HTML export must not load other weather models")
            profile.update(minimum=0.2, frost_maximum=5.)
            return profile

        corridor_calls = []
        def corridor(engine, bounds, settings, distance):
            corridor_calls.append((settings, distance))
            return np.array([False, True, True])

        network = SimpleNamespace(metadata={"available": True}, geographic_bounds=[5, 58, 30, 71],
                                  filters=lambda value: None, corridor_mask=corridor)
        vectors = dict(chunks=[], count=0)
        with patch("station_app.RoadNetwork", return_value=network), \
                patch("station_snapshot.road_chunks", return_value=vectors) as exported_roads, \
                patch("station_snapshot.basin_chunks", return_value=vectors), \
                patch.object(engine.weather, "available", [item for item in FullWeather.available
                             if item["phenomenon"] == "temperature"]), \
                patch.object(engine.weather, "profile", side_effect=distinct_limits), \
                patch.object(engine, "render", side_effect=capture):
            with patch("station_snapshot.coverage_chunks", return_value=vectors) as coverage:
                result = client.post("/api/snapshot-data", json=dict(
                    revision=revision, roads=dict(categories=["E"], classes=["1"]),
                    weather_roads=dict(categories=["E", "R"], classes=["1", "2"],
                                       corridor=True, distance_m=2000),
                    depth=20, coverage_classes=[5]))
                coverage.assert_called_once()
                self.assertEqual(coverage.call_args.args[1], [5])
                self.assertEqual(result.json["coverage"], vectors)
            self.assertEqual(result.status_code, 200, result.json)
            self.assertEqual(len(result.json["weather"]), 2)
            self.assertEqual({item["id"] for item in result.json["weather"]}, {"counties", "roads"})
            maps = {item["id"]: item for item in result.json["weather"]}
            self.assertEqual((maps["roads"]["minimum"], maps["roads"]["maximum"]), (0., 3.))
            self.assertEqual(maps["roads"]["distance_m"], 2000)
            self.assertIsNone(maps["counties"]["distance_m"])
            self.assertEqual(render_limits[0], (0., 3., 2))
            self.assertEqual(exported_roads.call_args.args[1]["categories"], ["E"])
            self.assertTrue(corridor_calls)
            self.assertTrue(all(settings["categories"] == ["E", "R"] and distance == 2000
                                for settings, distance in corridor_calls))
            self.assertTrue(all(base64.b64decode(item["url"].split(",")[1]).startswith(b"\x89PNG")
                                for item in result.json["weather"]))
            self.assertTrue(all(base64.b64decode(item["baseline"].split(",")[1]).startswith(b"\x89PNG")
                                for item in result.json["weather"]))
            self.assertTrue(all(base64.b64decode(item["reduction"].split(",")[1]).startswith(b"\x89PNG")
                                for item in result.json["weather"]))
            np.testing.assert_array_equal(rendered[0], engine.weather.profile(PHENOMENON_DATASETS["temperature"])["second"])
            np.testing.assert_array_equal(rendered[1], engine.weather.profile(PHENOMENON_DATASETS["temperature"])["baseline"])
            np.testing.assert_array_equal(rendered[2], [-1, -1, -3])
            np.testing.assert_array_equal(rendered[5], [-1, -1, -3])
            self.assertEqual(len(rendered), 6)
            self.assertTrue(all(mask is None for mask in render_masks[:3]))
            for mask in render_masks[3:]:
                np.testing.assert_array_equal(mask, [False, True, True])
            self.assertEqual(client.post("/api/snapshot-data", json=dict(revision=-1, roads={})).status_code, 400)
            self.assertEqual(client.post("/api/snapshot-data", json=dict(revision=revision, roads={}),
                headers={"Origin": "https://example.org"}).status_code, 403)
        self.assertEqual(client.get("/api/scenario").json, before)

    def test_both_temperature_extents_respect_counties_when_corridor_is_hidden(self):
        engine = FullEngine()
        client = create_app(engine).test_client()
        masks = []
        original_render = engine.render

        def capture(values, *args, **kwargs):
            masks.append(kwargs["cell_mask"].copy())
            return original_render(values, *args, **kwargs)

        network = SimpleNamespace(metadata={"available": True}, geographic_bounds=[5, 58, 30, 71],
            filters=lambda settings: None,
            corridor_mask=lambda engine, bounds, settings, distance: np.array([False, True, True]))
        vectors = dict(chunks=[], count=0)
        with patch("station_counties.CountyMask.selection", side_effect=lambda codes:
                   None if codes is None else tuple(codes)), \
                patch("station_counties.CountyMask.cells", return_value=np.array([True, True, False])), \
                patch("station_counties.CountyMask.geometry", return_value=box(0, 0, 1, 1)), \
                patch("station_counties.CountyMask.clip_image", side_effect=lambda image, bounds, codes: image), \
                patch("station_app.RoadNetwork", return_value=network), \
                patch("station_snapshot.road_chunks", return_value=vectors), \
                patch("station_snapshot.basin_chunks", return_value=vectors), \
                patch.object(engine, "render", side_effect=capture):
            selected = client.post("/api/action", json=dict(action="counties", value=["03"]))
            self.assertEqual(selected.status_code, 200, selected.json)
            result = client.post("/api/snapshot-data", json=dict(
                revision=selected.json["revision"], roads=dict(categories=["E"], classes=["1"]),
                weather_roads=dict(categories=["E", "R"], classes=["1", "2"], corridor=False, distance_m=500),
                basins_only=True))
            self.assertEqual(result.status_code, 200, result.json)
            self.assertEqual(result.json["weather"][1]["distance_m"], 500)
        self.assertEqual(len(masks), 6)
        for mask in masks[:3]:
            np.testing.assert_array_equal(mask, [True, True, False])
        for mask in masks[3:]:
            np.testing.assert_array_equal(mask, [False, True, False])

    def test_reduction_includes_all_changes_regardless_of_selected_station_or_comparison(self):
        from io import BytesIO
        engine = FullEngine()
        for scope in ("selected", "baseline"):
            with self.subTest(scope=scope), patch.object(engine, "prediction",
                    return_value=np.array([2, 5, 2], dtype=np.float32)):
                scenario = Scenario(engine, dataset=PHENOMENON_DATASETS["temperature"])
                scenario.apply(dict(action="remove_sensor", station_id="SN2", sensor="temperature"))
                scenario.apply(dict(action="add", lon=5, lat=60))
                scenario.apply(dict(action="select", kind="frost", station_id="SN1"))
                scenario.apply(dict(action="comparison", value=scope))
                selected = scenario.selected.copy()
                rendered = []

                def draw(current, values, mode, settings, **kwargs):
                    rendered.append((mode, values.copy()))
                    return BytesIO(b"image")

                with patch.object(scenario, "change", side_effect=AssertionError(
                        "Export must not use the selected-station comparison")):
                    maps = weather_maps(scenario, draw, dict(distance_m=500))
                self.assertEqual(len(maps), 2)
                for mode, values in (rendered[2], rendered[5]):
                    self.assertEqual(mode, "change")
                    np.testing.assert_array_equal(values, [1, -1, 1])
                self.assertEqual(scenario.comparison_scope, scope)
                self.assertEqual(scenario.selected, selected)

    def test_empty_network_export_uses_fixed_reference_and_handles_no_stations(self):
        from io import BytesIO
        engine = FullEngine()
        scenario = Scenario(engine, dataset=PHENOMENON_DATASETS["temperature"])
        scenario.apply(dict(action="network", value="empty"))
        rendered = []
        def draw(current, values, mode, settings, **kwargs):
            rendered.append((mode, values))
            return BytesIO(b"image")

        maps = weather_maps(scenario, draw, dict(distance_m=500))
        self.assertEqual(maps[0]["empty_reference_rmse"], 1.)
        self.assertIsNone(rendered[0][1])
        self.assertIsNone(rendered[2][1])
        rendered.clear()
        with patch.object(engine, "prediction", return_value=np.array([.8, 1., 1.2], dtype=np.float32)):
            scenario.apply(dict(action="add", lon=5, lat=60))
            weather_maps(scenario, draw, dict(distance_m=500))
        np.testing.assert_allclose(rendered[2][1], [.2, 0, 0])

    def test_basin_chunks_preserve_detail_depth_and_county_filters(self):
        polygon = box(250000, 6600000, 250300, 6600300).buffer(100, quad_segs=32)
        frame = gpd.GeoDataFrame(dict(lake_id=[1, 2, 3], max_depth_m=[30., 5., 40.],
            mean_depth_m=[10., 2., 12.], area_m2=[90000.] * 3),
            geometry=[polygon, polygon, box(600000, 7600000, 600500, 7600500)], crs=25833).to_crs(3857)
        basins = SimpleNamespace(prepare=lambda: True, polygons=frame, depth=float)
        result = basin_chunks(basins, 20)
        self.assertEqual(result["count"], 2)
        selected = basin_chunks(basins, 20, box(249000, 6599000, 251000, 6601000))
        collection = json.loads(gzip.decompress(base64.b64decode(selected["chunks"][0]["data"])))
        self.assertEqual(selected["count"], 1)
        feature = collection["features"][0]
        self.assertEqual(feature["properties"]["lake_id"], 1)
        self.assertEqual(feature["properties"]["max_depth_m"], 30.)
        self.assertEqual(feature["geometry"]["type"], "Polygon")
        self.assertEqual(len(feature["geometry"]["coordinates"][0]), len(polygon.exterior.coords))
        restored = gpd.GeoDataFrame.from_features(collection, crs=4326).to_crs(25833).geometry.iloc[0]
        self.assertLess(restored.hausdorff_distance(polygon), .01)
        self.assertEqual(basin_chunks(basins, 100)["count"], 0)


if __name__ == "__main__":
    unittest.main()
