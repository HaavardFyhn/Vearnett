import base64
import gzip
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from test_station_app import FakeWeather, FakeWeatherEngine
from station_app import PHENOMENON_DATASETS, create_app
from station_snapshot import basin_chunks, coverage_chunks


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

    def test_six_weather_maps_preserve_active_scenario_and_apply_sensor_edits(self):
        engine = FullEngine()
        client = create_app(engine).test_client()
        client.post("/api/action", json=dict(action="remove_sensor", station_id="SN2", sensor="temperature"))
        before = client.get("/api/scenario").json
        revision = client.get("/api/state").json["revision"]
        rendered, render_limits = [], []
        original_render = engine.render
        original_profile = engine.weather.profile

        def capture(values, *args, **kwargs):
            rendered.append(values.copy())
            render_limits.append(args[1:4])
            return original_render(values, *args, **kwargs)

        def distinct_limits(dataset):
            profile = original_profile(dataset)
            if profile["phenomenon"] == "temperature":
                profile.update(minimum=0.2, frost_maximum=5.)
            elif profile["phenomenon"] == "road_temperature":
                profile.update(minimum=0.4, frost_maximum=9.)
            return profile

        network = SimpleNamespace(metadata={"available": True}, geographic_bounds=[5, 58, 30, 71], filters=lambda value: None)
        vectors = dict(chunks=[], count=0)
        with patch("station_app.RoadNetwork", return_value=network), \
                patch("station_snapshot.road_chunks", return_value=vectors), \
                patch("station_snapshot.basin_chunks", return_value=vectors), \
                patch.object(engine.weather, "profile", side_effect=distinct_limits), \
                patch.object(engine, "render", side_effect=capture):
            with patch("station_snapshot.coverage_chunks", return_value=vectors) as coverage:
                result = client.post("/api/snapshot-data", json=dict(
                    revision=revision, roads={}, depth=20, coverage_classes=[5]))
                coverage.assert_called_once()
                self.assertEqual(coverage.call_args.args[1], [5])
                self.assertEqual(result.json["coverage"], vectors)
            self.assertEqual(result.status_code, 200, result.json)
            self.assertEqual(len(result.json["weather"]), 6)
            self.assertEqual({item["id"] for item in result.json["weather"]}, set(PHENOMENON_DATASETS))
            maps = {item["id"]: item for item in result.json["weather"]}
            self.assertEqual((maps["temperature"]["minimum"], maps["temperature"]["maximum"]), (0.2, 5.))
            self.assertEqual((maps["road_temperature"]["minimum"], maps["road_temperature"]["maximum"]), (0.2, 5.))
            self.assertEqual(render_limits[0], (0.2, 5., 2))
            self.assertEqual(render_limits[-2], render_limits[0])
            self.assertTrue(all(base64.b64decode(item["url"].split(",")[1]).startswith(b"\x89PNG")
                                for item in result.json["weather"]))
            self.assertTrue(all(base64.b64decode(item["baseline"].split(",")[1]).startswith(b"\x89PNG")
                                for item in result.json["weather"]))
            np.testing.assert_array_equal(rendered[0], engine.weather.profile(PHENOMENON_DATASETS["temperature"])["second"])
            np.testing.assert_array_equal(rendered[1], engine.weather.profile(PHENOMENON_DATASETS["temperature"])["baseline"])
            self.assertEqual(len(rendered), 12)
            self.assertEqual(client.post("/api/snapshot-data", json=dict(revision=-1, roads={})).status_code, 400)
            self.assertEqual(client.post("/api/snapshot-data", json=dict(revision=revision, roads={}),
                headers={"Origin": "https://example.org"}).status_code, 403)
        self.assertEqual(client.get("/api/scenario").json, before)

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
