from io import BytesIO
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
from PIL import Image
import pyogrio
from shapely.geometry import box

from station_kuldehull import KuldehullLayer, lake_settings


class KuldehullTests(unittest.TestCase):
    def test_persistent_preparation_reloads_without_source_reads_and_invalidates_on_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lakes.gpkg"
            polygons = gpd.GeoDataFrame(
                {"lake_id": [1], "max_depth_m": [12.], "volume_m3": [20000.]},
                geometry=[box(10, 60, 10.1, 60.1)], crs=4326,
            ).to_crs(25833)
            pyogrio.write_dataframe(polygons, path, layer="potential_lakes", driver="GPKG")
            first = KuldehullLayer(path)
            self.assertTrue(first.prepare())
            expected = first.mask((9.9, 59.9, 10.2, 60.2), 200, 150, 5)
            second = KuldehullLayer(path)
            with patch("station_kuldehull.pyogrio.read_dataframe", side_effect=AssertionError("Source read after offline preparation")):
                self.assertTrue(second.prepare())
                np.testing.assert_array_equal(second.mask((9.9, 59.9, 10.2, 60.2), 200, 150, 5), expected)
                self.assertEqual(second.point(10.05, 60.05)["volume_m3"], 20000.)
                with patch("station_kuldehull.rasterize", side_effect=AssertionError("Interactive overview polygon processing")):
                    overview = second.mask((9.9, 59.9, 10.2, 60.2), 3, 3, 5)
                    self.assertTrue(overview.any())
                    self.assertFalse(second.mask((9.9, 59.9, 10.2, 60.2), 3, 3, 12).any())
            before = path.stat()
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            with patch("station_kuldehull.pyogrio.read_dataframe", wraps=pyogrio.read_dataframe) as read:
                second.prepare()
                self.assertEqual(read.call_count, 1)
            self.assertEqual(len(list(first.cache_dir.glob("*/ready.json"))), 2)

    def test_depth_filter_and_clip_use_whole_basin_and_preserve_image_colours(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lakes.gpkg"
            polygons = gpd.GeoDataFrame(
                {"lake_id": [1, 2], "max_depth_m": [3., 12.]},
                geometry=[box(10, 60, 10.1, 60.1), box(10.2, 60, 10.3, 60.1)], crs=4326,
            ).to_crs(25833)
            pyogrio.write_dataframe(polygons, path, layer="potential_lakes", driver="GPKG")
            layer = KuldehullLayer(path)
            bounds = (9.9, 59.9, 10.4, 60.2)
            all_basins = layer.mask(bounds, 200, 150, 0)
            deep_basins = layer.mask(bounds, 200, 150, 3)
            self.assertGreater(all_basins.sum(), deep_basins.sum())
            self.assertGreater(deep_basins.sum(), 0)
            self.assertEqual(layer.mask(bounds, 200, 150, 12).sum(), 0)
            image = BytesIO()
            Image.new("RGBA", (200, 150), (100, 40, 20, 200)).save(image, format="PNG")
            clipped = layer.clip(image, SimpleNamespace(bounds=[[59.9, 9.9], [60.2, 10.4]]), 3)
            pixels = np.array(Image.open(clipped))
            np.testing.assert_array_equal(pixels[:, :, 3] > 0, deep_basins > 0)
            self.assertTrue((pixels[:, :, :3] == [100, 40, 20]).all())
            viewport = {"bounds": (10.15, 59.95, 10.35, 60.15), "width": 120, "height": 100}
            zoomed = layer.clip(image, SimpleNamespace(bounds=[[59.9, 9.9], [60.2, 10.4]]),
                                3, viewport=viewport)
            zoomed_pixels = np.array(Image.open(zoomed))
            self.assertEqual(zoomed_pixels.shape, (100, 120, 4))
            np.testing.assert_array_equal(
                zoomed_pixels[:, :, 3] > 0,
                layer.mask(viewport["bounds"], 120, 100, 3) > 0,
            )

    def test_invalid_depth_is_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                lake_settings({"only": True, "depth": value})

    def test_projected_geometry_is_reused_and_point_inspection_uses_original_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lakes.gpkg"
            polygons = gpd.GeoDataFrame(
                {"lake_id": [1], "max_depth_m": [12.], "mean_depth_m": [4.],
                 "area_m2": [5000.], "volume_m3": [20000.], "spill_level_m": [150.]},
                geometry=[box(10, 60, 10.1, 60.1)], crs=4326,
            ).to_crs(25833)
            pyogrio.write_dataframe(polygons, path, layer="potential_lakes", driver="GPKG")
            layer = KuldehullLayer(path)
            with patch("station_kuldehull.pyogrio.read_dataframe", wraps=pyogrio.read_dataframe) as read:
                layer.mask((9.9, 59.9, 10.2, 60.2), 200, 150, 0)
                layer.mask((9.95, 59.95, 10.15, 60.15), 200, 150, 5)
                self.assertEqual(read.call_count, 1)
            result = layer.point(10.05, 60.05, 5)
            self.assertTrue(result["found"])
            self.assertEqual(result["lake_id"], 1)
            self.assertEqual(result["volume_m3"], 20000.)
            self.assertEqual(result["spill_level_m"], 150.)
            self.assertFalse(layer.point(10.05, 60.05, 12)["found"])
            self.assertFalse(layer.point(10.15, 60.05)["found"])

    def test_map_and_preview_clipping_are_display_only_and_bypass_unfiltered_cache(self):
        from station_app import create_app
        from test_station_app import FakeWeatherEngine

        layer = Mock()
        layer.metadata = {"available": True, "label": "Kuldehull"}
        layer.clip.side_effect = lambda image, engine, depth: image
        with patch("station_app.KuldehullLayer", return_value=layer):
            client = create_app(FakeWeatherEngine()).test_client()
            before = client.get("/api/state").get_json()
            self.assertTrue(client.get("/api/kuldehull/options").json["available"])
            client.get("/api/map.png?mode=uncertainty&size=400")
            for depth in (3, 10):
                import json
                response = client.get("/api/map.png", query_string={
                    "mode": "uncertainty", "size": 400,
                    "kuldehull": json.dumps({"only": True, "depth": depth}),
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["X-Map-Cache"], "MISS")
                self.assertEqual(layer.clip.call_args.args[2], depth)
            preview = client.post("/api/preview", json={
                "lon": 5, "lat": 60, "mode": "affected",
                "kuldehull": {"only": True, "depth": 10},
            })
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(layer.clip.call_count, 3)
            after = client.get("/api/state").get_json()
            self.assertEqual(before["revision"], after["revision"])
            self.assertEqual(before["mean_rmse"], after["mean_rmse"])

    def test_point_api_returns_properties_without_changing_station_scenario(self):
        from station_app import create_app
        from test_station_app import FakeWeatherEngine

        layer = Mock()
        layer.point.return_value = {"found": True, "lake_id": 7, "max_depth_m": 20., "volume_m3": 30000.}
        with patch("station_app.KuldehullLayer", return_value=layer):
            client = create_app(FakeWeatherEngine()).test_client()
            before = client.get("/api/state").get_json()
            response = client.get("/api/kuldehull/point?lon=10&lat=60&depth=5")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["lake_id"], 7)
            self.assertEqual(response.json["volume_m3"], 30000.)
            layer.point.assert_called_once_with("10", "60", "5")
            after = client.get("/api/state").get_json()
            self.assertEqual(before["revision"], after["revision"])
            self.assertEqual(before["stations"], after["stations"])


if __name__ == "__main__":
    unittest.main()