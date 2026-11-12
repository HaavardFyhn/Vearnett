from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from station_app import create_app
from station_climate import ClimateZones
from station_counties import CountyMask
from test_station_app import FakeWeatherEngine
from test_station_filters import county_frame


class ClimateTests(unittest.TestCase):
    def test_native_classes_cache_nodata_and_point(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "zones.tif"
            with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                               dtype="uint8", crs=4326, transform=from_origin(10, 62, 1, 1), nodata=0) as source:
                source.write(np.array([[1, 2], [0, 1]], dtype=np.uint8), 1)
                source.update_tags(sonenavn=json.dumps({"1": "First", "2": "Second"}), metode="KMeans",
                    sone_statistikk=json.dumps({"1": {"vintertemp": -6.53, "kystpavirkning": 0.178}}))
            climate = ClimateZones(path, Path(folder) / "cache")
            self.assertEqual(len(climate.metadata()["zones"]), 2)
            image = climate.image(400)
            pixels = np.asarray(Image.open(image))
            self.assertTrue((pixels[:, :, 3] == 0).any())
            actual = set(map(tuple, pixels[pixels[:, :, 3] > 0, :3]))
            expected = {tuple(bytes.fromhex(zone["color"][1:])) for zone in climate.metadata()["zones"]}
            self.assertEqual(actual, expected)
            self.assertEqual(climate.image(400).getvalue(), image.getvalue())
            self.assertEqual(climate.point(10.5, 61.5)["zone"]["value"], 1)
            self.assertEqual(climate.point(10.5, 61.5)["statistics"],
                             {"vintertemp": -6.53, "kystpavirkning": 0.178})
            self.assertEqual(climate.point(11.5, 61.5)["zone"]["value"], 2)
            self.assertIsNone(climate.point(10.5, 60.5)["zone"])
            self.assertIsNone(climate.point(0, 0)["zone"])
            with self.assertRaises(ValueError):
                climate.image(123)
            with self.assertRaises(ValueError):
                climate.point(float("nan"), 60)
            clipped = climate.clip(climate.image(400),
                gpd.GeoDataFrame(geometry=[box(10, 60, 11, 62)], crs=4326), 400)
            rgba = np.asarray(Image.open(clipped))
            self.assertTrue((rgba[:, rgba.shape[1]//2+2:, 3] == 0).all())

    def test_live_routes_counties_and_station_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            climate = ClimateZones(Path("data") / "klimasoner.tif", Path(folder))
            roads = Mock()
            roads.corridor_mask.return_value = np.zeros(3, dtype=bool)
            with patch("station_climate.ClimateZones", return_value=climate), \
                    patch.object(CountyMask, "load", return_value=county_frame()):
                client = create_app(FakeWeatherEngine(), road_network=roads).test_client()
                before = client.get("/api/state").json
                self.assertEqual(len(client.get("/api/climate/options").json["zones"]), 8)
                response = client.get("/api/climate.png?size=400")
                self.assertEqual(response.status_code, 200)
                self.assertGreater(np.asarray(Image.open(BytesIO(response.data)))[:, :, 3].sum(), 0)
                response = client.get("/api/climate.png", query_string=dict(size=400,
                    roads=json.dumps(dict(corridor=True, bounds=[5, 58, 12, 64]))))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(np.asarray(Image.open(BytesIO(response.data)))[:, :, 3].sum(), 0)
                roads.corridor_mask.assert_called_once()
                client.post("/api/action", json=dict(action="counties", value=[]))
                response = client.get("/api/climate.png?size=400")
                self.assertEqual(np.asarray(Image.open(BytesIO(response.data)))[:, :, 3].sum(), 0)
                self.assertEqual(client.get("/api/state").json["fixed_count"], before["fixed_count"])
                self.assertEqual(client.get("/api/climate.png?size=123").status_code, 400)


if __name__ == "__main__":
    unittest.main()
