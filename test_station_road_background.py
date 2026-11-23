import station_app
from io import BytesIO
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import shapely
from PIL import Image
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point, box
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from station_road_background import RoadBackground, _raster_shapes


class RoadBackgroundTests(unittest.TestCase):
    def test_batched_rasterization_preserves_lines_parts_and_point_touches(self):
        lines = [LineString([(-5, 2), (5, 8), (15, 2)]),
                 MultiLineString([[(1, 1), (3, 9)], [(7, 1), (9, 9)]])]
        for geometries in (lines, lines + [Point(5, 5)]):
            with self.subTest(points=len(geometries) > 2):
                options = dict(out_shape=(64, 64), transform=from_bounds(0, 0, 10, 10, 64, 64),
                               dtype="uint8")
                expected = rasterize([(geometry, 1) for geometry in geometries], **options)
                actual = rasterize(_raster_shapes(np.array(geometries, dtype=object)), **options)
                np.testing.assert_array_equal(actual, expected)

    def test_tiles_filter_categories_cache_and_preserve_black_transparent_style(self):
        x, y = Transformer.from_crs(4326, 25833, always_xy=True).transform(10, 60)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roads.gpkg"
            frame = gpd.GeoDataFrame(dict(roadid=["{E6}", "{R3}", "{F51}", "{K1}", "{P1}",
                                                 "{S1}", None, '{"K1","f7"}']),
                geometry=[LineString([(x + offset * 100, y - 1000),
                                      (x + offset * 100, y + 1000)]) for offset in range(8)],
                crs=25833)
            frame.at[0, "geometry"] = LineString([(x - 300, y - 300), (x + 300, y + 300),
                                                  (x - 300, y + 300), (x + 300, y - 300)])
            frame.to_file(path, driver="GPKG")
            roads = RoadBackground(path)
            z = 12
            tx = int((10 + 180) / 360 * 2 ** z)
            ty = int((1 - math.asinh(math.tan(math.radians(60))) / math.pi) / 2 * 2 ** z)
            tile = roads.tile(z, tx, ty)
            self.assertEqual(len(roads.geometries), 4)
            with patch("station_road_background._raster_shapes",
                       side_effect=lambda geometries: [(geometry, 1) for geometry in geometries]), \
                    patch("station_road_background.shapely.clip_by_rect",
                          side_effect=lambda geometries, *bounds: shapely.intersection(
                              geometries, box(*bounds))):
                roads.tile.cache_clear()
                legacy = roads.tile(z, tx, ty)
            roads.tile.cache_clear()
            tile = roads.tile(z, tx, ty)
            self.assertEqual(tile, legacy)
            with Image.open(BytesIO(tile)) as image:
                pixels = np.asarray(image)
                self.assertEqual(image.size, (256, 256))
                self.assertTrue((pixels[:, :, :3] == 0).all())
                self.assertTrue((pixels[:, :, 3] > 0).any())
                self.assertTrue((pixels[:, :, 3] == 0).any())
            with patch.object(roads, "prepare", side_effect=AssertionError("Tile should be cached")):
                self.assertEqual(roads.tile(z, tx, ty), tile)
            with Image.open(BytesIO(roads.tile(12, 0, 0))) as empty:
                self.assertFalse(np.asarray(empty)[:, :, 3].any())
            for categories, reference, visible in (
                    (("E",), "", True), (("R",), "R3", True), (("F",), "F7", True),
                    (("E",), "F51", False), (("E", "R", "F"), "E600", False),
                    ((), "", False)):
                with self.subTest(categories=categories, reference=reference):
                    with Image.open(BytesIO(roads.tile(z, tx, ty, categories, reference))) as image:
                        self.assertEqual(bool(np.asarray(image)[:, :, 3].any()), visible)
            roads.counties = SimpleNamespace(geometry=lambda codes: (
                box(x - 100, y - 1000, x + 100, y) if codes == ("01",)
                else box(x + 10000, y, x + 11000, y + 1000)))
            with Image.open(BytesIO(roads.tile(z, tx, ty, ("E",), "", ("01",)))) as clipped, \
                    Image.open(BytesIO(roads.tile(z, tx, ty, ("E",)))) as whole:
                alpha = np.asarray(clipped)[:, :, 3]
                self.assertTrue(alpha.any())
                self.assertLess(np.count_nonzero(alpha), np.count_nonzero(np.asarray(whole)[:, :, 3]))
            with Image.open(BytesIO(roads.tile(z, tx, ty, ("E",), "", ("02",)))) as outside:
                self.assertFalse(np.asarray(outside)[:, :, 3].any())
            for args in ((19, 0, 0), (3, 8, 0), (3, 0, 8), (-1, 0, 0)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    roads.tile(*args)

    def test_tile_endpoint_is_cacheable_and_does_not_load_a_scenario(self):
        from test_station_app import FakeEngine
        with patch("station_road_background.RoadBackground.tile", return_value=b"png") as tile:
            client = station_app.create_app(FakeEngine()).test_client()
            response = client.get("/api/roads/background/6/33/18.png")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "image/png")
            self.assertIn("max-age=3600", response.headers["Cache-Control"])
            self.assertNotIn("Set-Cookie", response.headers)
            tile.assert_called_once_with(6, 33, 18, ("E", "F", "R"), "", None)
            response = client.get('/api/roads/background/6/33/18.png?categories=["R"]&reference=Rv3')
            self.assertEqual(response.status_code, 200)
            tile.assert_called_with(6, 33, 18, ("R",), "R3", None)
            for query in ('categories=["K"]', 'categories=null', 'reference=invalid'):
                with self.subTest(query=query):
                    response = client.get("/api/roads/background/6/33/18.png?" + query)
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("error", response.json)
