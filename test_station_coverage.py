import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from station_app import create_app
from station_coverage import CoverageLayer, RELATIVE, save_status_raster, COLORS, STATUSES
from test_station_app import FakeWeatherEngine
import rasterio
from rasterio.transform import from_origin, xy
from rasterio.warp import transform, transform_bounds


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        folder = self.root / RELATIVE
        folder.mkdir(parents=True)
        self.layer = CoverageLayer(self.root)
        self.profile = dict(width=3, height=2, crs="EPSG:32633",
                            transform=from_origin(250000, 6650000, 250, 250))
        occurrences = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
        relevant = pd.DataFrame(dict(ForekomstID=[1, 2, 3, 4, 5],
                                     Kandidatklasse=[1, 2, 3, 4, 5]))
        save_status_raster(self.layer.path, occurrences, mask, relevant, self.profile)
        (folder / "manifest.json").write_text(json.dumps(dict(created="2026-11-11", stations=849,
            owners=["MET.NO", "STATENS VEGVESEN"])), encoding="utf-8")
        with closing(sqlite3.connect(folder / "road_climate_coverage.gpkg")) as db, db:
            db.execute("CREATE TABLE scored_roads (Kandidatklasse INTEGER, ForekomstID INTEGER, length_m REAL)")
            db.executemany("INSERT INTO scored_roads VALUES (?, ?, ?)",
                           [(i, i, 1000 * i) for i in range(1, 6)] + [(5, 5, 250)])

    def test_metadata_five_categories_and_distinct_occurrences(self):
        options = self.layer.metadata()
        self.assertTrue(options["available"])
        self.assertEqual(options["stations"], 849)
        self.assertEqual([row["label"] for row in options["classes"]], list(STATUSES))
        self.assertEqual(options["classes"][4]["occurrences"], 1)
        self.assertEqual(options["classes"][4]["lines"], 2)
        self.assertEqual(options["classes"][4]["length_km"], 5.25)

    def test_portable_only_needs_small_raster_and_summary(self):
        options = self.layer.metadata()
        (self.layer.folder / "status_summary.json").write_text(json.dumps(options), encoding="utf-8")
        (self.layer.folder / "manifest.json").unlink()
        (self.layer.folder / "road_climate_coverage.gpkg").unlink()
        self.layer.metadata.cache_clear()
        self.assertEqual(self.layer.metadata(), options)

    def test_map_and_click_use_labels_recorded_with_saved_analysis(self):
        path = self.layer.folder / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["classification"] = dict(thresholds=[10, 30, 50, 70],
                                          statuses=["A", "B", "C", "D", "E"])
        path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self.layer.metadata()["classes"][4]["label"], "E")
        x, y = xy(self.profile["transform"], 1, 1)
        lon, lat = transform(32633, 4326, [x], [y])
        self.assertEqual(self.layer.point(lon[0], lat[0], [5])["label"], "E")

    def test_raster_only_contains_relevant_road_cells(self):
        values, _, _ = self.layer.source()
        np.testing.assert_array_equal(values, [[1, 2, 3], [4, 5, 0]])
        self.assertEqual(values.dtype, np.uint8)

    def test_filtered_images_have_only_selected_colors(self):
        bounds = list(transform_bounds(32633, 4326, 250000, 6649500, 250750, 6650000))
        for classes in ([1, 2, 3, 4, 5], [5], []):
            image = Image.open(self.layer.image(bounds, 200, 200, classes))
            pixels = np.array(image)
            actual = set(map(tuple, pixels[pixels[:, :, 3] > 0]))
            expected = {tuple(bytes.fromhex(COLORS[value - 1][1:])) + (255,) for value in classes}
            self.assertEqual(actual, expected)

    def test_click_identifies_cell_and_honors_filter(self):
        x, y = xy(self.profile["transform"], 1, 1)
        lon, lat = transform(32633, 4326, [x], [y])
        result = self.layer.point(lon[0], lat[0], [5])
        self.assertEqual(result["label"], STATUSES[4])
        self.assertFalse(self.layer.point(lon[0], lat[0], [4])["found"])
        self.assertFalse(self.layer.point(0, 0, [5])["found"])

    def test_invalid_queries_and_missing_data_are_explicit(self):
        for classes in ([0], [6], [True], ["5"], None):
            with self.assertRaises(ValueError):
                self.layer.classes(classes)
        with self.assertRaises(ValueError):
            self.layer.image([0, 60, 10, 70], 2001, 100, [5])
        with self.assertRaises(ValueError):
            self.layer.image([0, 70, 10, 60], 200, 100, [5])
        missing = CoverageLayer(self.root / "missing")
        self.assertFalse(missing.metadata()["available"])
        with self.assertRaisesRegex(ValueError, "Ingen lagret"):
            missing.source()

    def test_api_uses_saved_layer_and_rejects_bad_queries(self):
        with patch("station_coverage.CoverageLayer", return_value=self.layer):
            client = create_app(FakeWeatherEngine()).test_client()
        self.assertEqual(client.get("/api/coverage/options").json["stations"], 849)
        bounds = list(transform_bounds(32633, 4326, 250000, 6649500, 250750, 6650000))
        response = client.get("/api/coverage.png", query_string=dict(
            bounds=json.dumps(bounds), classes="[5]", width=200, height=200))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        self.assertEqual(client.get("/api/coverage.png", query_string=dict(
            bounds=json.dumps(bounds), classes="[6]")).status_code, 400)
        response = client.get("/api/coverage/point", query_string=dict(lon=0, lat=0, classes="[]"))
        self.assertEqual(response.json, {"found": False})


if __name__ == "__main__":
    unittest.main()
