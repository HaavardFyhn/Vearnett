from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from test_station_app import FakeWeatherEngine
from station_app import create_app
from station_rmse_export import export_rmse
import geopandas as gpd
import pyogrio
from shapely.geometry import box


class RMSEExportTests(unittest.TestCase):
    def read(self, payload):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rmse.gpkg"
            path.write_bytes(payload)
            self.assertEqual(set(pyogrio.list_layers(path)[:, 0]), {"rmse_norway", "rmse_visible"})
            return (pyogrio.read_dataframe(path, layer="rmse_norway"),
                    pyogrio.read_dataframe(path, layer="rmse_visible"))

    def test_native_polygons_threshold_roads_nodata_and_empty_layer(self):
        engine = FakeWeatherEngine()
        values = np.array([3, 4, np.inf])
        full, visible = self.read(export_rmse(engine, values, threshold=3).getvalue())
        np.testing.assert_array_equal(full.rmse, [3, 4])
        np.testing.assert_array_equal(visible.rmse, [4])
        self.assertEqual(full.crs.to_epsg(), 25833)
        np.testing.assert_array_equal(full.geometry.area, [1e6, 1e6])
        full, visible = self.read(export_rmse(engine, values, threshold=3,
            cell_mask=np.array([True, False, True])).getvalue())
        self.assertEqual(len(full), 2)
        self.assertTrue(visible.empty)

    def test_cold_basin_clip_retains_partial_cells_and_rmse(self):
        clip = gpd.GeoDataFrame(geometry=[box(-500, -500, 0, 500)], crs=25833)
        full, visible = self.read(export_rmse(FakeWeatherEngine(), np.array([3., 4., 5.]),
                                              clip=clip).getvalue())
        self.assertEqual(len(full), 3)
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible.rmse.iloc[0], 3)
        self.assertEqual(visible.geometry.area.iloc[0], 500000)

    def test_endpoint_exports_two_layers_and_current_filters_without_changing_scenario(self):
        roads = Mock()
        roads.corridor_mask.return_value = np.array([True, False, True])
        client = create_app(FakeWeatherEngine(), road_network=roads).test_client()
        before = client.get("/api/state").json
        response = client.post("/api/export.gpkg", json=dict(mode="uncertainty",
            revision=before["revision"], rmse_min=2.5,
            roads=dict(corridor=True, bounds=[5, 58, 12, 64])))
        self.assertEqual(response.status_code, 200)
        self.assertIn("norway_rmse.gpkg", response.headers["Content-Disposition"])
        full, visible = self.read(response.data)
        np.testing.assert_array_equal(full.rmse, [3, 4, 3])
        np.testing.assert_array_equal(visible.cell_id, [0, 2])
        self.assertEqual(client.get("/api/state").json["revision"], before["revision"])
        self.assertEqual(client.post("/api/export.gpkg", json=dict(
            revision=-1)).status_code, 400)
        self.assertEqual(client.post("/api/export.gpkg", json=dict(
            revision=before["revision"], mode="placement")).status_code, 400)

    def test_endpoint_applies_cold_basin_depth_and_only_mask(self):
        cold = Mock()
        cold.prepare.return_value = True
        cold.polygons = gpd.GeoDataFrame(dict(max_depth_m=[5., 20.]),
            geometry=[box(-500, -500, 0, 500), box(1500, -500, 2500, 500)], crs=25833)
        client = create_app(FakeWeatherEngine(), kuldehull_layer=cold).test_client()
        revision = client.get("/api/state").json["revision"]
        response = client.post("/api/export.gpkg", json=dict(revision=revision,
            kuldehull=dict(only=True, depth=10)))
        self.assertEqual(response.status_code, 200)
        full, visible = self.read(response.data)
        self.assertEqual(len(full), 3)
        np.testing.assert_array_equal(visible.cell_id, [2])


if __name__ == "__main__":
    unittest.main()
