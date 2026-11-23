from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from test_station_app import FakeWeatherEngine
from station_app import create_app, cutoff_distribution, default_dataset
from station_rmse_export import export_rmse
from station_weather import DIRECT_P95
import geopandas as gpd
import pyogrio
from shapely.geometry import box


class RMSEExportTests(unittest.TestCase):
    def test_p95_default_filter_and_export(self):
        engine = FakeWeatherEngine()
        original_weather = engine.weather
        engine.weather = Mock(wraps=original_weather)
        p95 = dict(original_weather.profile("temperature:continuous"),
                   id=DIRECT_P95, metric="P95", kind="direct_p95")
        engine.weather.available = [*original_weather.available, p95]
        engine.weather.profile.side_effect = lambda key: p95 if key == DIRECT_P95 else original_weather.profile(key)
        engine.weather.svv_profile.return_value = p95
        engine.weather.portable_manifest = None
        client = create_app(engine, initial_dataset=default_dataset(engine, "temperature")).test_client()
        state = client.get("/api/state").json
        self.assertEqual(state["metric"], "P95")
        for mode, scale in (("uncertainty", "log"), ("baseline", "linear")):
            legend = client.get(f"/api/legend?mode={mode}&count=2&legend_scale={scale}").json
            self.assertEqual(legend["values"], [0., 1., 3.])
            self.assertTrue(legend["fixed"])
        cutoff = client.post("/api/rmse-range", json={}).json
        self.assertEqual(cutoff["cutoffs"][50], 3.)
        self.assertAlmostEqual(cutoff["visible_percent"][50], 100/3)
        response = client.post("/api/export.gpkg", json=dict(revision=state["revision"], rmse_min=cutoff["cutoffs"][50]))
        self.assertEqual(response.status_code, 200)
        self.assertIn("norway_p95.gpkg", response.headers["Content-Disposition"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "p95.gpkg"
            path.write_bytes(response.data)
            self.assertEqual(set(pyogrio.list_layers(path)[:, 0]), {"p95_norway", "p95_visible"})
            np.testing.assert_array_equal(pyogrio.read_dataframe(path, layer="p95_visible").p95, [4.])

    def test_cutoff_counts_ties_and_only_finite_selected_cells(self):
        result = cutoff_distribution(np.array([0, 0, 2, np.inf, np.nan, 8]), [True]*5+[False])
        self.assertEqual(result["cells"], 3)
        self.assertEqual(result["cutoffs"][50], 0)
        self.assertAlmostEqual(result["visible_percent"][50], 100/3)

    def test_p95_raster_changes_from_blue_to_red_at_one_and_saturates_at_three(self):
        from affine import Affine
        from types import SimpleNamespace
        from unittest.mock import patch
        from station_app import StationEngine

        values = np.array([0., .9, 1., 1.1, 3., 4.], dtype=np.float32)
        grid = SimpleNamespace(web_height=1, web_width=len(values), web_transform=Affine.identity(),
                               minimum=0, maximum=3, grid=values, raster_shape=(1, len(values)),
                               rows=np.zeros(len(values), dtype=int), columns=np.arange(len(values)),
                               transform=Affine.identity())
        with patch("station_app.reproject", side_effect=lambda source, target, **kw: np.copyto(target, source)):
            pixels, numeric = StationEngine._render_pixels(grid, values, "uncertainty",
                                                           filter_missing=True, map_scale="direct_p95")
        rgb = pixels[0, :, :3]
        self.assertLess(rgb[1, 0], rgb[1, 2])
        np.testing.assert_array_equal(rgb[2], [247, 247, 247])
        self.assertGreater(rgb[3, 0], rgb[3, 2])
        np.testing.assert_array_equal(rgb[4], rgb[5])
        np.testing.assert_array_equal(numeric[0], values)

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
