import json
from pathlib import Path
from unittest.mock import patch
import time
import unittest

from flask import Flask
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.io import MemoryFile

from national_temperature_features import temperature_features
from national_temperature_uncertainty import forest_predict
from station_temperature_map import TemperatureMap, map_geometry, register_temperature_map
from valley_pilot import HOURS, REGIME_FIELDS


ROOT = Path(__file__).resolve().parent


class GeometryTests(unittest.TestCase):
    def test_exact_64km_window_sea_and_disconnected_land(self):
        origin = np.array([200000., 6800000.])
        height = np.zeros((80, 80))
        valid = np.ones_like(height, dtype=bool)
        valid[:, 42] = False
        rows, cols = np.indices(height.shape)
        table = np.zeros((height.size, 16))
        table[:, :2] = np.column_stack([origin[0]+cols.ravel()*1000, origin[1]-rows.ravel()*1000])
        selected, positions, graph, corner = map_geometry(
            table, np.array([[40, 40]]), height, valid, origin, 0)
        self.assertTrue((positions >= 0).all())
        self.assertTrue((positions < 64).all())
        self.assertEqual(len(selected), 64*34)
        self.assertTrue((cols.ravel()[selected] < 42).all())
        np.testing.assert_array_equal(graph[:, 0], 1)
        np.testing.assert_array_equal(corner, origin+[7500, -7500])
        valid[40, 40] = False
        with self.assertRaisesRegex(ValueError, "source"):
            map_geometry(table, np.array([[40, 40]]), height, valid, origin, 0)

    def test_geotiff_and_geojson_have_identical_georeferencing_and_nodata(self):
        model = TemperatureMap.__new__(TemperatureMap)
        model.to_lonlat = Transformer.from_crs(25833, 4326, always_xy=True)
        raster = np.full((64, 64), np.nan, np.float32)
        raster[2, 3] = -10
        corner = np.array([200500., 6800500.])
        metadata = dict(uncertainty_validated=False)
        geo = model.geojson(raster, corner, metadata)
        self.assertEqual(len(geo["features"]), 1)
        ring = geo["features"][0]["geometry"]["coordinates"][0]
        x, y = Transformer.from_crs(4326, 25833, always_xy=True).transform(*np.array(ring).T)
        np.testing.assert_allclose(x, [203500, 204500, 204500, 203500, 203500], atol=.001)
        np.testing.assert_allclose(y, [6798500, 6798500, 6797500, 6797500, 6798500], atol=.001)
        with MemoryFile(model.geotiff(raster, corner, metadata)) as memory, memory.open() as dataset:
            self.assertEqual(dataset.crs.to_epsg(), 25833)
            self.assertEqual(dataset.res, (1000., 1000.))
            self.assertEqual(dataset.bounds.right-dataset.bounds.left, 64000)
            self.assertEqual(dataset.bounds.top-dataset.bounds.bottom, 64000)
            self.assertEqual(dataset.read(1)[2, 3], -10)
            self.assertEqual(dataset.read(1, masked=True).count(), 1)
            self.assertEqual(dataset.tags()["uncertainty_validated"], "False")
        bands = np.stack([raster, np.where(np.isfinite(raster), 5., np.nan), raster-5, raster+5]).astype(np.float32)
        geo = model.geojson(bands, corner, metadata)
        properties = geo["features"][0]["properties"]
        self.assertEqual(properties["interval_radius_c"], 5)
        self.assertEqual(properties["lower_c"], -15)
        self.assertEqual(properties["upper_c"], -5)
        with MemoryFile(model.geotiff(bands, corner, metadata)) as memory, memory.open() as dataset:
            self.assertEqual(dataset.count, 4)
            np.testing.assert_array_equal(dataset.read()[:, 2, 3], [-10, 5, -15, -5])
            np.testing.assert_array_equal(dataset.read(masked=True).mask[0], dataset.read(masked=True).mask[1])
            self.assertIn("radius", dataset.descriptions[1])

    def test_portable_forest_float32_threshold_and_invalid_input(self):
        arrays = dict(left=np.array([1, -1, -1]), right=np.array([2, -1, -1]),
                      feature=np.array([0, -2, -2]), threshold=np.array([1., -2, -2]),
                      value=np.array([0., 3., 7.]), offsets=np.array([0, 3]))
        x = np.zeros((3, 40))
        x[:, 0] = [0, 1, 1.1]
        np.testing.assert_array_equal(forest_predict(arrays, x), [3, 3, 7])
        with self.assertRaises(ValueError):
            forest_predict(arrays, np.zeros((3, 39)))
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            forest_predict(arrays, x)

    def test_background_preparation_failure_is_explicit_and_retryable(self):
        class Backend:
            contexts = {}

            def validate(self, source, day):
                if source != "SN1":
                    raise ValueError("Unknown station")

            def prepare(self, source, day, progress):
                raise RuntimeError("MEPS unavailable")

        app = Flask(__name__)
        register_temperature_map(app, ROOT, Backend())
        client = app.test_client()
        self.assertEqual(client.post("/api/temperature-map/prepare?source=bad&day=2020-12-01").status_code, 400)
        with self.assertLogs(app.logger, level="ERROR"):
            self.assertEqual(client.post("/api/temperature-map/prepare?source=SN1&day=2020-12-01").status_code, 202)
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                status = client.get("/api/temperature-map/status").json
                if status["state"] == "failed":
                    break
                time.sleep(.01)
        self.assertEqual(status["state"], "failed")
        self.assertIn("MEPS unavailable", status["error"])
        with client.get("/temperature-map") as response:
            self.assertEqual(response.status_code, 200)
        with client.get("/api/temperature-map/grid") as response:
            self.assertEqual(response.status_code, 400)


class RealFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TemperatureMap(ROOT)

    def test_all_39_features_match_frozen_national_samples_and_saved_predictions(self):
        model = self.model
        day = "2024-12-15"
        forecast = json.loads((model.folder / "regimes" / f"{day}.json").read_text())
        ids = forecast["station_ids"]
        # Include integer input coordinates to exercise safe bearing division.
        table = model.station_table[[model.stations.index[model.stations.station_id.eq(s)][0] for s in ids]]
        fields = np.array([forecast["fields"][field] for field in REGIME_FIELDS]).transpose(1, 2, 0)
        xy = np.asarray(forecast["projection_xy"])
        result = ROOT / "outputs" / "national_transfer_returned" / "national_transfer_results"
        returned = pd.read_csv(result / "selected_pair_test.csv")
        with np.load(model.folder / "samples" / f"{day}.npz", allow_pickle=False) as data:
            for i in np.linspace(0, len(data["y"])-1, 20, dtype=int):
                source, target = str(data["source"][i]), str(data["target"][i])
                stamp = pd.Timestamp(str(data["time"][i]))
                x, delta = temperature_features(
                    table, ids.index(source), [ids.index(target)], data["x"][i:i+1, 28:33],
                    float(data["source_c"][i]), stamp, fields[HOURS.index(stamp.hour)], xy)
                np.testing.assert_allclose(x[0], data["x"][i], atol=1e-5)
                np.testing.assert_allclose(delta[0], data["forecast_delta_c"][i], atol=1e-5)
                with model.torch.inference_mode():
                    correction = float(model.model(model.torch.from_numpy((x-model.mean)/model.std))[0])
                expected = returned.loc[(returned.source == source) & (returned.target == target) &
                                        (returned.time == stamp.isoformat()), "prediction_c"].iloc[0]
                self.assertAlmostEqual(float(data["source_c"][i])+float(delta[0])+correction, expected, places=4)

    def test_observations_not_limited_to_paired_holdout_examples(self):
        model = self.model
        self.assertEqual(model.observation("SN55770", "2024-12-15T03:00:00+00:00"), 2.1)
        with np.load(model.folder / "samples" / "2024-12-15.npz", allow_pickle=False) as data:
            self.assertFalse((data["source"] == "SN55770").any())
        self.assertNotIn("2022-01-16", model.days)
        with self.assertRaises(ValueError):
            model.validate("SN55770", "2022-01-16")
        self.assertEqual(len(model.options()["stations"]), 1342)
        self.assertTrue(model.options()["uncertainty"]["available"])
        nontrain = model.catalogue.loc[model.catalogue.group.ne("train")].iloc[0]
        model.validate(nontrain.station_id, "2024-12-15")
        unavailable = model.catalogue.loc[model.catalogue.source_id.isna()].iloc[0]
        self.assertIsNone(model.observation(unavailable.station_id, "2024-12-15T03:00:00+00:00"))

    def test_portable_scales_match_returned_intervals(self):
        model = self.model
        returned = pd.read_csv(ROOT / "outputs" / "national_uncertainty_returned" /
                               "national_uncertainty_results" / "test_intervals.csv")
        day = "2024-12-15"
        with np.load(model.folder / "samples" / f"{day}.npz", allow_pickle=False) as data:
            saved = returned.loc[returned.time.str.startswith(day)]
            radii = model.uncertainty.radii(data["x"], saved.prediction_c.to_numpy())
            np.testing.assert_allclose(radii, saved.radius_c, atol=1e-5)

    def test_real_local_graph_matches_training_barriers(self):
        model = self.model
        source = model.stations.index[model.stations.station_id.eq("SN55770")][0]
        selected, positions, barriers, corner = map_geometry(
            model.table, model.positions, model.height, model.valid, model.origin, source)
        active = model.stations.loc[model.stations.group.isin(["train", "validation", "test"])].reset_index(drop=True)
        active_source = active.index[active.station_id.eq("SN55770")][0]
        with np.load(model.folder / "pairs.npz", allow_pickle=False) as pairs:
            count = 0
            for (a, b), expected in zip(pairs["pairs"], pairs["graph"]):
                if a != active_source:
                    continue
                reference = int(active.loc[b, "reference"])
                match = np.flatnonzero(selected == reference)
                if len(match):
                    np.testing.assert_allclose(barriers[match[0]], expected, atol=1e-6)
                    count += 1
        self.assertGreater(count, 0)

    def test_slider_recomputes_source_residual_and_missing_observation_never_defaults(self):
        import torch
        real = self.model
        model = TemperatureMap.__new__(TemperatureMap)
        model.catalogue, model.days = real.catalogue, real.days
        model.torch, model.mean, model.std = torch, np.zeros(39, np.float32), np.ones(39, np.float32)
        model.selected = "test"
        table = np.zeros((2, 16))
        table[:, :2] = [[200000, 6800000], [201000, 6800000]]
        fields = np.tile([[-5, 1, 0, .2], [-15, 1, 0, .2]], (3, 1, 1))
        model.contexts = {("SN55770", "2024-12-15"): dict(
            table=table, graph=np.array([[1, .3, .1, 0, 1]]),
            projection_xy=table[:, :2].astype(int), fields=fields, positions=np.array([[1, 2]]),
            top_left=np.array([199500, 6800500]), url=np.array("archive"), run_time=np.array("issue"))}
        recorded = []

        class Record(torch.nn.Module):
            def forward(self, x):
                recorded.append(x.numpy().copy())
                return torch.zeros(len(x))

        model.model = Record()
        model.uncertainty = real.uncertainty
        with patch.object(model, "observation", return_value=None):
            with self.assertRaisesRegex(ValueError, "No QC"):
                model.raster("SN55770", "2024-12-15", 3, None)
            for source_c in (-20, 0, 20):
                raster, _, metadata = model.raster("SN55770", "2024-12-15", 3, source_c)
                self.assertEqual(raster[1, 2], source_c-10)
                self.assertEqual(np.isfinite(raster).sum(), 1)
                self.assertEqual(recorded[-1][0, 0], source_c)
                self.assertEqual(recorded[-1][0, 25], source_c+5)
                self.assertTrue(metadata["synthetic_source"])
                self.assertFalse(metadata["uncertainty_validated"])
            for invalid in (float("nan"), float("inf"), 21):
                with self.assertRaises(ValueError):
                    model.raster("SN55770", "2024-12-15", 3, invalid)
            with self.assertRaises(ValueError):
                model.raster("SN55770", "2024-12-15", 9, 0)
            bands, _, metadata = model.raster("SN55770", "2024-12-15", 3, 0, True)
            self.assertEqual(bands.shape, (4, 64, 64))
            self.assertGreater(bands[1, 1, 2], 0)
            self.assertAlmostEqual(bands[2, 1, 2], bands[0, 1, 2]-bands[1, 1, 2], places=5)
            self.assertAlmostEqual(bands[3, 1, 2], bands[0, 1, 2]+bands[1, 1, 2], places=5)
            self.assertFalse(metadata["uncertainty_validated"])
            model.uncertainty = None
            with self.assertRaisesRegex(ValueError, "unavailable"):
                model.raster("SN55770", "2024-12-15", 3, 0, True)


if __name__ == "__main__":
    unittest.main()
