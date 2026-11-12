from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from senorge_long_term import LongTermNetwork, pooled_rmse
from station_senorge_gpu import SeNorgeGPUPairModel, load_profile
import test_senorge_network_map as fixtures
import test_station_senorge_gpu as app_fixtures


class LongTermTests(unittest.TestCase):
    def backend(self, folder):
        base = fixtures.SeNorgeNetworkMapTests().backend(folder)
        backend = LongTermNetwork.__new__(LongTermNetwork)
        backend.__dict__.update(base.__dict__)
        backend.identity = "long-term-test"
        backend.sensor_provenance = dict(element="air_temperature", stations=["A", "B", "OFF"])
        backend.pool_lock = threading.RLock()
        return backend

    def test_pool_squares_not_rmse_and_rank_after_averaging(self):
        scores = pooled_rmse([[1, 9], [6, 6]])
        np.testing.assert_allclose(scores, [np.sqrt(41), 6])
        self.assertEqual(int(np.argmin(scores)), 1)
        self.assertNotEqual(scores[0], 5)
        self.assertNotEqual(scores.min(), np.sqrt((1+36)/2))
        for bad in ([[1, np.nan]], [[-1, 2]], [[]]):
            with self.assertRaises(ValueError):
                pooled_rmse(bad)

    def test_all_dates_source_cache_identity_and_no_target_archive_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = self.backend(folder)
            calls = []
            def prediction(x, c):
                calls.append(x.copy())
                return np.zeros(len(x)), np.abs(c)
            backend.model.predict = prediction
            targets, rmse = backend.source_values(0, 0)
            np.testing.assert_array_equal(targets, [0, 1, 2])
            np.testing.assert_allclose(rmse, [0, np.sqrt(2.5), np.sqrt(2.5)])
            self.assertEqual(sum(len(x) for x in calls), 3*2)
            # Source temperatures [-2, -1] appear for every pair, never target values.
            np.testing.assert_array_equal(calls[0][:, 0], [-2, -1, -2, -1, -2, -1])
            with patch.object(backend.model, "predict", side_effect=AssertionError("Must resume")):
                np.testing.assert_array_equal(backend.source_values(0, 0)[1], rmse)
            receipt = next(backend.folder.glob("source_*.json"))
            import json
            self.assertEqual(json.loads(receipt.read_text())["dates"], 2)

    def test_network_best_second_full_period_then_removal_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = self.backend(folder)
            # Reference A has [1,9] errors; reference B [6,6]. B wins after pooling.
            def prediction(x, c):
                is_a = x[:, 7] == 100
                score = np.where(is_a, np.where(c == -2, 1., 9.), 6.)
                return np.zeros(len(x)), score
            backend.model.predict = prediction
            result = backend.prepare_long_term(0)
            np.testing.assert_array_equal(backend.context["winner"], [0, 1, 1, 1])
            np.testing.assert_allclose(backend.context["best"], [0, 0, 6, 6])
            np.testing.assert_allclose(backend.context["second"][:3], [6, np.sqrt(41), np.sqrt(41)])
            self.assertEqual(result["metadata"]["dates"], 2)
            self.assertNotIn("day", result["metadata"])
            with patch.object(backend.model, "predict", side_effect=AssertionError("Must restore")):
                self.assertEqual(backend.prepare_long_term(0), result)
            registry = app_fixtures.SeNorgeGPUStationTests().registry(backend)
            registry.models["senorge-gpu-long-term-backend"] = backend
            config = app_fixtures.SeNorgeGPUStationTests().config()
            config.pop("day")
            config["long_term"] = True
            profile = load_profile(registry, config)
            self.assertEqual(profile["start"], "2024-12-15")
            self.assertEqual(profile["end"], "2024-12-16")
            self.assertNotIn("Kartdato", profile["model_warning"])
            source = registry.target_cache
            adapter = profile["model"]
            values = adapter.predict_pairs(source, source, np.array([0, 1]), np.array([2, 2]))
            np.testing.assert_allclose(values, [np.sqrt(41), 6])
            self.assertGreaterEqual(values[0], values[1])

    def test_incomplete_dates_exclude_source_without_silent_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = self.backend(folder)
            backend.weather[0, 0, 1] = np.nan
            with self.assertRaises(ValueError):
                backend.source_values(0, 0)
            backend.prepare_long_term(0)
            self.assertEqual(backend.context["source_cell"][0], -1)
            self.assertIn("Incomplete", backend.context["reasons"][0])

    def test_pending_profile_does_not_compute_or_fall_back_to_single_day(self):
        with tempfile.TemporaryDirectory() as folder:
            backend = self.backend(folder)
            registry = app_fixtures.SeNorgeGPUStationTests().registry(backend)
            registry.models["senorge-gpu-long-term-backend"] = backend
            config = app_fixtures.SeNorgeGPUStationTests().config()
            config.pop("day")
            config["long_term"] = True
            with patch.object(backend, "prepare_long_term", side_effect=AssertionError("No inline compute")):
                profile = load_profile(registry, config)
            self.assertEqual(profile["status"]["state"], "missing")
            self.assertIsNone(profile["baseline"])
            self.assertEqual(len(profile["stations"]), 3)
            self.assertIn("x", profile["stations"])
            self.assertIn("3600", profile["model_warning"])

    def test_temperature_sensor_filter_reuses_original_observations_and_terrain(self):
        import pandas as pd
        from greedy_400_stations_pruned import TERRAIN
        from senorge_long_term import temperature_stations
        terrain = pd.DataFrame(dict(station_id=["TEMP", "NOTEMP", "NOFEATURE"],
            x=[1., 2., np.nan], y=3., elevation=4., coast_dist_fjord=5.,
            coast_dist_ocean=6., objtype="land"))
        for field in TERRAIN:
            terrain[field] = 0.
        observed = pd.DataFrame(dict(station_id=["TEMP", "NOFEATURE"],
            name=["Has sensor", "Missing terrain"], lat=61., lon=6.,
            first_time="2021-01-01", last_time="2025-01-01"))
        with patch("pyogrio.read_dataframe", return_value=terrain), \
             patch("station_weather.observation_stations", return_value=(observed, "2021", "2025")) as available, \
             patch("senorge_long_term.sha256", return_value="terrain-hash"):
            stations, provenance = temperature_stations(Path("unused"))
        self.assertEqual(stations.station_id.tolist(), ["TEMP"])
        self.assertEqual(available.call_args.args[2], "air_temperature")
        self.assertEqual(set(available.call_args.kwargs["station_ids"]), set(terrain.station_id))
        self.assertEqual(provenance["element"], "air_temperature")
        self.assertEqual(provenance["availability_start"], "2021")


if __name__ == "__main__":
    unittest.main()
