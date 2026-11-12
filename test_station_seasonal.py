from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from greedy_400_stations_pruned import make_cache, predict
from remote_senorge.train_seasonal_terrain import create_model
from station_seasonal import SeasonalBackend, SeasonalPairModel, TARGET_LABELS, WARNING
from station_senorge import BasinLookup
from station_weather import paired_predictions, WeatherRegistry
from test_station_senorge import StationSeNorgeTests
from test_station_app import FakeWeather, FakeWeatherEngine
from station_app import create_app


class SeasonalTests(unittest.TestCase):
    def test_encoded_head_matches_direct_network_all_targets_routes_and_identity(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(42)
        backend = SeasonalBackend.__new__(SeasonalBackend)
        backend.torch, backend.lock = torch, threading.RLock()
        backend.model = create_model("combined", torch).eval()
        backend.mean, backend.std = np.zeros(25, np.float32), np.full(25, 100000, np.float32)
        backend.metadata = dict(targets=list(TARGET_LABELS))
        backend.embeddings = {}
        backend.cells = {(1, 1): 0, (2, 1): 1, (3, 1): 2}
        patches = np.random.default_rng(42).normal(size=(3, 3, 2, 64, 64)).astype(np.float32)
        patches[:, :, 1] = 1
        with tempfile.TemporaryDirectory() as temp:
            frame = StationSeNorgeTests().frame()
            path = Path(temp) / "grid.csv.gz"
            frame.to_csv(path, index=False)
            basins = BasinLookup(path, Path(temp) / "missing.gpkg")
            cache = make_cache(frame)
            first, second = np.array([0, 1, 0]), np.array([1, 2, 0])
            with patch.object(backend, "terrain_input",
                              side_effect=lambda keys: patches[[int(k[0])-1 for k in keys]]) as load:
                for target in TARGET_LABELS:
                    model = SeasonalPairModel(backend, basins, target)
                    x, identical, coords = model.pair_inputs(cache, cache, first, second)
                    with torch.inference_mode():
                        expected = backend.model(torch.from_numpy(patches[first]),
                                                 torch.from_numpy(patches[second]),
                                                 torch.from_numpy((x-backend.mean)/backend.std)).numpy()
                    actual = model.predict_pairs(cache, cache, first, second)
                    direct = expected[:, model.target].copy()
                    direct[identical] = 0
                    np.testing.assert_allclose(actual, direct, atol=1e-7, rtol=1e-6)
                    np.testing.assert_array_equal(
                        paired_predictions(model, cache, cache, first, second, False), actual)
                    np.testing.assert_allclose(
                        model.predict_pairs(cache, cache, second, first), actual, atol=1e-7)
                    cartesian = predict(model, cache, cache, np.arange(3))
                    np.testing.assert_array_equal(np.diag(cartesian), 0)
                    np.testing.assert_allclose(cartesian, cartesian.T, atol=1e-7)
                self.assertEqual(len(backend.embeddings), 3)
                self.assertLessEqual(load.call_count, 2)

    def test_discovery_and_api_preserve_four_seasonal_choices_and_warning(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "outputs" / "senorge_seasonal_station_model"
            folder.mkdir(parents=True)
            for name in ("metadata.json", "combined.pt", "combined_normalization.npz", "seasonal_manifest.json"):
                (folder / name).touch()
            (root / "outputs" / "temperature_5y_prediction_rmse_station_features.gpkg").touch()
            modes = WeatherRegistry(root, None, None, None).available
            modes = [m for m in modes if m["kind"].startswith("seasonal_")]
            self.assertEqual(len(modes), 4)

        class Weather(FakeWeather):
            available = FakeWeather.available + modes

            def profile(self, identifier):
                result = super().profile(identifier)
                result.update(model_warning=WARNING, training_period="1996–2025")
                return result

        engine = FakeWeatherEngine()
        engine.weather = Weather()
        client = create_app(engine).test_client()
        for mode in modes:
            result = client.post("/api/action", json=dict(action="dataset", value=mode["id"]))
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.get_json()["dataset"], mode["id"])
            result = client.post("/api/action", json=dict(action="add", lon=5, lat=60))
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.get_json()["model_warning"], WARNING)
            result = client.post("/api/action", json=dict(action="undo"))
            self.assertEqual(result.status_code, 200)


if __name__ == "__main__":
    unittest.main()
