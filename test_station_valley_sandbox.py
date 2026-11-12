from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from flask import Flask

from station_valley_sandbox import propagate, register_sandbox_view, transfer_features, ValleySandbox


class SandboxTests(unittest.TestCase):
    def test_tree_hops_no_cycles_and_synthetic_feedback(self):
        stations = pd.DataFrame(dict(station_id=["A", "B", "C", "D", "SEA"]))
        table = np.zeros((5, 16))
        table[:, 0] = [0, 40000, 80000, 120000, 0]
        graph = np.ones((5, 5, 5))
        graph[4, :, 0] = graph[:, 4, 0] = 0
        received = []

        def predict(a, b, source):
            received.append(source)
            return source+2

        result = propagate(stations, table, graph, "A", -10, 3, predict)
        self.assertEqual([r["hop"] for r in result["nodes"]], [0, 1, 2, 3])
        self.assertEqual(received, [-10, -8, -6])
        self.assertEqual(result["unreachable"], ["SEA"])
        self.assertTrue(all(r["synthetic"] for r in result["nodes"]))
        self.assertEqual(len(propagate(stations, table, graph, "A", 20, 1, predict)["nodes"]), 2)
        for temperature in (-21, 21, float("nan")):
            with self.assertRaises(ValueError):
                propagate(stations, table, graph, "A", temperature, 3, predict)
        with self.assertRaises(ValueError):
            propagate(stations, table, graph, "A", 0, 4, predict)
        with self.assertRaises(ValueError):
            propagate(stations, table, graph, "A", 0, 3, lambda *args: float("nan"))

    def test_real_features_match_training_and_saved_test_predictions(self):
        root = Path(__file__).resolve().parent
        model = ValleySandbox(root)
        frame = pd.read_csv(root / "outputs" / "valley_neural_returned" /
                            "valley_neural_results" / "selected_test_predictions.csv")
        row = frame.iloc[0]
        result = model.simulate(row["source"], float(row["source_c"]), 1, row["time"])
        target = next(r for r in result["nodes"] if r["station_id"] == row["target"])
        self.assertAlmostEqual(target["temperature_c"], row["prediction_c"], places=4)
        source = model.catalogue.index[model.catalogue["station_id"] == row["source"]][0]
        destination = model.catalogue.index[model.catalogue["station_id"] == row["target"]][0]
        features, _ = transfer_features(model.table, model.graph, source, destination,
                                       row["source_c"], pd.Timestamp(row["time"]), model.forecast)
        with np.load(root / "outputs" / "valley_neural_training" / "training.npz", allow_pickle=False) as data:
            mask = (data["source"] == row["source"]) & (data["target"] == row["target"]) & (data["time"] == row["time"])
            self.assertEqual(mask.sum(), 1)
            np.testing.assert_allclose(features, data["x"][mask][0], atol=1e-5)
        cold = model.simulate("SN55770", -20, 3, row["time"])
        warm = model.simulate("SN55770", 20, 3, row["time"])
        self.assertEqual(cold["nodes"][0]["temperature_c"], -20)
        self.assertEqual(warm["nodes"][0]["temperature_c"], 20)
        self.assertNotEqual(cold["nodes"][1]["temperature_c"], warm["nodes"][1]["temperature_c"])
        with self.assertRaises(ValueError):
            model.simulate("SN55770", 0, 3, "2030-01-01")

    def test_routes_success_and_explicit_errors(self):
        root = Path(__file__).resolve().parent
        app = Flask(__name__)
        register_sandbox_view(app, root)
        with app.test_client() as client:
            response = client.get("/valley-sandbox")
            self.assertEqual(response.status_code, 200)
            response.close()
            options = client.get("/api/valley-sandbox/options")
            self.assertEqual(options.status_code, 200)
            self.assertEqual(len(options.json["stations"]), 11)
            response = client.get("/api/valley-sandbox/simulate", query_string=dict(
                source="SN55770", temperature=-10, hops=3, time=options.json["times"][-1]))
            self.assertEqual(response.status_code, 200)
            self.assertTrue(all(node["synthetic"] for node in response.json["nodes"]))
            response = client.get("/api/valley-sandbox/simulate", query_string=dict(
                source="SN55770", temperature=999, hops=3, time=options.json["times"][0]))
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.json)


if __name__ == "__main__":
    unittest.main()
