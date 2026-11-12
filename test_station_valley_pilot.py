import json
from pathlib import Path
import tempfile
import unittest

from flask import Flask
import numpy as np
import pandas as pd

from remote_senorge.train_remote import sha256, write_json
from station_valley_pilot import METHODS, ValleyResults, register_valley_view
from train_valley_pilot import scores


class ValleyViewTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "outputs" / "valley_regime_pilot"
        output = root / "outputs" / "valley_regime_pilot_results"
        data.mkdir(parents=True)
        output.mkdir(parents=True)
        pd.DataFrame([
            dict(station_id="SN1", name="source", lat=61., lon=6., group=0),
            dict(station_id="SN56960", name="target", lat=61.4, lon=6.4, group=2),
        ]).to_csv(data / "stations.csv", index=False)
        rows = pd.DataFrame([
            dict(source="SN1", target="SN56960", time=f"2025-01-0{i+1}T03:00:00+00:00",
                 night=f"2025-01-0{i+1}", split=2, source_c=-2., target_c=value, calm_clear=bool(i))
            for i, value in enumerate([-3., -6.])
        ])
        for i, name in enumerate(METHODS):
            rows[name+"_prediction_c"] = rows["target_c"]+i-1
        rows.to_csv(output / "test_predictions.csv", index=False)
        report = dict(
            selected_by_validation="graph_regime", limitations=["Pilot, not a national surface."],
            input_checksums={"stations.csv": sha256(data / "stations.csv")},
            model_checksums={}, variants={name: dict(
                test={**scores(rows, rows[name+"_prediction_c"]), "interval_radius_c": 8.})
                for name in METHODS})
        write_json(output / "report.json", report)
        return data, output

    def test_options_case_timeline_filters_and_error_colors_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            model = ValleyResults(root)
            options = model.options()
            self.assertEqual(options["rows"], 2)
            self.assertEqual(options["nights"], 2)
            self.assertEqual(options["selected_by_validation"], "graph_regime")
            self.assertEqual(len(options["methods"]), 4)
            self.assertEqual(model.options(condition="cold")["rows"], 1)
            self.assertEqual(model.options(condition="calm_clear")["rows"], 1)
            case = model.case("SN56960", options["times"][0], "all")
            self.assertEqual(case["observed_c"], -3)
            self.assertEqual(case["transfers"][0]["forecast_transfer_prediction_c"], -4)
            self.assertEqual(case["transfers"][0]["graph_regime_prediction_c"], -1)
            timeline = model.timeline("SN56960", "SN1", "all")
            self.assertEqual(len(timeline["points"]), 2)
            self.assertEqual(timeline["summary"]["graph_regime"]["rmse_c"], 2)
            for args in (("BAD", "all"), ("SN56960", "bad")):
                with self.assertRaises(ValueError):
                    model.options(*args)
            with self.assertRaises(ValueError):
                model.case("SN56960", "bad", "all")
            with self.assertRaises(ValueError):
                model.timeline("SN56960", "bad", "all")

    def test_prediction_tampering_input_checksums_and_holdouts_are_rejected(self):
        for change in ("prediction", "input", "split", "duplicates", "observed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data, output = self.fixture(root)
                path = output / "test_predictions.csv"
                rows = pd.read_csv(path)
                if change == "prediction":
                    rows.loc[0, "terrain_prediction_c"] = 50
                elif change == "split":
                    rows.loc[0, "split"] = 0
                elif change == "duplicates":
                    rows = pd.concat([rows, rows.iloc[:1]])
                elif change == "observed":
                    rows.loc[0, "target_c"] = np.nan
                else:
                    with (data / "stations.csv").open("a") as file:
                        file.write(" ")
                rows.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    ValleyResults(root)

    def test_api_readonly_exports_and_explicit_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            app = Flask(__name__)
            register_valley_view(app, root)
            client = app.test_client()
            options = client.get("/api/valley-pilot/options")
            self.assertEqual(options.status_code, 200)
            timestamp = options.get_json()["times"][0]
            case = client.get("/api/valley-pilot/case", query_string={"time": timestamp})
            self.assertEqual(case.status_code, 200)
            self.assertEqual(case.get_json()["observed_c"], -3)
            self.assertEqual(client.get("/api/valley-pilot/timeline?source=SN1").status_code, 200)
            for url in ("/api/valley-pilot/case", "/api/valley-pilot/case?time=bad",
                        "/api/valley-pilot/options?condition=bad", "/api/valley-pilot/options?target=bad",
                        "/api/valley-pilot/timeline?source=bad"):
                result = client.get(url)
                self.assertEqual(result.status_code, 400)
                self.assertIn("error", result.get_json())
            exported = client.get("/api/valley-pilot/export")
            self.assertEqual(exported.status_code, 200)
            self.assertIn("attachment", exported.headers["Content-Disposition"])
            self.assertIn(b"graph_regime_prediction_c", exported.data)
            exported.close()
            self.assertEqual(client.post("/api/valley-pilot/options").status_code, 405)

    def test_missing_results_return_explicit_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Flask(__name__)
            register_valley_view(app, Path(temp))
            response = app.test_client().get("/api/valley-pilot/options")
            self.assertEqual(response.status_code, 503)
            self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
