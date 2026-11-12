from pathlib import Path
import unittest

from flask import Flask
import numpy as np
import pandas as pd
from pyproj import Transformer

from station_model_evaluation import ModelEvaluation, metrics, register_model_evaluation


def fixture():
    model = ModelEvaluation.__new__(ModelEvaluation)
    model.transform = Transformer.from_crs(25833, 4326, always_xy=True)
    model.table = np.array([[200000., 6800000.], [201000., 6800000.], [202000., 6800000.]])
    model.stations = dict(T=dict(x=200000., y=6800000., name="Target"),
                          A=dict(x=201000., y=6800000., name="Source A"),
                          B=dict(x=202000., y=6800000., name="Source B"))
    model.frames = dict(senorge=pd.DataFrame(dict(
        source=["0", "1", "0"], target=["2", "2", "2"], date=["2024-12-15"]*3,
        variable=[0, 0, 1], split=["test"]*3, source_c=[-2., -2., -5.],
        target_c=[-10., -10., -12.], prediction_c=[-9., -7., -11.],
        predicted_rmse_c=[1., 3., 1.])),
        joint=pd.DataFrame(dict(target=["T"], time=["2024-12-15T03:00:00+00:00"],
            date=["2024-12-15"], hour=[3], split=["test"], prediction_c=[-9.],
            target_c=[-10.], radius_c=[2.], lower_c=[-11.], upper_c=[-7.], source_count=[2],
            source_readings_json=['[{"source":"A","source_c":-2},{"source":"B","source_c":-3}]'])))
    model.deletions = pd.DataFrame(dict(
        target=["T"]*2, time=["2024-12-15T03:00:00+00:00"]*2, removed_source=["A", "B"],
        prediction_c=[-8., -9.5], radius_c=[3., 1.], lower_c=[-11., -10.5], upper_c=[-5., -8.5],
        radius_change_when_removed_c=[1., -1.], temperature_change_when_removed_c=[1., -.5]))
    return model


class EvaluationTests(unittest.TestCase):
    def test_rmse_pooling_not_mean_rmse_or_joint_fusion(self):
        model = fixture()
        selected = model.select("senorge", "test", "2024-12-15")
        summary = metrics(selected, "senorge")
        self.assertEqual(summary["rows"], 2)
        self.assertAlmostEqual(summary["rmse_c"], np.sqrt(5))
        self.assertAlmostEqual(summary["predicted_pooled_rmse_c"], np.sqrt(5))
        geo = model.grid("senorge", selected)
        self.assertEqual(len(geo["features"]), 1)
        p = geo["features"][0]["properties"]
        self.assertEqual(p["prediction_c"], -8)
        self.assertAlmostEqual(p["rmse_c"], np.sqrt(5))
        self.assertNotIn("radius_c", p)
        self.assertIn("NOT a fusion", geo["aggregation"])

    def test_exact_1km_cell_and_source_drilldown(self):
        model = fixture()
        selected = model.select("senorge", "test", "2024-12-15", source="0")
        self.assertEqual(len(selected), 1)
        geo = model.grid("senorge", selected)["features"][0]
        ring = np.array(geo["geometry"]["coordinates"][0])
        x, y = Transformer.from_crs(4326, 25833, always_xy=True).transform(ring[:, 0], ring[:, 1])
        np.testing.assert_allclose(x, [201500, 202500, 202500, 201500, 201500], atol=.001)
        np.testing.assert_allclose(y, [6800500, 6800500, 6799500, 6799500, 6800500], atol=.001)
        detail = model.detail("senorge", selected, "2")
        self.assertEqual(len(detail["sources"]), 1)
        self.assertEqual(detail["sources"][0]["error_c"], 1)
        self.assertEqual(detail["sources"][0]["source"], "0")

    def test_signed_joint_removal_and_interval_not_rmse(self):
        model = fixture()
        part = model.select("joint", "test", "2024-12-15")
        geo = model.grid("joint", part)["features"][0]
        self.assertEqual(geo["geometry"]["type"], "Point")
        self.assertEqual(geo["properties"]["radius_c"], 2)
        self.assertEqual(geo["properties"]["rmse_c"], 1)
        self.assertNotIn("predicted_pooled_rmse_c", geo["properties"])
        detail = model.detail("joint", part, "T")
        self.assertEqual([s["radius_change_c"] for s in detail["sources"]], [1., -1.])
        part.loc[:, "source_count"] = 1
        part.loc[:, "source_readings_json"] = '[{"source":"A","source_c":-2}]'
        model.deletions = model.deletions.iloc[:0]
        detail = model.detail("joint", part, "T")
        self.assertTrue(detail["sole_source_no_deletion"])
        self.assertEqual(detail["sources"][0]["source"], "A")
        self.assertFalse(detail["sources"][0]["deletion_available"])

    def test_invalid_filters_and_empty_samples_are_explicit(self):
        model = fixture()
        for kwargs in (dict(model="unknown"), dict(date="2024-12-16"), dict(split="train"),
                       dict(variable="7"), dict(source="unknown")):
            args = dict(model="senorge", split="test", date="2024-12-15")
            args.update(kwargs)
            with self.assertRaises(ValueError):
                model.select(**args)
        with self.assertRaisesRegex(ValueError, "No evaluated"):
            model.select("joint", "test", "2024-12-15", hour="0")
        with self.assertRaisesRegex(ValueError, "single source"):
            model.select("joint", "test", "2024-12-15", source="A")
        with self.assertRaisesRegex(ValueError, "not in"):
            model.detail("senorge", model.frames["senorge"], "999")

    def test_routes_csv_and_errors(self):
        app = Flask(__name__)
        register_model_evaluation(app, Path(__file__).parent, fixture())
        client = app.test_client()
        q = "model=senorge&split=test&date=2024-12-15&variable=0&source=0"
        response = client.get("/api/model-evaluation/grid?"+q)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["summary"]["rows"], 1)
        response = client.get("/api/model-evaluation/detail?"+q+"&target=2")
        self.assertEqual(response.json["sources"][0]["error_c"], 1)
        response = client.get("/api/model-evaluation/export?"+q)
        self.assertIn("attachment", response.headers["Content-Disposition"])
        self.assertIn(b"error_c", response.data)
        self.assertEqual(len(response.data.splitlines()), 2)
        response = client.get("/api/model-evaluation/grid?model=invalid")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)
        for path in ("/model-evaluation", "/station_model_evaluation.js"):
            with client.get(path) as response:
                self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
