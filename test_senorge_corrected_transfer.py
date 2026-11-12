from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

import senorge_corrected_transfer as transfer


def data_fixture(count=200):
    rng = np.random.default_rng(42)
    source_c = np.linspace(-20., 20., count)
    x = rng.normal(size=(count, len(transfer.FEATURES))).astype(np.float32)
    x[:, 0] = source_c
    return dict(x=x, source_c=source_c, target_c=source_c-10,
                target=np.asarray([f"T{i%10}" for i in range(count)]),
                source=np.asarray([f"S{i%5}" for i in range(count)]),
                date=np.asarray([f"2000-01-{1+i%20:02d}" for i in range(count)]),
                block=np.asarray([f"{i%5}:40" for i in range(count)]))


class TransferTests(unittest.TestCase):
    def test_perfect_offset_has_zero_corrected_rmse_and_zero_learned_error(self):
        train, test = data_fixture(), data_fixture(50)
        test["source_c"] = np.linspace(-30., 30., 50)
        test["target_c"] = test["source_c"]-10
        test["x"][:, 0] = test["source_c"]
        for family in ("offset", "signed_linear", "terrain_tree"):
            model = transfer.fit_mean(train, family, 1)
            prediction = transfer.predict_mean(model, test)
            self.assertLess(transfer.score(test, prediction)["rmse_c"], 1e-10)
        offset = transfer.fit_mean(train, "offset", 1)
        oof = transfer.predict_mean(offset, train)
        error_model = ExtraTreesRegressor(n_estimators=4, random_state=42)
        error_model.fit(transfer.error_inputs(train, oof), (oof-train["target_c"])**2)
        np.testing.assert_array_equal(
            transfer.rmse_predictions(error_model, test, transfer.predict_mean(offset, test)), 0)
        self.assertEqual(transfer.score(test, test["source_c"])["rmse_c"], 10)

    def test_signed_direction_and_inference_features_never_use_target_temperature(self):
        table = np.zeros((2, 16))
        table[:, :3] = [[200000, 6800000, 100], [203000, 6804000, 1100]]
        dates = np.array(["2000-01-01"])
        x = transfer.features(table, np.array([0]), np.array([1]), np.array([2.]), dates, np.array([0]))
        reverse = transfer.features(table, np.array([1]), np.array([0]), np.array([2.]), dates, np.array([0]))
        self.assertEqual(x.shape, (1, len(transfer.FEATURES)))
        np.testing.assert_array_equal(x[0, 4:7], [3, 4, 5])
        np.testing.assert_array_equal(reverse[0, 4:7], [-3, -4, 5])
        np.testing.assert_array_equal(x[0, -14:], -reverse[0, -14:])
        with self.assertRaises(ValueError):
            transfer.features(table, np.array([0]), np.array([1]), np.array([np.nan]), dates, np.array([0]))

    def test_error_model_means_squared_error_not_absolute_error(self):
        class Error:
            def predict(self, x):
                return np.array([0., 9., 16.])
        data = data_fixture(3)
        np.testing.assert_array_equal(transfer.rmse_predictions(Error(), data, np.zeros(3)), [0, 3, 4])

    def test_regularized_linear_collinearity_does_not_explode_out_of_region(self):
        train, test = data_fixture(), data_fixture(50)
        rng = np.random.default_rng(2026)
        common = rng.normal(size=len(train["source_c"])).astype(np.float32)
        train["x"][:, 7] = common
        train["x"][:, 8] = common+np.float32(1e-6)
        train["target_c"] += common
        test["x"][:, 7] = .2
        test["x"][:, 8] = -.2
        prediction = transfer.predict_mean(transfer.fit_mean(train, "signed_linear", 1), test)
        self.assertLess(np.max(np.abs(prediction-test["source_c"])), 15)

    def test_sampling_and_missing_values_are_explicit(self):
        table = np.zeros((3, 16))
        table[:, 0] = [0, 1000, 2000]
        weather = np.ones((2, 3, 4), dtype=np.float32)
        weather[:, 1] = -9
        plan = pd.DataFrame(dict(date=["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]))
        pairs = dict(pairs=np.array([[0, 1], [0, 2]]), block=np.array(["0:0", "0:0"]))
        rows = transfer.sampled_rows(table, weather, plan, pairs, np.arange(2), np.arange(4), 16, 42)
        self.assertEqual(len(rows["target_c"]), 16)
        self.assertEqual(len(set(zip(rows["pair"], rows["date"], rows["variable"]))), 16)
        weather[:, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            transfer.sampled_rows(table, weather, plan, pairs, np.arange(2), np.arange(4), 16, 42)

    def test_real_geometry_keeps_endpoint_holdouts_disjoint(self):
        atlas = transfer.ROOT / "outputs" / "senorge_footprint_expanded"
        table = np.load(atlas / "grid_table.npy", mmap_mode="r", allow_pickle=False)
        try:
            with np.load(atlas / "references.npz", allow_pickle=False) as refs:
                result = transfer.geography_pairs(table, refs)
            self.assertEqual(len(result["pairs"]), 180*8)
            for split in range(3):
                pair = result["pairs"][result["split"] == split]
                self.assertFalse((pair[:, 0] == pair[:, 1]).any())
                for other in range(split+1, 3):
                    self.assertFalse(set(pair.ravel()) & set(result["pairs"][result["split"] == other].ravel()))
        finally:
            table._mmap.close()


if __name__ == "__main__":
    unittest.main()
