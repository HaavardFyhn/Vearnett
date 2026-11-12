import tempfile
from pathlib import Path
import unittest

import numpy as np

from national_joint_uncertainty import (
    ERROR_FEATURES, JointUncertainty, compare_diagnostics, portable_forest,
)
from national_temperature_uncertainty import forest_predict
from remote_senorge.train_valley_neural import sha256, write_json


def leaf_arrays():
    return dict(left=np.array([-1]), right=np.array([-1]), feature=np.array([-2]),
                threshold=np.array([-2.]), value=np.array([2.]), offsets=np.array([0, 1]))


class JointPortableTests(unittest.TestCase):
    def test_default_single_source_and_joint_feature_contracts(self):
        arrays = leaf_arrays()
        np.testing.assert_array_equal(forest_predict(arrays, np.zeros((3, 40))), [2]*3)
        np.testing.assert_array_equal(forest_predict(arrays, np.zeros((3, 120)), 120), [2]*3)
        with self.assertRaisesRegex(ValueError, "40-column"):
            forest_predict(arrays, np.zeros((3, 120)))
        with self.assertRaises(ValueError):
            forest_predict(arrays, np.full((3, 120), np.nan), 120)

    def test_portable_128_tree_predictions_match_sklearn(self):
        from sklearn.ensemble import ExtraTreesRegressor
        rng = np.random.default_rng(42)
        x = rng.normal(size=(200, 120)).astype(np.float32)
        y = np.abs(x[:, 0]-x[:, 20])
        model = ExtraTreesRegressor(n_estimators=128, max_depth=5, random_state=42).fit(x, y)
        np.testing.assert_allclose(forest_predict(portable_forest(model), x, 120), model.predict(x),
                                   atol=1e-12, rtol=1e-12)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            portable_forest(ExtraTreesRegressor(n_estimators=2).fit(x, y))

    def test_runtime_uses_verified_arrays_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "outputs" / "joint_uncertainty_returned"
            result = folder / "joint_uncertainty_results"
            result.mkdir(parents=True)
            write_json(result / "report.json", dict(schema="joint-residual-results-v1", state="complete",
                       error_features=ERROR_FEATURES, factor=3., scale_floor_c=.1))
            (result / "joint_scale.joblib").write_bytes(b"not loaded by runtime")
            np.savez_compressed(folder / "portable_joint_scale.npz", **leaf_arrays())
            write_json(folder / "verification.json", dict(all_test_configurations_verified=True,
                       report_sha256=sha256(result / "report.json"),
                       model_sha256=sha256(result / "joint_scale.joblib"),
                       arrays_sha256=sha256(folder / "portable_joint_scale.npz")))
            model = JointUncertainty(root)
            x, anchor = np.zeros((2, 6, 39)), np.zeros((2, 6))
            mask = np.array([[True]*6, [True]+[False]*5])
            np.testing.assert_array_equal(model.radii(x, anchor, mask, np.zeros(2)), [6, 6])
            with self.assertRaises(ValueError):
                model.radii(x, anchor, np.zeros_like(mask), np.zeros(2))
            (result / "joint_scale.joblib").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                JointUncertainty(root)

    def test_diagnostic_shapes_and_counts_are_verified(self):
        compare_diagnostics(dict(rows=3, cold=dict(rows=0)), dict(rows=3, cold=dict(rows=0)))
        with self.assertRaisesRegex(ValueError, "shape"):
            compare_diagnostics(dict(rows=3), dict(rows=3, cold=dict(rows=0)))
        with self.assertRaises(AssertionError):
            compare_diagnostics(dict(rows=4), dict(rows=3))


if __name__ == "__main__":
    unittest.main()
