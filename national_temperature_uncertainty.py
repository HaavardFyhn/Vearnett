"""Version-independent inference for the verified returned scalar error forest."""
import json
from pathlib import Path

import numpy as np

from remote_senorge.train_valley_neural import sha256, write_json


class TemperatureUncertainty:
    def __init__(self, root):
        folder = Path(root) / "outputs" / "national_uncertainty_returned"
        result = folder / "national_uncertainty_results"
        self.report = json.loads((result / "report.json").read_text())
        if self.report["schema"] != "national-residual-uncertainty-v2" or self.report["state"] != "complete":
            raise ValueError("A completed v2 uncertainty result is required")
        receipt = json.loads((folder / "portable_error_scale.json").read_text())
        if (sha256(result / "report.json") != receipt["report_sha256"] or
                sha256(result / "error_scale.joblib") != receipt["original_model_sha256"] or
                sha256(folder / "portable_error_scale.npz") != receipt["arrays_sha256"] or
                not receipt["all_test_scales_verified"]):
            raise ValueError("Portable uncertainty provenance/checksum mismatch")
        self.model_sha256 = receipt["original_model_sha256"]
        self.factor = float(self.report["factor"])
        if not np.isfinite(self.factor) or self.factor <= 0 or self.report["scale_floor_c"] != .1:
            raise ValueError("Invalid uncertainty calibration factor/floor")
        with np.load(folder / "portable_error_scale.npz", allow_pickle=False) as arrays:
            self.arrays = {key: arrays[key].copy() for key in arrays.files}
        self.run = json.loads((result / "run_config.json").read_text())

    def scales(self, features, prediction):
        return np.maximum(.1, forest_predict(self.arrays, np.column_stack([features, prediction])))

    def radii(self, features, prediction):
        return self.scales(features, prediction)*self.factor


def forest_predict(arrays, features, expected_features=40):
    """Traverse scalar sklearn trees using their original float32 input semantics."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != expected_features or not np.isfinite(x).all():
        raise ValueError(f"Expected finite {expected_features}-column error-scale features")
    left, right, feature, threshold, value, offsets = (
        arrays[key] for key in ("left", "right", "feature", "threshold", "value", "offsets"))
    predictions = np.zeros(len(x), dtype=np.float64)
    for start, stop in zip(offsets[:-1], offsets[1:]):
        nodes = np.full(len(x), start, dtype=np.int64)
        for _ in range(18):
            active = np.flatnonzero(left[nodes] >= 0)
            if not len(active):
                break
            current = nodes[active]
            nodes[active] = np.where(
                x[active, feature[current]] <= threshold[current], left[current], right[current])
            if (nodes[active] < start).any() or (nodes[active] >= stop).any():
                raise ValueError("Invalid portable tree child indices")
        else:
            raise ValueError("Portable tree exceeds expected maximum depth")
        predictions += value[nodes]
    if not np.isfinite(predictions).all() or not len(offsets) > 1:
        raise ValueError("Invalid portable error-scale output")
    return predictions/(len(offsets)-1)


def export_verified_forest(root):
    """Convert the user's trusted result, checking all saved test scales before persistence."""
    import joblib
    import pandas as pd
    from zipfile import ZipFile

    root = Path(root)
    folder = root / "outputs" / "national_uncertainty_returned"
    result = folder / "national_uncertainty_results"
    report = json.loads((result / "report.json").read_text())
    for name, digest in report["checksums"].items():
        if Path(name).name != name or sha256(result / name) != digest:
            raise ValueError(f"Returned uncertainty checksum mismatch: {name}")
    model = joblib.load(result / "error_scale.joblib")
    if model.n_features_in_ != 40:
        raise ValueError("Incorrect error model input shape")
    chunks = {key: [] for key in ("left", "right", "feature", "threshold", "value")}
    offsets = [0]
    for estimator in model.estimators_:
        tree = estimator.tree_
        if tree.max_depth > 16 or tree.value.shape[1:] != (1, 1):
            raise ValueError("Unsupported error tree")
        for key, values in (("left", tree.children_left), ("right", tree.children_right)):
            chunks[key].append(np.where(values >= 0, values+offsets[-1], -1))
        chunks["feature"].append(tree.feature)
        chunks["threshold"].append(tree.threshold)
        chunks["value"].append(tree.value[:, 0, 0])
        offsets.append(offsets[-1]+tree.node_count)
    arrays = {key: np.concatenate(values) for key, values in chunks.items()}
    arrays["offsets"] = np.asarray(offsets, np.int64)
    with ZipFile(root / "outputs" / "national_uncertainty_training_v2.zip") as archive:
        run = json.loads((result / "run_config.json").read_text())
        checks = json.loads(archive.read("package_checksums.json"))
        if run["training_sha256"] != checks["training.npz"]:
            raise ValueError("Portable error verification data provenance mismatch")
        with archive.open("training.npz") as stream, np.load(stream, allow_pickle=False) as data:
            test = data["split"] == "test"
            x = data["x"][test]
            source_ids, target_ids, times = data["source"][test], data["target"][test], data["time"][test]
    saved = pd.read_csv(result / "test_intervals.csv")
    if not (np.array_equal(source_ids, saved.source) and np.array_equal(target_ids, saved.target) and
            np.array_equal(times, saved.time)):
        raise ValueError("Portable error test-row identity mismatch")
    features = np.column_stack([x, saved.prediction_c.to_numpy()]).astype(np.float32)
    predicted = forest_predict(arrays, features)
    np.testing.assert_allclose(predicted, model.predict(features), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.maximum(.1, predicted), saved.scale_c, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.maximum(.1, predicted)*report["factor"], saved.radius_c, atol=1e-5)
    np.savez_compressed(folder / "portable_error_scale.npz", **arrays)
    write_json(folder / "portable_error_scale.json", dict(
        report_sha256=sha256(result / "report.json"),
        original_model_sha256=sha256(result / "error_scale.joblib"),
        arrays_sha256=sha256(folder / "portable_error_scale.npz"),
        all_test_scales_verified=True, test_rows=len(saved), trees=len(offsets)-1))


if __name__ == "__main__":
    export_verified_forest(Path(__file__).resolve().parent)
