"""Verify returned joint intervals and provide version-independent error-scale inference."""
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd

from national_temperature_uncertainty import forest_predict
from remote_senorge.train_joint_uncertainty import (
    ERROR_FEATURES, contribution_table, joint_error_features, scenarios, scenario_metadata,
)
from remote_senorge.train_national_transfer import fusion_model, grouped, score, subset
from remote_senorge.train_national_uncertainty import interval_diagnostics, simultaneous_factor
from remote_senorge.train_valley_neural import sha256, write_json


class JointUncertainty:
    def __init__(self, root):
        folder = Path(root) / "outputs" / "joint_uncertainty_returned"
        result = folder / "joint_uncertainty_results"
        self.report = json.loads((result / "report.json").read_text())
        receipt = json.loads((folder / "verification.json").read_text())
        if (self.report["schema"] != "joint-residual-results-v1" or
                self.report["state"] != "complete" or self.report["error_features"] != ERROR_FEATURES or
                not receipt["all_test_configurations_verified"] or
                receipt["report_sha256"] != sha256(result / "report.json") or
                receipt["model_sha256"] != sha256(result / "joint_scale.joblib") or
                receipt["arrays_sha256"] != sha256(folder / "portable_joint_scale.npz")):
            raise ValueError("Joint uncertainty verification/provenance mismatch")
        self.factor = float(self.report["factor"])
        if not np.isfinite(self.factor) or self.factor <= 0 or self.report["scale_floor_c"] != .1:
            raise ValueError("Invalid joint calibration factor/floor")
        with np.load(folder / "portable_joint_scale.npz", allow_pickle=False) as saved:
            self.arrays = {key: saved[key].copy() for key in saved.files}

    def radii(self, x, anchor, mask, prediction):
        features = joint_error_features(x, anchor, mask, prediction)
        return np.maximum(.1, forest_predict(self.arrays, features, len(ERROR_FEATURES)))*self.factor


def portable_forest(model):
    if model.n_features_in_ != len(ERROR_FEATURES) or len(model.estimators_) != 128:
        raise ValueError("Incorrect joint forest dimensions")
    chunks = {key: [] for key in ("left", "right", "feature", "threshold", "value")}
    offsets = [0]
    for estimator in model.estimators_:
        tree = estimator.tree_
        if tree.max_depth > 16 or tree.value.shape[1:] != (1, 1):
            raise ValueError("Unsupported joint error tree")
        for key, values in (("left", tree.children_left), ("right", tree.children_right)):
            chunks[key].append(np.where(values >= 0, values+offsets[-1], -1))
        chunks["feature"].append(tree.feature)
        chunks["threshold"].append(tree.threshold)
        chunks["value"].append(tree.value[:, 0, 0])
        offsets.append(offsets[-1]+tree.node_count)
    arrays = {key: np.concatenate(values) for key, values in chunks.items()}
    arrays["offsets"] = np.asarray(offsets, np.int64)
    return arrays


def compare_diagnostics(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("Diagnostic output shape mismatch")
    for key, value in expected.items():
        if isinstance(value, dict):
            compare_diagnostics(actual[key], value)
        elif isinstance(value, list):
            if len(actual[key]) != len(value):
                raise ValueError(f"Diagnostic list shape mismatch: {key}")
            for item, expected_item in zip(actual[key], value):
                compare_diagnostics(item, expected_item)
        elif value is None:
            if actual[key] is not None:
                raise ValueError(f"Diagnostic mismatch: {key}")
        elif isinstance(value, (int, float)):
            np.testing.assert_allclose(actual[key], value, atol=2e-4, rtol=2e-5, err_msg=key)
        elif actual[key] != value:
            raise ValueError(f"Diagnostic mismatch: {key}")


def verify(root):
    """Reproduce calibration and every returned test configuration; never refit or retune."""
    import hashlib
    import joblib
    import torch

    root = Path(root)
    folder = root / "outputs" / "joint_uncertainty_returned"
    result = folder / "joint_uncertainty_results"
    report = json.loads((result / "report.json").read_text())
    if (report["schema"] != "joint-residual-results-v1" or report["state"] != "complete" or
            report["error_features"] != ERROR_FEATURES or report["uncertainty_map_validated"] is not False):
        raise ValueError("Wrong joint result schema/state/features")
    for name, digest in report["checksums"].items():
        if Path(name).name != name or sha256(result / name) != digest:
            raise ValueError(f"Returned joint checksum mismatch: {name}")
    run = json.loads((result / "run_config.json").read_text())
    spec = json.loads((folder / "joint_spec.json").read_text())
    with ZipFile(root / "outputs" / "joint_uncertainty_training_v1.zip") as archive:
        checks = json.loads(archive.read("package_checksums.json"))
        if checks != json.loads((folder / "package_checksums.json").read_text()):
            raise ValueError("Returned package checksums changed")
        for name, digest in checks.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Original package checksum mismatch: {name}")
        for name, digest in {**run["inputs"], **run["code"]}.items():
            if digest != checks[name]:
                raise ValueError(f"Joint run provenance mismatch: {name}")
        for name in ("joint_spec.json", "manifest.json"):
            if archive.read(name) != (folder / name).read_bytes():
                raise ValueError(f"Returned metadata changed: {name}")
        with archive.open("training.npz") as stream, np.load(stream, allow_pickle=False) as saved:
            mask = np.isin(saved["split"], ["calibration", "test"])
            data = {key: saved[key][mask].copy() for key in saved.files}
        from io import BytesIO
        with np.load(BytesIO(archive.read("selected_normalization.npz")), allow_pickle=False) as saved:
            mean, std = saved["mean"].copy(), saved["std"].copy()
        weights = torch.load(BytesIO(archive.read("selected_fusion.pt")), map_location="cpu", weights_only=True)
    model = fusion_model(torch, spec["config"]["width"])
    model.load_state_dict(weights)
    torch.set_num_threads(4)
    args = SimpleNamespace(device="cpu", batch_size=2048)
    groups = grouped(data)
    # Only the user's checksummed forest is loaded, for conversion. Runtime uses arrays.
    forest = joblib.load(result / "joint_scale.joblib")
    arrays = portable_forest(forest)
    with np.load(result / "joint_oof_scale_sample.npz", allow_pickle=False) as saved:
        oof_x = saved["x"]
        if oof_x.shape != (300000, len(ERROR_FEATURES)) or not np.isfinite(saved["y"]).all():
            raise ValueError("Unexpected OOF scale fitting sample")
        np.testing.assert_allclose(forest_predict(arrays, oof_x, len(ERROR_FEATURES)),
                                   forest.predict(oof_x), atol=1e-10, rtol=1e-10)
    calibration = subset(groups, groups["split"] == "calibration")
    cal = scenarios(torch, model, data, calibration, mean, std, args)
    meta = scenario_metadata(calibration, cal)
    scales = np.maximum(.1, forest_predict(arrays, cal["features"], len(ERROR_FEATURES)))
    factor = simultaneous_factor(np.abs(cal["prediction"]-meta["target_c"]), scales, meta["night"])
    np.testing.assert_allclose(factor, report["factor"], atol=2e-5, rtol=2e-5)
    if len(set(meta["night"])) != report["calibration_nights"]:
        raise ValueError("Calibration night count mismatch")
    testing = subset(groups, groups["split"] == "test")
    test = scenarios(torch, model, data, testing, mean, std, args)
    meta = scenario_metadata(testing, test)
    scales = np.maximum(.1, forest_predict(arrays, test["features"], len(ERROR_FEATURES)))
    radii = scales*report["factor"]
    reproduced = contribution_table(meta, test, radii)
    returned = pd.read_csv(result / "test_contributions.csv", keep_default_na=False)
    if reproduced.columns.tolist() != returned.columns.tolist() or len(reproduced) != len(returned):
        raise ValueError("Returned contribution table shape mismatch")
    for key in reproduced.columns:
        if reproduced[key].dtype.kind in "fi":
            np.testing.assert_allclose(reproduced[key], returned[key], atol=2e-4, rtol=2e-5, err_msg=key)
        elif not np.array_equal(reproduced[key], returned[key]):
            raise ValueError(f"Returned scenario identity mismatch: {key}")
    # Use saved GPU values for exact coverage boundary reproduction after checking CPU parity.
    predictions, saved_radii = returned["prediction_c"].to_numpy(), returned["radius_c"].to_numpy()
    full = test["removed"] == ""
    compare_diagnostics(interval_diagnostics(subset(meta, full), np.arange(full.sum()),
                                            predictions[full], saved_radii[full]), report["full_joint_test"])
    compare_diagnostics(interval_diagnostics(meta, np.arange(len(predictions)), predictions, saved_radii),
                        report["full_and_deletions_test"])
    for count, diagnostics in report["by_source_count"].items():
        selected = test["features"][:, -1] == int(count)
        compare_diagnostics(interval_diagnostics(subset(meta, selected), np.arange(selected.sum()),
                            predictions[selected], saved_radii[selected]), diagnostics)
    for fold in report["folds"]:
        checkpoint = torch.load(result / f"{fold['scheme']}_fold{fold['fold']}.pt",
                                map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != spec["epochs"]:
            raise ValueError("Incomplete cross-fit checkpoint")
    np.savez_compressed(folder / "portable_joint_scale.npz", **arrays)
    changes = returned.loc[returned.removed_source.ne("")]
    full_predictions = returned.loc[returned.removed_source.eq("")].copy()
    full_predictions["source_count"] = test["features"][full, -1].astype(int)
    full_predictions["source_readings_json"] = [
        json.dumps([dict(source=str(data["source"][pair]), source_c=float(data["source_c"][pair]))
                    for pair in testing["index"][group][testing["mask"][group]]])
        for group in test["group"][full]]
    full_predictions.to_csv(folder / "verified_full_predictions.csv", index=False)
    receipt = dict(
        schema="verified-joint-uncertainty-v1", all_test_configurations_verified=True,
        archive_sha256=sha256(root / "outputs" / "joint_uncertainty_results.tar.gz"),
        report_sha256=sha256(result / "report.json"), model_sha256=sha256(result / "joint_scale.joblib"),
        arrays_sha256=sha256(folder / "portable_joint_scale.npz"),
        training_sha256=checks["training.npz"], calibrated_factor_reproduced=factor,
        calibration_configurations=len(cal["prediction"]), test_configurations=len(predictions),
        full_groups=int(full.sum()), valid_deletions=len(changes), oof_scale_rows=300000,
        full_temperature=score(subset(meta, full), predictions[full]),
        radius_change_when_removed=dict(negative=int((changes.radius_change_when_removed_c < 0).sum()),
            positive=int((changes.radius_change_when_removed_c > 0).sum()),
            minimum_c=float(changes.radius_change_when_removed_c.min()),
            maximum_c=float(changes.radius_change_when_removed_c.max())),
        verified_full_predictions_sha256=sha256(folder / "verified_full_predictions.csv"),
        map_integrated=False, uncertainty_map_validated=False,
        verification_note="Trusted sklearn1.7.2 forest converted under1.3.0 with explicit version warnings; "
                          "numeric traversal matches forest and every returned GPU test radius. "
                          "No dependency upgrades,refitting,interval tuning or app changes.")
    write_json(folder / "verification.json", receipt)
    JointUncertainty(root)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
