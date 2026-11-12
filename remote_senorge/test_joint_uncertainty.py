import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

if __package__:
    from . import train_joint_uncertainty as trainer
    from .test_national_uncertainty import fixture
else:
    import train_joint_uncertainty as trainer
    from test_national_uncertainty import fixture


def joint_fixture(folder):
    data, geometry = fixture(folder)
    # Two different readings per target/time,including training stations with each other as sources.
    extra = {key: value.copy() for key, value in data.items()}
    extra["source"] = np.array([f"B{(int(s[1:s.index('S')])+2)%8}S{s[-1]}" for s in data["source"]])
    extra["x"][:, 0] += 1
    extra["source_c"] += 1
    data = {key: np.concatenate([value, extra[key]]) for key, value in data.items()}
    np.savez(folder / "training.npz", **data)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["rows"] = {split: int((data["split"] == split).sum()) for split in
                        ("train", "selection", "calibration", "test")}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    config = dict(family="fusion", width=8, seed=42)
    torch.manual_seed(42)
    model = trainer.fusion_model(torch, 8)
    torch.save(model.state_dict(), folder / "selected_fusion.pt")
    spec = dict(schema="joint-residual-training-v1", config=config, epochs=1, station_xy=geometry)
    (folder / "joint_spec.json").write_text(json.dumps(spec))
    checks = {name: trainer.sha256(folder / name) for name in
              ("manifest.json", "training.npz", "selected_fusion.pt", "selected_normalization.npz", "joint_spec.json")}
    (folder / "package_checksums.json").write_text(json.dumps(checks))
    return data, geometry, spec


class JointTests(unittest.TestCase):
    def test_features_permutation_masking_and_count(self):
        x = np.arange(2*6*39, dtype=np.float32).reshape(2, 6, 39)
        anchor = np.arange(12, dtype=np.float32).reshape(2, 6)
        mask = np.array([[True, True, False, False, False, False], [True]*6])
        prediction = np.array([-10., 5.])
        features = trainer.joint_error_features(x, anchor, mask, prediction)
        self.assertEqual(features.shape, (2, 120))
        np.testing.assert_array_equal(features[:, -1], [2, 6])
        perm = [5, 2, 1, 4, 0, 3]
        np.testing.assert_allclose(features, trainer.joint_error_features(
            x[:, perm], anchor[:, perm], mask[:, perm], prediction))
        x[~mask] = 1e6
        np.testing.assert_allclose(features, trainer.joint_error_features(x, anchor, mask, prediction))
        with self.assertRaises(ValueError):
            trainer.joint_error_features(x, anchor, np.zeros_like(mask), prediction)

    def test_held_observations_excluded_from_all_joint_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            data, xy, spec = joint_fixture(Path(temp) / "data")
            groups = trainer.grouped(data)
            for fold in trainer.target_fold_masks(data, 4)+trainer.fold_masks(data, xy, 4):
                fit = trainer.available_groups(groups, fold["fit"])
                evaluate = trainer.available_groups(groups, fold["evaluate"])
                held = set(evaluate["target"])
                self.assertFalse(set(fit["target"]) & held)
                self.assertFalse(set(data["source"][fit["index"][fit["mask"]]]) & held)
                self.assertTrue((fit["index"][~fit["mask"]] == -1).all())
                self.assertTrue(fit["mask"].any(1).all())

    def test_full_and_deletion_scenarios_and_no_deleting_sole_source(self):
        with tempfile.TemporaryDirectory() as temp:
            data, xy, spec = joint_fixture(Path(temp) / "data")
            args = trainer.parse_args(["--device", "cpu", "--jobs", "1"])
            all_groups = trainer.grouped(data)
            groups = trainer.subset(all_groups, np.arange(len(all_groups["index"])) < 2)
            model = trainer.fusion_model(torch, 8).eval()
            torch.set_num_threads(1)
            result = trainer.scenarios(torch, model, data, groups, np.zeros(39), np.ones(39), args)
            self.assertEqual(len(result["prediction"]), 6)
            self.assertEqual((result["removed"] == "").sum(), 2)
            self.assertEqual((result["features"][:, -1] == 1).sum(), 4)
            meta = trainer.scenario_metadata(groups, result)
            table = trainer.contribution_table(meta, result, np.arange(6)+1.)
            self.assertEqual(len(table), 6)
            self.assertTrue((table.loc[table.removed_source.eq(""), "radius_change_when_removed_c"] == 0).all())
            radii = np.where(result["removed"] == "", 10., 3.)
            changes = trainer.contribution_table(meta, result, radii)
            self.assertTrue((changes.loc[changes.removed_source.ne(""), "radius_change_when_removed_c"] == -7).all())
            groups["mask"][:, 1:] = False
            result = trainer.scenarios(torch, model, data, groups, np.zeros(39), np.ones(39), args)
            self.assertEqual(len(result["prediction"]), 2)
            self.assertTrue((result["removed"] == "").all())

    def test_perfect_constant_relationship_is_zero_joint_error(self):
        model = trainer.fusion_model(torch, 8).eval()
        torch.set_num_threads(1)
        with torch.inference_mode():
            x = torch.zeros((4, 6, 39))
            mask = torch.tensor([[True]*6, [True]*3+[False]*3, [True]+[False]*5, [True]*6])
            prediction = model(x, torch.full((4, 6), -10.), mask).numpy()
        np.testing.assert_allclose(prediction, -10, atol=1e-6)

    def test_full_run_resume_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, output = Path(temp) / "data", Path(temp) / "results"
            data, _, _ = joint_fixture(folder)
            args = trainer.parse_args(["--data", str(folder), "--output", str(output), "--device", "cpu",
                                       "--folds", "4", "--jobs", "1", "--tree-rows", "400"])
            with patch("builtins.print"):
                report = trainer.run(args)
                with patch.object(trainer, "scenarios", side_effect=AssertionError("Completed run rescored")):
                    self.assertEqual(report, trainer.run(args))
            self.assertEqual(len(report["folds"]), 8)
            self.assertFalse(report["uncertainty_map_validated"])
            self.assertGreater(report["full_and_deletions_test"]["rows"], report["full_joint_test"]["rows"])
            self.assertEqual(report["calibration_nights"], 15)
            self.assertIn("by_source_count", report)
            self.assertIn("winter", report["folds"][0]["full_errors"])
            self.assertTrue((output / "joint_oof_scale_sample.npz").exists())
            args.tree_rows = 401
            with self.assertRaisesRegex(ValueError, "Changed"):
                trainer.run(args)

    def test_exact_refit_epoch_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            data, _, spec = joint_fixture(folder)
            spec["epochs"] = 2
            groups = trainer.grouped(data)
            fold = trainer.target_fold_masks(data, 4)[0]
            args = trainer.parse_args(["--device", "cpu", "--jobs", "1", "--batch-size", "16",
                                       "--output", str(Path(temp) / "resumed")])
            args.output.mkdir()
            torch.set_num_threads(1)
            original = trainer.atomic_torch_save

            def interrupted(torch, saved, path):
                original(torch, saved, path)
                if saved["epoch"] == 1:
                    raise RuntimeError("Interruption")

            with patch.object(trainer, "atomic_torch_save", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "Interruption"):
                    trainer.refit(torch, data, groups, fold, spec, args, time.monotonic()+120)
            with patch("builtins.print"):
                resumed = trainer.refit(torch, data, groups, fold, spec, args, time.monotonic()+120)[1]
                args.output = Path(temp) / "fresh"
                args.output.mkdir()
                fresh = trainer.refit(torch, data, groups, fold, spec, args, time.monotonic()+120)[1]
            for key in resumed:
                np.testing.assert_array_equal(resumed[key], fresh[key])

    def test_held_labels_do_not_affect_refit_and_expired_budget_never_emits_model(self):
        with tempfile.TemporaryDirectory() as temp:
            data, _, spec = joint_fixture(Path(temp) / "data")
            groups = trainer.grouped(data)
            fold = trainer.target_fold_masks(data, 4)[0]
            args = trainer.parse_args(["--device", "cpu", "--jobs", "1",
                                       "--output", str(Path(temp) / "first")])
            args.output.mkdir()
            torch.set_num_threads(1)
            with patch("builtins.print"):
                first = trainer.refit(torch, data, groups, fold, spec, args, time.monotonic()+120)[1]
                changed = {key: value.copy() for key, value in data.items()}
                changed["target_c"][np.isin(data["target"], fold["held_targets"])] += 100
                altered_groups = trainer.grouped(changed)
                args.output = Path(temp) / "second"
                args.output.mkdir()
                second = trainer.refit(torch, changed, altered_groups, fold, spec, args, time.monotonic()+120)[1]
            np.testing.assert_array_equal(first["prediction"], second["prediction"])
            args.output = Path(temp) / "expired"
            args.output.mkdir()
            self.assertIsNone(trainer.refit(torch, data, groups, fold, spec, args, time.monotonic()-1))
            self.assertFalse((args.output / "joint_scale.joblib").exists())


if __name__ == "__main__":
    unittest.main()
