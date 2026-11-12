"""Portable tests; synthetic observations never imply real forecast skill."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

if __package__:
    from . import train_valley_neural as training
else:
    import train_valley_neural as training


class NeuralTests(unittest.TestCase):
    def fixture(self, folder):
        folder.mkdir()
        rng = np.random.default_rng(123)
        n = 72
        x = rng.normal(size=(n, 40)).astype(np.float32)
        source_c = x[:, 0]*3-6
        target_c = source_c+x[:, 3]*.5
        masks = {}
        start = 0
        for name, count in (("train", 24), ("selection", 12), ("calibration", 12), ("test", 24)):
            masks[name] = np.zeros(n, bool)
            masks[name][start:start+count] = True
            start += count
        night = []
        timestamps = []
        for year, month, days in ((2022, 12, 24), (2023, 12, 12), (2024, 1, 12), (2024, 12, 24)):
            for day in range(1, days+1):
                stamp = pd.Timestamp(year=year, month=month, day=day, hour=3, tz="UTC")
                night.append(stamp.strftime("%Y-%m-%d"))
                timestamps.append(stamp.isoformat())
        data = dict(x=x, y=(target_c-source_c).astype(np.float32), source_c=source_c,
                    target_c=target_c, forecast_delta_c=x[:, 3]*.2,
                    source=np.array(["SOURCE"]*n), target=np.array(
                        ["TRAIN"]*24+["VALIDATION"]*24+["TEST"]*24),
                    night=np.array(night), time=np.array(timestamps),
                    calm_clear=np.arange(n) % 4 == 0, **masks)
        path = folder / "training.npz"
        np.savez_compressed(path, **data)
        manifest = dict(schema="valley-neural-transfer-v1", data_sha256=training.sha256(path),
                        terrain_count=16, regime_count=28, features=[f"feature{i}" for i in range(40)],
                        rows={name: int(mask.sum()) for name, mask in masks.items()},
                        limitations=["Exploratory synthetic test."])
        training.write_json(folder / "manifest.json", manifest)
        return data, manifest

    def args(self, folder, output):
        return training.parse_args([
            "--data", str(folder), "--output", str(output), "--device", "cpu", "--hours", "1",
            "--epochs", "4", "--patience", "3", "--widths", "8", "--seeds", "42", "--jobs", "1",
            "--batch-size", "16"])

    def test_all_architectures_forward_and_backward_finite(self):
        with tempfile.TemporaryDirectory() as temp:
            data, manifest = self.fixture(Path(temp) / "data")
            for config in training.trials([42], [8]):
                model = training.create_model(torch, config, manifest)
                x = torch.from_numpy(data["x"][:4])
                prediction = model(x)
                self.assertEqual(prediction.shape, (4,))
                np.testing.assert_array_equal(prediction.detach().numpy(), np.zeros(4))
                (prediction-1).square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertEqual(len(training.trials([42, 123, 2026], [32, 64, 128])), 63)

    def test_training_only_normalization_and_all_results_persist(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, output = Path(temp) / "data", Path(temp) / "result"
            data, manifest = self.fixture(folder)
            args = self.args(folder, output)
            with patch("builtins.print"):
                report = training.run(args)
            with np.load(output / "normalization.npz") as saved:
                np.testing.assert_allclose(saved["mean"], data["x"][data["train"]].mean(axis=0), atol=1e-6)
            self.assertEqual(len(report["trial_validation"]), 8)
            self.assertIn(report["selected"]["config"]["name"], [v["config"]["name"] for v in report["trial_validation"]])
            self.assertEqual(report["test"]["nights"], 24)
            self.assertEqual(report["calibration_nights"], 12)
            for name, checksum in report["checksums"].items():
                self.assertEqual(training.sha256(output / name), checksum)
            csv = pd.read_csv(output / "selected_test_predictions.csv")
            self.assertEqual(set(csv.target), {"TEST"})
            with patch("builtins.print"), patch.object(training, "train_trial", side_effect=AssertionError("retraining")):
                self.assertEqual(training.run(args)["selected"], report["selected"])
            with (output / "selected_test_predictions.csv").open("a") as file:
                file.write("corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"), patch("builtins.print"):
                training.run(args)

    def test_resume_reproduces_uninterrupted_completed_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            self.fixture(folder)
            output, fresh = Path(temp) / "resume", Path(temp) / "fresh"
            args = self.args(folder, output)
            original = training.atomic_torch_save
            calls = 0

            def interrupt(torch_module, payload, path):
                nonlocal calls
                original(torch_module, payload, path)
                calls += 1
                if calls == 2:
                    raise InterruptedError("simulated interruption after durable epoch")

            with patch.object(training, "atomic_torch_save", side_effect=interrupt), patch("builtins.print"):
                with self.assertRaises(InterruptedError):
                    training.run(args)
            with patch("builtins.print"):
                resumed = training.run(args)
                complete = training.run(self.args(folder, fresh))
            self.assertEqual(resumed["selected"]["config"], complete["selected"]["config"])
            pd.testing.assert_frame_equal(pd.read_csv(output / "selected_test_predictions.csv"),
                                          pd.read_csv(fresh / "selected_test_predictions.csv"))
            for name in (v["checkpoint"] for v in resumed["trial_validation"] if "checkpoint" in v):
                a = torch.load(output / name, weights_only=False)
                b = torch.load(fresh / name, weights_only=False)
                self.assertEqual(a["history"], b["history"])
                for key in a["best_state"]:
                    torch.testing.assert_close(a["best_state"][key], b["best_state"][key], rtol=0, atol=0)

    def test_guards_reject_leakage_corruption_changed_run_and_cuda_fallback(self):
        for kind in ("source", "night", "label", "checksum", "package", "time"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp) / "data"
                data, manifest = self.fixture(folder)
                if kind == "source":
                    data["source"][0] = "TEST"
                elif kind == "night":
                    data["night"][-1] = data["night"][0]
                elif kind == "time":
                    data["time"][-1] = "2023-12-24T03:00:00+00:00"
                elif kind == "label":
                    data["y"][0] = 999
                np.savez_compressed(folder / "training.npz", **data)
                if kind != "checksum":
                    manifest["data_sha256"] = training.sha256(folder / "training.npz")
                else:
                    manifest["data_sha256"] = "wrong"
                training.write_json(folder / "manifest.json", manifest)
                if kind == "package":
                    training.write_json(folder / "package_checksums.json", {"manifest.json": "bad"})
                with self.assertRaises(ValueError):
                    training.load_data(folder)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            self.fixture(folder)
            args = self.args(folder, Path(temp) / "out")
            args.device = "cuda:0"
            with patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    training.run(args)
            args.device = "cpu"
            with patch("builtins.print"):
                training.run(args)
            args.epochs = 5
            with patch("builtins.print"), self.assertRaisesRegex(ValueError, "configuration"):
                training.run(args)

    def test_risk_weights_do_not_depend_on_holdout_values(self):
        with tempfile.TemporaryDirectory() as temp:
            data, _ = self.fixture(Path(temp) / "data")
            before = training.sample_weights(data, 3.)
            changed = copy.deepcopy(data)
            changed["target_c"][~data["train"]] = -999
            changed["calm_clear"][~data["train"]] = True
            np.testing.assert_array_equal(before, training.sample_weights(changed, 3.))
            self.assertAlmostEqual(float(before.mean()), 1., places=6)
            prediction = data["target_c"].copy()
            self.assertEqual(training.radius(data, prediction), 0.)

    def test_budget_finishes_without_fabricating_trials(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            self.fixture(folder)
            args = self.args(folder, Path(temp) / "out")
            args.hours = 1e-12
            with patch("builtins.print"):
                report = training.run(args)
            self.assertEqual(len(report["trial_validation"]), 1)
            self.assertEqual(report["selected"]["config"]["name"], "forecast_transfer")
            self.assertTrue(report["budget_exhausted"])
            self.assertFalse((args.output / "selected_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
