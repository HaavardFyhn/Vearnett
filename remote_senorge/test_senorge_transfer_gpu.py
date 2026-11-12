import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

if __package__:
    from . import train_senorge_transfer_gpu as trainer
else:
    import train_senorge_transfer_gpu as trainer


def fixture(folder):
    folder.mkdir()
    rows = []
    for split, year in (("train", 2000), ("selection", 2017), ("calibration", 2019), ("test", 2023)):
        for block in range(6 if split == "train" else 2):
            for day in range(1, 11):
                for variable in (0, 1):
                    rows.append(dict(split=split, block=f"{split}{block}", source=f"{split}S{block}",
                                     target=f"{'selection' if split == 'calibration' else split}T{block}",
                                     date=f"{year}-01-{day:02d}", variable=variable,
                                     source_c=float(day-10), target_c=float(day-20)))
    data = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    x = np.random.default_rng(42).normal(size=(len(rows), len(trainer.FEATURES))).astype(np.float32)
    x[:, 0], x[:, 3] = data["source_c"], data["variable"]
    data["x"] = x
    np.savez(folder / "training.npz", **data)
    manifest = dict(schema="senorge-transfer-gpu-v1", features=trainer.FEATURES,
                    rows={split: int((data["split"] == split).sum()) for split in
                          ("train", "selection", "calibration", "test")})
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "package_checksums.json").write_text(json.dumps({
        name: trainer.sha256(folder / name) for name in ("training.npz", "manifest.json")}))
    return data


class GPUTransferTests(unittest.TestCase):
    def test_perfect_offset_and_zero_mse(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp) / "data")
            for name in ("offset", "ridge"):
                model = trainer.fit_baseline(data, name)
                prediction = trainer.baseline_predict(model, data)
                self.assertLess(trainer.score(data, prediction)["rmse_c"], 1e-10)
            self.assertEqual(trainer.score(data, data["source_c"])["rmse_c"], 10)

    def test_full_run_two_stages_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, out = Path(temp) / "data", Path(temp) / "results"
            fixture(folder)
            args = trainer.parse_args(["--data", str(folder), "--output", str(out), "--device", "cpu",
                                       "--epochs", "2", "--widths", "8", "--seeds", "42", "--jobs", "1"])
            with patch("builtins.print"):
                report = trainer.run(args)
                with patch.object(trainer, "train_network", side_effect=AssertionError("Completed run refit")):
                    self.assertEqual(report, trainer.run(args))
            self.assertIn(report["selected"]["mean"], ("offset", "ridge"))
            self.assertEqual(report["selected"]["error"], "pooled_mse")
            self.assertLess(report["assessments"]["test"]["temperature"]["rmse_c"], 1e-10)
            self.assertLess(report["assessments"]["test"]["error_model"]["predicted_pooled_rmse_c"], 1e-10)
            self.assertTrue((out / "test_predictions.csv").exists())
            args.batch_size += 1
            with self.assertRaisesRegex(ValueError, "Changed"):
                trainer.run(args)

    def test_validation_only_does_not_score_assessment_or_test(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, out = Path(temp) / "data", Path(temp) / "results"
            fixture(folder)
            args = trainer.parse_args(["--data", str(folder), "--output", str(out), "--device", "cpu",
                                       "--epochs", "1", "--widths", "8", "--seeds", "42", "--jobs", "1",
                                       "--validation-only"])
            with patch("builtins.print"), patch.object(trainer, "error_report", side_effect=AssertionError("Test scored")):
                report = trainer.run(args)
            self.assertEqual(report["assessments"], {})
            self.assertFalse((out / "test_predictions.csv").exists())

    def test_neural_epoch_resume_exact_and_budget_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp) / "data")
            x, y, w = data["x"], data["target_c"]-data["source_c"], trainer.balanced_weights(data)
            args = trainer.parse_args(["--device", "cpu", "--jobs", "1", "--epochs", "2", "--batch-size", "32",
                                       "--output", str(Path(temp) / "out")])
            args.output.mkdir()
            config = dict(width=8, seed=42, error_stage=False)
            torch.set_num_threads(1)
            original = trainer.atomic_torch_save

            def interrupt(torch, saved, path):
                original(torch, saved, path)
                if saved["epoch"] == 1:
                    raise RuntimeError("Interruption")

            with patch.object(trainer, "atomic_torch_save", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "Interruption"):
                    trainer.train_network(torch, x, y, w, None, config, args, "resume", time.monotonic()+120, 2)
            with patch("builtins.print"):
                resumed = trainer.train_network(torch, x, y, w, None, config, args, "resume", time.monotonic()+120, 2)
                fresh = trainer.train_network(torch, x, y, w, None, config, args, "fresh", time.monotonic()+120, 2)
            np.testing.assert_array_equal(
                trainer.infer(torch, resumed["model"], x, resumed["mean"], resumed["std"], args),
                trainer.infer(torch, fresh["model"], x, fresh["mean"], fresh["std"], args))
            self.assertIsNone(trainer.train_network(torch, x, y, w, None, config, args, "expired", time.monotonic()-1, 2))

    def test_training_normalization_not_held_target_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp) / "data")
            model = trainer.fit_baseline(trainer.subset(data, data["split"] == "train"), "ridge")
            altered = {key: value.copy() for key, value in data.items()}
            altered["target_c"][altered["split"] == "test"] += 100
            again = trainer.fit_baseline(trainer.subset(altered, altered["split"] == "train"), "ridge")
            np.testing.assert_array_equal(model["mean"], again["mean"])
            np.testing.assert_array_equal(model["model"].coef_, again["model"].coef_)

    def test_partial_oof_receipt_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, out = Path(temp) / "data", Path(temp) / "results"
            fixture(folder)
            args = trainer.parse_args(["--data", str(folder), "--output", str(out), "--device", "cpu",
                                       "--epochs", "1", "--widths", "8", "--seeds", "42", "--jobs", "1",
                                       "--validation-only"])
            with patch("builtins.print"):
                trainer.run(args)
            (out / "report.json").unlink()
            receipt = json.loads((out / "oof_receipt.json").read_text())
            receipt["sha256"] = "invalid"
            (out / "oof_receipt.json").write_text(json.dumps(receipt))
            with patch("builtins.print"), self.assertRaisesRegex(ValueError, "OOF checksum"):
                trainer.run(args)


if __name__ == "__main__":
    unittest.main()
