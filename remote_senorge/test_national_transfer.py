import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

if __package__:
    from . import train_national_transfer as trainer
else:
    import train_national_transfer as trainer


def fixture(folder):
    folder.mkdir()
    rows = []
    for split, year, first_day in (("train", 2020, 1), ("selection", 2023, 1),
                                   ("calibration", 2023, 16), ("test", 2024, 1)):
        for day in range(first_day, first_day+15):
            for target in range(2):
                for source in range(2):
                    rows.append(dict(source=f"S{source}", target=f"{split if split != 'calibration' else 'selection'}{target}",
                                     time=f"{year}-12-{day:02d}T03:00:00+00:00",
                                     night=f"{year}-12-{day:02d}", split=split, block=str(target),
                                     source_c=float(source), target_c=-10.+target, forecast_delta_c=-8.,
                                     calm_clear=True))
    data = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    x = np.random.default_rng(42).normal(size=(len(rows), 39)).astype(np.float32)
    x[:, 0] = data["source_c"]
    data["x"] = x
    np.savez(folder / "training.npz", **data)
    manifest = dict(schema="national-transfer-training-v1", features=[f"f{i}" for i in range(39)],
                    terrain_count=16, regime_count=28, limitations=["Synthetic fixture"],
                    rows={s: int((data["split"] == s).sum()) for s in trainer.SPLITS})
    (folder / "manifest.json").write_text(json.dumps(manifest))
    receipts = {name: trainer.sha256(folder / name) for name in ("training.npz", "manifest.json")}
    (folder / "package_checksums.json").write_text(json.dumps(receipts))
    return data


class NationalTests(unittest.TestCase):
    def test_grouped_mask_and_permutation_invariance(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp) / "data")
            groups = trainer.grouped(data)
            self.assertEqual(groups["mask"].sum(1).tolist(), [2]*120)
            model = trainer.fusion_model(torch, 8).eval()
            x = torch.randn(3, 6, 39)
            mask = torch.tensor([[True, True, False, False, False, False]]*3)
            anchor = torch.randn(3, 6)
            with torch.inference_mode():
                a = model(x, anchor, mask)
                order = [1, 0, 5, 4, 3, 2]
                b = model(x[:, order], anchor[:, order], mask[:, order])
                torch.testing.assert_close(a, b)
                x[:, 2:] = 999
                torch.testing.assert_close(a, model(x, anchor, mask))

    def test_balancing_duplicate_source_mass(self):
        meta = dict(target=np.array(["A", "A", "B"]), night=np.array(["N", "N", "N"]),
                    block=np.array(["1", "1", "2"]))
        w = trainer.balanced_weights(meta)
        self.assertEqual(w[0]+w[1], w[2])

    def test_corruption_and_spatial_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            data = fixture(folder)
            trainer.load_data(folder)
            data["source"] = data["source"].astype("U40")
            data["source"][0] = data["target"][data["split"] == "test"][0]
            np.savez(folder / "training.npz", **data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                trainer.load_data(folder)
            checks = json.loads((folder / "package_checksums.json").read_text())
            checks["training.npz"] = trainer.sha256(folder / "training.npz")
            (folder / "package_checksums.json").write_text(json.dumps(checks))
            with self.assertRaisesRegex(ValueError, "leakage"):
                trainer.load_data(folder)

    def test_full_run_completed_resume_and_baseline_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            fixture(folder)
            args = trainer.parse_args(["--data", str(folder), "--output", str(Path(temp) / "results"),
                                       "--device", "cpu", "--widths", "8", "--seeds", "42",
                                       "--epochs", "2", "--patience", "2", "--jobs", "1", "--tree-rows", "100"])
            summary = trainer.run(args)
            self.assertEqual(len(summary["pair_validation"]), 8)
            self.assertEqual(len(summary["fusion_validation"]), 6)
            self.assertTrue(np.isfinite(summary["reports"]["fusion"]["test"]["rmse_c"]))
            resumed = trainer.run(args)
            self.assertEqual(resumed, summary)
            args.hours = 39
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                trainer.run(args)

    def test_durable_epoch_interruption_resume_equals_fresh(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            fixture(folder)
            base = ["--data", str(folder), "--device", "cpu", "--widths", "8", "--seeds", "42",
                    "--epochs", "2", "--patience", "2", "--jobs", "1", "--tree-rows", "100"]
            args = trainer.parse_args(base+["--output", str(Path(temp) / "resume")])
            save = trainer.atomic_torch_save

            def interrupted(torch, payload, path):
                save(torch, payload, path)
                if "epoch" in payload and payload["epoch"] == 1:
                    raise RuntimeError("Simulated interruption")

            with patch.object(trainer, "atomic_torch_save", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    trainer.run(args)
            with patch("builtins.print"):
                trainer.run(args)
                fresh = trainer.parse_args(base+["--output", str(Path(temp) / "fresh")])
                trainer.run(fresh)
            for config in trainer.trials([8], [42]):
                a = torch.load(args.output / (config["name"]+".pt"), weights_only=False)
                b = torch.load(fresh.output / (config["name"]+".pt"), weights_only=False)
                self.assertEqual(a["history"], b["history"])
                for key, value in a["best_state"].items():
                    torch.testing.assert_close(value, b["best_state"][key], rtol=0, atol=0)

    def test_perfect_minus10_transfer_has_zero_corrected_error(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            data = fixture(folder)
            data["source_c"] = data["target_c"]+10.
            data["x"][:, 0] = data["source_c"]
            np.savez(folder / "training.npz", **data)
            checks = json.loads((folder / "package_checksums.json").read_text())
            checks["training.npz"] = trainer.sha256(folder / "training.npz")
            (folder / "package_checksums.json").write_text(json.dumps(checks))
            args = trainer.parse_args(["--data", str(folder), "--output", str(Path(temp) / "result"),
                                      "--device", "cpu", "--widths", "8", "--seeds", "42",
                                      "--epochs", "1", "--patience", "1", "--jobs", "1", "--tree-rows", "100"])
            with patch("builtins.print"):
                summary = trainer.run(args)
            offset = next(c for c in summary["pair_validation"] if c["name"] == "constant_offset")
            self.assertLess(offset["best"], 1e-10)
            self.assertLess(summary["reports"]["pair"]["test"]["rmse_c"], 1e-10)


if __name__ == "__main__":
    unittest.main()
