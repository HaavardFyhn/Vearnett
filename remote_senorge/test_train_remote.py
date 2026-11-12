import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

try:
    from . import train_remote as trainer
except ImportError:
    import train_remote as trainer


class RemoteTests(unittest.TestCase):
    def test_checkpoint_retries_windows_file_lock_and_remains_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.json"
            trainer.write_json(path, dict(completed=10))
            replace = Path.replace
            error = PermissionError("locked")
            error.winerror = 5
            attempts = []

            def locked_once(pending, target):
                attempts.append(target)
                self.assertEqual(json.loads(path.read_text()), dict(completed=10))
                if len(attempts) == 1:
                    raise error
                return replace(pending, target)

            with patch.object(Path, "replace", autospec=True, side_effect=locked_once), \
                    patch.object(trainer.time, "sleep") as sleep, \
                    self.assertLogs(trainer.__name__, level="WARNING"):
                trainer.write_json(path, dict(completed=20))
            self.assertEqual(json.loads(path.read_text()), dict(completed=20))
            sleep.assert_called_once_with(.1)

    def test_checkpoint_persistent_lock_and_other_errors_surface(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.json"
            trainer.write_json(path, dict(completed=10))
            locked = PermissionError("locked")
            locked.winerror = 32
            for error, expected in ((locked, 8), (OSError("disk failure"), 1)):
                with patch.object(Path, "replace", side_effect=error) as replace, \
                        patch.object(trainer.time, "sleep"), \
                        patch.object(trainer.logging.getLogger(trainer.__name__), "warning"):
                    with self.assertRaises(OSError):
                        trainer.write_json(path, dict(completed=20))
                self.assertEqual(replace.call_count, expected)
                self.assertEqual(json.loads(path.read_text()), dict(completed=10))

    def fixture(self, folder):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(120, 25)).astype("float32")
        y = (np.abs(x[:, 0]) + 0.3*np.abs(x[:, 14])).astype("float32")
        test = np.arange(120) >= 90
        np.savez_compressed(folder / "training.npz", x=x, y=y, train=~test, test=test,
                            block_1=test.astype(int), block_2=test.astype(int),
                            grid_id_1=np.arange(120), grid_id_2=np.arange(120)+120)
        trainer.write_json(folder / "manifest.json", dict(
            dataset_sha256=trainer.sha256(folder / "training.npz"),
            features=[f"feature_{i}" for i in range(25)], base_feature_count=13,
            training_source="synthetic test", limitations=["synthetic"],
        ))

    def test_hash_and_spatial_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            self.fixture(p)
            data, meta = trainer.load_data(p)
            data["block_1"][:] = 1
            np.savez(p / "training.npz", **data)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                trainer.load_data(p)
            meta["dataset_sha256"] = trainer.sha256(p / "training.npz")
            trainer.write_json(p / "manifest.json", meta)
            with self.assertRaisesRegex(ValueError, "leakage"):
                trainer.load_data(p)

    def test_cpu_training_resume_and_holdout_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            self.fixture(p)
            args = argparse.Namespace(data=p, output=p / "results", backends=["extratrees"],
                                      device="cpu", trees=4, boost_rounds=4, jobs=1)
            result = trainer.run(args)
            self.assertEqual(len(result["experiments"]), 2)
            experiment = result["experiments"]["extratrees_with_basins"]
            with np.load(args.output / "extratrees_with_basins_holdout.npz") as held:
                self.assertAlmostEqual(np.sqrt(np.mean((held["actual"]-held["predicted"])**2)),
                                       experiment["validation"]["rmse"])
            receipt = args.output / "extratrees_with_basins.json"
            timestamp = receipt.stat().st_mtime_ns
            with self.assertRaises(FileExistsError):
                trainer.run(args)
            self.assertEqual(json.loads((args.output / "status.json").read_text())["state"], "complete")
            (args.output / "summary.json").unlink()
            trainer.run(args)
            self.assertEqual(timestamp, receipt.stat().st_mtime_ns)
            (args.output / "summary.json").unlink()
            with (args.output / "extratrees_with_basins_holdout.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                trainer.run(args)

    def test_invalid_training_values_and_endpoints(self):
        for field in ("x", "y", "block_1"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                p = Path(temp)
                self.fixture(p)
                data, meta = trainer.load_data(p)
                if field == "block_1":
                    data[field] = data[field][:-1]
                elif field == "y":
                    data[field][0] = -1
                else:
                    data[field][0, 0] = np.nan
                np.savez(p / "training.npz", **data)
                meta["dataset_sha256"] = trainer.sha256(p / "training.npz")
                trainer.write_json(p / "manifest.json", meta)
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    trainer.load_data(p)

    def test_distance_bin_scores_include_sparse_and_empty_bins(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            self.fixture(p)
            data, meta = trainer.load_data(p)
            meta["features"][0] = "distance_m"
            data["x"][:, 0] = np.linspace(1000, 100000, len(data["y"]))
            np.savez(p / "training.npz", **data)
            meta["dataset_sha256"] = trainer.sha256(p / "training.npz")
            trainer.write_json(p / "manifest.json", meta)
            args = argparse.Namespace(data=p, output=p / "results", backends=["extratrees"],
                                      device="cpu", trees=4, boost_rounds=4, jobs=1)
            result = trainer.run(args)
            bins = result["experiments"]["extratrees_with_basins"]["validation"]["by_distance_km"]
            self.assertEqual(sum(b["pairs"] for b in bins), 30)
            self.assertIsNone(bins[0]["rmse"])
            with np.load(args.output / "extratrees_with_basins_holdout.npz") as held:
                mask = data["x"][data["test"], 0] >= 90000
                self.assertEqual(bins[-1]["pairs"], int(mask.sum()))
                self.assertAlmostEqual(bins[-1]["rmse"], np.sqrt(np.mean(
                    (held["actual"][mask]-held["predicted"][mask])**2)))

    def test_cuda_fallback_is_failure_not_success(self):
        model = Mock()
        model.get_booster().save_config.return_value = json.dumps(
            {"learner": {"generic_param": {"device": "cpu"}}})
        with self.assertRaisesRegex(RuntimeError, "refusing CPU fallback"):
            trainer.verify_device(model, "xgboost", "cuda:0")
        model.get_booster().save_config.return_value = json.dumps(
            {"learner": {"generic_param": {"device": "cuda:0"}}})
        self.assertEqual(trainer.verify_device(model, "xgboost", "cuda:0"), "cuda:0")


if __name__ == "__main__":
    unittest.main()
