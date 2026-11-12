import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

if __package__:
    from . import train_seasonal_terrain as trainer
    from .train_remote import sha256, write_json
else:
    import train_seasonal_terrain as trainer
    from train_remote import sha256, write_json


class SeasonalTerrainTests(unittest.TestCase):
    def fixture(self, folder):
        rng = np.random.default_rng(42)
        xy = np.array([[block*100000 + offset, 50000] for block in range(10)
                       for offset in (50000, 51000)], dtype=float)
        pairs = np.repeat(np.arange(20).reshape(10, 2), 4, axis=0)
        split = np.repeat([0]*6+[1]*2+[2]*2, 4)
        x = rng.normal(size=(len(pairs), 25)).astype(np.float32)
        x[:, 0], x[:, 5], x[:, 6] = 1000, 1000, 0
        y = (.5 + np.abs(x[:, 1:5])*.1).astype(np.float32)
        np.savez(folder / "seasonal_training.npz", x=x, y=y, pairs=pairs, xy=xy, split=split)
        np.save(folder / "terrain.npy", rng.normal(0, .05, size=(20, 3, 64, 64)).astype(np.float16))
        np.save(folder / "terrain_valid.npy", np.ones((20, 3, 64, 64), dtype=np.uint8))
        manifest = dict(targets=["winter_tg", "summer_tg", "winter_tn", "summer_tn"],
                        patch_radius_m=16500, limitations=["synthetic"],
                        checksums={name: sha256(folder / name) for name in
                                   ("seasonal_training.npz", "terrain.npy", "terrain_valid.npy")})
        write_json(folder / "seasonal_manifest.json", manifest)
        return manifest

    def test_exact_endpoint_symmetry_and_positive_four_outputs(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        for mode in ("terrain", "combined"):
            net = trainer.create_model(mode, torch).eval()
            a, b = torch.randn(2, 3, 2, 64, 64), torch.randn(2, 3, 2, 64, 64)
            x = torch.randn(2, 25)
            with torch.no_grad():
                actual, reverse = net(a, b, x), net(b, a, x)
            self.assertEqual(tuple(actual.shape), (2, 4))
            self.assertTrue(torch.equal(actual, reverse))
            self.assertTrue((actual > 0).all())

    def test_checksums_and_spatial_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            manifest = self.fixture(p)
            data, _, _, _ = trainer.load_dataset(p)
            data["xy"][-2:] = data["xy"][:2]
            np.savez(p / "seasonal_training.npz", **data)
            with self.assertRaisesRegex(ValueError, "Checksum"):
                trainer.load_dataset(p)
            manifest["checksums"]["seasonal_training.npz"] = sha256(p / "seasonal_training.npz")
            write_json(p / "seasonal_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Spatial leakage"):
                trainer.load_dataset(p)

    def test_patch_boundary_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            manifest = self.fixture(p)
            data, _, _, _ = trainer.load_dataset(p)
            data["xy"][0, 0] = 1000
            np.savez(p / "seasonal_training.npz", **data)
            manifest["checksums"]["seasonal_training.npz"] = sha256(p / "seasonal_training.npz")
            write_json(p / "seasonal_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "support crosses"):
                trainer.load_dataset(p)

    def test_all_three_experiments_cpu_and_completed_experiment_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            self.fixture(p)
            args = argparse.Namespace(data=p, output=p / "results", epochs=2, patience=2,
                                      batch_size=8, jobs=1, learning_rate=.001,
                                      boost_rounds=4, device="cpu", modes=["xgboost", "terrain", "combined"])
            result = trainer.train(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(set(result["experiments"]), set(args.modes))
            for mode in args.modes:
                with np.load(args.output / f"{mode}_holdout.npz") as held:
                    rmse = np.sqrt(np.mean((held["actual"][:, 0]-held["predicted"][:, 0])**2))
                    self.assertAlmostEqual(rmse, result["experiments"][mode]["test"]["winter_tg"]["rmse"],
                                           places=6)
            weights = args.output / "combined.pt"
            timestamp = weights.stat().st_mtime_ns
            with self.assertRaises(FileExistsError):
                trainer.train(args)
            (args.output / "summary.json").unlink()
            resumed = trainer.train(args)
            self.assertEqual(resumed["experiments"], result["experiments"])
            self.assertEqual(timestamp, weights.stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
