import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

if __package__:
    from .footprint_model import TARGETS, create_model, masked_mse
    from .train_footprint import run, load_data
    from .train_remote import write_json, sha256
else:
    from footprint_model import TARGETS, create_model, masked_mse
    from train_footprint import run, load_data
    from train_remote import write_json, sha256


class FootprintTests(unittest.TestCase):
    def fixture(self, folder):
        rng = np.random.default_rng(42)
        n, pixels = 6, 8
        scenes = rng.normal(0, .1, (n, 2, 240, 240)).astype(np.float16)
        scenes[:, 1] = 1
        local = rng.normal(0, .1, (n, 2, 128, 128)).astype(np.float16)
        local[:, 1] = 1
        features = rng.normal(0, 1, (n, 25, pixels, pixels)).astype(np.float32)
        labels = rng.uniform(.1, 2, (n, 6, pixels, pixels)).astype(np.float32)
        labels[:, :, pixels//2, pixels//2] = 0
        mask = np.ones_like(labels, dtype=bool)
        mask[:, :, 0, 0] = False
        for name, array in dict(scenes=scenes, local10=local, features=features,
                                labels=labels, label_mask=mask).items():
            np.save(folder / f"{name}.npy", array)
        np.savez(folder / "references.npz", refs=np.arange(n), split=np.repeat([0, 1, 2], 2),
                 xy=np.array([[50000+i//2*100000+i%2*1000, 6650000] for i in range(n)]))
        names = [p.name for p in folder.iterdir()]
        write_json(folder / "footprint_manifest.json", dict(
            targets=TARGETS, pixels=pixels, limitations=["synthetic"],
            checksums={name: sha256(folder / name) for name in names}))

    def test_map_shape_reference_conditioning_and_gradient(self):
        import torch
        torch.manual_seed(42)
        torch.set_num_threads(2)
        for fine in (False, True):
            model = create_model(torch, width=8, fine=fine, pixels=8)
            scene = torch.randn(2, 2, 240, 240)
            local = torch.randn(2, 2, 128, 128)
            features = torch.randn(2, 25, 8, 8)
            predicted = model(scene, local, features)
            self.assertEqual(tuple(predicted.shape), (2, 6, 8, 8))
            self.assertTrue((predicted > 0).all())
            mask = torch.ones_like(predicted, dtype=torch.bool)
            mask[:, :, 0, 0] = False
            actual = torch.ones_like(predicted)
            actual[:, :, 0, 0] = float("nan")
            loss = masked_mse(torch, predicted, actual, mask)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.stem[0].weight.grad)
            self.assertTrue(torch.isfinite(model.stem[0].weight.grad).all())
            if fine:
                self.assertIsNotNone(model.fine_encoder[0].weight.grad)
            with self.assertRaises(ValueError):
                masked_mse(torch, predicted, actual, torch.zeros_like(mask))

    def test_actual_cpu_experiments_selection_and_saved_models(self):
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            args = argparse.Namespace(data=root, output=root / "results", device="cpu", hours=1.,
                                      epochs=2, batch_size=2, patience=2, jobs=2,
                                      widths=[8], seeds=[42], learning_rate=.001, skip_baseline=True)
            summary = run(args)
            self.assertEqual(len(summary["trials"]), 2)
            self.assertEqual(list(summary["test"]), TARGETS)
            self.assertEqual(summary["selected"]["validation_mse"],
                             min(t["validation_mse"] for t in summary["trials"]))
            self.assertEqual(summary["status"], "complete")
            selected = summary["selected"]
            model = create_model(torch, selected["width"], selected["fine"], pixels=8)
            model.load_state_dict(torch.load(args.output / "selected_model.pt", weights_only=True))
            self.assertTrue((args.output / "selected_test_maps.npz").is_file())
            for name, digest in summary["checksums"].items():
                self.assertEqual(sha256(args.output / name), digest)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_hash_and_spatial_leakage_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            arrays, _, _ = load_data(root)
            for a in arrays.values():
                a._mmap.close()
            with np.load(root / "references.npz") as stored:
                refs = {name: stored[name] for name in stored.files}
            refs["xy"][2] = [51000, 6650000]
            np.savez(root / "references.npz", **refs)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_data(root)
            manifest = json.loads((root / "footprint_manifest.json").read_text())
            manifest["checksums"]["references.npz"] = sha256(root / "references.npz")
            write_json(root / "footprint_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_data(root)


if __name__ == "__main__":
    unittest.main()
