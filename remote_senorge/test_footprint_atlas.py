import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

if __package__:
    from .footprint_atlas import ABSOLUTE_FIELDS, AtlasDataset, SPATIAL_FIELDS
    from .footprint_model import TARGETS, create_model, masked_mse
    from .train_footprint import run, load_data, normalization
    from .train_remote import write_json, sha256
else:
    from footprint_atlas import ABSOLUTE_FIELDS, AtlasDataset, SPATIAL_FIELDS
    from footprint_model import TARGETS, create_model, masked_mse
    from train_footprint import run, load_data, normalization
    from train_remote import write_json, sha256


class AtlasTests(unittest.TestCase):
    def fixture(self, root, absolute=True):
        table = np.zeros((6, 16), float)
        table[:, 0] = [75000, 76000, 225000, 226000, 375000, 376000]
        table[:, 1] = 6675000
        table[:, 2] = [500, 600, 1500, 1600, 2500, 2600]
        table[:, 3] = [1000, 2000, 11000, 12000, 21000, 22000]
        table[:, 4] = [3000, 4000, 13000, 14000, 23000, 24000]
        table[:, 6:] = 1
        np.save(root / "grid_table.npy", table)
        shape = (830, 3840)
        terrain = np.lib.format.open_memmap(root / "terrain_height.npy", mode="w+",
                                           dtype=np.float32, shape=shape)
        terrain[:] = .6
        terrain[:, 100:110] = .7
        terrain.flush()
        terrain._mmap.close()
        np.save(root / "terrain_valid.npy", np.ones(shape, np.uint8))
        spatial = np.zeros((len(SPATIAL_FIELDS), 83, 384), np.float16)
        spatial[0] = 1
        spatial[1, 41, 41] = .2
        np.save(root / "spatial10.npy", spatial)
        mapping = np.full((6, 64, 64), -1, np.int32)
        for i in range(6):
            mapping[i, 32, 32] = i
            mapping[i, 32, 33 if i % 2 == 0 else 31] = i+1 if i % 2 == 0 else i-1
        np.savez(root / "references.npz", refs=np.arange(6, dtype=np.int32),
                 split=np.repeat([0, 1, 2], 2), xy=table[:, :2], mapping=mapping)
        labels = np.zeros((6, 6, 64, 64), np.float32)
        labels[:] = np.linspace(.1, .6, 6)[None, :, None, None]
        labels[:, :, 32, 32] = 0
        np.save(root / "labels.npy", labels)
        np.save(root / "label_mask.npy", np.repeat((mapping >= 0)[:, None], 6, axis=1))
        manifest = dict(schema=2, architecture="reference-resunet-spatial10-absolute-v2" if absolute
                        else "reference-resunet-spatial10-v2",
                        targets=TARGETS, pixels=64, block_size_m=150000,
                        atlas_origin=[34000, 6716000], spatial_fields=SPATIAL_FIELDS,
                        limitations=["synthetic"], geography={},
                        checksums={p.name: sha256(p) for p in root.iterdir()})
        if absolute:
            manifest["absolute_features"] = ABSOLUTE_FIELDS
        write_json(root / "footprint_manifest.json", manifest)
        return manifest

    def test_lazy_alignment_features_and_handle_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            data, refs, _ = load_data(root)
            scene = data["scenes"][0]
            fine = data["local10"][0]
            self.assertEqual(scene.shape, (2, 800, 800))
            self.assertAlmostEqual(float(scene[0, 100, 100]), .1, places=6)
            self.assertEqual(fine.shape, (len(SPATIAL_FIELDS), 80, 80))
            self.assertAlmostEqual(float(fine[1, 40, 40]), .2, places=3)
            features = data["features"][0]
            self.assertEqual(features.shape, (31, 64, 64))
            np.testing.assert_array_equal(features[25:, 32, 33], [500, 600, 3000, 4000, 1000, 2000])
            np.testing.assert_array_equal(features[25:, 0, 0], 0)
            np.testing.assert_array_equal(data["features"][1][25:, 32, 31],
                                          [600, 500, 4000, 3000, 2000, 1000])
            other_region = data["features"][2]
            np.testing.assert_array_equal(other_region[:25, 32, 33], features[:25, 32, 33])
            self.assertFalse(np.array_equal(other_region[25:, 32, 33], features[25:, 32, 33]))
            self.assertEqual(features[0, 32, 33], 1000)
            self.assertEqual(features[0, 0, 0], 0)
            self.assertEqual(refs["mapping"][0, 32, 32], 0)
            mean, _ = normalization(data, [0, 1])
            self.assertEqual(mean[0], 500)
            np.testing.assert_array_equal(mean[25:], [550, 550, 3500, 3500, 1500, 1500])
            data["_atlas"].close()
            for name in ("terrain_height.npy", "terrain_valid.npy", "spatial10.npy", "labels.npy"):
                (root / name).rename(root / (name+".released"))

    def test_spatial_encoder_preserves_location_and_trains(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = create_model(torch, 8, True, 8, len(SPATIAL_FIELDS), 31)
        local = torch.randn(2, len(SPATIAL_FIELDS), 24, 24, requires_grad=True)
        scene, features = torch.randn(2, 2, 240, 240), torch.randn(2, 31, 8, 8, requires_grad=True)
        predicted = model(scene, local, features)
        self.assertEqual(tuple(model.fine_encoder(local).shape), (2, 32, 24, 24))
        loss = masked_mse(torch, predicted, torch.ones_like(predicted), torch.ones_like(predicted, dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(local.grad).all())
        self.assertGreater(float(local.grad.abs().sum()), 0)
        self.assertGreater(float(features.grad[:, 25:].abs().sum()), 0)
        with torch.no_grad():
            shifted = model(scene, torch.flip(local, [-1]), features)
            self.assertGreater(float((predicted-shifted).abs().max()), 1e-6)

    def test_expanded_actual_cpu_run_and_selected_checkpoint(self):
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            args = argparse.Namespace(data=root, output=root / "results", device="cpu", hours=1.,
                                      epochs=1, batch_size=2, patience=2, jobs=2,
                                      widths=[8], seeds=[42], learning_rate=.001, skip_baseline=True)
            summary = run(args)
            self.assertEqual(summary["architecture"], "reference-resunet-spatial10-absolute-v2")
            self.assertEqual(len(summary["trials"]), 1)
            self.assertEqual(len(summary["regional_test"]), 1)
            self.assertEqual(summary["spatial_channels"], len(SPATIAL_FIELDS))
            self.assertEqual(summary["pair_channels"], 31)
            model = create_model(torch, 8, True, 64, len(SPATIAL_FIELDS), 31)
            model.load_state_dict(torch.load(args.output / "selected_model.pt", weights_only=True))
            for name, digest in summary["checksums"].items():
                self.assertEqual(sha256(args.output / name), digest)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_legacy_relative_only_atlas_remains_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, absolute=False)
            data, _, _ = load_data(root)
            try:
                self.assertEqual(data["features"][0].shape, (25, 64, 64))
                mean, std = normalization(data, [0, 1])
                self.assertEqual(mean.shape, (25,))
                self.assertEqual(std.shape, (25,))
            finally:
                data["_atlas"].close()

    def test_absolute_feature_schema_rejects_reordering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            manifest["absolute_features"] = list(reversed(ABSOLUTE_FIELDS))
            write_json(root / "footprint_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "absolute endpoint feature schema"):
                AtlasDataset(root)

    def test_hash_duplicate_reference_and_leakage_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            with np.load(root / "references.npz") as file:
                refs = {key: file[key] for key in file.files}
            refs["split"][1] = 1
            np.savez(root / "references.npz", **refs)
            with self.assertRaisesRegex(ValueError, "checksum"):
                AtlasDataset(root)
            manifest["checksums"]["references.npz"] = sha256(root / "references.npz")
            write_json(root / "footprint_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "partition|leakage"):
                AtlasDataset(root)
            refs["split"] = np.repeat([0, 1, 2], 2)
            refs["refs"][1] = refs["refs"][0]
            np.savez(root / "references.npz", **refs)
            manifest["checksums"]["references.npz"] = sha256(root / "references.npz")
            write_json(root / "footprint_manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                AtlasDataset(root)

    def test_expanded_six_xgboost_baselines_use_matching_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            args = argparse.Namespace(data=root, output=root / "results", device="cpu", hours=1.,
                                      epochs=1, batch_size=2, patience=2, jobs=2,
                                      widths=[8], seeds=[42], learning_rate=.001, skip_baseline=False)
            summary = run(args)
            self.assertEqual(list(summary["xgboost_test"]), TARGETS)
            for record in summary["xgboost_test"].values():
                self.assertEqual(record["sampling"]["used_by_split"], [2, 2, 2])
                self.assertEqual(record["sampling"]["available_by_split"], [2, 2, 2])
                self.assertLess(record["rmse"], 1e-6)

    def test_interrupted_epoch_resumes_model_optimizer_and_history(self):
        if __package__:
            from . import train_footprint as trainer
        else:
            import train_footprint as trainer
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            args = argparse.Namespace(data=root, output=root / "results", device="cpu", hours=1.,
                                      epochs=2, batch_size=2, patience=2, jobs=2,
                                      widths=[8], seeds=[42], learning_rate=.001, skip_baseline=True)
            save = trainer.atomic_save

            def interrupted(torch, path, value):
                save(torch, path, value)
                if path.name.endswith("_resume.pt"):
                    raise InterruptedError("synthetic interruption after saved epoch")

            with patch.object(trainer, "atomic_save", side_effect=interrupted):
                with self.assertRaises(InterruptedError):
                    run(args)
            manifest_path = root / "footprint_manifest.json"
            original = json.loads(manifest_path.read_text())
            changed = dict(original)
            changed.pop("absolute_features")
            changed["architecture"] = "reference-resunet-spatial10-v2"
            write_json(manifest_path, changed)
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                run(args)
            write_json(manifest_path, original)
            summary = run(args)
            self.assertEqual(summary["trials"][0]["epochs"], 2)
            history = json.loads((args.output / (summary["selected"]["name"]+"_history.json")).read_text())
            self.assertEqual([e["epoch"] for e in history["epochs"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
