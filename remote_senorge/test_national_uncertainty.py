import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

if __package__:
    from . import train_national_uncertainty as trainer
    from .train_national_transfer import sha256
else:
    import train_national_uncertainty as trainer
    from train_national_transfer import sha256


def fixture(folder):
    folder.mkdir()
    rows, geometry = [], {}
    for block in range(8):
        for station in range(2):
            geometry[f"B{block}S{station}"] = [block*100000+40000+station*1000, 6700000]
    for split, year, first_day in (("train", 2020, 1), ("selection", 2023, 1),
                                   ("calibration", 2023, 16), ("test", 2024, 1)):
        for day in range(first_day, first_day+15):
            for block in range(8 if split == "train" else 2):
                for station in range(2):
                    source = f"B{(block+2)%8}S{station}"
                    target = f"B{block}S{station}" if split == "train" else f"{split if split!='calibration' else 'selection'}{block}"
                    if target not in geometry:
                        geometry[target] = [block*100000+40000, 7000000]
                    rows.append(dict(source=source, target=target, block=f"{block}:67", split=split,
                                     source_c=0., target_c=-10., forecast_delta_c=-8., calm_clear=True,
                                     night=f"{year}-12-{day:02d}", time=f"{year}-12-{day:02d}T03:00:00+00:00"))
    data = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    data["x"] = np.random.default_rng(42).normal(size=(len(rows), 39)).astype(np.float32)
    np.savez(folder / "training.npz", **data)
    manifest = dict(schema="national-transfer-training-v1", terrain_count=16, regime_count=28,
                    features=[f"f{i}" for i in range(39)], rows={
                        s: int((data["split"] == s).sum()) for s in ["train", "selection", "calibration", "test"]})
    (folder / "manifest.json").write_text(json.dumps(manifest))
    config = dict(family="forecast_residual", width=8, dropout=.1, seed=42)
    model = trainer.create_model(torch, config, manifest)
    torch.save(model.state_dict(), folder / "selected_pair.pt")
    np.savez(folder / "selected_normalization.npz", mean=np.zeros(39), std=np.ones(39))
    spec = dict(config=config, epochs=1, station_xy=geometry)
    (folder / "uncertainty_spec.json").write_text(json.dumps(spec))
    files = ("training.npz", "manifest.json", "selected_pair.pt", "selected_normalization.npz", "uncertainty_spec.json")
    (folder / "package_checksums.json").write_text(json.dumps({f: sha256(folder / f) for f in files}))
    return data, geometry


class UncertaintyTests(unittest.TestCase):
    def test_night_max_not_pair_quantile(self):
        nights = np.repeat([f"N{i}" for i in range(20)], 10)
        errors = np.ones(200)
        errors[::10] = 10
        self.assertEqual(trainer.simultaneous_factor(errors, np.ones(200), nights), 10)
        with self.assertRaises(ValueError):
            trainer.simultaneous_factor(errors, np.zeros(200), nights)

    def test_fold_sources_targets_and_buffers_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            data, xy = fixture(Path(temp) / "data")
            folds = trainer.fold_masks(data, xy, 4)
            evaluated = np.zeros(len(data["target_c"]), int)
            for fold in folds:
                fit, evaluate = fold["fit"], fold["evaluate"]
                evaluated += evaluate
                self.assertFalse(set(data["target"][fit]) & set(data["target"][evaluate]))
                self.assertFalse(set(data["source"][fit]) & set(data["target"][evaluate]))
                self.assertTrue((data["split"][fit] == "train").all())
                self.assertFalse(np.isin(data["block"][fit], fold["held_blocks"]).any())
            self.assertLessEqual(evaluated.max(), 1)

    def test_unseen_targets_keep_nearby_sources_but_never_fit_held_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            data, xy = fixture(Path(temp) / "data")
            # Explicit nearest-source transfers in each block.
            train_ids = np.flatnonzero(data["split"] == "train")
            for i in train_ids:
                target = data["target"][i]
                data["source"][i] = target[:-1]+str(1-int(target[-1]))
            folds = trainer.target_fold_masks(data, 4)
            evaluated = np.zeros(len(data["target_c"]), int)
            held_targets = []
            for fold in folds:
                held = fold["held_targets"]
                held_targets.extend(held)
                fit, evaluate = fold["fit"], fold["evaluate"]
                self.assertFalse(np.isin(data["source"][fit], held).any())
                self.assertFalse(np.isin(data["target"][fit], held).any())
                self.assertTrue(np.isin(data["target"][evaluate], held).all())
                self.assertFalse(np.isin(data["source"][evaluate], held).any())
                self.assertTrue(fit.any())
                evaluated += evaluate
            self.assertEqual(set(held_targets), set(data["target"][train_ids]))
            self.assertTrue((evaluated[train_ids] == 1).all())
            self.assertFalse(evaluated[data["split"] != "train"].any())

    def test_full_run_durable_resume_and_no_mean_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            fixture(folder)
            args = trainer.parse_args(["--data", str(folder), "--output", str(Path(temp) / "results"),
                                       "--device", "cpu", "--folds", "4", "--jobs", "1",
                                       "--tree-rows", "100", "--batch-size", "64"])
            before = sha256(folder / "selected_pair.pt")
            with patch("builtins.print"):
                report = trainer.run(args)
                resumed = trainer.run(args)
            self.assertEqual(report, resumed)
            self.assertEqual(before, sha256(folder / "selected_pair.pt"))
            self.assertFalse(report["uncertainty_map_validated"])
            self.assertEqual(report["test"]["nights"], 15)
            self.assertGreater(report["test"]["radius_median_c"], 0)
            self.assertEqual(len(report["folds"]), 8)
            self.assertEqual(set(report["crossfit_diagnostics"]), {"target", "region"})
            self.assertIn("winter", report["crossfit_diagnostics"]["target"])
            self.assertIn("distance_band", report["test"])
            self.assertIn("block", report["test"])
            self.assertIn("target", report["test"])
            args.hours = 39
            with self.assertRaisesRegex(ValueError, "Changed"):
                trainer.run(args)

    def test_holdout_temperatures_do_not_change_crossfit(self):
        with tempfile.TemporaryDirectory() as temp:
            data, xy = fixture(Path(temp) / "data")
            first = trainer.fold_masks(data, xy, 4)+trainer.target_fold_masks(data, 4)
            data["target_c"][data["split"] != "train"] += 100
            second = trainer.fold_masks(data, xy, 4)+trainer.target_fold_masks(data, 4)
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a["fit"], b["fit"])
                np.testing.assert_array_equal(a["evaluate"], b["evaluate"])

    def test_per_block_night_coverage_not_pooled_pairs(self):
        data = dict(target=np.array(["A", "B"]*20), night=np.repeat([f"N{i}" for i in range(20)], 2),
                    block=np.array(["0:67", "1:67"]*20),
                    target_c=np.zeros(40), calm_clear=np.ones(40, bool),
                    x=np.zeros((40, 39)), time=np.repeat("2020-12-01T03:00:00+00:00", 40))
        prediction = np.zeros(40)
        prediction[1::2] = 2
        report = trainer.interval_diagnostics(data, np.arange(40), prediction, np.ones(40))
        self.assertEqual(report["pair_coverage"], .5)
        self.assertEqual(report["night_coverage"], 0)
        self.assertEqual(report["block"]["0:67"]["night_coverage"], 1)
        self.assertEqual(report["block"]["1:67"]["night_coverage"], 0)

    def test_target_and_region_checkpoint_names_and_exact_epoch_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "data"
            data, xy = fixture(folder)
            manifest = json.loads((folder / "manifest.json").read_text())
            spec = json.loads((folder / "uncertainty_spec.json").read_text())
            spec["epochs"] = 2
            args = trainer.parse_args(["--device", "cpu", "--jobs", "1", "--batch-size", "64",
                                       "--output", str(Path(temp) / "resumed")])
            args.output.mkdir()
            torch.set_num_threads(1)
            fold = trainer.target_fold_masks(data, 4)[0]
            save = trainer.atomic_torch_save

            def interrupt(torch, payload, path):
                save(torch, payload, path)
                if payload["epoch"] == 1:
                    raise RuntimeError("Simulated interruption")

            with patch.object(trainer, "atomic_torch_save", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    trainer.refit_fold(torch, data, manifest, spec, fold, args, time.monotonic()+120)
            with patch("builtins.print"):
                resumed = trainer.refit_fold(torch, data, manifest, spec, fold, args, time.monotonic()+120)
                fresh_args = trainer.parse_args(["--device", "cpu", "--jobs", "1", "--batch-size", "64",
                                                  "--output", str(Path(temp) / "fresh")])
                fresh_args.output.mkdir()
                fresh = trainer.refit_fold(torch, data, manifest, spec, fold, fresh_args, time.monotonic()+120)
                region = trainer.fold_masks(data, xy, 4)[0]
                trainer.refit_fold(torch, data, manifest, spec, region, args, time.monotonic()+120)
            np.testing.assert_array_equal(resumed[0], fresh[0])
            np.testing.assert_array_equal(resumed[1], fresh[1])
            self.assertTrue((args.output / "target_fold0.pt").exists())
            self.assertTrue((args.output / "region_fold0.pt").exists())


if __name__ == "__main__":
    unittest.main()
