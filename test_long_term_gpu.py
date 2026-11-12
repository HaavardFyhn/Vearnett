import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from remote_senorge import compute_long_term_gpu as runner
from remote_senorge import train_senorge_transfer_gpu as trainer
from senorge_corrected_transfer import FEATURES
from senorge_transfer_map import SeNorgeTransferModel
import test_senorge_long_term as fixtures
from import_long_term_gpu import import_results, verify_archive


class LongTermGPUTests(unittest.TestCase):
    def test_checksum_without_python311_file_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.bin"
            content = b"checksum compatibility"*100000
            path.write_bytes(content)
            with patch.object(hashlib, "file_digest", None, create=True):
                self.assertEqual(runner.sha256(path), hashlib.sha256(content).hexdigest())

    def setup_package(self, folder):
        root = Path(folder)
        backend = fixtures.LongTermTests().backend(folder)
        backend.days = ["2020-02-29", "2024-12-15", "2025-07-01"]
        backend.weather = np.repeat(backend.weather[:, :, :1], 3, axis=2)
        backend.weather[:, :, 1] += 3
        backend.weather[:, :, 2] += 12
        package = root / "package"
        package.mkdir()
        selected = dict(mean_anchor="ridge", mean_checkpoint="mean.pt", error_checkpoint="error.pt")
        spec = dict(identity=backend.identity, catalogue_sha=backend.catalogue_sha,
                    features=FEATURES, selected=selected)
        (package / "spec.json").write_text(json.dumps(spec))
        ridge = dict(mean=np.zeros(len(FEATURES)), std=np.ones(len(FEATURES))*100,
                     coef=np.linspace(-.01, .01, len(FEATURES)), intercept=np.array(.3))
        np.savez(package / "portable_ridge.npz", **ridge)
        networks, models = {}, {}
        torch.manual_seed(2026)
        for name, error in (("mean.pt", False), ("error.pt", True)):
            model = trainer.build_model(torch, len(FEATURES)+int(error), 8, error)
            with torch.no_grad():
                model[6].weight.normal_(0, .1)
                model[6].bias.fill_(.3)
            torch.save(model.state_dict(), package / name)
            norm = name+".npz"
            mean, std = np.zeros(len(FEATURES)+int(error)), np.ones(len(FEATURES)+int(error))*100
            np.savez(package / norm, mean=mean, std=std)
            networks[name] = dict(config=dict(width=8, error_stage=error), weights=name, normalization=norm)
            models[name] = (model.eval(), mean, std)
        (package / "portable_ridge.json").write_text(json.dumps(dict(networks=networks)))
        reference = SeNorgeTransferModel.__new__(SeNorgeTransferModel)
        reference.torch, reference.selected = torch, selected
        reference.ridge, reference.networks = ridge, models
        backend.model = reference
        cells = np.array([0, 1], np.int32)
        targets = [np.sort(backend.tree.query_ball_point(backend.table[c, :2], 46000)).astype(np.int32)
                   for c in cells]
        np.savez(package / "inputs.npz", table=backend.table, cells=cells,
                 temperatures=backend.weather[:, cells], days=np.asarray(backend.days),
                 valid=np.ones((2, len(cells)), bool), targets=np.concatenate(targets),
                 offsets=np.r_[0, np.cumsum([len(x) for x in targets])])
        code = Path(runner.__file__)
        (package / code.name).write_bytes(code.read_bytes())
        (root / "remote_senorge").mkdir()
        (root / "remote_senorge" / code.name).write_bytes(code.read_bytes())
        self.recheck(package)
        return backend, package

    def recheck(self, package):
        checks = {p.name: runner.sha256(p) for p in package.iterdir()
                  if p.name != "package_checksums.json"}
        (package / "package_checksums.json").write_text(json.dumps(checks))

    def execute(self, package, output, benchmark=False):
        args = argparse.Namespace(package=package, output=output, device="cpu",
            batch_size=4, variables=[0, 1], benchmark=benchmark)
        with patch.dict("sys.modules", {"train_senorge_transfer_gpu": trainer}):
            runner.run(args)

    def archive(self, root, output):
        path = root / "returned.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            archive.add(output, arcname="long_term_results")
        return path

    def test_exact_feature_and_pooled_prediction_equivalence_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, package = self.setup_package(folder)
            output = Path(folder) / "results"
            self.execute(package, output)
            self.execute(package, output)
            for receipt in output.glob("source_*.json"):
                metadata = json.loads(receipt.read_text())
                cell, variable = metadata["source_cell"], metadata["variable"]
                targets, expected = backend.source_values(cell, variable)
                with np.load(receipt.with_suffix(".npz")) as saved:
                    np.testing.assert_array_equal(saved["targets"], targets)
                    np.testing.assert_allclose(saved["rmse"], expected, rtol=2e-6, atol=2e-6)
                    np.testing.assert_array_equal(saved["date_count"], [3]*len(targets))

    def test_verified_import_builds_original_network_and_removal_cannot_improve(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, package = self.setup_package(folder)
            output = Path(folder) / "results"
            self.execute(package, output)
            archive = self.archive(Path(folder), output)
            with patch("import_long_term_gpu.LongTermNetwork", return_value=backend):
                import_results(Path(folder), archive, package)
                digests = {p.name: runner.sha256(p) for p in backend.folder.glob("source_*")}
                import_results(Path(folder), archive, package)
                self.assertEqual(digests, {p.name: runner.sha256(p) for p in backend.folder.glob("source_*")})
            for variable in (0, 1):
                backend.prepare_long_term(variable)
                valid = np.isfinite(backend.context["best"])
                self.assertTrue(np.all(backend.context["second"][valid] >= backend.context["best"][valid]))
                self.assertTrue(backend.network_paths(variable)[2].exists())

    def test_benchmark_is_not_importable(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, package = self.setup_package(folder)
            output = Path(folder) / "results"
            self.execute(package, output, benchmark=True)
            with tarfile.open(self.archive(Path(folder), output), "r:gz") as archive:
                with self.assertRaisesRegex(ValueError, "benchmark"):
                    verify_archive(archive, package, backend)

    def test_completed_variable_checkpoint_imports_while_other_variable_is_running(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, package = self.setup_package(folder)
            output = Path(folder) / "results"
            self.execute(package, output)
            (output / "report.json").unlink()
            (output / "status.json").write_text(json.dumps(dict(state="running", variable=1, completed=1)))
            for receipt in list(output.glob("source_*.json")):
                if json.loads(receipt.read_text())["variable"] == 1:
                    receipt.with_suffix(".npz").unlink()
                    receipt.unlink()
            archive = self.archive(Path(folder), output)
            with patch("import_long_term_gpu.LongTermNetwork", return_value=backend):
                import_results(Path(folder), archive, package, variables=[0], checkpoint=True)
            self.assertTrue(backend.network_paths(0)[2].exists())
            self.assertFalse(backend.network_paths(1)[2].exists())
            with tarfile.open(archive, "r:gz") as opened:
                with self.assertRaisesRegex(ValueError, "Missing remote output"):
                    verify_archive(opened, package, backend, variables=[1], checkpoint=True)
            first = runner.sha256(backend.network_paths(0)[1])
            self.execute(package, output)
            archive = self.archive(Path(folder), output)
            with patch("import_long_term_gpu.LongTermNetwork", return_value=backend):
                import_results(Path(folder), archive, package)
            self.assertTrue(backend.network_paths(1)[2].exists())
            self.assertEqual(first, runner.sha256(backend.network_paths(0)[1]))

    def test_incomplete_checkpoint_never_publishes_any_sources_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, package = self.setup_package(folder)
            output = Path(folder) / "results"
            self.execute(package, output)
            (output / "report.json").unlink()
            (output / "status.json").write_text(json.dumps(dict(state="running", variable=0)))
            receipt = next(p for p in output.glob("source_*.json")
                           if json.loads(p.read_text())["variable"] == 0)
            receipt.unlink()
            archive = self.archive(Path(folder), output)
            with patch("import_long_term_gpu.LongTermNetwork", return_value=backend):
                with self.assertRaisesRegex(ValueError, "Missing remote output"):
                    import_results(Path(folder), archive, package, variables=[0], checkpoint=True)
            self.assertEqual(list(backend.folder.glob("source_*")), [])
            self.assertFalse(backend.network_paths(0)[2].exists())

    def test_checkpoint_rejects_benchmark_and_corrupted_completed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, package = self.setup_package(folder)
            output = Path(folder) / "results"
            self.execute(package, output)
            (output / "status.json").write_text(json.dumps(dict(state="benchmark")))
            with tarfile.open(self.archive(Path(folder), output), "r:gz") as archive:
                with self.assertRaisesRegex(ValueError, "benchmark"):
                    verify_archive(archive, package, backend, variables=[0], checkpoint=True)
            (output / "status.json").write_text(json.dumps(dict(state="running")))
            receipt = next(p for p in output.glob("source_*.json")
                           if json.loads(p.read_text())["variable"] == 0)
            receipt.with_suffix(".npz").write_bytes(b"corrupt")
            with tarfile.open(self.archive(Path(folder), output), "r:gz") as archive:
                with self.assertRaisesRegex(ValueError, "receipt identity"):
                    verify_archive(archive, package, backend, variables=[0], checkpoint=True)

    def test_checksum_failure_and_changed_package_refuse_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            _, package = self.setup_package(folder)
            output = Path(folder) / "results"
            self.execute(package, output)
            source = next(output.glob("source_*.npz"))
            source.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Changed completed"):
                self.execute(package, output)
            (package / "spec.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Package checksum"):
                self.execute(package, Path(folder) / "other")
            self.recheck(package)
            with self.assertRaisesRegex(ValueError, "different package"):
                self.execute(package, output)

    def test_import_rejects_tampered_output_before_publishing(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, package = self.setup_package(folder)
            output = Path(folder) / "results"
            self.execute(package, output)
            next(output.glob("source_*.npz")).write_bytes(b"corrupt")
            with tarfile.open(self.archive(Path(folder), output), "r:gz") as archive:
                with self.assertRaisesRegex(ValueError, "Changed remote output"):
                    verify_archive(archive, package, backend)
            self.assertEqual(list(backend.folder.glob("source_*")), [])

    def test_incomplete_date_sources_are_skipped_not_partially_averaged(self):
        with tempfile.TemporaryDirectory() as folder:
            backend, package = self.setup_package(folder)
            with np.load(package / "inputs.npz") as saved:
                data = {key: saved[key].copy() for key in saved.files}
            data["valid"][1, 0] = False
            data["temperatures"][1, 0, 1] = np.nan
            backend.weather[1, 0, 1] = np.nan
            np.savez(package / "inputs.npz", **data)
            self.recheck(package)
            output = Path(folder) / "results"
            self.execute(package, output)
            self.assertEqual(len(list(output.glob("source_*.npz"))), 3)
            with tarfile.open(self.archive(Path(folder), output), "r:gz") as archive:
                verified, _, _ = verify_archive(archive, package, backend)
            self.assertEqual(len(verified), 3)


if __name__ == "__main__":
    unittest.main()
