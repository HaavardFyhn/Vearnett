"""Verify returned footprint results and install a separate experimental app model."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import numpy as np

from remote_senorge.footprint_atlas import ABSOLUTE_FIELDS, AtlasDataset
from remote_senorge.footprint_model import TARGETS, metrics
from remote_senorge.train_remote import sha256, write_json


ROOT = Path(__file__).resolve().parent
RESULTS = "footprint_v2_absolute_results"


def verify_holdout(folder, atlas, summary):
    with np.load(folder / "selected_test_maps.npz", allow_pickle=False) as held:
        ids = np.flatnonzero(atlas.refs["split"] == 2)
        np.testing.assert_array_equal(held["reference_indices"], ids)
        np.testing.assert_array_equal(held["xy"], atlas.refs["xy"][ids])
        actual, predicted, mask = held["actual"], held["predicted"], held["mask"]
        shape = (len(ids), 6, 64, 64)
        if actual.shape != shape or predicted.shape != shape or mask.shape != shape or mask.dtype != bool:
            raise ValueError("Invalid returned test map shapes")
        if not np.isfinite(predicted).all() or (predicted < 0).any():
            raise ValueError("Invalid returned CNN predictions")
        for row, ref in enumerate(ids):
            np.testing.assert_array_equal(actual[row], atlas.labels[ref])
            expected = atlas.mask[ref].copy()
            expected[:, 32, 32] = False
            np.testing.assert_array_equal(mask[row], expected)
        computed = metrics(actual, predicted, mask)
        for target in TARGETS:
            for field in ("cells", "rmse", "mae", "bias", "r2"):
                if not np.isclose(computed[target][field], summary["test"][target][field], rtol=1e-7, atol=1e-10):
                    raise ValueError(f"Reported CNN score differs: {target}/{field}")
        for i, target in enumerate(TARGETS):
            with np.load(folder / f"xgboost_{target}_holdout.npz", allow_pickle=False) as baseline:
                np.testing.assert_array_equal(baseline["actual"], actual[:, i][mask[:, i]])
                p = baseline["predicted"]
                if p.shape != baseline["actual"].shape or not np.isfinite(p).all() or (p < 0).any():
                    raise ValueError(f"Invalid returned baseline: {target}")
                error = p.astype(float)-baseline["actual"]
                scores = dict(cells=len(p), rmse=float(np.sqrt(np.mean(error**2))),
                              mae=float(np.mean(abs(error))), bias=float(error.mean()))
                for field, value in scores.items():
                    if not np.isclose(value, summary["xgboost_test"][target][field], rtol=1e-7, atol=1e-10):
                        raise ValueError(f"Reported baseline score differs: {target}/{field}")


def install(root=ROOT):
    root = Path(root)
    prepared = root / "outputs" / "senorge_footprint_expanded"
    destination = root / "outputs" / "senorge_footprint_station_model"
    if destination.exists():
        raise FileExistsError("Footprint model already installed; refusing to overwrite it")
    with tempfile.TemporaryDirectory(prefix="footprint-install-", dir=root / "outputs") as temporary, ExitStack() as stack:
        folder = Path(temporary) / "model"
        folder.mkdir()
        atlas = AtlasDataset(prepared)
        stack.callback(atlas.close)
        with tarfile.open(root / "outputs" / "footprint_v2_absolute_results.tar.gz", "r:gz") as archive:
            members = {}
            for member in archive.getmembers():
                if member.name in members:
                    raise ValueError(f"Duplicate returned archive member: {member.name}")
                members[member.name] = member

            def read(name):
                member = members.get(name)
                if member is None or not member.isfile() or member.size > 512*1024**2:
                    raise ValueError(f"Missing/invalid returned member: {name}")
                with archive.extractfile(member) as source:
                    return source.read()

            returned_manifest = read("footprint_manifest.json")
            if returned_manifest != (prepared / "footprint_manifest.json").read_bytes():
                raise ValueError("Returned footprint dataset manifest differs from local dataset")
            summary = json.loads(read(f"{RESULTS}/summary.json"))
            run = json.loads(read(f"{RESULTS}/run_config.json"))
            if (summary["status"] != "complete" or summary["targets"] != TARGETS or
                    summary["architecture"] != "reference-resunet-spatial10-absolute-v2" or
                    summary["pair_channels"] != 31 or summary["spatial_channels"] != 11 or
                    summary["features"][-6:] != ABSOLUTE_FIELDS or
                    run["manifest_sha256"] != sha256(prepared / "footprint_manifest.json")):
                raise ValueError("Unexpected returned model architecture/provenance")
            if summary["selected"] != min(summary["trials"], key=lambda value: value["validation_mse"]):
                raise ValueError("Selected model was not the validation-best trial")
            allowed = {"selected_model.pt", "normalization.npz", "selected_test_maps.npz", "run_config.json"}
            allowed.update(f"xgboost_{target}{suffix}" for target in TARGETS for suffix in (".ubj", "_holdout.npz"))
            if set(summary["checksums"]) != allowed:
                raise ValueError("Unexpected returned checksum allowlist")
            for name, digest in summary["checksums"].items():
                (folder / name).write_bytes(read(f"{RESULTS}/{name}"))
                if sha256(folder / name) != digest:
                    raise ValueError(f"Returned model checksum mismatch: {name}")
            for trial in summary["trials"]:
                name = trial["name"]
                if Path(name).name != name:
                    raise ValueError("Invalid trial name")
                best = read(f"{RESULTS}/{name}_best.pt")
                import hashlib
                if hashlib.sha256(best).hexdigest() != trial["model_sha256"]:
                    raise ValueError(f"Returned trial checksum mismatch: {name}")
                if json.loads(read(f"{RESULTS}/{name}.json")) != trial:
                    raise ValueError(f"Trial receipt differs from summary: {name}")
            (folder / "footprint_manifest.json").write_bytes(returned_manifest)
            write_json(folder / "summary.json", summary)
        verify_holdout(folder, atlas, summary)
        with np.load(folder / "normalization.npz", allow_pickle=False) as norm:
            if (norm["mean"].shape != (31,) or norm["std"].shape != (31,) or
                    not np.isfinite(norm["mean"]).all() or not np.isfinite(norm["std"]).all() or
                    (norm["std"] <= 0).any()):
                raise ValueError("Invalid31-channel normalization")
        from station_footprint import FootprintBackend
        metadata = dict(architecture=summary["architecture"], targets=TARGETS, selected=summary["selected"],
                        manifest_sha256=sha256(folder / "footprint_manifest.json"),
                        checksums={name: sha256(folder / name) for name in
                                   ("selected_model.pt", "normalization.npz", "footprint_manifest.json", "summary.json")},
                        test=summary["test"], baseline_test=summary["xgboost_test"])
        write_json(folder / "metadata.json", metadata)
        backend = FootprintBackend(root, folder, atlas=atlas)
        differences = []
        with np.load(folder / "selected_test_maps.npz", allow_pickle=False) as held:
            for row in (0, len(held["reference_indices"])//2, len(held["reference_indices"])-1):
                i = int(held["reference_indices"][row])
                ref = int(atlas.refs["refs"][i])
                predicted, mapping = backend.predict_reference(ref)
                np.testing.assert_array_equal(mapping, atlas.refs["mapping"][i])
                expected = held["predicted"][row]
                maximum = float(np.abs(predicted-expected).max())
                if maximum > .002:
                    raise ValueError(f"CPU inference differs from returned GPU map by{maximum} degrees")
                differences.append(maximum)
        write_json(folder / "cpu_verification.json", dict(references=3, maximum_difference_c= max(differences),
                                                          tolerance_c=.002))
        folder.rename(destination)
    print(json.dumps(dict(installed=str(destination), test=summary["test"],
                          cpu_maximum_difference_c=max(differences)), indent=2), flush=True)
    return destination


if __name__ == "__main__":
    install()
