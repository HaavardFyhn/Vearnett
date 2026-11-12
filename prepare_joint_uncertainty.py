"""Package the fixed selected joint mean model and unseen-target residual experiment."""
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import torch

from prepare_national_dataset import ROOT, load_splits
from remote_senorge.train_valley_neural import sha256, write_json


def prepare(output):
    archive_path = output.with_suffix(".zip")
    if output.exists() or archive_path.exists():
        raise FileExistsError("Choose a new output; refusing joint package overwrite")
    result = ROOT / "outputs" / "national_transfer_returned" / "national_transfer_results"
    summary = json.loads((result / "summary.json").read_text())
    for name, digest in summary["checksums"].items():
        if Path(name).name != name or sha256(result / name) != digest:
            raise ValueError(f"National result corrupt: {name}")
    chosen = summary["fusion_selected"]
    if chosen != min(summary["fusion_validation"], key=lambda x: (x["best"], x["name"])):
        raise ValueError("Joint mean model was not selected on validation")
    checkpoint = torch.load(result / (chosen["name"]+".pt"), map_location="cpu", weights_only=False)
    epoch = min(checkpoint["history"], key=lambda x: x["selection_balanced_rmse_c"])["epoch"]
    data = ROOT / "outputs" / "national_station_dataset"
    load_splits(data)
    stations = pd.read_csv(data / "station_splits.csv")
    spec = dict(schema="joint-residual-training-v1", config=chosen["config"], epochs=epoch,
                station_xy={row.station_id: [float(row.x), float(row.y)] for row in stations.itertuples()},
                selected_summary_sha256=sha256(result / "summary.json"),
                source_limit=6, source_scope="Frozen training sources for fitting/validation; later nontraining map sources are extrapolation",
                map_timestamp="2024-12-15T03:00:00+00:00", hours=40,
                coverage="90%simultaneous evaluated joint locations and valid leave-one-source-out configurations per night",
                primary_residuals="Unseen-target OOF,including full and all valid single-source deletions",
                diagnostic_residuals="Separate buffered geographic rounds; not pooled into scale fitting",
                contribution="Radius and temperature change after removing a source without replacement; may be negative")
    output.mkdir(parents=True)
    write_json(output / "joint_spec.json", spec)
    files = {"manifest.json": result.parent / "manifest.json", "joint_spec.json": output / "joint_spec.json",
             "selected_fusion.pt": result / "selected_fusion.pt",
             "selected_normalization.npz": result / "normalization.npz"}
    for name in ("train_joint_uncertainty.py", "train_national_uncertainty.py", "train_national_transfer.py",
                 "train_valley_neural.py", "test_joint_uncertainty.py", "test_national_uncertainty.py"):
        files[name] = ROOT / "remote_senorge" / name
    files["README.md"] = ROOT / "remote_senorge" / "JOINT_UNCERTAINTY_README.md"
    receipts = {name: sha256(path) for name, path in files.items()}
    original = ROOT / "outputs" / "national_transfer_training.zip"
    with ZipFile(original) as source:
        if source.read("manifest.json") != (result.parent / "manifest.json").read_bytes():
            raise ValueError("Original package and returned joint model manifest differ")
        checks = json.loads(source.read("package_checksums.json"))
        with source.open("training.npz") as stream:
            receipts["training.npz"] = hashlib.file_digest(stream, "sha256").hexdigest()
        if receipts["training.npz"] != checks["training.npz"]:
            raise ValueError("Original training data corrupt")
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            with source.open("training.npz") as stream, archive.open("training.npz", "w") as target:
                shutil.copyfileobj(stream, target)
            for name, path in files.items():
                archive.write(path, name)
            archive.writestr("package_checksums.json", json.dumps(receipts, indent=2))
            archive.writestr("requirements.txt", "numpy>=1.24,<3\npandas>=1.5,<3\nscikit-learn>=1.3,<2\njoblib>=1.3,<2\n")
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Joint package CRC failure")
        for name, digest in receipts.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Joint archive member corrupt: {name}")
            if name in files and sha256(files[name]) != digest:
                raise ValueError(f"Source changed while packaging: {name}")
    write_json(output / "package_receipt.json", dict(bytes=archive_path.stat().st_size,
               zip_sha256=sha256(archive_path), fixed_epoch=epoch, members=receipts))
    print(f"Verified joint package: {archive_path}; fixed mean refit epochs={epoch}", flush=True)
    return archive_path


if __name__ == "__main__":
    prepare(ROOT / "outputs" / "joint_uncertainty_training_v1")
