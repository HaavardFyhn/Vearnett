"""Package geographic residual-error training without changing the mean predictor."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import torch

from prepare_national_dataset import ROOT, load_splits
from remote_senorge.train_remote import sha256, write_json


def prepare(output):
    archive_path = output.with_suffix(".zip")
    if output.exists() or archive_path.exists():
        raise FileExistsError("Choose a new output; refusing uncertainty package overwrite")
    original = ROOT / "outputs" / "national_transfer_training.zip"
    result = ROOT / "outputs" / "national_transfer_returned" / "national_transfer_results"
    summary = json.loads((result / "summary.json").read_text())
    for name, digest in summary["checksums"].items():
        if Path(name).name != name or sha256(result / name) != digest:
            raise ValueError(f"National result corrupt: {name}")
    chosen = summary["pair_selected"]
    if chosen != min(summary["pair_validation"], key=lambda x: (x["best"], x["name"])):
        raise ValueError("Mean model not the validation-selected winner")
    if chosen["config"]["family"] != "forecast_residual":
        raise ValueError("Current cross-fit package supports the selected residual MLP only")
    checkpoint = torch.load(result / (chosen["name"]+".pt"), map_location="cpu", weights_only=False)
    best_epoch = min(checkpoint["history"], key=lambda x: x["selection_balanced_rmse_c"])["epoch"]
    data_folder = ROOT / "outputs" / "national_station_dataset"
    load_splits(data_folder)
    stations = pd.read_csv(data_folder / "station_splits.csv")
    geometry = {row.station_id: [float(row.x), float(row.y)] for row in stations.itertuples()}
    spec = dict(schema="national-crossfit-error-v2", config=chosen["config"], epochs=best_epoch,
                station_xy=geometry, selected_summary_sha256=sha256(result / "summary.json"),
                selected_weights_sha256=sha256(result / "selected_pair.pt"),
                coverage="90%simultaneous evaluated station-pair coverage per night",
                crossfit="Five unseen-target rounds plus five buffered geographic rounds; error scale fits target rounds only",
                fixed_epoch="Original selection best epoch; no OOF/test early stopping")
    with ZipFile(original) as original_zip:
        if original_zip.read("manifest.json") != (result.parent / "manifest.json").read_bytes():
            raise ValueError("Returned model and national package differ")
        for name, digest in json.loads(original_zip.read("package_checksums.json")).items():
            import hashlib
            with original_zip.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Original package corrupt: {name}")
        output.mkdir(parents=True)
        write_json(output / "uncertainty_spec.json", spec)
        files = {
            "manifest.json": result.parent / "manifest.json",
            "uncertainty_spec.json": output / "uncertainty_spec.json",
            "selected_pair.pt": result / "selected_pair.pt",
            "selected_normalization.npz": result / "normalization.npz",
            "train_national_transfer.py": ROOT / "remote_senorge" / "train_national_transfer.py",
            "train_valley_neural.py": ROOT / "remote_senorge" / "train_valley_neural.py",
            "train_national_uncertainty.py": ROOT / "remote_senorge" / "train_national_uncertainty.py",
            "test_national_uncertainty.py": ROOT / "remote_senorge" / "test_national_uncertainty.py",
            "README.md": ROOT / "remote_senorge" / "NATIONAL_UNCERTAINTY_README.md",
        }
        receipts = {name: sha256(path) for name, path in files.items()}
        import hashlib
        with original_zip.open("training.npz") as stream:
            receipts["training.npz"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            with original_zip.open("training.npz") as source, archive.open("training.npz", "w") as target:
                import shutil
                shutil.copyfileobj(source, target)
            for name, path in files.items():
                archive.write(path, name)
            archive.writestr("package_checksums.json", json.dumps(receipts, indent=2))
            archive.writestr("requirements.txt", "numpy>=1.24,<3\npandas>=1.5,<3\nscikit-learn>=1.3,<2\njoblib>=1.3,<2\n")
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Uncertainty archive CRC failure")
        for name, digest in receipts.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Uncertainty archive member corrupt: {name}")
    write_json(output / "package_receipt.json", dict(bytes=archive_path.stat().st_size,
               zip_sha256=sha256(archive_path), members=receipts))
    print(f"Verified uncertainty package: {archive_path}; fixed cross-fit epochs={best_epoch}")
    return archive_path


if __name__ == "__main__":
    prepare(ROOT / "outputs" / "national_uncertainty_training_v2")
