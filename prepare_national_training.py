"""Verify ready national shards and export an allowlisted remote training ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from prepare_national_dataset import ROOT, load_config, load_splits, split_for, validate_regime
from remote_senorge.train_national_transfer import load_data
from remote_senorge.train_remote import sha256, write_json


def prepare(data, output):
    archive_path = output.with_suffix(".zip")
    if output.exists() or archive_path.exists():
        raise FileExistsError("Refusing to overwrite a training package; choose a new output")
    load_config(data)
    load_splits(data)
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    if manifest["state"] != "ready" or manifest["schema"] != "national-station-transfer-v1":
        raise ValueError("National dataset not ready")
    if sha256(data / "splits.json") != manifest["provenance"]["split_sha256"]:
        raise ValueError("Dataset split fingerprint mismatch")
    weather_receipt = json.loads((data / "regimes_progress.json").read_text())
    if sha256(data / "regimes_progress.json") != manifest["provenance"]["forecast_receipt_sha256"]:
        raise ValueError("Dataset forecast receipt mismatch")
    if sha256(data / "pairs.npz") != manifest["pairs_sha256"]:
        raise ValueError("Dataset pair graph mismatch")
    stations = pd.read_csv(data / "station_splits.csv")
    active = stations.loc[stations["group"].isin(["train", "validation", "test"])].reset_index(drop=True)
    blocks = active.set_index("station_id")[["block_x", "block_y"]].to_dict("index")
    source_ids = active.loc[active["group"] == "train", "station_id"].to_numpy()
    target_groups = active.set_index("station_id")["group"].to_dict()
    columns = {k: [] for k in ("x", "source_c", "target_c", "forecast_delta_c",
                               "source", "target", "night", "time", "split", "calm_clear", "block")}
    excluded = []
    for number, (day, digest) in enumerate(sorted(manifest["shards"].items()), 1):
        path = data / "samples" / f"{day}.npz"
        if sha256(path) != digest:
            raise ValueError(f"National shard checksum mismatch: {day}")
        forecast_path = data / "regimes" / f"{day}.json"
        if sha256(forecast_path) != weather_receipt[day]:
            raise ValueError(f"National forecast checksum mismatch: {day}")
        forecast = json.loads(forecast_path.read_text())
        validate_regime(forecast, pd.Timestamp(day).date(), active)
        with np.load(path, allow_pickle=False) as arrays:
            n = len(arrays["y"])
            if (arrays["x"].shape != (n, 39) or not np.isfinite(arrays["x"]).all() or
                    not np.isfinite(arrays["y"]).all() or
                    not np.allclose(arrays["y"], arrays["target_c"]-arrays["source_c"], atol=1e-5)):
                raise ValueError(f"Invalid national feature/label shard: {day}")
            if not np.isin(arrays["source"], source_ids).all() or not (arrays["night"] == day).all():
                raise ValueError(f"Invalid national source/night identity: {day}")
            for target in np.unique(arrays["target"]):
                expected = split_for(pd.Timestamp(day).date(), target_groups[target])
                if not (arrays["split"][arrays["target"] == target] == expected).all():
                    raise ValueError(f"Frozen geographic split mismatch: {day}/{target}")
            if forecast.get("state") == "excluded_missing_run":
                if len(arrays["y"]):
                    raise ValueError("Excluded night contains examples")
                excluded.append(day)
            for name in columns:
                if name == "block":
                    columns[name].append(np.array([
                        f"{blocks[t]['block_x']}:{blocks[t]['block_y']}" for t in arrays["target"]], dtype=str))
                else:
                    columns[name].append(arrays[name].copy())
        if number % 50 == 0:
            print(f"Verified {number}/{len(manifest['shards'])}shards", flush=True)
    arrays = {name: np.concatenate(parts) for name, parts in columns.items()}
    counts = {s: int((arrays["split"] == s).sum()) for s in ("train", "selection", "calibration", "test")}
    if counts != {s: counts_["rows"] for s, counts_ in manifest["counts"].items()}:
        raise ValueError("Dataset count mismatch")
    output.mkdir(parents=True)
    np.savez_compressed(output / "training.npz", **arrays)
    training = dict(
        schema="national-transfer-training-v1", features=manifest["features"], terrain_count=16, regime_count=28,
        rows=counts, counts=manifest["counts"], source_manifest_sha256=sha256(data / "dataset_manifest.json"),
        data_sha256=sha256(output / "training.npz"), excluded_nights=excluded,
        group_policy=manifest["splits"], limitations=[
            "Contemporaneous winter station transfer,not future forecasts or daily minima.",
            "1.56million pairs share timestamps/targets; train/evaluate with equal geographic-block and target-night mass.",
            "Only two validation and two test geographic blocks; national coverage is not national accuracy certification.",
            "No per-heldout-station or pair offset fitted; pooled offsets/lapse-rate regressions use training only.",
            "Fusion jointly learns masked source weights/corrections; not a GNN or learned spatial covariance.",
            "Source errors may be correlated; do not divide error by sqrt(station count) or treat weights as station importance.",
            "Calibrated maximum-per-night intervals can be broad; no coverage guarantee under regional/weather shifts.",
            "MEPS may assimilate earlier heldout station readings. No snow/history inputs or continuous1km uncertainty map.",
            "40hours is accumulated measured fitting budget,not strict wall time; tree fit/current epoch may overrun.",
            "Fixed split experiment,not leave-region-out crossvalidation. More independent holdouts needed before deployment.",
        ])
    write_json(output / "manifest.json", training)
    return write_archive(output)


def write_archive(output):
    archive_path = output.with_suffix(".zip")
    files = {"training.npz": output / "training.npz", "manifest.json": output / "manifest.json",
             "train_national_transfer.py": ROOT / "remote_senorge" / "train_national_transfer.py",
             "train_valley_neural.py": ROOT / "remote_senorge" / "train_valley_neural.py",
             "test_national_transfer.py": ROOT / "remote_senorge" / "test_national_transfer.py",
             "README.md": ROOT / "remote_senorge" / "NATIONAL_TRANSFER_README.md"}
    receipts = {name: sha256(path) for name, path in files.items()}
    write_json(output / "package_checksums.json", {name: receipts[name] for name in ("training.npz", "manifest.json")})
    # Vectorized portable validation checks every row's metadata,identity,split and temperature.
    load_data(output)
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("package_checksums.json", json.dumps(receipts, indent=2))
        archive.writestr("requirements.txt", "numpy>=1.24,<3\npandas>=1.5,<3\nscikit-learn>=1.3,<2\njoblib>=1.3,<2\n")
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        for name, digest in receipts.items():
            with archive.open(name) as file:
                actual = hashlib.file_digest(file, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"ZIP member checksum mismatch: {name}")
    write_json(output / "package_receipt.json", dict(bytes=archive_path.stat().st_size,
               zip_sha256=sha256(archive_path), members=receipts))
    print(f"Verified national training package: {archive_path} ({archive_path.stat().st_size:,}bytes)")
    return archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs" / "national_station_dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "national_transfer_training")
    args = parser.parse_args()
    prepare(args.data.resolve(), args.output.resolve())
