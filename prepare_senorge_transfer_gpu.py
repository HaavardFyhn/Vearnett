"""Export expanded raw seNorge samples and a credential-free two-stage GPU trainer."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from senorge_corrected_transfer import ROOT, FEATURES, geography_pairs, sampled_rows
from remote_senorge.train_valley_neural import sha256, write_json


def prepare(args):
    output = args.output
    archive_path = output.with_suffix(".zip")
    if output.exists() or archive_path.exists():
        raise FileExistsError("Choose a new package output; refusing overwrite")
    atlas = args.atlas
    manifest = json.loads((atlas / "footprint_manifest.json").read_text())
    for name in ("grid_table.npy", "references.npz", "selected_dates.csv"):
        if sha256(atlas / name) != manifest["checksums"][name]:
            raise ValueError(f"Frozen atlas input changed: {name}")
    if json.loads((atlas / "cache" / "weather_progress.json").read_text()) != dict(completed_years=30, years=30):
        raise ValueError("The complete30-year seNorge cache is required")
    table = np.load(atlas / "grid_table.npy", mmap_mode="r", allow_pickle=False)
    weather = np.load(atlas / "cache" / "sampled_weather.npy", mmap_mode="r", allow_pickle=False)
    try:
        plan = pd.read_csv(atlas / "selected_dates.csv")
        if weather.shape != (2, len(table), 3600) or len(plan) != 3600 or set(plan.year) != set(range(1996, 2026)):
            raise ValueError("Unexpected seNorge source shape/date coverage")
        with np.load(atlas / "references.npz", allow_pickle=False) as refs:
            pairs = geography_pairs(table, refs, tuple(args.references), args.targets)
        years = plan.year.to_numpy()
        definitions = dict(train=(0, years <= 2015, args.train_rows),
                           selection=(1, (years >= 2016) & (years <= 2018), args.selection_rows),
                           calibration=(1, (years >= 2019) & (years <= 2020), args.selection_rows),
                           test=(2, years >= 2021, args.test_rows))
        parts = []
        counts, missing, geography = {}, {}, {}
        for i, (name, (group, dates, count)) in enumerate(definitions.items()):
            rows = sampled_rows(table, weather, plan, pairs, np.flatnonzero(pairs["split"] == group),
                                np.flatnonzero(dates), count, 2026+i)
            counts[name], missing[name] = len(rows["target_c"]), rows.pop("missing_rows")
            rows.pop("pair")
            rows["split"] = np.repeat(name, counts[name])
            geography[name] = dict(blocks=len(set(rows["block"])), sources=len(set(rows["source"])),
                                   targets=len(set(rows["target"])), dates=len(set(rows["date"])))
            parts.append(rows)
            print(f"Prepared {name}: {counts[name]:,}rows, {geography[name]}", flush=True)
        data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        provenance = {name: sha256(atlas / name) for name in
                      ("footprint_manifest.json", "grid_table.npy", "references.npz", "selected_dates.csv",
                       "cache/sampled_weather.npy", "cache/source_config.json")}
    finally:
        table._mmap.close()
        weather._mmap.close()
    output.mkdir(parents=True)
    np.savez_compressed(output / "training.npz", **data)
    write_json(output / "manifest.json", dict(
        schema="senorge-transfer-gpu-v1", features=FEATURES, rows=counts, missing_rows=missing,
        geography=geography, provenance=provenance, preparer_sha256=sha256(Path(__file__)),
        sampling=dict(references=args.references, targets_per_source=args.targets, seed=2026),
        default_fitting_hours=40, limitations=[
            "Sampled3600seNorge dates1996-2025,both daily mean/minimum; not all cells/days.",
            "Frozen original spatial splits; original period/geography and pilot test already inspected.",
            "Final test records are not training inputs. No Frost,MEPS,credentials or existing app models packaged.",
            "Signed mean trained first; error stage uses OOF squared residuals,not raw temperature differences.",
            "RMSE is not interval radius; no positive error floor or guaranteed grid/station accuracy.",
        ]))
    files = {"training.npz": output / "training.npz", "manifest.json": output / "manifest.json",
             "senorge_corrected_transfer.py": ROOT / "senorge_corrected_transfer.py",
             "README.md": ROOT / "remote_senorge" / "SENORGE_TRANSFER_GPU_README.md"}
    for name in ("train_senorge_transfer_gpu.py", "test_senorge_transfer_gpu.py", "train_valley_neural.py"):
        files[name] = ROOT / "remote_senorge" / name
    receipts = {name: sha256(path) for name, path in files.items()}
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("package_checksums.json", json.dumps(receipts, indent=2))
        archive.writestr("requirements.txt", "numpy>=1.24,<3\npandas>=1.5,<3\nscikit-learn>=1.3,<2\njoblib>=1.3,<2\n")
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("GPU archive CRC failure")
        for name, digest in receipts.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest or sha256(files[name]) != digest:
                    raise ValueError(f"Package/source mismatch: {name}")
    write_json(output / "package_receipt.json", dict(bytes=archive_path.stat().st_size,
               zip_sha256=sha256(archive_path), members=receipts))
    print(f"Verified GPU package: {archive_path}", flush=True)
    return archive_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=ROOT / "outputs" / "senorge_footprint_expanded")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "senorge_transfer_gpu_v1")
    parser.add_argument("--references", nargs=3, type=int, default=[2000, 200, 200])
    parser.add_argument("--targets", type=int, default=16)
    parser.add_argument("--train-rows", type=int, default=1000000)
    parser.add_argument("--selection-rows", type=int, default=100000)
    parser.add_argument("--test-rows", type=int, default=200000)
    args = parser.parse_args()
    if min(*args.references, args.targets, args.train_rows, args.selection_rows, args.test_rows) < 1:
        parser.error("Use positive bounded dataset sizes")
    return args


if __name__ == "__main__":
    prepare(parse_args())
