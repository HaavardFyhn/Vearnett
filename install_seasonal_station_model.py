"""Verify returned seasonal experiments before installing the portable CNN."""
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

from remote_senorge.train_remote import sha256, write_json


ROOT = Path(__file__).resolve().parent


def install(root=ROOT):
    outputs = root / "outputs"
    local = outputs / "senorge_seasonal_terrain"
    folder = outputs / "senorge_seasonal_station_model"
    folder.mkdir(exist_ok=True)
    modes = ("xgboost", "terrain", "combined")
    names = {"seasonal_manifest.json", "seasonal_results/summary.json",
             "seasonal_results/run_config.json"}
    for mode in modes:
        names.update({f"seasonal_results/{mode}.json", f"seasonal_results/{mode}_holdout.npz"})
    names.update({"seasonal_results/combined.pt", "seasonal_results/combined_normalization.npz"})
    with tarfile.open(outputs / "seasonal_terrain_results.tar.gz") as archive:
        found = set()
        for member in archive:
            if member.name not in names:
                continue
            if member.name in found or not member.isfile() or member.size > 100*1024**2:
                raise ValueError(f"Unexpected returned member: {member.name}")
            with archive.extractfile(member) as source, (folder / Path(member.name).name).open("wb") as target:
                shutil.copyfileobj(source, target)
            found.add(member.name)
        if found != names:
            raise ValueError(f"Missing returned files: {names-found}")
    manifest_path = folder / "seasonal_manifest.json"
    if sha256(manifest_path) != sha256(local / "seasonal_manifest.json"):
        raise ValueError("Returned seasonal manifest differs from the original dataset")
    manifest = json.loads(manifest_path.read_text())
    run = json.loads((folder / "run_config.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if run["manifest_sha256"] != sha256(manifest_path) or summary["status"] != "complete":
        raise ValueError("Incomplete results or manifest fingerprint mismatch")
    with np.load(local / "seasonal_training.npz", allow_pickle=False) as data:
        indices = np.flatnonzero(data["split"] == 2)
        for mode in modes:
            receipt = json.loads((folder / f"{mode}.json").read_text())
            if receipt != summary["experiments"][mode]:
                raise ValueError(f"Summary/receipt mismatch: {mode}")
            for name in (f"{mode}_holdout.npz", *(
                    ("combined.pt", "combined_normalization.npz") if mode == "combined" else ())):
                if sha256(folder / name) != receipt["checksums"][name]:
                    raise ValueError(f"Returned checksum mismatch: {name}")
            with np.load(folder / f"{mode}_holdout.npz", allow_pickle=False) as held:
                np.testing.assert_array_equal(held["pair_indices"], indices)
                np.testing.assert_array_equal(held["pairs"], data["pairs"][indices])
                np.testing.assert_array_equal(held["actual"], data["y"][indices])
                pred = held["predicted"]
                if pred.shape != (len(indices), 4) or not np.isfinite(pred).all() or (pred < 0).any():
                    raise ValueError(f"Invalid holdout predictions: {mode}")
                for i, target in enumerate(manifest["targets"]):
                    rmse = float(np.sqrt(np.mean((pred[:, i].astype(float)-held["actual"][:, i])**2)))
                    if not np.isclose(rmse, receipt["test"][target]["rmse"], rtol=1e-7):
                        raise ValueError(f"Reported test score mismatch: {mode}/{target}")
    write_json(folder / "metadata.json", dict(
        features=manifest["features"], targets=manifest["targets"], period="1996–2025",
        selected_experiment="combined", architecture=run["architecture"],
        selection="Lowest RMSE for all four targets on the single buffered spatial test. "
                  "Selection uses this test; no additional independent selection test.",
        model_sha256=sha256(folder / "combined.pt"),
        normalization_sha256=sha256(folder / "combined_normalization.npz"),
        manifest_sha256=sha256(manifest_path), validation=summary["experiments"]["combined"]["test"]))
    print(json.dumps({mode: {name: value["rmse"] for name, value in
                            summary["experiments"][mode]["test"].items()} for mode in modes}, indent=2))
    return folder


if __name__ == "__main__":
    install()
