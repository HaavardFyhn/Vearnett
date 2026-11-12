"""Inspect returned results and install only portable UBJ, never server joblib files."""
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

import numpy as np

from remote_senorge.train_remote import sha256, write_json


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def install():
    folder = OUT / "senorge_30yr_station_model"
    folder.mkdir(exist_ok=True)
    names = {"results/summary.json", "results/run_config.json", "manifest.json", "installed_versions.txt"}
    for backend in ("extratrees", "xgboost"):
        for option in ("with_basins", "without_basins"):
            name = f"{backend}_{option}"
            names.update({f"results/{name}.json", f"results/{name}_holdout.npz"})
    names.add("results/xgboost_with_basins.ubj")
    with tarfile.open(OUT / "senorge_results_30yr.tar.gz") as archive:
        found = set()
        for member in archive:
            if member.name not in names:
                continue
            if not member.isfile() or member.size > 100*1024**2:
                raise ValueError(f"Unexpected archive member: {member.name}")
            destination = folder / Path(member.name).name
            with archive.extractfile(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            found.add(member.name)
        if found != names:
            raise ValueError(f"Missing files: {names-found}")
    manifest = json.loads((folder / "manifest.json").read_text())
    local = OUT / "senorge_grid_pairs_1m_1996_2025_random"
    summary = json.loads((folder / "summary.json").read_text())
    dataset_sha = json.loads((local / "dataset_prepared.json").read_text())
    if summary["status"] != "complete" or dataset_sha["accepted_pairs"] != summary["pairs"]:
        raise ValueError("Incomplete or unexpected dataset")
    run = json.loads((folder / "run_config.json").read_text())
    if not manifest["dataset_sha256"] == summary["dataset_sha256"] == run["dataset_sha256"]:
        raise ValueError("Dataset fingerprint mismatch")
    from station_senorge import PAIR_FEATURES
    if manifest["features"] != PAIR_FEATURES or run["features"] != PAIR_FEATURES:
        raise ValueError("Unexpected feature order")
    with zipfile.ZipFile(OUT / "senorge_server_package_1996_2025.zip") as package:
        original = json.loads(package.read("manifest.json"))
        if manifest != original:
            raise ValueError("Returned manifest differs from the prepared dataset")
        with package.open("grid_features.csv.gz") as source, (folder / "grid_features.csv.gz").open("wb") as target:
            shutil.copyfileobj(source, target)
    inspection = {}
    for name, experiment in summary["experiments"].items():
        receipt = json.loads((folder / f"{name}.json").read_text())
        if receipt != experiment:
            raise ValueError(f"Receipt differs from summary: {name}")
        holdout = folder / f"{name}_holdout.npz"
        if sha256(holdout) != receipt["file_sha256"][holdout.name]:
            raise ValueError(f"Holdout checksum mismatch: {name}")
        with np.load(holdout, allow_pickle=False) as data:
            actual, predicted = data["actual"], data["predicted"]
            if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
                raise ValueError(f"Invalid holdout: {name}")
            error = predicted-actual
            rmse = float(np.sqrt(np.mean(error**2)))
            if not np.isclose(rmse, receipt["validation"]["rmse"]):
                raise ValueError(f"Invalid reported RMSE: {name}")
            inspection[name] = dict(**receipt["validation"], bias=float(error.mean()),
                                    absolute_error_p95=float(np.quantile(np.abs(error), .95)),
                                    pairs=len(actual), actual_device=receipt["actual_device"])
    model_path = folder / "xgboost_with_basins.ubj"
    expected = summary["experiments"]["xgboost_with_basins"]["file_sha256"][model_path.name]
    if sha256(model_path) != expected:
        raise ValueError("Portable model checksum mismatch")
    write_json(folder / "metadata.json", dict(
        features=PAIR_FEATURES, model_sha256=expected, dataset_sha256=manifest["dataset_sha256"],
        basin_grid_sha256=sha256(folder / "grid_features.csv.gz"),
        period="1996–2025", selected_experiment="xgboost_with_basins",
        selection="Lowest RMSE on the single fixed spatial holdout; no additional independent selection test.",
        validation=inspection["xgboost_with_basins"]))
    write_json(folder / "inspection.json", inspection)
    print(json.dumps(inspection, indent=2))
    print("Installed portable UBJ model; server joblib models were not extracted.")


if __name__ == "__main__":
    install()
