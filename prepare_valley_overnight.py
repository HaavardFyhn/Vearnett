"""Export verified valley samples and an allowlisted, credential-free GPU trainer."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from remote_senorge.train_remote import sha256, write_json
from train_valley_pilot import load_prepared_samples


ROOT = Path(__file__).resolve().parent


def prepare(data, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite package: {output}")
    zip_path = output.with_suffix(".zip")
    if zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {zip_path}")
    config, samples, features, labels, names, sensors, raw = load_prepared_samples(data)
    validation_nights = sorted(samples.loc[samples["split"] == 1, "night"].unique())
    if len(validation_nights) < 30:
        raise ValueError("At least30validation nights required")
    middle = len(validation_nights)//2
    selection = samples["night"].isin(validation_nights[:middle]).to_numpy()
    calibration = samples["night"].isin(validation_nights[middle:]).to_numpy()
    # Freeze paired examples and QC provenance once; no remote GIS/Frost access.
    arrays = dict(
        x=features["graph_regime"], y=labels, source_c=samples["source_c"].to_numpy(),
        target_c=samples["target_c"].to_numpy(), forecast_delta_c=samples["forecast_delta_c"].to_numpy(),
        train=samples["split"].eq(0).to_numpy(), selection=selection, calibration=calibration,
        test=samples["split"].eq(2).to_numpy(), calm_clear=samples["calm_clear"].to_numpy(),
        source=samples["source"].to_numpy(dtype=str), target=samples["target"].to_numpy(dtype=str),
        night=samples["night"].to_numpy(dtype=str), time=samples["time"].to_numpy(dtype=str))
    output.mkdir(parents=True)
    path = output / "training.npz"
    np.savez_compressed(path, **arrays)
    manifest = dict(
        schema="valley-neural-transfer-v1", features=names["graph_regime"],
        terrain_count=len(names["terrain"]), regime_count=len(names["regime"]),
        data_sha256=sha256(path), rows={name: int(arrays[name].sum())
                                      for name in ("train", "selection", "calibration", "test")},
        nights={name: len(set(arrays["night"][arrays[name]]))
                for name in ("train", "selection", "calibration", "test")},
        input_checksums={name: sha256(data / name) for name in
                         ("config.json", "terrain.npz", "stations.csv", "regimes_progress.json", "observations_progress.json")},
        sensor_choices=sensors.to_dict("records"),
        qc_rejections=dict(Counter(raw.loc[~raw["accepted"].eq(True), "rejection"])),
        provenance=dict(bbox=config["bbox"], groups=config["groups"], winters=config["winters"]),
        limitations=[
            "Already-inspected Haukedal test: exploratory, not a new untouched test.",
            "Only one accepted validation target and one test target; small training geography.",
            "Source observations are contemporaneous: station transfer, not future forecasting.",
            "MEPS may assimilate earlier observations from held-out stations.",
            "Graph inputs are terrain-connectivity summaries, not a GNN or learned covariance.",
            "No snow state, observed inversion labels or national uncertainty surface.",
            "Night-block conformal coverage can fail under spatial/temporal distribution shift.",
            "40hours is a maximum, not a minimum; extra epochs cannot create new information.",
        ])
    write_json(output / "manifest.json", manifest)
    files = {
        "training.npz": path, "manifest.json": output / "manifest.json",
        "train_valley_neural.py": ROOT / "remote_senorge" / "train_valley_neural.py",
        "test_valley_neural.py": ROOT / "remote_senorge" / "test_valley_neural.py",
        "README.md": ROOT / "remote_senorge" / "VALLEY_NEURAL_README.md",
    }
    receipt = {name: sha256(source) for name, source in files.items()}
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, source in files.items():
            archive.write(source, name)
        archive.writestr("package_checksums.json", json.dumps(receipt, indent=2))
        archive.writestr("requirements.txt", "numpy>=1.24,<3\npandas>=1.5,<3\n")
    with ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Portable ZIP CRC verification failed")
        import hashlib
        for name, checksum in receipt.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError(f"Portable member checksum mismatch: {name}")
    write_json(output / "package_receipt.json", dict(
        zip_sha256=sha256(zip_path), bytes=zip_path.stat().st_size, members=receipt))
    print(f"Verified portable package: {zip_path} ({zip_path.stat().st_size:,}bytes)", flush=True)
    print(json.dumps(dict(rows=manifest["rows"], nights=manifest["nights"]), indent=2), flush=True)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs" / "valley_regime_pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "valley_neural_training")
    args = parser.parse_args()
    prepare(args.data.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
