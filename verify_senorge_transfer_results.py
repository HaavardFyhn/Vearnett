"""Reproduce returned seNorge GPU predictions without refitting or test selection."""
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd

from remote_senorge import train_senorge_transfer_gpu as trainer


def verify(root):
    import torch

    root = Path(root)
    folder = root / "outputs" / "senorge_transfer_gpu_returned"
    result = folder / "senorge_transfer_gpu_results"
    report = json.loads((result / "report.json").read_text())
    if report["state"] != "complete" or report["schema"] != "senorge-transfer-gpu-results-v1" or report["validation_only"]:
        raise ValueError("Completed full seNorge results required")
    for name, digest in report["checksums"].items():
        if Path(name).name != name or trainer.sha256(result / name) != digest:
            raise ValueError(f"Returned seNorge checksum mismatch: {name}")
    run = json.loads((result / "run_config.json").read_text())
    with ZipFile(root / "outputs" / "senorge_transfer_gpu_v1.zip") as archive:
        checks = json.loads(archive.read("package_checksums.json"))
        if checks != json.loads((folder / "package_checksums.json").read_text()):
            raise ValueError("Changed seNorge package")
        for name, digest in {**run["inputs"], **run["code"]}.items():
            if digest != checks[name]:
                raise ValueError(f"Changed seNorge run provenance: {name}")
        for name in ("manifest.json",):
            if archive.read(name) != (folder / name).read_bytes():
                raise ValueError("Changed seNorge manifest")
        with archive.open("training.npz") as stream, np.load(stream, allow_pickle=False) as saved:
            mask = np.isin(saved["split"], ["calibration", "test"])
            data = {key: saved[key][mask].copy() for key in saved.files}
    selected = report["selected"]
    args = SimpleNamespace(device="cpu", batch_size=4096)
    torch.set_num_threads(4)

    def network(name, x):
        info = receipt["networks"][name]
        if info["original_sha256"] != trainer.sha256(result / name):
            raise ValueError("Changed original neural checkpoint")
        for output, digest in info["checksums"].items():
            if trainer.sha256(folder / output) != digest:
                raise ValueError("Changed portable neural checkpoint")
        config = info["config"]
        model = trainer.build_model(torch, x.shape[1], config["width"], config["error_stage"])
        model.load_state_dict(torch.load(folder / info["weights"], map_location="cpu", weights_only=True))
        with np.load(folder / info["normalization"], allow_pickle=False) as saved:
            return trainer.infer(torch, model, x, saved["mean"], saved["std"], args)

    receipt = json.loads((folder / "portable_ridge.json").read_text())
    if (receipt["model_sha256"] != trainer.sha256(result / "ridge.joblib") or
            receipt["arrays_sha256"] != trainer.sha256(folder / "portable_ridge.npz")):
        raise ValueError("Portable ridge checksum mismatch")
    with np.load(folder / "portable_ridge.npz", allow_pickle=False) as saved:
        ridge = {key: saved[key].copy() for key in saved.files}
    summaries = {}
    for split in ("calibration", "test"):
        part = trainer.subset(data, data["split"] == split)
        if selected["mean_anchor"] == "offset":
            prediction = part["source_c"].astype(np.float64)+ridge["offset"]
        else:
            prediction = part["source_c"].astype(np.float64)+(
                (part["x"].astype(np.float64)-ridge["mean"])/ridge["std"])@ridge["coef"]+ridge["intercept"]
        if selected["mean_checkpoint"]:
            prediction += network(selected["mean_checkpoint"], part["x"])
        mse = (network(selected["error_checkpoint"], np.column_stack([part["x"], prediction]).astype(np.float32))
               if selected["error_checkpoint"] else np.full(len(prediction), selected["pooled_mse"]))
        if not np.isfinite(mse).all() or (mse < 0).any():
            raise ValueError("Invalid seNorge MSE")
        frame = pd.read_csv(result / f"{split}_predictions.csv",
                            dtype={"source": str, "target": str, "block": str})
        for key in ("source", "target", "date", "variable", "block"):
            if not np.array_equal(frame[key], part[key]):
                raise ValueError(f"seNorge evaluation identity mismatch: {key}")
        for key in ("source_c", "target_c"):
            np.testing.assert_allclose(frame[key], part[key], atol=1e-5, rtol=1e-6)
        np.testing.assert_allclose(frame.prediction_c, prediction, atol=1e-4, rtol=2e-5)
        np.testing.assert_allclose(frame.predicted_rmse_c, np.sqrt(mse), atol=1e-4, rtol=2e-5)
        from national_joint_uncertainty import compare_diagnostics
        compare_diagnostics(trainer.score(part, frame.prediction_c.to_numpy()),
                            report["assessments"][split]["temperature"])
        compare_diagnostics(trainer.error_report(part, frame.prediction_c.to_numpy(),
                            frame.predicted_rmse_c.to_numpy()), report["assessments"][split]["error_model"])
        summaries[split] = len(frame)
    receipt = dict(schema="verified-senorge-transfer-v1", all_predictions_verified=True, rows=summaries,
                   report_sha256=trainer.sha256(result / "report.json"),
                   manifest_sha256=trainer.sha256(folder / "manifest.json"),
                   archive_sha256=trainer.sha256(root / "outputs" / "senorge_transfer_gpu_results.tar.gz"),
                   csv_sha256={name: trainer.sha256(result / f"{name}_predictions.csv") for name in summaries},
                   missing_installed_versions=not (folder / "installed_versions.txt").exists(),
                   note="Archive omitted installed_versions.txt. Checksummed models/data/code and all300000 "
                        "assessment/test temperature and RMSE predictions reproduced on CPU. "
                        "No training,selection or recalibration; runtime evaluation uses verified CSVs.")
    trainer.write_json(folder / "verification.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


def export_ridge(root):
    """Run only in a matched sklearn/NumPy environment; export trusted numeric parameters."""
    import joblib
    import torch
    folder = Path(root) / "outputs" / "senorge_transfer_gpu_returned"
    result = folder / "senorge_transfer_gpu_results"
    report = json.loads((result / "report.json").read_text())
    for name in ("ridge.joblib", "offset.joblib"):
        if trainer.sha256(result / name) != report["checksums"][name]:
            raise ValueError(f"Model checksum mismatch: {name}")
    model = joblib.load(result / "ridge.joblib")
    offset = joblib.load(result / "offset.joblib")
    arrays = dict(mean=model["mean"], std=model["std"], coef=model["model"].coef_,
                  intercept=model["model"].intercept_, offset=offset["offset"])
    x = np.random.default_rng(42).normal(size=(1000, len(trainer.FEATURES)))
    np.testing.assert_allclose(model["model"].predict((x-arrays["mean"])/arrays["std"]),
        ((x-arrays["mean"])/arrays["std"])@arrays["coef"]+arrays["intercept"], atol=1e-10, rtol=1e-10)
    np.savez_compressed(folder / "portable_ridge.npz", **arrays)
    networks = {}
    for name in (report["selected"]["mean_checkpoint"], report["selected"]["error_checkpoint"]):
        if name is None:
            continue
        if trainer.sha256(result / name) != report["checksums"][name]:
            raise ValueError("Changed neural checkpoint")
        saved = torch.load(result / name, map_location="cpu", weights_only=False)
        weights, norm = f"portable_{name}", f"portable_{Path(name).stem}_normalization.npz"
        torch.save(saved["best_state"], folder / weights)
        np.savez_compressed(folder / norm, mean=saved["mean"], std=saved["std"])
        networks[name] = dict(config=saved["config"], weights=weights, normalization=norm,
                              original_sha256=trainer.sha256(result / name),
                              checksums={key: trainer.sha256(folder / key) for key in (weights, norm)})
    trainer.write_json(folder / "portable_ridge.json", dict(model_sha256=trainer.sha256(result / "ridge.joblib"),
                       arrays_sha256=trainer.sha256(folder / "portable_ridge.npz"), networks=networks))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--export-ridge"]:
        export_ridge(Path(__file__).resolve().parent)
    elif not sys.argv[1:]:
        verify(Path(__file__).resolve().parent)
    else:
        raise SystemExit("Use no arguments to verify, or --export-ridge for matched-runtime conversion.")
