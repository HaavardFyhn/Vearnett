"""Portable regime-gated station-transfer experiment; no GIS, network or credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    staged = path.with_suffix(".pending.json")
    staged.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    staged.replace(path)


def load_data(folder):
    folder = Path(folder)
    checks = folder / "package_checksums.json"
    if checks.exists():
        for name, digest in json.loads(checks.read_text()).items():
            if Path(name).name != name or sha256(folder / name) != digest:
                raise ValueError(f"Portable package checksum mismatch: {name}")
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema"] != "valley-neural-transfer-v1":
        raise ValueError("Unsupported portable valley schema")
    if sha256(folder / "training.npz") != manifest["data_sha256"]:
        raise ValueError("Valley data checksum mismatch")
    with np.load(folder / "training.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    n = len(data["y"])
    if (data["x"].shape != (n, len(manifest["features"])) or
            not 0 < manifest["terrain_count"] < manifest["regime_count"] < data["x"].shape[1]):
        raise ValueError("Invalid feature shape/counts")
    if len(set(manifest["features"])) != len(manifest["features"]):
        raise ValueError("Duplicate feature names")
    for name in ("x", "y", "source_c", "target_c", "forecast_delta_c"):
        if not np.isfinite(data[name]).all() or (name != "x" and data[name].shape != (n,)):
            raise ValueError(f"Invalid numeric data: {name}")
    if not np.allclose(data["y"], data["target_c"]-data["source_c"], atol=1e-5):
        raise ValueError("Transfer labels differ from observed temperatures")
    for name in ("source", "target", "night", "time", "calm_clear"):
        if data[name].shape != (n,):
            raise ValueError(f"Invalid metadata shape: {name}")
    masks = [data[name] for name in ("train", "selection", "calibration", "test")]
    if any(mask.dtype != bool or mask.shape != (n,) or mask.sum() < 2 for mask in masks):
        raise ValueError("Invalid split masks")
    if not (np.stack(masks).sum(axis=0) == 1).all():
        raise ValueError("Overlapping or unassigned examples")
    if data["calm_clear"].dtype != bool:
        raise ValueError("Invalid calm-clear eligibility")
    train, selection, calibration, test = masks
    train_ids = set(data["source"][train]) | set(data["target"][train])
    validation_ids = set(data["target"][selection | calibration])
    test_ids = set(data["target"][test])
    if (train_ids & (validation_ids | test_ids) or validation_ids & test_ids or
            set(data["source"]) & (validation_ids | test_ids)):
        raise ValueError("Spatial endpoint leakage")
    for i, a in enumerate(masks):
        for b in masks[i+1:]:
            if set(data["night"][a]) & set(data["night"][b]):
                raise ValueError("Night leakage")
    stamps = pd.to_datetime(data["time"], utc=True)
    winter = stamps.year-(stamps.month < 12).astype(int)
    if (not np.isin(stamps.month, [12, 1, 2]).all() or
            not np.isin(stamps.hour, [0, 3, 6]).all() or
            not (stamps.minute == 0).all() or not (stamps.second == 0).all() or
            not np.array_equal(stamps.strftime("%Y-%m-%d").to_numpy(), data["night"]) or
            not (winter[train] == 2022).all() or
            not (winter[selection | calibration] == 2023).all() or
            not (winter[test] == 2024).all()):
        raise ValueError("Invalid held-out winter/timestamp")
    frame = pd.DataFrame({key: data[key] for key in ("source", "target", "time")})
    if frame.duplicated().any():
        raise ValueError("Duplicate paired examples")
    values = pd.DataFrame(dict(target=data["target"], time=data["time"], value=data["target_c"]))
    if (values.groupby(["target", "time"])["value"].nunique() != 1).any():
        raise ValueError("Conflicting target observations")
    for name, mask in zip(("train", "selection", "calibration", "test"), masks):
        if manifest["rows"][name] != int(mask.sum()):
            raise ValueError("Manifest split counts differ")
    return data, manifest


def trials(seeds, widths):
    result = []
    for family in ("terrain_mlp", "forecast_residual", "regime_gate", "graph_gate"):
        for width in widths:
            for seed in seeds:
                for risk_weight in ((1.,) if family == "terrain_mlp" else (1., 3.)):
                    result.append(dict(
                        name=f"{family}_w{width}_s{seed}_risk{int(risk_weight)}",
                        family=family, width=width, seed=seed, risk_weight=risk_weight,
                        learning_rate=.001, weight_decay=.01, dropout=.1))
    return result


def create_model(torch, config, manifest):
    nn = torch.nn
    family = config["family"]
    count = (manifest["terrain_count"] if family == "terrain_mlp" else
             len(manifest["features"]) if family == "graph_gate" else manifest["regime_count"])
    width = config["width"]

    def trunk():
        return nn.Sequential(nn.Linear(count, width), nn.SiLU(), nn.LayerNorm(width),
                             nn.Dropout(config["dropout"]), nn.Linear(width, width),
                             nn.SiLU(), nn.Linear(width, 1))

    class Transfer(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([trunk() for _ in range(3 if family.endswith("gate") else 1)])
            if family.endswith("gate"):
                gate_count = manifest["regime_count"]-manifest["terrain_count"]
                self.gate = nn.Sequential(nn.Linear(gate_count, 16), nn.SiLU(), nn.Linear(16, 3))
            for expert in self.experts:
                nn.init.zeros_(expert[-1].weight)
                nn.init.zeros_(expert[-1].bias)

        def forward(self, x):
            values = torch.cat([expert(x[:, :count]) for expert in self.experts], dim=1)
            if family.endswith("gate"):
                weights = self.gate(x[:, manifest["terrain_count"]:manifest["regime_count"]]).softmax(dim=1)
                return (weights*values).sum(dim=1)
            return values[:, 0]

    return Transfer()


def sample_weights(data, risk_weight):
    # Each training night has equal mass before the predeclared risk emphasis.
    train = data["train"]
    _, inverse, counts = np.unique(data["night"][train], return_inverse=True, return_counts=True)
    weights = 1./counts[inverse]
    risk = data["calm_clear"][train] | (data["target_c"][train] <= -5)
    weights *= np.where(risk, risk_weight, 1.)
    return (weights/weights.mean()).astype(np.float32)


def metrics(data, prediction, mask):
    error = np.asarray(prediction)[mask]-data["target_c"][mask]
    if not np.isfinite(error).all() or not len(error):
        raise ValueError("Nonfinite or empty predictions")
    report = dict(rows=len(error), nights=len(set(data["night"][mask])),
                  rmse_c=float(np.sqrt(np.mean(error**2))), mae_c=float(np.abs(error).mean()),
                  bias_c=float(error.mean()))
    for name, subset in (("calm_clear", data["calm_clear"][mask]),
                         ("cold_below_minus5", data["target_c"][mask] <= -5)):
        report[name] = dict(rows=int(subset.sum()), rmse_c=float(np.sqrt(np.mean(error[subset]**2)))
                           if subset.any() else None)
    return report


def radius(data, prediction):
    mask = data["calibration"]
    errors = np.abs(prediction[mask]-data["target_c"][mask])
    worst = pd.Series(errors, index=data["night"][mask]).groupby(level=0).max().to_numpy()
    rank = math.ceil((len(worst)+1)*.9)
    if rank > len(worst):
        raise ValueError("Too few calibration nights for finite90%interval")
    return float(np.partition(worst, rank-1)[rank-1])


def atomic_torch_save(torch, payload, path):
    staged = path.with_suffix(".pending.pt")
    torch.save(payload, staged)
    staged.replace(path)


def train_trial(torch, data, manifest, config, args, output, mean, std, deadline):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = create_model(torch, config, manifest).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    fingerprint = hashlib.sha256(json.dumps(
        dict(data=manifest["data_sha256"], config=config, epochs=args.epochs,
             patience=args.patience, batch_size=args.batch_size, device=args.device,
             features=manifest["features"]), sort_keys=True).encode()).hexdigest()
    checkpoint = output / (config["name"]+".pt")
    epoch, stale, best, elapsed = 0, 0, math.inf, 0.
    best_state = None
    completed = False
    history = []
    if checkpoint.exists():
        # Checkpoints are generated locally by this run; never load untrusted pickle files.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Incompatible checkpoint; use a fresh output folder")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, stale, best, elapsed = (saved[key] for key in ("epoch", "stale", "best", "elapsed"))
        best_state, history, completed = saved["best_state"], saved["history"], saved["completed"]
        torch.set_rng_state(saved["rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    train_ids = np.flatnonzero(data["train"])
    selection_ids = np.flatnonzero(data["selection"])
    x = torch.from_numpy(((data["x"]-mean)/std).astype(np.float32)).to(args.device)
    residual = config["family"] != "terrain_mlp"
    anchor = data["source_c"]+(data["forecast_delta_c"] if residual else 0)
    labels = torch.tensor((data["target_c"]-anchor).astype(np.float32), device=args.device)
    weights = torch.from_numpy(sample_weights(data, config["risk_weight"])).to(args.device)
    selection_index = torch.tensor(selection_ids, device=args.device)
    if completed:
        return dict(config=config, best=best, epochs=epoch, elapsed=elapsed, checkpoint=checkpoint.name)
    for next_epoch in range(epoch+1, args.epochs+1):
        if time.monotonic() >= deadline:
            break
        started = time.monotonic()
        model.train()
        order = torch.randperm(len(train_ids), device=args.device)
        indices = torch.tensor(train_ids, device=args.device)
        for start in range(0, len(train_ids), args.batch_size):
            local = order[start:start+args.batch_size]
            ids = indices[local]
            error = model(x[ids])-labels[ids]
            # Huber limits individual extremes; weighting uses training observations only.
            loss = (torch.nn.functional.huber_loss(error, torch.zeros_like(error), delta=2.,
                                                  reduction="none")*weights[local]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            delta = model(x[selection_index]).cpu().numpy()
        validation = float(np.sqrt(np.mean((delta+anchor[selection_ids]-data["target_c"][selection_ids])**2)))
        if not np.isfinite(validation):
            raise ValueError("Nonfinite validation score")
        if validation < best-1e-6:
            best, stale = validation, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        elapsed += time.monotonic()-started
        epoch = next_epoch
        history.append(dict(epoch=epoch, selection_rmse_c=validation))
        completed = stale >= args.patience or epoch >= args.epochs
        atomic_torch_save(torch, dict(
            fingerprint=fingerprint, model=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch, stale=stale, best=best, best_state=best_state, elapsed=elapsed,
            completed=completed, history=history, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []), checkpoint)
        if epoch == 1 or epoch % 25 == 0 or completed:
            print(f"{config['name']} epoch{epoch}: selection RMSE{validation:.4f}, best{best:.4f}", flush=True)
        if completed:
            break
    if best_state is None:
        return None
    return dict(config=config, best=best, epochs=epoch, elapsed=elapsed, checkpoint=checkpoint.name)


def run(args):
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(args.jobs)
    data, manifest = load_data(args.data)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # Prevent two workers from modifying the same checkpoint set.
    import os
    lock_path = output / "training.lock"
    with lock_path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            try:
                return run_locked(torch, data, manifest, args, output)
            except Exception as error:
                write_json(output / "status.json", dict(state="failed", error=str(error)))
                raise
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def run_locked(torch, data, manifest, args, output):
    configurations = trials(args.seeds, args.widths)
    settings = dict(schema="valley-neural-run-v1", data_sha256=manifest["data_sha256"],
                    manifest_sha256=sha256(args.data / "manifest.json"), trials=configurations,
                    epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                    device=args.device, jobs=args.jobs, hours=args.hours,
                    trainer_sha256=sha256(Path(__file__)))
    path = output / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("Run configuration changed; use a new output folder")
    write_json(path, settings)
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        for name, checksum in summary["checksums"].items():
            if sha256(output / name) != checksum:
                raise ValueError(f"Completed result checksum mismatch: {name}")
        print("Completed results verified; not retraining or rescoring test.", flush=True)
        return summary
    mean = data["x"][data["train"]].mean(axis=0).astype(np.float32)
    std = data["x"][data["train"]].std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Invalid training-only normalization")
    np.savez(output / "normalization.npz", mean=mean, std=std)
    spent = 0.
    for config in configurations:
        checkpoint = output / (config["name"]+".pt")
        if checkpoint.exists():
            spent += float(torch.load(checkpoint, map_location="cpu", weights_only=False)["elapsed"])
    deadline = time.monotonic()+max(0, args.hours*3600-spent)
    candidates = []
    baseline = data["source_c"]+data["forecast_delta_c"]
    baseline_score = metrics(data, baseline, data["selection"])["rmse_c"]
    candidates.append(dict(config=dict(name="forecast_transfer", family="baseline"),
                           best=baseline_score, epochs=0, elapsed=0.))
    write_json(output / "status.json", dict(state="training", trials=len(configurations), completed=0))
    for index, config in enumerate(configurations):
        checkpoint = output / (config["name"]+".pt")
        if time.monotonic() >= deadline and not checkpoint.exists():
            break
        result = train_trial(torch, data, manifest, config, args, output, mean, std, deadline)
        if result is not None:
            candidates.append(result)
        write_json(output / "status.json", dict(state="training", attempted=index+1,
                                                trials=len(configurations)))
    selected = min(candidates, key=lambda item: (item["best"], item["config"]["name"]))
    prediction = baseline.copy()
    if selected["config"]["family"] != "baseline":
        checkpoint = output / selected["checkpoint"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = create_model(torch, selected["config"], manifest).to(args.device)
        model.load_state_dict(saved["best_state"])
        model.eval()
        x = torch.from_numpy(((data["x"]-mean)/std).astype(np.float32)).to(args.device)
        with torch.inference_mode():
            corrections = np.concatenate([
                model(x[start:start+args.batch_size]).cpu().numpy()
                for start in range(0, len(x), args.batch_size)])
        prediction = data["source_c"]+corrections
        if selected["config"]["family"] != "terrain_mlp":
            prediction += data["forecast_delta_c"]
        atomic_torch_save(torch, saved["best_state"], output / "selected_model.pt")
    r = radius(data, prediction)
    test = data["test"]
    error = np.abs(prediction[test]-data["target_c"][test])
    night_error = pd.Series(error, index=data["night"][test]).groupby(level=0).max().to_numpy()
    frame = pd.DataFrame({name: data[name][test] for name in
                          ("source", "target", "time", "night", "source_c", "target_c", "calm_clear")})
    frame["prediction_c"] = prediction[test]
    frame["forecast_transfer_prediction_c"] = baseline[test]
    frame.to_csv(output / "selected_test_predictions.csv", index=False)
    summary = dict(
        state="complete", selected=selected, trial_validation=candidates,
        selection_policy="Minimum RMSE on selection-validation nights only; baseline can win. Calibration/test never select trials.",
        normalization_policy="Means/std from training rows only.",
        selection=metrics(data, prediction, data["selection"]),
        test=metrics(data, prediction, test), baseline_test=metrics(data, baseline, test),
        interval_radius_c=r, empirical_pair_coverage_90=float((error <= r).mean()),
        empirical_night_coverage_90=float((night_error <= r).mean()),
        calibration_nights=len(set(data["night"][data["calibration"]])),
        maximum_hours=args.hours, measured_training_hours=sum(item["elapsed"] for item in candidates)/3600,
        budget_exhausted=time.monotonic() >= deadline,
        limitations=manifest["limitations"], checksums={})
    files = ["normalization.npz", "run_config.json", "selected_test_predictions.csv"]
    if selected["config"]["family"] != "baseline":
        files.append("selected_model.pt")
    summary["checksums"] = {name: sha256(output / name) for name in files}
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", dict(state="complete", selected=selected["config"]["name"]))
    print(json.dumps(dict(selected=selected["config"]["name"], test=summary["test"],
                          measured_training_hours=summary["measured_training_hours"],
                          interval_radius_c=r), indent=2), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("valley_neural_results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hours", type=float, default=40.)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args(argv)
    if (not math.isfinite(args.hours) or args.hours <= 0 or min(args.epochs, args.patience, args.batch_size) < 1 or
            not 1 <= args.jobs <= 8 or min(args.widths) < 8 or max(args.widths) > 512 or
            min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds) or
            len(set(args.widths)) != len(args.widths) or
            not (args.device == "cpu" or args.device.startswith("cuda:"))):
        parser.error("Invalid budget, device, trials or training parameters")
    return args


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            # Do not overwrite another process's live status if lock acquisition failed.
            print(f"Training failed: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
