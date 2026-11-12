"""GPU signed seNorge transfer followed by OOF squared-error prediction."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import senorge_corrected_transfer as transfer
from sklearn.linear_model import Ridge

if __package__:
    from .train_valley_neural import atomic_torch_save, sha256, write_json
    from senorge_corrected_transfer import FEATURES, error_report
else:
    from train_valley_neural import atomic_torch_save, sha256, write_json
    from senorge_corrected_transfer import FEATURES, error_report


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


def balanced_weights(data):
    frame = pd.DataFrame({key: data[key] for key in ("block", "target", "date")})
    keys = ["block", "target", "date"]
    frame["count"] = frame.groupby(keys)["block"].transform("size")
    unique = frame[keys].drop_duplicates().groupby("block").size()
    return (1/(frame["count"]*frame["block"].map(unique))).to_numpy(dtype=np.float64)


def normalization(x, weights):
    mean = np.average(x.astype(np.float64), axis=0, weights=weights)
    std = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    std[std < 1e-8] = 1
    return mean, std


def fit_baseline(data, kind):
    w = balanced_weights(data)
    delta = data["target_c"].astype(np.float64)-data["source_c"]
    if kind == "offset":
        return dict(kind=kind, offset=float(np.average(delta, weights=w)))
    if kind != "ridge":
        raise ValueError("Invalid baseline")
    mean, std = normalization(data["x"], w)
    model = Ridge(alpha=10.)
    model.fit((data["x"].astype(np.float64)-mean)/std, delta, sample_weight=w/w.mean())
    return dict(kind=kind, model=model, mean=mean, std=std)


def baseline_predict(model, data):
    if model["kind"] == "offset":
        delta = np.full(len(data["source_c"]), model["offset"])
    else:
        delta = model["model"].predict((data["x"].astype(np.float64)-model["mean"])/model["std"])
    return data["source_c"].astype(np.float64)+delta


def build_model(torch, inputs, width, error_stage=False):
    nn = torch.nn
    layers = [nn.Linear(inputs, width), nn.SiLU(), nn.LayerNorm(width), nn.Dropout(.1),
              nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)]
    nn.init.zeros_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)
    if error_stage:
        layers.append(nn.Softplus())
    return nn.Sequential(*layers)


def infer(torch, model, x, mean, std, args):
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), args.batch_size):
            values = torch.from_numpy(((x[start:start+args.batch_size]-mean)/std).astype(np.float32)).to(args.device)
            result.append(model(values).squeeze(-1).cpu().numpy())
    output = np.concatenate(result).astype(np.float64)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite GPU predictions")
    return output


def train_network(torch, x, y, weights, validation, config, args, name, deadline, fixed_epochs=None):
    """Exact epoch resume; tuning uses validation only; cross-fit epochs are frozen."""
    path = args.output / f"{name}.pt"
    mean, std = normalization(x, weights)
    torch.manual_seed(config["seed"])
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config["seed"])
    model = build_model(torch, x.shape[1], config["width"], config["error_stage"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    epoch, elapsed, stale, best, best_epoch, done, best_state = 0, 0., 0, None, 0, False, None
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("Changed neural checkpoint configuration")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, elapsed, stale, best, best_epoch, done, best_state = (
            saved[key] for key in ("epoch", "elapsed", "stale", "best", "best_epoch", "done", "best_state"))
        torch.set_rng_state(saved["rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    weights = (weights/weights.mean()).astype(np.float32)
    maximum = fixed_epochs if fixed_epochs is not None else args.epochs
    for next_epoch in range(epoch+1, maximum+1):
        if done:
            break
        if time.monotonic() >= deadline:
            return None
        started = time.monotonic()
        model.train()
        order = torch.randperm(len(y)).numpy()
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start+args.batch_size]
            values = torch.from_numpy(((x[ids]-mean)/std).astype(np.float32)).to(args.device)
            labels = torch.from_numpy(y[ids].astype(np.float32)).to(args.device)
            w = torch.from_numpy(weights[ids]).to(args.device)
            prediction = model(values).squeeze(-1)
            losses = ((prediction-labels)**2 if config["error_stage"] else
                      torch.nn.functional.huber_loss(prediction, labels, delta=2., reduction="none"))
            loss = (losses*w).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        if validation is None:
            value = 0.
        else:
            vx, vy, vw = validation
            predicted = infer(torch, model, vx, mean, std, args)
            value = float(np.average((predicted-vy)**2, weights=vw))
        if best is None or value < best-1e-8 or fixed_epochs is not None:
            best, best_epoch, stale = value, next_epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        epoch = next_epoch
        done = epoch >= maximum or (fixed_epochs is None and stale >= args.patience)
        elapsed += time.monotonic()-started
        atomic_torch_save(torch, dict(
            config=config, model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
            elapsed=elapsed, stale=stale, best=best, best_epoch=best_epoch, done=done,
            best_state=best_state, mean=mean, std=std, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []), path)
        if epoch == 1 or epoch % 10 == 0 or done:
            print(f"{name}: epoch {epoch}/{maximum}, selection MSE {value:.5f}", flush=True)
    if not done:
        return None
    model.load_state_dict(best_state)
    return dict(model=model, mean=mean, std=std, best=best, best_epoch=best_epoch, config=config, path=path)


def score(data, prediction):
    error = prediction-data["target_c"]
    if error.shape != data["target_c"].shape or not np.isfinite(error).all():
        raise ValueError("Invalid temperature predictions")
    return dict(rows=len(error), targets=len(set(data["target"])), blocks=len(set(data["block"])),
                dates=len(set(data["date"])), rmse_c=float(np.sqrt(np.mean(error**2))),
                balanced_rmse_c=float(np.sqrt(np.average(error**2, weights=balanced_weights(data)))),
                mae_c=float(np.abs(error).mean()), bias_c=float(error.mean()))


def load_data(folder):
    checks = json.loads((folder / "package_checksums.json").read_text())
    for name, digest in checks.items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError(f"Package checksum mismatch: {name}")
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema"] != "senorge-transfer-gpu-v1" or manifest["features"] != FEATURES:
        raise ValueError("Wrong GPU dataset/feature schema")
    with np.load(folder / "training.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    n = len(data["target_c"])
    if data["x"].shape != (n, len(FEATURES)) or not np.isfinite(data["x"]).all():
        raise ValueError("Invalid signed feature arrays")
    for key in ("source_c", "target_c", "source", "target", "date", "variable", "block", "split"):
        if data[key].shape != (n,):
            raise ValueError(f"Invalid shape: {key}")
    if (not np.isfinite(data["source_c"]).all() or not np.isfinite(data["target_c"]).all() or
            not np.isin(data["variable"], [0, 1]).all() or not np.isin(
                data["split"], ["train", "selection", "calibration", "test"]).all()):
        raise ValueError("Invalid temperature/split metadata")
    if not np.allclose(data["x"][:, 0], data["source_c"], atol=1e-6) or not np.array_equal(
            data["x"][:, 3], data["variable"]):
        raise ValueError("Source temperature/variable feature mismatch")
    frame = pd.DataFrame({key: data[key] for key in ("source", "target", "date", "variable")})
    if frame.duplicated().any() or (data["source"] == data["target"]).any():
        raise ValueError("Duplicate or self transfer")
    years = pd.to_datetime(data["date"]).year
    endpoints = {}
    for name, (low, high) in dict(train=(1996, 2015), selection=(2016, 2018),
                                  calibration=(2019, 2020), test=(2021, 2025)).items():
        mask = data["split"] == name
        if mask.sum() != manifest["rows"][name] or mask.sum() < 2 or not (
                (years[mask] >= low) & (years[mask] <= high)).all():
            raise ValueError("Split count/year leakage")
        endpoints[name] = set(data["source"][mask]) | set(data["target"][mask])
    if (endpoints["train"] & (endpoints["selection"] | endpoints["calibration"] | endpoints["test"]) or
            endpoints["test"] & (endpoints["selection"] | endpoints["calibration"])):
        raise ValueError("Geographic endpoint leakage")
    return data, manifest


def run_locked(torch, args):
    data, _ = load_data(args.data)
    settings = dict(args={key: str(value) for key, value in vars(args).items()},
                    inputs={name: sha256(args.data / name) for name in ("training.npz", "manifest.json")},
                    code={name: sha256(path) for name, path in (
                        ("train_senorge_transfer_gpu.py", Path(__file__)),
                        ("senorge_corrected_transfer.py", Path(transfer.__file__)),
                        ("train_valley_neural.py", Path(__file__).with_name("train_valley_neural.py")))})
    config_path = args.output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("Changed settings/code/data; use a fresh output")
    write_json(config_path, settings)
    if (args.output / "report.json").exists():
        report = json.loads((args.output / "report.json").read_text())
        for name, digest in report["checksums"].items():
            if Path(name).name != name or sha256(args.output / name) != digest:
                raise ValueError("Completed output checksum mismatch")
        print("Completed GPU experiment verified; no rescoring", flush=True)
        return report
    elapsed = sum(torch.load(path, map_location="cpu", weights_only=False)["elapsed"]
                  for path in args.output.glob("*.pt"))
    deadline = time.monotonic()+max(0, args.hours*3600-elapsed)
    train = subset(data, data["split"] == "train")
    selection = subset(data, data["split"] == "selection")
    tw, sw = balanced_weights(train), balanced_weights(selection)
    candidates, validation = {}, {}
    for kind in ("offset", "ridge"):
        path = args.output / f"{kind}.joblib"
        if path.exists():
            model = joblib.load(path)
        else:
            model = fit_baseline(train, kind)
            joblib.dump(model, path)
        candidates[kind] = dict(baseline=model)
        validation[kind] = score(selection, baseline_predict(model, selection))
    ridge_train = baseline_predict(candidates["ridge"]["baseline"], train)
    ridge_selection = baseline_predict(candidates["ridge"]["baseline"], selection)
    for width in args.widths:
        for seed in args.seeds:
            name = f"temperature_w{width}_s{seed}"
            write_json(args.output / "status.json", dict(state="mean_selection", trial=name))
            result = train_network(torch, train["x"], train["target_c"]-ridge_train, tw,
                                   (selection["x"], selection["target_c"]-ridge_selection, sw),
                                   dict(width=width, seed=seed, error_stage=False), args, name, deadline)
            if result is None:
                return exhausted(args)
            candidates[name] = result
            prediction = ridge_selection+infer(torch, result["model"], selection["x"], result["mean"], result["std"], args)
            validation[name] = score(selection, prediction)
    winner = min(validation, key=lambda name: (validation[name]["balanced_rmse_c"], name))
    selected = candidates[winner]
    oof_path = args.output / "oof_predictions.npz"
    oof_receipt = args.output / "oof_receipt.json"
    if oof_path.exists():
        expected = dict(mean=winner, sha256=sha256(oof_path))
        if not oof_receipt.exists() or json.loads(oof_receipt.read_text()) != expected:
            raise ValueError("OOF checksum/selected mean mismatch; use a fresh output")
        with np.load(oof_path, allow_pickle=False) as saved:
            oof = saved["prediction"].copy()
            if not np.array_equal(saved["squared_error"], (oof-train["target_c"])**2):
                raise ValueError("OOF squared-error mismatch")
    else:
        oof = np.full(len(train["target_c"]), np.nan)
        blocks = sorted(set(train["block"]))
        if len(blocks) < args.folds:
            raise ValueError("Insufficient geographic blocks for requested cross-fits")
        for fold in range(args.folds):
            if time.monotonic() >= deadline:
                return exhausted(args)
            evaluate = np.isin(train["block"], blocks[fold::args.folds])
            fit = ~evaluate
            fitting, evaluation = subset(train, fit), subset(train, evaluate)
            if (set(fitting["source"]) | set(fitting["target"])) & (
                    set(evaluation["source"]) | set(evaluation["target"])):
                raise ValueError("OOF endpoint leakage")
            write_json(args.output / "status.json", dict(state="cross_fitting", fold=fold, mean=winner))
            if winner in ("offset", "ridge"):
                prediction = baseline_predict(fit_baseline(fitting, winner), evaluation)
            else:
                baseline = fit_baseline(fitting, "ridge")
                anchor = baseline_predict(baseline, fitting)
                refit = train_network(torch, fitting["x"], fitting["target_c"]-anchor, balanced_weights(fitting),
                                      None, selected["config"], args, f"oof_fold{fold}", deadline,
                                      fixed_epochs=selected["best_epoch"])
                if refit is None:
                    return exhausted(args)
                prediction = baseline_predict(baseline, evaluation)+infer(
                    torch, refit["model"], evaluation["x"], refit["mean"], refit["std"], args)
            oof[evaluate] = prediction
        if not np.isfinite(oof).all():
            raise ValueError("Incomplete out-of-fold predictions")
        pending = args.output / "oof_predictions.pending.npz"
        np.savez_compressed(pending, prediction=oof, squared_error=(oof-train["target_c"])**2)
        pending.replace(oof_path)
        write_json(oof_receipt, dict(mean=winner, sha256=sha256(oof_path)))
    if oof.shape != train["target_c"].shape or not np.isfinite(oof).all():
        raise ValueError("Invalid saved OOF temperatures")
    ex = np.column_stack([train["x"], oof]).astype(np.float32)
    ey = (oof-train["target_c"])**2
    # Reserve whole training blocks for error-model selection. Never use final test/assessment to tune.
    blocks = sorted(set(train["block"]))
    held = blocks[::5]
    reserve = np.isin(train["block"], held)
    fit = ~reserve
    if not fit.any() or not reserve.any():
        raise ValueError("Insufficient error-model fit/selection geography")
    ew = balanced_weights(subset(train, fit))
    vw = balanced_weights(subset(train, reserve))
    pooled = float(np.average(ey[fit], weights=ew))
    errors = dict(pooled_mse=float(np.average((pooled-ey[reserve])**2, weights=vw)))
    error_candidates = {}
    for width in args.widths:
        for seed in args.seeds:
            name = f"error_w{width}_s{seed}"
            write_json(args.output / "status.json", dict(state="error_selection", trial=name))
            result = train_network(torch, ex[fit], ey[fit], ew, (ex[reserve], ey[reserve], vw),
                                   dict(width=width, seed=seed, error_stage=True), args, name, deadline)
            if result is None:
                return exhausted(args)
            errors[name], error_candidates[name] = result["best"], result
    error_winner = min(errors, key=lambda name: (errors[name], name))

    def mean_predict(part):
        if winner in ("offset", "ridge"):
            return baseline_predict(selected["baseline"], part)
        return baseline_predict(candidates["ridge"]["baseline"], part)+infer(
            torch, selected["model"], part["x"], selected["mean"], selected["std"], args)

    def mse_predict(part, prediction):
        if error_winner == "pooled_mse":
            return np.full(len(prediction), pooled)
        result = error_candidates[error_winner]
        return infer(torch, result["model"], np.column_stack([part["x"], prediction]),
                     result["mean"], result["std"], args)

    assessments = {}
    if not args.validation_only:
        for split in ("calibration", "test"):
            part = subset(data, data["split"] == split)
            prediction = mean_predict(part)
            mse = mse_predict(part, prediction)
            if (mse < 0).any() or not np.isfinite(mse).all():
                raise ValueError("Invalid squared-error prediction")
            rmse = np.sqrt(mse)
            assessments[split] = dict(temperature=score(part, prediction), error_model=error_report(part, prediction, rmse),
                                     offset=score(part, baseline_predict(candidates["offset"]["baseline"], part)),
                                     ridge=score(part, baseline_predict(candidates["ridge"]["baseline"], part)),
                                     raw_difference=score(part, part["source_c"]))
            frame = pd.DataFrame({key: part[key] for key in ("source", "target", "date", "variable", "block", "source_c", "target_c")})
            frame["prediction_c"], frame["predicted_rmse_c"] = prediction, rmse
            frame.to_csv(args.output / f"{split}_predictions.csv", index=False)
    deployment = dict(mean=winner, error=error_winner, mean_anchor="ridge" if winner not in ("offset", "ridge") else winner,
                      pooled_mse=pooled, error_selection_blocks=held,
                      mean_checkpoint=None if winner in ("offset", "ridge") else f"{winner}.pt",
                      error_checkpoint=None if error_winner == "pooled_mse" else f"{error_winner}.pt")
    write_json(args.output / "selected.json", deployment)
    report = dict(schema="senorge-transfer-gpu-results-v1", state="complete", selected=deployment,
                  mean_validation=validation, error_validation_squared_error_loss=errors,
                  oof_temperature=score(train, oof), assessments=assessments,
                  validation_only=args.validation_only, limitations=[
                      "seNorge grid reproduction,not station accuracy,future forecasting or calibrated intervals.",
                      "Only original18train/four validation/four test geographic blocks; already-inspected period/geography.",
                      "Error model learns squared OOF residuals; reserve OOF blocks select its settings,not independent certification.",
                      "No final-test mean/error selection. RMSE=sqrt(predicted MSE),not interval radius or absolute error.",
                      "Baselines may win; exact learned offsets can have zero error,neural softplus alone does not guarantee exact zero.",
                      "Sampled3600dates,expanded reference/pair sampling,not every national cell/day.",
                  ], checksums={})
    names = ["run_config.json", "selected.json", "offset.joblib", "ridge.joblib", "oof_predictions.npz", "oof_receipt.json"]
    names += [path.name for path in args.output.glob("*.pt")]
    if not args.validation_only:
        names += ["calibration_predictions.csv", "test_predictions.csv"]
    report["checksums"] = {name: sha256(args.output / name) for name in names}
    write_json(args.output / "report.json", report)
    write_json(args.output / "status.json", dict(state="complete", mean=winner, error=error_winner,
                                               validation_only=args.validation_only))
    print(json.dumps(dict(mean=winner, error=error_winner, assessments=assessments), indent=2), flush=True)
    return report


def exhausted(args):
    write_json(args.output / "status.json", dict(state="budget_exhausted", ready=False,
                                               detail="Incomplete stages; no completed error/temperature deployment."))
    print("Fitting budget exhausted; results are incomplete. See status.json.", flush=True)
    return None


def run(args):
    import torch
    import os
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    torch.set_num_threads(args.jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "training.lock").open("a+b") as lock:
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
                return run_locked(torch, args)
            except Exception as error:
                write_json(args.output / "status.json", dict(state="failed", error=str(error)))
                raise
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("senorge_transfer_gpu_results"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--hours", type=float, default=40)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args(argv)
    if (not np.isfinite(args.hours) or args.hours <= 0 or
            min(args.epochs, args.patience, args.batch_size, args.jobs, args.folds, *args.widths) < 1 or
            args.folds < 2):
        parser.error("Use positive settings and at least two folds")
    return args


if __name__ == "__main__":
    if run(parse_args()) is None:
        raise SystemExit(2)
