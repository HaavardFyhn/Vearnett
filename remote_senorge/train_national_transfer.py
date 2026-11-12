"""Portable national corrected-transfer and masked multi-station fusion experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

if __package__:
    from .train_valley_neural import sha256, write_json, atomic_torch_save, create_model
else:
    from train_valley_neural import sha256, write_json, atomic_torch_save, create_model


SPLITS = ("train", "selection", "calibration", "test")


def load_data(folder):
    checks = json.loads((folder / "package_checksums.json").read_text())
    for name, digest in checks.items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError(f"Package checksum mismatch: {name}")
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema"] != "national-transfer-training-v1":
        raise ValueError("Wrong national training schema")
    with np.load(folder / "training.npz", allow_pickle=False) as saved:
        data = {k: saved[k].copy() for k in saved.files}
    n = len(data["target_c"])
    if data["x"].shape != (n, 39) or not np.isfinite(data["x"]).all():
        raise ValueError("Invalid national feature shape/values")
    for key in ("source_c", "target_c", "forecast_delta_c"):
        if data[key].shape != (n,) or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid temperatures: {key}")
    for key in ("source", "target", "time", "night", "split", "block", "calm_clear"):
        if data[key].shape != (n,):
            raise ValueError(f"Invalid metadata: {key}")
    if not np.isin(data["split"], SPLITS).all() or data["calm_clear"].dtype != bool:
        raise ValueError("Invalid splits/calm-clear metadata")
    frame = pd.DataFrame({k: data[k] for k in ("source", "target", "time", "night", "split", "target_c")})
    if frame.duplicated(["source", "target", "time"]).any() or (data["source"] == data["target"]).any():
        raise ValueError("Duplicate/self pairs")
    train = data["split"] == "train"
    train_ids = set(data["source"][train]) | set(data["target"][train])
    validation_ids = set(data["target"][np.isin(data["split"], ["selection", "calibration"])])
    test_ids = set(data["target"][data["split"] == "test"])
    if (train_ids & (validation_ids | test_ids) or validation_ids & test_ids or
            set(data["source"]) & (validation_ids | test_ids)):
        raise ValueError("Spatial endpoint leakage")
    stamps = pd.to_datetime(data["time"], utc=True)
    winter = stamps.year-(stamps.month < 12).astype(int)
    if (not np.isin(stamps.month, [12, 1, 2]).all() or not np.isin(stamps.hour, [0, 3, 6]).all() or
            (stamps.minute != 0).any() or (stamps.second != 0).any() or
            not np.array_equal(stamps.strftime("%Y-%m-%d").to_numpy(), data["night"]) or
            not np.isin(winter[train], [2020, 2021, 2022]).all() or
            not (winter[np.isin(data["split"], ["selection", "calibration"])] == 2023).all() or
            not (winter[data["split"] == "test"] == 2024).all()):
        raise ValueError("Winter/time leakage")
    for i, a in enumerate(SPLITS):
        mask = data["split"] == a
        if mask.sum() != manifest["rows"][a] or mask.sum() < 2:
            raise ValueError("Invalid manifest split count")
        for b in SPLITS[i+1:]:
            if set(data["night"][mask]) & set(data["night"][data["split"] == b]):
                raise ValueError("Shared split nights")
    if (frame.groupby(["target", "time"])["target_c"].nunique() != 1).any():
        raise ValueError("Conflicting target observations")
    return data, manifest


def grouped(data):
    """One example per simultaneous target/time; retain every available source."""
    frame = pd.DataFrame({k: data[k] for k in ("target", "time", "split", "night", "block", "target_c")})
    groups = frame.groupby(["target", "time"], sort=True).indices
    index = np.full((len(groups), 6), -1, np.int64)
    first = []
    for row, ids in enumerate(groups.values()):
        ids = sorted(ids, key=lambda i: (data["x"][i, 4], data["source"][i]))
        if len(ids) > 6 or any(len(set(data[k][ids])) != 1 for k in
                               ("split", "night", "block", "target_c")):
            raise ValueError("Invalid simultaneous fusion group")
        index[row, :len(ids)] = ids
        first.append(ids[0])
    mask = index >= 0
    result = {k: data[k][first] for k in ("target", "time", "split", "night", "block", "target_c")}
    result.update(index=index, mask=mask, calm_clear=np.array([
        data["calm_clear"][ids[ids >= 0]].all() for ids in index], bool))
    return result


def balanced_weights(data):
    frame = pd.DataFrame({k: data[k] for k in ("target", "night", "block")})
    # Equal block mass, equal target-night mass inside each block, equal pair mass inside each target-night.
    counts = frame.groupby(["block", "target", "night"]).size()
    block_counts = counts.groupby(level=0).size()
    return np.array([1/(counts[(r.block, r.target, r.night)]*block_counts[r.block])
                     for r in frame.itertuples()], np.float64)


def score(data, prediction):
    error = np.asarray(prediction)-data["target_c"]
    if error.shape != data["target_c"].shape or not np.isfinite(error).all():
        raise ValueError("Nonfinite or mismatched predictions")
    w = balanced_weights(data)
    report = dict(rows=len(error), targets=len(set(data["target"])), nights=len(set(data["night"])),
                  rmse_c=float(np.sqrt(np.mean(error**2))),
                  balanced_rmse_c=float(np.sqrt(np.average(error**2, weights=w))),
                  mae_c=float(np.abs(error).mean()), bias_c=float(error.mean()))
    for key, mask in (("cold", data["target_c"] <= -5), ("calm_clear", data["calm_clear"])):
        report[key] = dict(rows=int(mask.sum()), rmse_c=float(np.sqrt(np.mean(error[mask]**2)))
                           if mask.any() else None)
    report["blocks"] = {}
    for block in sorted(set(data["block"])):
        part = data["block"] == block
        report["blocks"][block] = dict(rows=int(part.sum()), rmse_c=float(np.sqrt(np.mean(error[part]**2))))
    return report


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()}


def fused_average(prediction, groups):
    indices, mask = groups["index"], groups["mask"]
    return (prediction[np.maximum(indices, 0)]*mask).sum(1)/mask.sum(1)


def calibrate(data, prediction):
    errors = pd.Series(np.abs(prediction-data["target_c"]), index=data["night"]).groupby(level=0).max().to_numpy()
    rank = int(np.ceil((len(errors)+1)*.9))
    if rank > len(errors):
        raise ValueError("Too few calibration nights")
    return float(np.partition(errors, rank-1)[rank-1])


def fusion_model(torch, width):
    nn = torch.nn

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.correction = nn.Sequential(nn.Linear(39, width), nn.SiLU(), nn.LayerNorm(width),
                                            nn.Dropout(.1), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
            self.gate = nn.Sequential(nn.Linear(39, width), nn.SiLU(), nn.Linear(width, 1))
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)

        def forward(self, x, anchor, mask):
            logits = self.gate(x).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = logits.softmax(dim=1)
            # Correlated source readings are learned jointly; not an independence-based variance calculation.
            return ((anchor+self.correction(x).squeeze(-1))*weights).sum(1)

    return Fusion()


def trials(widths, seeds):
    return [dict(name=f"{family}_w{width}_s{seed}", family=family, width=width, seed=seed,
                 dropout=.1, learning_rate=.001, weight_decay=.01)
            for width in widths for seed in seeds
            for family in ("forecast_residual", "regime_gate", "graph_gate", "fusion")]


def infer(torch, model, x, anchor, mask, indices, batch_size, fusion):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            values = model(x[ids], anchor[ids], mask[ids]) if fusion else model(x[ids])+anchor[ids]
            output.append(values.cpu().numpy())
    return np.concatenate(output)


def fit_trial(torch, data, groups, manifest, config, args, output, norm, deadline):
    torch.manual_seed(config["seed"])
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config["seed"])
    fusion = config["family"] == "fusion"
    meta = groups if fusion else data
    normalized = ((data["x"]-norm["mean"])/norm["std"]).astype(np.float32)
    baseline = data["source_c"]+data["forecast_delta_c"]
    if fusion:
        ids = np.maximum(groups["index"], 0)
        normalized, baseline = normalized[ids], baseline[ids]
    x = torch.from_numpy(normalized).to(args.device)
    anchor = torch.from_numpy(baseline.astype(np.float32)).to(args.device)
    mask = torch.from_numpy(groups["mask"]).to(args.device) if fusion else None
    target = torch.from_numpy(meta["target_c"].astype(np.float32)).to(args.device)
    train_ids = np.flatnonzero(meta["split"] == "train")
    selection_ids = np.flatnonzero(meta["split"] == "selection")
    w = balanced_weights(subset(meta, meta["split"] == "train"))
    w = torch.from_numpy((w/w.mean()).astype(np.float32)).to(args.device)
    model = (fusion_model(torch, config["width"]) if fusion else create_model(torch, config, manifest)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    checkpoint = output / (config["name"]+".pt")
    epoch, stale, best, elapsed, completed, best_state, history = 0, 0, math.inf, 0., False, None, []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, stale, best, elapsed, completed, best_state, history = (
            saved[k] for k in ("epoch", "stale", "best", "elapsed", "completed", "best_state", "history"))
        torch.set_rng_state(saved["rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    selection_meta = subset(meta, meta["split"] == "selection")
    for next_epoch in range(epoch+1, args.epochs+1):
        if completed or time.monotonic() >= deadline:
            break
        start_time = time.monotonic()
        model.train()
        order = torch.randperm(len(train_ids)).numpy()
        for start in range(0, len(order), args.batch_size):
            local = order[start:start+args.batch_size]
            ids = train_ids[local]
            pred = model(x[ids], anchor[ids], mask[ids]) if fusion else model(x[ids])+anchor[ids]
            loss = (torch.nn.functional.huber_loss(pred, target[ids], delta=2., reduction="none")*w[local]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neural loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        pred = infer(torch, model, x, anchor, mask, selection_ids, args.batch_size, fusion)
        value = score(selection_meta, pred)["balanced_rmse_c"]
        if value < best-1e-6:
            best, stale = value, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        epoch = next_epoch
        elapsed += time.monotonic()-start_time
        completed = stale >= args.patience or epoch >= args.epochs
        history.append(dict(epoch=epoch, selection_balanced_rmse_c=value))
        atomic_torch_save(torch, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          epoch=epoch, stale=stale, best=best, elapsed=elapsed, completed=completed,
                          best_state=best_state, history=history, rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []), checkpoint)
        if epoch == 1 or epoch % 10 == 0 or completed:
            print(f"{config['name']} epoch{epoch}: balanced selection RMSE{value:.4f}", flush=True)
    if best_state is None:
        return None
    return dict(name=config["name"], config=config, best=best, epochs=epoch, elapsed=elapsed)


def run(args):
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    torch.set_num_threads(args.jobs)
    args.output.mkdir(parents=True, exist_ok=True)
    from contextlib import contextmanager

    @contextmanager
    def locked():
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
                yield
            finally:
                lock.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    with locked():
        try:
            return run_locked(torch, args)
        except Exception as error:
            write_json(args.output / "status.json", dict(state="failed", error=str(error)))
            raise


def run_locked(torch, args):
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor
    output = args.output
    data, manifest = load_data(args.data)
    configs = trials(args.widths, args.seeds)
    settings = dict(schema="national-transfer-run-v1", manifest_sha256=sha256(args.data / "manifest.json"),
                    trainer_sha256=sha256(Path(__file__)), helper_sha256=sha256(Path(__file__).with_name("train_valley_neural.py")),
                    configs=configs, hours=args.hours, epochs=args.epochs, patience=args.patience,
                    batch_size=args.batch_size, device=args.device, jobs=args.jobs, tree_rows=args.tree_rows,
                    validation_only=args.validation_only)
    path = output / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("Incompatible resume settings/code/data; use a new output")
    write_json(path, settings)
    if (output / "summary.json").exists():
        summary = json.loads((output / "summary.json").read_text())
        for name, digest in summary["checksums"].items():
            if Path(name).name != name or sha256(output / name) != digest:
                raise ValueError(f"Completed result corrupted: {name}")
        print("Completed run verified; not rescoring test", flush=True)
        return summary
    groups = grouped(data)
    train = data["split"] == "train"
    w = balanced_weights(subset(data, train))
    mean = np.average(data["x"][train], axis=0, weights=w).astype(np.float32)
    std = np.sqrt(np.average((data["x"][train]-mean)**2, axis=0, weights=w)).astype(np.float32)
    std[std < 1e-6] = 1.
    norm = dict(mean=mean, std=std)
    np.savez(output / "normalization.npz", **norm)
    selection = data["split"] == "selection"
    fusion_selection = groups["split"] == "selection"
    baseline = data["source_c"]+data["forecast_delta_c"]
    design = np.column_stack([np.ones(len(data["x"])), data["x"][:, 5]])
    sw = np.sqrt(w)
    coefficients = np.linalg.lstsq(design[train]*sw[:, None],
                                  (data["target_c"][train]-data["source_c"][train])*sw, rcond=None)[0]
    fixed = float(np.average(data["target_c"][train]-data["source_c"][train], weights=w))
    pair_predictions = dict(copy_source=data["source_c"], constant_offset=data["source_c"]+fixed,
                            elevation_offset=data["source_c"]+design@coefficients, forecast_transfer=baseline)
    np.savez(output / "offset_baselines.npz", fixed=fixed, coefficients=coefficients)
    candidates, fusion_candidates = [], []
    for name, prediction in pair_predictions.items():
        candidates.append(dict(name=name, best=score(subset(data, selection), prediction[selection])["balanced_rmse_c"]))
        fusion_candidates.append(dict(name="mean_"+name, best=score(
            subset(groups, fusion_selection), fused_average(prediction, groups)[fusion_selection])["balanced_rmse_c"]))
    # Classical model is a bounded, weighted control and is eligible to win either task.
    tree_path = output / "weather_trees.joblib"
    tree_receipt = output / "weather_trees.json"
    if tree_receipt.exists():
        receipt = json.loads(tree_receipt.read_text())
        if sha256(tree_path) != receipt["sha256"]:
            raise ValueError("Tree checkpoint checksum mismatch")
        tree = joblib.load(tree_path)
    else:
        started = time.monotonic()
        ids = np.flatnonzero(train)
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(ids), min(args.tree_rows, len(ids)), replace=False)
        tree = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=16, max_depth=18,
                                   random_state=42, n_jobs=args.jobs)
        tree.fit(data["x"][ids[chosen]], data["target_c"][ids[chosen]]-baseline[ids[chosen]],
                 sample_weight=w[chosen]/w[chosen].mean())
        joblib.dump(tree, tree_path)
        write_json(tree_receipt, dict(sha256=sha256(tree_path), elapsed=time.monotonic()-started,
                                     fitted_rows=len(chosen)))
    spent = json.loads(tree_receipt.read_text())["elapsed"]
    tree_selection = baseline[selection]+tree.predict(data["x"][selection])
    candidates.append(dict(name="weather_trees", best=score(subset(data, selection), tree_selection)["balanced_rmse_c"]))
    # Infer only train/selection at this stage; calibration/test scores stay unopened until selection.
    selection_pair_ids = np.flatnonzero(selection)
    group_selection_ids = groups["index"][fusion_selection]
    all_selection_prediction = baseline.copy()
    all_selection_prediction[selection_pair_ids] = tree_selection
    fusion_candidates.append(dict(name="mean_weather_trees", best=score(subset(groups, fusion_selection),
        (all_selection_prediction[np.maximum(group_selection_ids, 0)]*(group_selection_ids >= 0)).sum(1)/
        (group_selection_ids >= 0).sum(1))["balanced_rmse_c"]))
    for config in configs:
        checkpoint = output / (config["name"]+".pt")
        if checkpoint.exists():
            spent += torch.load(checkpoint, map_location="cpu", weights_only=False)["elapsed"]
    deadline = time.monotonic()+max(0, args.hours*3600-spent)
    for number, config in enumerate(configs, 1):
        write_json(output / "status.json", dict(state="training", trial=number, trials=len(configs), name=config["name"]))
        result = fit_trial(torch, data, groups, manifest, config, args, output, norm, deadline)
        if result is not None:
            (fusion_candidates if config["family"] == "fusion" else candidates).append(result)
    pair_winner = min(candidates, key=lambda c: (c["best"], c["name"]))
    fusion_winner = min(fusion_candidates, key=lambda c: (c["best"], c["name"]))
    if args.validation_only:
        files = ["run_config.json", "normalization.npz", "offset_baselines.npz",
                 "weather_trees.joblib", "weather_trees.json"]
        summary = dict(state="validation_smoke_complete", pair_selected=pair_winner, fusion_selected=fusion_winner,
                       pair_validation=candidates, fusion_validation=fusion_candidates,
                       no_holdout_evaluation="Calibration/test predictions and metrics not computed",
                       checksums={name: sha256(output / name) for name in files})
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", dict(state="validation_smoke_complete"))
        print("Validation-only smoke complete; calibration/test not evaluated", flush=True)
        return summary
    def selected_prediction(winner, fusion=False):
        name = winner["name"]
        if name.startswith("mean_"):
            base = selected_prediction(dict(name=name[5:]))
            return fused_average(base, groups)
        if name in pair_predictions:
            return pair_predictions[name]
        if name == "weather_trees":
            return baseline+tree.predict(data["x"])
        config = winner["config"]
        model = (fusion_model(torch, config["width"]) if fusion else create_model(torch, config, manifest)).to(args.device)
        saved = torch.load(output / (name+".pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(saved["best_state"])
        atomic_torch_save(torch, saved["best_state"], output / ("selected_fusion.pt" if fusion else "selected_pair.pt"))
        x = ((data["x"]-mean)/std).astype(np.float32)
        anchor, mask = baseline, None
        if fusion:
            ids = np.maximum(groups["index"], 0)
            x, anchor, mask = x[ids], baseline[ids], torch.from_numpy(groups["mask"]).to(args.device)
        return infer(torch, model, torch.from_numpy(x).to(args.device),
                     torch.from_numpy(anchor.astype(np.float32)).to(args.device), mask,
                     np.arange(len(x)), args.batch_size, fusion)
    pair_prediction = selected_prediction(pair_winner)
    fusion_prediction = selected_prediction(fusion_winner, True)
    reports = {}
    for task, meta, pred, base in (
            ("pair", data, pair_prediction, baseline),
            ("fusion", groups, fusion_prediction, fused_average(baseline, groups))):
        calibration = meta["split"] == "calibration"
        test = meta["split"] == "test"
        r = calibrate(subset(meta, calibration), pred[calibration])
        error = np.abs(pred[test]-meta["target_c"][test])
        night = pd.Series(error, index=meta["night"][test]).groupby(level=0).max().to_numpy()
        reports[task] = dict(test=score(subset(meta, test), pred[test]),
                             baseline_test=score(subset(meta, test), base[test]),
                             interval_radius_c=r, empirical_night_coverage=float((night <= r).mean()))
        frame = pd.DataFrame({k: meta[k][test] for k in ("target", "time", "night", "block", "target_c")})
        if task == "pair":
            frame["source"] = data["source"][test]
        else:
            frame["source_count"] = groups["mask"][test].sum(1)
            frame["sources"] = [";".join(data["source"][ids[ids >= 0]]) for ids in groups["index"][test]]
        frame["prediction_c"], frame["baseline_prediction_c"] = pred[test], base[test]
        frame.to_csv(output / f"selected_{task}_test.csv", index=False)
    files = ["normalization.npz", "offset_baselines.npz", "run_config.json", "weather_trees.joblib",
             "weather_trees.json", "selected_pair_test.csv", "selected_fusion_test.csv"]
    files += [p.name for p in output.glob("selected_*.pt")]
    elapsed = json.loads(tree_receipt.read_text())["elapsed"]
    elapsed += sum(c.get("elapsed", 0.) for c in candidates+fusion_candidates)
    summary = dict(state="complete", pair_selected=pair_winner, fusion_selected=fusion_winner,
                   pair_validation=candidates, fusion_validation=fusion_candidates, reports=reports,
                   measured_training_hours=elapsed/3600, budget_exhausted=time.monotonic() >= deadline,
                   selection_policy="Separate pair/fusion winners,minimum equal-block target-night selection RMSE; no test selection.",
                   limitations=manifest["limitations"],
                   checksums={name: sha256(output / name) for name in files})
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", dict(state="complete", pair=pair_winner["name"], fusion=fusion_winner["name"]))
    print(json.dumps(dict(pair=pair_winner["name"], fusion=fusion_winner["name"], reports=reports), indent=2), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("national_transfer_results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hours", type=float, default=40.)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    parser.add_argument("--tree-rows", type=int, default=300000)
    parser.add_argument("--validation-only", action="store_true", help="Smoke mode: no calibration/test evaluation")
    args = parser.parse_args(argv)
    if (args.hours <= 0 or not math.isfinite(args.hours) or
            min(args.epochs, args.patience, args.batch_size, args.jobs, args.tree_rows, *args.widths) <= 0 or
            len(set(args.widths)) != len(args.widths) or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Use positive finite settings and unique widths/seeds")
    return args


if __name__ == "__main__":
    run(parse_args())
