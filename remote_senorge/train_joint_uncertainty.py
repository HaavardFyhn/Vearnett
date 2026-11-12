"""Joint fusion residual intervals and leave-one-reading-out network diagnostics."""
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

if __package__:
    from .train_national_transfer import load_data, grouped, subset, balanced_weights, fusion_model
    from .train_national_uncertainty import (
        normalize, target_fold_masks, fold_masks, simultaneous_factor, interval_diagnostics,
        residual_diagnostics, parse_args,
    )
    from .train_valley_neural import atomic_torch_save, write_json, sha256
else:
    from train_national_transfer import load_data, grouped, subset, balanced_weights, fusion_model
    from train_national_uncertainty import (
        normalize, target_fold_masks, fold_masks, simultaneous_factor, interval_diagnostics,
        residual_diagnostics, parse_args,
    )
    from train_valley_neural import atomic_torch_save, write_json, sha256


ERROR_FEATURES = [f"{stat}_feature_{i}" for stat in ("mean", "minimum", "maximum") for i in range(39)]
ERROR_FEATURES += ["joint_prediction_c", "anchor_std_c", "source_count"]


def available_groups(groups, pair_mask):
    result = {key: value.copy() for key, value in groups.items()}
    result["mask"] &= pair_mask[np.maximum(result["index"], 0)]
    keep = result["mask"].any(1)
    result = subset(result, keep)
    result["index"][~result["mask"]] = -1
    return result


def batch_inputs(torch, data, groups, ids, mean, std, device):
    indices = np.maximum(groups["index"][ids], 0)
    mask = groups["mask"][ids]
    x = data["x"][indices]
    anchor = data["source_c"][indices]+data["forecast_delta_c"][indices]
    return (
        torch.from_numpy(((x-mean)/std).astype(np.float32)).to(device),
        torch.from_numpy(anchor.astype(np.float32)).to(device),
        torch.from_numpy(mask).to(device),
    )


def joint_error_features(x, anchor, mask, prediction):
    """Permutation-invariant inference inputs; no target temperature or observed residual."""
    count = mask.sum(1)
    if (not count.size or (count < 1).any() or x.shape != (*mask.shape, 39) or
            anchor.shape != mask.shape or prediction.shape != count.shape or
            not np.isfinite(x[mask]).all() or not np.isfinite(anchor[mask]).all() or
            not np.isfinite(prediction).all()):
        raise ValueError("Invalid joint error-feature inputs")
    values = np.where(mask[:, :, None], x, 0)
    average = values.sum(1)/count[:, None]
    low = np.where(mask[:, :, None], x, np.inf).min(1)
    high = np.where(mask[:, :, None], x, -np.inf).max(1)
    anchor_mean = np.where(mask, anchor, 0).sum(1)/count
    spread = np.sqrt(np.where(mask, (anchor-anchor_mean[:, None])**2, 0).sum(1)/count)
    return np.column_stack([average, low, high, prediction, spread, count]).astype(np.float32)


def scenarios(torch, model, data, groups, mean, std, args):
    """Full joint input plus each valid deletion, without replacing the deleted source."""
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(groups["index"]), args.batch_size):
            ids = np.arange(start, min(start+args.batch_size, len(groups["index"])))
            x, anchor, mask = batch_inputs(torch, data, groups, ids, mean, std, args.device)
            for removed_slot in range(-1, 6):
                current = mask.clone()
                eligible = np.ones(len(ids), bool)
                if removed_slot >= 0:
                    eligible = (groups["mask"][ids, removed_slot] &
                                (groups["mask"][ids].sum(1) > 1))
                    current[:, removed_slot] = False
                local = np.flatnonzero(eligible)
                if not len(local):
                    continue
                prediction = model(x[local], anchor[local], current[local]).cpu().numpy()
                pair_ids = np.maximum(groups["index"][ids[local]], 0)
                features = joint_error_features(
                    data["x"][pair_ids], anchor[local].cpu().numpy(), current[local].cpu().numpy(), prediction)
                removed = (np.full(len(local), "", dtype=data["source"].dtype) if removed_slot < 0 else
                           data["source"][pair_ids[:, removed_slot]])
                outputs.append(dict(group=ids[local], prediction=prediction, features=features, removed=removed))
    if not outputs:
        raise ValueError("No joint configurations to evaluate")
    return {key: np.concatenate([part[key] for part in outputs]) for key in outputs[0]}


def scenario_metadata(groups, result):
    metadata = {key: groups[key][result["group"]] for key in
                ("target", "time", "night", "block", "target_c", "split", "calm_clear")}
    metadata["x"] = result["features"][:, :39]
    return metadata


def refit(torch, data, groups, fold, spec, args, deadline):
    fitting = available_groups(groups, fold["fit"])
    evaluation = available_groups(groups, fold["evaluate"])
    if min(len(fitting["index"]), len(evaluation["index"])) < 2:
        raise ValueError("Insufficient joint groups after held-source removal")
    mean, std, _ = normalize(data, fold["fit"])
    weights = balanced_weights(fitting)
    weights = (weights/weights.mean()).astype(np.float32)
    torch.manual_seed(spec["config"]["seed"]+fold["fold"])
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec["config"]["seed"]+fold["fold"])
    model = fusion_model(torch, spec["config"]["width"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    path = args.output / f"{fold['scheme']}_fold{fold['fold']}.pt"
    epoch, elapsed = 0, 0.
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, elapsed = saved["epoch"], saved["elapsed"]
        torch.set_rng_state(saved["rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for epoch in range(epoch+1, spec["epochs"]+1):
        if time.monotonic() >= deadline:
            return None
        started = time.monotonic()
        model.train()
        order = torch.randperm(len(fitting["index"])).numpy()
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start+args.batch_size]
            x, anchor, mask = batch_inputs(torch, data, fitting, ids, mean, std, args.device)
            prediction = model(x, anchor, mask)
            label = torch.from_numpy(fitting["target_c"][ids].astype(np.float32)).to(args.device)
            weight = torch.from_numpy(weights[ids]).to(args.device)
            loss = (torch.nn.functional.huber_loss(prediction, label, delta=2., reduction="none")*weight).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite joint refit loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        elapsed += time.monotonic()-started
        atomic_torch_save(torch, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          epoch=epoch, elapsed=elapsed, rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []), path)
        print(f"joint {fold['scheme']} fold{fold['fold']} epoch{epoch}/{spec['epochs']}", flush=True)
    result = scenarios(torch, model, data, evaluation, mean, std, args)
    return evaluation, result


def contribution_table(metadata, result, radii):
    frame = pd.DataFrame({key: metadata[key] for key in ("target", "time", "night", "block", "target_c")})
    frame["removed_source"] = result["removed"]
    frame["prediction_c"], frame["radius_c"] = result["prediction"], radii
    full = frame.loc[frame.removed_source.eq(""), ["target", "time", "prediction_c", "radius_c"]]
    if full.duplicated(["target", "time"]).any():
        raise ValueError("Duplicate full joint predictions")
    full = full.rename(columns={"prediction_c": "full_prediction_c", "radius_c": "full_radius_c"})
    frame = frame.merge(full, on=["target", "time"], how="left", validate="many_to_one")
    if frame["full_radius_c"].isna().any():
        raise ValueError("Deletion scenario has no full reference")
    frame["radius_change_when_removed_c"] = frame["radius_c"]-frame["full_radius_c"]
    frame["temperature_change_when_removed_c"] = frame["prediction_c"]-frame["full_prediction_c"]
    frame["lower_c"], frame["upper_c"] = frame["prediction_c"]-frame["radius_c"], frame["prediction_c"]+frame["radius_c"]
    return frame


def run_locked(torch, args):
    data, manifest = load_data(args.data)
    spec = json.loads((args.data / "joint_spec.json").read_text())
    if (spec["schema"] != "joint-residual-training-v1" or spec["config"]["family"] != "fusion" or
            type(spec["epochs"]) is not int or spec["epochs"] < 1):
        raise ValueError("Invalid fixed joint specification")
    settings = dict(schema="joint-uncertainty-run-v1", settings={key: str(value) for key, value in vars(args).items()},
                    inputs={name: sha256(args.data / name) for name in
                            ("training.npz", "manifest.json", "joint_spec.json", "selected_fusion.pt",
                             "selected_normalization.npz")},
                    code={name: sha256(Path(__file__).with_name(name)) for name in
                          ("train_joint_uncertainty.py", "train_national_transfer.py",
                           "train_national_uncertainty.py", "train_valley_neural.py")})
    config_path = args.output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("Changed joint run settings/code/data; use a fresh output")
    write_json(config_path, settings)
    if (args.output / "report.json").exists():
        report = json.loads((args.output / "report.json").read_text())
        for name, digest in report["checksums"].items():
            if Path(name).name != name or sha256(args.output / name) != digest:
                raise ValueError("Completed joint result checksum mismatch")
        print("Completed joint result verified; no rescoring", flush=True)
        return report
    groups = grouped(data)
    elapsed = sum(torch.load(path, map_location="cpu", weights_only=False)["elapsed"]
                  for path in args.output.glob("*_fold*.pt"))
    receipt_path = args.output / "joint_scale.json"
    if receipt_path.exists():
        elapsed += json.loads(receipt_path.read_text())["elapsed"]
    deadline = time.monotonic()+max(0, args.hours*3600-elapsed)
    diagnostics, samples = [], []
    for scheme, folds in (("target", target_fold_masks(data, args.folds)),
                           ("region", fold_masks(data, spec["station_xy"], args.folds))):
        for fold in folds:
            write_json(args.output / "status.json", dict(state="cross_fitting", scheme=scheme, fold=fold["fold"]))
            completed = refit(torch, data, groups, fold, spec, args, deadline)
            if completed is None:
                write_json(args.output / "status.json", dict(state="budget_exhausted", ready=False))
                return None
            evaluation, result = completed
            meta = scenario_metadata(evaluation, result)
            full = result["removed"] == ""
            error = result["prediction"]-meta["target_c"]
            diagnostics.append(dict(scheme=scheme, fold=fold["fold"], full_groups=int(full.sum()),
                                    configurations=len(error), targets=len(set(meta["target"])),
                                    blocks=len(set(meta["block"])), full_rmse_c=float(np.sqrt(np.mean(error[full]**2))),
                                    deletion_rmse_c=float(np.sqrt(np.mean(error[~full]**2))) if (~full).any() else None,
                                    held_targets=fold.get("held_targets", []), held_blocks=fold.get("held_blocks", []),
                                    fit_pairs=int(fold["fit"].sum()), evaluated_pairs=int(fold["evaluate"].sum()),
                                    full_errors=residual_diagnostics(meta, np.flatnonzero(full), result["prediction"][full])))
            if scheme == "target":
                # Stratified equal-fold sampling,then target-night/block balancing within the sample.
                rng = np.random.default_rng(2026+fold["fold"])
                ids = rng.choice(len(error), min(len(error), args.tree_rows//args.folds), replace=False)
                samples.append(dict(x=result["features"][ids], y=np.abs(error[ids]),
                                    **{key: meta[key][ids] for key in ("target", "night", "block")}))
    scale_path = args.output / "joint_scale.joblib"
    if receipt_path.exists():
        if sha256(scale_path) != json.loads(receipt_path.read_text())["sha256"]:
            raise ValueError("Joint error-scale checkpoint checksum mismatch")
        scale_model = joblib.load(scale_path)
    else:
        if time.monotonic() >= deadline:
            write_json(args.output / "status.json", dict(state="budget_exhausted", ready=False))
            return None
        sample = {key: np.concatenate([part[key] for part in samples]) for key in samples[0]}
        weights = balanced_weights(sample)
        started = time.monotonic()
        scale_model = ExtraTreesRegressor(n_estimators=128, max_depth=16, min_samples_leaf=64,
                                          random_state=2026, n_jobs=args.jobs)
        scale_model.fit(sample["x"], sample["y"], sample_weight=weights/weights.mean())
        joblib.dump(scale_model, scale_path)
        np.savez_compressed(args.output / "joint_oof_scale_sample.npz", **sample)
        write_json(receipt_path, dict(sha256=sha256(scale_path), elapsed=time.monotonic()-started,
                                     rows=len(sample["y"]), features=ERROR_FEATURES))
    model = fusion_model(torch, spec["config"]["width"]).to(args.device)
    model.load_state_dict(torch.load(args.data / "selected_fusion.pt", map_location="cpu", weights_only=True))
    with np.load(args.data / "selected_normalization.npz", allow_pickle=False) as norm:
        mean, std = norm["mean"], norm["std"]
    calibration = subset(groups, groups["split"] == "calibration")
    cal = scenarios(torch, model, data, calibration, mean, std, args)
    meta = scenario_metadata(calibration, cal)
    scales = np.maximum(.1, scale_model.predict(cal["features"]))
    factor = simultaneous_factor(np.abs(cal["prediction"]-meta["target_c"]), scales, meta["night"])
    test_groups = subset(groups, groups["split"] == "test")
    test = scenarios(torch, model, data, test_groups, mean, std, args)
    meta = scenario_metadata(test_groups, test)
    radii = np.maximum(.1, scale_model.predict(test["features"]))*factor
    contribution_table(meta, test, radii).to_csv(args.output / "test_contributions.csv", index=False)
    full = test["removed"] == ""
    report = dict(schema="joint-residual-results-v1", state="complete", factor=factor, scale_floor_c=.1,
                  error_features=ERROR_FEATURES, folds=diagnostics, calibration_nights=len(set(calibration["night"])),
                  full_joint_test=interval_diagnostics(subset(meta, full), np.arange(full.sum()),
                                                       test["prediction"][full], radii[full]),
                  full_and_deletions_test=interval_diagnostics(meta, np.arange(len(radii)), test["prediction"], radii),
                  uncertainty_map_validated=False, limitations=[
                      "Already-inspected test; only two calibration and two test blocks. Exploratory,not national certification.",
                      "Calibration targets simultaneous evaluated joint locations AND valid single-source-deletion scenarios per night.",
                      "Source-removal radius changes are model diagnostics,not causal value or additive uncertainty contributions.",
                      "No replacement of a removed source; one remaining source is allowed; removal of the sole source gives NoData.",
                      "At most six connected sources; new catalogue sources,slider values and unobserved raster cells are extrapolations.",
                      "No independence assumption,no division by sqrt(number of sources),no test-based model selection.",
                      "Scale fits target OOF configurations only; buffered region refits are separate diagnostics.",
                  ], checksums={})
    counts = test["features"][:, -1]
    report["by_source_count"] = {
        str(int(count)): interval_diagnostics(subset(meta, counts == count), np.arange(int((counts == count).sum())),
                                             test["prediction"][counts == count], radii[counts == count])
        for count in np.unique(counts)}
    for name in ("run_config.json", "joint_scale.joblib", "joint_scale.json", "test_contributions.csv",
                 "joint_oof_scale_sample.npz"):
        report["checksums"][name] = sha256(args.output / name)
    write_json(args.output / "report.json", report)
    write_json(args.output / "status.json", dict(state="complete", uncertainty_map_validated=False))
    print(json.dumps(report["full_and_deletions_test"], indent=2), flush=True)
    return report


def run(args):
    import torch
    import os
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    if args.tree_rows < args.folds:
        raise ValueError("Tree sample budget must cover every target fold")
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


if __name__ == "__main__":
    run(parse_args())
