"""Geographic cross-fit residual scale, calibrated to simultaneous station-night coverage."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

if __package__:
    from .train_national_transfer import load_data, balanced_weights, subset
    from .train_valley_neural import sha256, write_json, atomic_torch_save, create_model
else:
    from train_national_transfer import load_data, balanced_weights, subset
    from train_valley_neural import sha256, write_json, atomic_torch_save, create_model


def fold_masks(data, geometry, folds=5):
    """Leave geographic blocks out with a10km buffer; never train on held-out source observations."""
    train = data["split"] == "train"
    blocks = sorted(set(data["block"][train]), key=lambda b: sha256_text(b))
    if len(blocks) < folds:
        raise ValueError("Too few training geographic blocks for cross-fitting")
    assignment = {block: i % folds for i, block in enumerate(blocks)}
    groups = []
    for fold in range(folds):
        held = [block for block in blocks if assignment[block] == fold]
        excluded = set()
        for block in held:
            bx, by = map(int, block.split(":"))
            left, bottom = bx*100000, by*100000
            for station, xy in geometry.items():
                dx = max(left-xy[0], xy[0]-(left+100000), 0)
                dy = max(bottom-xy[1], xy[1]-(bottom+100000), 0)
                if np.hypot(dx, dy) <= 10000:
                    excluded.add(station)
        allowed_source = ~np.isin(data["source"], list(excluded))
        allowed_target = ~np.isin(data["target"], list(excluded))
        fit = train & allowed_source & allowed_target & ~np.isin(data["block"], held)
        evaluate = train & np.isin(data["block"], held) & allowed_source
        if fit.sum() < 2 or evaluate.sum() < 2:
            raise ValueError(f"Fold{fold}has insufficient fitting or cross-fit pairs; do not relax buffers silently")
        groups.append(dict(fold=fold, scheme="region", held_blocks=held, fit=fit, evaluate=evaluate))
    return groups


def target_fold_masks(data, folds=5):
    """Withhold targets everywhere; their observations cannot enter fitting as sources."""
    train = data["split"] == "train"
    frame = pd.DataFrame(dict(target=data["target"][train], block=data["block"][train])).drop_duplicates()
    if frame["target"].duplicated().any() or frame["target"].nunique() < folds:
        raise ValueError("Invalid or insufficient target station geography")
    assignment = {}
    # Distribute held stations across geography rather than withholding entire blocks.
    offset = 0
    for block, rows in frame.groupby("block", sort=True):
        targets = sorted(rows["target"], key=sha256_text)
        for i, target in enumerate(targets):
            assignment[target] = (offset+i) % folds
        offset += len(targets)
    result = []
    for fold in range(folds):
        held = [station for station, number in assignment.items() if number == fold]
        held_source = np.isin(data["source"], held)
        held_target = np.isin(data["target"], held)
        fit = train & ~held_source & ~held_target
        evaluate = train & held_target & ~held_source
        if fit.sum() < 2 or evaluate.sum() < 2:
            raise ValueError(f"Target fold{fold}has insufficient available-source pairs")
        result.append(dict(fold=fold, scheme="target", held_targets=held, fit=fit, evaluate=evaluate))
    return result


def residual_diagnostics(data, ids, prediction):
    """Descriptive OOF errors,not in-sample error-scale validation or coverage claims."""
    error = prediction-data["target_c"][ids]
    distance = data["x"][ids, 4]
    stamps = pd.to_datetime(data["time"][ids], utc=True)
    winter = stamps.year-(stamps.month < 12).astype(int)
    frame = pd.DataFrame(dict(block=data["block"][ids], target=data["target"][ids],
                              night=data["night"][ids], winter=winter, squared_error=error**2,
                              distance_band=np.where(distance <= 25, "0-25km",
                                            np.where(distance <= 50, "25-50km", "50-100km"))))
    report = dict(rows=len(ids), targets=frame["target"].nunique(), blocks=frame["block"].nunique(),
                  rmse_c=float(np.sqrt(np.mean(error**2))),
                  comparison="Descriptive cross-fit mean-predictor errors; no held-out scale coverage claimed")
    for dimension in ("block", "winter", "distance_band"):
        report[dimension] = {
            str(key): dict(rows=len(part), targets=part["target"].nunique(), nights=part["night"].nunique(),
                           rmse_c=float(np.sqrt(part["squared_error"].mean())))
            for key, part in frame.groupby(dimension)}
    for name, mask in (("cold", data["target_c"][ids] <= -5),
                       ("calm_clear", data["calm_clear"][ids])):
        report[name] = dict(rows=int(mask.sum()), rmse_c=float(np.sqrt(np.mean(error[mask]**2)))
                           if mask.any() else None)
    return report


def interval_diagnostics(data, ids, prediction, radii):
    error = np.abs(prediction-data["target_c"][ids])
    covered = error <= radii
    frame = pd.DataFrame(dict(block=data["block"][ids], target=data["target"][ids], night=data["night"][ids],
                              covered=covered, radius=radii,
                              distance_band=np.where(data["x"][ids, 4] <= 25, "0-25km",
                                            np.where(data["x"][ids, 4] <= 50, "25-50km", "50-100km"))))

    def summarize(part):
        return dict(rows=len(part), targets=part["target"].nunique(), nights=part["night"].nunique(),
                    night_coverage=float(part.groupby("night")["covered"].all().mean()),
                    pair_coverage=float(part["covered"].mean()),
                    radius_median_c=float(part["radius"].median()),
                    radius_p90_c=float(part["radius"].quantile(.9)))

    report = summarize(frame)
    for dimension in ("block", "target", "distance_band"):
        report[dimension] = {str(key): summarize(part) for key, part in frame.groupby(dimension)}
    for name, mask in (("cold", data["target_c"][ids] <= -5), ("calm_clear", data["calm_clear"][ids])):
        report[name] = summarize(frame.loc[mask]) if mask.any() else dict(rows=0)
    return report


def sha256_text(text):
    import hashlib
    return hashlib.sha256(f"uncertainty-v1-2026-{text}".encode()).hexdigest()


def normalize(data, fit):
    weights = balanced_weights(subset(data, fit))
    mean = np.average(data["x"][fit], axis=0, weights=weights).astype(np.float32)
    std = np.sqrt(np.average((data["x"][fit]-mean)**2, axis=0, weights=weights)).astype(np.float32)
    std[std < 1e-6] = 1
    return mean, std, (weights/weights.mean()).astype(np.float32)


def predict(torch, model, data, indices, mean, std, device, batch_size):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            x = torch.from_numpy(((data["x"][ids]-mean)/std).astype(np.float32)).to(device)
            correction = model(x).cpu().numpy()
            result.append(data["source_c"][ids]+data["forecast_delta_c"][ids]+correction)
    return np.concatenate(result)


def error_features(x, prediction):
    # Inputs available at inference, never the target observation or observed error.
    return np.column_stack([x, prediction]).astype(np.float32)


def simultaneous_factor(errors, scales, nights, coverage=.9):
    errors, scales = np.asarray(errors), np.asarray(scales)
    if (not np.isfinite(errors).all() or not np.isfinite(scales).all() or
            (errors < 0).any() or (scales <= 0).any()):
        raise ValueError("Invalid calibration error/scale")
    maxima = pd.Series(errors/scales, index=nights).groupby(level=0).max().to_numpy()
    rank = int(math.ceil((len(maxima)+1)*coverage))
    if rank > len(maxima):
        raise ValueError("Insufficient calibration nights for finite simultaneous factor")
    return float(np.partition(maxima, rank-1)[rank-1])


def refit_fold(torch, data, manifest, spec, fold, args, deadline):
    path = args.output / f"{fold['scheme']}_fold{fold['fold']}.pt"
    ids = np.flatnonzero(fold["fit"])
    mean, std, weights = normalize(data, fold["fit"])
    torch.manual_seed(spec["config"]["seed"]+fold["fold"])
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(spec["config"]["seed"]+fold["fold"])
    model = create_model(torch, spec["config"], manifest).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    epoch, elapsed = 0, 0.
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, elapsed = saved["epoch"], saved["elapsed"]
        torch.set_rng_state(saved["rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for next_epoch in range(epoch+1, spec["epochs"]+1):
        if time.monotonic() >= deadline:
            return None
        started = time.monotonic()
        model.train()
        order = torch.randperm(len(ids)).numpy()
        for start in range(0, len(ids), args.batch_size):
            local = order[start:start+args.batch_size]
            batch = ids[local]
            x = torch.from_numpy(((data["x"][batch]-mean)/std).astype(np.float32)).to(args.device)
            label = torch.from_numpy((data["target_c"][batch]-data["source_c"][batch]-
                                      data["forecast_delta_c"][batch]).astype(np.float32)).to(args.device)
            weight = torch.from_numpy(weights[local]).to(args.device)
            loss = (torch.nn.functional.huber_loss(model(x), label, delta=2., reduction="none")*weight).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite cross-fit loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        elapsed += time.monotonic()-started
        atomic_torch_save(torch, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          epoch=next_epoch, elapsed=elapsed, rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []), path)
        print(f"{fold['scheme']} cross-fit fold{fold['fold']} epoch{next_epoch}/{spec['epochs']}", flush=True)
    evaluation = np.flatnonzero(fold["evaluate"])
    return evaluation, predict(torch, model, data, evaluation, mean, std, args.device, args.batch_size)


def run_locked(args):
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    torch.set_num_threads(args.jobs)
    data, manifest = load_data(args.data)
    spec = json.loads((args.data / "uncertainty_spec.json").read_text())
    if (spec["config"]["family"] != "forecast_residual" or
            type(spec["epochs"]) is not int or spec["epochs"] < 1):
        raise ValueError("Invalid fixed selected cross-fit specification")
    if spec.get("schema") not in (None, "national-crossfit-error-v2"):
        raise ValueError("Use the updated dual-crossfit v2 package,not the earlier regional-only specification")
    geometry = {key: tuple(xy) for key, xy in spec["station_xy"].items()}
    settings = dict(schema="national-uncertainty-run-v2", manifest=sha256(args.data / "manifest.json"),
                    training_sha256=sha256(args.data / "training.npz"),
                    selected_weights_sha256=sha256(args.data / "selected_pair.pt"),
                    normalization_sha256=sha256(args.data / "selected_normalization.npz"),
                    spec=sha256(args.data / "uncertainty_spec.json"), code=sha256(Path(__file__)),
                    transfer_code=sha256(Path(__file__).with_name("train_national_transfer.py")),
                    neural_code=sha256(Path(__file__).with_name("train_valley_neural.py")),
                    folds=args.folds, hours=args.hours, batch_size=args.batch_size, jobs=args.jobs,
                    device=args.device, tree_rows=args.tree_rows)
    output = args.output
    path = output / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("Changed cross-fit run settings/code/data; use a fresh output")
    write_json(path, settings)
    if (output / "report.json").exists():
        report = json.loads((output / "report.json").read_text())
        for name, digest in report["checksums"].items():
            if Path(name).name != name or sha256(output / name) != digest:
                raise ValueError("Completed uncertainty output checksum mismatch")
        print("Completed uncertainty results verified; no re-evaluation", flush=True)
        return report
    spent = sum(torch.load(p, map_location="cpu", weights_only=False)["elapsed"]
                for p in output.glob("*_fold*.pt"))
    if (output / "error_scale.json").exists():
        spent += json.loads((output / "error_scale.json").read_text())["elapsed"]
    deadline = time.monotonic()+max(0, args.hours*3600-spent)
    fold_ids = np.full(len(data["target_c"]), -1, np.int16)
    oof = np.full(len(fold_ids), np.nan, np.float32)
    target_folds = target_fold_masks(data, args.folds)
    region_folds = fold_masks(data, geometry, args.folds)
    diagnostics, fold_report = {}, []
    for scheme, folds in (("target", target_folds), ("region", region_folds)):
        scheme_fold = np.full(len(fold_ids), -1, np.int16)
        scheme_oof = np.full(len(oof), np.nan, np.float32)
        for fold in folds:
            write_json(output / "status.json", dict(state="cross_fitting", scheme=scheme,
                       fold=fold["fold"], rounds_per_scheme=args.folds, total_rounds=2*args.folds))
            result = refit_fold(torch, data, manifest, spec, fold, args, deadline)
            if result is None:
                write_json(output / "status.json", dict(state="budget_exhausted", ready=False,
                           detail="Incomplete dual cross-fit; no uncertainty model emitted. New budget needs new output."))
                return None
            ids, values = result
            if (scheme_fold[ids] >= 0).any():
                raise ValueError("Overlapping cross-fit evaluations")
            scheme_fold[ids], scheme_oof[ids] = fold["fold"], values
            fold_report.append(dict(scheme=scheme, fold=fold["fold"],
                                    held_blocks=fold.get("held_blocks", []),
                                    held_targets=fold.get("held_targets", []),
                                    fit_rows=int(fold["fit"].sum()), evaluated_rows=len(ids)))
        eligible_scheme = scheme_fold >= 0
        ids = np.flatnonzero(eligible_scheme)
        diagnostics[scheme] = residual_diagnostics(data, ids, scheme_oof[ids])
        diagnostics[scheme]["unavailable_source_pairs"] = int(
            ((data["split"] == "train") & ~eligible_scheme).sum())
        train = data["split"] == "train"
        diagnostics[scheme]["training_pair_fraction"] = float(eligible_scheme.sum()/train.sum())
        diagnostics[scheme]["targets_without_evaluation"] = sorted(
            set(data["target"][train])-set(data["target"][ids]))
        diagnostics[scheme]["original_training_distance_bands"] = {
            name: int(mask.sum()) for name, mask in (
                ("0-25km", train & (data["x"][:, 4] <= 25)),
                ("25-50km", train & (data["x"][:, 4] > 25) & (data["x"][:, 4] <= 50)),
                ("50-100km", train & (data["x"][:, 4] > 50)))}
        np.savez_compressed(output / f"{scheme}_crossfit_residuals.npz", row=ids,
                            fold=scheme_fold[ids], prediction_c=scheme_oof[ids],
                            residual_c=scheme_oof[ids]-data["target_c"][ids])
        if scheme == "target":
            fold_ids, oof = scheme_fold, scheme_oof
    eligible = fold_ids >= 0
    if not eligible.any():
        raise ValueError("No geographic out-of-fold residuals")
    tree_path, tree_receipt = output / "error_scale.joblib", output / "error_scale.json"
    if tree_receipt.exists():
        receipt = json.loads(tree_receipt.read_text())
        if sha256(tree_path) != receipt["sha256"]:
            raise ValueError("Error-scale checkpoint corrupt")
        scale_model = joblib.load(tree_path)
    else:
        if time.monotonic() >= deadline:
            write_json(output / "status.json", dict(state="budget_exhausted", ready=False))
            return None
        started = time.monotonic()
        indices = np.flatnonzero(eligible)
        rng = np.random.default_rng(2026)
        chosen = rng.choice(len(indices), min(len(indices), args.tree_rows), replace=False)
        ids = indices[chosen]
        w = balanced_weights(subset(data, eligible))[chosen]
        scale_model = ExtraTreesRegressor(n_estimators=128, max_depth=16, min_samples_leaf=64,
                                           random_state=2026, n_jobs=args.jobs)
        scale_model.fit(error_features(data["x"][ids], oof[ids]),
                        np.abs(oof[ids]-data["target_c"][ids]), sample_weight=w/w.mean())
        joblib.dump(scale_model, tree_path)
        write_json(tree_receipt, dict(sha256=sha256(tree_path), elapsed=time.monotonic()-started, rows=len(ids)))
    # The production mean predictor remains the original validation-selected model.
    model = create_model(torch, spec["config"], manifest).to(args.device)
    model.load_state_dict(torch.load(args.data / "selected_pair.pt", map_location="cpu", weights_only=True))
    with np.load(args.data / "selected_normalization.npz", allow_pickle=False) as norm:
        mean, std = norm["mean"], norm["std"]
    calibration = data["split"] == "calibration"
    cal_ids = np.flatnonzero(calibration)
    cal_pred = predict(torch, model, data, cal_ids, mean, std, args.device, args.batch_size)
    cal_scale = np.maximum(.1, scale_model.predict(error_features(data["x"][cal_ids], cal_pred)))
    factor = simultaneous_factor(np.abs(cal_pred-data["target_c"][cal_ids]), cal_scale, data["night"][cal_ids])
    # No tuning after this exploratory test evaluation.
    test_ids = np.flatnonzero(data["split"] == "test")
    test_pred = predict(torch, model, data, test_ids, mean, std, args.device, args.batch_size)
    scale = np.maximum(.1, scale_model.predict(error_features(data["x"][test_ids], test_pred)))
    radii = scale*factor
    error = np.abs(test_pred-data["target_c"][test_ids])
    covered = error <= radii
    frame = pd.DataFrame({k: data[k][test_ids] for k in ("source", "target", "time", "night", "block", "target_c")})
    frame["prediction_c"], frame["scale_c"], frame["radius_c"], frame["covered"] = test_pred, scale, radii, covered
    frame.to_csv(output / "test_intervals.csv", index=False)
    report = dict(
        schema="national-residual-uncertainty-v2", state="complete",
        target="90%simultaneous coverage of all evaluated single-source station transfers per night",
        factor=factor, scale_floor_c=.1, crossfit_rows=int(eligible.sum()),
        training_pairs_not_crossfit=int(((data["split"] == "train") & ~eligible).sum()),
        error_scale_training_scheme="Unseen-target out-of-fold residuals only; region rounds are separate diagnostics",
        folds=fold_report, crossfit_diagnostics=diagnostics,
        calibration_nights=len(set(data["night"][cal_ids])),
        test=interval_diagnostics(data, test_ids, test_pred, radii),
        uncertainty_map_validated=False,
        limitations=[
            "Test already inspected in temperature experiment; uncertainty evaluation is exploratory.",
            "Only two heldout geographic test blocks; spatial/temporal shifts violate exchangeability.",
            "Calibrated for evaluated station pairs per night,not all4096unobserved cells or national maps.",
            "No simultaneous-grid coverage guarantee; synthetic slider and domino are unvalidated.",
            "Fixed selected one-source predictor; fusion residuals require separate training/calibration.",
            "Target cross-fit excludes held stations as both fitting targets and observed sources; nearby nonheld sources remain.",
            "Region diagnostics exclude held-block targets/sources and10km buffers from fitting; source outside buffer needed for evaluation.",
            "Training winters2020-22get descriptive OOF breakdowns,not independent temporal coverage certification.",
            "Interval calibration/final testing still use only two regions each; dual cross-fitting does not create fresh test regions.",
            "OOF residual model learns absolute error scale,not physical uncertainty or learned covariance.",
        ],
        checksums={})
    for name in ("run_config.json", "error_scale.joblib", "error_scale.json",
                 "target_crossfit_residuals.npz", "region_crossfit_residuals.npz", "test_intervals.csv"):
        report["checksums"][name] = sha256(output / name)
    write_json(output / "report.json", report)
    write_json(output / "status.json", dict(state="complete", uncertainty_map_validated=False))
    print(json.dumps(report["test"], indent=2), flush=True)
    return report


def run(args):
    import os
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
                return run_locked(args)
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, default=Path("national_uncertainty_results"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--hours", type=float, default=40.)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--tree-rows", type=int, default=300000)
    args = p.parse_args(argv)
    if (not np.isfinite(args.hours) or args.hours <= 0 or
            min(args.folds, args.batch_size, args.jobs, args.tree_rows) < 1 or args.folds < 2):
        p.error("Use positive finite settings and at least two folds")
    return args


if __name__ == "__main__":
    run(parse_args())
