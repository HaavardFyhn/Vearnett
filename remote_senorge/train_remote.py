"""Portable CPU/GPU comparison using prepared seNorge-only pairs; no GIS dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    for attempt in range(8):
        try:
            pending.replace(path)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 7:
                raise
            delay = min(.1*2**attempt, 2.)
            logging.getLogger(__name__).warning(
                "Checkpoint replacement temporarily blocked for %s; retrying in %.1fs (%d/7): %s",
                path, delay, attempt+1, error)
            time.sleep(delay)


def load_data(folder: Path, target: str | None = None) -> tuple[dict, dict]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    path = folder / "training.npz"
    if sha256(path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset SHA256 mismatch; repeat the transfer")
    targets = manifest.get("targets")
    if targets is not None:
        if target not in targets:
            raise ValueError(f"Select --target from: {', '.join(targets)}")
        manifest.update(target_metric=target, target=targets[target])
    elif target is not None:
        raise ValueError("--target is only supported for multi-target packages")
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files if not key.startswith("y_")}
        if targets is not None:
            data["y"] = saved[f"y_{target}"]
    x, y = data["x"], data["y"]
    if (y.ndim != 1 or x.shape != (len(y), len(manifest["features"]))
            or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0)):
        raise ValueError("Invalid training matrix")
    if not 0 < manifest["base_feature_count"] < x.shape[1]:
        raise ValueError("Invalid base feature count")
    for key in ("grid_id_1", "grid_id_2", "block_1", "block_2"):
        if data[key].shape != y.shape or not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f"Invalid endpoint array: {key}")
    train, test = data["train"], data["test"]
    if train.dtype != bool or test.dtype != bool or train.shape != y.shape or test.shape != y.shape:
        raise ValueError("Invalid split masks")
    if np.any(train & test) or min(train.sum(), test.sum()) < 2:
        raise ValueError("Overlapping or empty spatial holdout")
    if np.intersect1d(data["block_1"][train], np.r_[data["block_1"][test], data["block_2"][test]]).size:
        raise ValueError("Spatial leakage at first endpoints")
    if np.intersect1d(data["block_2"][train], np.r_[data["block_1"][test], data["block_2"][test]]).size:
        raise ValueError("Spatial leakage at second endpoints")
    return data, manifest


def make_model(backend: str, args: argparse.Namespace):
    if backend == "extratrees":
        return ExtraTreesRegressor(n_estimators=args.trees, min_samples_leaf=8,
                                   n_jobs=args.jobs, random_state=42, max_features=1.0)
    import xgboost
    return xgboost.XGBRegressor(
        n_estimators=args.boost_rounds, max_depth=8, learning_rate=0.05,
        subsample=0.9, colsample_bytree=1.0, objective="reg:squarederror",
        tree_method="hist", device=args.device, n_jobs=args.jobs, random_state=42,
    )


def verify_device(model, backend: str, requested: str) -> str:
    if backend == "extratrees":
        return "cpu"
    config = json.loads(model.get_booster().save_config())
    actual = config["learner"]["generic_param"]["device"]
    if requested.startswith("cuda") and not actual.startswith("cuda"):
        raise RuntimeError(f"GPU requested but XGBoost used {actual}; refusing CPU fallback")
    return actual


def run(args: argparse.Namespace) -> dict:
    target = getattr(args, "target", None)
    data, manifest = load_data(args.data, target) if target is not None else load_data(args.data)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Completed results exist; choose a different --output")
    xgb_provenance = {}
    if "xgboost" in args.backends:
        import xgboost
        xgb_provenance = dict(version=xgboost.__version__, build=xgboost.build_info())
        if args.device.startswith("cuda") and not xgboost.build_info().get("USE_CUDA", False):
            raise RuntimeError("Installed XGBoost has no CUDA support")
    config = dict(dataset_sha256=manifest["dataset_sha256"], trees=args.trees,
                  boost_rounds=args.boost_rounds, device=args.device, backends=args.backends,
                  jobs=args.jobs, numpy=np.__version__, sklearn=sklearn.__version__,
                  python=platform.python_version(), xgboost=xgb_provenance,
                  features=manifest["features"], base_feature_count=manifest["base_feature_count"])
    if target is not None:
        config["target_metric"] = target
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run settings changed; choose another --output")
    write_json(config_path, config)
    start = time.monotonic()
    x, y, train, test = (data[key] for key in ("x", "y", "train", "test"))
    if args.device.startswith("cuda") and "xgboost" in args.backends:
        probe = make_model("xgboost", args)
        probe.set_params(n_estimators=2)
        probe.fit(x[:256], y[:256])
        verify_device(probe, "xgboost", args.device)
        del probe
    experiments = {}
    for backend in args.backends:
        for use_basins in (False, True):
            name = backend + ("_with_basins" if use_basins else "_without_basins")
            columns = np.arange(x.shape[1] if use_basins else manifest["base_feature_count"])
            receipt = output / f"{name}.json"
            model_path = output / f"{name}.joblib"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if saved["model_sha256"] != sha256(model_path):
                    raise ValueError(f"Model checkpoint changed: {name}")
                for filename, expected in saved["file_sha256"].items():
                    if sha256(output / filename) != expected:
                        raise ValueError(f"Experiment checkpoint changed: {filename}")
                experiments[name] = saved
                print(f"CACHE {name}", flush=True)
                continue
            write_json(output / "status.json", dict(state="running", experiment=name, stage="spatial_validation"))
            print(f"VALIDATE {name}: {train.sum():,} train / {test.sum():,} test pairs", flush=True)
            model = make_model(backend, args)
            model.fit(x[train][:, columns], y[train])
            device = verify_device(model, backend, args.device)
            prediction = model.predict(x[test][:, columns])
            scores = dict(mae=float(mean_absolute_error(y[test], prediction)),
                          rmse=float(np.sqrt(mean_squared_error(y[test], prediction))),
                          r2=float(r2_score(y[test], prediction)))
            if "distance_m" in manifest["features"]:
                distance = x[test, manifest["features"].index("distance_m")]
                scores["by_distance_km"] = []
                for lower, upper in ((1, 5), (5, 10), (10, 20), (20, 30), (30, 50),
                                     (50, 75), (75, 90), (90, 100)):
                    mask = (distance >= lower*1000) & (
                        distance <= upper*1000 if upper == 100 else distance < upper*1000)
                    scores["by_distance_km"].append(dict(
                        lower=lower, upper=upper, pairs=int(mask.sum()),
                        rmse=float(np.sqrt(mean_squared_error(y[test][mask], prediction[mask]))) if mask.any() else None,
                        mae=float(mean_absolute_error(y[test][mask], prediction[mask])) if mask.any() else None))
            np.savez_compressed(output / f"{name}_holdout.npz", actual=y[test], predicted=prediction,
                                grid_id_1=data["grid_id_1"][test], grid_id_2=data["grid_id_2"][test])
            del model
            write_json(output / "status.json", dict(state="running", experiment=name, stage="final_refit"))
            print(f"REFIT {name}: {len(y):,} pairs, validation={scores}", flush=True)
            model = make_model(backend, args)
            model.fit(x[:, columns], y)
            verify_device(model, backend, args.device)
            pending = output / f"{name}.pending.joblib"
            joblib.dump(model, pending)
            pending.replace(model_path)
            if backend == "xgboost":
                model.save_model(output / f"{name}.ubj")
            result = dict(validation=scores, actual_device=device,
                          features=[manifest["features"][i] for i in columns],
                          model_sha256=sha256(model_path),
                          feature_importance=dict(zip(
                              [manifest["features"][i] for i in columns],
                              model.feature_importances_.astype(float).tolist())))
            files = [output / f"{name}_holdout.npz"]
            if backend == "xgboost":
                files.append(output / f"{name}.ubj")
            result["file_sha256"] = {path.name: sha256(path) for path in files}
            write_json(receipt, result)
            experiments[name] = result
            del model
    summary = dict(status="complete", dataset_sha256=manifest["dataset_sha256"],
                   pairs=len(y), training_source=manifest["training_source"],
                   spatial_train=int(train.sum()), spatial_test=int(test.sum()),
                   experiments=experiments, elapsed_minutes=(time.monotonic()-start)/60,
                   validation="Same fixed spatial holdout for every experiment. No test-set tuning/early stopping.",
                   limitations=manifest["limitations"],
                   target_metric=manifest.get("target_metric", "rmse"))
    summary["sampling"] = manifest.get("sampling")
    summary["target"] = manifest.get("target")
    write_json(output / "summary.json", summary)
    write_json(output / "status.json", dict(state="complete"))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--target", help="Target to train from a multi-target package")
    parser.add_argument("--backends", nargs="+", choices=["extratrees", "xgboost"],
                        default=["extratrees", "xgboost"])
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "cuda:1", "cpu"])
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--boost-rounds", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.trees, args.boost_rounds, args.jobs) < 1:
        parser.error("Trees, rounds and jobs must be positive")
    args.backends = list(dict.fromkeys(args.backends))
    args.output.mkdir(parents=True, exist_ok=True)
    # Linux advisory lock releases automatically if SSH/job/process dies.
    import fcntl
    with (args.output / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this output directory") from error
        try:
            run(args)
        except Exception as error:
            if not (args.output / "summary.json").exists():
                write_json(args.output / "status.json", dict(state="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
