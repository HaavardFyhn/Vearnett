"""GPU weekend experiments for reference-conditioned six-target footprint maps."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from .footprint_model import ARCHITECTURE, TARGETS, create_model, masked_mse, metrics
    from .train_remote import sha256, write_json, verify_device
else:
    from footprint_model import ARCHITECTURE, TARGETS, create_model, masked_mse, metrics
    from train_remote import sha256, write_json, verify_device


@contextmanager
def output_lock(folder):
    import os
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "training.lock").open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_data(folder):
    manifest = json.loads((folder / "footprint_manifest.json").read_text())
    if manifest["targets"] != TARGETS or manifest["pixels"] % 8:
        raise ValueError("Unexpected footprint schema/targets")
    if manifest.get("schema") == 2:
        if __package__:
            from .footprint_atlas import AtlasDataset
        else:
            from footprint_atlas import AtlasDataset
        dataset = AtlasDataset(folder)
        arrays = dataset.arrays()
        arrays["_atlas"] = dataset
        return arrays, dataset.refs, manifest
    for name, digest in manifest["checksums"].items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError(f"Dataset checksum/allowlist mismatch: {name}")
    with np.load(folder / "references.npz", allow_pickle=False) as stored:
        refs = {name: stored[name] for name in stored.files}
    arrays = {}
    try:
        for name in ("scenes", "local10", "features", "labels", "label_mask"):
            arrays[name] = np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        n, pixels = len(refs["refs"]), manifest["pixels"]
        expected = dict(scenes=(n, 2, (pixels+16)*10, (pixels+16)*10),
                        local10=(n, 2, 128, 128), features=(n, 25, pixels, pixels),
                        labels=(n, 6, pixels, pixels), label_mask=(n, 6, pixels, pixels))
        if any(arrays[name].shape != shape for name, shape in expected.items()):
            raise ValueError("Invalid footprint array shapes")
        if refs["split"].shape != (n,) or not np.isin(refs["split"], [0, 1, 2]).all():
            raise ValueError("Invalid reference split")
        if arrays["label_mask"].dtype != np.bool_:
            raise ValueError("Label mask must be boolean")
        xy = refs["xy"]
        if xy.shape != (n, 2) or not np.isfinite(xy).all():
            raise ValueError("Invalid reference coordinates")
        position = xy % 100000
        radius = (pixels/2+8+.5)*1000
        if ((position < radius) | (position > 100000-radius)).any():
            raise ValueError("Input scene crosses its spatial partition block")
        memberships = []
        for group in range(3):
            ids = np.flatnonzero(refs["split"] == group)
            if len(ids) < 2:
                raise ValueError("Each split needs at least two references")
            memberships.append(set(map(tuple, np.floor(xy[ids]/100000).astype(int))))
            count = arrays["label_mask"][ids].sum(axis=(0, 2, 3))
            if (count == 0).any():
                raise ValueError("Every split needs valid labels for all six targets")
        if any(memberships[a] & memberships[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Geographic leakage between map splits")
        return arrays, refs, manifest
    except Exception:
        for data in arrays.values():
            data._mmap.close()
        raise


def normalization(arrays, ids):
    count = 0
    channels = arrays["_atlas"].feature_count if "_atlas" in arrays else 25
    total, square = np.zeros(channels), np.zeros(channels)
    for i in ids:
        keep = arrays["label_mask"][i].any(axis=0)
        values = arrays["features"][i][:, keep].astype(float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training features")
        count += values.shape[1]
        total += values.sum(axis=1)
        square += (values**2).sum(axis=1)
    if count == 0:
        raise ValueError("No training feature pixels")
    mean = total/count
    std = np.sqrt(np.maximum(square/count-mean**2, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def atomic_save(torch, path, value):
    pending = path.with_suffix(".pending.pt")
    torch.save(value, pending)
    pending.replace(path)


def run(args):
    import torch
    from torch.utils.data import DataLoader, Dataset
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        device_name = torch.cuda.get_device_name(torch.device(args.device))
        torch.ones(8, device=args.device).sum().item()
    else:
        device_name = "cpu"
    if args.hours <= 0 or args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Positive budget, epochs, batch size and patience required")
    if args.output.joinpath("summary.json").exists():
        raise FileExistsError("Completed footprint results exist; choose another output")
    data, refs, manifest = load_data(args.data)
    try:
        return fit(args, torch, DataLoader, Dataset, data, refs, manifest, device_name)
    finally:
        for value in data.values():
            if hasattr(value, "_mmap"):
                value._mmap.close()
        if "_atlas" in data:
            data["_atlas"].close()


def fit(args, torch, DataLoader, Dataset, arrays, refs, manifest, device_name):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.jobs)
    architecture = manifest.get("architecture", ARCHITECTURE)
    spatial_channels = len(manifest["spatial_fields"]) if manifest.get("schema") == 2 else 0
    config = dict(architecture=architecture, manifest_sha256=sha256(args.data / "footprint_manifest.json"),
                  torch=torch.__version__, numpy=np.__version__, device=args.device,
                  epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                  seeds=args.seeds, widths=args.widths, learning_rate=args.learning_rate,
                  hours=args.hours, jobs=args.jobs, baseline=not args.skip_baseline)
    config_path = out / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run settings changed; choose another output")
    write_json(config_path, config)
    started = time.monotonic()
    budget_path = out / "budget.json"
    consumed = json.loads(budget_path.read_text())["seconds"] if budget_path.exists() else 0
    deadline = started+max(0, args.hours*3600-consumed)
    train_ids, val_ids, test_ids = [np.flatnonzero(refs["split"] == group) for group in range(3)]
    mean, std = normalization(arrays, train_ids)
    pair_channels = len(mean)
    np.savez(out / "normalization.npz", mean=mean, std=std)

    class Maps(Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            i = self.indices[index]
            scene = arrays["scenes"][i].astype(np.float32)
            local = arrays["local10"][i].astype(np.float32)
            features = (arrays["features"][i]-mean[:, None, None])/std[:, None, None]
            labels = arrays["labels"][i].copy()
            mask = arrays["label_mask"][i].copy()
            # Identity is a policy, not evidence of learned accuracy; exclude it from losses/scores.
            centre = manifest["pixels"]//2
            mask[:, centre, centre] = False
            if (not all(np.isfinite(v).all() for v in (scene, local, features, labels)) or
                    (labels[mask] < 0).any() or
                    not np.isin(scene[1], [0, 1]).all() or
                    (not spatial_channels and not np.isin(local[1], [0, 1]).all()) or
                    (spatial_channels and ((local[0] < 0) | (local[0] > 1)).any())):
                raise ValueError("Invalid map values, labels or terrain validity masks")
            return scene, local, features, labels, mask

    def loader(indices, shuffle=False, seed=42):
        return DataLoader(Maps(indices), batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
                          generator=torch.Generator().manual_seed(seed),
                          pin_memory=args.device.startswith("cuda"))

    def evaluate(model, ids):
        predicted, actual, masks = [], [], []
        model.eval()
        with torch.inference_mode():
            for scene, local, features, labels, mask in loader(ids):
                p = model(scene.to(args.device), local.to(args.device), features.to(args.device))
                p = p.cpu().numpy()
                if not np.isfinite(p).all():
                    raise ValueError("Nonfinite footprint prediction")
                # Identity rule mirrors existing app semantics, not an observed zero-error station claim.
                p[:, :, manifest["pixels"]//2, manifest["pixels"]//2] = 0
                predicted.append(p)
                actual.append(labels.numpy())
                masks.append(mask.numpy())
        a, p, m = np.concatenate(actual), np.concatenate(predicted), np.concatenate(masks)
        scores = metrics(a, p, m)
        return float(np.mean([s["rmse"]**2 for s in scores.values()])), scores, (a, p, m)

    trials = []
    budget_exhausted = False
    for width in args.widths:
        for seed in args.seeds:
            for fine in ((True,) if spatial_channels else (False, True)):
                kind = "spatial10" if spatial_channels else "fine10" if fine else "context"
                name = f"resunet_w{width}_{kind}_s{seed}"
                receipt = out / f"{name}.json"
                if receipt.exists():
                    saved = json.loads(receipt.read_text())
                    if sha256(out / f"{name}_best.pt") != saved["model_sha256"]:
                        raise ValueError(f"Trial checkpoint checksum mismatch: {name}")
                    trials.append(saved)
                    continue
                if time.monotonic() >= deadline:
                    budget_exhausted = True
                    break
                torch.manual_seed(seed)
                if args.device.startswith("cuda"):
                    torch.cuda.manual_seed_all(seed)
                model = create_model(torch, width, fine, manifest["pixels"], spatial_channels,
                                     pair_channels).to(args.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.001)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=.5, patience=max(2, args.patience//3), min_lr=1e-6)
                state_path = out / f"{name}_resume.pt"
                best_path = out / f"{name}_best.pt"
                first_epoch, best, stale, history = 0, float("inf"), 0, []
                if state_path.exists():
                    state = torch.load(state_path, map_location=args.device, weights_only=True)
                    model.load_state_dict(state["model"])
                    optimizer.load_state_dict(state["optimizer"])
                    scheduler.load_state_dict(state["scheduler"])
                    first_epoch, best, stale, history = state["epoch"], state["best"], state["stale"], state["history"]
                write_json(out / "status.json", dict(state="running", trial=name, epoch=first_epoch))
                for epoch in range(first_epoch, args.epochs):
                    if time.monotonic() >= deadline or stale >= args.patience:
                        break
                    model.train()
                    loss_sum, batches = 0., 0
                    for scene, local, features, actual, mask in loader(train_ids, True, seed+epoch):
                        optimizer.zero_grad(set_to_none=True)
                        pred = model(scene.to(args.device), local.to(args.device), features.to(args.device))
                        loss = masked_mse(torch, pred, actual.to(args.device), mask.to(args.device))
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite footprint training loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                        loss_sum += loss.item()
                        batches += 1
                    value, validation, _ = evaluate(model, val_ids)
                    scheduler.step(value)
                    history.append(dict(epoch=epoch+1, train_mse=loss_sum/batches,
                                        validation_mse=value, learning_rate=optimizer.param_groups[0]["lr"]))
                    print(f"{name} epoch {epoch+1}: train={loss_sum/batches:.6f} validation={value:.6f}", flush=True)
                    if value < best:
                        best, stale = value, 0
                        atomic_save(torch, best_path, model.state_dict())
                    else:
                        stale += 1
                    atomic_save(torch, state_path, dict(
                        model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), epoch=epoch+1, best=best, stale=stale, history=history))
                    write_json(out / f"{name}_history.json", dict(epochs=history))
                    write_json(budget_path, dict(seconds=consumed+time.monotonic()-started))
                if not best_path.exists():
                    raise RuntimeError("Budget expired before a complete epoch; increase budget in a new output")
                model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=True))
                value, validation, _ = evaluate(model, val_ids)
                record = dict(name=name, width=width, fine=fine, seed=seed, validation_mse=value,
                              validation=validation, epochs=len(history),
                              stopped_by_budget=time.monotonic() >= deadline,
                              model_sha256=sha256(best_path))
                write_json(receipt, record)
                trials.append(record)
                del optimizer, scheduler, model
            if budget_exhausted:
                break
        if budget_exhausted:
            break
    if not trials:
        raise RuntimeError("No completed trial to evaluate")
    # Test maps are not evaluated per trial or used for choosing architecture/seed.
    selected = min(trials, key=lambda trial: trial["validation_mse"])
    model = create_model(torch, selected["width"], selected["fine"], manifest["pixels"],
                         spatial_channels, pair_channels).to(args.device)
    model.load_state_dict(torch.load(out / f"{selected['name']}_best.pt", map_location=args.device, weights_only=True))
    _, test_scores, (actual, predicted, mask) = evaluate(model, test_ids)
    np.savez_compressed(out / "selected_test_maps.npz", actual=actual, predicted=predicted,
                        mask=mask, reference_indices=test_ids, xy=refs["xy"][test_ids])
    atomic_save(torch, out / "selected_model.pt", model.state_dict())
    summary = dict(status="complete", architecture=architecture, targets=TARGETS,
                   spatial_channels=spatial_channels,
                   pair_channels=pair_channels, features=manifest.get("features"),
                   selected=selected, trials=trials, test=test_scores, device=device_name,
                   seconds=consumed+time.monotonic()-started, limitations=manifest["limitations"],
                   selection="Validation only; test scored once for selected CNN. Not an untouched geography relative to prior experiments.")
    if spatial_channels:
        block_ids = np.floor(refs["xy"][test_ids]/manifest["block_size_m"]).astype(int)
        summary["regional_test"] = {
            f"{bx},{by}": metrics(actual[(block_ids == [bx, by]).all(axis=1)],
                                  predicted[(block_ids == [bx, by]).all(axis=1)],
                                  mask[(block_ids == [bx, by]).all(axis=1)])
            for bx, by in np.unique(block_ids, axis=0)}
        summary["geography"] = manifest["geography"]
    if not args.skip_baseline:
        summary["xgboost_test"] = baseline(args, arrays, refs, out)
    files = ["selected_model.pt", "normalization.npz", "selected_test_maps.npz", "run_config.json"]
    if not args.skip_baseline:
        files.extend(f"xgboost_{target}{suffix}" for target in TARGETS for suffix in (".ubj", "_holdout.npz"))
    summary["checksums"] = {name: sha256(out / name) for name in files}
    write_json(out / "summary.json", summary)
    write_json(out / "status.json", dict(state="complete", selected=selected["name"]))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def baseline(args, arrays, refs, out):
    from xgboost import XGBRegressor
    scores = {}
    expanded = "_atlas" in arrays
    for i, name in enumerate(TARGETS):
        groups = []
        available = []
        for part in range(3):
            xx, yy = [], []
            ids = np.flatnonzero(refs["split"] == part)
            counts = []
            for j in ids:
                keep = arrays["label_mask"][j, i]
                centre = keep.shape[0]//2
                counts.append(int(keep.sum())-int(keep[centre, centre]))
            total = sum(counts)
            available.append(total)
            cap = (1000000 if part == 0 else 100000) if expanded and part < 2 else total
            selected = np.sort(np.random.default_rng(42).choice(total, cap, replace=False)) if cap < total else None
            offset = 0
            for j, count in zip(ids, counts):
                keep = arrays["label_mask"][j, i].copy()
                centre = keep.shape[0]//2
                keep[centre, centre] = False
                if selected is not None:
                    local = selected[np.searchsorted(selected, offset):np.searchsorted(selected, offset+count)]-offset
                    pixels = np.flatnonzero(keep)
                    keep[:] = False
                    keep.ravel()[pixels[local]] = True
                offset += count
                if not keep.any():
                    continue
                xx.append(arrays["features"][j][:, keep].T)
                yy.append(arrays["labels"][j, i][keep])
            groups.append((np.concatenate(xx), np.concatenate(yy)))
        model = XGBRegressor(n_estimators=1500, max_depth=8, learning_rate=.05, tree_method="hist",
                             device=args.device, n_jobs=args.jobs, random_state=42, early_stopping_rounds=20)
        model.fit(*groups[0], eval_set=[groups[1]], verbose=False)
        verify_device(model, "xgboost", args.device)
        p = np.maximum(model.predict(groups[2][0]), 0)
        error = p.astype(float)-groups[2][1]
        model.save_model(out / f"xgboost_{name}.ubj")
        np.savez_compressed(out / f"xgboost_{name}_holdout.npz", actual=groups[2][1], predicted=p)
        scores[name] = dict(cells=len(p), rmse=float(np.sqrt(np.mean(error**2))),
                            mae=float(np.mean(abs(error))), bias=float(error.mean()))
        if expanded:
            scores[name]["sampling"] = dict(seed=42, available_by_split=available,
                                           used_by_split=[len(group[1]) for group in groups],
                                           policy="Uniform pixels without replacement; train<=1000000,validation<=100000,test=all")
        print(f"XGBoost {name}: RMSE {scores[name]['rmse']:.6f}", flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("footprint_results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hours", type=float, default=40)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--widths", nargs="+", type=int, default=[16, 24])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    with output_lock(args.output):
        if (args.output / "summary.json").exists():
            raise FileExistsError("Completed footprint results exist; choose another output")
        try:
            run(args)
        except Exception as error:
            write_json(args.output / "status.json", dict(state="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
