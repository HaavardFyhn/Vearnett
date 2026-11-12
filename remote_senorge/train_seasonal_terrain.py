"""Compare seasonal XGBoost and symmetric terrain CNNs on fixed buffered spatial splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
if __package__:
    from .train_remote import sha256, write_json
else:
    from train_remote import sha256, write_json


def load_dataset(folder):
    manifest = json.loads((folder / "seasonal_manifest.json").read_text())
    for name, expected in manifest["checksums"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    with np.load(folder / "seasonal_training.npz", allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    x, y, pairs, split = (data[key] for key in ("x", "y", "pairs", "split"))
    if x.shape != (len(y), 25) or y.shape != (len(x), 4) or pairs.shape != (len(x), 2):
        raise ValueError("Invalid seasonal matrix shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid seasonal training values")
    if split.shape != (len(y),) or not np.isin(split, [-1, 0, 1, 2]).all():
        raise ValueError("Invalid spatial split")
    if any((split == group).sum() < 2 for group in range(3)):
        raise ValueError("Need nonempty training, validation and test partitions")
    terrain = np.load(folder / "terrain.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(folder / "terrain_valid.npy", mmap_mode="r", allow_pickle=False)
    if terrain.shape != valid.shape or terrain.shape[1:] != (3, 64, 64):
        raise ValueError("Invalid terrain patch shapes")
    if pairs.min() < 0 or pairs.max() >= len(terrain):
        raise ValueError("Pair references outside terrain table")
    xy = data["xy"]
    if xy.shape != (len(terrain), 2) or not np.isfinite(xy).all():
        raise ValueError("Invalid endpoint coordinates")
    blocks = np.floor(xy/100000).astype(np.int64)
    radius = manifest["patch_radius_m"]
    if radius < 16500:
        raise ValueError("Insufficient terrain support buffer")
    membership = []
    for group in range(3):
        ids = np.unique(pairs[split == group])
        within = xy[ids] % 100000
        if ((within < radius) | (within > 100000-radius)).any():
            raise ValueError("Terrain support crosses a spatial block boundary")
        membership.append(set(map(tuple, blocks[ids])))
    if any(membership[a] & membership[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Spatial leakage between partitions")
    return data, terrain, valid, manifest


def create_model(mode, torch):
    from torch import nn

    class TerrainPairNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = nn.ModuleList([
                nn.Sequential(nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.SiLU(),
                              nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                              nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten())
                for _ in range(3)])
            self.head = nn.Sequential(nn.Linear(192 + (25 if mode == "combined" else 3), 128),
                                      nn.SiLU(), nn.Linear(128, 64), nn.SiLU(),
                                      nn.Linear(64, 4), nn.Softplus())

        def encode(self, patches):
            return torch.cat([encoder(patches[:, i]) for i, encoder in enumerate(self.encoders)], dim=1)

        def forward(self, a, b, x):
            ea, eb = self.encode(a), self.encode(b)
            # Commutative features make endpoint exchange invariant by construction.
            features = x if mode == "combined" else x[:, [0, 5, 6]]
            return self.head(torch.cat([ea+eb, torch.abs(ea-eb), features], dim=1))

    return TerrainPairNet()


def score(actual, predicted, names, distance):
    result = {}
    for i, name in enumerate(names):
        error = predicted[:, i].astype(float)-actual[:, i]
        variance = np.sum((actual[:, i]-actual[:, i].mean())**2)
        bins = []
        for lower, upper in ((1, 5), (5, 10), (10, 20), (20, 50), (50, 75), (75, 90), (90, 100)):
            keep = (distance >= lower*1000) & (
                distance <= upper*1000 if upper == 100 else distance < upper*1000)
            bins.append(dict(lower_km=lower, upper_km=upper, pairs=int(keep.sum()),
                             rmse=float(np.sqrt(np.mean(error[keep]**2))) if keep.any() else None))
        result[name] = dict(rmse=float(np.sqrt(np.mean(error**2))),
                            mae=float(np.mean(abs(error))),
                            r2=float(1-np.sum(error**2)/variance) if variance > 0 else None,
                            bias=float(error.mean()), by_distance=bins)
    return result


def train(args):
    import torch
    if (args.output / "summary.json").exists():
        raise FileExistsError("Completed seasonal results already exist; choose another output")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in PyTorch")
    data, terrain, valid, manifest = load_dataset(args.data)
    try:
        return fit(args, data, terrain, valid, manifest)
    finally:
        terrain._mmap.close()
        valid._mmap.close()


def fit(args, data, terrain, valid, manifest):
    import torch
    import xgboost
    import sklearn
    from torch.utils.data import DataLoader, Dataset
    from xgboost import XGBRegressor
    if __package__:
        from .train_remote import verify_device
    else:
        from train_remote import verify_device

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError("Completed seasonal results already exist; choose another output")
    config = dict(seed=42, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                  device=args.device, jobs=args.jobs, learning_rate=args.learning_rate,
                  boost_rounds=args.boost_rounds, torch=torch.__version__,
                  xgboost=xgboost.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
                  architecture="symmetric-multiscale-cnn-v1",
                  prediction_policy="nonnegative RMSE; XGBoost clipped at zero, CNN softplus",
                  manifest_sha256=sha256(args.data / "seasonal_manifest.json"),
                  modes=args.modes)
    if (out / "run_config.json").exists() and json.loads((out / "run_config.json").read_text()) != config:
        raise ValueError("Run settings changed; choose another output")
    write_json(out / "run_config.json", config)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(args.jobs)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(42)
    x, y, pairs, split = (data[key] for key in ("x", "y", "pairs", "split"))
    train_ids, validation_ids, test_ids = [np.flatnonzero(split == group) for group in range(3)]
    mean = x[train_ids].mean(axis=0, dtype=np.float64).astype("float32")
    std = x[train_ids].std(axis=0, dtype=np.float64).astype("float32")
    std = np.maximum(std, 1e-6)
    normalized = (x-mean)/std

    class Pairs(Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            i = self.indices[index]
            patches = []
            for endpoint in pairs[i]:
                height = np.asarray(terrain[endpoint], dtype=np.float32)
                mask = np.asarray(valid[endpoint], dtype=np.float32)
                if not np.isfinite(height).all() or not np.isin(mask, [0, 1]).all():
                    raise ValueError("Nonfinite terrain or invalid validity mask")
                patches.append(np.stack([height, mask], axis=1))
            return patches[0], patches[1], normalized[i], y[i]

    loaders = {}
    generator = torch.Generator().manual_seed(42)
    for key, ids in (("train", train_ids), ("validation", validation_ids), ("test", test_ids)):
        loaders[key] = DataLoader(Pairs(ids), batch_size=args.batch_size, shuffle=key == "train",
                                  num_workers=0, generator=generator, pin_memory=args.device.startswith("cuda"))

    def evaluate(model, loader):
        model.eval()
        predictions, truths = [], []
        with torch.no_grad():
            for a, b, features, target in loader:
                pred = model(a.to(args.device), b.to(args.device), features.to(args.device))
                predictions.append(pred.cpu().numpy())
                truths.append(target.numpy())
        predicted, actual = np.concatenate(predictions), np.concatenate(truths)
        if not np.isfinite(predicted).all():
            raise RuntimeError("Nonfinite network predictions")
        return predicted, float(np.mean((predicted-actual)**2))

    results = {}
    for mode in args.modes:
        receipt = out / f"{mode}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            for name, digest in saved["checksums"].items():
                if sha256(out / name) != digest:
                    raise ValueError(f"Completed checkpoint mismatch: {name}")
            results[mode] = saved
            print(f"CACHE {mode}", flush=True)
            continue
        write_json(out / "status.json", dict(state="running", mode=mode))
        start = time.monotonic()
        files = []
        if mode == "xgboost":
            predicted = np.empty((len(test_ids), 4), dtype=np.float32)
            for i, name in enumerate(manifest["targets"]):
                model = XGBRegressor(n_estimators=args.boost_rounds, max_depth=8,
                                     learning_rate=.05, subsample=.9, tree_method="hist",
                                     device=args.device, n_jobs=args.jobs, random_state=42,
                                     early_stopping_rounds=args.patience)
                model.fit(x[train_ids], y[train_ids, i],
                          eval_set=[(x[validation_ids], y[validation_ids, i])], verbose=False)
                verify_device(model, "xgboost", args.device)
                predicted[:, i] = np.maximum(model.predict(x[test_ids]), 0)
                path = out / f"xgboost_{name}.ubj"
                model.save_model(path)
                files.append(path)
        else:
            torch.manual_seed(42)
            generator.manual_seed(42)
            model = create_model(mode, torch).to(args.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            best, stale, history = float("inf"), 0, []
            weights = out / f"{mode}.pt"
            for epoch in range(args.epochs):
                model.train()
                total, count = 0., 0
                for a, b, features, target in loaders["train"]:
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(a.to(args.device), b.to(args.device), features.to(args.device))
                    loss = torch.nn.functional.mse_loss(pred, target.to(args.device))
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    total += loss.item()*len(target)
                    count += len(target)
                _, validation_loss = evaluate(model, loaders["validation"])
                history.append(dict(epoch=epoch+1, train_mse=total/count, validation_mse=validation_loss))
                print(f"{mode} epoch {epoch+1}: train={total/count:.6f} validation={validation_loss:.6f}", flush=True)
                if validation_loss < best:
                    best, stale = validation_loss, 0
                    pending = weights.with_suffix(".pending.pt")
                    torch.save(model.state_dict(), pending)
                    pending.replace(weights)
                else:
                    stale += 1
                write_json(out / f"{mode}_history.json", dict(epochs=history))
                if stale >= args.patience:
                    break
            model.load_state_dict(torch.load(weights, map_location=args.device, weights_only=True))
            predicted, _ = evaluate(model, loaders["test"])
            np.savez(out / f"{mode}_normalization.npz", mean=mean, std=std)
            files.extend([weights, out / f"{mode}_normalization.npz", out / f"{mode}_history.json"])
        if not np.isfinite(predicted).all():
            raise RuntimeError(f"Nonfinite predictions from {mode}")
        heldout = out / f"{mode}_holdout.npz"
        np.savez_compressed(heldout, actual=y[test_ids], predicted=predicted,
                            pair_indices=test_ids, pairs=pairs[test_ids], distance_m=x[test_ids, 0])
        files.append(heldout)
        result = dict(test=score(y[test_ids], predicted, manifest["targets"], x[test_ids, 0]),
                      minutes=(time.monotonic()-start)/60,
                      checksums={p.name: sha256(p) for p in files})
        write_json(receipt, result)
        results[mode] = result
    summary = dict(status="complete", experiments=results, train_pairs=len(train_ids),
                   validation_pairs=len(validation_ids), test_pairs=len(test_ids),
                   targets=manifest["targets"],
                   limitations=manifest["limitations"],
                   evaluation="Buffered spatial test; validation only for early stopping. No final all-pair refit.",
                   cuda_device=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else "cpu")
    write_json(out / "summary.json", summary)
    write_json(out / "status.json", dict(state="complete"))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("seasonal_results"))
    parser.add_argument("--device", choices=["cpu", "cuda:0", "cuda:1"], default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--boost-rounds", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--modes", nargs="+", choices=["xgboost", "terrain", "combined"],
                        default=["xgboost", "terrain", "combined"])
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.jobs, args.boost_rounds, args.learning_rate) <= 0:
        parser.error("Training parameters must be positive")
    args.modes = list(dict.fromkeys(args.modes))
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / "training.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            train(args)
        except Exception as error:
            if not (args.output / "summary.json").exists():
                write_json(args.output / "status.json", dict(state="failed", error=str(error)))
            raise


if __name__ == "__main__":
    main()
