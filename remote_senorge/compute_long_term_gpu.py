"""Portable exact all-date seNorge inference; no fitting, GIS, network or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False))
    pending.replace(path)


class GPUInference:
    def __init__(self, folder, device):
        import torch
        from train_senorge_transfer_gpu import build_model
        self.torch, self.device = torch, device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; refusing a silent CPU fallback")
        torch.set_num_threads(4)
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        folder = Path(folder)
        self.spec = json.loads((folder / "spec.json").read_text())
        self.selected = self.spec["selected"]
        self.networks = {}
        portable = json.loads((folder / "portable_ridge.json").read_text())
        for name in (self.selected["mean_checkpoint"], self.selected["error_checkpoint"]):
            if name is None:
                continue
            info = portable["networks"][name]
            config = info["config"]
            n = len(self.spec["features"])+int(config["error_stage"])
            model = build_model(torch, n, config["width"], config["error_stage"])
            model.load_state_dict(torch.load(folder / info["weights"], map_location="cpu", weights_only=True))
            with np.load(folder / info["normalization"], allow_pickle=False) as norm:
                self.networks[name] = (model.to(device).eval(),
                    torch.as_tensor(norm["mean"], device=device, dtype=torch.float64),
                    torch.as_tensor(norm["std"], device=device, dtype=torch.float64))
        with np.load(folder / "portable_ridge.npz", allow_pickle=False) as saved:
            self.ridge = {key: torch.as_tensor(saved[key], device=device, dtype=torch.float64)
                          for key in saved.files}

    def network(self, name, x):
        model, mean, std = self.networks[name]
        return model(((x.double()-mean)/std).float()).squeeze(-1).double()

    def mse(self, x, source_c):
        torch, r = self.torch, self.ridge
        with torch.inference_mode():
            if self.selected["mean_anchor"] == "ridge":
                prediction = source_c.double()+((x.double()-r["mean"])/r["std"])@r["coef"]+r["intercept"]
            elif self.selected["mean_anchor"] == "offset":
                prediction = source_c.double()+r["offset"]
            else:
                raise ValueError("Unknown frozen mean anchor")
            if self.selected["mean_checkpoint"]:
                prediction += self.network(self.selected["mean_checkpoint"], x)
            if self.selected["error_checkpoint"]:
                mse = self.network(self.selected["error_checkpoint"],
                                   torch.cat([x, prediction.float()[:, None]], dim=1))
            else:
                mse = torch.full((len(x),), self.selected["pooled_mse"], device=self.device, dtype=torch.float64)
            if not torch.isfinite(mse).all() or (mse < 0).any():
                raise ValueError("Invalid predicted squared corrected error")
            return mse

    def footprint(self, table, cell, targets, temperatures, days, variable, batch_size):
        from senorge_corrected_transfer import features
        import pandas as pd
        torch, device = self.torch, self.device
        count = len(days)
        if not np.isfinite(temperatures).all():
            raise ValueError("Incomplete source temperatures; cannot pool a subset of dates")
        static = features(table, np.full(len(targets), cell), targets, np.zeros(len(targets)),
                          np.repeat(days[0], len(targets)), np.full(len(targets), variable))
        static = torch.as_tensor(static, device=device)
        doy = pd.to_datetime(days).dayofyear.to_numpy()
        season = torch.as_tensor(np.column_stack([np.sin(2*np.pi*doy/365.25),
            np.cos(2*np.pi*doy/365.25)]).astype(np.float32), device=device)
        temperature = torch.as_tensor(temperatures, device=device, dtype=torch.float32)
        sums, counts = np.zeros(len(targets), np.float64), np.zeros(len(targets), np.int32)
        total = len(targets)*count
        with torch.inference_mode():
            for start in range(0, total, batch_size):
                flat = torch.arange(start, min(start+batch_size, total), device=device)
                pair, date = flat//count, flat % count
                x = static[pair].clone()
                x[:, 0], x[:, 1:3] = temperature[date], season[date]
                mse = self.mse(x, temperature[date]).cpu().numpy()
                ids = np.arange(start, min(start+batch_size, total))//count
                sums += np.bincount(ids, weights=mse, minlength=len(targets))
                counts += np.bincount(ids, minlength=len(targets)).astype(np.int32)
        if not np.all(counts == count):
            raise ValueError("Incomplete all-date footprint")
        scores = np.sqrt(sums/count).astype(np.float32)
        scores[targets == cell] = 0
        return scores, counts


def run(args):
    folder, output = args.package.resolve(), args.output.resolve()
    checks = json.loads((folder / "package_checksums.json").read_text())
    for name, digest in checks.items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError(f"Package checksum mismatch: {name}")
    package_hash = sha256(folder / "package_checksums.json")
    output.mkdir(parents=True, exist_ok=True)
    config = dict(package_sha256=package_hash, variables=args.variables)
    receipt = output / "run_config.json"
    if receipt.exists() and json.loads(receipt.read_text()) != config:
        raise ValueError("Output belongs to a different package/variable selection")
    write_json(receipt, config)
    infer = GPUInference(folder, args.device)
    spec = infer.spec
    with np.load(folder / "inputs.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    started = time.monotonic()
    outputs = {}
    sources = len(data["cells"])
    for variable in args.variables:
        for i, cell in enumerate(data["cells"]):
            if args.benchmark and i:
                break
            if not data["valid"][variable, i]:
                continue
            key = hashlib.sha256(f"{spec['identity']}:{cell}:{variable}".encode()).hexdigest()
            name = f"source_{key}.npz"
            path, meta_path = output / name, output / f"source_{key}.json"
            targets = data["targets"][data["offsets"][i]:data["offsets"][i+1]]
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if sha256(path) != meta["sha256"] or meta["package_sha256"] != package_hash:
                    raise ValueError("Changed completed GPU source output")
            else:
                local_name = f"seed_{key}.npz"
                if not args.benchmark and (folder / local_name).exists():
                    content = (folder / local_name).read_bytes()
                    pending = path.with_suffix(".pending.npz")
                    pending.write_bytes(content)
                    pending.replace(path)
                else:
                    scores, counts = infer.footprint(data["table"], int(cell), targets,
                        data["temperatures"][variable, i], data["days"], variable, args.batch_size)
                    pending = path.with_suffix(".pending.npz")
                    np.savez_compressed(pending, targets=targets, rmse=scores, date_count=counts)
                    pending.replace(path)
                write_json(meta_path, dict(sha256=sha256(path), dates=len(data["days"]),
                    source_cell=int(cell), variable=variable, identity=spec["identity"],
                    package_sha256=package_hash))
            outputs[name] = sha256(path)
            outputs[meta_path.name] = sha256(meta_path)
            elapsed = time.monotonic()-started
            write_json(output / "status.json", dict(state="benchmark" if args.benchmark else "running",
                variable=variable, completed=i+1, total=sources, elapsed_s=elapsed))
            print(f"variable={variable} source={i+1}/{sources} elapsed_s={elapsed:.1f}", flush=True)
    write_json(output / "report.json", dict(state="benchmark" if args.benchmark else "complete",
        package_sha256=package_hash, variables=args.variables, unique_sources=sources,
        dates=len(data["days"]), checksums=outputs, elapsed_s=time.monotonic()-started,
        device=args.device))
    write_json(output / "status.json", dict(state="benchmark" if args.benchmark else "complete"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("long_term_results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--variables", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Use a positive batch size")
    run(args)
