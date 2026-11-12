"""Verify remote long-term predictions before publishing original-app cache receipts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from senorge_long_term import LongTermNetwork
from senorge_corrected_transfer import features
from remote_senorge.train_valley_neural import sha256


def verify_archive(archive, package, backend, variables=(0, 1), checkpoint=False):
    variables = tuple(variables)
    if not variables or len(set(variables)) != len(variables) or any(v not in (0, 1) for v in variables):
        raise ValueError("Choose distinct daily mean/minimum variables")
    package = Path(package)
    spec = json.loads((package / "spec.json").read_text())
    checks = json.loads((package / "package_checksums.json").read_text())
    for name, digest in checks.items():
        if Path(name).name != name or sha256(package / name) != digest:
            raise ValueError(f"Local package changed: {name}")
    if spec["identity"] != backend.identity or spec["catalogue_sha"] != backend.catalogue_sha:
        raise ValueError("Current app/model/station identity differs from the GPU package")
    if sha256(package / "compute_long_term_gpu.py") != sha256(
            backend.root / "remote_senorge" / "compute_long_term_gpu.py"):
        raise ValueError("GPU inference code changed since packaging")
    with np.load(package / "inputs.npz", allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    members = {}
    for member in archive.getmembers():
        if member.name.startswith("long_term_results/") and member.isfile():
            name = member.name[len("long_term_results/"):]
            if "/" in name or "\\" in name or name in members:
                raise ValueError("Invalid or duplicate result filename")
            members[name] = member

    def read(name):
        if name not in members:
            raise ValueError(f"Missing remote output: {name}")
        return archive.extractfile(members[name]).read()

    package_hash = sha256(package / "package_checksums.json")
    if checkpoint:
        config, status = json.loads(read("run_config.json")), json.loads(read("status.json"))
        if (config["package_sha256"] != package_hash or
                not set(variables).issubset(config["variables"]) or
                status["state"] not in ("running", "complete")):
            raise ValueError("Incomplete, benchmark-only or mismatched GPU checkpoint")
        report = dict(state="checkpoint", package_sha256=package_hash,
                      variables=list(variables), device="GPU checkpoint (per-source verified)")
    else:
        report = json.loads(read("report.json"))
        if (report["state"] != "complete" or report["package_sha256"] != package_hash or
                not set(variables).issubset(report["variables"]) or
                report["dates"] != len(backend.days) or
                report["unique_sources"] != len(data["cells"])):
            raise ValueError("Incomplete, benchmark-only or mismatched GPU run")
    verified, samples = [], []
    expected_names = set()
    for variable in variables:
        for i, cell in enumerate(data["cells"]):
            if not data["valid"][variable, i]:
                continue
            key = hashlib.sha256(f"{backend.identity}:{cell}:{variable}".encode()).hexdigest()
            name, metadata_name = f"source_{key}.npz", f"source_{key}.json"
            expected_names.update((name, metadata_name))
            content, metadata_bytes = read(name), read(metadata_name)
            if not checkpoint:
                for filename, value in ((name, content), (metadata_name, metadata_bytes)):
                    if hashlib.sha256(value).hexdigest() != report["checksums"].get(filename):
                        raise ValueError(f"Changed remote output: {filename}")
            metadata = json.loads(metadata_bytes)
            if (metadata["sha256"] != hashlib.sha256(content).hexdigest() or
                    metadata["identity"] != backend.identity or metadata["source_cell"] != int(cell) or
                    metadata["variable"] != variable or metadata["dates"] != len(backend.days) or
                    metadata["package_sha256"] != package_hash):
                raise ValueError("Source receipt identity mismatch")
            targets = data["targets"][data["offsets"][i]:data["offsets"][i+1]]
            with np.load(io.BytesIO(content), allow_pickle=False) as saved:
                scores, counts = saved["rmse"], saved["date_count"]
                if (not np.array_equal(saved["targets"], targets) or
                        scores.shape != targets.shape or counts.shape != targets.shape or
                        not np.all(counts == len(backend.days)) or
                        not np.isfinite(scores).all() or (scores < 0).any() or
                        not np.all(scores[targets == cell] == 0)):
                    raise ValueError("Invalid all-date GPU footprint")
                choices = targets[targets != cell]
                if len(choices):
                    target = choices[len(choices)//2]
                    samples.append((variable, int(cell), int(target),
                                    float(scores[np.searchsorted(targets, target)])))
            verified.append((name, metadata_name, metadata))
    if not checkpoint:
        returned_names = set(report["checksums"])
        if (not expected_names.issubset(returned_names) or
                (set(variables) == set(report["variables"]) and returned_names != expected_names)):
            raise ValueError("Remote result set does not match all expected eligible footprints")
    validate_samples(backend, samples)
    return verified, read, report


def validate_samples(backend, samples):
    """Reconstruct one nonidentity pair over EVERY date for EVERY source/variable."""
    days = np.asarray(backend.days)
    count = len(days)
    per_batch = max(1, 65536//count)
    for start in range(0, len(samples), per_batch):
        batch = samples[start:start+per_batch]
        variable, source, target, expected = map(np.asarray, zip(*batch))
        date = np.tile(np.arange(count), len(batch))
        source = np.repeat(source, count)
        variable = np.repeat(variable, count)
        temperature = np.asarray(backend.weather[variable, source, date], dtype=np.float64)
        x = features(backend.table, source, np.repeat(target, count),
                     temperature, days[date], variable)
        _, predicted = backend.model.predict(x, temperature)
        actual = np.sqrt(np.square(predicted).reshape(len(batch), count).mean(axis=1))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5,
                                   err_msg="GPU all-date pooled predictions differ from verified CPU model")
    print(f"Verified every output and {len(samples)} independent all-date CPU pairs", flush=True)


def import_results(root, archive_path, package, variables=(0, 1), checkpoint=False):
    variables = tuple(variables)
    from precompute_frost import computation_lock
    backend = LongTermNetwork(root)
    with computation_lock(backend.folder):
        with tarfile.open(archive_path, "r:gz") as archive:
            verified, read, report = verify_archive(archive, package, backend, variables, checkpoint)
            for name, metadata_name, metadata in verified:
                path, receipt = backend.folder / name, backend.folder / metadata_name
                if receipt.exists():
                    local = json.loads(receipt.read_text())
                    if sha256(path) != local["sha256"]:
                        raise ValueError("Changed existing local source footprint")
                    with np.load(path, allow_pickle=False) as saved, np.load(
                            io.BytesIO(read(name)), allow_pickle=False) as returned:
                        np.testing.assert_array_equal(saved["targets"], returned["targets"])
                        np.testing.assert_array_equal(saved["date_count"], returned["date_count"])
                        np.testing.assert_allclose(saved["rmse"], returned["rmse"], rtol=2e-5, atol=2e-5)
                    continue
                pending = path.with_suffix(".pending.npz")
                pending.write_bytes(read(name))
                pending.replace(path)
                pending_receipt = receipt.with_suffix(".pending.json")
                pending_receipt.write_text(json.dumps(metadata))
                pending_receipt.replace(receipt)
            for variable in variables:
                backend.prepare_long_term(variable)
                backend.write_progress(variable, state="ready", completed=len(backend.stations),
                    total=len(backend.stations), device=report["device"])
    print(f"Long-term variables {list(variables)} are ready in the original app", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--package", type=Path, default=Path("outputs") / "senorge_long_term_gpu_v1")
    parser.add_argument("--variable", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--checkpoint", action="store_true",
                        help="Verify complete selected variables from source receipts before the final report exists")
    args = parser.parse_args()
    import_results(Path(__file__).parent, args.archive, args.package, args.variable, args.checkpoint)
