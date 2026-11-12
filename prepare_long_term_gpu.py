"""Package only eligible temperature-station inputs and frozen seNorge weights."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from senorge_long_term import LongTermNetwork
from senorge_corrected_transfer import FEATURES
from remote_senorge.train_valley_neural import sha256


def prepare(root, output):
    root, output = Path(root), Path(output)
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Choose a new package path; refusing to overwrite")
    backend = LongTermNetwork(root)
    cells = []
    for _, row in backend.stations.iterrows():
        x, y = backend.to_xy.transform(row.lon, row.lat)
        _, cell = backend.tree.query([x, y])
        if (np.max(np.abs(backend.table[cell, :2]-[x, y])) <= 500 and
                np.isfinite(backend.weather[:, cell]).all(axis=1).any()):
            cells.append(int(cell))
    cells = np.unique(cells).astype(np.int32)
    targets = [np.sort(backend.tree.query_ball_point(backend.table[cell, :2], 46000)).astype(np.int32)
               for cell in cells]
    output.mkdir(parents=True)
    np.savez_compressed(output / "inputs.npz", table=np.asarray(backend.table), cells=cells,
        temperatures=np.asarray(backend.weather[:, cells, :]), days=np.asarray(backend.days),
        valid=np.isfinite(backend.weather[:, cells, :]).all(axis=2),
        offsets=np.r_[0, np.cumsum([len(x) for x in targets])], targets=np.concatenate(targets))
    spec = dict(schema="senorge-long-term-gpu-v1", identity=backend.identity,
        catalogue_sha=backend.catalogue_sha, sensor_provenance=backend.sensor_provenance,
        features=FEATURES, selected=backend.model.selected, dates=len(backend.days),
        station_count=len(backend.stations), unique_sources=len(cells),
        network_keys={str(v): backend.network_paths(v)[0] for v in (0, 1)})
    (output / "spec.json").write_text(json.dumps(spec, indent=2))
    files = {
        "compute_long_term_gpu.py": root / "remote_senorge" / "compute_long_term_gpu.py",
        "train_senorge_transfer_gpu.py": root / "remote_senorge" / "train_senorge_transfer_gpu.py",
        "train_valley_neural.py": root / "remote_senorge" / "train_valley_neural.py",
        "senorge_corrected_transfer.py": root / "senorge_corrected_transfer.py",
        "README.md": root / "remote_senorge" / "LONG_TERM_GPU_README.md"}
    portable = root / "outputs" / "senorge_transfer_gpu_returned"
    info = json.loads((portable / "portable_ridge.json").read_text())
    names = {"portable_ridge.json", "portable_ridge.npz"}
    for network in info["networks"].values():
        names.update(network["checksums"])
    files.update({name: portable / name for name in names})
    for variable in (0, 1):
        for cell in cells:
            key = hashlib.sha256(f"{backend.identity}:{cell}:{variable}".encode()).hexdigest()
            path, receipt = backend.folder / f"source_{key}.npz", backend.folder / f"source_{key}.json"
            if receipt.exists():
                if sha256(path) != json.loads(receipt.read_text())["sha256"]:
                    raise ValueError("Changed local completed footprint")
                files[f"seed_{key}.npz"] = path
    for name, path in files.items():
        (output / name).write_bytes(path.read_bytes())
    checks = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    (output / "package_checksums.json").write_text(json.dumps(checks, indent=2))
    archive_path = output.with_suffix(".zip")
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in output.iterdir():
            archive.write(path, path.name)
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Package CRC failure")
        for name, digest in checks.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("Package checksum failure")
    print(json.dumps(dict(package=str(archive_path), bytes=archive_path.stat().st_size,
        sha256=sha256(archive_path), stations=len(backend.stations), unique_sources=len(cells),
        dates=len(backend.days), seeded_footprints=sum(name.startswith("seed_") for name in files)), indent=2))
    return archive_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs") / "senorge_long_term_gpu_v1")
    args = parser.parse_args()
    prepare(Path(__file__).parent, args.output)
