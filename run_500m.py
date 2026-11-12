"""Resumable offline 500 m preparation; reuses the fitted seNorge temperature model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from grid_resolution import cell_policy, weather_folder


ROOT = Path(__file__).resolve().parent
MODELS = [
    ("wind", "prediction_rmse"),
    ("precipitation", "precipitation_occurrence_error"),
    ("snow", "snow_occurrence_error"),
    ("humidity", "prediction_rmse"),
    ("road_temperature", "prediction_rmse"),
]


def commands(phenomenon, metric):
    folder = weather_folder(ROOT, 500) / phenomenon
    return [
        [sys.executable, "-u", str(ROOT / "main.py"), "--phenomenon", phenomenon,
         "--metric", metric, "--years", "5", "--grid-spacing-m", "500",
         "--station-cell-inputs", "--output-dir", str(folder), "--from-step", "information",
         "--to-step", "information", "--skip-download", "--force"],
        [sys.executable, "-u", str(ROOT / "precompute_frost.py"), "--phenomenon", phenomenon,
         "--kind", "class" if metric.endswith("occurrence_error") else "continuous",
         "--cell-m", "500", "--station-network", "both", "--prepare-maps"],
    ]


def write_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
    pending.replace(path)


def execute(command, title, threads, staging=None):
    folder = weather_folder(ROOT, 500)
    folder.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, WEATHER_CPU_THREADS=str(threads),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        WEATHER_CALCULATION_PROGRESS=str(folder / f"{title}_progress.json"))
    if staging:
        environment["WEATHER_MODEL_STAGING"] = staging
    else:
        environment.pop("WEATHER_MODEL_STAGING", None)
    print(f"START {title}: {subprocess.list2cmdline(command)}", flush=True)
    with (folder / f"{title}.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{title} failed ({result.returncode}); see {folder / (title + '.log')}")
    print(f"DONE {title}", flush=True)


def completed(path, fingerprint):
    if not path.is_file():
        return False
    if json.loads(path.read_text(encoding="utf-8")).get("fingerprint") != fingerprint:
        raise ValueError(f"Inputs/code changed since {path}; use a fresh remote work directory")
    return True


def workflow_fingerprint():
    digest = hashlib.sha256()
    paths = sorted(ROOT.glob("*.py")) + [ROOT / "remote_500m_inputs.json"]
    for path in paths:
        if path.name.startswith("test_") or not path.is_file():
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_inputs():
    from remote_senorge.train_remote import sha256
    manifest = json.loads((ROOT / "remote_500m_inputs.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            reason = "missing file"
        elif path.stat().st_size != entry["bytes"]:
            reason = f"size mismatch: expected {entry['bytes']} bytes, got {path.stat().st_size}"
        else:
            actual = sha256(path)
            reason = None if actual == entry["sha256"] else (
                f"SHA256 mismatch: expected {entry['sha256']}, got {actual}")
        if reason:
            failures.append(f"{entry['path']}: {reason}")
    if failures:
        raise ValueError("Transferred input verification failed; re-upload these files (do not skip verification):\n"
                         + "\n".join(failures))
    print(f"Verified {len(manifest['files'])} input files", flush=True)


def run(args):
    folder = weather_folder(ROOT, 500)
    fingerprint = workflow_fingerprint()
    phases = ("terrain", "train", "maps", "roads", "stage") if args.phase == "all" else (args.phase,)
    if args.dry_run:
        print(f"CPU training: at most {args.jobs} parallel jobs, {args.threads} threads/job. GPU not required.")
        for name, metric in MODELS:
            for command in commands(name, metric):
                print(subprocess.list2cmdline(command))
        print("Order: verify inputs -> terrain -> parallel training -> sequential maps -> roads -> portable stage")
        return
    verify_inputs()
    folder.mkdir(parents=True, exist_ok=True)
    from precompute_frost import computation_lock
    with computation_lock(folder):
        terrain_done = folder / "terrain_done.json"
        if "terrain" in phases and not completed(terrain_done, fingerprint):
            execute([sys.executable, "-u", str(ROOT / "build_trainmap.py"), "--spacing-m", "500", "--force"],
                    "terrain", args.threads)
            import pyogrio
            import numpy as np
            grid = pyogrio.read_dataframe(ROOT / "outputs" / "norway_grid_500m.gpkg",
                                          columns=["x", "y"], read_geometry=False)
            if grid.empty or not np.isfinite(grid.to_numpy()).all() or not np.isclose(grid.to_numpy() % 500, 0).all():
                raise ValueError("Invalid 500 m terrain lattice")
            write_json(terrain_done, dict(fingerprint=fingerprint, cells=len(grid)))
        if not completed(terrain_done, fingerprint):
            raise ValueError("Complete --phase terrain first")

        def train(item):
            phenomenon, metric = item
            model_folder = folder / phenomenon
            model_folder.mkdir(exist_ok=True)
            marker = model_folder / "trained.json"
            if not completed(marker, fingerprint):
                execute(commands(phenomenon, metric)[0], phenomenon + "_train", args.threads)
                write_json(marker, dict(fingerprint=fingerprint, station_input_policy=cell_policy(500)))

        if "train" in phases:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(train, MODELS))
        if "maps" in phases:
            marker = folder / "temperature_maps_done.json"
            if not completed(marker, fingerprint):
                execute([sys.executable, "-u", str(ROOT / "precompute_frost.py"), "--phenomenon", "temperature",
                         "--kind", "senorge30", "--cell-m", "500", "--station-network", "both", "--prepare-maps"],
                        "temperature_maps", args.threads)
                write_json(marker, dict(fingerprint=fingerprint))
            for phenomenon, metric in MODELS:
                model_folder = folder / phenomenon
                if not completed(model_folder / "trained.json", fingerprint):
                    raise ValueError(f"Complete --phase train first: {phenomenon}")
                marker = model_folder / "ready.json"
                if completed(marker, fingerprint):
                    continue
                execute(commands(phenomenon, metric)[1], phenomenon + "_maps", args.threads, staging=phenomenon)
                write_json(marker, dict(fingerprint=fingerprint, station_input_policy=cell_policy(500),
                                        phenomenon=phenomenon))
        if "roads" in phases:
            marker = folder / "roads_done.json"
            if not completed(marker, fingerprint):
                execute([sys.executable, "-u", str(ROOT / "precompute_road_grid.py"), "--cell-m", "500"],
                        "roads", args.threads)
                write_json(marker, dict(fingerprint=fingerprint))
        if "stage" in phases:
            required = [folder / "temperature_maps_done.json", folder / "roads_done.json",
                        *(folder / phenomenon / "ready.json" for phenomenon, _ in MODELS)]
            if not all(completed(path, fingerprint) for path in required):
                raise ValueError("Finish maps and roads before creating the portable stage")
            marker = folder / f"{args.stage_name}_done.json"
            if not completed(marker, fingerprint):
                execute([sys.executable, "-u", str(ROOT / "build_portable.py"), "--stage-only",
                         "--cell-m", "500", "--name", args.stage_name], "portable_stage", args.threads)
                write_json(marker, dict(fingerprint=fingerprint))
            print(f"RETURN STAGE: {ROOT / 'outputs' / 'portable_build' / args.stage_name / 'data'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--phase", choices=["all", "terrain", "train", "maps", "roads", "stage"], default="all")
    parser.add_argument("--stage-name", default="Remote_500m")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Check all input files without starting calculations")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.threads < 1 or args.jobs * args.threads > (os.cpu_count() or 1):
        parser.error("--jobs × --threads must fit the available logical CPUs")
    if Path(args.stage_name).name != args.stage_name:
        parser.error("--stage-name must be a folder name")
    if args.verify_only:
        verify_inputs()
        return
    if not args.dry_run:
        print(f"Free storage: {shutil.disk_usage(ROOT).free / 1024**3:.1f} GiB; input data stays in place", flush=True)
    run(args)


if __name__ == "__main__":
    main()
