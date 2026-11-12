"""Train the detailed seNorge pilot, precompute 1 km Station Lab surfaces, and launch the app."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

import precompute_frost
from station_kuldehull import DEFAULT_PATH as KULDEHULL_PATH, KuldehullLayer
from pipeline_config import ROOT, WORK
from senorge_short_range_pilot import (
    DEFAULT_ARCHIVE,
    DEFAULT_FROST_PAIRS,
    DEFAULT_FROST_STATIONS,
    DEFAULT_GRID,
    STATIONLAB_PREFIX,
    run_pilot,
)


DEFAULT_PILOT_OUTPUT = WORK / "senorge_short_range_temperature_night"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _available_port(start, attempts=20):
    for port in range(start, start + attempts):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise OSError(f"No free local port in {start}-{start + attempts - 1}")


def _launch_station_lab(cell_m, port, log_path):
    port = _available_port(port)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    command = [
        sys.executable, "-u", str(ROOT / "station_app.py"),
        "--cell-m", str(cell_m), "--phenomenon", "temperature", "--port", str(port),
    ]
    options = dict(cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log_handle,
                   stderr=subprocess.STDOUT, close_fds=True)
    if os.name == "nt":
        options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **options)
    finally:
        log_handle.close()
    print(f"Station Lab started: http://127.0.0.1:{port} (PID {process.pid})")
    print(f"Station Lab log: {log_path}")
    return process.pid, port


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--frost-pairs", type=Path, default=DEFAULT_FROST_PAIRS)
    parser.add_argument("--frost-stations", type=Path, default=DEFAULT_FROST_STATIONS)
    parser.add_argument("--pilot-output", type=Path, default=DEFAULT_PILOT_OUTPUT)
    parser.add_argument("--max-grid-pairs", type=int, default=20_000)
    parser.add_argument("--max-frost-pairs", type=int, default=20_000)
    parser.add_argument("--date-step-days", type=int, default=1)
    parser.add_argument("--spatial-repeats", type=int, default=5)
    parser.add_argument("--cell-m", type=int, choices=[1000, 2500, 5000], default=1000)
    parser.add_argument("--port", type=int, default=8911)
    parser.add_argument("--skip-training", action="store_true",
                        help="Reuse the existing seNorge profile and resume any precompute checkpoints")
    parser.add_argument("--skip-grunnett", action="store_true",
                        help="Do not precompute the original Frost temperature profile")
    parser.add_argument("--no-launch", action="store_true",
                        help="Stop after precomputation instead of starting Station Lab")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.max_grid_pairs < 1 or args.max_frost_pairs < 20:
        parser.error("Pair counts must be positive; Frost sample must contain at least 20 pairs")
    if args.date_step_days < 1 or args.spatial_repeats < 1:
        parser.error("Date step and spatial repeats must be positive")
    if args.port < 1 or args.port > 65515:
        parser.error("Port must be between 1 and 65515")
    if args.log is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = WORK / f"senorge_stationlab_night_{stamp}.log"
    return args


def run(args):
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as log:
        tee = _Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Night run started: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Detailed pilot output: {args.pilot_output}")
            if args.skip_training:
                required = [
                    STATIONLAB_PREFIX.with_name(STATIONLAB_PREFIX.name + suffix)
                    for suffix in ("_model.joblib", "_metadata.json", "_station_features.gpkg")
                ]
                missing = [path for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"Cannot resume without the profile artifacts: {missing}")
                print("Training skipped; reusing the existing seNorge profile.")
            else:
                print("Training the detailed seNorge/Frost profile...")
                run_pilot(
                    archive_dir=args.archive_dir,
                    grid_path=args.grid,
                    frost_pairs_path=args.frost_pairs,
                    frost_stations_path=args.frost_stations,
                    output_dir=args.pilot_output,
                    max_grid_pairs=args.max_grid_pairs,
                    max_frost_pairs=args.max_frost_pairs,
                    date_step_days=args.date_step_days,
                    spatial_repeats=args.spatial_repeats,
                    export_stationlab=True,
                )

            if not args.skip_grunnett:
                print(f"Precomputing Grunnett temperature surface at {args.cell_m} m...")
                precompute_frost.main([
                    "--phenomenon", "temperature", "--cell-m", str(args.cell_m),
                    "--kind", "continuous",
                ])
            print(f"Precomputing seNorge-pilot temperature surface at {args.cell_m} m...")
            precompute_frost.main([
                "--phenomenon", "temperature", "--cell-m", str(args.cell_m),
                "--kind", "senorge",
            ])

            print("Preparing persistent Kuldehull display assets...", flush=True)
            if not KuldehullLayer(KULDEHULL_PATH).prepare():
                print("Kuldehull source not present; preparation skipped.", flush=True)
            if not args.no_launch:
                _launch_station_lab(
                    args.cell_m, args.port,
                    WORK / f"station_app_senorge_{args.cell_m}m.log",
                )
            print(f"Night run completed: {datetime.now().isoformat(timespec='seconds')}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except Exception:
        traceback.print_exc()
        print("Night run failed. Rerun with --skip-training to resume the precompute stage.",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())