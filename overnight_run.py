"""Run a logged collection/analysis queue, or a selected phenomenon overnight.

Run from the pipeline folder:
    python -u overnight_run.py
    python -u overnight_run.py --phenomenon humidity --years 5
    python -u overnight_run.py --plan temperature-priority
    python -u overnight_run.py --plan wind-priority
    python -u overnight_run.py --plan road-temperature-priority
    python overnight_run.py --phenomenon humidity --dry-run
Every command is logged. The legacy queue continues after failures; a selected
phenomenon or priority plan stops before dependent steps if a command fails.
"""
from pathlib import Path
from datetime import datetime
from contextlib import closing
import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pipeline_config import DATABASE, FROST_CLIENT_ID_ENV, NORWAY_POLYGON_FILE, PHENOMENA

ROOT = Path(__file__).resolve().parent
LOG = ROOT / f"overnight_{datetime.now():%Y%m%d_%H%M%S}.log"
PY = sys.executable
TEMPERATURE_PRIORITY = [
    ("humidity", 5000),
    ("temperature", 2500),
    ("humidity", 2500),
    ("temperature", 1000),
    ("humidity", 1000),
]
WIND_PRIORITY_ANALYSES = [("wind", "prediction_rmse")]
ROAD_TEMPERATURE_PRIORITY_ANALYSES = [("road_temperature", "prediction_rmse")]
SINGLE_5000M_PLANS = {
    "wind-priority": ("wind", WIND_PRIORITY_ANALYSES),
    "road-temperature-priority": ("road_temperature", ROAD_TEMPERATURE_PRIORITY_ANALYSES),
}

# Existing national grids are retained.
# Operational occurrence models run before experimental continuous snow/rain RMSE.
COMMANDS = [
    ("Preflight syntax check", [
        PY, "-m", "py_compile", "main.py", "information_gain.py",
        "pipeline_config.py", "download_frost.py", "check_frost_db.py"
    ]),
    ("Database report before download", [
        PY, "-u", "check_frost_db.py"
    ]),

    # --force-download is intentionally omitted: existing months should be kept
    # and only missing months in the requested five-year window downloaded.
    ("Download missing data for 5y temperature", [
        PY, "-u", "main.py", "--phenomenon", "temperature", "--years", "5",
        "--from-step", "download", "--to-step", "download"
    ]),
    ("Download missing data for 5y snow", [
        PY, "-u", "main.py", "--phenomenon", "snow", "--years", "5",
        "--from-step", "download", "--to-step", "download"
    ]),
    ("Download missing data for 5y precipitation", [
        PY, "-u", "main.py", "--phenomenon", "precipitation", "--years", "5",
        "--from-step", "download", "--to-step", "download"
    ]),
    ("Database report after download", [
        PY, "-u", "check_frost_db.py"
    ]),

    # Highest-priority temperature analyses.
    ("Temperature 5y continuous RMSE - full grid", [
        PY, "-u", "main.py", "--phenomenon", "temperature",
        "--metric", "prediction_rmse", "--years", "5",
        "--from-step", "information", "--skip-download", "--force"
    ]),
    ("Temperature 5y three-class error - full grid", [
        PY, "-u", "main.py", "--phenomenon", "temperature",
        "--metric", "temperature_class_error", "--years", "5",
        "--from-step", "information", "--skip-download", "--force"
    ]),

    # New occurrence models: run before experimental amount/depth RMSE models.
    ("Snow 5y occurrence error - full grid", [
        PY, "-u", "main.py", "--phenomenon", "snow",
        "--metric", "snow_occurrence_error", "--years", "5",
        "--from-step", "information", "--skip-download", "--force"
    ]),
    ("Precipitation 5y wet-dry occurrence error - full grid", [
        PY, "-u", "main.py", "--phenomenon", "precipitation",
        "--metric", "precipitation_occurrence_error", "--years", "5",
        "--from-step", "information", "--skip-download", "--force"
    ]),

    # Lower-priority experimental layers. These run if time remains.
    ("Snow 5y continuous depth RMSE - experimental full grid", [
        PY, "-u", "main.py", "--phenomenon", "snow",
        "--metric", "prediction_rmse", "--years", "5",
        "--from-step", "information", "--skip-download", "--force"
    ]),
    ("Precipitation 5y continuous amount RMSE - experimental full grid", [
        PY, "-u", "main.py", "--phenomenon", "precipitation",
        "--metric", "prediction_rmse", "--years", "5",
        "--from-step", "information", "--skip-download", "--force"
    ]),
    ("Final database report", [
        PY, "-u", "check_frost_db.py"
    ]),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--phenomenon", choices=PHENOMENA,
                        help="Run only this phenomenon with continuous RMSE; no phenomenon/plan preserves the legacy queue")
    selection.add_argument("--plan", choices=["temperature-priority", "precipitation-priority", *SINGLE_5000M_PLANS],
                           help="Select a fixed priority queue; wind/road-temperature plans download and retrain RMSE at 5000 m only")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--grid-spacing-m", "--grid-spacing", type=int, nargs="+",
                        choices=[1000, 2500, 5000], default=[5000, 2500, 1000],
                        help="Analysis grids, run coarse to fine (default: 5000 2500 1000)")
    parser.add_argument("--skip-download", action="store_true", help="Use observations already in the database")
    parser.add_argument("--force", action="store_true", help="Rebuild analysis outputs, never force re-download")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing logs or starting work")
    args = parser.parse_args(argv)
    if args.years < 1:
        parser.error("--years must be at least 1")
    args.grid_spacing_m = sorted(set(args.grid_spacing_m), reverse=True)
    if args.plan == "temperature-priority" and args.grid_spacing_m != [5000, 2500, 1000]:
        parser.error("--plan temperature-priority has a fixed grid sequence; omit --grid-spacing-m")
    if args.plan in SINGLE_5000M_PLANS:
        if args.grid_spacing_m != [5000, 2500, 1000]:
            parser.error(f"--plan {args.plan} is fixed at 5000 m; omit --grid-spacing-m")
        args.grid_spacing_m = [5000]
    if args.plan == "precipitation-priority":
        args.grid_spacing_m = [5000]
    if args.phenomenon is None and args.plan is None and (args.years != 5 or args.grid_spacing_m != [5000, 2500, 1000] or args.skip_download or args.force):
        parser.error("Select --phenomenon or --plan when changing years, grid spacing, download or force settings")
    return args


def commands_for(args):
    if args.phenomenon is None and args.plan is None:
        return COMMANDS
    preflight = ("Preflight syntax check", [PY, "-m", "py_compile", "main.py", "information_gain.py",
                                           "pipeline_config.py", "download_frost.py", "build_trainmap.py",
                                           "contour_terrain_metrics.py", "overnight_run.py"])
    if args.plan == "precipitation-priority":
        commands = [preflight]
        base = [PY, "-u", "main.py", "--phenomenon", "precipitation", "--years", str(args.years)]
        if not args.skip_download:
            commands.append(("Download daily and fallback ten-minute precipitation",
                             [*base, "--from-step", "download", "--to-step", "download"]))
        for metric in ("precipitation_occurrence_error", "prediction_rmse"):
            commands.append((f"Retrain precipitation {metric}",
                             [*base, "--metric", metric, "--grid-spacing-m", "5000",
                              "--from-step", "information", "--to-step", "information",
                              "--skip-download", "--force"]))
        if args.years == 5:
            commands.append(("Prepare updated precipitation maps for both networks",
                             [PY, "-u", "precompute_frost.py", "--phenomenon", "precipitation",
                              "--kind", "class", "--cell-m", "5000", "2500", "1000",
                              "--prepare-maps", "--station-network", "both"]))
        return commands
    if args.plan in SINGLE_5000M_PLANS:
        plan_phenomenon, plan_analyses = SINGLE_5000M_PLANS[args.plan]
        commands = [preflight]
        if not args.skip_download:
            commands.append((f"Download missing {args.years}y {plan_phenomenon} data",
                             [PY, "-u", "main.py", "--phenomenon", plan_phenomenon, "--years", str(args.years),
                              "--from-step", "download", "--to-step", "download"]))
        for phenomenon, metric in plan_analyses:
            output = ROOT / "outputs" / f"{phenomenon}_{args.years}y_5000m"
            analysis = [PY, "-u", "main.py", "--phenomenon", phenomenon, "--years", str(args.years),
                        "--metric", metric, "--grid-spacing-m", "5000", "--output-dir", str(output),
                        "--from-step", "information", "--to-step", "information", "--skip-download",
                        "--force"]
            commands.append((f"Train {phenomenon} {metric} at 5000 m using minimum single-station baseline", analysis))
        return commands
    schedule = TEMPERATURE_PRIORITY if args.plan else [(args.phenomenon, spacing) for spacing in args.grid_spacing_m]
    commands = [preflight]
    downloaded = set()
    for phenomenon, spacing in schedule:
        base = [PY, "-u", "main.py", "--phenomenon", phenomenon, "--years", str(args.years)]
        if not args.skip_download and phenomenon not in downloaded:
            commands.append((f"Download missing {args.years}y {phenomenon}",
                             [*base, "--from-step", "download", "--to-step", "download"]))
            downloaded.add(phenomenon)
        output = ROOT / "outputs" / f"{phenomenon}_{args.years}y_{spacing}m"
        analysis = [*base, "--grid-spacing-m", str(spacing), "--output-dir", str(output),
                    "--from-step", "information", "--to-step", "information",
                    "--metric", "prediction_rmse", "--skip-download"]
        if args.force:
            analysis.append("--force")
        commands.append((f"{phenomenon} {args.years}y continuous RMSE - {spacing}m full grid", analysis))
    return commands


def check_prerequisites(args):
    grids = [ROOT / "outputs" / f"norway_grid_{spacing}m.gpkg" for spacing in args.grid_spacing_m]
    for path in (DATABASE, NORWAY_POLYGON_FILE, *grids):
        if not path.exists():
            raise FileNotFoundError(f"Required overnight input is missing: {path}")
    if args.plan in SINGLE_5000M_PLANS:
        terrain_sources = (
            ROOT / "data" / "Basisdata_0000_Norge_25833_N100Kartdata_FGDB.gdb",
            ROOT / "data" / "Basisdata_0000_Norge_25833_NorgesMaritimeGrenser_FGDB.gdb",
        )
        for path in terrain_sources:
            if not path.exists():
                raise FileNotFoundError(f"Required {args.plan} terrain input is missing: {path}")
        import pyogrio
        grid_path = ROOT / "outputs" / "norway_grid_5000m.gpkg"
        grid_info = pyogrio.read_info(grid_path, layer="trainmap")
        required = {"elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype",
                    "relief_10km", "rugstd_10km", "tpi_10km",
                    "relief_25km", "rugstd_25km", "tpi_25km"}
        missing = required - set(grid_info.get("fields", []))
        if missing:
            raise RuntimeError(f"Existing 5000 m trainmap layer is missing fields: {sorted(missing)}")
    if not args.skip_download and not os.getenv(FROST_CLIENT_ID_ENV):
        raise RuntimeError(f"Set {FROST_CLIENT_ID_ENV} before downloading; its value is never logged")
    with closing(sqlite3.connect(DATABASE.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM stations WHERE inside_norway=1").fetchone()[0]
    if not count:
        raise RuntimeError("No Norwegian stations in the database; run fetch_stations.py first")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    print(f"Preflight: {count} Norwegian stations; {', '.join(grid.name for grid in grids)}; free disk {free_gib:.1f} GiB", flush=True)
    if not args.skip_download and free_gib < 30:
        print("WARNING: less than 30 GiB free. Multi-year raw observations and SQLite WAL may require substantial space; check capacity before leaving this run unattended", flush=True)


def emit(text):
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def run_queue(commands, stop_on_error=False):
    queue_start = time.monotonic()
    results = []
    emit(f"Started {datetime.now().isoformat(timespec='seconds')}")
    emit(f"Python: {PY}")
    emit(f"Working directory: {ROOT}")
    emit(f"Commands queued: {len(commands)}")
    for number, (title, cmd) in enumerate(commands, start=1):
        step_start = time.monotonic()
        emit("\n" + "=" * 80)
        emit(f"STEP {number}/{len(commands)}")
        emit(f"START {datetime.now().isoformat(timespec='seconds')} | {title}")
        emit("COMMAND: " + subprocess.list2cmdline(cmd))
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, errors="replace")
            if proc.stdout is not None:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    with LOG.open("a", encoding="utf-8") as output:
                        output.write(line)
            exit_code = proc.wait()
            elapsed = time.monotonic() - step_start
            status = "OK" if exit_code == 0 else "FAILED"
            results.append((title, status, exit_code, elapsed))
            emit(f"END {datetime.now().isoformat(timespec='seconds')} | {status} | "
                 f"exit={exit_code} | elapsed={elapsed / 60:.1f} min | {title}")
        except Exception as error:
            elapsed = time.monotonic() - step_start
            results.append((title, "FAILED TO START", None, elapsed))
            emit(f"FAILED TO START | elapsed={elapsed / 60:.1f} min | {type(error).__name__}: {error}")
        if stop_on_error and results[-1][1] != "OK":
            emit("Stopping: dependent steps will not run after a failed command")
            break
    emit("\n" + "=" * 80)
    emit("SUMMARY")
    for title, status, exit_code, elapsed in results:
        emit(f"{status:15} | exit={str(exit_code):>4} | {elapsed / 60:8.1f} min | {title}")
    emit(f"\nQueue finished {datetime.now().isoformat(timespec='seconds')}")
    emit(f"Total elapsed: {(time.monotonic() - queue_start) / 3600:.2f} hours")
    emit(f"Log: {LOG.name}")
    return 0 if all(result[1] == "OK" for result in results) else 1


def main(argv=None):
    global LOG
    args = parse_args(argv)
    commands = commands_for(args)
    if args.dry_run:
        for number, (title, command) in enumerate(commands, 1):
            print(f"{number}. {title}\n   {subprocess.list2cmdline(command)}")
        print("Dry run: no downloads, model training, reports or log files were started")
        return 0
    selected = args.phenomenon or args.plan
    if selected is not None:
        check_prerequisites(args)
        LOG = ROOT / f"overnight_{selected}_{datetime.now():%Y%m%d_%H%M%S}.log"
    return run_queue(commands, stop_on_error=selected is not None)


if __name__ == "__main__":
    sys.exit(main())
