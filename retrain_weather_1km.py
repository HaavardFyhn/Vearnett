"""Rebuild remaining Station Lab weather models offline; publish only ready 1 km maps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from calculation_progress import report
from precompute_frost import computation_lock


ROOT = Path(__file__).resolve().parent
MODELS = [
    ("wind", "prediction_rmse"),
    ("precipitation", "precipitation_occurrence_error"),
    ("snow", "snow_occurrence_error"),
    ("humidity", "prediction_rmse"),
    ("road_temperature", "prediction_rmse"),
]


def commands(phenomenon, metric):
    output = ROOT / "outputs" / "weather_1km" / phenomenon
    return [
        [sys.executable, "-u", str(ROOT / "main.py"), "--phenomenon", phenomenon,
         "--metric", metric, "--years", "5", "--grid-spacing-m", "1000",
         "--station-cell-inputs", "--output-dir", str(output), "--from-step", "information",
         "--to-step", "information", "--skip-download", "--force"],
        [sys.executable, "-u", str(ROOT / "precompute_frost.py"), "--phenomenon", phenomenon,
         "--kind", "class" if metric.endswith("occurrence_error") else "continuous",
         "--cell-m", "1000", "--station-network", "both", "--prepare-maps"],
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        for phenomenon, metric in MODELS:
            for command in commands(phenomenon, metric):
                print(subprocess.list2cmdline(command))
        return 0
    os.environ["WEATHER_CALCULATION_PROGRESS"] = str(ROOT / "outputs" / "weather_calculation_progress.json")
    os.environ["WEATHER_CALCULATION_STEPS"] = str(2 * len(MODELS))
    with computation_lock(ROOT / "outputs" / "weather_1km"):
        for number, (phenomenon, metric) in enumerate(MODELS):
            folder = ROOT / "outputs" / "weather_1km" / phenomenon
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "ready.json").exists():
                print(f"SKIP {phenomenon}: already published; keeping its live model", flush=True)
                continue
            for offset, command in enumerate(commands(phenomenon, metric), 1):
                os.environ["WEATHER_CALCULATION_STEP"] = str(number * 2 + offset)
                os.environ["WEATHER_CALCULATION_TITLE"] = phenomenon
                os.environ["WEATHER_MODEL_STAGING"] = phenomenon
                report("Starter modelltrening" if offset == 1 else "Starter kartberegning")
                print(subprocess.list2cmdline(command), flush=True)
                result = subprocess.run(command, cwd=ROOT, check=False)
                if result.returncode:
                    report("Beregningen feilet", state="failed",
                           error=f"{phenomenon}: command exited {result.returncode}")
                    return result.returncode
            temporary = folder / "ready.json.tmp"
            temporary.write_text(json.dumps(dict(station_input_policy="cell-1000m-v1", phenomenon=phenomenon)),
                                 encoding="utf-8")
            temporary.replace(folder / "ready.json")
        report("Alle 1 km-modeller og kart er klare. Start appen på nytt.", state="ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
