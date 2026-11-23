"""Verify and reuse the SSH host's existing terrain inputs, then run all four maps."""
import json
from pathlib import Path
import subprocess
import sys

from file_hash import sha256


def main():
    code = Path(__file__).resolve().parent
    stage = code / "stage"
    existing = code.parent / "outputs" / "portable_build" / "Remote_500m" / "data"
    shared = json.loads((code / "shared_inputs.json").read_text(encoding="utf-8"))
    for relative, expected in shared.items():
        source = existing / relative
        if not source.is_file() or sha256(source) != expected:
            raise SystemExit(f"Missing or different 500 m input: {source}; copy the matching local file and rerun.")
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink() and not target.exists():
            target.symlink_to(source)
        elif target.resolve() != source.resolve():
            raise SystemExit(f"Unexpected staged input: {target}")
    subprocess.run([sys.executable, "-u", str(code / "placement_gain.py"),
                    "--dataset", "temperature:direct_p95", "--station-network", "all-networks",
                    "--data-root", str(stage), "--work-dir", str(code / "work"),
                    "--workers", "14", "--radius-km", "100", "--overwrite", *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
