"""Atomic, read-only-in-the-app progress reporting for offline calculations."""
import json
import os
from pathlib import Path
import tempfile
import time


def report(stage, completed=0, total=None, started=None, state="running", error=None):
    destination = os.environ.get("WEATHER_CALCULATION_PROGRESS")
    elapsed = None if started is None else max(0, time.monotonic() - started)
    remaining = (elapsed * (total - completed) / completed
                 if elapsed is not None and total and 0 < completed <= total else None)
    payload = dict(stage=stage, completed=completed, total=total, state=state,
                   elapsed_seconds=elapsed, remaining_seconds=remaining, error=error,
                   step=os.environ.get("WEATHER_CALCULATION_STEP"),
                   steps=os.environ.get("WEATHER_CALCULATION_STEPS"),
                   title=os.environ.get("WEATHER_CALCULATION_TITLE"), updated_at=time.time())
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.stem, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return payload


def read_progress(root):
    path = Path(root) / "outputs" / "weather_calculation_progress.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    result["stale"] = result["state"] == "running" and time.time() - result["updated_at"] > 180
    return result
