"""Persist Frost station holders separately from model and observation caches."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile


OWNER_FILE = Path("data") / "station_owners.json"


def owner_counts(stations):
    counts = Counter()
    seen = set()
    unknown = 0
    for station in stations:
        identifier = station.get("station_id")
        if not identifier or identifier in seen or station.get("removed"):
            continue
        seen.add(identifier)
        owners = station.get("owners", [])
        if owners:
            counts.update(set(owners))
        else:
            unknown += 1
    return dict(stations=len(seen), unknown=unknown,
                owners=[dict(name=name, count=count) for name, count in
                        sorted(counts.items(), key=lambda item: (-item[1], item[0]))])


@lru_cache(maxsize=4)
def _read_catalog(path, modified, size):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not isinstance(data.get("stations"), dict) or not isinstance(data.get("updated"), str):
        raise ValueError("Ugyldig fil med stasjonseiere")
    for identifier, station in data["stations"].items():
        if (not isinstance(identifier, str) or not isinstance(station, dict)
                or not isinstance(station.get("owners"), list)
                or any(not isinstance(owner, str) or not owner.strip() for owner in station["owners"])):
            raise ValueError("Ugyldige stasjonseiere")
    return data


def load_catalog(*roots):
    for root in roots:
        path = Path(root) / OWNER_FILE
        if path.is_file():
            stat = path.stat()
            return _read_catalog(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    return None


def refresh(output):
    from fetch_stations import _request_all
    from pipeline_config import FROST_BASE_URL, FROST_CLIENT_ID_ENV

    client_id = os.environ.get(FROST_CLIENT_ID_ENV)
    if not client_id:
        raise RuntimeError(f"Set environment variable {FROST_CLIENT_ID_ENV} before updating owners")
    stations = {}
    for item in _request_all(FROST_BASE_URL + "/sources/v0.jsonld",
                            dict(country="NO", types="SensorSystem", fields="id,name,stationHolders"), client_id):
        identifier = item.get("id")
        holders = item.get("stationHolders", [])
        if not isinstance(identifier, str) or not identifier.startswith("SN"):
            raise ValueError("Frost returned an invalid station ID")
        if not isinstance(holders, list) or any(not isinstance(holder, str) or not holder.strip() for holder in holders):
            raise ValueError(f"Frost returned invalid stationHolders for {identifier}")
        stations[identifier.split(":")[0]] = dict(name=item.get("name", ""),
                                                 owners=sorted(set(holder.strip() for holder in holders)))
    if not stations:
        raise ValueError("Frost returned no Norwegian stations; previous owner data was not replaced")
    data = dict(schema=1, updated=datetime.now(timezone.utc).isoformat(),
                source="Frost sources/v0 stationHolders", stations=stations)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    summary = owner_counts(dict(station_id=identifier, **station) for identifier, station in stations.items())
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["Eier", "Stasjoner"])
        writer.writerows((item["name"], item["count"]) for item in summary["owners"])
        writer.writerow(["Ikke oppgitt", summary["unknown"]])
    print(f"Stored holders for {summary['stations']:,} Norwegian Frost stations; "
          f"{len(summary['owners'])} owners; {summary['unknown']} without holders")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / OWNER_FILE)
    refresh(parser.parse_args().output)
