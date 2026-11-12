"""Resumable all-date corrected RMSE, pooled before choosing a reference station."""
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from remote_senorge.train_valley_neural import sha256
from senorge_corrected_transfer import features
from senorge_network_map import SeNorgeNetworkMap


WARNING = (
    "Long-term corrected predicted RMSE = sqrt(mean predicted squared corrected error), "
    "using all 3600 sampled seNorge dates in 1996-2025, equally weighted. "
    "Pool dates separately for each station-target pair BEFORE selecting the best station. "
    "No daily station switching. Only stations eligible in the original app's temperature "
    "setting are used (air_temperature observations and usable terrain features). "
    "Those Frost sensor locations use archived seNorge temperatures, "
    "not actual Frost readings. This is best-source coverage, not multi-station fusion or "
    "independent validation. Only references within 46 km; unsupported cells remain NoData. "
    "Removing a station selects the next-best long-term source and cannot lower RMSE. "
    "Known source-cell identity has zero transfer error."
)


def pooled_rmse(rmse):
    values = np.asarray(rmse, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Finite nonnegative pair-by-date RMSE predictions required")
    return np.sqrt(np.mean(np.square(values), axis=1))


def temperature_stations(root):
    """Use exactly the original continuous-temperature registry's sensor eligibility."""
    import pyogrio
    from greedy_400_stations_pruned import TERRAIN
    from station_weather import observation_stations, WEATHER
    root = Path(root)
    path = root / "outputs" / "temperature_5y_prediction_rmse_station_features.gpkg"
    terrain = pyogrio.read_dataframe(path).drop_duplicates("station_id")
    observations, start, end = observation_stations(
        root / "frost_history.db", root / "frost_db_report" / "station_element_all.csv",
        WEATHER["temperature"][1], station_ids=terrain["station_id"])
    fields = ["station_id", "x", "y", "elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype", *TERRAIN]
    stations = observations[["station_id", "name", "lat", "lon", "first_time", "last_time"]].merge(
        terrain[fields], on="station_id", how="inner").dropna(subset=fields[1:]).reset_index(drop=True)
    if stations.empty:
        raise ValueError("No eligible temperature-sensor stations from the original app")
    provenance = dict(element=WEATHER["temperature"][1], availability_start=start, availability_end=end,
                      terrain_sha256=sha256(path),
                      stations=stations[["station_id", "lat", "lon"]].to_dict("records"))
    return stations, provenance


class LongTermNetwork(SeNorgeNetworkMap):
    def __init__(self, root):
        super().__init__(root)
        self.stations, self.sensor_provenance = temperature_stations(root)
        self.catalogue_sha = hashlib.sha256(json.dumps(self.sensor_provenance, sort_keys=True).encode()).hexdigest()
        self.folder = self.root / "outputs" / "senorge_long_term"
        self.folder.mkdir(exist_ok=True)
        self.pool_lock = threading.RLock()
        self.identity = hashlib.sha256(json.dumps(dict(
            schema="all-dates-fixed-reference-v1", code=sha256(Path(__file__)),
            adapter=sha256(self.root / "senorge_transfer_map.py"),
            features=sha256(self.root / "senorge_corrected_transfer.py"),
            manifest=self.model.manifest,
            portable=sha256(self.root / "outputs" / "senorge_transfer_gpu_returned" / "portable_ridge.json"),
            days=self.days), sort_keys=True).encode()).hexdigest()

    def network_paths(self, variable):
        key = hashlib.sha256(f"{self.identity}:{self.catalogue_sha}:{variable}".encode()).hexdigest()
        return key, self.folder / f"{key}.npz", self.folder / f"{key}.json"

    def write_progress(self, variable, **values):
        key, _, _ = self.network_paths(variable)
        path = self.folder / f"{key}.progress.json"
        pending = path.with_suffix(".pending.json")
        pending.write_text(json.dumps(values))
        pending.replace(path)

    def source_values(self, cell, variable):
        """Persist the complete native footprint once per unique source cell/variable."""
        key = hashlib.sha256(f"{self.identity}:{cell}:{variable}".encode()).hexdigest()
        path, receipt = self.folder / f"source_{key}.npz", self.folder / f"source_{key}.json"
        with self.pool_lock:
            if receipt.exists():
                metadata = json.loads(receipt.read_text())
                if sha256(path) != metadata["sha256"]:
                    raise ValueError("Changed long-term source footprint")
                with np.load(path, allow_pickle=False) as saved:
                    return saved["targets"].copy(), saved["rmse"].copy()
            temperatures = np.asarray(self.weather[variable, cell], dtype=np.float64)
            if not np.isfinite(temperatures).all():
                raise ValueError(f"Source cell {cell} lacks one or more sampled dates; cannot pool all dates")
            targets = np.sort(self.tree.query_ball_point(self.table[cell, :2], 46000)).astype(np.int32)
            days = np.asarray(self.days)
            count = len(days)
            total = len(targets)*count
            sums = np.zeros(len(targets), np.float64)
            counts = np.zeros(len(targets), np.int32)
            for start in range(0, total, 65536):
                flat = np.arange(start, min(start+65536, total))
                pair, date = flat//count, flat % count
                x = features(self.table, np.full(len(flat), cell), targets[pair],
                             temperatures[date], days[date], np.full(len(flat), variable))
                _, predicted_rmse = self.model.predict(x, temperatures[date])
                squared = np.square(predicted_rmse.astype(np.float64))
                if not np.isfinite(squared).all():
                    raise ValueError("Invalid all-date squared-error predictions")
                np.add.at(sums, pair, squared)
                np.add.at(counts, pair, 1)
            if not np.all(counts == count):
                raise ValueError("Incomplete all-date source footprint")
            rmse = np.sqrt(sums/count).astype(np.float32)
            rmse[targets == cell] = 0
            pending = path.with_suffix(".pending.npz")
            np.savez_compressed(pending, targets=targets, rmse=rmse, date_count=counts)
            pending.replace(path)
            pending_receipt = receipt.with_suffix(".pending.json")
            pending_receipt.write_text(json.dumps(dict(
                sha256=sha256(path), dates=count, source_cell=int(cell), variable=variable,
                identity=self.identity)))
            pending_receipt.replace(receipt)
            return targets, rmse

    def prepare_long_term(self, variable, progress=lambda completed, total: None):
        if variable not in (0, 1):
            raise ValueError("Choose daily mean or daily minimum")
        key, path, receipt = self.network_paths(variable)
        if receipt.exists():
            metadata = json.loads(receipt.read_text())
            if sha256(path) != metadata["sha256"]:
                raise ValueError("Changed long-term network predictions")
            with np.load(path, allow_pickle=False) as saved:
                context = {name: saved[name].copy() for name in saved.files}
        else:
            n = len(self.table)
            best, second = np.full(n, np.inf, np.float32), np.full(n, np.inf, np.float32)
            winner = np.full(n, -1, np.int32)
            source_cells = np.full(len(self.stations), -1, np.int32)
            reasons = []
            for station, row in self.stations.iterrows():
                x, y = self.to_xy.transform(row.lon, row.lat)
                _, cell = self.tree.query([x, y])
                if np.max(np.abs(self.table[cell, :2]-[x, y])) > 500:
                    reasons.append("Outside supported native land grid")
                elif not np.isfinite(self.weather[variable, cell]).all():
                    reasons.append("Incomplete temperature archive across sampled dates")
                else:
                    reasons.append("")
                    source_cells[station] = cell
                    targets, rmse = self.source_values(int(cell), variable)
                    better = rmse < best[targets]
                    second[targets] = np.where(better, best[targets], np.minimum(second[targets], rmse))
                    best[targets] = np.minimum(best[targets], rmse)
                    winner[targets] = np.where(better, station, winner[targets])
                progress(station+1, len(self.stations))
            valid = np.isfinite(best)
            if not valid.any() or np.any(second[valid] < best[valid]):
                raise ValueError("Invalid long-term network ranking")
            context = dict(best=best, second=second, winner=winner,
                           source_cell=source_cells, reasons=np.asarray(reasons, dtype=str))
            pending = path.with_suffix(".pending.npz")
            np.savez_compressed(pending, **context)
            pending.replace(path)
            metadata = dict(key=key, sha256=sha256(path), dates=len(self.days),
                first_date=min(self.days), last_date=max(self.days), variable=variable,
                cells=int(valid.sum()), no_prediction_cells=int((~valid).sum()),
                eligible_stations=int((source_cells >= 0).sum()), warning=WARNING,
                aggregation="sqrt(mean predicted MSE per pair), then minimum over fixed references")
            metadata["temperature_sensors"] = self.sensor_provenance
            pending_receipt = receipt.with_suffix(".pending.json")
            pending_receipt.write_text(json.dumps(metadata, indent=2))
            pending_receipt.replace(receipt)
        self.context, self.metadata, self.key = context, metadata, key
        return dict(metadata=metadata)


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variable", type=int, choices=[0, 1], nargs="+", default=[0, 1])
    args = parser.parse_args()
    backend = LongTermNetwork(Path(__file__).parent)
    from precompute_frost import computation_lock
    with computation_lock(backend.folder):
        for variable in args.variable:
            backend.write_progress(variable, state="queued", completed=0, total=len(backend.stations))
        for variable in args.variable:
            started = time.monotonic()
            def progress(completed, total):
                elapsed = time.monotonic()-started
                backend.write_progress(variable, state="running", completed=completed, total=total,
                                       elapsed_s=elapsed)
                print(f"variable={variable} station={completed}/{total} elapsed_s={elapsed:.1f}", flush=True)
            progress(0, len(backend.stations))
            try:
                print(json.dumps(backend.prepare_long_term(variable, progress)["metadata"]), flush=True)
                backend.write_progress(variable, state="ready", completed=len(backend.stations),
                                       total=len(backend.stations), elapsed_s=time.monotonic()-started)
            except Exception as error:
                backend.write_progress(variable, state="failed", completed=0, total=len(backend.stations),
                                       error=str(error))
                raise
