"""New seNorge GPU model adapter for the original Station Lab registry."""
import json
import threading

import numpy as np
from scipy.spatial import cKDTree

from greedy_400_stations_pruned import TERRAIN
from senorge_corrected_transfer import features
from senorge_network_map import SeNorgeNetworkMap, WARNING
from senorge_long_term import LongTermNetwork, WARNING as LONG_TERM_WARNING


PROFILE_LOCK = threading.RLock()


def refresh_progress(profile):
    path = profile.get("progress_path")
    if path is not None and path.exists():
        progress = json.loads(path.read_text())
        total, completed = progress["total"], progress["completed"]
        state = progress["state"]
        profile["status"].update(state=state, completed=completed, total=total,
            percent=100*completed/total if total else 0,
            error=progress.get("error") or
                  (f"Langtidsberegning {completed}/{total} stasjoner" if state == "running" else
                   "Langtidsberegning venter" if state == "queued" else None))


class SeNorgeGPUPairModel:
    allow_nodata = True

    def __init__(self, backend, day, variable):
        self.backend, self.day, self.variable = backend, day, variable
        self.date_index = None if day is None else backend.days.index(day)
        self.lock = threading.RLock()

    def cells(self, cache):
        x, y = cache.get("senorge_x", cache["x"]), cache.get("senorge_y", cache["y"])
        xy = np.column_stack([x, y])
        _, cells = self.backend.tree.query(xy)
        supported = np.max(np.abs(self.backend.table[cells, :2]-xy), axis=1) <= 500
        return cells.astype(np.int64), supported

    def predict_pairs(self, source, target, source_ids, target_ids):
        a, source_supported = self.cells(source)
        b, target_supported = self.cells(target)
        sources, targets = a[source_ids], b[target_ids]
        if self.day is None:
            result = np.full(len(source_ids), np.inf, np.float32)
            distance = np.linalg.norm(self.backend.table[sources, :2]-self.backend.table[targets, :2], axis=1)
            for cell in np.unique(sources[source_supported[source_ids]]):
                rows = np.flatnonzero((sources == cell) & source_supported[source_ids] &
                                     target_supported[target_ids] & (distance <= 46000))
                if not len(rows) or not np.isfinite(self.backend.weather[self.variable, cell]).all():
                    continue
                with self.lock:
                    footprint, rmse = self.backend.source_values(int(cell), self.variable)
                positions = np.searchsorted(footprint, targets[rows])
                inside = positions < len(footprint)
                inside[inside] &= footprint[positions[inside]] == targets[rows[inside]]
                result[rows[inside]] = rmse[positions[inside]]
            return result
        temperature = np.asarray(self.backend.weather[self.variable, sources, self.date_index])
        distance = np.linalg.norm(self.backend.table[sources, :2]-self.backend.table[targets, :2], axis=1)
        keep = (source_supported[source_ids] & target_supported[target_ids] &
                np.isfinite(temperature) & (distance <= 46000))
        result = np.full(len(source_ids), np.inf, np.float32)
        if keep.any():
            with self.lock:
                inputs = features(self.backend.table, sources[keep], targets[keep], temperature[keep],
                                  np.repeat(self.day, keep.sum()), np.full(keep.sum(), self.variable))
                _, rmse = self.backend.model.predict(inputs, temperature[keep])
            rmse[sources[keep] == targets[keep]] = 0
            result[keep] = rmse
        return result


def load_profile(registry, config):
    with PROFILE_LOCK:
        return _load_profile(registry, config)


def _load_profile(registry, config):
    long_term = config.get("long_term", False)
    backend_key = "senorge-gpu-long-term-backend" if long_term else "senorge-gpu-network-backend"
    if backend_key not in registry.models:
        registry.models[backend_key] = (LongTermNetwork if long_term else SeNorgeNetworkMap)(registry.root)
    backend = registry.models[backend_key]
    if long_term:
        _, path, receipt = backend.network_paths(config["variable"])
        if not receipt.exists():
            stations = backend.stations.copy()
            stations["x"], stations["y"] = backend.to_xy.transform(stations.lon.to_numpy(), stations.lat.to_numpy())
            stations["first_time"], stations["last_time"] = min(backend.days), max(backend.days)
            profile = dict(**config, baseline=None, second=None, winner=None,
                stations=stations, cache_path=receipt, cell_m=registry.cell_m,
                progress_path=receipt.with_suffix(".progress.json"),
                minimum=.0001, maximum=6.5, start=min(backend.days), end=max(backend.days),
                missing_features=0, classes=None, method="Langtids-RMSE per par, deretter beste faste stasjon",
                training_period="1996–2025", model_warning=LONG_TERM_WARNING, allow_nodata=True,
                status=dict(state="missing", completed=0, total=len(backend.stations), percent=0,
                    label=f"{config['label']} · RMSE", cell_m=registry.cell_m,
                    error="Langtidsberegning kreves: python -u senorge_long_term.py"))
            refresh_progress(profile)
            return profile
        result = backend.prepare_long_term(config["variable"])
    else:
        result = backend.prepare_network(config["day"], config["variable"])
    context, metadata = backend.context, result["metadata"]
    adapter = SeNorgeGPUPairModel(backend, None if long_term else config["day"], config["variable"])
    target_cells, supported = adapter.cells(registry.target_cache)
    baseline, second = np.full(len(target_cells), np.inf, np.float32), np.full(len(target_cells), np.inf, np.float32)
    winner = np.full(len(target_cells), -1, np.int32)
    baseline[supported], second[supported], winner[supported] = (
        context["best"][target_cells[supported]], context["second"][target_cells[supported]],
        context["winner"][target_cells[supported]])
    stations = backend.stations.copy()
    stations["x"], stations["y"] = backend.to_xy.transform(stations.lon.to_numpy(), stations.lat.to_numpy())
    _, indices = cKDTree(registry.grid[["x", "y"]].to_numpy()).query(stations[["x", "y"]].to_numpy())
    for field in ("elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype", *TERRAIN):
        stations[field] = registry.grid.iloc[indices][field].to_numpy()
    stations["objtype_code"] = registry.grid.iloc[indices]["objtype_code"].to_numpy()
    start = min(backend.days) if long_term else config["day"]
    end = max(backend.days) if long_term else config["day"]
    stations["first_time"], stations["last_time"] = start, end
    finite = baseline[np.isfinite(baseline)]
    if not len(finite):
        raise ValueError("Ingen støttede seNorge GPU-prediksjoner i appens rutenett")
    lower, upper = np.percentile(finite, [0.5, 99.5])
    return dict(**config, model=adapter, baseline=baseline, second=second, winner=winner,
        stations=stations, cache_path=backend.folder / f"{metadata['key']}.npz", cell_m=registry.cell_m,
        minimum=max(float(lower), .0001), maximum=6.5,
        frost_maximum=max(float(upper), max(float(lower), .0001)*1.01),
        start=start, end=end, missing_features=0, classes=None,
        method=("Langtids-RMSE per par, deretter beste faste Frost-referanse · 46 km" if long_term else
                "Eksakt minimum korrigert RMSE over Frost-lokasjoner · seNorge-temperatur, 46 km"),
        training_period="1996–2025", model_warning=(LONG_TERM_WARNING if long_term else
                                                 WARNING+f" Kartdato: {config['day']}. ")+
        f"{int((~np.isfinite(baseline)).sum())} appruter uten støttet referanse er NoData.",
        allow_nodata=True,
        status=dict(state="ready", completed=len(stations), total=len(stations), percent=100,
                    label=f"{config['label']} · RMSE", cell_m=registry.cell_m, error=None))
