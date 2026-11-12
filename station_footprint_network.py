"""Frost-only minima and exact leave-one-station-out footprint contributions."""
import argparse
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from remote_senorge.footprint_model import TARGETS
from remote_senorge.train_remote import sha256, write_json
from station_senorge import BasinLookup
from station_weather import save_exact_cache
from station_footprint import WARNING, TARGET_LABELS, raster_png, raster_export


NETWORK_WARNING = (
    "Frost-nett: laveste anslåtte RMSE fra én tilgjengelig stasjon per 1 km-rute, "
    "ikke samlet assimilering. Bare stasjonenes trente 64 × 64 km footprint brukes. "
    "Ingen dekning gir NoData. Bidrag = RMSE uten valgt stasjon minus RMSE med hele nettet. "
    "Hvis valgt stasjon gir eneste dekning, vises dekningstap separat uten numerisk bidrag. "
    "Stasjoner i samme 1 km-rute har identiske footprint og erstatter hverandre i dette anslaget. "
) + WARNING


def update_best(best, second, winner, runner, counts, ids, values, station):
    """Stable ties follow catalogue order; a different station may have the same reference."""
    if values.shape != (6, len(ids)) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid station footprint values")
    counts[ids] += 1
    for target in range(6):
        p = values[target]
        improved = p < best[target, ids]
        other = ~improved & (p < second[target, ids])
        new_ids = ids[improved]
        second[target, new_ids] = best[target, new_ids]
        runner[target, new_ids] = winner[target, new_ids]
        best[target, new_ids] = p[improved]
        winner[target, new_ids] = station
        next_ids = ids[other]
        second[target, next_ids] = p[other]
        runner[target, next_ids] = station


class FootprintNetwork:
    def __init__(self, backend, stations, cache=None):
        self.backend = backend
        self.stations = sorted((dict(item) for item in stations if item["supported"]),
                               key=lambda item: item["station_id"])
        if not self.stations or len({item["station_id"] for item in self.stations}) != len(self.stations):
            raise ValueError("Frost network needs unique supported station IDs")
        self.excluded = [item["station_id"] for item in stations if not item["supported"]]
        signature = dict(schema="frost-footprint-best-two-v1",
                         model=backend.metadata["checksums"]["selected_model.pt"],
                         dataset=backend.metadata["manifest_sha256"],
                         stations=[(item["station_id"], item["reference"]) for item in self.stations])
        self.key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        base = Path(cache) if cache is not None else backend.root / "outputs" / "station_app_cache"
        self.folder = base / f"footprint_network_{self.key}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.signature = signature
        self.lock = threading.RLock()
        self.arrays = None
        xy = backend.atlas.table[:, :2]
        self.origin = (int(xy[:, 0].min()), int(xy[:, 1].max()))
        self.cols = np.round((xy[:, 0]-self.origin[0])/1000).astype(int)
        self.rows = np.round((self.origin[1]-xy[:, 1])/1000).astype(int)
        self.shape = (int(self.rows.max())+1, int(self.cols.max())+1)

    def status(self):
        path = self.folder / "status.json"
        if not path.exists():
            return dict(state="missing", completed=0, stations=len(self.stations))
        return json.loads(path.read_text())

    def map_path(self, ref):
        return self.folder / f"reference_{int(ref)}.npz"

    def saved_map(self, ref):
        path = self.map_path(ref)
        if not path.exists():
            values, mapping = self.backend.predict_reference(ref)
            save_exact_cache(path, values, mapping, np.empty(0, np.int32), 1)
        with np.load(path, allow_pickle=False) as data:
            values, mapping = data["baseline"].copy(), data["second"].copy()
        if (values.shape != (6, 64, 64) or mapping.shape != (64, 64) or
                not np.issubdtype(mapping.dtype, np.integer) or mapping.min() < -1 or
                mapping.max() >= len(self.backend.atlas.table) or
                not np.isfinite(values).all() or (values < 0).any()):
            raise ValueError(f"Invalid cached reference footprint: {ref}")
        return values, mapping

    def prepare(self):
        from precompute_frost import computation_lock
        with computation_lock(self.folder):
            if self.status()["state"] == "ready":
                self.load()
                return
            try:
                write_json(self.folder / "signature.json", self.signature)
                n = len(self.backend.atlas.table)
                path = self.folder / "best_two.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as data:
                        arrays = dict(best=data["baseline"].copy(), second=data["second"].copy(),
                                      winner=data["winner"][0].copy(), runner=data["winner"][1].copy(),
                                      counts=data["counts"].copy())
                        completed = int(data["completed"])
                else:
                    arrays = dict(best=np.full((6, n), np.inf, np.float32),
                                  second=np.full((6, n), np.inf, np.float32),
                                  winner=np.full((6, n), -1, np.int32),
                                  runner=np.full((6, n), -1, np.int32),
                                  counts=np.zeros(n, np.int32))
                    completed = 0
                self.validate(arrays, completed)
                write_json(self.folder / "status.json", dict(state="running", completed=completed,
                                                            stations=len(self.stations)))
                for index in range(completed, len(self.stations)):
                    record = self.stations[index]
                    values, mapping = self.saved_map(record["reference"])
                    keep = mapping >= 0
                    update_best(**arrays, ids=mapping[keep], values=values[:, keep], station=index)
                    if (index+1) % 16 == 0 or index+1 == len(self.stations):
                        # Reuse the repository's atomic/Windows-lock-safe checkpoint writer.
                        save_exact_cache(path, arrays["best"], arrays["second"],
                                         np.stack([arrays["winner"], arrays["runner"]]), index+1,
                                         extra=dict(counts=arrays["counts"]))
                        write_json(self.folder / "status.json", dict(state="running", completed=index+1,
                                                                    stations=len(self.stations)))
                        print(f"Frost footprints: {index+1}/{len(self.stations)} stations", flush=True)
                self.arrays = arrays
                write_json(self.folder / "complete.json", dict(
                    checksums={p.name: sha256(p) for p in self.folder.glob("*.npz")}))
                write_json(self.folder / "status.json", dict(state="ready", completed=len(self.stations),
                                                            stations=len(self.stations)))
            except Exception as error:
                write_json(self.folder / "status.json", dict(state="failed", error=str(error),
                                                            stations=len(self.stations)))
                raise

    def validate(self, arrays, completed):
        n = len(self.backend.atlas.table)
        if (not 0 <= completed <= len(self.stations) or set(arrays) !=
                {"best", "second", "winner", "runner", "counts"} or
                any(arrays[name].shape != (6, n) for name in ("best", "second", "winner", "runner")) or
                arrays["counts"].shape != (n,)):
            raise ValueError("Invalid network checkpoint shapes/progress")
        best, second = arrays["best"], arrays["second"]
        if (np.isnan(best).any() or np.isnan(second).any() or (best < 0).any() or
                (second < best).any() or (arrays["counts"] < 0).any() or
                (arrays["counts"] > completed).any()):
            raise ValueError("Invalid network minima/coverage")
        covered, two = arrays["counts"] > 0, arrays["counts"] > 1
        if (not np.array_equal(np.isfinite(best), np.broadcast_to(covered, best.shape)) or
                not np.array_equal(np.isfinite(second), np.broadcast_to(two, second.shape)) or
                not np.issubdtype(arrays["counts"].dtype, np.integer)):
            raise ValueError("Network coverage differs from minima")
        for name, support in (("winner", covered), ("runner", two)):
            ids = arrays[name]
            if (not np.issubdtype(ids.dtype, np.integer) or
                    (ids[:, support] < 0).any() or (ids[:, support] >= completed).any() or
                    (ids[:, ~support] != -1).any()):
                raise ValueError("Invalid best/second station IDs")
        if (arrays["winner"][:, two] == arrays["runner"][:, two]).any():
            raise ValueError("Best and second station must be distinct")

    def load(self):
        with self.lock:
            if self.arrays is not None:
                return self.arrays
            if self.status()["state"] != "ready":
                raise ValueError("Frost-nettkartet er ikke ferdig; se fremdrift")
            receipt = json.loads((self.folder / "complete.json").read_text())
            for name, digest in receipt["checksums"].items():
                if Path(name).name != name or sha256(self.folder / name) != digest:
                    raise ValueError(f"Network cache checksum mismatch: {name}")
            with np.load(self.folder / "best_two.npz", allow_pickle=False) as data:
                arrays = dict(best=data["baseline"].copy(), second=data["second"].copy(),
                              winner=data["winner"][0].copy(), runner=data["winner"][1].copy(),
                              counts=data["counts"].copy())
                completed = int(data["completed"])
            self.validate(arrays, completed)
            self.arrays = arrays
            return arrays

    def station_index(self, identifier):
        index = next((i for i, record in enumerate(self.stations) if record["station_id"] == identifier), None)
        if index is None:
            raise ValueError("Velg en Frost-stasjon med modellstøtte")
        return index

    def field(self, target, mode="network", station=None):
        if target not in TARGETS or mode not in ("network", "reduction", "lost"):
            raise ValueError("Ugyldig mål eller nettkart")
        a = self.load()
        t = TARGETS.index(target)
        covered = a["counts"] > 0
        if mode == "network":
            return a["best"][t].copy(), covered, np.zeros(len(covered), bool)
        index = self.station_index(station)
        won = a["winner"][t] == index
        lost = won & (a["counts"] == 1)
        valid = covered & ~lost
        values = np.zeros(len(covered), np.float32)
        comparable = won & (a["counts"] > 1)
        values[comparable] = a["second"][t, comparable]-a["best"][t, comparable]
        if mode == "lost":
            return lost.astype(np.float32), covered, lost
        return values, valid, lost

    def describe(self, target, mode="network", station=None):
        from pyproj import Transformer
        from rasterio.warp import transform_bounds
        values, valid, lost = self.field(target, mode, station)
        left, top = self.origin[0]-500, self.origin[1]+500
        bottom, right = top-self.shape[0]*1000, left+self.shape[1]*1000
        web = transform_bounds(25833, 3857, left, bottom, right, top, densify_pts=21)
        to_geo = Transformer.from_crs(3857, 4326, always_xy=True)
        west, south = to_geo.transform(web[0], web[1])
        east, north = to_geo.transform(web[2], web[3])
        covered = self.load()["counts"] > 0
        return dict(mode=mode, station=station, target=target, label=TARGET_LABELS[target], cell_m=1000,
                    shape=list(self.shape), bounds=[[south, west], [north, east]], web_bounds=list(web),
                    projected_bounds=[left, bottom, right, top], valid_cells=int(valid.sum()),
                    covered_cells=int(covered.sum()), uncovered_cells=int((~covered).sum()),
                    affected_area_km2=int((valid & (values > 0)).sum()) if mode == "reduction" else None,
                    lost_coverage_area_km2=int(lost.sum()),
                    mean_c=float(values[valid].mean()) if valid.any() else None,
                    mean_definition="Mean over numerically supported1km land cells; excludes loss-of-coverage cells.",
                    stations=len(self.stations), excluded_stations=self.excluded,
                    warning=NETWORK_WARNING, selected=self.backend.metadata["selected"]["name"],
                    model_sha256=self.backend.metadata["checksums"]["selected_model.pt"],
                    dataset_sha256=self.backend.metadata["manifest_sha256"],
                    test=self.backend.metadata["test"][target], baseline_test=self.backend.metadata["baseline_test"][target],
                    interpretation="Best-single-Frost-station representation RMSE; removal contribution is without-minus-with RMSE.",
                    station_catalogue=[dict(index=i, station_id=s["station_id"], name=s["name"],
                                            reference=s["reference"]) for i, s in enumerate(self.stations)])

    def raster(self, target, mode="network", station=None):
        from rasterio.transform import from_origin
        values, valid, _ = self.field(target, mode, station)
        raster = np.full(self.shape, -9999, np.float32)
        raster[self.rows[valid], self.cols[valid]] = values[valid]
        return raster, from_origin(self.origin[0]-500, self.origin[1]+500, 1000, 1000)

    def png(self, target, mode="network", station=None):
        raster, transform = self.raster(target, mode, station)
        return raster_png(raster, transform, self.describe(target, mode, station),
                          maximum=1 if mode == "lost" else 3)

    def export(self, target, mode="network", station=None):
        raster, transform = self.raster(target, mode, station)
        rasters = {f"frost_{mode}_{target}_1000m.tif": raster}
        if mode == "reduction":
            rasters["lost_coverage_1000m.tif"] = self.raster(target, "lost", station)[0]
        a = self.load()
        covered = a["counts"] > 0
        for name, values in (("station_coverage_count_1000m.tif", a["counts"]),
                             ("best_station_index_1000m.tif", a["winner"][TARGETS.index(target)])):
            extra = np.full(self.shape, -9999, np.float32)
            extra[self.rows[covered], self.cols[covered]] = values[covered]
            rasters[name] = extra
        return raster_export(rasters, transform, self.describe(target, mode, station))

    def point(self, target, x, y, selected=None):
        if target not in TARGETS or not np.isfinite([x, y]).all():
            raise ValueError("Ugyldig mål eller kartpunkt")
        if selected is not None:
            self.station_index(selected)
        key = tuple(BasinLookup.cell_keys([x], [y])[0])
        index = self.backend.keys.get(key)
        if index is None or self.load()["counts"][index] == 0:
            return dict(supported=False, value_c=None, ranking=[], cell_m=1000,
                        reason="Ingen Frost-footprint dekker denne1 km-landruten")
        ranked = []
        t = TARGETS.index(target)
        for record in self.stations:
            cx, cy = self.backend.atlas.table[record["reference"], :2]/1000
            row, col = int(cy-key[1])+32, int(key[0]-cx)+32
            if 0 <= row < 64 and 0 <= col < 64:
                values, mapping = self.saved_map(record["reference"])
                if mapping[row, col] == index:
                    ranked.append(dict(station_id=record["station_id"], name=record["name"],
                                       value_c=float(values[t, row, col]), reference=record["reference"]))
        ranked.sort(key=lambda item: (item["value_c"], item["station_id"]))
        if not ranked:
            raise ValueError("Network coverage differs from saved footprints")
        best = ranked[0]["value_c"]
        for item in ranked:
            other = [candidate["value_c"] for candidate in ranked if candidate["station_id"] != item["station_id"]]
            item["removal_reduction_c"] = float(min(other)-best) if other else None
            item["loses_coverage"] = not other
        return dict(supported=True, value_c=best, best_station=ranked[0]["station_id"],
                    coverage_count=len(ranked), ranking=ranked, cell_m=1000,
                    x=int(key[0]*1000), y=int(key[1]*1000),
                    selected_station=next((item for item in ranked if item["station_id"] == selected), None))


def main():
    # Numerical/GIS import order matches the existing application environment.
    import train_senorge_grid_pairs
    from station_footprint import FootprintBackend, frost_catalogue
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    backend = FootprintBackend(args.root)
    try:
        network = FootprintNetwork(backend, frost_catalogue(args.root, backend))
        network.prepare()
        print(json.dumps(network.status()), flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
