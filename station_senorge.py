"""CPU inference adapter for the independently trained seNorge grid-pair model."""
from pathlib import Path
import os
import threading

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from greedy_400_stations_pruned import FEATURES, TERRAIN


BASIN_FIELDS = ["basin_present", "basin_count", "basin_max_depth_m", "basin_volume_m3"]
PAIR_FEATURES = FEATURES + [f"{field}_{stat}" for field in BASIN_FIELDS for stat in ("min", "max", "diff")]
WARNING = (
    "Eksperimentell seNorge-modell 1996–2025, ikke validert mot uavhengige stasjoner. "
    "Anslaget gjelder historisk temperatur-likhet i interpolerte 1 km-data, ikke en temperaturprognose. "
    "Trent på 1–100 km; avstander over 100 km ekstrapoleres uten dokumentert nøyaktighet. "
    "Ingen romlige testpar ved 90–100 km. Bassenger knyttes til nærmeste 1 km-rutesenter, "
    "og samme 1 km-rute får feil 0 som en eksplisitt identitetsregel."
)


def station_cell_features(stations, grid):
    """Use the same source cells as manually placed grid-pair stations."""
    fields = ["x", "y", "elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype", *TERRAIN]
    if grid.empty or not np.isfinite(grid[["x", "y"]].to_numpy()).all():
        raise ValueError("Ugyldig rutenett for stasjonsegenskaper")
    if not np.isfinite(stations[["x", "y"]].to_numpy()).all():
        raise ValueError("Ugyldige stasjonskoordinater")
    _, indices = cKDTree(grid[["x", "y"]].to_numpy()).query(stations[["x", "y"]].to_numpy())
    cells = grid.iloc[indices]
    if cells[fields].isna().any().any():
        raise ValueError("Manglende terrengverdier i stasjonenes terrengruter")
    result = stations.copy()
    for field in fields:
        result[field] = cells[field].to_numpy()
    return result


class BasinLookup:
    def __init__(self, grid_path, basin_path):
        frame = pd.read_csv(grid_path, float_precision="round_trip")
        keys = self.cell_keys(frame.x.to_numpy(), frame.y.to_numpy())
        self.values = pd.DataFrame(frame[BASIN_FIELDS].to_numpy(dtype=np.float64),
                                   index=pd.MultiIndex.from_arrays(keys.T))
        if not self.values.index.is_unique or not np.isfinite(self.values.to_numpy()).all():
            raise ValueError("Ugyldig bassengtabell for seNorge-modellen")
        self.basin_path = Path(basin_path)
        self.basins = None
        self.lock = threading.RLock()

    @staticmethod
    def cell_keys(x, y):
        return np.floor(np.column_stack([x, y])/1000 + .5).astype(np.int64)

    def lookup(self, x, y):
        keys = self.cell_keys(x, y)
        with self.lock:
            index = pd.MultiIndex.from_arrays(keys.T)
            result = self.values.reindex(index)
            missing = result.isna().any(axis=1).to_numpy()
            if missing.any():
                # Cells excluded from weather training still need real basin geometry, not zero defaults.
                import geopandas as gpd
                from shapely import box
                if self.basins is None:
                    self.basins = gpd.read_file(self.basin_path, columns=[
                        "lake_id", "max_depth_m", "volume_m3", "geometry"]).to_crs(25833)
                unique = np.unique(keys[missing], axis=0)
                centers = unique*1000
                cells = box(centers[:, 0]-500, centers[:, 1]-500,
                            centers[:, 0]+500, centers[:, 1]+500)
                left, right = self.basins.sindex.query(cells, predicate="intersects")
                areas = self.basins.geometry.iloc[right].reset_index(drop=True).intersection(
                    gpd.GeoSeries(cells[left], crs=25833)).area.to_numpy()
                left, right = left[areas > 0], right[areas > 0]
                values = np.zeros((len(unique), 4), dtype=np.float64)
                if len(left):
                    hits = pd.DataFrame(dict(cell=left,
                                             depth=self.basins.max_depth_m.to_numpy()[right],
                                             volume=self.basins.volume_m3.to_numpy()[right]))
                    grouped = hits.groupby("cell").agg(count=("depth", "size"), depth=("depth", "max"),
                                                       volume=("volume", "sum"))
                    values[grouped.index, 0] = 1
                    values[grouped.index, 1:] = grouped.to_numpy()
                added = pd.DataFrame(values, index=pd.MultiIndex.from_arrays(unique.T))
                self.values = pd.concat([self.values, added])
                result = self.values.reindex(index)
            values = result.to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Ugyldige bassengegenskaper ved prediksjon")
            return values


class SeNorgePairModel:
    n_features_in_ = len(PAIR_FEATURES)

    def __init__(self, model_path, basins):
        import xgboost
        self.model = xgboost.XGBRegressor()
        self.model.load_model(model_path)
        self.model.set_params(device="cpu", n_jobs=int(os.environ.get("WEATHER_CPU_THREADS", "4")))
        if self.model.n_features_in_ != self.n_features_in_:
            raise ValueError("seNorge-modellen har feil antall egenskaper")
        self.basins = basins

    def pair_inputs(self, source, target, source_ids, target_ids):
        source = {key: source.get(f"senorge_{key}", source[key]) for key in
                  ("x", "y", "elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype_code", *TERRAIN)}
        target = {key: target.get(f"senorge_{key}", target[key]) for key in source}
        sx, sy = source["x"][source_ids].astype(float), source["y"][source_ids].astype(float)
        tx, ty = target["x"][target_ids].astype(float), target["y"][target_ids].astype(float)
        columns = [
            np.hypot(sx-tx, sy-ty),
            np.abs(source["elevation"][source_ids].astype(float)-target["elevation"][target_ids].astype(float)),
            np.abs(source["coast_dist_fjord"][source_ids].astype(float)-target["coast_dist_fjord"][target_ids].astype(float)),
            np.abs(source["coast_dist_ocean"][source_ids].astype(float)-target["coast_dist_ocean"][target_ids].astype(float)),
            (source["objtype_code"][source_ids] == target["objtype_code"][target_ids]).astype(np.float32),
            np.abs(sx-tx), np.abs(sy-ty),
        ]
        columns.extend(np.abs(source[key][source_ids].astype(float)-target[key][target_ids].astype(float)) for key in TERRAIN)
        a, b = self.basins.lookup(sx, sy), self.basins.lookup(tx, ty)
        for i in range(len(BASIN_FIELDS)):
            columns.extend([np.minimum(a[:, i], b[:, i]), np.maximum(a[:, i], b[:, i]),
                            np.abs(a[:, i]-b[:, i])])
        x = np.column_stack(columns).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Ikke-endelige seNorge-paregenskaper")
        identical = (self.basins.cell_keys(sx, sy) == self.basins.cell_keys(tx, ty)).all(axis=1)
        return x, identical, (sx, sy, tx, ty)

    def predict_pairs(self, source, target, source_ids, target_ids):
        x, identical, _ = self.pair_inputs(source, target, source_ids, target_ids)
        result = np.maximum(self.model.predict(x), 0).astype(np.float32)
        result[identical] = 0
        if not np.isfinite(result).all():
            raise ValueError("seNorge-modellen ga ikke-endelige prediksjoner")
        return result
