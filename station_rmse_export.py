"""Native RMSE grid polygons, with an independently filtered display layer."""
from io import BytesIO
from pathlib import Path
import tempfile

import geopandas as gpd
import numpy as np
import pyogrio
from shapely import box


def export_rmse(engine, values, threshold=0, cell_mask=None, clip=None, metadata=None):
    values = np.asarray(values)
    if values.shape != (len(engine.grid),):
        raise ValueError("RMSE surface does not match the grid")
    finite = np.isfinite(values)
    indices = np.flatnonzero(finite)
    x = engine.grid["x"].to_numpy()[indices]
    y = engine.grid["y"].to_numpy()[indices]
    half = engine.cell_m/2
    full = gpd.GeoDataFrame(dict(cell_id=indices, rmse=values[indices]),
                           geometry=box(x-half, y-half, x+half, y+half), crs=25833)
    selected = finite.copy()
    if threshold > 0:
        selected &= values > threshold
    if cell_mask is not None:
        selected &= cell_mask
    visible = full.loc[selected[indices]].copy()
    if clip is not None:
        if clip.empty:
            visible = visible.iloc[:0].copy()
        elif not visible.empty:
            visible = gpd.clip(visible, clip.to_crs(full.crs), keep_geom_type=True)
            visible = visible.loc[~visible.geometry.is_empty & (visible.geometry.area > 0)].copy()
    info = {str(key): str(value) for key, value in (metadata or {}).items()}
    info.update(rmse_min=str(threshold), cell_size_m=str(engine.cell_m))
    with tempfile.TemporaryDirectory(prefix="station_rmse_") as folder:
        path = Path(folder) / "norway_rmse.gpkg"
        for name, frame in (("rmse_norway", full), ("rmse_visible", visible)):
            pyogrio.write_dataframe(frame, path, layer=name, driver="GPKG",
                                    geometry_type="Unknown", layer_metadata=info)
        return BytesIO(path.read_bytes())
