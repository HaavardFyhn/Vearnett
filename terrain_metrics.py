import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from pipeline_config import BASE_GRID_FILE, GRID_FILE

RADII_M = (10_000, 25_000)

def run(force=False):
    if GRID_FILE.exists() and not force:
        print(f"SKIP terrain: {GRID_FILE.name} exists")
        return GRID_FILE
    grid = gpd.read_file(BASE_GRID_FILE)
    if grid.crs is None or grid.crs.to_epsg() != 25833:
        grid = grid.to_crs(25833)
    grid = grid.dropna(subset=["elevation"]).copy()
    grid["x"], grid["y"] = grid.geometry.x, grid.geometry.y
    xy = grid[["x", "y"]].to_numpy(float)
    z = grid["elevation"].to_numpy(float)
    tree = cKDTree(xy)
    for radius in RADII_M:
        suffix = f"{radius // 1000}km"
        groups = tree.query_ball_point(xy, r=radius)
        relief=np.full(len(grid),np.nan); rugged=np.full(len(grid),np.nan); tpi=np.full(len(grid),np.nan)
        for i, ids in enumerate(groups):
            v=z[np.asarray(ids,dtype=int)]; v=v[np.isfinite(v)]
            if len(v)>=3:
                relief[i]=v.max()-v.min(); rugged[i]=v.std(); tpi[i]=z[i]-v.mean()
        grid[f"relief_{suffix}"]=relief
        grid[f"rugstd_{suffix}"]=rugged
        grid[f"tpi_{suffix}"]=tpi
    GRID_FILE.parent.mkdir(parents=True,exist_ok=True)
    grid.to_file(GRID_FILE,layer="terrain_grid",driver="GPKG")
    print(f"DONE terrain: {GRID_FILE}")
    return GRID_FILE

if __name__ == "__main__": run()
