from scipy.spatial import cKDTree
import numpy as np
import geopandas as gpd
from pathlib import Path    
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "outputs"

RADIUS_KM =  100
WORK.mkdir(exist_ok=True)
GRID_FILE = WORK / "norway_grid_terrain.gpkg"
grid = gpd.read_file(
    GRID_FILE
)

grid["x"] = grid.geometry.x
grid["y"] = grid.geometry.y

xy = grid[["x", "y"]].to_numpy()

tree = cKDTree(xy)

sample = grid.sample(
    50,
    random_state=42
)

for i, row in sample.iterrows():

    idx = tree.query_ball_point(
        [row["x"], row["y"]],
        r=RADIUS_KM * 1000
    )

    print(
        f"{len(idx):5d} targets "
        f"within {RADIUS_KM} km"
    )



candidates = grid.sample(
    500,
    random_state=42
)

start = time.time()

total = 0

for _, row in candidates.iterrows():

    idx = tree.query_ball_point(
        [row["x"], row["y"]],
        r=RADIUS_KM * 1000
    )

    total += len(idx)

elapsed = time.time() - start

print(
    f"Average targets:"
)

print(
    total / len(candidates)
)

print(
    f"Runtime:"
)

print(
    elapsed
)
