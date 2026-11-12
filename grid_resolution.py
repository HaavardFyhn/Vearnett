"""Canonical placement grid and model policy for supported app resolutions."""
from pathlib import Path


def source_spacing(cell_m):
    if cell_m not in (500, 1000, 2500, 5000):
        raise ValueError("Supported grid spacing: 500, 1000, 2500 or 5000 m")
    return min(cell_m, 1000)


def cell_policy(cell_m):
    if cell_m not in (500, 1000):
        raise ValueError("Cell-based station inputs require a 500 or 1000 m grid")
    return f"cell-{cell_m}m-v1"


def training_spacing(path):
    for spacing in (500, 1000):
        if Path(path).name == f"norway_grid_{spacing}m.gpkg":
            return spacing
    raise ValueError("Cell-based station inputs require norway_grid_500m.gpkg or norway_grid_1000m.gpkg")


def weather_folder(root, spacing):
    cell_policy(spacing)
    return Path(root) / "outputs" / ("weather_1km" if spacing == 1000 else "weather_500m")
