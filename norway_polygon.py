"""Create a reusable Norway land polygon before Frost downloading.

The polygon is the union of all N100_Arealdekke_omrade features except
objtype='Havflate'. Existing output is skipped unless force=True.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import geopandas as gpd

CRS = "EPSG:25833"
ROOT = Path(__file__).resolve().parent
DEFAULT_N100_GDB = (
    ROOT / "data" /
    "Basisdata_0000_Norge_25833_N100Kartdata_FGDB.gdb"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "norway_polygon.gpkg"
AREA_LAYER_CANDIDATES = (
    "N100_Arealdekke_omrade",
    "N_100_Arealdekke",
    "N100_Arealdekke",
)


def _layers(path: Path) -> list[str]:
    try:
        import pyogrio
        return pyogrio.list_layers(path)[:, 0].tolist()
    except Exception:
        import fiona
        return list(fiona.listlayers(path))


def _resolve_layer(path: Path, requested: str | None) -> str:
    available = _layers(path)
    if requested:
        if requested not in available:
            raise ValueError(
                f"Layer {requested!r} not found in {path}. Available: {available}"
            )
        return requested

    folded = {name.casefold(): name for name in available}
    for candidate in AREA_LAYER_CANDIDATES:
        if candidate.casefold() in folded:
            return folded[candidate.casefold()]

    raise ValueError(
        f"No N100 area layer found in {path}. Available: {available}"
    )


def build(
    output: Path = DEFAULT_OUTPUT,
    force: bool = False,
    n100_gdb: Path = DEFAULT_N100_GDB,
    area_layer: str | None = None,
) -> Path:
    output = Path(output)
    n100_gdb = Path(n100_gdb)

    if output.exists() and not force:
        print(f"SKIP Norway polygon; exists: {output}")
        return output

    if not n100_gdb.exists():
        raise FileNotFoundError(f"Missing N100 geodatabase: {n100_gdb}")

    layer = _resolve_layer(n100_gdb, area_layer)
    print(f"Reading Norway areas from {layer}...")
    areas = gpd.read_file(n100_gdb, layer=layer)

    if areas.crs is None:
        raise ValueError(f"Layer {layer} has no CRS")
    if areas.crs.to_epsg() != 25833:
        areas = areas.to_crs(CRS)

    obj_field = next(
        (column for column in areas.columns if str(column).casefold() == "objtype"),
        None,
    )
    if obj_field is None:
        raise ValueError(
            f"objtype not found in {layer}. Available: {list(areas.columns)}"
        )

    valid = areas.geometry.notna() & ~areas.geometry.is_empty
    is_ocean = (
        areas[obj_field].astype(str).str.strip().str.casefold().eq("havflate")
    )
    land = areas.loc[valid & ~is_ocean, ["geometry"]].copy()
    if land.empty:
        raise ValueError("No non-Havflate polygons were found")

    print(f"Unioning {len(land):,} non-ocean polygons...")
    geometry = land.geometry.union_all()
    if geometry is None or geometry.is_empty:
        raise ValueError("Norway polygon union produced an empty geometry")

    result = gpd.GeoDataFrame(
        {"name": ["Norway"]},
        geometry=[geometry],
        crs=CRS,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    result.to_file(output, layer="norway_polygon", driver="GPKG")

    print(f"Saved Norway polygon: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable Norway land polygon")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n100-gdb", type=Path, default=DEFAULT_N100_GDB)
    parser.add_argument("--area-layer")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(
        output=args.output,
        force=args.force,
        n100_gdb=args.n100_gdb,
        area_layer=args.area_layer,
    )


if __name__ == "__main__":
    main()
