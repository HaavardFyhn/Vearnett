"""Reproduce Dekningsgrad's 250 m climate-occurrence scoring on our model stations."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import tempfile

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import rowcol
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation, label
import shapely
from shapely.geometry import shape

from file_hash import sha256
from station_owners import load_catalog
from station_roads import RoadNetwork
from station_weather import PHENOMENON_DATASETS


ROOT = Path(__file__).resolve().parent
ALLOWED_OWNERS = frozenset(("STATENS VEGVESEN", "MET.NO"))
FIELDS = ["ForekomstID", "Klimasone", "LokaltForekomstnummer", "Klimaceller250m",
          "Vegceller250m", "Overgangsvegceller250m", "AndelVegIOvergang"]
STATUS = ["Godt representert", "Relativt god dekning", "Mulig dekningsmangel",
          "Interessant kandidatområde", "Svært interessant kandidatområde"]

SCORE_THRESHOLDS = (25, 45, 65, 85)


def klassifiser_score(score):
    if score <= SCORE_THRESHOLDS[0]:
        return 1, STATUS[0]
    if score <= SCORE_THRESHOLDS[1]:
        return 2, STATUS[1]
    if score <= SCORE_THRESHOLDS[2]:
        return 3, STATUS[2]
    if score <= SCORE_THRESHOLDS[3]:
        return 4, STATUS[3]
    return 5, STATUS[4]


def validate_classification():
    if (len(SCORE_THRESHOLDS) != 4 or
            not all(np.isfinite(value) and 0 <= value < 100 for value in SCORE_THRESHOLDS) or
            any(left >= right for left, right in zip(SCORE_THRESHOLDS, SCORE_THRESHOLDS[1:]))):
        raise ValueError("Score thresholds must be four strictly increasing finite values from 0 to below 100")
    if len(STATUS) != 5 or any(not isinstance(value, str) or not value.strip() for value in STATUS):
        raise ValueError("Classification requires five non-empty status labels")


def climate_occurrences(climate, roads, cell_m=250):
    """Same 8-neighbour boundaries, square dilation and component IDs as the notebook."""
    valid = (climate >= 1) & (climate <= 8)
    road_mask = (roads > 0) & valid
    boundary = np.zeros(climate.shape, dtype=bool)
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        shifted = np.roll(climate, (dr, dc), (0, 1))
        neighbour_valid = np.roll(valid, (dr, dc), (0, 1))
        different = valid & neighbour_valid & (climate != shifted)
        if dr == -1:
            different[-1, :] = False
        if dr == 1:
            different[0, :] = False
        if dc == -1:
            different[:, -1] = False
        if dc == 1:
            different[:, 0] = False
        boundary |= different
    transition = binary_dilation(boundary, structure=np.ones((3, 3), dtype=bool),
                                 iterations=max(1, int(np.ceil(500 / cell_m)))) & valid
    occurrences = np.zeros(climate.shape, dtype=np.int32)
    records = []
    offset = 0
    for zone in range(1, 9):
        components, count = label(climate == zone, structure=np.ones((3, 3), dtype=np.uint8))
        cell_counts = np.bincount(components.ravel(), minlength=count + 1)
        road_counts = np.bincount(components[road_mask], minlength=count + 1)
        transition_counts = np.bincount(components[road_mask & transition], minlength=count + 1)
        kept = np.flatnonzero(road_counts[1:]) + 1
        lookup = np.zeros(count + 1, dtype=np.int32)
        lookup[kept] = offset + np.arange(1, len(kept) + 1)
        # Bincount/lookup avoids rescanning the national raster for every component.
        occurrences += lookup[components]
        for component in kept:
            records.append([int(lookup[component]), zone, int(component), int(cell_counts[component]),
                            int(road_counts[component]), int(transition_counts[component]),
                            float(transition_counts[component] / road_counts[component])])
        offset += len(kept)
        print(f"Zone {zone}: {count:,} raw occurrences, {len(kept):,} with roads", flush=True)
    return occurrences, transition, road_mask, pd.DataFrame(records, columns=FIELDS)


def model_stations(data_root, owner_catalog):
    manifest_path = data_root / "outputs" / "portable_weather.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stations = {}
    for dataset in PHENOMENON_DATASETS.values():
        if dataset not in manifest["profiles"]:
            raise ValueError(f"Missing station profile: {dataset}")
        profile_path = data_root / manifest["profiles"][dataset]["profile"]
        profile = joblib.load(profile_path)
        for row in profile["stations"].to_dict("records"):
            identifier = str(row["station_id"])
            owners = owner_catalog["stations"].get(identifier, {}).get("owners", [])
            if not ALLOWED_OWNERS.intersection(owners):
                continue
            longitude, latitude = float(row["lon"]), float(row["lat"])
            if not np.isfinite([longitude, latitude]).all() or not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError(f"Invalid original position for {identifier}")
            if identifier in stations:
                previous = stations[identifier]
                if not np.allclose([previous["lon"], previous["lat"]], [longitude, latitude], rtol=0, atol=1e-8):
                    raise ValueError(f"Conflicting original model positions for {identifier}")
                previous["datasets"].append(dataset)
            else:
                stations[identifier] = dict(station_id=identifier, name=row["name"], lon=longitude,
                    lat=latitude, owners="; ".join(owners), datasets=[dataset])
        del profile
        gc.collect()
    if not stations:
        raise ValueError("No model stations match STATENS VEGVESEN or MET.NO")
    frame = pd.DataFrame(stations.values()).sort_values("station_id").reset_index(drop=True)
    frame["datasets"] = frame["datasets"].map("; ".join)
    return frame, manifest


def station_occurrences(stations, climate, occurrences, transition, transform, crs):
    stations = stations.copy()
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(
        stations.lon.to_numpy(), stations.lat.to_numpy())
    rows, cols = (np.asarray(values) for values in rowcol(transform, x, y))
    inside = (rows >= 0) & (rows < climate.shape[0]) & (cols >= 0) & (cols < climate.shape[1])
    ids = np.flatnonzero(inside)
    inside[ids] &= (climate[rows[ids], cols[ids]] >= 1) & (climate[rows[ids], cols[ids]] <= 8)
    ids = np.flatnonzero(inside)
    stations["plot_x"], stations["plot_y"] = x, y
    stations["InnenforKart"] = inside
    stations["Klimasone"], stations["ForekomstID"], stations["Overgangssone"] = 0, 0, False
    stations.loc[ids, "Klimasone"] = climate[rows[ids], cols[ids]]
    stations.loc[ids, "ForekomstID"] = occurrences[rows[ids], cols[ids]]
    stations.loc[ids, "Overgangssone"] = transition[rows[ids], cols[ids]]
    stations["KobletTilForekomst"] = stations.InnenforKart & (stations.ForekomstID > 0)
    return stations


def score_occurrences(occurrences, stations):
    validate_classification()
    frame = occurrences.copy()
    linked = stations.loc[stations.KobletTilForekomst]
    counts = linked.groupby("ForekomstID").size()
    transition_counts = linked.loc[linked.Overgangssone].groupby("ForekomstID").size()
    frame["AntallStasjoner"] = frame.ForekomstID.map(counts).fillna(0).astype(int)
    frame["AntallOvergangsstasjoner"] = frame.ForekomstID.map(transition_counts).fillna(0).astype(int)
    frame["Relevant"] = frame.Vegceller250m >= 20
    relevant = frame.loc[frame.Relevant].copy()
    if relevant.empty:
        raise ValueError("No occurrences have at least 20 road cells")
    density = relevant.AntallStasjoner.sum() / relevant.Vegceller250m.sum()
    relevant["ForventetAntallStasjoner"] = relevant.Vegceller250m * density
    expected = relevant.ForventetAntallStasjoner.to_numpy()
    coverage = np.ones(len(relevant), dtype=float)
    np.divide(relevant.AntallStasjoner.to_numpy(), expected, out=coverage, where=expected > 0)
    relevant["Dekningsgrad"] = np.clip(coverage, 0, 1)
    relevant["DekningsmangelPoeng"] = 50 * (1 - relevant.Dekningsgrad)
    relevant["OvergangsmangelPoeng"] = (30 * relevant.AndelVegIOvergang.clip(0, 1) *
                                         (relevant.AntallOvergangsstasjoner == 0).astype(float))
    percentile = (np.ones(len(relevant)) if len(relevant) <= 1 else
                  relevant.Vegceller250m.astype(float).rank(method="average", pct=True).to_numpy())
    relevant["VegnettBetydningPoeng"] = 20 * percentile
    relevant["Kandidatscore"] = (relevant.DekningsmangelPoeng + relevant.OvergangsmangelPoeng +
                               relevant.VegnettBetydningPoeng).clip(0, 100).round(1)
    classified = relevant.Kandidatscore.map(klassifiser_score)
    relevant["Kandidatklasse"] = classified.map(lambda value: value[0])
    relevant["Status"] = classified.map(lambda value: value[1])
    relevant["UtenStasjon"] = relevant.AntallStasjoner == 0
    return frame, relevant


def occurrence_polygons(occurrences, road_mask, relevant, transform, crs):
    lookup = np.zeros(int(occurrences.max()) + 1, dtype=bool)
    lookup[relevant.ForekomstID.to_numpy()] = True
    mask = road_mask & lookup[occurrences]
    parts = defaultdict(list)
    for geometry, identifier in shapes(occurrences, mask=mask, transform=transform, connectivity=4):
        parts[int(identifier)].append(shape(geometry))
    records = [dict(ForekomstID=identifier, geometry=shapely.union_all(geometries))
               for identifier, geometries in parts.items()]
    return gpd.GeoDataFrame(records, crs=crs).merge(relevant, on="ForekomstID", validate="one_to_one")


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 0 else []
    if geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        return [line for part in geometry.geoms for line in line_parts(part)]
    return []


def export_lines(network, polygons, output):
    polygons = polygons.to_crs(25833).reset_index(drop=True)
    index = polygons.sindex
    settings = dict(categories=["E", "R", "F"], classes=network.metadata["classes"])
    rows = network.matching_rows(network.geographic_bounds, settings)
    if "bounds" in network.attributes and len(rows):
        bounds = network.attributes["bounds"][:, rows]
        rows = rows[np.lexsort(((bounds[0] + bounds[2]) / 2,
            np.floor((bounds[0] + bounds[2]) / 100000),
            np.floor((bounds[1] + bounds[3]) / 100000)))]
    fields = [name for name in polygons.columns if name != "geometry"]
    total = 0
    missing = 0
    for start in range(0, len(rows), 5000):
        ids = network.attributes["fid"][rows[start:start + 5000]].tolist()
        roads = network.read_selected(ids, network.geographic_bounds, settings)
        if roads.empty:
            continue
        west, south, east, north = roads.total_bounds
        extent = shapely.box(west - 1, south - 1, east + 1, north + 1)
        local = polygons.iloc[index.query(extent, predicate="intersects")].copy().reset_index(drop=True)
        # Clip each occurrence once per spatial batch, not its national geometry for every road.
        local.geometry = shapely.intersection(local.geometry.to_numpy(), extent)
        pairs = local.sindex.query(roads.geometry, predicate="intersects")
        records = []
        for road_index, polygon_index in pairs.T:
            road = roads.iloc[road_index]
            occurrence = local.iloc[polygon_index]
            intersection = shapely.intersection(road.geometry, occurrence.geometry)
            for part_number, geometry in enumerate(line_parts(intersection)):
                record = {field: occurrence[field] for field in fields}
                record.update(source_fid=int(road.name), edge_id=str(road.get("edge_id", "")),
                    roadid=str(road.get("roadid", "")), municipality=str(road.get("municipality", "")),
                    part=part_number, length_m=float(geometry.length), geometry=geometry)
                records.append(record)
        if records:
            frame = gpd.GeoDataFrame(records, crs=25833)
            pyogrio.write_dataframe(frame, output, layer="scored_roads", driver="GPKG", append=total > 0,
                                    geometry_type="LineString")
            uncovered = frame.loc[frame.UtenStasjon]
            if not uncovered.empty:
                pyogrio.write_dataframe(uncovered, output, layer="roads_without_station", driver="GPKG",
                                        append=missing > 0, geometry_type="LineString")
            total += len(frame)
            missing += len(uncovered)
        if start % 50000 == 0:
            print(f"Roads processed: {min(start + 5000, len(rows)):,}/{len(rows):,}; "
                  f"LineStrings: {total:,}; without station: {missing:,}", flush=True)
    if not total:
        raise ValueError("No road LineStrings intersect the relevant raster cells")
    if not missing:
        empty = frame.iloc[:0].copy()
        pyogrio.write_dataframe(empty, output, layer="roads_without_station", driver="GPKG",
                                geometry_type="LineString")
    return dict(scored_lines=total, lines_without_station=missing)


def run(args):
    validate_classification()
    inputs = args.inputs.resolve()
    data_root = args.data_root.resolve()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory; preserving existing results: {destination}")
    owners = load_catalog(ROOT, data_root)
    if owners is None:
        raise ValueError("Owner metadata missing; run python station_owners.py")
    climate_path = inputs / "klimasoner_v1_0.tif"
    road_path = inputs / "vegnett_ERF_250m.tif"
    stations, manifest = model_stations(data_root, owners)
    print(f"Unique model stations with permitted holders: {len(stations):,}", flush=True)
    with rasterio.open(road_path) as source:
        road_values = source.read(1)
        profile = source.profile.copy()
        if source.crs is None or not source.crs.is_projected or source.res != (250, 250):
            raise ValueError("The reference method requires the original projected 250 m road raster")
    climate = np.zeros(road_values.shape, dtype=np.uint8)
    with rasterio.open(climate_path) as source:
        reproject(source.read(1), climate, src_transform=source.transform, src_crs=source.crs,
            src_nodata=source.nodata if source.nodata is not None else 0,
            dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=0,
            resampling=Resampling.nearest)
    occurrences, transition, road_mask, information = climate_occurrences(climate, road_values)
    stations = station_occurrences(stations, climate, occurrences, transition, profile["transform"], profile["crs"])
    information, relevant = score_occurrences(information, stations)
    print(f"Road cells: {road_mask.sum():,}; transition road cells: {(transition & road_mask).sum():,}; "
          f"occurrences: {len(information):,}; relevant: {len(relevant):,}; "
          f"without station: {relevant.UtenStasjon.sum():,}", flush=True)
    polygons = occurrence_polygons(occurrences, road_mask, relevant, profile["transform"], profile["crs"])
    road_source = data_root / "data" / "Vegnett_VegScore_full.gpkg"
    if not road_source.is_file():
        road_source = data_root / "data" / "ProssExport.gdb"
    network = RoadNetwork(road_source, portable_cache=manifest["cell_m"] == 500)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix="coverage_") as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        result = export_lines(network, polygons, staged / "road_climate_coverage.gpkg")
        stations.to_csv(staged / "stations.csv", index=False, encoding="utf-8-sig")
        information.to_csv(staged / "all_occurrences.csv", index=False)
        relevant.sort_values("Kandidatscore", ascending=False).to_csv(staged / "candidates.csv", index=False)
        from station_coverage import save_status_raster
        save_status_raster(staged / "status.tif", occurrences, road_mask, relevant, profile)
        for name, array in (("occurrences", occurrences), ("transition", transition.astype("uint8"))):
            raster_profile = dict(profile, driver="GTiff", count=1, dtype=str(array.dtype), nodata=0, compress="lzw")
            with rasterio.open(staged / f"{name}.tif", "w", **raster_profile) as output:
                output.write(array, 1)
        provenance = dict(schema=1, created=datetime.now(timezone.utc).isoformat(),
            method="Dekningsgrad notebook: 250 m, eight-neighbour occurrences, 500 m square dilation, 20 road cells, 50/30/20 scoring",
            classification=dict(thresholds=list(SCORE_THRESHOLDS), statuses=list(STATUS)),
            station_selection="Union of six full model profiles; original lon/lat; unique station_id",
            owners=sorted(ALLOWED_OWNERS), owner_data_updated=owners["updated"],
            stations=len(stations), linked_stations=int(stations.KobletTilForekomst.sum()),
            occurrences=len(information), relevant_occurrences=len(relevant),
            occurrences_without_station=int(relevant.UtenStasjon.sum()), **result,
            climate=dict(path=str(climate_path), sha256=sha256(climate_path)),
            road_raster=dict(path=str(road_path), sha256=sha256(road_path)),
            road_lines=str(road_source), model_data_root=str(data_root),
            geometry="Original ERF road lines clipped to the scored 250 m road cells; exploded to LineStrings; EPSG:25833")
        (staged / "manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        staged.rename(destination)
    print(json.dumps(result), flush=True)
    print(f"COMPLETE: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data" / "Dekningsgrad")
    parser.add_argument("--data-root", type=Path, default=ROOT / "outputs" / "portable_build" / "Remote_500m" / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "road_climate_coverage")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
