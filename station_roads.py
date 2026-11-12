"""Read-only road-network views and sampling of the existing weather-error grid."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
import shapely


CATEGORY_NAMES = dict(E="Europaveg", R="Riksveg", F="Fylkesveg", K="Kommunal veg",
                      P="Privat veg", S="Skogsbilveg", U="Ukjent kategori")
ROAD_FIELDS = ["edge_id", "roadclass", "roadtype", "roadid", "length_m", "streetname",
               "municipality", "isferry", "isbridge", "istunnel"]
SCORE_COMPONENTS = [
    ("VS_Vegbase", "Vegbase", "#0072b2"),
    ("VS_Eksponering", "Eksponering", "#e69f00"),
    ("VS_Kjoreforhold", "Kjøreforhold", "#009e73"),
    ("VS_Naturfare", "Naturfare", "#cc79a7"),
    ("VS_Skred", "Skred", "#d55e00"),
    ("VS_Vaerutsatt", "Værutsatt", "#56b4e9"),
]
ROAD_LAYER = "Vegnett_VegScore_full"


def road_references(value):
    if value is None or isinstance(value, (float, np.floating)) and np.isnan(value):
        return []
    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    return [item.strip() for item in next(csv.reader([text], skipinitialspace=True)) if item.strip()]


def road_categories(value):
    categories = set()
    for reference in road_references(value):
        match = re.match(r"([A-Za-z])", reference)
        if match:
            categories.add(match[1].upper())
    return categories or {"U"}


def normalize_reference(value):
    match = re.fullmatch(r"\s*([A-Za-z])\s*[vV]?\s*(\d+)([A-Za-z]?)\s*", str(value))
    if match is None:
        raise ValueError("Bruk et vegnavn som E6, Rv3 eller Fv17")
    return match[1].upper() + str(int(match[2])) + match[3].upper()


def identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def save_road_arrays(path, **arrays):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        np.savez_compressed(staged, **arrays)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def indexed_road_source(source, cache_dir, layer=ROAD_LAYER):
    """Keep the existing indexed SQLite reader behind a compact GDB layer cache."""
    signature = dict(schema="gdb-road-source-v1", source=str(source.resolve()), layer=layer,
                     files=[(path.name, path.stat().st_size, path.stat().st_mtime_ns)
                            for path in sorted(source.iterdir())
                            if path.is_file() and not path.name.lower().endswith(".lock")])
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"road_source_{key}.gpkg"
    if target.is_file():
        return target
    info = pyogrio.read_info(source, layer=layer)
    required = ["linkid", "vegnokkel", "roadcategory", "kommunenavn",
                "istunnel", "isferry", "VS_Total", "VS_Total_klasse", "fylkesnr"]
    missing = set(required) - set(info["fields"])
    if missing:
        raise ValueError(f"{layer} mangler felter: {sorted(missing)}")
    fields = required + [field for field, _, _ in SCORE_COMPONENTS if field in info["fields"]]
    if CRS(info["crs"]).to_epsg() != 25833:
        raise ValueError("Vegnettet må være i EPSG:25833")
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".gpkg", delete=False) as handle:
        staged = Path(handle.name)
    staged.unlink()
    try:
        count = int(info["features"])
        if count <= 0:
            raise ValueError(f"{layer} er tomt")
        for offset in range(0, count, 50000):
            frame = pyogrio.read_dataframe(source, layer=layer, columns=fields,
                                           skip_features=offset, max_features=min(50000, count-offset),
                                           force_2d=True)
            if len(frame) != min(50000, count-offset):
                raise ValueError(f"{layer} ble endret under innlesing")
            pyogrio.write_dataframe(frame, staged, layer=layer, driver="GPKG", append=offset > 0)
            print(f"Vegnett fra {source.name}, lag {layer}: {offset+len(frame)}/{count}", flush=True)
        if int(pyogrio.read_info(staged, layer=layer)["features"]) != count:
            raise ValueError("Ufullstendig vegnettmellomlager")
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)
    return target


class RoadNetwork:
    def __init__(self, path, limit=None, cache_dir=None, portable_cache=False):
        self.path, self.limit = Path(path), limit
        self.portable_cache = portable_cache
        self.cache_dir = self.path.parent / ".station_road_cache" if cache_dir is None else Path(cache_dir)
        self.lock = threading.RLock()
        self.cache = OrderedDict()
        self.overviews = OrderedDict()
        self.overview_samples = OrderedDict()
        self.grid_links = {}
        self.display_data = None
        self.county_display_data = OrderedDict()
        self.display_samples = {}
        self.masks = OrderedDict()
        self.trees = {}
        self.metadata = dict(available=self.path.exists())
        if not self.path.exists():
            return
        self.source_path = self.path
        if self.path.suffix.lower() == ".gdb":
            self.path = indexed_road_source(self.source_path, self.cache_dir)
        info = pyogrio.read_info(self.path)
        self.score_network = "VS_Total" in info["fields"]
        self.score_fields = [field for field, _, _ in SCORE_COMPONENTS if field in info["fields"]]
        self.has_counties = "fylkesnr" in info["fields"]
        self.read_fields = (["linkid", "vegnokkel", "roadcategory", "kommunenavn",
                             "istunnel", "isferry", "VS_Total", "VS_Total_klasse"]
                            if self.score_network else ROAD_FIELDS)
        self.fid_column = info["fid_column"]
        if CRS(info["crs"]).to_epsg() != 25833:
            raise ValueError("Vegnettet må være i EPSG:25833")
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            self.table, self.geometry_column, _ = connection.execute(
                "SELECT table_name,column_name,srs_id FROM gpkg_geometry_columns LIMIT 1").fetchone()
            self.rtree = "rtree_" + self.table + "_" + self.geometry_column
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (self.rtree,)).fetchone():
                raise ValueError("Vegnettet mangler romlig indeks")
        self.attributes, lookup = self.load_attributes()
        self.county_rows = ({code: np.flatnonzero(self.attributes["county"] == code)
                             for code in np.unique(self.attributes["county"])}
                            if self.has_counties else {})
        self.attribute_rows = pd.Index(self.attributes["fid"])
        self.category_bits = {item["code"]: 1 << index for index, item in enumerate(lookup["categories"])}
        self.reference_ids = lookup["references"]
        self.to_xy = Transformer.from_crs(4326, 25833, always_xy=True)
        self.to_lonlat = Transformer.from_crs(25833, 4326, always_xy=True)
        self.native_bounds = tuple(info["total_bounds"])
        self.geographic_bounds = self.to_lonlat.transform_bounds(*self.native_bounds, densify_pts=21)
        self.metadata.update(
            layer=self.table, features=int(info["features"]), limit=limit,
            classes=lookup["classes"], class_counts=lookup["class_counts"], categories=lookup["categories"],
            bounds=list(self.geographic_bounds))
        if self.score_network:
            scores = self.attributes["vs_total"]
            finite = scores[np.isfinite(scores)]
            self.metadata.update(score_network=True, class_field="VS_Total_klasse",
                score_field="VS_Total", score_min=float(finite.min()), score_max=float(finite.max()),
                class_ranges=lookup["class_ranges"])

    def load_attributes(self):
        stat = self.path.stat()
        signature = dict(schema="road-lookup-county-v3", source=str(self.path.resolve()),
                         size=stat.st_size, modified=stat.st_mtime_ns, layer=self.table)
        if self.portable_cache:
            from file_hash import sha256
            signature = dict(schema="road-lookup-portable-v1", source=sha256(self.path), layer=self.table)
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
        self.index_path = self.cache_dir / f"road_lookup_{key}.npz"
        fields = ("fid", "bounds", "roadclass", "category_mask", "priority")
        if self.score_network:
            fields += ("vs_total",)
        if self.has_counties:
            fields += ("county",)
        if self.index_path.exists():
            try:
                with np.load(self.index_path, allow_pickle=False) as stored:
                    arrays = {field: stored[field].copy() for field in fields}
                    lookup = json.loads(str(stored["lookup"]))
                count = len(arrays["fid"])
                if (lookup["signature"] != signature or arrays["bounds"].shape != (4, count) or
                        any(arrays[field].shape != (count,) for field in fields if field != "bounds")):
                    raise ValueError("Ugyldig vegindeks")
                return arrays, lookup
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                pass
        query = (f"SELECT {identifier(self.fid_column)} AS fid,roadid,"
                 f"COALESCE(CAST(roadclass AS TEXT),'U') AS roadclass FROM {identifier(self.table)}")
        if self.score_network:
            query = (f"SELECT {identifier(self.fid_column)} AS fid,vegnokkel AS roadid,"
                     "COALESCE(CAST(CAST(VS_Total_klasse AS INTEGER) AS TEXT),'U') AS roadclass,"
                     f"VS_Total FROM {identifier(self.table)}")
        if self.has_counties:
            query = query.replace(" FROM ", ",fylkesnr FROM ", 1)
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            frame = pd.read_sql_query(query, connection).set_index("fid")
        fids, bounds = pyogrio.read_bounds(self.path, layer=self.table)
        frame = frame.reindex(fids)
        frame["roadid"] = frame["roadid"].fillna("")
        codes = {value: sorted(road_categories(value)) for value in frame["roadid"].unique()}
        categories = sorted({code for values in codes.values() for code in values})
        bits = {code: 1 << index for index, code in enumerate(categories)}
        masks = {value: sum(bits[code] for code in values) for value, values in codes.items()}
        normalized = {}
        for value in codes:
            references = set()
            for reference in road_references(value):
                try:
                    references.add(normalize_reference(reference))
                except ValueError:
                    continue
            normalized[value] = references
        references = {}
        for fid, value in zip(fids, frame["roadid"]):
            for reference in normalized[value]:
                references.setdefault(reference, []).append(int(fid))
        classes = frame["roadclass"].to_numpy(dtype=str)
        arrays = dict(fid=fids, bounds=bounds, roadclass=classes,
            category_mask=frame["roadid"].map(masks).to_numpy(dtype=np.uint32),
            priority=np.array([int(value) if value.isdigit() else 99 for value in classes], dtype=np.int16))
        if self.has_counties:
            arrays["county"] = frame["fylkesnr"].fillna("").astype(str).str.zfill(2).to_numpy(dtype=str)
        class_counts = {str(value): int(count) for value, count in frame["roadclass"].value_counts().items()}
        lookup = dict(signature=signature, references=references, class_counts=class_counts,
            classes=sorted(class_counts, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)),
            categories=[dict(code=code, label=CATEGORY_NAMES.get(code, code + " (fra roadid)"),
                             links=int(np.count_nonzero(arrays["category_mask"] & bits[code]))) for code in categories])
        if self.score_network:
            arrays["vs_total"] = frame["VS_Total"].to_numpy(dtype=np.float64)
            lookup["class_ranges"] = {}
            for code in lookup["classes"]:
                values = arrays["vs_total"][classes == code]
                values = values[np.isfinite(values)]
                lookup["class_ranges"][code] = [float(values.min()), float(values.max())] if len(values) else None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            np.savez_compressed(staged, **arrays, lookup=np.asarray(json.dumps(lookup)))
            staged.replace(self.index_path)
        finally:
            staged.unlink(missing_ok=True)
        return arrays, lookup

    def filters(self, data):
        categories = data.get("categories", ["E", "R", "F"])
        classes = data.get("classes", [value for value in self.metadata.get("classes", [])
                           if value.isdigit() and int(value) < 5])
        if not isinstance(categories, list) or not isinstance(classes, list):
            raise ValueError("Vegkategori og vegklasse må være lister")
        categories = sorted(set(str(value) for value in categories))
        classes = sorted(set(str(value) for value in classes))
        allowed = {item["code"] for item in self.metadata.get("categories", [])}
        if not set(categories) <= allowed or not set(classes) <= set(self.metadata.get("classes", [])):
            raise ValueError("Ukjent vegkategori eller vegklasse")
        route = str(data.get("reference", "")).strip()
        return categories, classes, normalize_reference(route) if route else None

    def bounds(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError("Kartutsnittet må inneholde vest, sør, øst og nord")
        west, south, east, north = map(float, values)
        if not np.isfinite([west, south, east, north]).all() or not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
            raise ValueError("Ugyldig kartutsnitt")
        return west, south, east, north

    def matching_rows(self, bounds, data, padding=0):
        west, south, east, north = self.bounds(bounds)
        categories, classes, reference = self.filters(data)
        extent = self.geographic_bounds
        clipped = (max(west, extent[0]), max(south, extent[1]), min(east, extent[2]), min(north, extent[3]))
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3] or not categories or not classes:
            return np.empty(0, dtype=np.int64)
        xmin, ymin, xmax, ymax = self.to_xy.transform_bounds(*clipped, densify_pts=21)
        codes = data.get("_county_key")
        candidates = np.arange(len(self.attributes["fid"]))
        if codes is not None and self.has_counties:
            parts = [self.county_rows[code] for code in codes if code in self.county_rows]
            candidates = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        minxs, minys, maxxs, maxys = self.attributes["bounds"][:, candidates]
        category_mask = sum(self.category_bits[code] for code in categories)
        selected = ((maxxs >= xmin-padding) & (minxs <= xmax+padding) &
                    (maxys >= ymin-padding) & (minys <= ymax+padding) &
                    np.isin(self.attributes["roadclass"][candidates], classes) &
                    ((self.attributes["category_mask"][candidates] & category_mask) != 0))
        county = data.get("_county_geometry")
        if county is not None:
            if county.is_empty:
                return np.empty(0, dtype=np.int64)
            a, b, c, d = county.bounds
            selected &= (maxxs >= a) & (minxs <= c) & (maxys >= b) & (minys <= d)
        if reference is not None:
            selected &= np.isin(self.attributes["fid"][candidates], self.reference_ids.get(reference, []))
        rows = candidates[np.flatnonzero(selected)]
        return rows[np.lexsort((self.attributes["fid"][rows], self.attributes["priority"][rows]))]

    def viewport_footprint(self, bounds, padding=0):
        west, south, east, north = self.bounds(bounds)
        extent = self.geographic_bounds
        visible = (max(west, extent[0]-.5), max(south, extent[1]-.5),
                   min(east, extent[2]+.5), min(north, extent[3]+.5))
        footprint = shapely.segmentize(shapely.box(*visible), max_segment_length=.1)
        footprint = shapely.transform(footprint, lambda coordinates: np.column_stack(
            self.to_xy.transform(coordinates[:, 0], coordinates[:, 1])))
        return footprint.buffer(padding) if padding else footprint

    def components(self, data):
        if not self.metadata["available"] or not self.score_network:
            raise ValueError("Vegnettet har ingen VegScore-komponenter")
        if data.get("fid") is not None:
            fid = data["fid"]
            if isinstance(fid, bool) or not isinstance(fid, int) or fid < 0:
                raise ValueError("Ugyldig veglenke")
            fids = [fid] if fid in self.attributes["fid"] else []
            point = None
        else:
            lon, lat = float(data["lon"]), float(data["lat"])
            if not np.isfinite([lon, lat]).all() or not -180 <= lon <= 180 or not -90 <= lat <= 90:
                raise ValueError("Ugyldig kartposisjon")
            x, y = self.to_xy.transform(lon, lat)
            point = shapely.Point(x, y)
            west, south = self.to_lonlat.transform(x - 3000, y - 3000)
            east, north = self.to_lonlat.transform(x + 3000, y + 3000)
            rows = self.matching_rows([west, south, east, north], data)
            fids = self.attributes["fid"][rows].tolist()
        if not fids:
            raise ValueError("Ingen veglenke funnet her. Zoom inn og velg en veg.")
        frame = pyogrio.read_dataframe(self.path, layer=self.table, fids=fids,
                                      columns=["linkid", "vegnokkel", *self.score_fields], fid_as_index=True)
        if frame.empty:
            raise ValueError("Veglenken finnes ikke i datakilden")
        if point is not None:
            distances = frame.geometry.distance(point)
            frame = frame.loc[[distances.idxmin()]]
            if float(distances.min()) > 3000:
                raise ValueError("Ingen veglenke funnet her. Zoom inn og velg en veg.")
        row = frame.iloc[0]
        components = []
        for field, label, colour in SCORE_COMPONENTS:
            value = row.get(field)
            if pd.isna(value):
                share_value = None
            else:
                share_value = float(value)
                if not np.isfinite(share_value) or share_value < 0:
                    raise ValueError(f"Ugyldig VegScore-komponent: {label}")
            components.append(dict(label=label, colour=colour, value=share_value))
        total = sum(item["value"] for item in components if item["value"] is not None)
        for item in components:
            value = item.pop("value")
            item["share"] = None if value is None else value / total if total else 0
        return dict(fid=int(frame.index[0]), roadid=str(row["vegnokkel"]), edge_id=str(row["linkid"]),
                    components=components)

    def read(self, bounds, data, padding=0, fid_filter=None):
        import geopandas as gpd

        if not self.metadata["available"]:
            raise ValueError("Vegnettfilen er ikke tilgjengelig")
        bounds = self.bounds(bounds)
        categories, classes, reference = self.filters(data)
        restricted_ids = None if fid_filter is None else tuple(fid_filter)
        key = (bounds, tuple(categories), tuple(classes), reference, padding, restricted_ids, data.get("_county_key"))
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        empty = gpd.GeoDataFrame(columns=ROAD_FIELDS + (
            ["VS_Total", "VS_Total_klasse"] if self.score_network else []), geometry=[], crs=25833)
        rows = self.matching_rows(bounds, data, padding)
        if fid_filter is not None:
            rows = rows[np.isin(self.attributes["fid"][rows], fid_filter)]
        truncated = self.limit is not None and len(rows) > self.limit
        if not len(rows):
            return empty, truncated
        selected_ids = self.attributes["fid"][rows[:self.limit]].tolist()
        roads = self.read_selected(selected_ids, bounds, data, padding)
        result = roads, truncated
        with self.lock:
            self.cache[key] = result
            if len(self.cache) > 6:
                self.cache.popitem(last=False)
        return result

    def read_selected(self, selected_ids, bounds, data, padding=0):
        """Read an explicit batch without viewport limits or display aggregation."""
        if len(selected_ids) > 2000:
            selection = identifier(self.fid_column) + " IN (" + ",".join(str(int(fid)) for fid in selected_ids) + ")"
            roads = pyogrio.read_dataframe(self.path, layer=self.table, where=selection,
                                          columns=self.read_fields, fid_as_index=True)
        else:
            roads = pyogrio.read_dataframe(self.path, layer=self.table, fids=selected_ids,
                                          columns=self.read_fields, fid_as_index=True)
        roads = roads.reindex(selected_ids)
        if self.score_network:
            roads = roads.rename(columns=dict(linkid="edge_id", vegnokkel="roadid",
                                               roadcategory="roadtype", kommunenavn="municipality"))
            roads["roadclass"] = roads["VS_Total_klasse"].map(
                lambda value: str(int(value)) if pd.notna(value) else "U")
            roads["length_m"] = roads.geometry.length
            roads["streetname"], roads["isbridge"] = "", False
            for field in ("istunnel", "isferry"):
                roads[field] = roads[field].astype(str).str.lower().isin(["1", "true", "ja"])
        roads = roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
        roads.geometry = shapely.force_2d(roads.geometry.to_numpy())
        clip_geometry = self.viewport_footprint(bounds, padding)
        if data.get("_county_geometry") is not None:
            clip_geometry = shapely.intersection(clip_geometry, data["_county_geometry"])
        shapely.prepare(clip_geometry)
        geometries = roads.geometry.to_numpy(copy=True)
        crossing = ~shapely.covers(clip_geometry, geometries)
        if crossing.any():
            geometries[crossing] = shapely.intersection(geometries[crossing], clip_geometry)
        roads.geometry = geometries
        roads = roads.loc[roads.geometry.geom_type.isin(["LineString", "MultiLineString"]) & (roads.geometry.length > 0)].copy()
        masks = self.attributes["category_mask"][self.attribute_rows.get_indexer(roads.index)]
        category_codes = {int(mask): [code for code, bit in self.category_bits.items() if int(mask) & bit]
                  for mask in np.unique(masks)}
        roads["categories"] = [category_codes[int(mask)] for mask in masks]
        return roads

    def grid_values(self, engine, coordinates, values):
        if values is None:
            return np.full(len(coordinates), np.nan)
        key = id(engine.grid)
        with self.lock:
            if key not in self.trees:
                centres = engine.grid[["x", "y"]].to_numpy()
                self.trees[key] = cKDTree(centres), centres
            tree, centres = self.trees[key]
        _, nearest = tree.query(coordinates)
        valid = (np.abs(centres[nearest] - coordinates) <= engine.cell_m / 2 + 0.01).all(axis=1)
        result = np.full(len(coordinates), np.nan)
        result[valid] = values[nearest[valid]]
        return result

    def grid_link_key(self, engine):
        coordinates = engine.grid[["x", "y"]].to_numpy(dtype=np.float64)
        signature = f"road-grid-v1:{self.index_path.stem}:{engine.cell_m}:".encode() + coordinates.tobytes()
        return hashlib.sha256(signature).hexdigest()[:24]

    def load_grid_links(self, engine, distance):
        key = self.grid_link_key(engine)
        paths = []
        for path in self.cache_dir.glob(f"road_grid_{key}_*.npz"):
            try:
                maximum = float(path.stem.rsplit("_", 1)[1])
            except ValueError:
                continue
            if maximum >= distance:
                paths.append((maximum, path))
        for maximum, path in sorted(paths):
            with self.lock:
                if path in self.grid_links:
                    return self.grid_links[path]
            try:
                with np.load(path, allow_pickle=False) as stored:
                    links = {field: stored[field].copy() for field in ("cell", "road", "distance")}
                    if str(stored["key"]) != key or float(stored["maximum"]) != maximum:
                        continue
                cells, roads, distances = links["cell"], links["road"], links["distance"]
                if (cells.ndim != 1 or roads.shape != cells.shape or distances.shape != cells.shape or
                        (cells < 0).any() or (cells >= len(engine.grid)).any() or
                        (roads < 0).any() or (roads >= len(self.attributes["fid"])).any() or
                        not np.isfinite(distances).all() or (distances < 0).any() or (distances > maximum).any()):
                    continue
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                continue
            links["path"] = path
            with self.lock:
                self.grid_links[path] = links
            return links
        return None

    def prepare_grid(self, engine, maximum=500, progress=None):
        maximum = float(maximum)
        if not np.isfinite(maximum) or not 0 <= maximum <= 5000:
            raise ValueError("Maksimal vegavstand må være mellom 0 og 5000 meter")
        saved = self.load_grid_links(engine, maximum)
        if saved is not None:
            return saved["path"]
        roads = pyogrio.read_dataframe(self.path, layer=self.table, columns=[], fid_as_index=True)
        roads = roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty &
                          roads.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        geometries = shapely.force_2d(roads.geometry.to_numpy())
        road_rows = self.attribute_rows.get_indexer(roads.index)
        if (road_rows < 0).any():
            raise ValueError("Vegfilen er endret etter at vegindeksen ble lastet")
        tree = shapely.STRtree(geometries)
        coordinates = engine.grid[["x", "y"]].to_numpy(dtype=float)
        half = engine.cell_m / 2
        cells, owners, distances = [], [], []
        for start in range(0, len(coordinates), 2048):
            points = coordinates[start:start+2048]
            squares = shapely.box(points[:, 0]-half, points[:, 1]-half, points[:, 0]+half, points[:, 1]+half)
            hits = tree.query(squares, predicate="dwithin", distance=maximum)
            cells.append((hits[0]+start).astype(np.int32))
            owners.append(road_rows[hits[1]].astype(np.int32))
            distances.append(shapely.distance(squares[hits[0]], geometries[hits[1]]))
            if progress is not None:
                progress(min(start+2048, len(coordinates)), len(coordinates))
        key = self.grid_link_key(engine)
        path = self.cache_dir / f"road_grid_{key}_{maximum:g}.npz"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            np.savez_compressed(staged, cell=np.concatenate(cells), road=np.concatenate(owners),
                                distance=np.concatenate(distances), key=np.asarray(key), maximum=np.asarray(maximum))
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)
        return path

    def corridor_mask(self, engine, bounds, data, distance):
        distance = float(distance)
        if not np.isfinite(distance) or not 0 <= distance <= 5000:
            raise ValueError("Vegkorridoren må være mellom 0 og 5000 meter")
        categories, classes, reference = self.filters(data)
        key = (id(engine.grid), tuple(self.bounds(bounds)), tuple(categories), tuple(classes), reference, distance,
               data.get("_county_key"))
        with self.lock:
            if key in self.masks:
                self.masks.move_to_end(key)
                return self.masks[key]
        mask = np.zeros(len(engine.grid), dtype=bool)
        links = self.load_grid_links(engine, distance) if data.get("_county_geometry") is None else None
        if links is not None:
            rows = self.matching_rows(bounds, data, padding=distance)[:self.limit]
            if len(rows):
                footprint = self.viewport_footprint(bounds, distance)
                shapely.prepare(footprint)
                inside = shapely.covers(footprint, shapely.box(*self.attributes["bounds"][:, rows]))
                selected = np.zeros(len(self.attributes["fid"]), dtype=bool)
                selected[rows[inside]] = True
                hits = selected[links["road"]] & (links["distance"] <= distance)
                mask[links["cell"][hits]] = True
                boundary = self.attributes["fid"][rows[~inside]]
                if len(boundary):
                    roads, _ = self.read(bounds, data, padding=distance, fid_filter=boundary)
                    mask |= self.spatial_mask(engine, roads, distance)
        else:
            roads, _ = self.read(bounds, data, padding=distance)
            mask = self.spatial_mask(engine, roads, distance)
        with self.lock:
            self.masks[key] = mask
            if len(self.masks) > 6:
                self.masks.popitem(last=False)
        return mask

    def spatial_mask(self, engine, roads, distance):
        mask = np.zeros(len(engine.grid), dtype=bool)
        if roads.empty:
            return mask
        xmin, ymin, xmax, ymax = roads.total_bounds
        half = engine.cell_m / 2
        xs, ys = engine.grid["x"].to_numpy(), engine.grid["y"].to_numpy()
        candidates = np.flatnonzero((xs >= xmin-distance-half) & (xs <= xmax+distance+half) &
                                    (ys >= ymin-distance-half) & (ys <= ymax+distance+half))
        squares = shapely.box(xs[candidates]-half, ys[candidates]-half,
                              xs[candidates]+half, ys[candidates]+half)
        hits = shapely.STRtree(roads.geometry.to_numpy()).query(squares, predicate="dwithin", distance=distance)
        mask[candidates[np.unique(hits[0])]] = True
        return mask

    def display_path(self):
        return self.cache_dir / f"road_display_v1_{self.index_path.stem}.npz"

    def load_display(self):
        if self.display_data is not None:
            return self.display_data
        path = self.display_path()
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as stored:
                data = {field: stored[field].copy() for field in
                        ("raw", "raw_offsets", "display", "display_offsets", "lengths")}
            count = len(self.attributes["fid"])
            if data["lengths"].shape != (count,):
                return None
            for field in ("raw", "display"):
                offsets = data[field + "_offsets"]
                if (offsets.shape != (count+1,) or offsets[0] != 0 or
                        offsets[-1] != len(data[field]) or (np.diff(offsets) < 0).any()):
                    return None
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            return None
        self.display_data = data
        return data

    @staticmethod
    def unpack_display(data, field, rows):
        offsets = data[field + "_offsets"]
        buffer = data[field]
        return shapely.from_wkb([buffer[offsets[row]:offsets[row+1]].tobytes() for row in rows])

    def load_display_samples(self, engine):
        key = self.grid_link_key(engine)
        if key in self.display_samples:
            return self.display_samples[key]
        path = self.cache_dir / f"road_samples_v1_{key}.npz"
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as stored:
                data = {field: stored[field].copy() for field in ("road", "cell", "weight")}
            if (data["road"].ndim != 1 or data["cell"].shape != data["road"].shape or
                    data["weight"].shape != data["road"].shape or (data["road"] < 0).any() or
                    (data["road"] >= len(self.attributes["fid"])).any() or (data["cell"] < 0).any() or
                    (data["cell"] >= len(engine.grid)).any() or not np.isfinite(data["weight"]).all()):
                return None
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            return None
        self.display_samples[key] = data
        return data

    def prepare_display(self, engine):
        if self.load_display() is not None and self.load_display_samples(engine) is not None:
            for code in self.county_rows:
                self.load_county_display(code)
            return self.display_path()
        if not hasattr(self, "prepared_source"):
            source = pyogrio.read_dataframe(self.path, layer=self.table, columns=[], fid_as_index=True)
            source = source.loc[source.geometry.notna() & ~source.geometry.is_empty &
                                source.geometry.geom_type.isin(["LineString", "MultiLineString"])]
            rows = self.attribute_rows.get_indexer(source.index)
            if (rows < 0).any():
                raise ValueError("Vegfilen er endret etter at vegindeksen ble lastet")
            self.prepared_source = np.empty(len(self.attributes["fid"]), dtype=object)
            self.prepared_source.fill(shapely.LineString())
            self.prepared_source[rows] = shapely.force_2d(source.geometry.to_numpy())
        geometries = self.prepared_source
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.load_display() is None:
            displayed = shapely.transform(shapely.simplify(geometries, 150000 / 128, preserve_topology=True),
                lambda points: np.round(np.column_stack(self.to_lonlat.transform(points[:, 0], points[:, 1])), 5))
            arrays = dict(lengths=shapely.length(geometries))
            for field, values in (("raw", geometries), ("display", displayed)):
                encoded = shapely.to_wkb(values)
                sizes = np.fromiter((len(value) for value in encoded), dtype=np.int64, count=len(encoded))
                arrays[field] = np.frombuffer(b"".join(encoded), dtype=np.uint8)
                arrays[field + "_offsets"] = np.r_[0, np.cumsum(sizes)]
            save_road_arrays(self.display_path(), **arrays)
        if self.load_display_samples(engine) is None:
            parts, owners = shapely.get_parts(geometries, return_index=True)
            lengths = shapely.length(parts)
            counts = np.maximum(1, np.ceil(lengths / max(250, min(engine.cell_m / 2, 1000))).astype(np.int64))
            part_ids = np.repeat(np.arange(len(parts)), counts)
            starts = np.repeat(np.cumsum(counts) - counts, counts)
            positions = (np.arange(len(part_ids))-starts+.5) / counts[part_ids] * lengths[part_ids]
            points = shapely.get_coordinates(shapely.line_interpolate_point(parts[part_ids], positions))
            cells = self.grid_values(engine, points, np.arange(len(engine.grid)))
            valid = np.isfinite(cells)
            save_road_arrays(self.cache_dir / f"road_samples_v1_{self.grid_link_key(engine)}.npz",
                road=owners[part_ids][valid].astype(np.int32), cell=cells[valid].astype(np.int32),
                weight=lengths[part_ids][valid] / counts[part_ids][valid])
        for code in self.county_rows:
            self.load_county_display(code)
        return self.display_path()

    def load_county_display(self, code):
        if code in self.county_display_data:
            self.county_display_data.move_to_end(code)
            return self.county_display_data[code]
        rows = self.county_rows.get(code)
        if rows is None:
            return None
        path = self.display_path().with_name(self.display_path().stem + f"_county_{code}.npz")
        if path.is_file():
            with np.load(path, allow_pickle=False) as stored:
                result = {field: stored[field].copy() for field in stored.files}
        else:
            display = self.load_display()
            if display is None:
                return None
            result = dict(rows=rows, lengths=display["lengths"][rows])
            for field in ("raw", "display"):
                offsets = display[field + "_offsets"]
                chunks = [display[field][offsets[row]:offsets[row+1]] for row in rows]
                result[field] = np.concatenate(chunks)
                result[field + "_offsets"] = np.r_[0, np.cumsum([len(chunk) for chunk in chunks])]
            save_road_arrays(path, **result)
        if not np.array_equal(result["rows"], rows):
            raise ValueError(f"Ugyldig fylkesindeks for vegnett: {path}")
        self.county_display_data[code] = result
        if len(self.county_display_data) > 2:
            self.county_display_data.popitem(last=False)
        return result

    def prepared_overview(self, engine, current, bounds, data, rows):
        codes = data.get("_county_key")
        if codes and self.has_counties:
            pieces = [self.load_county_display(code) for code in sorted(set(codes)) if code in self.county_rows]
            if not pieces or any(piece is None for piece in pieces):
                return None
            display = {field: np.concatenate([piece[field] for piece in pieces])
                       for field in ("rows", "lengths", "raw", "display")}
            for field in ("raw", "display"):
                sizes = np.concatenate([np.diff(piece[field + "_offsets"]) for piece in pieces])
                display[field + "_offsets"] = np.r_[0, np.cumsum(sizes)]
            display_rows = pd.Index(display["rows"]).get_indexer(rows[:self.limit])
        else:
            display = self.load_display()
            display_rows = rows[:self.limit]
        samples = self.load_display_samples(engine) if current is not None else None
        if display is None or (current is not None and samples is None):
            return None
        rows = rows[:self.limit]
        lengths = display["lengths"][display_rows].copy()
        footprint = self.viewport_footprint(bounds)
        if data.get("_county_geometry") is not None:
            footprint = shapely.intersection(footprint, data["_county_geometry"])
        shapely.prepare(footprint)
        boundary = ~shapely.covers(footprint, shapely.box(*self.attributes["bounds"][:, rows]))
        if boundary.any():
            raw = self.unpack_display(display, "raw", display_rows[boundary])
            lengths[boundary] = shapely.length(shapely.intersection(raw, footprint))
        keep = lengths > 0
        rows, lengths, display_rows = rows[keep], lengths[keep], display_rows[keep]
        geometries = self.unpack_display(display, "display", display_rows)
        clip = boundary[keep]
        if data.get("_county_geometry") is not None and clip.any():
            geographic_clip = shapely.transform(footprint, lambda coordinates: np.column_stack(
                self.to_lonlat.transform(coordinates[:, 0], coordinates[:, 1])))
            geometries[clip] = shapely.intersection(geometries[clip], geographic_clip)
        categories, _, _ = self.filters(data)
        labels = np.array([next(code for code in categories if int(mask) & self.category_bits[code])
                           for mask in self.attributes["category_mask"][rows]])
        values = np.full(len(rows), np.nan)
        if samples is not None:
            owners, cells, weights = samples["road"], samples["cell"], samples["weight"]
            valid = np.isfinite(current[cells])
            totals = np.bincount(owners[valid], weights=weights[valid], minlength=len(self.attributes["fid"]))
            errors = np.bincount(owners[valid], weights=current[cells[valid]]*weights[valid], minlength=len(totals))
            values = np.divide(errors, totals, out=np.full(len(totals), np.nan), where=totals > 0)[rows]
        return self.group_overview(geometries, labels, values, lengths, data,
                                   self.limit is not None and len(self.matching_rows(bounds, data)) > self.limit,
                                   self.attributes["vs_total"][rows] if self.score_network else None,
                                   self.attributes["roadclass"][rows] if self.score_network else None)

    def view(self, engine, current, reduction, threshold, bounds, data, zoom=8):
        categories, classes, reference = self.filters(data)
        geometry_only = data.get("geometry_only") is True
        colouring = data.get("colour") == "error" and current is not None
        surface_key = hashlib.sha256(np.asarray(current).tobytes()).digest() if colouring else None
        overview_key = (tuple(self.bounds(bounds)), tuple(categories), tuple(classes), reference, float(zoom),
                        id(engine.grid) if colouring else None, surface_key,
                        data.get("colour_min"), data.get("colour_max"), data.get("colour"), data.get("_county_key"),
                        geometry_only)
        with self.lock:
            if overview_key in self.overviews:
                self.overviews.move_to_end(overview_key)
                return dict(self.overviews[overview_key])
        if float(zoom) < 8:
            rows = self.matching_rows(bounds, data)
            if len(rows) > 2000:
                result = self.prepared_overview(engine, current if colouring else None, bounds, data, rows)
                if result is not None:
                    with self.lock:
                        self.overviews[overview_key] = result
                        if len(self.overviews) > 4:
                            self.overviews.popitem(last=False)
                    return dict(result)
        roads, truncated = self.read(bounds, data)
        if float(zoom) < 8 and len(roads) > 2000:
            result = self.overview(roads, truncated, data, zoom, engine, current if colouring else None)
            with self.lock:
                self.overviews[overview_key] = result
                if len(self.overviews) > 2:
                    self.overviews.popitem(last=False)
            return dict(result)
        features, coordinates, lengths, offsets = [], [], [], [0]
        geometries = roads.geometry.to_numpy()
        total_length = float(shapely.length(geometries).sum())
        step = max(250, min(engine.cell_m / 2, 1000), total_length / 40000)
        tolerance = max(0, 150000 / 2 ** min(max(float(zoom), 3), 18))
        for geometry in geometries:
            if geometry_only:
                offsets.append(0)
                continue
            parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
            for part in parts:
                count = max(1, int(np.ceil(part.length / step)))
                positions = (np.arange(count) + 0.5) * part.length / count
                sampled = shapely.line_interpolate_point(part, positions)
                coordinates.extend(shapely.get_coordinates(sampled).tolist())
                lengths.extend([part.length / count] * count)
            offsets.append(len(coordinates))
        coordinates = np.asarray(coordinates).reshape(-1, 2)
        weights = np.asarray(lengths)
        errors = np.empty(0) if geometry_only else self.grid_values(engine, coordinates, current)
        gains = np.empty(0) if geometry_only else self.grid_values(engine, coordinates, reduction)
        displayed = shapely.transform(shapely.simplify(geometries, tolerance, preserve_topology=True),
                                      lambda points: np.column_stack(self.to_lonlat.transform(points[:, 0], points[:, 1])))
        records = roads[ROAD_FIELDS + (["VS_Total", "VS_Total_klasse"] if self.score_network else [])].to_dict("records")
        for index, (fid, geometry, display, record, categories) in enumerate(
            zip(roads.index, geometries, displayed, records, roads["categories"])):
            start, end = offsets[index:index+2]
            values, length_weights = errors[start:end], weights[start:end]
            valid = np.isfinite(values) & (length_weights > 0)
            mean = float(np.average(values[valid], weights=length_weights[valid])) if valid.any() else None
            maximum = float(values[valid].max()) if valid.any() else None
            properties = {key: None if pd.isna(value) else value for key, value in record.items()}
            properties.update(fid=int(fid), categories=categories, mean_error=mean,
                              max_error=maximum, displayed_km=float(geometry.length / 1000))
            features.append(dict(type="Feature", geometry=shapely.geometry.mapping(display), properties=properties))
        valid = np.isfinite(errors) & (weights > 0)
        errors, weights, gains = errors[valid], weights[valid], gains[valid]
        if len(errors):
            order = np.argsort(errors)
            percentile = float(errors[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() * 0.9)]])
            mean, mapped = float(np.average(errors, weights=weights)), float(weights.sum() / 1000)
            improved = None if reduction is None else float(weights[gains > threshold].sum() / 1000)
        else:
            mean = percentile = improved = None
            mapped = 0
        payload = dict(type="FeatureCollection", features=features,
                       summary=dict(links=len(features), truncated=truncated, displayed_km=total_length/1000,
                                    mapped_km=mapped, mean_error=mean, p90_error=percentile,
                                    improved_km=improved, sampling_m=step, corridor_applies_to="Viste veglenker i kartutsnittet"))
        result = json.loads(json.dumps(payload, default=lambda value: value.item() if isinstance(value, np.generic) else None,
                                       allow_nan=False))
        if geometry_only:
            with self.lock:
                self.overviews[overview_key] = result
                if len(self.overviews) > 4:
                    self.overviews.popitem(last=False)
        return result

    def overview_errors(self, engine, roads, current):
        key = (id(engine.grid), id(roads))
        with self.lock:
            stored = self.overview_samples.get(key)
        if stored is None:
            parts, owners = shapely.get_parts(roads.geometry.to_numpy(), return_index=True)
            lengths = shapely.length(parts)
            step = max(250, min(engine.cell_m / 2, 1000))
            counts = np.maximum(1, np.ceil(lengths / step).astype(np.int64))
            part_ids = np.repeat(np.arange(len(parts)), counts)
            starts = np.repeat(np.cumsum(counts) - counts, counts)
            positions = (np.arange(len(part_ids)) - starts + .5) / counts[part_ids] * lengths[part_ids]
            coordinates = shapely.get_coordinates(shapely.line_interpolate_point(parts[part_ids], positions))
            cells = self.grid_values(engine, coordinates, np.arange(len(engine.grid)))
            valid = np.isfinite(cells)
            stored = (roads, cells[valid].astype(np.int64), owners[part_ids][valid],
                      lengths[part_ids][valid] / counts[part_ids][valid])
            with self.lock:
                self.overview_samples[key] = stored
                if len(self.overview_samples) > 2:
                    self.overview_samples.popitem(last=False)
        _, cells, owners, weights = stored
        valid = np.isfinite(current[cells])
        totals = np.bincount(owners[valid], weights=weights[valid], minlength=len(roads))
        errors = np.bincount(owners[valid], weights=current[cells[valid]] * weights[valid], minlength=len(roads))
        return np.divide(errors, totals, out=np.full(len(roads), np.nan), where=totals > 0)

    def overview(self, roads, truncated, data, zoom, engine=None, current=None):
        categories, _, _ = self.filters(data)
        labels = roads["categories"].map(lambda codes: next(code for code in codes if code in categories)).to_numpy()
        geometries = roads.geometry.to_numpy()
        tolerance = 150000 / 2 ** min(max(float(zoom), 3), 18)
        displayed = shapely.transform(shapely.simplify(geometries, tolerance, preserve_topology=True),
            lambda points: np.round(np.column_stack(self.to_lonlat.transform(points[:, 0], points[:, 1])), 5))
        values = self.overview_errors(engine, roads, current) if current is not None else np.full(len(roads), np.nan)
        return self.group_overview(displayed, labels, values, shapely.length(geometries), data, truncated,
                                   roads["VS_Total"].to_numpy() if self.score_network else None,
                                   roads["roadclass"].to_numpy() if self.score_network else None)

    def group_overview(self, displayed, labels, values, lengths, data, truncated, scores=None, classes=None):
        if scores is not None and str(data.get("colour", "")).startswith("score"):
            features = []
            for category, code in sorted(set(zip(labels, classes))):
                selected = (labels == category) & (classes == code)
                geometry = shapely.multilinestrings(shapely.get_parts(displayed[selected]))
                finite = selected & np.isfinite(scores)
                properties = dict.fromkeys(ROAD_FIELDS)
                properties.update(roadid=CATEGORY_NAMES.get(category, category), streetname="VS_Total oversikt",
                    categories=[category], roadclass=code, VS_Total_klasse=None if code == "U" else int(code),
                    VS_Total=float(np.average(scores[finite], weights=lengths[finite])) if finite.any() else None,
                    overview=True, links=int(selected.sum()), mean_error=None, max_error=None,
                    displayed_km=float(lengths[selected].sum()/1000))
                features.append(dict(type="Feature", geometry=json.loads(shapely.to_geojson(geometry)), properties=properties))
            return dict(type="FeatureCollection", features=features,
                summary=dict(links=len(scores), truncated=truncated, overview=True,
                    displayed_km=float(lengths.sum()/1000), mapped_km=None, mean_error=None, p90_error=None,
                    improved_km=None, sampling_m=None, corridor_applies_to="Alle filtrerte veglenker i kartutsnittet"))
        buckets = np.full(len(values), -1, dtype=np.int16)
        finite = np.isfinite(values)
        if finite.any():
            minimum = max(float(data.get("colour_min", np.percentile(values[finite], .5))), .0001)
            maximum = max(float(data.get("colour_max", np.percentile(values[finite], 99.5))), minimum * 1.01)
            boundaries = np.geomspace(minimum, maximum, 17)
            buckets[finite] = np.clip(np.searchsorted(boundaries, values[finite], side="right") - 1, 0, 15)
        features = []
        for category, bucket in sorted(set(zip(labels, buckets))):
            selected = (labels == category) & (buckets == bucket)
            geometry = shapely.multilinestrings(shapely.get_parts(displayed[selected]))
            properties = dict.fromkeys(ROAD_FIELDS)
            properties.update(roadid=CATEGORY_NAMES.get(category, category), streetname="Oversikt",
                categories=[category], overview=True, links=int(selected.sum()),
                colour_value=float(np.sqrt(boundaries[bucket] * boundaries[bucket+1])) if bucket >= 0 else None,
                mean_error=float(np.average(values[selected], weights=lengths[selected])) if bucket >= 0 else None,
                max_error=float(values[selected].max()) if bucket >= 0 else None,
                displayed_km=float(lengths[selected].sum() / 1000))
            features.append(dict(type="Feature", geometry=json.loads(shapely.to_geojson(geometry)), properties=properties))
        return dict(type="FeatureCollection", features=features,
            summary=dict(links=len(values), truncated=truncated, overview=True,
                displayed_km=float(lengths.sum() / 1000), mapped_km=None,
                mean_error=None, p90_error=None, improved_km=None, sampling_m=None,
                corridor_applies_to="Alle filtrerte veglenker i kartutsnittet"))