"""Complete, spatially indexed vector data for read-only standalone maps."""
import base64
import gzip

import numpy as np
import shapely


def packed_chunk(frame):
    geographic = frame.to_crs(4326)
    if geographic.empty:
        return None
    # Sub-centimetre rounding avoids storing projection floating-point noise.
    geographic.geometry = shapely.transform(geographic.geometry.to_numpy(),
                                             lambda coordinates: np.round(coordinates, 7))
    content = geographic.to_json(drop_id=True, na="null", show_bbox=True)
    payload = gzip.compress(content.encode("utf-8"), compresslevel=6, mtime=0)
    return dict(bounds=geographic.total_bounds.tolist(), count=len(frame),
                data=base64.b64encode(payload).decode("ascii"))


def spatial_order(bounds):
    center_x = (bounds[:, 0] + bounds[:, 2]) / 2
    center_y = (bounds[:, 1] + bounds[:, 3]) / 2
    return np.lexsort((center_x, np.floor(center_x / 50000), np.floor(center_y / 50000)))


def road_chunks(roads, settings, county_geometry=None, county_codes=None):
    settings = dict(settings)
    if county_codes is not None:
        settings.update(_county_key=tuple(county_codes), _county_geometry=county_geometry)
    bounds = list(roads.geographic_bounds)
    rows = roads.matching_rows(bounds, settings)
    rows = rows[spatial_order(roads.attributes["bounds"][:, rows].T)]
    chunks = []
    for start in range(0, len(rows), 1000):
        ids = roads.attributes["fid"][rows[start:start + 1000]].tolist()
        frame = roads.read_selected(ids, bounds, settings)
        fields = [name for name in ("edge_id", "roadid", "streetname", "municipality", "VS_Total",
                  "VS_Total_klasse", "length_m", "isbridge", "istunnel", "isferry", "categories")
                  if name in frame]
        chunk = packed_chunk(frame[fields + ["geometry"]])
        if chunk is not None:
            chunks.append(chunk)
    return dict(chunks=chunks, count=sum(chunk["count"] for chunk in chunks))


def basin_chunks(basins, depth, county_geometry=None):
    if not basins.prepare():
        raise ValueError("Kuldehull-data mangler")
    frame = basins.polygons
    frame = frame.loc[frame.max_depth_m >= basins.depth(depth)]
    if county_geometry is not None:
        import geopandas as gpd
        county = gpd.GeoSeries([county_geometry], crs=25833).to_crs(frame.crs).iloc[0]
        frame = frame.loc[frame.geometry.intersects(county)].copy()
        frame.geometry = frame.geometry.intersection(county)
        frame = frame.loc[~frame.geometry.is_empty & frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    order = spatial_order(frame.geometry.bounds.to_numpy())
    chunks = []
    for start in range(0, len(order), 500):
        chunk = packed_chunk(frame.iloc[order[start:start + 500]])
        if chunk is not None:
            chunks.append(chunk)
    return dict(chunks=chunks, count=sum(chunk["count"] for chunk in chunks))


def coverage_chunks(layer, classes, county_geometry=None):
    import geopandas as gpd
    from rasterio.features import shapes, geometry_mask
    from shapely.geometry import shape

    classes = layer.classes(classes)
    values, transform, crs = layer.source()
    mask = np.isin(values, classes)
    county = None
    if county_geometry is not None:
        county = gpd.GeoSeries([county_geometry], crs=25833).to_crs(crs).iloc[0]
        if county.is_empty:
            return dict(chunks=[], count=0)
        mask &= geometry_mask([county], out_shape=values.shape, transform=transform,
                              invert=True, all_touched=True)
    records = [dict(value=int(value), geometry=shape(geometry))
               for geometry, value in shapes(values, mask=mask, transform=transform)]
    if not records:
        return dict(chunks=[], count=0)
    frame = gpd.GeoDataFrame(records, crs=crs)
    if county is not None:
        frame.geometry = frame.geometry.intersection(county)
        frame = frame.loc[~frame.geometry.is_empty & (frame.geometry.area > 0)]
    order = spatial_order(frame.geometry.bounds.to_numpy())
    chunks = [packed_chunk(frame.iloc[order[start:start + 500]])
              for start in range(0, len(order), 500)]
    return dict(chunks=chunks, count=sum(chunk["count"] for chunk in chunks))


def weather_maps(current, draw, road_settings, depth, basins_only):
    from station_app import PHENOMENON_DATASETS
    if current.engine.weather is None:
        raise ValueError("Værmodellene mangler; HTML-eksporten krever alle seks modeller")
    available = {item["id"] for item in current.engine.weather.available}
    maps = []
    for phenomenon, dataset in PHENOMENON_DATASETS.items():
        if dataset not in available:
            raise ValueError(f"Værmodellen for {phenomenon} mangler; eksporten krever alle seks modeller")
        current.layers = [dataset]
        with current.as_dataset(dataset):
            current.sync_edits()
            if not current.ready():
                raise ValueError(f"Grunnflaten for {phenomenon} er ikke klar")
            values = current.surface()
            if values is None:
                raise ValueError(f"Ingen stasjoner bidrar til {phenomenon}")
            profile = current.profile()
            image = draw(current, values, "uncertainty", road_settings,
                display_size=1600, kuldehull=dict(only=True, depth=depth) if basins_only else None)
            reference = current.reference() if current.network == "frost" else current.base()
            if reference is None:
                raise ValueError(f"Grunnflaten for {phenomenon} mangler")
            baseline = draw(current, reference, "baseline", road_settings,
                display_size=1600, kuldehull=dict(only=True, depth=depth) if basins_only else None)
            limits = current.limits()
            maps.append(dict(id=phenomenon, dataset=dataset, label=profile["label"], unit=profile["unit"],
                minimum=float(limits[0]), maximum=float(limits[1]), bounds=current.engine.bounds,
                baseline="data:image/png;base64," + base64.b64encode(baseline.getvalue()).decode("ascii"),
                url="data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")))
    return maps
