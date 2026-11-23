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


def weather_maps(current, draw, road_settings):
    from station_app import lab_datasets
    if current.engine.weather is None:
        raise ValueError("Temperaturmodellen mangler for HTML-eksport")
    available = {item["id"] for item in current.engine.weather.available}
    dataset = lab_datasets(current.engine)["temperature"]
    if dataset not in available:
        raise ValueError("Temperaturmodellen mangler for HTML-eksport")
    current.layers = [dataset]
    with current.as_dataset(dataset):
        current.sync_edits()
        if not current.ready():
            raise ValueError("Grunnflaten for temperatur er ikke klar")
        values = current.surface()
        if values is None and not current.uses_empty_reference():
            raise ValueError("Ingen stasjoner bidrar til temperatur")
        profile = current.profile()
        reference = current.reference() if current.network == "frost" else current.base()
        if reference is None and not current.uses_empty_reference():
            raise ValueError("Grunnflaten for temperatur mangler")
        if current.uses_empty_reference():
            reduction = None if values is None else np.maximum(current.empty_reference_rmse - values, 0)
        else:
            reduction = current.surface_difference(reference, values)
        limits = current.limits()
        maps = []
        for identifier, label, settings in (
                ("counties", "Temperatur", None),
                ("roads", "Temperatur", dict(road_settings, corridor=True))):
            image = draw(current, values, "uncertainty", settings, display_size=1600)
            baseline = draw(current, reference, "baseline", settings, display_size=1600)
            change = draw(current, reduction, "change", settings, display_size=1600)
            maps.append(dict(id=identifier, dataset=dataset, label=label, unit=profile["unit"],
                minimum=float(limits[0]), maximum=float(limits[1]), bounds=current.engine.bounds,
                empty_reference_rmse=current.empty_reference_rmse if current.uses_empty_reference() else None,
                distance_m=settings["distance_m"] if settings else None,
                reduction="data:image/png;base64," + base64.b64encode(change.getvalue()).decode("ascii"),
                baseline="data:image/png;base64," + base64.b64encode(baseline.getvalue()).decode("ascii"),
                url="data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")))
        return maps


def placement_map(current, clip, road_mask, road_settings):
    from station_app import lab_datasets
    dataset = lab_datasets(current.engine)["temperature"]
    network = current.placement_network()
    layer = current.engine.placement_layer(dataset, station_network=network)
    if layer.get("station_network") != network or layer["cell_m"] != 500:
        raise ValueError("Lagret plasseringsgevinst på 500 m mangler for dette stasjonsnettet")
    active = current.engine.at_resolution(layer["cell_m"])
    image = active.render(layer["values"], "placement", layer["minimum"], layer["maximum"],
                          display_size=1600)
    image = clip(current, image, active=active)
    road_image = active.render(layer["values"], "placement", layer["minimum"], layer["maximum"],
                               display_size=1600, cell_mask=road_mask(current, dict(road_settings, corridor=True), active))
    road_image = clip(current, road_image, active=active)
    return dict(url="data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii"),
                roads_url="data:image/png;base64," + base64.b64encode(road_image.getvalue()).decode("ascii"),
                bounds=layer["bounds"], station_network=layer["station_network"],
                empty_reference_rmse=layer.get("empty_reference_rmse"))
