"""Versioned, validated editable scenario documents; no model arrays are saved."""
import math


def object_value(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} må være et JSON-objekt")
    return value


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Ugyldig {name}")
    return value


def cell(engine, value):
    value = object_value(value, "Terrengrute")
    longitude, latitude = number(value.get("lon"), "lengdegrad"), number(value.get("lat"), "breddegrad")
    if not -180 <= longitude <= 180 or not -85 <= latitude <= 85:
        raise ValueError("Koordinatene er utenfor gyldig område")
    index = engine.snap(longitude, latitude)
    location = engine.station(index)
    if any(abs(location[field] - number(value.get(field), field)) > 1e-6 for field in ("x", "y")):
        raise ValueError("Terrengrutenettet er endret siden scenariet ble lagret")
    return index


def saved_cell(engine, index, position):
    location = engine.station(index)
    return dict(x=location["x"], y=location["y"], **position)


def export_scenario(current):
    with current.lock:
        edits = []
        for station_id, edit in current.station_edits.items():
            saved = dict(station_id=station_id, removed=bool(edit.get("removed")))
            if not saved["removed"]:
                location = current.engine.station(edit["candidate_index"])
                saved["position"] = saved_cell(current.engine, edit["candidate_index"], dict(
                    lon=edit.get("display_lon", location["lon"]), lat=edit.get("display_lat", location["lat"])))
            edits.append(saved)
        selected = current.selection()
        if selected is not None:
            if selected["kind"] == "manual":
                index = selected["index"]
                selected = dict(kind="manual", position=saved_cell(current.engine, index,
                    current.added_positions.get(index, {key: current.engine.station(index)[key] for key in ("lon", "lat")})))
            elif selected["kind"] == "frost":
                selected = dict(kind="frost", station_id=selected["station_id"])
            else:
                selected = dict(kind="greedy", index=selected["index"])
        return dict(cell_m=current.engine.cell_m, dataset=current.dataset, layers=list(current.layers),
            network=current.network, baseline=current.baseline, pross_only=current.pross_only,
            include_met=current.include_met,
            station_since=current.station_since, counties=current.counties, combine=current.combine,
            threshold=current.threshold, comparison_scope=current.comparison_scope,
            station_edits=edits, removed_sensors={identifier: list(sensors)
                for identifier, sensors in current.station_sensor_removals.items()},
            proposals=[dict(position=saved_cell(current.engine, index, current.added_positions.get(
                index, {key: current.engine.station(index)[key] for key in ("lon", "lat")})),
                sensors=list(current.added_sensors.get(index, current.layer_phenomena()))) for index in current.added],
            selected=selected)


def restore_scenario(current, payload, county_layer):
    from station_app import Scenario, PHENOMENON_DATASETS

    data = object_value(payload, "Scenario")
    spacing = number(data.get("cell_m"), "oppløsning")
    if spacing not in (500, 1000, 2500, 5000):
        raise ValueError("Ugyldig oppløsning")
    engine = current.engine if spacing == current.engine.cell_m else current.engine.at_resolution(spacing)
    weather = getattr(engine, "weather", None)
    available = {item["id"] for item in weather.available} if weather is not None else {current.dataset}
    if not isinstance(data.get("dataset"), str) or data["dataset"] not in available:
        raise ValueError("Den lagrede værmodellen er ikke tilgjengelig")
    restored = Scenario(engine, dataset=data["dataset"])
    restored.county_layer = county_layer
    restored.initial_dataset = current.initial_dataset
    restored.initial_baseline = current.initial_baseline
    layers = data.get("layers")
    if (not isinstance(layers, list) or any(not isinstance(layer, str) or layer not in available for layer in layers)
            or len(set(layers)) != len(layers) or layers and layers[0] != restored.dataset):
        raise ValueError("Ugyldige værkartlag")
    if weather is not None:
        for layer in layers:
            weather.profile(layer)
    restored.layers = layers
    baseline = data.get("baseline")
    if isinstance(baseline, bool) or not isinstance(baseline, int) or not 0 <= baseline <= len(engine.stations):
        raise ValueError("Ugyldig stasjonsantall")
    restored.baseline = baseline
    network, pross = data.get("network"), data.get("pross_only")
    if network not in ("frost", "greedy", "empty") or not isinstance(pross, bool):
        raise ValueError("Ugyldig stasjonsnett")
    restored.network, restored.pross_only = network, pross
    include_met = data.get("include_met", False)
    if not isinstance(include_met, bool) or include_met and (network != "frost" or not pross):
        raise ValueError("Ugyldig valg av MET-stasjoner")
    if include_met and (not hasattr(engine, "station_owner_catalog") or engine.station_owner_catalog() is None):
        raise ValueError("Stasjonseiere mangler for Vegværstasjoner + MET-stasjoner")
    restored.include_met = include_met
    restored.station_since = restored.validate_station_date(data.get("station_since"))
    counties = county_layer.selection(data.get("counties"))
    restored.counties = None if counties is None else list(counties)
    for action, key in (("combine", "combine"), ("threshold", "threshold"), ("comparison", "comparison_scope")):
        if action == "threshold":
            number(data.get(key), "terskel")
        restored.apply(dict(action=action, value=data.get(key)))
    edits, proposals = data.get("station_edits"), data.get("proposals")
    if not isinstance(edits, list) or len(edits) > 10000 or not isinstance(proposals, list) or len(proposals) > 40:
        raise ValueError("Ugyldig liste over stasjonsendringer")
    catalog = engine.sensor_catalog() if hasattr(engine, "sensor_catalog") else {}
    for edit in edits:
        edit = object_value(edit, "Stasjonsendring")
        identifier = edit.get("station_id")
        if not isinstance(identifier, str) or identifier in restored.station_edits or identifier not in catalog:
            raise ValueError("Ukjent eller duplisert stasjon")
        if not isinstance(edit.get("removed"), bool):
            raise ValueError("Ugyldig stasjonsendring")
        if edit["removed"]:
            restored.station_edits[identifier] = dict(removed=True)
        else:
            index = cell(engine, edit.get("position"))
            restored.station_edits[identifier] = dict(removed=False, candidate_index=index,
                display_lon=edit["position"]["lon"], display_lat=edit["position"]["lat"])
    removed = object_value(data.get("removed_sensors"), "Fjernede sensorer")
    for identifier, sensors in removed.items():
        if (identifier not in catalog or not isinstance(sensors, list) or
                any(not isinstance(sensor, str) or sensor not in catalog.get(identifier, ()) for sensor in sensors)):
            raise ValueError("Ugyldige fjernede sensorer")
        restored.station_sensor_removals[identifier] = tuple(dict.fromkeys(sensors))
    for proposal in proposals:
        proposal = object_value(proposal, "Stasjonsforslag")
        index = cell(engine, proposal.get("position"))
        sensors = proposal.get("sensors")
        if (index in restored.added or not isinstance(sensors, list) or not sensors or
                any(not isinstance(sensor, str) or sensor not in PHENOMENON_DATASETS for sensor in sensors)):
            raise ValueError("Ugyldige eller dupliserte stasjonsforslag")
        restored.added.append(index)
        restored.added_sensors[index] = tuple(dict.fromkeys(sensors))
        restored.added_positions[index] = {key: proposal["position"][key] for key in ("lon", "lat")}
    restored.sync_edits()
    selected = data.get("selected")
    if selected is not None:
        selected = object_value(selected, "Valgt stasjon")
        kind = selected.get("kind")
        if kind == "manual":
            selected = dict(kind=kind, index=cell(engine, selected.get("position")))
        elif kind == "frost":
            if not isinstance(selected.get("station_id"), str):
                raise ValueError("Ugyldig valgt værstasjon")
        elif kind != "greedy" or isinstance(selected.get("index"), bool) or not isinstance(selected.get("index"), int):
            raise ValueError("Ugyldig valgt stasjon")
        restored.apply(dict(selected, action="select"))
    restored.revision = current.revision + 1
    return restored
