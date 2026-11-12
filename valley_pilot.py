"""Shared, testable parts of the winter station-transfer pilot."""
from __future__ import annotations

import heapq
import re

import numpy as np


HOURS = (0, 3, 6)
WINTERS = (2022, 2023, 2024)
REGIME_FIELDS = ("temperature_c", "wind_x_ms", "wind_y_ms", "cloud_fraction")
TERRAIN_FIELDS = (
    "distance_km", "signed_height_difference_km", "source_height_km", "target_height_km",
    "source_ocean_distance_km", "target_ocean_distance_km",
    "source_fjord_distance_km", "target_fjord_distance_km",
    "source_tpi10_km", "target_tpi10_km", "source_relief10_km", "target_relief10_km",
)
GRAPH_FIELDS = ("connected", "source_pass_rise_km", "target_pass_rise_km",
                "straight_ridge_rise_km", "straight_route_supported")


def winter_year(timestamp) -> int:
    """December belongs to the winter beginning in that calendar year."""
    if timestamp.month not in (12, 1, 2):
        raise ValueError("Pilot accepts DJF only")
    return timestamp.year if timestamp.month == 12 else timestamp.year - 1


def frost_records(payload: dict) -> list[dict]:
    """Retain QC and sensor identity; never pool different sensors."""
    import pandas as pd
    records = []
    for item in payload.get("data", []):
        source = item["sourceId"]
        stamp = pd.Timestamp(item["referenceTime"])
        if stamp.tzinfo is None:
            raise ValueError("Frost observation timestamp must include timezone")
        stamp = stamp.tz_convert("UTC")
        for observation in item.get("observations", []):
            if observation.get("elementId") != "air_temperature":
                continue
            offset = observation.get("timeOffset")
            resolution = observation.get("timeResolution")
            level = observation.get("level")
            level_ok = level is None or (
                level.get("levelType") == "height_above_ground" and
                level.get("unit") == "m" and float(level["value"]) == 2.)
            value = observation.get("value")
            quality = observation.get("qualityCode")
            reason = None
            if observation.get("unit") != "degC":
                reason = "unit"
            elif offset != "PT0H" or resolution != "PT1H":
                reason = "time_definition"
            elif not level_ok:
                reason = "sensor_height"
            elif quality != 0:
                # Code1 includes corrected/interpolated values: exclude from this pilot.
                reason = "quality_not_zero"
            elif value is None or not np.isfinite(float(value)) or not -60 <= float(value) <= 45:
                reason = "physical_range"
            elif stamp.minute != 0 or stamp.second != 0 or stamp.hour not in HOURS:
                reason = "not_selected_hour"
            elif stamp.month not in (12, 1, 2):
                reason = "not_winter"
            sensor = observation.get("timeSeriesId")
            if sensor is None:
                reason = "missing_sensor_identity"
            records.append(dict(
                station_id=source.split(":")[0], source_id=source, sensor_id=sensor,
                time=stamp.isoformat(), value=value, quality_code=quality,
                unit=observation.get("unit"), time_offset=offset, time_resolution=resolution,
                level=level, accepted=reason is None, rejection=reason))
    return records


def select_observations(raw):
    """Choose one sensor per station using training-period metadata coverage only."""
    import pandas as pd
    required = {"station_id", "source_id", "sensor_id", "time", "value", "accepted"}
    if not required <= set(raw.columns):
        raise ValueError("Missing quality-preserving Frost columns")
    accepted = raw.loc[raw["accepted"].eq(True)].copy()
    if accepted.empty:
        raise ValueError("No qualityCode0 hourly nighttime observations")
    accepted["time"] = pd.to_datetime(accepted["time"], utc=True)
    accepted["winter"] = accepted["time"].map(winter_year)
    train = accepted.loc[accepted["winter"] == WINTERS[0]]
    counts = train.groupby(["station_id", "source_id", "sensor_id"]).size().rename("rows").reset_index()
    if counts.empty:
        raise ValueError("No accepted observations in training winter")
    choices = counts.sort_values(
        ["station_id", "rows", "source_id", "sensor_id"],
        ascending=[True, False, True, True]).drop_duplicates("station_id")
    accepted = accepted.merge(choices[["station_id", "source_id", "sensor_id"]], how="inner")
    keys = ["station_id", "time"]
    if accepted.duplicated(keys).any():
        raise ValueError("Ambiguous duplicate observations from the selected sensor")
    return accepted, choices


def parse_ascii(text: str, name: str) -> np.ndarray:
    """Parse a bounded DAP ASCII array, checking every row and declared shape."""
    if "---------------------------------------------" not in text:
        raise ValueError("Not a DAP ASCII response")
    body = text.split("---------------------------------------------", 1)[1]
    match = re.search(r"(?m)^" + re.escape(name) + r"((?:\[\d+\])+)\s*\n", body)
    if match is None:
        raise ValueError(f"Missing DAP array: {name}")
    shape = tuple(map(int, re.findall(r"\d+", match.group(1))))
    tail = body[match.end():].split("\n\n", 1)[0]
    if len(shape) == 1:
        tokens = tail.strip().split(",")
        if len(tokens) != shape[0]:
            raise ValueError(f"Invalid DAP vector length: {name}")
        array = np.array([float(value) for value in tokens], np.float64)
    else:
        array = np.full(shape, np.nan, np.float64)
        visited = set()
        for line in tail.splitlines():
            row = re.fullmatch(r"((?:\[\d+\])+),\s*(.*)", line.strip())
            if row is None:
                raise ValueError(f"Malformed DAP row: {name}")
            indices = tuple(map(int, re.findall(r"\d+", row.group(1))))
            values = [float(value) for value in row.group(2).split(",")]
            if (len(indices) != len(shape)-1 or indices in visited or
                    any(i >= bound for i, bound in zip(indices, shape[:-1])) or
                    len(values) != shape[-1]):
                raise ValueError(f"Invalid or duplicate DAP row: {name}")
            array[indices] = values
            visited.add(indices)
        if len(visited) != int(np.prod(shape[:-1])):
            raise ValueError(f"Incomplete DAP array: {name}")
    if not np.isfinite(array).all() or (np.abs(array) > 1e10).any():
        raise ValueError(f"Missing/nonfinite DAP values: {name}")
    return array


def minimax_heights(height: np.ndarray, valid: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Minimum attainable highest cell along four-neighbour land paths."""
    if height.shape != valid.shape or height.ndim != 2 or not np.isfinite(height[valid]).all():
        raise ValueError("Invalid graph terrain")
    if not valid[source]:
        raise ValueError("Graph source is not supported land")
    result = np.full(height.shape, np.inf)
    result[source] = height[source]
    queue = [(float(height[source]), *source)]
    while queue:
        peak, row, col = heapq.heappop(queue)
        if peak != result[row, col]:
            continue
        for r, c in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
            if not (0 <= r < height.shape[0] and 0 <= c < height.shape[1]) or not valid[r, c]:
                continue
            candidate = max(peak, float(height[r, c]))
            if candidate < result[r, c]:
                result[r, c] = candidate
                heapq.heappush(queue, (candidate, r, c))
    return result


def graph_features(height, valid, positions):
    count = len(positions)
    result = np.zeros((count, count, len(GRAPH_FIELDS)), np.float32)
    for source, position in enumerate(positions):
        passes = minimax_heights(height, valid, tuple(position))
        source_z = height[tuple(position)]
        for target, destination in enumerate(positions):
            peak = passes[tuple(destination)]
            if not np.isfinite(peak):
                continue
            target_z = height[tuple(destination)]
            steps = max(abs(destination-position))+1
            line = np.rint(np.linspace(position, destination, steps)).astype(int)
            straight_valid = bool(valid[line[:, 0], line[:, 1]].all())
            straight_rise = (max(0, height[line[:, 0], line[:, 1]].max()-max(source_z, target_z))
                             if straight_valid else 0)
            result[source, target] = [1, (peak-source_z)/1000,
                                      (peak-target_z)/1000, straight_rise/1000, int(straight_valid)]
    return result


def spatial_groups(stations):
    """Predeclared geographic challenge groups, not verified valley boundaries."""
    required = ("SN56960", "SN55820")
    if any(identifier not in set(stations["station_id"]) for identifier in required):
        raise ValueError("Both Haukedal and Fjærland–Bremuseet anchors need supported terrain")
    anchors = stations.set_index("station_id").loc[list(required), ["x", "y"]].to_numpy()
    xy = stations[["x", "y"]].to_numpy()
    distances = np.linalg.norm(xy[:, None]-anchors[None], axis=2)
    nearest = distances.argmin(axis=1)
    groups = np.zeros(len(stations), np.int32)
    near = distances.min(axis=1) <= 6000
    groups[near & (nearest == 0)] = 2  # Haukedal: final test.
    groups[near & (nearest == 1)] = 1  # Fjærland–Bremuseet: validation/calibration.
    if any((groups == split).sum() < 1 for split in (0, 1, 2)):
        raise ValueError("Empty geographic split")
    return groups


def conformal_radius(errors: np.ndarray, coverage: float = .9) -> float:
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or not np.isfinite(errors).all() or (errors < 0).any() or not 0 < coverage < 1:
        raise ValueError("Invalid conformal calibration errors")
    rank = int(np.ceil((len(errors)+1)*coverage))
    if rank > len(errors):
        raise ValueError("Too few independent calibration nights for finite interval")
    return float(np.partition(errors, rank-1)[rank-1])
