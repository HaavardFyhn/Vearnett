"""Identify road-weather stations from their MET/Frost station names."""
import numpy as np
import pandas as pd


PLACEMENT_SUFFIXES = {"svv": "", "svv-met": "_svv_met", "all": "_all", "empty": "_empty"}
EMPTY_REFERENCE_RMSE = .5
PLACEMENT_LABELS = {"svv": "Vegværstasjoner", "svv-met": "Vegværstasjoner + MET-stasjoner",
                    "all": "Alle værstasjoner", "empty": "Tomt nett"}


def svv_mask(names):
    """Recognize leading Norwegian road references, not embedded place names."""
    names = pd.Series(names, dtype="string")
    return names.str.match(
        r"(?i)^\s*(?:EV?|RV?|FV?|KV?|PV?|SV?)\s*\d+(?=$|[\s:./_\-])",
        na=False,
    ).to_numpy(dtype=bool)


def pross_excluded(stations):
    return frozenset(np.flatnonzero(~svv_mask(stations["name"])).tolist())


def svv_met_excluded(stations, owner_catalog):
    if owner_catalog is None:
        raise ValueError("Stasjonseiere mangler for Vegværstasjoner + MET-stasjoner")
    met = stations["station_id"].astype(str).map(
        lambda identifier: "MET.NO" in owner_catalog["stations"].get(identifier, {}).get("owners", [])
    ).to_numpy(dtype=bool)
    return frozenset(np.flatnonzero(~(svv_mask(stations["name"]) | met)).tolist())
