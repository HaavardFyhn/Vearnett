"""Identify road-weather stations from their MET/Frost station names."""
import numpy as np
import pandas as pd


def svv_mask(names):
    """Recognize leading Norwegian road references, not embedded place names."""
    names = pd.Series(names, dtype="string")
    return names.str.match(
        r"(?i)^\s*(?:EV?|RV?|FV?|KV?|PV?|SV?)\s*\d+(?=$|[\s:./_\-])",
        na=False,
    ).to_numpy(dtype=bool)


def pross_excluded(stations):
    return frozenset(np.flatnonzero(~svv_mask(stations["name"])).tolist())
