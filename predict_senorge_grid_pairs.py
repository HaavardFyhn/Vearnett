"""Prediker RMSE for vilkårlige par av tilgjengelige 1 km-ruter, uten stasjoner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import train_senorge_grid_pairs as training
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd


def predict(model_dir: Path, pairs: pd.DataFrame) -> pd.DataFrame:
    summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["features"] != training.FEATURES:
        raise ValueError("Modellen er ufullstendig eller har uventet egenskapsskjema")
    if not {"grid_id_1", "grid_id_2"} <= set(pairs):
        raise ValueError("CSV må inneholde grid_id_1 og grid_id_2")
    grid = gpd.read_file(model_dir / "grid_features.gpkg")
    lookup = pd.Series(np.arange(len(grid)), index=grid.grid_id)
    positions = np.column_stack([pairs[key].map(lookup) for key in ("grid_id_1", "grid_id_2")])
    if not np.isfinite(positions).all():
        raise ValueError("Et eller flere grid_id finnes ikke i modellens rutenett")
    indices = positions.astype(int)
    features = training.pair_features(grid, indices)
    maximum = summary.get("maximum_distance_m", 50_000)
    if not features.distance_m.between(1000 - 1e-5, maximum).all():
        raise ValueError(f"Modellen er bare trent for ulike ruter med avstand 1–{maximum/1000:g} km")
    model = joblib.load(model_dir / "model.joblib")
    if model.n_features_in_ != len(training.FEATURES):
        raise ValueError("Modellens antall egenskaper stemmer ikke")
    result = pairs.copy()
    result["predicted_rmse_c"] = model.predict(features.to_numpy())
    result["distance_m"] = features.distance_m.to_numpy()
    for endpoint in (0, 1):
        for field in training.LAKE_FIELDS:
            result[f"{field}_{endpoint+1}"] = grid[field].to_numpy()[indices[:, endpoint]]
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=training.WORK / "senorge_grid_pairs_1m_basins")
    parser.add_argument("--pairs", type=Path, required=True, help="CSV med grid_id_1 og grid_id_2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = predict(args.model_dir, pd.read_csv(args.pairs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Lagret {len(result):,} parprediksjoner i {args.output}")


if __name__ == "__main__":
    main()
