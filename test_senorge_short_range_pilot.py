import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from senorge_short_range_pilot import (
    _pair_regression_rmse,
    sample_virtual_pairs,
)
import overnight_senorge_stationlab as overnight


class SenorgeShortRangePilotTests(unittest.TestCase):
    def test_overnight_resume_reuses_profile_and_runs_only_requested_precompute(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "temperature_5y_prediction_rmse_senorge"
            for suffix in ("_model.joblib", "_metadata.json", "_station_features.gpkg"):
                prefix.with_name(prefix.name + suffix).touch()
            args = overnight.parse_args([
                "--skip-training", "--skip-grunnett", "--no-launch",
                "--log", str(Path(directory) / "night.log"),
            ])
            with patch.object(overnight, "STATIONLAB_PREFIX", prefix), \
                    patch.object(overnight.precompute_frost, "main", return_value=0) as precompute, \
                    patch.object(overnight, "KuldehullLayer") as cold_basins, \
                    patch.object(overnight, "run_pilot", side_effect=AssertionError("Training should be skipped")):
                self.assertEqual(overnight.run(args), 0)
            cold_basins.return_value.prepare.assert_called_once()
            precompute.assert_called_once_with([
                "--phenomenon", "temperature", "--cell-m", "1000", "--kind", "senorge",
            ])

    def test_virtual_pairs_are_local_and_include_current_terrain_features(self):
        xs, ys = np.meshgrid(np.arange(8) * 1000.0, np.arange(8) * 1000.0)
        grid = pd.DataFrame({
            "x": xs.ravel(), "y": ys.ravel(),
            "elevation": np.arange(xs.size, dtype=float),
            "coast_dist_fjord": np.full(xs.size, 1000.0),
            "coast_dist_ocean": np.full(xs.size, 2000.0),
            "objtype": np.where(xs.ravel() < 4000, "forest", "mountain"),
            "relief_10km": np.arange(xs.size, dtype=float),
            "rugstd_10km": np.arange(xs.size, dtype=float) / 10,
            "tpi_10km": np.zeros(xs.size),
            "relief_25km": np.arange(xs.size, dtype=float) * 2,
            "rugstd_25km": np.arange(xs.size, dtype=float) / 20,
            "tpi_25km": np.zeros(xs.size),
        })

        pairs = sample_virtual_pairs(
            grid, max_pairs=40, min_distance_m=1000, max_distance_m=5000,
            distance_scale_m=2000, random_state=4,
        )

        self.assertEqual(len(pairs), 40)
        self.assertTrue(pairs.distance_m.between(1000, 5000).all())
        self.assertIn("relief_diff_10km", pairs)
        self.assertIn("same_objtype", pairs)

    def test_virtual_pair_rmse_matches_two_direction_ols(self):
        source = np.arange(20, dtype=np.float32)
        target = 3 * source + np.sin(source)
        series = np.stack([source, target], axis=1)[:, None, :]

        labels, split = _pair_regression_rmse(series, 0.8)

        train_source, train_target = source[:split], target[:split]
        valid_source, valid_target = source[split:], target[split:]
        forward = np.polyfit(train_source, train_target, 1)
        reverse = np.polyfit(train_target, train_source, 1)
        forward_rmse = np.sqrt(np.mean((valid_target - np.polyval(forward, valid_source)) ** 2))
        reverse_rmse = np.sqrt(np.mean((valid_source - np.polyval(reverse, valid_target)) ** 2))
        self.assertAlmostEqual(labels[0], (forward_rmse + reverse_rmse) / 2, places=5)


if __name__ == "__main__":
    unittest.main()