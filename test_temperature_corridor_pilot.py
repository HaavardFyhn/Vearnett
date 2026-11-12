import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from rasterio.transform import from_origin

from temperature_corridor_pilot import (
    DtmTileSampler,
    corridor_profile_features,
    raster_corridor_profile_features,
    sample_pairs_by_distance,
    spatial_holdout_indices,
)


class TemperatureCorridorPilotTests(unittest.TestCase):
    def test_distance_weighted_sample_selects_shorter_pairs_more_often(self):
        pairs = pd.DataFrame({
            "distance_m": np.linspace(10_000, 1_000_000, 10_000),
            "pair_id": np.arange(10_000),
        })

        selected = sample_pairs_by_distance(pairs, max_pairs=1_000,
                                            distance_scale_km=300, random_state=42)

        self.assertEqual(len(selected), 1_000)
        self.assertEqual(selected.pair_id.nunique(), 1_000)
        self.assertLess(selected.distance_m.median(), pairs.distance_m.median())

    def test_profile_detects_ridge_above_endpoint_elevation_line(self):
        terrain_xy = np.array([[0, 0], [1000, 0], [2000, 0], [3000, 0], [4000, 0]])
        elevation = np.array([0, 0, 100, 0, 0], dtype=np.float32)

        result = corridor_profile_features(
            (0, 0), (4000, 0), 0, 0, cKDTree(terrain_xy), elevation,
            sample_step_m=1000, max_snap_m=100,
        )

        self.assertAlmostEqual(result["corridor_ridge_above_line_m"], 100)
        self.assertEqual(result["corridor_valley_below_line_m"], 0)
        self.assertAlmostEqual(result["corridor_coverage_fraction"], 1)

    def test_profile_reports_missing_corridor_coverage(self):
        terrain_xy = np.array([[0, 0], [4000, 0]])
        elevation = np.array([0, 0], dtype=np.float32)

        result = corridor_profile_features(
            (0, 0), (4000, 0), 0, 0, cKDTree(terrain_xy), elevation,
            sample_step_m=1000, max_snap_m=100,
        )

        self.assertEqual(result["corridor_coverage_fraction"], 0)
        self.assertEqual(result["corridor_ridge_above_line_m"], 0)

    def test_spatial_holdout_keeps_heldout_stations_out_of_training(self):
        stations = pd.DataFrame({
            "x": [10_000, 20_000, 210_000, 220_000],
            "y": [10_000, 20_000, 210_000, 220_000],
        }, index=["A", "B", "C", "D"])
        pairs = pd.DataFrame({
            "station_1": ["A", "A", "B", "C", "C", "D"],
            "station_2": ["B", "C", "C", "D", "A", "B"],
        })

        train_ids, test_ids, heldout, _ = spatial_holdout_indices(
            pairs, stations, test_fraction=0.5, block_size_m=100_000,
        )

        train_stations = set(pairs.iloc[train_ids][["station_1", "station_2"]].to_numpy().ravel())
        test_stations = set(pairs.iloc[test_ids][["station_1", "station_2"]].to_numpy().ravel())
        self.assertTrue(train_stations.isdisjoint(heldout))
        self.assertTrue(test_stations.intersection(heldout))

    def test_dtm_sampler_reads_offset_tiles_and_profile_ridge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tile_paths = [root / "west.tif", root / "east.tif"]
            arrays = [np.zeros((2, 2), dtype=np.float32),
                      np.full((2, 2), 100, dtype=np.float32)]
            arrays[0][1, 1] = -32767
            for index, (path, values) in enumerate(zip(tile_paths, arrays)):
                with rasterio.open(
                    path, "w", driver="GTiff", width=2, height=2, count=1,
                    dtype="float32", crs="EPSG:25833",
                    transform=from_origin(index * 25, 20, 10, 10), nodata=-32767,
                ) as target:
                    target.write(values, 1)

            with DtmTileSampler(tile_paths) as sampler:
                result = raster_corridor_profile_features(
                    (0, 5), (45, 5), 0, 0, sampler, sample_step_m=10,
                )
                nodata_sample = sampler.sample(np.array([[15, 5]]))
            self.assertAlmostEqual(result["corridor_ridge_above_line_m"], 100)
            self.assertGreater(result["corridor_coverage_fraction"], 0)
            self.assertTrue(np.isnan(nodata_sample[0]))


if __name__ == "__main__":
    unittest.main()