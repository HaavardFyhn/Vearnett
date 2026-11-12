"""Checks for the disk-backed nationwide DTM10 workflow."""
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

import potential_lakes as base
import potential_lakes_dtm10_norway as dtm10


class Dtm10DiskWorkflowTests(unittest.TestCase):
    def test_existing_mosaic_is_validated_and_reused_by_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mosaic.tif"
            with rasterio.open(path, "w", driver="GTiff", width=4, height=3, count=1,
                               dtype="float32", crs="EPSG:25833",
                               transform=from_origin(500000, 6500000, 10, 10)) as dataset:
                dataset.write(np.ones((3, 4), dtype=np.float32), 1)

            mosaic_path, bounds, shape = dtm10.inspect_mosaic(path)

            self.assertEqual(mosaic_path, path.resolve())
            self.assertEqual(bounds, (500000.0, 6499970.0, 500040.0, 6500000.0))
            self.assertEqual(shape, (3, 4))

    def test_monitor_reports_zero_progress_for_tiled_flood(self):
        class StopAfterFirstPoll:
            calls = 0

            def wait(self, _):
                self.calls += 1
                return self.calls > 1

        output = StringIO()
        with redirect_stdout(output):
            dtm10.monitor_flood(
                np.zeros(3, dtype=np.int64), 3_250_441_586,
                time.perf_counter(), StopAfterFirstPoll(),
            )
        self.assertIn("Priority-Flood chunks", output.getvalue())

    def test_server_axis_order_variant_is_accepted_only_for_utm33_grs80(self):
        self.assertTrue(dtm10.is_supported_dtm_crs(CRS.from_epsg(25833)))
        self.assertTrue(dtm10.is_supported_dtm_crs(CRS.from_wkt(CRS.from_epsg(25833).to_wkt())))
        self.assertTrue(dtm10.is_supported_dtm_crs(CRS.from_epsg(3045)))
        self.assertFalse(dtm10.is_supported_dtm_crs(CRS.from_epsg(25832)))
        self.assertFalse(dtm10.is_supported_dtm_crs(CRS.from_epsg(32633)))
        self.assertFalse(dtm10.is_supported_dtm_crs(None))

    def test_disk_backed_flood_matches_existing_priority_flood(self):
        elevation = np.array([
            [8, 8, 8, 8, 8, 8],
            [8, 3, 3, 8, 4, 8],
            [8, 3, 1, 7, 2, 8],
            [8, 8, 8, 8, 8, 8],
        ], dtype=np.float32)
        valid = np.ones(elevation.shape, dtype=bool)
        valid[0, 5] = False
        outlets = np.zeros(elevation.shape, dtype=bool)
        outlets[0, 0] = True
        expected = np.where(valid, elevation, base.NODATA).astype(np.float32)
        expected_count = base.priority_flood(expected, valid, outlets)

        with tempfile.TemporaryDirectory() as directory:
            actual = np.memmap(Path(directory) / "filled.f32", mode="w+",
                               dtype=np.float32, shape=elevation.shape)
            state = np.memmap(Path(directory) / "state.u8", mode="w+",
                              dtype=np.uint8, shape=elevation.shape)
            try:
                actual[:] = expected
                state[:] = 0
                state[~valid] = dtm10.INVALID
                state[outlets] = dtm10.OUTLET
                actual_count = dtm10.priority_flood_disk(actual, state)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual_count, expected_count)
            finally:
                actual._mmap.close()
                state._mmap.close()

    def test_tiled_flood_matches_reference_for_multi_outlet_chunk_boundaries(self):
        rng = np.random.default_rng(42)
        elevation = rng.integers(0, 40, size=(23, 29)).astype(np.float32)
        valid = np.ones(elevation.shape, dtype=bool)
        valid[5:8, 8:12] = False
        outlets = np.zeros(elevation.shape, dtype=bool)
        outlets[0, 0] = True
        outlets[-1, -1] = True
        outlets[12, 14] = True
        expected = np.where(valid, elevation, base.NODATA).astype(np.float32)
        expected_count = base.priority_flood(expected, valid, outlets)

        for chunk_size in (1, 4, 7, 16):
            with self.subTest(chunk_size=chunk_size):
                state = np.zeros(elevation.shape, dtype=np.uint8)
                state[~valid] = dtm10.INVALID
                state[outlets] = dtm10.OUTLET
                filled = np.full(elevation.shape, np.inf, dtype=np.float32)

                resolved, tile_runs = dtm10.priority_flood_tiled(
                    elevation, filled, state, chunk_size,
                )

                np.testing.assert_array_equal(filled[valid], expected[valid])
                self.assertEqual(resolved, expected_count)
                self.assertGreater(tile_runs, 0)
                self.assertTrue(np.all(state[valid & ~outlets] == dtm10.LAND_CLOSED))

    def test_tiled_flood_matches_reference_for_randomized_surfaces(self):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            elevation = rng.integers(0, 100, size=(11 + seed, 15 + seed)).astype(np.float32)
            valid = rng.random(elevation.shape) >= 0.04
            outlets = np.zeros(elevation.shape, dtype=bool)
            outlets[0, 0] = True
            outlets[-1, -1] = True
            outlets[elevation.shape[0] // 2, elevation.shape[1] // 2] = True
            valid[outlets] = True
            expected = np.where(valid, elevation, base.NODATA).astype(np.float32)
            expected_count = base.priority_flood(expected, valid, outlets)

            for chunk_size in (1, 3, 6):
                with self.subTest(seed=seed, chunk_size=chunk_size):
                    state = np.zeros(elevation.shape, dtype=np.uint8)
                    state[~valid] = dtm10.INVALID
                    state[outlets] = dtm10.OUTLET
                    filled = np.full(elevation.shape, np.inf, dtype=np.float32)
                    resolved, _ = dtm10.priority_flood_tiled(
                        elevation, filled, state, chunk_size,
                    )
                    np.testing.assert_array_equal(filled[valid], expected[valid])
                    self.assertEqual(resolved, expected_count)

    def test_sill_grouping_uses_original_dem_elevation(self):
        labels = np.zeros((7, 7), dtype=np.int32)
        labels[3, 2] = 1
        labels[3, 4] = 2
        original_dem = np.full((7, 7), 120, dtype=np.float32)
        original_dem[3, 3] = 115
        state = np.full((7, 7), dtm10.LAND_CLOSED, dtype=np.uint8)
        basins = pd.DataFrame({
            "lake_id": [1, 2],
            "area_m2": [2500.0, 2500.0],
            "volume_m3": [10_000.0, 20_000.0],
            "max_depth_m": [4.0, 8.0],
            "spill_level_m": [100.0, 100.0],
        })

        _, _, groups, merged = dtm10.group_basins(labels, original_dem, state, basins, 20.0)

        self.assertEqual(merged, 1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups.iloc[0].member_basin_count, 2)


if __name__ == "__main__":
    unittest.main()