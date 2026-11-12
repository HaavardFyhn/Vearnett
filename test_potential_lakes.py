"""Synthetic checks for the potential-lake Priority-Flood analysis."""
import unittest

import numpy as np
import pandas as pd
from rasterio.transform import from_origin

import potential_lakes as pl


class PotentialLakeTests(unittest.TestCase):
    def test_32v_test_center_transforms_to_project_crs(self):
        x, y = pl.transform_center((366850, 6540763), "EPSG:25832")
        self.assertAlmostEqual(x, 22767.043, places=2)
        self.assertAlmostEqual(y, 6568220.604, places=2)

    def test_outside_norway_is_zero_elevation_with_boundary_ring_as_outlet(self):
        dem = np.full((7, 7), 120.0, np.float32)
        valid = np.ones((7, 7), dtype=bool)
        norway = np.zeros((7, 7), dtype=bool)
        norway[2:5, 2:5] = True

        outlet = pl.prepare_ocean_outlets(dem, valid, norway)
        expected_outlet = ~norway & pl.ndimage.binary_dilation(
            norway, structure=np.ones((3, 3), dtype=bool),
        )
        np.testing.assert_array_equal(outlet, expected_outlet)
        self.assertTrue(np.all(dem[~norway] == 0))
        self.assertTrue(valid[outlet].all())
        self.assertFalse(valid[~norway & ~outlet].any())
        self.assertTrue(valid[norway].all())

    def test_sill_tolerance_groups_nearby_pools_without_changing_raw_basins(self):
        labels = np.zeros((7, 7), dtype=np.int32)
        labels[3, 2] = 1
        labels[3, 4] = 2
        dem = np.full((7, 7), 120.0, dtype=np.float32)
        dem[3, 3] = 115.0
        report = np.ones((7, 7), dtype=bool)
        basins = pd.DataFrame({
            "lake_id": [1, 2],
            "area_m2": [2500.0, 2500.0],
            "volume_m3": [10_000.0, 20_000.0],
            "max_depth_m": [4.0, 8.0],
            "spill_level_m": [100.0, 100.0],
        })

        mapping_20, grouped_20, groups_20, merged_20 = pl.group_basins_by_sill(
            labels.copy(), dem, report, basins, 20.0,
        )
        mapping_10, grouped_10, groups_10, merged_10 = pl.group_basins_by_sill(
            labels.copy(), dem, report, basins, 10.0,
        )

        self.assertEqual(merged_20, 1)
        self.assertEqual(mapping_20[1], mapping_20[2])
        self.assertEqual(len(groups_20), 1)
        self.assertEqual(groups_20.iloc[0].member_basin_count, 2)
        self.assertEqual(groups_20.iloc[0].group_volume_sum_m3, 30_000)
        self.assertEqual(merged_10, 0)
        self.assertNotEqual(mapping_10[1], mapping_10[2])
        self.assertEqual(len(groups_10), 2)

    def setUp(self):
        self.z = np.full((20, 30), 100., np.float32)
        self.z[3:10, 3:10] = 110
        self.z[5:8, 5:8] = 95
        self.z[12:18, 3:10] = 110
        self.z[14:17, 5:8] = 96
        self.z[15, 0:6] = 96
        self.z[3:10, 15:22] = 110
        self.z[5:8, 17:20] = 94
        self.valid = np.ones_like(self.z, bool)
        self.sea = np.zeros_like(self.z, bool)
        self.sea[0, 0] = True
        self.transform = from_origin(0, 1000, 50, 50)

    def run_case(self, min_depth=1, min_area=0, report=None):
        return pl.find_potential_lakes(self.z, self.valid, self.sea, self.transform, min_depth, min_area,
                                       report=report)

    def test_closed_bowl_depth_volume_and_spill_level(self):
        depth, labels, table = self.run_case()
        lake = table.loc[table.max_depth_m == 15].iloc[0]
        self.assertAlmostEqual(lake.max_depth_m, 15)
        self.assertAlmostEqual(lake.spill_level_m, 110)
        self.assertEqual(lake.area_m2, 9 * 2500)
        self.assertAlmostEqual(lake.volume_m3, 9 * 15 * 2500)
        self.assertEqual(depth[6, 6], 15)
        self.assertEqual(labels[6, 6], lake.lake_id)

    def test_basin_is_filled_without_any_lake_or_river_mask(self):
        depth, _, table = self.run_case()
        self.assertEqual(len(table), 3)
        self.assertAlmostEqual(depth[6, 18], 16)

    def test_basin_connected_to_sea_retains_only_to_spill_level(self):
        depth, labels, table = self.run_case()
        self.assertEqual(depth[15, 6], 4)
        self.assertGreater(labels[15, 6], 0)
        self.assertTrue(np.any(table.spill_level_m == 100))

    def test_sea_is_the_only_outlet_even_at_raster_edge(self):
        self.sea[:] = False
        self.sea[19, 29] = True
        self.z[:] = 100
        self.z[:5, :5] = 110
        self.z[1:3, 1:3] = 95
        depth, _, table = self.run_case()
        self.assertEqual(depth[1, 1], 15)
        self.assertEqual(len(table), 1)

    def test_sea_cell_drains_its_basin(self):
        self.sea[6, 6] = True
        depth, _, table = self.run_case()
        self.assertEqual(depth[5, 5], 0)
        self.assertEqual(len(table), 2)

    def test_reporting_mask_and_size_filters(self):
        report = np.ones_like(self.valid)
        report[:, :12] = False
        self.assertEqual(len(self.run_case(report=report)[2]), 1)
        self.assertEqual(len(self.run_case(min_depth=20)[2]), 0)
        self.assertEqual(len(self.run_case(min_area=30_000)[2]), 1)


if __name__ == "__main__":
    unittest.main()
