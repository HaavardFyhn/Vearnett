"""Focused checks for contour-derived potential-lake terrain."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString

import potential_lakes as pl
import potential_lakes_contours as plc


class ContourInterpolationTests(unittest.TestCase):
    def test_completed_contour_tile_is_reused_on_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            n50 = root / "n50.gdb"
            norway = root / "norway.gpkg"
            n50.write_bytes(b"contour source")
            norway.write_bytes(b"norway boundary")
            args = SimpleNamespace(
                cell_m=10.0,
                tile_cells=2,
                max_gap_m=5.0,
                n50=n50,
                norway=norway,
                out_dir=root / "out",
            )
            transform = from_origin(0, 20, 10, 10)
            expected = np.full((2, 2), 123, dtype=np.float32)
            with patch.object(plc, "interpolate_contour_tile", return_value=expected) as interpolate:
                first = plc.build_contour_surface(args, (2, 2), transform, (0, 0, 20, 20))
                interpolate.assert_called_once()
            np.testing.assert_array_equal(first, expected)
            first._mmap.close()

            with patch.object(plc, "interpolate_contour_tile", side_effect=AssertionError("tile should resume")):
                resumed = plc.build_contour_surface(args, (2, 2), transform, (0, 0, 20, 20))
            np.testing.assert_array_equal(resumed, expected)
            resumed._mmap.close()

    def test_default_grid_is_50m_for_national_screening(self):
        args = plc.parse_args([])
        self.assertEqual(args.cell_m, 50.0)
        self.assertEqual(args.tile_cells, 1000)
        self.assertEqual(args.max_gap_m, 15_000.0)

    def test_outside_norway_is_the_assumed_ocean_outlet(self):
        norway = np.array([[True, True, False], [True, False, False]], dtype=bool)
        np.testing.assert_array_equal(plc.assumed_sea_mask(norway), ~norway)

    def test_centerlines_anchor_interpolated_elevation_and_xy_grid(self):
        contours = [
            LineString([(50, 450), (450, 450)]),
            LineString([(50, 50), (450, 50)]),
        ]
        transform = from_origin(0, 500, 100, 100)
        surface = plc.interpolate_contour_geometries(
            contours, np.array([20, 40], dtype=np.float32), (5, 5), transform, 5,
        )
        np.testing.assert_allclose(surface[:, 2], [20, 25, 30, 35, 40], atol=2)
        self.assertEqual(rasterio_xy(transform, 2, 2), (250.0, 250.0))

    def test_progress_counter_reports_processed_cells(self):
        dem = np.full((8, 8), 10, dtype=np.float32)
        valid = np.ones((8, 8), dtype=bool)
        sea = np.zeros((8, 8), dtype=bool)
        sea[0, 0] = True
        counter = np.zeros(1, dtype=np.int64)
        processed = pl.priority_flood(dem, valid, sea, counter)
        self.assertEqual(counter[0], processed)
        self.assertEqual(processed, dem.size)


def rasterio_xy(transform, row, col):
    from rasterio.transform import xy
    return xy(transform, row, col)


if __name__ == "__main__":
    unittest.main()