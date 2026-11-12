"""Tests for direct polygonization of mapped N50 depression contours."""
import unittest
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box

import potential_lakes_loops as loops


def split_rectangle(xmin, ymin, xmax, ymax):
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    return [LineString([start, end]) for start, end in zip(corners, corners[1:])]


class DepressionLoopTests(unittest.TestCase):
    def test_open_segments_form_nested_closed_depressions_with_depth_and_area(self):
        rows = []
        for elevation, bounds in ((100, (0, 0, 100, 100)),
                                  (80, (20, 20, 80, 80)),
                                  (60, (45, 45, 55, 55))):
            for line in split_rectangle(*bounds):
                rows.append({"objtype": "Høydekurve", "hoyde": elevation, "geometry": line})
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:25833")

        contours = loops.polygonize_contours(frame, min_loop_area_m2=0)
        families = loops.nest_contour_loops(contours)
        records, loop_records = loops.basin_records(
            contours, families, box(-10, -10, 110, 110), 0, 0,
        )

        self.assertEqual(len(contours), 3)
        self.assertEqual(len(records), 1)
        basin = records[0]
        self.assertEqual(basin["area_m2"], 10_000)
        self.assertEqual(basin["depth_span_lower_bound_m"], 40)
        self.assertAlmostEqual(basin["volume_above_lowest_contour_m3"], 173_000)
        self.assertEqual(basin["max_depth_assumed_m"], 60)
        self.assertAlmostEqual(basin["volume_floor_step_assumed_m3"], 76_000)
        self.assertEqual(basin["deepest_contour_x"], 50)
        self.assertEqual(basin["deepest_contour_y"], 50)
        self.assertEqual(len(loop_records), 3)

    def test_single_closed_contour_has_unknown_depth(self):
        rows = [{"objtype": "Forsenkningskurve", "hoyde": 40, "geometry": line}
                for line in split_rectangle(0, 0, 100, 100)]
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:25833")
        contours = loops.polygonize_contours(frame, min_loop_area_m2=0)
        families = loops.nest_contour_loops(contours)
        records, _ = loops.basin_records(
            contours, families, box(-10, -10, 110, 110), 0, 0,
        )

        self.assertEqual(len(records), 1)
        self.assertTrue(np.isnan(records[0]["depth_span_lower_bound_m"]))
        self.assertEqual(records[0]["max_depth_assumed_m"], 20)
        self.assertEqual(records[0]["volume_floor_step_assumed_m3"], 200_000)
        self.assertEqual(records[0]["depth_status"], "assumed one floor step below single contour")

    def test_single_ordinary_contour_is_not_reported_as_a_basin(self):
        rows = [{"objtype": "Høydekurve", "hoyde": 40, "geometry": line}
                for line in split_rectangle(0, 0, 100, 100)]
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:25833")
        contours = loops.polygonize_contours(frame, min_loop_area_m2=0)
        families = loops.nest_contour_loops(contours)
        records, _ = loops.basin_records(
            contours, families, box(-10, -10, 110, 110), 0, 0,
        )

        self.assertEqual(records, [])

    def test_depression_label_is_not_shared_by_other_loops_at_same_elevation(self):
        rows = [{"objtype": "Høydekurve", "hoyde": 40, "geometry": line}
                for line in split_rectangle(0, 0, 100, 100)]
        rows.append({"objtype": "Forsenkningskurve", "hoyde": 40,
                     "geometry": LineString([(0, 0), (100, 0)])})
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:25833")

        contours = loops.polygonize_contours(frame, min_loop_area_m2=0)
        self.assertEqual(len(contours), 1)
        self.assertEqual(contours[0]["source_types"], ["Høydekurve"])
        self.assertEqual(contours[0]["geometry"].area, 10_000)

        families = loops.nest_contour_loops(contours)
        records, _ = loops.basin_records(
            contours, families, box(-1, -1, 101, 101), 0, 0,
        )
        self.assertEqual(records, [])

    def test_contours_outside_norway_do_not_inflate_depth_span(self):
        rows = []
        for elevation, bounds in ((100, (0, 0, 100, 100)),
                                  (40, (85, 20, 95, 80))):
            for line in split_rectangle(*bounds):
                objtype = "Forsenkningskurve" if elevation == 100 else "Høydekurve"
                rows.append({"objtype": objtype, "hoyde": elevation, "geometry": line})
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:25833")
        contours = loops.polygonize_contours(frame, min_loop_area_m2=0)
        families = loops.nest_contour_loops(contours)
        records, _ = loops.basin_records(
            contours, families, box(-10, -10, 80, 110), 0, 0,
        )

        self.assertEqual(len(records), 1)
        self.assertTrue(np.isnan(records[0]["depth_span_lower_bound_m"]))
        self.assertEqual(records[0]["contour_level_count"], 1)

    def test_geopackage_writer_round_trips_without_fiona(self):
        rows = [{"basin_id": 1, "geometry": box(0, 0, 10, 10)}]
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:25833")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "basins.gpkg"
            loops.write_geopackage(frame, path, "basins")
            result = loops.pyogrio.read_dataframe(path, layer="basins")

        self.assertEqual(result.basin_id.tolist(), [1])
        self.assertEqual(result.geometry.iloc[0].area, 100)


if __name__ == "__main__":
    unittest.main()