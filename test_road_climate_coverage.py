import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyogrio
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation, label
from shapely.geometry import LineString, box

from road_climate_coverage import (FIELDS, climate_occurrences, export_lines, klassifiser_score, model_stations,
                                   occurrence_polygons, score_occurrences, station_occurrences)
from station_weather import PHENOMENON_DATASETS


class CoverageTests(unittest.TestCase):
    def test_classification_includes_notebook_boundaries_and_accepts_changed_limits(self):
        scores = [0, 20, 20.1, 40, 40.1, 60, 60.1, 80, 80.1, 100]
        self.assertEqual([klassifiser_score(score)[0] for score in scores],
                         [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        frame = pd.DataFrame([[1, 1, 1, 20, 20, 0, 0]], columns=FIELDS)
        stations = pd.DataFrame(dict(ForekomstID=[1], Overgangssone=[False], KobletTilForekomst=[True]))
        with patch("road_climate_coverage.SCORE_THRESHOLDS", (10, 30, 60, 90)), \
                patch("road_climate_coverage.STATUS", ["a", "b", "c", "d", "e"]):
            _, result = score_occurrences(frame, stations)
            self.assertEqual(result.Kandidatscore.iloc[0], 20)
            self.assertEqual(result.Kandidatklasse.iloc[0], 2)
            self.assertEqual(result.Status.iloc[0], "b")
        with patch("road_climate_coverage.SCORE_THRESHOLDS", (20, 20, 60, 80)):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                score_occurrences(frame, stations)

    def test_raster_edges_do_not_create_wrapped_climate_boundaries(self):
        climate = np.zeros((8, 8), dtype=np.uint8)
        climate[3, 0] = 1
        climate[3, -1] = 2
        _, transition, _, _ = climate_occurrences(climate, np.ones_like(climate))
        self.assertFalse(transition.any())

    def test_components_and_dilation_match_notebook_loops(self):
        climate = np.zeros((12, 14), dtype=np.uint8)
        climate[1:6, 1:6] = 1
        climate[6:11, 6:11] = 1  # Diagonally connected by the notebook's 8-neighbour rule.
        climate[1:7, 7:12] = 2
        climate[0, 0] = 8
        roads = np.zeros_like(climate)
        roads[2, 2:6] = 1
        roads[8, 8] = 1
        roads[3, 8:11] = 1
        actual, transition, mask, frame = climate_occurrences(climate, roads)
        valid = (climate >= 1) & (climate <= 8)
        boundary = np.zeros_like(valid)
        for r in range(climate.shape[0]):
            for c in range(climate.shape[1]):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < climate.shape[0] and 0 <= nc < climate.shape[1]
                                and valid[r, c] and valid[nr, nc] and climate[r, c] != climate[nr, nc]):
                            boundary[r, c] = True
        reference_transition = binary_dilation(boundary, structure=np.ones((3, 3)), iterations=2) & valid
        expected = np.zeros_like(actual)
        records = []
        identifier = 0
        for zone in range(1, 9):
            components, count = label(climate == zone, structure=np.ones((3, 3)))
            for component in range(1, count + 1):
                part = components == component
                road_cells = int((part & mask).sum())
                if road_cells == 0:
                    continue
                identifier += 1
                expected[part] = identifier
                trans = int((part & mask & reference_transition).sum())
                records.append([identifier, zone, component, int(part.sum()), road_cells, trans, trans / road_cells])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(transition, reference_transition)
        pd.testing.assert_frame_equal(frame, pd.DataFrame(records, columns=FIELDS))

    def test_score_cohort_percentile_ties_and_zero_station_rule(self):
        frame = pd.DataFrame([
            [1, 1, 1, 50, 20, 10, .5],
            [2, 2, 1, 80, 40, 10, .25],
            [3, 3, 1, 80, 40, 0, 0],
            [4, 4, 1, 19, 19, 19, 1],
        ], columns=FIELDS)
        stations = pd.DataFrame(dict(ForekomstID=[2, 2, 4], Overgangssone=[True, False, True],
                                     KobletTilForekomst=[True] * 3))
        all_rows, result = score_occurrences(frame, stations)
        self.assertEqual(result.ForekomstID.tolist(), [1, 2, 3])
        np.testing.assert_allclose(result.Kandidatscore, [71.7, 16.7, 66.7])
        self.assertEqual(result.Kandidatklasse.tolist(), [4, 1, 4])
        self.assertEqual(result.UtenStasjon.tolist(), [True, False, True])
        self.assertFalse(all_rows.loc[3, "Relevant"])
        stations["KobletTilForekomst"] = False
        _, empty_network = score_occurrences(frame, stations)
        # Preserve the notebook's perhaps surprising zero-global-density convention.
        self.assertTrue((empty_network.Dekningsgrad == 1).all())
        self.assertTrue((empty_network.DekningsmangelPoeng == 0).all())

    def test_station_uses_original_coordinates_not_model_cell_and_counts_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            profiles = {}
            for number, dataset in enumerate(PHENOMENON_DATASETS.values()):
                path = root / f"profile_{number}.joblib"
                frame = pd.DataFrame(dict(station_id=["SN1", "SN2", "SN3"],
                    name=["A", "B", "C"], lon=[10., 11., 12.], lat=[63., 64., 65.],
                    x=[0., 0., 0.], y=[0., 0., 0.]))
                joblib.dump(dict(stations=frame), path)
                profiles[dataset] = dict(profile=path.name)
            (root / "outputs" / "portable_weather.json").write_text(json.dumps(dict(profiles=profiles)))
            catalog = dict(stations={"SN1": dict(owners=["STATENS VEGVESEN"]),
                                     "SN2": dict(owners=["MET.NO", "OTHER"]),
                                     "SN3": dict(owners=["MET HAV OG IS"])})
            selected, _ = model_stations(root, catalog)
            self.assertEqual(selected.station_id.tolist(), ["SN1", "SN2"])
            self.assertEqual(selected.lon.tolist(), [10., 11.])
            self.assertTrue(selected.datasets.str.count(";").eq(5).all())
            keys = list(profiles)
            metadata = {key: profiles.pop(key) for key in keys[1:]}
            (root / "outputs" / "portable_weather.json").write_text(json.dumps(
                dict(profiles=profiles, station_profiles=metadata)))
            compact, _ = model_stations(root, catalog)
            pd.testing.assert_frame_equal(compact, selected)

    def test_station_assignment_handles_outside_and_no_road_occurrences(self):
        climate = np.ones((3, 3), dtype=np.uint8)
        occurrences = np.ones((3, 3), dtype=np.int32)
        occurrences[0, 0] = 0
        stations = pd.DataFrame(dict(lon=[.5, 1.5, 10.], lat=[2.5, 1.5, 10.]))
        result = station_occurrences(stations, climate, occurrences, np.zeros((3, 3), dtype=bool),
                                     from_origin(0, 3, 1, 1), 4326)
        self.assertEqual(result.KobletTilForekomst.tolist(), [False, True, False])

    def test_single_relevant_occurrence_retains_notebook_full_percentile(self):
        frame = pd.DataFrame([[1, 1, 1, 20, 20, 0, 0]], columns=FIELDS)
        stations = pd.DataFrame(dict(ForekomstID=[1], Overgangssone=[False], KobletTilForekomst=[True]))
        _, result = score_occurrences(frame, stations)
        self.assertEqual(result.VegnettBetydningPoeng.iloc[0], 20)
        self.assertEqual(result.Kandidatscore.iloc[0], 20)
        self.assertEqual(result.Kandidatklasse.iloc[0], 1)

    def test_actual_lines_are_split_and_retain_missing_station_and_score_attributes(self):
        climate_ids = np.array([[1, 2], [1, 2]], dtype=np.int32)
        relevant = pd.DataFrame(dict(ForekomstID=[1, 2], Klimasone=[1, 2], AntallStasjoner=[0, 1],
                                     Kandidatscore=[90., 20.], UtenStasjon=[True, False]))
        polygons = occurrence_polygons(climate_ids, np.ones((2, 2), dtype=bool), relevant,
                                        from_origin(0, 500, 250, 250), 25833)
        road = gpd.GeoDataFrame(dict(edge_id=["a"], roadid=["Ev6"]),
            geometry=[LineString([(-50, 250), (550, 250)])], index=[15], crs=25833)
        network = SimpleNamespace(metadata={"classes": ["1"]}, geographic_bounds=[0, 0, 1, 1],
            attributes={"fid": np.array([15])}, matching_rows=lambda *args: np.array([0]),
            read_selected=lambda *args: road)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.gpkg"
            summary = export_lines(network, polygons, path)
            scored = pyogrio.read_dataframe(path, layer="scored_roads")
            uncovered = pyogrio.read_dataframe(path, layer="roads_without_station")
            self.assertEqual(summary, dict(scored_lines=2, lines_without_station=1))
            self.assertTrue(scored.geometry.geom_type.eq("LineString").all())
            self.assertAlmostEqual(scored.geometry.length.sum(), 500)
            self.assertEqual(uncovered.ForekomstID.tolist(), [1])
            self.assertEqual(uncovered.Kandidatscore.tolist(), [90])
            self.assertEqual(scored.source_fid.tolist(), [15, 15])
            self.assertEqual(scored.crs.to_epsg(), 25833)


if __name__ == "__main__":
    unittest.main()
