import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from station_app import Scenario
from station_owners import OWNER_FILE, load_catalog, owner_counts, refresh
from test_station_sensor_proposals import SensorEngine


class StationOwnersTests(unittest.TestCase):
    def test_owner_counts_deduplicate_station_sensors_and_coowners(self):
        counts = owner_counts([
            dict(station_id="SN1", owners=["A", "A", "B"]),
            dict(station_id="SN1", owners=["A"]),
            dict(station_id="SN2", owners=["B"]),
            dict(station_id="SN3", owners=[]),
            dict(station_id="SN4", owners=["A"], removed=True),
            dict(owners=["A"]),
        ])
        self.assertEqual(counts, dict(stations=3, unknown=1, owners=[
            dict(name="B", count=2), dict(name="A", count=1)]))

    def test_refresh_persists_holders_and_csv_without_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / OWNER_FILE
            with patch.dict(os.environ, {"OWNER_TEST_CLIENT": "not-a-real-credential"}), \
                    patch("pipeline_config.FROST_CLIENT_ID_ENV", "OWNER_TEST_CLIENT"), \
                    patch("fetch_stations._request_all", return_value=[
                        dict(id="SN1", name="First", stationHolders=["B", "A", "A"]),
                        dict(id="SN2", name="Second")]) as request:
                refresh(path)
                self.assertEqual(request.call_args.args[1]["fields"], "id,name,stationHolders")
            catalog = load_catalog(Path(directory))
            self.assertEqual(catalog["stations"]["SN1"]["owners"], ["A", "B"])
            self.assertNotIn("not-a-real-credential", path.read_text(encoding="utf-8"))
            with path.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(list(csv.reader(handle, delimiter=";"))[-1], ["Ikke oppgitt", "1"])
            previous = path.read_bytes()
            with patch.dict(os.environ, {"OWNER_TEST_CLIENT": "test"}), \
                    patch("pipeline_config.FROST_CLIENT_ID_ENV", "OWNER_TEST_CLIENT"), \
                    patch("fetch_stations._request_all", return_value=[]):
                with self.assertRaisesRegex(ValueError, "no Norwegian stations"):
                    refresh(path)
            self.assertEqual(path.read_bytes(), previous)

    def test_missing_and_invalid_catalog_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_catalog(directory))
            path = Path(directory) / OWNER_FILE
            path.parent.mkdir()
            path.write_text(json.dumps(dict(schema=1, updated="today", stations={"SN1": {"owners": "A"}})))
            with self.assertRaises(ValueError):
                load_catalog(directory)

    def test_owners_follow_moved_station_and_removed_station_is_not_counted(self):
        engine = SensorEngine()
        catalog = dict(updated="2026-11-11", stations={"SN1": dict(owners=["A", "B"]),
                                                      "SN2": dict(owners=["A"])})
        with patch.object(engine, "station_owner_catalog", return_value=catalog, create=True):
            scenario = Scenario(engine)
            initial = scenario.state()
            self.assertTrue(initial["station_owners"]["available"])
            self.assertEqual(initial["station_owners"]["owners"][0], dict(name="A", count=2))
            with patch.object(engine, "snap", return_value=7), patch.object(engine, "station", return_value=dict(
                    id=7, x=7000, y=0, lon=13, lat=64, elevation=15)):
                scenario.apply(dict(action="move_frost", station_id="SN1", lon=13, lat=64))
                marker = next(item for item in scenario.state()["fixed_stations"] if item["station_id"] == "SN1")
                self.assertEqual(marker["owners"], ["A", "B"])
                self.assertIn("original_position", marker)
            scenario.apply(dict(action="remove_frost", station_id="SN2"))
            summary = scenario.state()["station_owners"]
            self.assertEqual(summary["owners"], [dict(name="A", count=1), dict(name="B", count=1)])


if __name__ == "__main__":
    unittest.main()
