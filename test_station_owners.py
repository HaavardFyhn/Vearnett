import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd

from station_app import Scenario, create_app
from station_owners import OWNER_FILE, load_catalog, owner_counts, refresh
from station_pross import svv_met_excluded
from test_station_sensor_proposals import SensorEngine


class StationOwnersTests(unittest.TestCase):
    def test_combined_membership_uses_exact_owner_and_keeps_roads_without_owner_data(self):
        stations = pd.DataFrame(dict(station_id=["SN1", "SN2", "SN3", "SN4", "SN5"],
            name=["E6 Road", "Private name", "MET.NO in name", "Unknown", "Fv7 Road"]),
            index=[10, 20, 30, 40, 50])
        catalog = dict(stations={"SN2": dict(owners=["OTHER", "MET.NO"]),
                                 "SN3": dict(owners=["MET"]),
                                 "SN5": dict(owners=["MET.NO", "MET.NO"])})
        self.assertEqual(svv_met_excluded(stations, catalog), frozenset([2, 3]))
        with self.assertRaisesRegex(ValueError, "Stasjonseiere mangler"):
            svv_met_excluded(stations, None)

    def test_app_combined_network_preserves_other_sensors_and_can_switch_back(self):
        engine = SensorEngine()
        catalog = dict(updated="today", stations={"SN2": dict(owners=["MET.NO"])})
        with patch.object(engine, "station_owner_catalog", return_value=catalog, create=True):
            client = create_app(engine).test_client()
            combined = client.post("/api/action", json=dict(action="network", value="svv_met"))
            self.assertEqual(combined.status_code, 200, combined.json)
            self.assertTrue(combined.json["include_met"])
            self.assertTrue(combined.json["pross_only"])
            self.assertEqual({row["station_id"] for row in combined.json["fixed_stations"]}, {"SN2", "SN3"})
            snow = next(row for row in combined.json["fixed_stations"] if row["station_id"] == "SN3")
            self.assertEqual(snow["sensors"], ["snow"])
            saved = client.get("/api/scenario").json
            all_stations = client.post("/api/action", json=dict(action="network", value="frost", pross_only=False))
            self.assertFalse(all_stations.json["include_met"])
            self.assertEqual({row["station_id"] for row in all_stations.json["fixed_stations"]}, {"SN1", "SN2", "SN3"})
            restored = client.post("/api/scenario", json=saved)
            self.assertTrue(restored.json["include_met"])
            self.assertEqual({row["station_id"] for row in restored.json["fixed_stations"]}, {"SN2", "SN3"})

    def test_missing_owners_rejects_combined_network_without_changing_scenario(self):
        engine = SensorEngine()
        with patch.object(engine, "station_owner_catalog", return_value=None, create=True):
            client = create_app(engine).test_client()
            before = client.get("/api/scenario").json
            response = client.post("/api/action", json=dict(action="network", value="svv_met"))
            self.assertEqual(response.status_code, 400)
            self.assertIn("Stasjonseiere mangler", response.json["error"])
            self.assertEqual(client.get("/api/scenario").json, before)

    def test_combined_membership_also_filters_metadata_without_a_ready_surface(self):
        engine = SensorEngine()
        with patch.object(engine, "station_owner_catalog",
                          return_value=dict(stations={"SN2": dict(owners=["MET.NO"])}), create=True):
            scenario = Scenario(engine)
            scenario.pross_only = scenario.include_met = True
            stations = engine.weather.profile("temperature:continuous")["stations"]
            self.assertEqual(scenario.station_mask(stations, False), frozenset([0]))

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
