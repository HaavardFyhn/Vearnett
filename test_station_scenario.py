import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from station_app import Scenario, create_app
from station_scenario import export_scenario, restore_scenario
from test_station_sensor_proposals import SensorEngine


class Counties:
    def selection(self, value):
        if value is not None and (not isinstance(value, list) or any(code not in ("03", "11") for code in value)):
            raise ValueError("Invalid county")
        return value


class ScenarioSaveTests(unittest.TestCase):
    def test_roundtrip_altered_station_sensors_proposals_and_settings(self):
        engine = SensorEngine()
        original = Scenario(engine)
        original.apply(dict(action="add_sensor", station_id="SN1", sensors=["snow"]))
        original.apply(dict(action="remove_sensor", station_id="SN2", sensor="temperature"))
        with patch.object(engine, "snap", return_value=7), patch.object(engine, "station", return_value=dict(
                id=7, x=7000, y=0, lon=13, lat=64, elevation=15)):
            original.apply(dict(action="move_frost", station_id="SN1", lon=13, lat=64))
            original.apply(dict(action="comparison", value="baseline"))
            original.apply(dict(action="combine", value="rss"))
            original.apply(dict(action="threshold", value=.2))
            original.counties = ["03"]
            original.station_since = "2021-01-01"
            payload = json.loads(json.dumps(export_scenario(original)))
            restored = restore_scenario(Scenario(engine), payload, Counties())
            self.assertEqual(export_scenario(restored), export_scenario(original))
            self.assertEqual(restored.attached_proposal("SN1"), 7)
            np.testing.assert_array_equal(restored.surface(), original.surface())

    def test_invalid_import_leaves_current_scenario_unchanged(self):
        scenario = Scenario(SensorEngine())
        payload = export_scenario(scenario)
        for field, value in (("cell_m", float("nan")), ("layers", ["bad"]), ("network", "bad"),
                             ("threshold", -1), ("counties", ["bad"]), ("removed_sensors", {"SN1": ["snow"]})):
            bad = copy.deepcopy(payload)
            bad[field] = value
            with self.assertRaises(ValueError):
                restore_scenario(scenario, bad, Counties())
            self.assertEqual(export_scenario(scenario), payload)

    def test_api_roundtrip_and_reject_bad_version_and_cross_origin(self):
        client = create_app(SensorEngine()).test_client()
        client.post("/api/action", json=dict(action="add", lon=12, lat=63, sensors=["snow"]))
        saved = client.get("/api/scenario").json
        client.post("/api/action", json=dict(action="reset"))
        response = client.post("/api/scenario", json=saved)
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json["stations"][0]["sensors"], ["snow"])
        bad = dict(saved, version=999)
        self.assertEqual(client.post("/api/scenario", json=bad).status_code, 400)
        self.assertEqual(len(client.get("/api/state").json["stations"]), 1)
        self.assertEqual(client.post("/api/scenario", json=saved,
            headers={"Origin": "https://example.org"}).status_code, 403)
        saved["view"] = {"padding": "x" * 10000}
        self.assertEqual(client.post("/api/scenario", json=saved).status_code, 200)
        saved["view"]["padding"] = "x" * (2 * 1024 * 1024)
        self.assertEqual(client.post("/api/scenario", json=saved).status_code, 400)

    def test_manual_selection_uses_saved_coordinates_not_old_index(self):
        engine = SensorEngine()
        original = Scenario(engine)
        original.apply(dict(action="add", lon=12, lat=63, sensors=["snow"]))
        original.apply(dict(action="select", kind="manual", index=original.added[0]))
        payload = json.loads(json.dumps(export_scenario(original)))
        with patch.object(engine, "snap", return_value=7):
            restored = restore_scenario(Scenario(engine), payload, Counties())
        self.assertEqual(restored.added, [7])
        self.assertEqual(restored.selected, dict(kind="manual", index=7))

    def test_500m_roundtrip_and_changed_grid_rejection(self):
        engine = SensorEngine()
        engine.cell_m = 500
        original = Scenario(engine)
        original.apply(dict(action="add", lon=12, lat=63, sensors=["snow", "wind"]))
        payload = json.loads(json.dumps(export_scenario(original)))
        restored = restore_scenario(Scenario(engine), payload, Counties())
        self.assertEqual(export_scenario(restored), export_scenario(original))
        payload["proposals"][0]["position"]["x"] += 500
        with self.assertRaisesRegex(ValueError, "Terrengrutenettet"):
            restore_scenario(original, payload, Counties())

    def test_bad_field_shapes_return_validation_errors(self):
        scenario = Scenario(SensorEngine())
        saved = json.loads(json.dumps(export_scenario(scenario)))
        for field, value in (("dataset", {}), ("layers", [{}]), ("selected", {}),
                             ("selected", {"kind": "greedy"}), ("cell_m", 750)):
            payload = dict(saved, **{field: value})
            with self.assertRaises(ValueError):
                restore_scenario(scenario, payload, Counties())

    def test_edit_hidden_by_date_filter_is_preserved(self):
        engine = SensorEngine()
        original = Scenario(engine)
        original.apply(dict(action="remove_frost", station_id="SN1"))
        original.apply(dict(action="station_since", value="2099-01-01"))
        payload = json.loads(json.dumps(export_scenario(original)))
        restored = restore_scenario(Scenario(engine), payload, Counties())
        self.assertEqual(restored.station_edits, original.station_edits)


if __name__ == "__main__":
    unittest.main()
