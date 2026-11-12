import unittest
import numpy as np
import pandas as pd

from station_pross import pross_excluded, svv_mask


class ProssStationTests(unittest.TestCase):
    def test_road_prefixes_not_coordinates_or_embedded_names(self):
        names = ["E18 DRAMMEN", "FV307 GRAVDAL", "RV13 LATEFOSS", " e 6 TEST",
                 "F17 TEST", "KV5738 TEST", "Pv12 TEST", "SV1 TEST", "EV39 TEST",
                 "BERGEN E18", "EIDFJORD", "300 KV SAUDA", "", None, "E18th STREET"]
        np.testing.assert_array_equal(svv_mask(names), [True]*9 + [False]*6)
        frame = pd.DataFrame(dict(name=names, x=0, y=0))
        self.assertEqual(pross_excluded(frame), frozenset(range(9, 15)))
        frame.loc[0, "name"] = "OSLO"
        self.assertIn(0, pross_excluded(frame))

    def test_old_location_model_id_resolves_to_met_network(self):
        from station_app import Scenario
        from test_station_app import FakeWeatherEngine
        scenario = Scenario(FakeWeatherEngine(), dataset="temperature:senorge30_pross")
        self.assertEqual(scenario.dataset, "temperature:senorge30")
        self.assertEqual(scenario.layers, ["temperature:senorge30"])
