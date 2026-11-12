import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import train_senorge_grid_pairs
import numpy as np
from flask import Flask
from pyproj import Transformer
from rasterio.io import MemoryFile

from station_footprint import register_footprint_view
from station_footprint_network import FootprintNetwork, update_best


class NetworkTests(unittest.TestCase):
    def fixture(self, root, duplicate=False):
        table = np.zeros((4, 16), float)
        table[:, :2] = [[75000, 6675000], [76000, 6675000], [77000, 6675000], [78000, 6675000]]
        maps = {}
        for ref, ids, errors in ((0, [0, 1, 2], [0., .5, 1.]),
                                 (1, [0, 1], [.4, 0.]),
                                 (2, [1, 2], [.5, 0.])):
            mapping = np.full((64, 64), -1, np.int32)
            values = np.zeros((6, 64, 64), np.float32)
            for index, error in zip(ids, errors):
                col = 32+index-ref
                mapping[32, col] = index
                values[:, 32, col] = error
            maps[ref] = values, mapping
        model = SimpleNamespace(
            root=root, atlas=SimpleNamespace(table=table),
            metadata=dict(checksums={"selected_model.pt": "fixture"}, manifest_sha256="fixture",
                          selected=dict(name="fixture"), test={t: {} for t in
                          ("annual_tg", "annual_tn", "winter_tg", "winter_tn", "summer_tg", "summer_tn")}),
            keys={(int(x/1000), int(y/1000)): i for i, (x, y) in enumerate(table[:, :2])},
            predict_reference=lambda ref: maps[ref])
        model.metadata["baseline_test"] = model.metadata["test"]
        stations = [dict(station_id=f"SN{i}", reference=i, name=f"station{i}", supported=True)
                    for i in range(3)]
        if duplicate:
            stations.append(dict(station_id="SN3", reference=0, name="same cell", supported=True))
        stations.append(dict(station_id="SN99", reference=None, name="outside", supported=False))
        return model, stations

    def test_minima_and_removal_equal_bruteforce_with_ties(self):
        rng = np.random.default_rng(1)
        predictions = rng.integers(0, 5, (8, 6, 20)).astype(np.float32)
        masks = rng.random((8, 20)) > .4
        arrays = dict(best=np.full((6, 20), np.inf, np.float32),
                      second=np.full((6, 20), np.inf, np.float32),
                      winner=np.full((6, 20), -1, np.int32),
                      runner=np.full((6, 20), -1, np.int32), counts=np.zeros(20, np.int32))
        for station in range(8):
            ids = np.flatnonzero(masks[station])
            update_best(**arrays, ids=ids, values=predictions[station][:, ids], station=station)
        full = np.where(masks[:, None], predictions, np.inf)
        np.testing.assert_array_equal(arrays["best"], full.min(axis=0))
        np.testing.assert_array_equal(arrays["second"], np.sort(full, axis=0)[1])
        np.testing.assert_array_equal(arrays["counts"], masks.sum(axis=0))
        for station in range(8):
            without = np.delete(full, station, axis=0).min(axis=0)
            computed = np.where(arrays["winner"] == station, arrays["second"], arrays["best"])
            np.testing.assert_array_equal(computed, without)

    def test_network_maps_rankings_ties_duplicates_and_loss_of_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            model, stations = self.fixture(Path(temp))
            net = FootprintNetwork(model, stations)
            net.prepare()
            self.assertEqual(net.status()["state"], "ready")
            np.testing.assert_array_equal(net.load()["counts"], [2, 3, 2, 0])
            np.testing.assert_array_equal(net.field("winter_tn")[0][:3], [0, 0, 0])
            # Station0 wins cell0; station1 is the next-best with0.4C error.
            values, valid, lost = net.field("winter_tn", "reduction", "SN0")
            np.testing.assert_allclose(values, [.4, 0, 0, 0])
            np.testing.assert_array_equal(valid, [True, True, True, False])
            self.assertFalse(lost.any())
            ranked = net.point("winter_tn", 75000, 6675000, "SN0")
            self.assertEqual(ranked["best_station"], "SN0")
            self.assertAlmostEqual(ranked["ranking"][0]["removal_reduction_c"], .4, places=6)
            self.assertEqual(ranked["ranking"][1]["removal_reduction_c"], 0)
            tie = net.point("winter_tn", 76000, 6675000)
            self.assertEqual([s["station_id"] for s in tie["ranking"]], ["SN1", "SN0", "SN2"])
            self.assertFalse(net.point("winter_tn", 78000, 6675000)["supported"])
            # Same grid reference still represents distinct replaceable Frost station IDs.
            duplicate = FootprintNetwork(model, self.fixture(Path(temp), duplicate=True)[1])
            duplicate.prepare()
            values, _, _ = duplicate.field("winter_tn", "reduction", "SN0")
            self.assertEqual(values[0], 0)
            single = FootprintNetwork(model, [stations[0]])
            single.prepare()
            values, valid, lost = single.field("winter_tn", "reduction", "SN0")
            self.assertFalse(valid.any())
            np.testing.assert_array_equal(lost, [True, True, True, False])
            point = single.point("winter_tn", 75000, 6675000)
            self.assertTrue(point["ranking"][0]["loses_coverage"])
            self.assertIsNone(point["ranking"][0]["removal_reduction_c"])
            self.assertEqual(single.describe("winter_tn", "reduction", "SN0")["lost_coverage_area_km2"], 3)
            self.assertIsNone(single.describe("winter_tn", "reduction", "SN0")["mean_c"])

    def test_checkpoint_resume_atomic_coverage_and_cache_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            model, base = self.fixture(Path(temp))
            stations = [dict(base[i % 3], station_id=f"SN{i:02}") for i in range(20)]
            net = FootprintNetwork(model, stations)
            saved_map = net.saved_map
            calls = []

            def interrupt(ref):
                calls.append(ref)
                if len(calls) == 18:
                    raise InterruptedError("test interruption")
                return saved_map(ref)

            with patch.object(net, "saved_map", side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    net.prepare()
            with np.load(net.folder / "best_two.npz") as data:
                self.assertEqual(int(data["completed"]), 16)
                self.assertEqual(data["counts"].max(), 16)
            resumed = FootprintNetwork(model, stations)
            resumed.prepare()
            fresh = FootprintNetwork(model, stations, cache=Path(temp) / "fresh")
            fresh.prepare()
            for name in resumed.load():
                np.testing.assert_array_equal(resumed.load()[name], fresh.load()[name])
            loaded = FootprintNetwork(model, stations)
            loaded.load()
            with loaded.map_path(0).open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                FootprintNetwork(model, stations).load()

    def test_network_geotiff_has_native_grid_and_loss_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            model, stations = self.fixture(Path(temp))
            net = FootprintNetwork(model, [stations[0]])
            net.prepare()
            with ZipFile(net.export("summer_tg", "reduction", "SN0")) as archive:
                metadata = json.loads(archive.read("footprint.json"))
                self.assertEqual(metadata["lost_coverage_area_km2"], 3)
                for name in ("frost_reduction_summer_tg_1000m.tif", "lost_coverage_1000m.tif"):
                    with MemoryFile(archive.read(name)) as memory:
                        with memory.open() as src:
                            self.assertEqual(src.res, (1000, 1000))
                            self.assertEqual(src.crs.to_epsg(), 25833)
                            self.assertEqual(src.xy(0, 0), (75000, 6675000))
                            values = src.read(1)
                            np.testing.assert_array_equal(values[0, :3], [-9999]*3 if
                                                          name.startswith("frost_") else [1]*3)
                            self.assertEqual(values[0, 3], -9999)
                self.assertIn("best_station_index_1000m.tif", archive.namelist())

    def test_network_api_frost_only_rank_and_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            model, stations = self.fixture(Path(temp))
            net = FootprintNetwork(model, stations)
            net.prepare()
            app = Flask(__name__)
            register_footprint_view(app, Path(temp), backend=model, stations=stations)
            client = app.test_client()
            self.assertEqual(client.get("/api/footprint/network/status").get_json()["state"], "ready")
            self.assertEqual(client.get("/api/footprint/network?target=annual_tn").status_code, 200)
            result = client.get("/api/footprint/network?mode=reduction&station=SN0").get_json()
            self.assertEqual(result["mode"], "reduction")
            self.assertEqual(result["stations"], 3)
            self.assertEqual(result["excluded_stations"], ["SN99"])
            for query in ("mode=invalid", "mode=reduction&station=SN99", "mode=reduction",
                          "target=bad", "cell_m=2500"):
                self.assertEqual(client.get(f"/api/footprint/network?{query}").status_code, 400)
            lon, lat = Transformer.from_crs(25833, 4326, always_xy=True).transform(75000, 6675000)
            point_url = f"/api/footprint/network/point?point_lon={lon}&point_lat={lat}"
            response = client.get(point_url+"&station=SN0")
            self.assertEqual(response.status_code, 200, response.get_json())
            point = response.get_json()
            self.assertEqual(point["best_station"], "SN0")
            self.assertEqual((point["x"], point["y"]), (75000, 6675000))
            self.assertEqual(point["coverage_count"], 2)
            self.assertAlmostEqual(point["selected_station"]["removal_reduction_c"], .4, places=6)
            self.assertEqual(client.get(point_url+"&station=SN99").status_code, 400)
            lon, lat = Transformer.from_crs(25833, 4326, always_xy=True).transform(78000, 6675000)
            outside = client.get(f"/api/footprint/network/point?point_lon={lon}&point_lat={lat}")
            self.assertEqual(outside.status_code, 200)
            self.assertFalse(outside.get_json()["supported"])
            self.assertIsNone(outside.get_json()["value_c"])
            image = client.get("/api/footprint/network.png?mode=lost&station=SN0")
            self.assertEqual(image.status_code, 200)
            self.assertEqual(image.headers["X-Cell-Size-M"], "1000")
            export = client.get("/api/footprint/network/export?mode=reduction&station=SN0")
            self.assertEqual(export.status_code, 200)
            self.assertEqual(client.post("/api/footprint/network/prepare").get_json()["state"], "ready")


if __name__ == "__main__":
    unittest.main()
