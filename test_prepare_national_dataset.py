from contextlib import closing
from datetime import date
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pyproj import Transformer
import requests

import prepare_national_dataset as national
from remote_senorge.train_remote import sha256, write_json


def stations_fixture():
    stations, qc = [], []
    for block in range(12):
        for i in range(3):
            identifier = f"SN{block*3+i}"
            stations.append(dict(station_id=identifier, name=identifier, x=block*100000+50000+i*1000,
                                 y=6500000+block*100000, lon=20., lat=65.,
                                 block_x=block, block_y=65+block))
            qc.append(dict(station_id=identifier, source_id=identifier+":0", sensor_id=0,
                           eligible=True, train_nights=100, validation_nights=60, test_nights=60))
    return pd.DataFrame(stations), pd.DataFrame(qc)


def weather(day, stations):
    n = len(stations)
    fields = dict(temperature_c=np.full((3, n), -2.).tolist(),
                  wind_x_ms=np.ones((3, n)).tolist(), wind_y_ms=np.zeros((3, n)).tolist(),
                  cloud_fraction=np.full((3, n), .2).tolist())
    return dict(date=str(day), run_time=national.forecast_url(day)[1].isoformat(), hours=list(national.HOURS),
                station_ids=stations["station_id"].tolist(), fields=fields,
                projection_xy=stations[["x", "y"]].to_numpy().tolist())


class NationalTests(unittest.TestCase):
    def test_five_complete_winters_and_split_rules(self):
        days = list(national.dates())
        self.assertEqual(len(days), 451)
        self.assertIn(date(2024, 2, 29), days)
        self.assertEqual(national.split_for(date(2021, 1, 1), "train"), "train")
        self.assertIsNone(national.split_for(date(2024, 12, 1), "train"))
        self.assertIsNone(national.split_for(date(2020, 12, 1), "test"))
        self.assertEqual(national.split_for(date(2024, 12, 1), "test"), "test")
        self.assertEqual(national.split_for(date(2023, 12, 1), "validation"), "selection")
        self.assertEqual(national.split_for(date(2023, 12, 2), "validation"), "calibration")

    def test_initialization_units_support_catalogue_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atlas = root / "outputs" / "senorge_footprint_expanded"
            atlas.mkdir(parents=True)
            output = root / "dataset"
            output.mkdir()
            origin = [200000, 6500000]
            table = np.ones((36, 16))
            table[:, :2] = [[origin[0]+c*1000, origin[1]-r*1000] for r in range(6) for c in range(6)]
            np.save(atlas / "grid_table.npy", table)
            np.save(atlas / "terrain_height.npy", np.full((60, 60), 2., np.float32))
            valid = np.ones((60, 60), np.uint8)
            valid[:10, :10] = 0
            np.save(atlas / "terrain_valid.npy", valid)
            write_json(atlas / "footprint_manifest.json",
                       dict(crs="EPSG:25833", context_resolution_m=100, atlas_origin=origin))
            lon, lat = Transformer.from_crs(25833, 4326, always_xy=True).transform(table[:, 0], table[:, 1])
            with closing(sqlite3.connect(root / "frost_history.db")) as conn:
                conn.execute("CREATE TABLE stations(station_id TEXT,name TEXT,lat REAL,lon REAL,elevation REAL,inside_norway INTEGER)")
                conn.executemany("INSERT INTO stations VALUES(?,?,?,?,?,1)",
                                 [(f"SN{i}", f"Station{i}", float(lat[i]), float(lon[i]), 2000.) for i in range(36)])
                conn.commit()
            national.initialize(root, output)
            config = national.load_config(output)
            self.assertEqual(config["winters"], list(national.WINTERS))
            self.assertEqual(len(pd.read_csv(output / "stations.csv")), 35)
            with np.load(output / "terrain.npz") as terrain:
                self.assertEqual(terrain["height"][1, 1], 2000.)
                self.assertFalse(terrain["valid"][0, 0])
                self.assertEqual(terrain["table"].shape, (35, 16))
            (output / "stations.csv").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                national.initialize(root, output)

    def test_qc_audit_training_sensor_only_and_duplicate_rejection(self):
        stations = pd.DataFrame(dict(station_id=["SN1", "SN2"]))
        rows = []
        for stamp in pd.date_range("2020-12-01", periods=30, tz="UTC"):
            rows.append(dict(station_id="SN1", source_id="SN1:0", sensor_id=0,
                             time=stamp.isoformat(), value=-2., accepted=True))
        for stamp in pd.date_range("2024-12-01", periods=60, tz="UTC"):
            rows.append(dict(station_id="SN1", source_id="SN1:0", sensor_id=1,
                             time=stamp.isoformat(), value=30., accepted=True))
        rows.append(dict(station_id="SN2", accepted=False, rejection="quality_not_zero"))
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            write_json(output / "observations_progress.json", {})
            with patch.object(national, "verified_observation_batches", return_value=[dict(records=rows)]):
                qc = national.audit(output, stations)
            self.assertEqual(qc.iloc[0]["sensor_id"], 0)
            self.assertEqual(qc.iloc[0]["test_nights"], 0)
            self.assertTrue(qc.iloc[0]["eligible"])
            self.assertFalse(qc.iloc[1]["eligible"])
            with patch.object(national, "verified_observation_batches", return_value=[dict(records=rows+[rows[0]])]):
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    national.audit(output, stations)

    def test_deterministic_geographic_blocks_buffer_and_qc_failures(self):
        stations, qc = stations_fixture()
        first, blocks = national.choose_groups(stations, qc)
        second, again = national.choose_groups(stations.sample(frac=1, random_state=3), qc)
        self.assertEqual(blocks, again)
        self.assertEqual(first.set_index("station_id")["group"].to_dict(),
                         second.set_index("station_id")["group"].to_dict())
        self.assertEqual(sum(b["group"] == "test" for b in blocks), 2)
        self.assertEqual(sum(b["group"] == "validation" for b in blocks), 2)
        self.assertEqual(first["group"].eq("train").sum(), 24)
        chosen = blocks[0]
        border = pd.DataFrame([dict(station_id="BORDER", name="BORDER",
                                   x=chosen["block_x"]*100000-5000,
                                   y=chosen["block_y"]*100000+50000, lon=20, lat=65,
                                   block_x=chosen["block_x"]-1, block_y=chosen["block_y"])])
        extra = pd.DataFrame([dict(station_id="BORDER", source_id="BORDER:0", sensor_id=0,
                                  eligible=True, train_nights=30, validation_nights=0, test_nights=0)])
        buffered, same = national.choose_groups(pd.concat([stations, border]), pd.concat([qc, extra]))
        self.assertEqual(same, blocks)
        self.assertEqual(buffered.set_index("station_id").loc["BORDER", "group"], "excluded_buffer")
        with self.assertRaisesRegex(ValueError, "geographic blocks"):
            national.choose_groups(stations.iloc[:9], qc.iloc[:9])

    def test_freeze_provenance_and_geojson(self):
        stations, qc = stations_fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_json(folder / "config.json", {})
            write_json(folder / "observations_progress.json", {})
            write_json(folder / "observations_config.json", {})
            national.atomic_csv(folder / "qc_audit.csv", qc)
            write_json(folder / "qc_audit.json", {})
            with patch.object(national, "audit", return_value=qc):
                split = national.freeze(folder, stations)
            self.assertEqual(national.load_splits(folder), split)
            geo = json.loads((folder / "station_coverage.geojson").read_text())
            self.assertEqual(len(geo["features"]), 36)
            self.assertEqual(geo["features"][0]["geometry"]["coordinates"], [20., 65.])
            (folder / "qc_audit.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                national.freeze(folder, stations)

    def test_bounded_rectangles_cover_every_station_once(self):
        random = np.random.default_rng(42)
        ix, iy = random.integers(0, 1200, (2, 1000))
        subsets = national.rectangles(ix, iy)
        seen = []
        for indices, c0, c1, r0, r1 in subsets:
            self.assertLessEqual((c1-c0+1)*(r1-r0+1), 4096)
            seen.extend(indices)
            self.assertTrue(((ix[indices] >= c0) & (ix[indices] <= c1)).all())
        self.assertEqual(sorted(seen), list(range(1000)))
        self.assertLess(len(subsets), 1000)
        with self.assertRaises(ValueError):
            national.rectangles(np.array([]), np.array([]))

    def test_cached_observation_resume_and_missing_credentials(self):
        day = date(2020, 12, 1)
        stations, _ = stations_fixture()
        payload = dict(date=str(day), station_ids=stations["station_id"].tolist(), records=[])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch.dict("os.environ", {"FROST_CLIENT_ID": "test-placeholder"}), patch.object(
                    national, "download_observations", return_value=payload) as download:
                national.observations(folder, stations, [day])
                self.assertEqual(download.call_count, 1)
            with patch.dict("os.environ", {}, clear=True), patch.object(national, "download_observations") as download:
                national.observations(folder, stations, [day])
                download.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "FROST_CLIENT_ID"):
                    national.observations(folder, stations, [date(2020, 12, 2)])
            path = folder / "observations" / f"{day}_b000.json"
            path.write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                national.observations(folder, stations, [day])

    def test_graph_reuses_training_sources_sea_barrier_and_distance(self):
        table = np.ones((4, 16))
        table[:, :2] = [[0, 0], [1000, 0], [3000, 0], [200000, 0]]
        positions = np.array([[0, 0], [0, 1], [0, 3], [0, 4]])
        height = np.zeros((5, 5))
        valid = np.ones((5, 5), bool)
        valid[:, 2] = False
        pairs, graph, excluded = national.pair_graph(
            table, positions, height, valid, ["train", "validation", "test", "test"])
        self.assertEqual(pairs.tolist(), [[0, 1]])
        self.assertEqual(graph.shape, (1, 5))
        self.assertEqual({r["target_index"] for r in excluded}, {0, 2, 3})

    def test_forecast_chunk_resume_and_shape_checks(self):
        day = date(2020, 12, 1)
        stations = pd.DataFrame(dict(station_id=["SN1", "SN2"], lon=[6., 7.], lat=[61., 62.],
                                     x=[0., 1.], y=[0., 0.]))
        meta = weather(day, stations)
        meta.update(ix=[0, 1], iy=[0, 0], url="https://example.invalid/archive")
        del meta["fields"]
        text = "stub"

        def parser(_, name):
            value = 270. if name.startswith("air") else .2 if name.startswith("cloud") else 1.
            return np.full((3, 1, 1, 2), value)

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch.object(national, "regime_metadata", return_value=meta), patch.object(
                    national, "http", return_value=text) as http, patch.object(national, "parse_ascii", side_effect=parser):
                national.regimes(folder, stations, [day])
                self.assertEqual(http.call_count, 1)
            saved = json.loads((folder / "regimes" / f"{day}.json").read_text())
            self.assertAlmostEqual(saved["fields"]["temperature_c"][0][0], -3.15)
            with patch.object(national, "http") as http:
                national.regimes(folder, stations, [day])
                http.assert_not_called()
            saved["run_time"] = "2020-12-01T12:00:00+00:00"
            with self.assertRaisesRegex(ValueError, "issue time"):
                national.validate_regime(saved, day, stations)

    def test_interrupted_forecast_reuses_durable_chunk(self):
        day = date(2020, 12, 1)
        stations = pd.DataFrame(dict(station_id=["SN1", "SN2"], lon=[6., 7.], lat=[61., 62.],
                                     x=[0., 100000.], y=[0., 100000.]))
        meta = weather(day, stations)
        meta.update(ix=[0, 100], iy=[0, 100], url="https://example.invalid/archive")
        del meta["fields"]

        def parser(_, name):
            return np.full((3, 1, 1, 1), 270. if name.startswith("air") else .2)

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch.object(national, "regime_metadata", return_value=meta), patch.object(
                    national, "http", side_effect=["first", RuntimeError("Interrupted")]), patch.object(
                    national, "parse_ascii", side_effect=parser):
                with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                    national.regimes(folder, stations, [day])
            receipt = json.loads((folder / "regimes_progress.json").read_text())
            self.assertIn(f"{day}_c000", receipt)
            self.assertNotIn(str(day), receipt)

            with patch.object(national, "regime_metadata") as metadata, patch.object(
                    national, "http", return_value="second") as http, patch.object(
                    national, "parse_ascii", side_effect=parser):
                national.regimes(folder, stations, [day])
                metadata.assert_not_called()
                self.assertEqual(http.call_count, 1)

    def test_frost_qc_parser_remains_strict_for_national_inputs(self):
        from valley_pilot import frost_records
        payload = dict(data=[dict(sourceId="SN1:0", referenceTime="2020-12-01T03:00:00Z",
                                 observations=[dict(elementId="air_temperature", value=-3., unit="degC",
                                                    timeOffset="PT0H", timeResolution="PT1H",
                                                    qualityCode=6, timeSeriesId=0)])])
        rows = frost_records(payload)
        self.assertFalse(rows[0]["accepted"])
        self.assertEqual(rows[0]["rejection"], "quality_not_zero")
        payload["data"][0]["observations"][0]["qualityCode"] = 0
        self.assertTrue(frost_records(payload)[0]["accepted"])

    def test_missing_archive_run_confirmed_excluded_and_resumed(self):
        stations, _ = stations_fixture()
        day = date(2022, 1, 16)
        response = requests.Response()
        response.status_code = 404
        response.url = national.forecast_url(day)[0]+".das"
        xml = ('<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">'
               '<dataset urlPath="meps25epsarchive/2022/01/15/meps_det_2_5km_20220115T06Z.nc"/>'
               '</catalog>')
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch.object(national, "regime_metadata", side_effect=requests.HTTPError(response=response)), patch.object(
                    national, "http", return_value=xml):
                national.regimes(folder, stations, [day])
            payload = json.loads((folder / "regimes" / f"{day}.json").read_text())
            self.assertEqual(payload["state"], "excluded_missing_run")
            self.assertNotIn("fields", payload)
            national.validate_regime(payload, day, stations)
            with patch.object(national, "regime_metadata") as metadata, patch.object(national, "http") as http:
                national.regimes(folder, stations, [day])
                metadata.assert_not_called()
                http.assert_not_called()
            expected_path = national.forecast_url(day)[0].split("/thredds/dodsC/", 1)[1]
            with patch.object(national, "http", return_value=xml.replace(
                    "meps25epsarchive/2022/01/15/meps_det_2_5km_20220115T06Z.nc", expected_path)):
                with self.assertRaisesRegex(ValueError, "not confirmed"):
                    national.confirmed_missing_run(None, day, stations)
            for status in (401, 500):
                response.status_code = status
                with patch.object(national, "regime_metadata", side_effect=requests.HTTPError(response=response)):
                    with self.assertRaises(requests.HTTPError):
                        national.regimes(folder, stations, [date(2022, 1, 17)])
            payload["fields"] = {}
            with self.assertRaisesRegex(ValueError, "exclusion"):
                national.validate_regime(payload, day, stations)

    def test_documented_frost_empty_batches_not_auth_or_pagination_errors(self):
        from prepare_valley_pilot import download_observations
        stations, _ = stations_fixture()
        day = date(2020, 12, 1)
        for status in (404, 412, 401, 403, 500):
            response = requests.Response()
            response.status_code = status
            response.url = "https://frost.met.no/observations/v0.jsonld?sources=SN1"
            error = requests.HTTPError(response=response)
            with patch("prepare_valley_pilot.http", side_effect=error):
                if status in (404, 412):
                    payload = download_observations(None, day, stations, "placeholder", allow_empty=True)
                    self.assertEqual(payload["provider_no_data_status"], status)
                    self.assertEqual(payload["records"], [])
                else:
                    with self.assertRaises(requests.HTTPError):
                        download_observations(None, day, stations, "placeholder", allow_empty=True)
                with self.assertRaises(requests.HTTPError):
                    download_observations(None, day, stations, "placeholder")
        response.status_code = 404
        first = json.dumps(dict(data=[], **{"@nextLink": "https://frost.met.no/observations/v0.jsonld?offset=1"}))
        with patch("prepare_valley_pilot.http", side_effect=[first, requests.HTTPError(response=response)]):
            with self.assertRaises(requests.HTTPError):
                download_observations(None, day, stations, "placeholder", allow_empty=True)

    def test_daily_feature_label_shape_and_endpoint_leakage(self):
        stations = pd.DataFrame(dict(station_id=["SN1", "SN2"], group=["train", "test"],
                                     x=[0., 1000.], y=[0., 0.]))
        day = date(2024, 12, 1)
        table = np.ones((2, 16))
        table[:, :2] = stations[["x", "y"]].to_numpy()
        temperatures = {(station, (pd.Timestamp(day, tz="UTC")+pd.Timedelta(hours=hour)).isoformat()): value
                        for station, value in (("SN1", -2.), ("SN2", -5.)) for hour in national.HOURS}
        arrays = national.day_samples(day, stations, table, np.array([[0, 1]]),
                                      np.array([[1., 0., 0., 0., 1.]]), temperatures, weather(day, stations))
        national.validate_samples(arrays, stations)
        self.assertEqual(arrays["x"].shape, (3, 39))
        np.testing.assert_array_equal(arrays["y"], [-3., -3., -3.])
        self.assertTrue(arrays["calm_clear"].all())
        arrays["source"][:] = "SN2"
        with self.assertRaisesRegex(ValueError, "leakage"):
            national.validate_samples(arrays, stations)

    def test_full_sharded_build_checksum_resume_and_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            all_stations = pd.DataFrame(dict(station_id=[f"SN{i}" for i in range(6)],
                                             name=[f"Station{i}" for i in range(6)],
                                             group=["train", "train", "validation", "validation", "test", "test"],
                                             source_id=[f"SN{i}:0" for i in range(6)], sensor_id=[0]*6,
                                             x=np.arange(6)*1000., y=np.zeros(6),
                                             block_x=[0, 0, 1, 1, 2, 2], block_y=[0]*6))
            national.atomic_csv(folder / "station_splits.csv", all_stations)
            write_json(folder / "splits.json", dict(schema=national.SCHEMA, policy=national.POLICY,
                                                   blocks=[dict(block_x=1, block_y=0, group="validation"),
                                                           dict(block_x=2, block_y=0, group="test")], checksums={}))
            table = np.ones((6, 16))
            table[:, :2] = all_stations[["x", "y"]].to_numpy()
            national.atomic_npz(folder / "terrain.npz", table=table, positions=np.zeros((6, 2), int),
                                 height=np.zeros((6, 6)), valid=np.ones((6, 6), bool))
            obs_receipt, weather_receipt = {}, {}
            (folder / "observations").mkdir()
            (folder / "regimes").mkdir()
            for day in national.dates():
                key = f"{day}_b000"
                records = []
                for station in all_stations.itertuples():
                    for hour in national.HOURS:
                        records.append(dict(
                            station_id=station.station_id, source_id=station.source_id, sensor_id=0,
                            time=(pd.Timestamp(day, tz="UTC")+pd.Timedelta(hours=hour)).isoformat(),
                            value=-2., accepted=True, quality_code=0, unit="degC",
                            time_offset="PT0H", time_resolution="PT1H"))
                path = folder / "observations" / f"{key}.json"
                write_json(path, dict(date=str(day), station_ids=all_stations["station_id"].tolist(), records=records))
                obs_receipt[key] = sha256(path)
                path = folder / "regimes" / f"{day}.json"
                write_json(path, weather(day, all_stations))
                weather_receipt[str(day)] = sha256(path)
            write_json(folder / "observations_progress.json", obs_receipt)
            excluded_day = date(2022, 1, 16)
            xml = ('<catalog xmlns="http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0">'
                   '<dataset urlPath="meps25epsarchive/2022/01/15/other.nc"/></catalog>')
            with patch.object(national, "http", return_value=xml):
                excluded = national.confirmed_missing_run(None, excluded_day, all_stations)
            path = folder / "regimes" / f"{excluded_day}.json"
            write_json(path, excluded)
            weather_receipt[str(excluded_day)] = sha256(path)
            write_json(folder / "regimes_progress.json", weather_receipt)
            pairs = np.array([[1, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]])
            graph = np.tile([1., 0., 0., 0., 1.], (6, 1))
            with patch.object(national, "pair_graph", return_value=(pairs, graph, [])), patch("builtins.print"):
                national.build(folder, all_stations, list(national.dates()))
                national.build(folder, all_stations, list(national.dates()))
            manifest = json.loads((folder / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["state"], "ready")
            self.assertEqual(len(manifest["shards"]), 451)
            self.assertEqual(manifest["counts"]["test"]["targets"], 2)
            self.assertEqual(manifest["counts"]["train"]["nights"], 269)
            self.assertEqual(manifest["counts"]["selection"]["nights"], 46)
            self.assertEqual(manifest["forecast_nights"], dict(requested=451, available=450, excluded=1))
            self.assertEqual(manifest["excluded_forecast_nights"][0]["date"], "2022-01-16")
            with np.load(folder / "samples" / "2022-01-16.npz") as shard:
                self.assertEqual(shard["x"].shape, (0, 39))
            path = folder / "samples" / "2024-12-01.npz"
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"), patch("builtins.print"):
                national.build(folder, all_stations, [date(2024, 12, 1)])


if __name__ == "__main__":
    unittest.main()
