import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import joblib

from prepare_valley_pilot import dates, download_observations, forecast_url, projection_from_das, update_groups, validate_day
from remote_senorge.train_remote import sha256, write_json
from train_valley_pilot import build_samples, compare, run, scores, validate_splits, verify_inputs
from valley_pilot import (
    GRAPH_FIELDS, HOURS, REGIME_FIELDS, conformal_radius, frost_records, graph_features,
    minimax_heights, parse_ascii, select_observations, spatial_groups, winter_year,
)


class ValleyPilotTests(unittest.TestCase):
    def observation(self, **changes):
        observation = dict(elementId="air_temperature", value=-10., qualityCode=0, unit="degC",
                           timeOffset="PT0H", timeResolution="PT1H", timeSeriesId=0,
                           level=dict(levelType="height_above_ground", unit="m", value=2))
        observation.update(changes)
        return dict(data=[dict(sourceId="SN1:0", referenceTime="2023-01-01T03:00:00Z",
                               observations=[observation])])

    def test_frost_quality_time_sensor_and_physical_guards(self):
        accepted = frost_records(self.observation())[0]
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["source_id"], "SN1:0")
        self.assertEqual(accepted["sensor_id"], 0)
        self.assertEqual(accepted["level"]["value"], 2)
        for change in (dict(qualityCode=1), dict(qualityCode=2), dict(qualityCode=7),
                       dict(qualityCode=None), dict(value=-80), dict(value=float("nan")),
                       dict(value=None), dict(unit="K"), dict(timeOffset="PT6H"),
                       dict(timeResolution="PT10M"), dict(timeSeriesId=None),
                       dict(level=dict(levelType="height_above_ground", unit="m", value=10))):
            record = frost_records(self.observation(**change))[0]
            self.assertFalse(record["accepted"], change)
            self.assertIsNotNone(record["rejection"])
        payload = self.observation()
        payload["data"][0]["referenceTime"] = "2023-01-01T03:00:00"
        with self.assertRaisesRegex(ValueError, "timezone"):
            frost_records(payload)

    def test_one_sensor_selected_without_validation_value_or_coverage(self):
        rows = []
        for sensor, times in ((0, ["03", "06"]), (1, ["03"])):
            for hour in times:
                payload = self.observation(timeSeriesId=sensor)
                payload["data"][0]["referenceTime"] = f"2023-01-01T{hour}:00:00Z"
                rows.extend(frost_records(payload))
        for day in range(1, 20):
            payload = self.observation(timeSeriesId=1)
            payload["data"][0]["referenceTime"] = f"2024-01-{day:02}T03:00:00Z"
            rows.extend(frost_records(payload))
        chosen, sensors = select_observations(pd.DataFrame(rows))
        self.assertEqual(sensors["sensor_id"].tolist(), [0])
        self.assertEqual(len(chosen), 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_observations(pd.DataFrame(rows+[rows[0]]))

    def test_dap_parser_exact_shapes_rows_and_missing_values(self):
        text = """Dataset {} name;
---------------------------------------------
x[2]
0.0, 2500.0

temperature.temperature[2][1][2][2]
[0][0][0], 1, 2
[0][0][1], 3, 4
[1][0][0], 5, 6
[1][0][1], 7, 8

temperature.time[2]
0, 3600
"""
        np.testing.assert_array_equal(parse_ascii(text, "x"), [0, 2500])
        array = parse_ascii(text, "temperature.temperature")
        np.testing.assert_array_equal(array.reshape(-1), np.arange(1, 9))
        for broken in (text.replace("[0][0][1], 3, 4\n", ""),
                       text.replace("[0][0][1], 3, 4", "[0][0][0], 3, 4"),
                       text.replace("[0][0][1], 3, 4", "[0][0][1], 3"),
                       text.replace("[0][0][1], 3, 4", "[0][0][1], 3, 9.969E36")):
            with self.assertRaises(ValueError):
                parse_ascii(broken, "temperature.temperature")
        with self.assertRaises(ValueError):
            parse_ascii("HTTP503", "x")

    def test_pass_graph_detects_separate_valleys_detours_and_nodata(self):
        terrain = np.array([[0, 900, 0], [0, 900, 0], [0, 200, 0]], float)
        valid = np.ones_like(terrain, bool)
        positions = np.array([[0, 0], [0, 2]])
        graph = graph_features(terrain, valid, positions)
        self.assertEqual(graph.shape, (2, 2, len(GRAPH_FIELDS)))
        self.assertAlmostEqual(graph[0, 1, 1], .2)
        self.assertAlmostEqual(graph[0, 1, 3], .9)
        self.assertEqual(graph[0, 1, 4], 1)
        valid[:, 1] = False
        graph = graph_features(terrain, valid, positions)
        np.testing.assert_array_equal(graph[0, 1], 0)
        with self.assertRaises(ValueError):
            minimax_heights(terrain, valid, (0, 1))
        valid[2, 1] = True
        graph = graph_features(terrain, valid, positions)
        self.assertEqual(graph[0, 1, 0], 1)
        self.assertEqual(graph[0, 1, 4], 0)
        self.assertEqual(graph[0, 1, 3], 0)

    def fixture(self):
        ids = ["SN1", "SN2", "SN3", "SN55820", "SN56960"]
        stations = pd.DataFrame(dict(station_id=ids, group=[0, 0, 0, 1, 2]))
        table = np.zeros((5, 16), np.float32)
        table[:, 0] = np.arange(5)*1000
        table[:, 1] = 6840000
        table[:, 2] = [100, 150, 200, 180, 120]
        graph = np.zeros((5, 5, len(GRAPH_FIELDS)), np.float32)
        graph[:, :, 0] = 1
        rng = np.random.default_rng(42)
        rows, regimes = [], {}
        for year in (2022, 2023, 2024):
            for stamp in pd.date_range(f"{year}-12-01", periods=40, tz="UTC"):
                day = stamp.date().isoformat()
                field = rng.normal(size=(3, 5, 4))
                field[:, :, 0] -= 8
                field[:, :, 1:3] *= 3
                field[:, :, 3] = rng.uniform(size=(3, 5))
                run = stamp-pd.Timedelta(hours=12)
                regimes[day] = dict(
                    date=day, run_time=run.isoformat(), station_ids=ids, hours=list(HOURS),
                    bearing=np.zeros((5, 5, 2)).tolist(),
                    fields={name: field[:, :, i].tolist() for i, name in enumerate(
                        ("temperature_c", "wind_x_ms", "wind_y_ms", "cloud_fraction"))})
                for hour_index, hour in enumerate(HOURS):
                    shared = rng.normal()
                    for i, identifier in enumerate(ids):
                        temperature = shared-8+field[hour_index, i, 0]*.1-field[hour_index, i, 3]*2
                        rows.append(dict(station_id=identifier, time=stamp+pd.Timedelta(hours=hour),
                                         value=temperature))
        observations = pd.DataFrame(rows)
        return stations, table, graph, observations, regimes

    def test_samples_keep_heldout_stations_and_winters_out_of_training(self):
        stations, table, graph, obs, regimes = self.fixture()
        samples, features, y, names = build_samples(stations, table, graph, obs, regimes)
        self.assertEqual(set(samples.loc[samples["split"] == 0, "target"]), {"SN1", "SN2", "SN3"})
        self.assertEqual(set(samples.loc[samples["split"] == 1, "target"]), {"SN55820"})
        self.assertEqual(set(samples.loc[samples["split"] == 2, "target"]), {"SN56960"})
        self.assertFalse(set(samples["source"]) & {"SN55820", "SN56960"})
        for name, values in features.items():
            self.assertEqual(values.shape, (len(samples), len(names[name])))
            self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(y, samples["target_c"]-samples["source_c"], atol=1e-6)
        leaked = samples.copy()
        leaked.loc[leaked["split"] == 0, "source"] = "SN56960"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(leaked)
        damaged = copy.deepcopy(regimes)
        damaged["2022-12-01"]["run_time"] = "2022-12-01T00:00:00+00:00"
        with self.assertRaisesRegex(ValueError, "precede"):
            build_samples(stations, table, graph, obs, damaged)

    def test_sources_require_training_period_qc_availability_before_neighbour_cap(self):
        stations, table, graph, observations, regimes = self.fixture()
        heldout = ["SN55820", "SN56960"]
        extra = pd.DataFrame(dict(station_id=[f"EMPTY{i}" for i in range(7)], group=[0]*7))
        expanded = pd.concat([stations, extra], ignore_index=True)
        terrain = np.vstack([table, np.repeat(table[4:5], 7, axis=0)])
        edges = np.zeros((len(expanded), len(expanded), len(GRAPH_FIELDS)), np.float32)
        edges[:, :, 0] = 1
        weather = copy.deepcopy(regimes)
        for payload in weather.values():
            payload["station_ids"] = expanded.station_id.tolist()
            payload["bearing"] = np.zeros((len(expanded), len(expanded), 2)).tolist()
            for field in REGIME_FIELDS:
                payload["fields"][field] = np.pad(
                    np.asarray(payload["fields"][field]), ((0, 0), (0, 7))).tolist()
        samples, _, _, _ = build_samples(expanded, terrain, edges, observations, weather)
        self.assertEqual(set(samples.loc[samples["split"] == 2, "source"]), {"SN1", "SN2", "SN3"})
        self.assertFalse(set(samples["source"]) & set(extra.station_id))
        # Later availability must not qualify a source that had no training QC data.
        later = observations.loc[observations["time"].map(winter_year) == 2023].copy()
        later["station_id"] = "EMPTY0"
        later = later.drop_duplicates(["station_id", "time"])
        samples, _, _, _ = build_samples(
            expanded, terrain, edges, pd.concat([observations, later]), weather)
        self.assertFalse(set(samples["source"]) & set(extra.station_id))

    def test_all_four_comparisons_intervals_and_persistence_inputs(self):
        stations, table, graph, obs, regimes = self.fixture()
        samples, features, y, _ = build_samples(stations, table, graph, obs, regimes)
        models, predictions, report = compare(samples, features, y, trees=12, jobs=1)
        self.assertEqual(set(models), {"terrain", "regime", "graph_regime"})
        self.assertEqual(set(predictions), {"forecast_transfer", *models})
        self.assertIn(report["selected_by_validation"], predictions)
        for values in report["variants"].values():
            self.assertEqual(values["selection"]["nights"], 20)
            self.assertEqual(values["calibration_nights"], 20)
            self.assertEqual(values["test"]["nights"], 40)
            self.assertEqual(values["test"]["target_stations"], 1)
            self.assertGreaterEqual(values["test"]["interval_radius_c"], 0)
            self.assertGreaterEqual(values["test"]["empirical_pair_coverage_90"], 0)
            self.assertLessEqual(values["test"]["empirical_pair_coverage_90"], 1)

    def test_complete_verified_pipeline_persists_reloadable_models_and_results(self):
        stations, table, graph, observations, regimes = self.fixture()
        identifiers = stations["station_id"].tolist()
        with tempfile.TemporaryDirectory() as temp:
            folder, output = Path(temp) / "data", Path(temp) / "results"
            folder.mkdir()
            stations.to_csv(folder / "stations.csv", index=False)
            np.savez_compressed(folder / "terrain.npz", table=table, graph=graph)
            write_json(folder / "config.json", dict(
                schema=1, terrain_checksums={name: sha256(folder / name)
                                             for name in ("terrain.npz", "stations.csv")}))
            receipts = {stage: {} for stage in ("observations", "regimes")}
            for stage in receipts:
                (folder / stage).mkdir()
            for day in dates():
                key = day.isoformat()
                forecast = regimes.get(key, dict(
                    date=key, run_time=(pd.Timestamp(key, tz="UTC")-pd.Timedelta(hours=12)).isoformat(),
                    station_ids=identifiers, hours=list(HOURS),
                    fields={field: np.zeros((3, 5)).tolist() for field in REGIME_FIELDS},
                    bearing=np.zeros((5, 5, 2)).tolist()))
                rows = []
                for row in observations.loc[observations["time"].dt.date == day].itertuples(index=False):
                    payload = self.observation(value=row.value)
                    payload["data"][0]["sourceId"] = row.station_id+":0"
                    payload["data"][0]["referenceTime"] = row.time.isoformat()
                    rows.extend(frost_records(payload))
                payloads = dict(regimes=forecast,
                                observations=dict(date=key, station_ids=identifiers, records=rows))
                for stage, payload in payloads.items():
                    path = folder / stage / f"{key}.json"
                    write_json(path, payload)
                    receipts[stage][key] = sha256(path)
            for stage, hashes in receipts.items():
                write_json(folder / f"{stage}_progress.json", hashes)
            args = SimpleNamespace(data=folder, output=output, trees=4, jobs=1)
            with patch("builtins.print"):
                run(args)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["qc_rejections"], {})
            self.assertEqual(len(report["model_checksums"]), 3)
            for name, digest in report["model_checksums"].items():
                self.assertEqual(sha256(output / name), digest)
                model = joblib.load(output / name)
                self.assertEqual(model.n_features_in_, len(report["features"][name[:-7]]))
            self.assertEqual(pd.read_csv(output / "test_predictions.csv")["target"].unique().tolist(), ["SN56960"])
            path = folder / "observations" / "2022-12-01.json"
            with path.open("a") as file:
                file.write(" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_inputs(folder)

    def test_conformal_and_metrics_do_not_count_correlated_rows_as_nights(self):
        self.assertEqual(conformal_radius(np.arange(1, 10)), 9)
        with self.assertRaisesRegex(ValueError, "Too few"):
            conformal_radius(np.arange(8))
        s = pd.DataFrame(dict(target_c=[0, 0], target=["A", "A"],
                              night=["2024-01-01"]*2, calm_clear=[False, True]))
        metric = scores(s, np.array([0., 2.]), radius=1.)
        self.assertEqual(metric["empirical_pair_coverage_90"], .5)
        self.assertEqual(metric["empirical_night_simultaneous_coverage_90"], 0)
        self.assertEqual(metric["nights"], 1)
        self.assertIsNone(metric["cold_below_minus5"]["rmse_c"])

    def test_geographic_groups_do_not_use_observation_values(self):
        stations = pd.DataFrame(dict(
            station_id=["SN56960", "SN55820", "SN1", "SN2"],
            x=[41000, 37000, 41500, 100000], y=[6840000, 6832000, 6840100, 6850000]))
        np.testing.assert_array_equal(spatial_groups(stations), [2, 1, 2, 0])

    def test_group_migration_preserves_order_graph_and_provider_caches(self):
        stations = pd.DataFrame(dict(
            station_id=["SN56960", "SN55820", "SN56895", "SN1"],
            x=[41000, 55000, 37000, 100000], y=[6840000, 6860000, 6832000, 6850000],
            group=[2, 0, 1, 0]))
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            stations.to_csv(folder / "stations.csv", index=False)
            np.savez(folder / "terrain.npz", graph=np.ones((4, 4, 5)))
            (folder / "regimes_progress.json").write_text('{"2024-01-15":"untouched"}')
            write_json(folder / "config.json", dict(
                terrain_checksums={name: sha256(folder / name) for name in ("stations.csv", "terrain.npz")}))
            before = {name: sha256(folder / name) for name in ("terrain.npz", "regimes_progress.json")}
            with patch("builtins.print"):
                update_groups(folder)
            updated = pd.read_csv(folder / "stations.csv")
            pd.testing.assert_frame_equal(updated.drop(columns="group"), stations.drop(columns="group"))
            np.testing.assert_array_equal(updated["group"], [2, 1, 0, 0])
            for name, checksum in before.items():
                self.assertEqual(sha256(folder / name), checksum)
            config = json.loads((folder / "config.json").read_text())
            self.assertEqual(config["terrain_checksums"]["stations.csv"], sha256(folder / "stations.csv"))

    def test_download_identity_pagination_and_cached_day_validation(self):
        stations = pd.DataFrame(dict(station_id=["SN1"]))
        payload = self.observation()
        payload["@nextLink"] = "https://example.invalid/observations"
        with patch("prepare_valley_pilot.http", return_value=__import__("json").dumps(payload)):
            with self.assertRaisesRegex(ValueError, "pagination"):
                download_observations(None, pd.Timestamp("2023-01-01").date(), stations, "not-a-secret")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_day(dict(date="2023-01-01", station_ids=["SN1"], records=[dict(station_id="SN2")]),
                         pd.Timestamp("2023-01-01").date(), stations, "observations")
        self.assertTrue(forecast_url(pd.Timestamp("2023-01-01").date())[0].endswith(".nc"))
        self.assertTrue(forecast_url(pd.Timestamp("2024-01-01").date())[0].endswith(".ncml"))
        projection = projection_from_das("""
projection_lambert {
String grid_mapping_name "lambert_conformal_conic";
Float64 standard_parallel 63.3;
Float64 longitude_of_central_meridian 15;
Float64 latitude_of_projection_origin 63.3;
Float64 earth_radius 6371000;
}
""")
        self.assertEqual(projection.to_cf()["grid_mapping_name"], "lambert_conformal_conic")


if __name__ == "__main__":
    unittest.main()
