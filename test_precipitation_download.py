from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import pandas as pd

import download_frost as downloader
import information_gain
import overnight_run


START = datetime(2026, 2, 1, tzinfo=timezone.utc)
END = datetime(2026, 3, 1, tzinfo=timezone.utc)


def day_readings(day=START, sid="SN1", value=0.1):
    first = day - timedelta(days=1) + timedelta(hours=6, minutes=10)
    return [{"sourceId": f"{sid}:0", "referenceTime": downloader._iso(first + timedelta(minutes=10 * i)),
             "observations": [{"elementId": downloader.PRECIP_TEN_MINUTES, "value": value,
                               "timeResolution": "PT10M", "timeSeriesId": 0, "qualityCode": 0}]}
            for i in range(144)]


class PrecipitationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "frost.db"
        self.patch = patch.object(downloader, "DATABASE", self.database)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        with closing(downloader._connect()) as cx:
            cx.commit()

    def rows(self, items, start=START, end=END):
        with redirect_stdout(StringIO()):
            return downloader.precipitation_daily_rows(items, start, end)

    def download(self, items, **kwargs):
        with patch.object(downloader, "fetch_batch_recursive", return_value=iter(items)), \
                redirect_stdout(StringIO()):
            return downloader.download_precipitation_fallback("test", ["SN1"], START, END, **kwargs)

    def test_exact_144_slot_sum_and_reporting_day(self):
        rows = self.rows(day_readings())
        self.assertEqual(rows[0][:3], ("SN1", "2026-02-01T00:00:00Z", downloader.PRECIP_DAILY))
        self.assertAlmostEqual(rows[0][3], 14.4)

    def test_90_percent_threshold_counts_missing_as_zero_without_scaling(self):
        self.assertEqual(downloader.PRECIP_MIN_DAILY_READINGS, 130)
        self.assertAlmostEqual(self.rows(day_readings()[:130])[0][3], 13.0)
        self.assertEqual(self.rows(day_readings()[:129]), [])
        self.assertAlmostEqual(self.rows(day_readings()[:-1])[0][3], 14.3)
        self.assertEqual(self.rows([]), [])

    def test_duplicate_intervals_are_not_summed_twice(self):
        readings = day_readings()
        self.assertAlmostEqual(self.rows(readings + readings)[0][3], 14.4)
        self.assertAlmostEqual(self.rows(readings[:-1] + [readings[0]])[0][3], 14.3)
        self.assertEqual(self.rows(readings[:129] * 2), [])

    def test_conflicting_duplicates_fail_explicitly(self):
        readings = day_readings()[:130]
        duplicate = day_readings(value=0.2)[0]
        with self.assertRaisesRegex(ValueError, "Conflicting precipitation"):
            self.rows(readings + [duplicate])

    def test_invalid_rain_and_bad_quality_make_day_incomplete(self):
        for value, quality in ((-1, 0), (float("nan"), 0), (float("inf"), 0),
                               (None, 0), (0.1, 5), (0.1, 6), (0.1, 7)):
            readings = day_readings()[:130]
            readings[0]["observations"][0].update(value=value, qualityCode=quality)
            with self.subTest(value=value, quality=quality):
                self.assertEqual(self.rows(readings), [])

    def test_other_sensor_is_not_combined(self):
        readings = day_readings()[:129]
        other_sensor = day_readings()[-1]
        other_sensor["observations"][0]["timeSeriesId"] = 1
        self.assertEqual(self.rows(readings + [other_sensor]), [])

    def test_version_change_reprocesses_previously_rejected_days(self):
        with closing(sqlite3.connect(self.database)) as cx, cx:
            cx.execute("INSERT INTO precipitation_fallback_status VALUES(?,?,?,?)",
                       ("SN1", "2026-02", downloader._iso(END), "complete144_06utc_v1"))
        self.assertEqual(self.download(day_readings()[:130]), 1)
        with closing(sqlite3.connect(self.database)) as cx:
            version = cx.execute("SELECT version FROM precipitation_fallback_status").fetchone()[0]
            self.assertEqual(version, downloader.PRECIP_AGGREGATION_VERSION)
        self.assertEqual(self.download(day_readings()[:130]), 0)

    def test_month_boundary_and_first_interval_after_six(self):
        readings = day_readings() + day_readings(START + timedelta(days=1), value=0.2)
        rows = self.rows(readings)
        self.assertEqual([row[1][:10] for row in rows], ["2026-02-01", "2026-02-02"])
        self.assertAlmostEqual(rows[1][3], 28.8)

    def test_current_incomplete_day_is_not_published(self):
        self.assertEqual(self.rows(day_readings(), end=START + timedelta(hours=5)), [])
        self.assertEqual(len(self.rows(day_readings(), end=START + timedelta(hours=6))), 1)

    def test_daily_and_dual_resolution_stations_are_excluded(self):
        def series(url, params, client):
            ids = ["SN1", "SN2"] if params["elements"] == downloader.PRECIP_DAILY else ["SN2", "SN3"]
            return iter({"sourceId": f"{sid}:0", "timeSeriesId": 0, "timeResolution": "PT10M",
                         "validFrom": downloader._iso(START)}
                        for sid in ids)
        with patch.object(downloader, "_request_all", side_effect=series), redirect_stdout(StringIO()):
            self.assertEqual(downloader.precipitation_fallback_sources(
                "test", [["SN1", "SN2", "SN3"]], START, END), {"SN3": [(START, END)]})

    def test_success_is_resumable_and_stores_only_canonical_daily_data(self):
        self.assertEqual(self.download(day_readings()), 1)
        with patch.object(downloader, "fetch_batch_recursive", side_effect=AssertionError("No repeat request")):
            self.assertEqual(downloader.download_precipitation_fallback("test", ["SN1"], START, END), 0)
        with closing(sqlite3.connect(self.database)) as cx, cx:
            self.assertEqual(cx.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 1)
            self.assertEqual(cx.execute("SELECT element FROM observations").fetchone()[0], downloader.PRECIP_DAILY)
            self.assertEqual(cx.execute("SELECT COUNT(*) FROM precipitation_daily_derived").fetchone()[0], 1)

    def test_native_daily_total_takes_priority_even_with_millisecond_timestamp(self):
        with closing(sqlite3.connect(self.database)) as cx, cx:
            cx.execute("INSERT INTO observations VALUES(?,?,?,?)",
                       ("SN1", "2026-02-01T00:00:00.000Z", downloader.PRECIP_DAILY, 9.0))
        self.assertEqual(self.download(day_readings()), 0)
        with closing(sqlite3.connect(self.database)) as cx, cx:
            self.assertEqual(cx.execute("SELECT value FROM observations").fetchall(), [(9.0,)])

    def test_native_download_replaces_derived_day_without_double_count(self):
        self.download(day_readings())
        native = [{"sourceId": "SN1:0", "referenceTime": "2026-02-01T00:00:00.000Z",
                   "observations": [{"elementId": downloader.PRECIP_DAILY, "value": 9.0}]}]
        with patch.object(downloader, "fetch_batch_recursive", return_value=iter(native)), redirect_stdout(StringIO()):
            downloader.download_month("test", downloader.PRECIP_DAILY, START, END, [["SN1"]])
        with closing(sqlite3.connect(self.database)) as cx, cx:
            self.assertEqual(cx.execute("SELECT value FROM observations").fetchall(), [(9.0,)])
            self.assertEqual(cx.execute("SELECT COUNT(*) FROM precipitation_daily_derived").fetchone()[0], 0)

    def test_request_error_does_not_mark_fallback_success(self):
        with patch.object(downloader, "fetch_batch_recursive", side_effect=RuntimeError("HTTP failed")), \
                redirect_stdout(StringIO()), self.assertRaises(RuntimeError):
            downloader.download_precipitation_fallback("test", ["SN1"], START, END)
        with closing(sqlite3.connect(self.database)) as cx, cx:
            self.assertEqual(cx.execute("SELECT COUNT(*) FROM precipitation_fallback_status").fetchone()[0], 0)

    def test_force_refresh_does_not_duplicate_daily_total(self):
        self.assertEqual(self.download(day_readings()), 1)
        self.assertEqual(self.download(day_readings(value=0.2), force=True), 1)
        with closing(sqlite3.connect(self.database)) as cx:
            values = cx.execute("SELECT value FROM observations").fetchall()
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(values[0][0], 28.8)

    def test_single_station_response_limit_is_split_by_time(self):
        def fetch(client, stations, element, start, end):
            if end - start > timedelta(days=1):
                raise RuntimeError("exceeds the maximum number allowed")
            return iter([{"start": start}])
        with patch.object(downloader, "fetch_batch", side_effect=fetch):
            result = list(downloader.fetch_batch_recursive(
                "test", ["SN1"], downloader.PRECIP_TEN_MINUTES, START, START + timedelta(days=2)))
        self.assertEqual(len(result), 2)

    def test_unrelated_http_412_is_not_reported_as_success(self):
        from unittest.mock import Mock
        response = Mock(status_code=412, text="invalid request", url="https://example.invalid")
        response.json.return_value = {"error": {"reason": "Invalid time interval"}}
        with patch.object(downloader.requests, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Frost HTTP 412"):
                list(downloader._request_all("https://example.invalid", {}, "test"))

    def test_real_frost_no_series_412_is_a_valid_empty_period(self):
        from unittest.mock import Mock
        response = Mock(status_code=412)
        response.json.return_value = {"error": {"reason":
            "No time series found for this combination of parameters, check /observations/availableTimeSeries for more information."}}
        with patch.object(downloader.requests, "get", return_value=response), redirect_stdout(StringIO()):
            self.assertEqual(list(downloader._request_all("https://example.invalid", {}, "test")), [])

    def test_existing_daily_month_completion_does_not_skip_fallback(self):
        with closing(sqlite3.connect(self.database)) as cx, cx:
            cx.execute("INSERT INTO download_status VALUES(?,?,1)", (downloader.PRECIP_DAILY, "2026-02"))
        with patch.object(downloader, "station_id_batches", return_value=iter([["SN1", "SN2", "SN3"]])), \
                patch.object(downloader, "_months", return_value=iter([(START, END)])), \
                patch.object(downloader, "precipitation_fallback_sources", return_value={"SN3": [(START, END)]}), \
                patch.object(downloader, "fetch_batch_recursive", return_value=iter(day_readings(sid="SN3"))) as fetch, \
                patch.dict(downloader.os.environ, {downloader.FROST_CLIENT_ID_ENV: "test"}), \
                redirect_stdout(StringIO()):
            downloader.run("precipitation", 1)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(fetch.call_args.args[1:3], (["SN3"], downloader.PRECIP_TEN_MINUTES))

    def test_derived_daily_totals_are_used_by_occurrence_model_pair_training(self):
        rows = []
        for day in range(120):
            for index in range(3):
                sid = f"SN{index}"
                rain = float((day + index) % 4) if (day + index) % 3 else 0.0
                rows.append((sid, downloader._iso(START + timedelta(days=day)), downloader.PRECIP_DAILY, rain))
        with closing(sqlite3.connect(self.database)) as cx, cx:
            cx.executemany("INSERT INTO observations VALUES(?,?,?,?)", rows)
        downloader._touch_marker(downloader.PRECIP_DAILY)
        paths = information_gain.paths("precipitation", 1, "precipitation_occurrence_error",
                                       output_dir=Path(self.temp.name))
        stations = pd.DataFrame({
            "station_id": ["SN0", "SN1", "SN2"], "x": [0., 1000., 2000.], "y": [0., 1000., 2000.],
            "elevation": [10., 20., 30.], "coast_dist_fjord": [1., 2., 3.],
            "coast_dist_ocean": [1., 2., 3.], "objtype": ["Skog"] * 3,
            **{feature: [0., 1., 2.] for feature in information_gain.TERRAIN}})
        with patch.object(information_gain, "DATABASE", self.database), redirect_stdout(StringIO()):
            pairs, selected = information_gain.build_pairs(
                "precipitation", 1, stations, paths, "precipitation_occurrence_error")
        self.assertEqual(set(selected.index), {"SN0", "SN1", "SN2"})
        self.assertEqual(len(pairs), 3)
        self.assertTrue(pairs.target_error.between(0, 1).all())
        self.assertIn("observation_update_ns", paths["metadata"].read_text())

    def test_no_ten_minute_requests_outside_series_lifetime(self):
        with patch.object(downloader, "station_id_batches", return_value=iter([["SN1"]])), \
                patch.object(downloader, "_months", return_value=iter([(START, END)])), \
                patch.object(downloader, "precipitation_fallback_sources",
                             return_value={"SN1": [(END + timedelta(days=1), END + timedelta(days=30))]}), \
                patch.object(downloader, "download_month", return_value=0), \
                patch.object(downloader, "download_precipitation_fallback") as download, \
                patch.dict(downloader.os.environ, {downloader.FROST_CLIENT_ID_ENV: "test"}), \
                redirect_stdout(StringIO()):
            downloader.run("precipitation", 1)
        download.assert_not_called()

    def test_daily_batch_commits_data_before_resume_status(self):
        def fail(*args):
            raise RuntimeError("status write failed")
        native = [{"sourceId": "SN1:0", "referenceTime": downloader._iso(START),
                   "observations": [{"elementId": downloader.PRECIP_DAILY, "value": 9.0}]}]
        with patch.object(downloader, "fetch_batch_recursive", return_value=iter(native)), \
                patch.object(downloader, "mark_batch_complete", side_effect=fail), \
                redirect_stdout(StringIO()), self.assertRaises(RuntimeError):
            downloader.download_month("test", downloader.PRECIP_DAILY, START, END, [["SN1"]])
        with closing(sqlite3.connect(self.database)) as cx, cx:
            self.assertEqual(cx.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 0)
            self.assertEqual(cx.execute("SELECT COUNT(*) FROM batch_status").fetchone()[0], 0)

    def test_queue_retrains_operational_model_then_prepares_both_networks(self):
        commands = overnight_run.commands_for(overnight_run.parse_args(["--plan", "precipitation-priority"]))
        metrics = [command[command.index("--metric") + 1] for _, command in commands if "--metric" in command]
        self.assertEqual(metrics, ["precipitation_occurrence_error", "prediction_rmse"])
        self.assertTrue(all("--all-candidates" not in command for _, command in commands))
        final = commands[-1][1]
        self.assertEqual(final[final.index("--station-network") + 1], "both")

    def test_training_run_never_calls_placement_functions_and_invalidates_old_model(self):
        paths = information_gain.paths("precipitation", 5, "precipitation_occurrence_error",
                                       output_dir=Path(self.temp.name))
        paths["metadata"].write_text("old")
        stations = gpd.GeoDataFrame({"station_id": ["SN1"]}, geometry=gpd.points_from_xy([1], [1]), crs=25833)
        norway = gpd.GeoDataFrame(geometry=stations.buffer(1), crs=25833)
        def build(*args, **kwargs):
            paths["metadata"].write_text("new")
            return pd.DataFrame({"target_error": [0.1]}), stations
        with patch.object(information_gain, "load_grid", return_value=stations), \
                patch.object(information_gain, "station_features", return_value=stations), \
                patch.object(information_gain.pyogrio, "read_dataframe", return_value=norway), \
                patch.object(information_gain, "build_pairs", side_effect=build), \
                patch.object(information_gain, "fit_error_model") as fit, \
                patch.object(information_gain, "evaluate", side_effect=AssertionError("No placement gain")), \
                patch.object(information_gain, "current_error", side_effect=AssertionError("No national baseline")), \
                redirect_stdout(StringIO()):
            information_gain.run("precipitation", 5, metric="precipitation_occurrence_error",
                                 output_dir=Path(self.temp.name))
        self.assertTrue(fit.call_args.kwargs["force"])


if __name__ == "__main__":
    unittest.main()
