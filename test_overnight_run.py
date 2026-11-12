"""Overnight humidity tests using mocked HTTP/processes and temporary observations."""
from contextlib import closing, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import download_frost
import information_gain
import main
import overnight_run
from pipeline_config import PHENOMENA


class OvernightTests(unittest.TestCase):
    def test_humidity_cli_and_configuration(self):
        with patch.object(sys, "argv", ["main.py", "--phenomenon", "humidity", "--years", "5"]):
            args = main.parse_args()
        self.assertEqual(args.phenomenon, "humidity")
        self.assertEqual(PHENOMENA["humidity"]["element"], "relative_humidity")
        self.assertEqual(PHENOMENA["humidity"]["daily_agg"], "mean")

    def test_selected_queue_is_humidity_only_and_resumable(self):
        commands = overnight_run.commands_for(overnight_run.parse_args(["--phenomenon", "humidity"]))
        self.assertEqual(len(commands), 5)
        for _, command in commands[1:]:
            self.assertEqual(command[command.index("--phenomenon") + 1], "humidity")
            self.assertEqual(command[command.index("--years") + 1], "5")
            self.assertNotIn("--force", command)
            self.assertNotIn("--force-download", command)
        self.assertEqual([command[command.index("--grid-spacing-m") + 1] for _, command in commands[2:]],
                         ["5000", "2500", "1000"])
        self.assertEqual(sum("--to-step" in command and command[command.index("--to-step") + 1] == "download"
                             for _, command in commands), 1)
        self.assertNotIn("--all-candidates", commands[-1][1])

    def test_skip_download_and_force_only_affect_selected_analysis(self):
        args = overnight_run.parse_args(["--phenomenon", "humidity", "--skip-download", "--force"])
        commands = overnight_run.commands_for(args)
        self.assertEqual(len(commands), 4)
        for _, command in commands[1:]:
            self.assertIn("--force", command)
            self.assertIn("--skip-download", command)
        self.assertTrue(all("--force-download" not in command for _, command in commands))

    def test_grid_selection_alias_and_order(self):
        args = overnight_run.parse_args(["--phenomenon", "humidity", "--grid-spacing", "1000", "5000", "2500", "5000"])
        self.assertEqual(args.grid_spacing_m, [5000, 2500, 1000])
        single = overnight_run.commands_for(overnight_run.parse_args(["--phenomenon", "humidity", "--grid-spacing-m", "2500"]))
        self.assertEqual(len(single), 3)
        self.assertEqual(single[-1][1][single[-1][1].index("--grid-spacing-m") + 1], "2500")

    def test_analysis_outputs_are_isolated_per_grid(self):
        commands = overnight_run.commands_for(overnight_run.parse_args(["--phenomenon", "humidity"]))[2:]
        outputs = [Path(command[command.index("--output-dir") + 1]) for _, command in commands]
        self.assertEqual(len(set(outputs)), 3)
        for output in outputs:
            for radius in (None, 20000):
                paths = information_gain.paths("humidity", 5, "prediction_rmse", radius, output_dir=output)
                self.assertTrue(all(path.parent == output for path in paths.values()))
        legacy = information_gain.paths("humidity", 5, "prediction_rmse")
        self.assertEqual(legacy["model"].parent, information_gain.WORK)

    def test_main_accepts_output_directory_and_grid_alias(self):
        with patch.object(sys, "argv", ["main.py", "--phenomenon", "humidity", "--grid-spacing", "2500", "--output-dir", "outputs/humidity_2500m"]):
            args = main.parse_args()
        self.assertEqual(args.grid_spacing_m, 2500)
        self.assertEqual(args.output_dir, Path("outputs/humidity_2500m"))

    def test_legacy_queue_is_unchanged_when_no_phenomenon_selected(self):
        self.assertIs(overnight_run.commands_for(overnight_run.parse_args([])), overnight_run.COMMANDS)

    def test_temperature_priority_exact_analysis_order_and_outputs(self):
        args = overnight_run.parse_args(["--plan", "temperature-priority"])
        analyses = [command for _, command in overnight_run.commands_for(args) if "--metric" in command]
        expected = [("humidity", 5000), ("temperature", 2500), ("humidity", 2500),
                    ("temperature", 1000), ("humidity", 1000)]
        actual = [(command[command.index("--phenomenon") + 1], int(command[command.index("--grid-spacing-m") + 1]))
                  for command in analyses]
        self.assertEqual(actual, expected)
        self.assertEqual(len({command[command.index("--output-dir") + 1] for command in analyses}), 5)
        for command in analyses:
            self.assertEqual(command[command.index("--metric") + 1], "prediction_rmse")
            self.assertNotIn("--all-candidates", command)
            self.assertNotIn("--force-download", command)

    def test_temperature_priority_downloads_once_just_before_first_use(self):
        commands = [command for _, command in overnight_run.commands_for(
            overnight_run.parse_args(["--plan", "temperature-priority"]))]
        downloads = [(index, command[command.index("--phenomenon") + 1]) for index, command in enumerate(commands)
                     if "--to-step" in command and command[command.index("--to-step") + 1] == "download"]
        self.assertEqual(downloads, [(1, "humidity"), (3, "temperature")])
        self.assertIn("--metric", commands[2])
        self.assertTrue(all("--force" not in commands[index] for index, _ in downloads))

    def test_temperature_priority_skip_download_force_and_fail_fast(self):
        args = overnight_run.parse_args(["--plan", "temperature-priority", "--skip-download", "--force"])
        commands = overnight_run.commands_for(args)
        self.assertEqual(len(commands), 6)
        for _, command in commands[1:]:
            self.assertIn("--force", command)
            self.assertIn("--skip-download", command)
        with patch.object(overnight_run, "check_prerequisites"), \
                patch.object(overnight_run, "LOG"), \
                patch.object(overnight_run, "run_queue", return_value=0) as run:
            self.assertEqual(overnight_run.main(["--plan", "temperature-priority", "--skip-download"]), 0)
        self.assertTrue(run.call_args.kwargs["stop_on_error"])

    def test_priority_plan_rejects_conflicting_overrides(self):
        for arguments in [["--plan", "temperature-priority", "--phenomenon", "humidity"],
                          ["--plan", "temperature-priority", "--grid-spacing", "2500"]]:
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                overnight_run.parse_args(arguments)

    def test_dry_run_has_no_execution_or_logs(self):
        with patch.object(overnight_run, "run_queue", side_effect=AssertionError("Must not execute")), \
                patch.object(overnight_run, "check_prerequisites", side_effect=AssertionError("Must not inspect inputs")), \
                patch.object(overnight_run, "emit", side_effect=AssertionError("Must not log")), \
                redirect_stdout(StringIO()):
            self.assertEqual(overnight_run.main(["--phenomenon", "humidity", "--dry-run"]), 0)

    def test_selected_queue_stops_after_failure(self):
        process = Mock(stdout=None)
        process.wait.return_value = 1
        with patch.object(overnight_run.subprocess, "Popen", return_value=process) as launch, \
                patch.object(overnight_run, "emit"):
            result = overnight_run.run_queue([("download", ["download"]), ("analysis", ["analysis"])], stop_on_error=True)
        self.assertEqual(result, 1)
        self.assertEqual(launch.call_count, 1)

    def test_humidity_class_metrics_fail_before_any_pipeline_work(self):
        args = SimpleNamespace(phenomenon="humidity", years=5, metric="temperature_class_error")
        with patch.object(main, "parse_args", return_value=args):
            with self.assertRaisesRegex(ValueError, "prediction_rmse only"):
                main.main()

    def test_preflight_checks_inputs_without_database_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            polygon = root / "outputs/norway_polygon.gpkg"
            for spacing in (5000, 2500, 1000):
                (root / "outputs" / f"norway_grid_{spacing}m.gpkg").touch()
            polygon.touch()
            database = root / "frost.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE stations(station_id TEXT,inside_norway INTEGER)")
                connection.execute("INSERT INTO stations VALUES('SN1',1)")
                connection.commit()
            modified = database.stat().st_mtime_ns
            args = overnight_run.parse_args(["--phenomenon", "humidity", "--skip-download"])
            with patch.object(overnight_run, "ROOT", root), patch.object(overnight_run, "DATABASE", database), \
                    patch.object(overnight_run, "NORWAY_POLYGON_FILE", polygon), redirect_stdout(StringIO()):
                overnight_run.check_prerequisites(args)
                (root / "outputs/norway_grid_2500m.gpkg").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "2500m"):
                    overnight_run.check_prerequisites(args)
            self.assertEqual(database.stat().st_mtime_ns, modified)


class DownloadSafetyTests(unittest.TestCase):
    def test_exhausted_http_retries_raise(self):
        with patch.object(download_frost.requests, "get", return_value=Mock(status_code=503)) as request, \
                patch.object(download_frost.time_module, "sleep"), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, "6 attempts"):
                list(download_frost._request_all("https://example.invalid", {}, "test"))
        self.assertEqual(request.call_count, 6)

    def test_no_matching_series_remains_a_valid_empty_result(self):
        response = Mock(status_code=412)
        response.json.return_value = {"error": {"reason": "No matching time series"}}
        with patch.object(download_frost.requests, "get", return_value=response), redirect_stdout(StringIO()):
            self.assertEqual(list(download_frost._request_all("https://example.invalid", {}, "test")), [])

    def test_failed_batch_does_not_mark_month_or_batch_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "frost.db"
            connections = []
            original = download_frost._connect

            def connect():
                connection = original()
                connections.append(connection)
                return connection

            try:
                with patch.object(download_frost, "DATABASE", database), \
                        patch.object(download_frost, "_connect", side_effect=connect), \
                        patch.object(download_frost, "fetch_batch_recursive", side_effect=RuntimeError("failed")), \
                        redirect_stdout(StringIO()):
                    with self.assertRaises(RuntimeError):
                        download_frost.download_month("test", "relative_humidity",
                            datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 2, 1, tzinfo=timezone.utc), [["SN1"]])
            finally:
                for connection in connections:
                    connection.close()
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM download_status").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM batch_status").fetchone()[0], 0)


class HumidityAnalysisTests(unittest.TestCase):
    def test_current_error_uses_lowest_predicted_single_station_error(self):
        class DistanceErrorModel:
            def predict(self, features):
                return features[:, 0]

        def cache(x):
            size = len(x)
            return {
                "x": np.asarray(x, dtype=np.float32),
                "y": np.zeros(size, dtype=np.float32),
                "elevation": np.zeros(size, dtype=np.float32),
                "coast_dist_fjord": np.zeros(size, dtype=np.float32),
                "coast_dist_ocean": np.zeros(size, dtype=np.float32),
                "objtype_code": np.zeros(size, dtype=np.int16),
                **{feature: np.zeros(size, dtype=np.float32) for feature in information_gain.TERRAIN},
            }

        stations = cache(np.arange(10, dtype=np.float32))
        targets = cache(np.array([2.2], dtype=np.float32))
        error = information_gain.current_error(
            DistanceErrorModel(), targets, stations, "prediction_rmse",
        )

        self.assertAlmostEqual(error[0], 0.2, places=5)

    def test_candidate_gain_is_nonnegative_under_minimum_baseline(self):
        class DistanceErrorModel:
            def predict(self, features):
                return features[:, 0]

        candidate = {
            "x": np.array([2.0, 0.5], dtype=np.float32),
            "y": np.zeros(2, dtype=np.float32),
            "elevation": np.zeros(2, dtype=np.float32),
            "coast_dist_fjord": np.zeros(2, dtype=np.float32),
            "coast_dist_ocean": np.zeros(2, dtype=np.float32),
            "objtype_code": np.zeros(2, dtype=np.int16),
            **{field: np.zeros(2, dtype=np.float32) for field in information_gain.TERRAIN},
        }
        target = {
            "x": np.array([0.0], dtype=np.float32),
            "y": np.array([0.0], dtype=np.float32),
            "elevation": np.array([0.0], dtype=np.float32),
            "coast_dist_fjord": np.array([0.0], dtype=np.float32),
            "coast_dist_ocean": np.array([0.0], dtype=np.float32),
            "objtype_code": np.array([0], dtype=np.int16),
            "error": np.array([1.0], dtype=np.float32),
            **{field: np.array([0.0], dtype=np.float32) for field in information_gain.TERRAIN},
        }

        gain, improved = information_gain.evaluate(
            candidate,
            pd.DataFrame({"x": [0.0], "y": [0.0]}),
            DistanceErrorModel(),
            "prediction_rmse",
            target_cache=target,
        )

        np.testing.assert_allclose(gain, [0.0, 0.5], atol=1e-6)
        np.testing.assert_array_equal(improved, [0, 1])

    def test_wind_is_a_supported_frost_phenomenon(self):
        with patch.object(sys, "argv", ["main.py", "--phenomenon", "wind", "--years", "5"]):
            args = main.parse_args()
        self.assertEqual(args.phenomenon, "wind")
        self.assertEqual(PHENOMENA["wind"]["element"], "wind_speed")

    def test_wind_priority_plan_downloads_wind_then_retrains_requested_models(self):
        args = overnight_run.parse_args(["--plan", "wind-priority"])
        commands = overnight_run.commands_for(args)
        self.assertEqual(len(commands), 3)
        download = commands[1][1]
        self.assertEqual(download[download.index("--phenomenon") + 1], "wind")
        self.assertEqual(download[download.index("--from-step") + 1], "download")
        analyses = [command for _, command in commands if "--metric" in command]
        actual = [(command[command.index("--phenomenon") + 1], command[command.index("--metric") + 1])
                  for command in analyses]
        self.assertEqual(actual, [("wind", "prediction_rmse")])
        for command in analyses:
            self.assertIn("--force", command)
            self.assertNotIn("--all-candidates", command)
            self.assertIn("--skip-download", command)
            self.assertEqual(command[command.index("--grid-spacing-m") + 1], "5000")

    def test_wind_priority_download_can_be_skipped_and_grid_is_fixed(self):
        commands = overnight_run.commands_for(
            overnight_run.parse_args(["--plan", "wind-priority", "--skip-download"])
        )
        self.assertEqual(len(commands), 2)
        self.assertFalse(any(
            "--to-step" in command and command[command.index("--to-step") + 1] == "download"
            for _, command in commands
        ))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            overnight_run.parse_args(["--plan", "wind-priority", "--grid-spacing-m", "2500"])

    def test_road_temperature_plan_downloads_then_trains_continuous_rmse_at_5000m(self):
        self.assertEqual(PHENOMENA["road_temperature"]["element"], "mean(road_surface_temperature PT1M)")
        self.assertEqual(PHENOMENA["road_temperature"]["daily_agg"], "mean")
        commands = overnight_run.commands_for(overnight_run.parse_args(["--plan", "road-temperature-priority"]))
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[1][1][commands[1][1].index("--phenomenon") + 1], "road_temperature")
        analysis = commands[2][1]
        self.assertEqual(analysis[analysis.index("--metric") + 1], "prediction_rmse")
        self.assertEqual(analysis[analysis.index("--grid-spacing-m") + 1], "5000")
        self.assertTrue(analysis[analysis.index("--output-dir") + 1].endswith("road_temperature_5y_5000m"))

    def test_minute_values_are_stored_as_daily_station_means(self):
        import download_frost
        sums = {}
        for sid, ref, value in [("SN1", "2024-01-01T00:01:00.000Z", 1.0), ("SN1", "2024-01-01T23:59:00.000Z", 3.0),
                                ("SN1", "2024-01-02T00:00:00.000Z", -2.0)]:
            total, n = sums.get((sid, ref[:10]), (0.0, 0))
            sums[(sid, ref[:10])] = (total + value, n + 1)
        rows = sorted(download_frost._daily_mean_rows(sums, "mean(road_surface_temperature PT1M)"))
        self.assertEqual(rows, [("SN1", "2024-01-01T00:00:00Z", "mean(road_surface_temperature PT1M)", 2.0),
                                ("SN1", "2024-01-02T00:00:00Z", "mean(road_surface_temperature PT1M)", -2.0)])

    def test_humidity_daily_pairs_use_existing_continuous_model_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "humidity.db"
            dates = pd.date_range("2022-01-01", periods=800, tz="UTC")
            rows = [(identifier, date.isoformat(), "relative_humidity",
                     float(60 + 12 * np.sin(day / 37) + index * 2 + np.cos(day / (11 + index))))
                    for index, identifier in enumerate(["SN1", "SN2", "SN3"])
                    for day, date in enumerate(dates)]
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("CREATE TABLE observations(station_id TEXT,reference_time TEXT,element TEXT,value REAL)")
                connection.executemany("INSERT INTO observations VALUES(?,?,?,?)", rows)
                connection.commit()
            stations = pd.DataFrame({"station_id": ["SN1", "SN2", "SN3"], "x": [0, 1000, 3000], "y": [0, 0, 1000],
                "elevation": [10, 20, 30], "coast_dist_fjord": [100, 200, 300], "coast_dist_ocean": [1000, 2000, 3000],
                "objtype": ["Skog"] * 3, **{field: [0., 1., 2.] for field in information_gain.TERRAIN}})
            paths = dict(pairs=root / "pairs.csv", metadata=root / "metadata.json")
            with patch.object(information_gain, "DATABASE", database), redirect_stdout(StringIO()):
                pairs, selected = information_gain.build_pairs("humidity", 5, stations, paths, "prediction_rmse")
            self.assertEqual(len(pairs), 3)
            self.assertTrue(np.isfinite(pairs["target_error"]).all())
            self.assertEqual(information_gain.paths("humidity", 5, "prediction_rmse")["model"].name,
                             "humidity_5y_prediction_rmse_model.joblib")


if __name__ == "__main__":
    unittest.main()