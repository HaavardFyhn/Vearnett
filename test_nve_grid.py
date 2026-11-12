"""Tests for native NVE cell alignment, response handling and ArcGIS exports."""
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

import numpy as np

import download_nve_grid as nve


def response(x=262500, y=6650500, theme="tm", data=None):
    return dict(Theme=theme, FullName="Temperature", X=x, Y=y, Unit="Celcius",
                StartDate="15.01.2025 06:00:00", EndDate="16.01.2025 06:00:00",
                TimeResolution=1440, NoDataValue=65535,
                Data=[-0.1, 0.9] if data is None else data)


class NveGridTests(unittest.TestCase):
    def test_grid_is_aligned_north_up_and_point_snaps_to_containing_cell(self):
        xs, ys = nve.grid_centres(None, [262100, 6649100, 264000, 6651000])
        np.testing.assert_array_equal(xs, [262500, 263500])
        np.testing.assert_array_equal(ys, [6650500, 6649500])
        xs, ys = nve.grid_centres([-1, 6650123], None)
        np.testing.assert_array_equal(xs, [-500])
        np.testing.assert_array_equal(ys, [6650500])

    def test_no_data_is_not_exported_as_weather_and_values_are_not_rescaled(self):
        metadata, values = nve.parse_series(response(data=[65535, -0.1]), "tm", 262500, 6650500)
        np.testing.assert_allclose(values, [nve.NODATA, -0.1])
        self.assertEqual(metadata["start"], "2025-01-15T06:00:00")
        _, values = nve.parse_series(response(data=[None, 0]), "tm", 262500, 6650500)
        np.testing.assert_array_equal(values, [nve.NODATA, 0])

    def test_invalid_series_is_rejected(self):
        cases = [
            dict(Data=[1]), dict(X=1), dict(Theme="rr"), dict(TimeResolution=0),
            dict(TimeResolution=float("nan")), dict(Data=[[1, 2]]),
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                nve.parse_series(dict(response(), **changes), "tm", 262500, 6650500)

    def test_three_hour_time_axis(self):
        payload = dict(response(), StartDate="15.01.2025 03:00:00",
                       EndDate="15.01.2025 09:00:00", TimeResolution=180, Data=[1, 2, 3])
        metadata, values = nve.parse_series(payload, "tm", 262500, 6650500)
        self.assertEqual(metadata["step_minutes"], 180)
        self.assertEqual(len(values), 3)

    def test_large_grid_is_rejected_before_allocating_coordinates(self):
        with patch.object(np, "arange", side_effect=AssertionError("Must not allocate")):
            with self.assertRaisesRegex(ValueError, "exceeds --max-cells"):
                nve.grid_centres(None, [0, 0, 1e12, 1e12])

    def test_exports_exact_native_grid_and_resumes_without_network(self):
        import rasterio

        with tempfile.TemporaryDirectory() as directory:
            args = nve.parse_args([
                "--start", "2025-01-15", "--end", "2025-01-16",
                "--bbox", "262000", "6649000", "264000", "6651000",
                "--themes", "tm", "--output-dir", directory, "--request-delay", "0",
            ])
            def get(url, **kwargs):
                fields = url.split("/")
                x, y = int(fields[-5]), int(fields[-4])
                value = (x - 262500) / 1000 + (6650500 - y) / 500
                result = Mock()
                result.__enter__ = Mock(return_value=result)
                result.__exit__ = Mock(return_value=False)
                result.json.return_value = response(x, y, data=[value, value + 10])
                return result
            with patch.object(requests.Session, "get", side_effect=get) as request:
                paths = nve.download(args)
                self.assertEqual(request.call_count, 4)
            self.assertEqual(len(paths), 2)
            with rasterio.open(paths[0]) as source:
                self.assertEqual(source.crs.to_epsg(), 32633)
                self.assertEqual(source.res, (1000, 1000))
                self.assertEqual(tuple(source.bounds), (262000, 6649000, 264000, 6651000))
                self.assertEqual(source.xy(0, 0), (262500, 6650500))
                self.assertEqual(source.nodata, nve.NODATA)
                self.assertEqual(source.tags()["valid_time"], "2025-01-15T06:00:00")
                np.testing.assert_array_equal(source.read(1), [[0, 1], [2, 3]])
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["series"][0]["valid_values"], 8)
            with patch.object(requests.Session, "get", side_effect=AssertionError("Must use cache")):
                self.assertEqual(nve.download(args), paths)
            args.end = "2025-01-17"
            with self.assertRaisesRegex(ValueError, "another query"):
                nve.download(args)

    def test_dry_run_does_not_create_output_or_request(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            with patch.object(requests.Session, "get", side_effect=AssertionError("No network")):
                result = nve.main(["--point", "262500", "6650500", "--start", "2025-01-15",
                                   "--end", "2025-01-15", "--output-dir", str(output), "--dry-run"])
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())

    def test_http_failure_is_explicit_and_nonzero(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(StringIO()) as stderr:
            with patch.object(requests.Session, "get", side_effect=requests.HTTPError("503")):
                result = nve.main(["--point", "262500", "6650500", "--start", "2025-01-15",
                                   "--end", "2025-01-15", "--output-dir", directory])
            self.assertEqual(result, 1)
            self.assertIn("503", stderr.getvalue())
            self.assertFalse((Path(directory) / "manifest.json").exists())

    def test_only_explicit_missing_cell_error_is_masked(self):
        error = {"Error": "Tema: tm. No cell exists for coordinates 262500, 6650500."}
        self.assertTrue(nve.is_missing_cell(error, "tm", 262500, 6650500))
        self.assertFalse(nve.is_missing_cell(error, "rr", 262500, 6650500))
        self.assertFalse(nve.is_missing_cell({"Error": "Invalid date"}, "tm", 262500, 6650500))

    def test_missing_cell_is_masked_in_export_and_recorded(self):
        import rasterio

        with tempfile.TemporaryDirectory() as directory:
            args = nve.parse_args([
                "--start", "2025-01-15", "--end", "2025-01-16",
                "--bbox", "262000", "6650000", "264000", "6651000",
                "--themes", "tm", "--output-dir", directory, "--request-delay", "0",
            ])
            def get(url, **kwargs):
                result = Mock()
                result.__enter__ = Mock(return_value=result)
                result.__exit__ = Mock(return_value=False)
                if "/262500/" in url:
                    result.status_code = 400
                    result.json.return_value = {
                        "Error": "Tema: tm. No cell exists for coordinates 262500, 6650500.",
                    }
                else:
                    result.status_code = 200
                    result.json.return_value = response(x=263500)
                return result
            with patch.object(requests.Session, "get", side_effect=get):
                paths = nve.download(args)
            with rasterio.open(paths[0]) as source:
                self.assertEqual(source.read(1, masked=True).mask.tolist(), [[True, False]])
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertEqual(manifest["series"][0]["missing_cells"], [[262500, 6650500]])

    def test_invalid_arguments_are_rejected(self):
        base = ["--point", "262500", "6650500", "--output-dir", "unused"]
        for dates in [
            ["--start", "2025-01-16", "--end", "2025-01-15"],
            ["--start", "2025-01-15T03", "--end", "2025-01-15"],
            ["--start", "bad", "--end", "2025-01-15"],
        ]:
            with self.subTest(dates=dates), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                nve.parse_args(base + dates)


if __name__ == "__main__":
    unittest.main()
