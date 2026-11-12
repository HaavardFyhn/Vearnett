"""Tests for whole-country seNorge daily downloads and grid exports."""
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import netCDF4
import numpy as np

import download_senorge_norway as national


def create_daily(path):
    with netCDF4.Dataset(path, "w") as ds:
        for name, length in [("time", 1), ("Y", 2), ("X", 2)]:
            ds.createDimension(name, length)
        ds.createVariable("X", "f8", ("X",))[:] = [262500, 263500]
        ds.createVariable("Y", "f8", ("Y",))[:] = [6650500, 6651500]
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "hours since 2025-01-13 00:00:00"
        t[:] = [6]
        projection = ds.createVariable("UTM_Zone_33", "i4")
        projection.proj4 = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"
        for name, units in [("tg", "Celsius"), ("rr", "kg/m^2")]:
            var = ds.createVariable(name, "f4", ("time", "Y", "X"), fill_value=-999.99)
            var[:] = [[[1, 2], [3, -999.99]]]
            var.units = units
            var.grid_mapping = "UTM_Zone_33"
            var.long_name = "daily value, 06 UTC to previous day 06 UTC"
        ds.institution = "MET Norway"
        ds.license = "CC-BY-4.0"
        ds.version = "test"


class NationalSeNorgeTests(unittest.TestCase):
    def test_reuse_references_original_without_copying(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, reuse = root / "new", root / "old"
            output.mkdir()
            reuse.mkdir()
            original = reuse / "seNorge2018_2024.nc"
            original.write_bytes(b"retained original")
            record = dict(year=2024, netcdf=str(original.resolve()), bytes=original.stat().st_size,
                          source="https://example.test/2024.nc", days=366, width=2, height=2)
            with patch.object(national, "fetch_archive") as fetch, patch.object(
                national, "validate_archive", return_value=record
            ) as validate:
                national.collect_annual(date(2024, 1, 1), date(2024, 12, 31),
                                        output, Mock(), False, reuse)
            self.assertEqual(fetch.call_args.args[2], original)
            validate.assert_called_once_with(original, 2024)
            self.assertFalse((output / "netcdf" / original.name).exists())
            self.assertEqual(original.read_bytes(), b"retained original")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["archives"][0]["netcdf"], str(original.resolve()))
            self.assertEqual(manifest["status"], "complete")

    def test_archive_transfer_resumes_with_if_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seNorge2018_2021.nc"
            path.with_suffix(".pending.nc").write_bytes(b"first")
            national.write_json(path.with_suffix(".transfer.json"), dict(
                url=f"{national.ARCHIVE_URL}/seNorge2018_2021.nc",
                total=10, validator='"revision1"',
            ))
            session = MagicMock()
            response = session.get.return_value.__enter__.return_value
            response.status_code = 206
            response.headers = {"Content-Length": "5", "Content-Range": "bytes 5-9/10",
                                "ETag": '"revision1"'}
            response.iter_content.return_value = [b"last!"]
            national.fetch_archive(session, 2021, path)
            self.assertEqual(path.read_bytes(), b"firstlast!")
            self.assertEqual(session.get.call_args.kwargs["headers"]["Range"], "bytes=5-")
            self.assertEqual(session.get.call_args.kwargs["headers"]["If-Range"], '"revision1"')

    def test_server_ignoring_range_replaces_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "year.nc"
            path.with_suffix(".pending.nc").write_bytes(b"old")
            national.write_json(path.with_suffix(".transfer.json"), dict(
                url=f"{national.ARCHIVE_URL}/seNorge2018_2021.nc", total=10, validator='"old"',
            ))
            session = MagicMock()
            response = session.get.return_value.__enter__.return_value
            response.status_code = 200
            response.headers = {"Content-Length": "4", "ETag": '"new"'}
            response.iter_content.return_value = [b"new!"]
            national.fetch_archive(session, 2021, path)
            self.assertEqual(path.read_bytes(), b"new!")

    def test_incorrect_range_does_not_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "year.nc"
            path.with_suffix(".pending.nc").write_bytes(b"first")
            national.write_json(path.with_suffix(".transfer.json"), dict(
                url=f"{national.ARCHIVE_URL}/seNorge2018_2021.nc", total=10, validator='"v1"',
            ))
            session = MagicMock()
            response = session.get.return_value.__enter__.return_value
            response.status_code = 206
            response.headers = {"Content-Length": "6", "Content-Range": "bytes 4-9/10"}
            with self.assertRaisesRegex(ValueError, "Content-Range"):
                national.fetch_archive(session, 2021, path)
            self.assertEqual(path.with_suffix(".pending.nc").read_bytes(), b"first")
            self.assertFalse(path.exists())

    def test_failed_annual_stream_preserves_partial_for_restart(self):
        import requests

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "year.nc"
            session = MagicMock()
            response = session.get.return_value.__enter__.return_value
            response.status_code = 200
            response.headers = {"Content-Length": "10", "ETag": '"v1"'}
            response.iter_content.return_value = [b"first"]
            with patch.object(national.time, "sleep"), self.assertRaises(requests.RequestException):
                national.fetch_archive(session, 2021, path)
            self.assertEqual(path.with_suffix(".pending.nc").read_bytes(), b"first")
            self.assertFalse(path.exists())

    def test_full_leap_year_validation_and_nonzero_time_index_export(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "year.nc"
            with netCDF4.Dataset(path, "w") as ds:
                for name, length in [("time", 366), ("Y", 2), ("X", 2)]:
                    ds.createDimension(name, length)
                ds.createVariable("X", "f8", ("X",))[:] = [262500, 263500]
                ds.createVariable("Y", "f8", ("Y",))[:] = [6650500, 6651500]
                t = ds.createVariable("time", "f8", ("time",))
                t.units = "hours since 2024-01-01 00:00:00"
                t[:] = np.arange(366) * 24 + 6
                projection = ds.createVariable("UTM_Zone_33", "i4")
                projection.proj4 = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"
                projection.grid_mapping_name = "transverse_mercator"
                projection.utm_zone_number = 33
                for name, units in [("tg", "Celsius"), ("rr", "kg/m^2")]:
                    var = ds.createVariable(name, "f4", ("time", "Y", "X"), fill_value=-999.99)
                    var[:] = np.arange(366)[:, None, None] + np.ones((366, 2, 2))
                    var.units = units
                    var.grid_mapping = "UTM_Zone_33"
                    var.long_name = "daily value"
                ds.institution = "MET Norway"
                ds.license = "CC-BY-4.0"
                ds.version = "test"
            record = national.validate_archive(path, 2024)
            self.assertEqual(record["days"], 366)
            self.assertEqual(record["variables"]["tg"]["maximum"], 366)
            result = national.export_day(path, date(2024, 2, 29), folder, "url", time_index=59)
            self.assertEqual(result[0]["minimum"], 60)
            with self.assertRaisesRegex(ValueError, "every consecutive day"):
                national.validate_archive(path, 2023)
            with netCDF4.Dataset(path, "a") as ds:
                ds.variables["rr"][200] = -999.99
            with self.assertRaisesRegex(ValueError, "All rr cells are missing"):
                national.validate_archive(path, 2024)

    def test_inclusive_dates_cross_year_boundary(self):
        self.assertEqual(list(national.daily_dates(date(2024, 12, 31), date(2025, 1, 2))),
                         [date(2024, 12, 31), date(2025, 1, 1), date(2025, 1, 2)])

    def test_export_preserves_mask_units_time_and_north_up_orientation(self):
        import rasterio

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "day.nc"
            create_daily(path)
            records = national.export_day(path, date(2025, 1, 13), folder, "https://example.test/day.nc")
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["valid_cells"], 3)
            self.assertEqual(records[1]["units"], "mm")
            with rasterio.open(records[0]["geotiff"]) as raster:
                self.assertEqual(raster.crs.to_epsg(), 32633)
                self.assertEqual(raster.res, (1000, 1000))
                self.assertEqual(raster.tags()["valid_time"], "2025-01-13T06:00:00")
                np.testing.assert_array_equal(raster.read(1), [[3, national.NODATA], [1, 2]])

    def test_incorrect_day_and_resolution_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "day.nc"
            create_daily(path)
            with self.assertRaisesRegex(ValueError, "time does not match"):
                national.export_day(path, date(2025, 1, 14), folder, "url")
            with netCDF4.Dataset(path, "a") as ds:
                ds.variables["X"][:] = [262500, 264500]
            with self.assertRaisesRegex(ValueError, "native 1 km"):
                national.export_day(path, date(2025, 1, 13), folder, "url")

    def test_cached_file_avoids_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "day.nc"
            create_daily(path)
            session = Mock()
            national.fetch_file(session, date(2025, 1, 13), path)
            session.get.assert_not_called()

    def test_failed_download_cleans_pending_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "day.nc"
            session = MagicMock()
            response = session.get.return_value.__enter__.return_value
            response.headers = {"Content-Length": "10"}
            response.iter_content.return_value = [b"short"]
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                national.fetch_file(session, date(2025, 1, 13), path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".pending.nc").exists())


if __name__ == "__main__":
    unittest.main()
