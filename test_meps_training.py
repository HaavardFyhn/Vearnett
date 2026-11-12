"""Tests for the opt-in, sequential MEPS/Frost training-data downloader."""
from contextlib import closing, redirect_stderr
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from io import StringIO
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pyproj
import rasterio
import xarray as xr

import download_meps_training as meps


class MepsTrainingTests(unittest.TestCase):
    def test_archive_layout_switches_from_legacy_file_to_surface_ncml(self):
        self.assertEqual(meps.dataset_path(datetime(2023, 9, 30, tzinfo=timezone.utc)),
                         "2023/09/30/meps_det_2_5km_20230930T00Z.nc")
        self.assertEqual(meps.dataset_path(datetime(2023, 10, 1, 6, tzinfo=timezone.utc)),
                         "2023/10/01/meps_det_sfc_20231001T06Z.ncml")
        self.assertIn("/ncss/grid/", meps.ncss_url(datetime(2021, 10, 23, tzinfo=timezone.utc)))

    def test_dry_run_counts_sequential_requests_without_network_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "training.sqlite"
            with patch.object(meps, "WORK", Path(directory)), patch("builtins.print"), \
                    patch.object(meps, "collect", side_effect=AssertionError("Dry run must not collect")):
                result = meps.main(["--start", "2021-10-23", "--end", "2021-10-26",
                                    "--cycles", "0", "6", "--database", str(output), "--dry-run"])
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())

    def test_observation_matching_selects_nearest_value_per_station_and_element(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frost.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE stations(station_id TEXT,name TEXT,lat REAL,lon REAL,elevation REAL,inside_norway INTEGER)")
                connection.execute("CREATE TABLE observations(station_id TEXT,reference_time TEXT,element TEXT,value REAL)")
                connection.execute("INSERT INTO stations VALUES('SN1','A',60,10,100,1)")
                connection.executemany("INSERT INTO observations VALUES(?,?,?,?)", [
                    ("SN1", "2021-10-23T11:20:00.000Z", "air_temperature", 8.),
                    ("SN1", "2021-10-23T12:10:00.000Z", "air_temperature", 9.),
                    ("SN1", "2021-10-23T11:50:00.000Z", "relative_humidity", 70.),
                ])
                connection.commit()
                valid, result = meps.observations_near(connection, datetime(2021,10,23,0,tzinfo=timezone.utc),
                    ["temperature", "humidity"], 12, 60)
            self.assertEqual(valid.hour, 12)
            self.assertEqual(result[("SN1", "air_temperature")]["observation"], 9.)
            self.assertEqual(result[("SN1", "relative_humidity")]["observation"], 70.)

    def test_observation_matching_can_be_limited_to_bbox(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frost.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE stations(station_id TEXT,name TEXT,lat REAL,lon REAL,elevation REAL,inside_norway INTEGER)")
                connection.execute("CREATE TABLE observations(station_id TEXT,reference_time TEXT,element TEXT,value REAL)")
                connection.executemany("INSERT INTO stations VALUES(?,?,?,?,?,1)", [
                    ("SN1", "Oslo", 60.0, 10.7, 100),
                    ("SN2", "Bergen", 60.4, 5.3, 100),
                ])
                connection.executemany("INSERT INTO observations VALUES(?,?,?,?)", [
                    ("SN1", "2024-01-15T12:00:00.000Z", "air_temperature", -8.),
                    ("SN2", "2024-01-15T12:00:00.000Z", "air_temperature", 2.),
                ])
                connection.commit()
                _, result = meps.observations_near(
                    connection, datetime(2024,1,15,0,tzinfo=timezone.utc),
                    ["temperature"], 12, 60, bbox=(10.3, 59.6, 11.2, 60.2),
                )
        self.assertIn(("SN1", "air_temperature"), result)
        self.assertNotIn(("SN2", "air_temperature"), result)

    def test_spatial_sampler_converts_meps_kelvin_and_fraction_to_frost_units(self):
        projection = dict(grid_mapping_name="lambert_conformal_conic", standard_parallel=[63.3,63.3],
            longitude_of_central_meridian=15., latitude_of_projection_origin=63.3,
            false_easting=0., false_northing=0., earth_radius=6371000.)
        crs = pyproj.CRS.from_cf(projection)
        transformer = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
        x, y = transformer.transform(10., 60.)
        dataset = xr.Dataset({
            "projection_lambert": xr.DataArray(0, attrs=projection),
            "air_temperature_2m": xr.DataArray([[[[283.15, 280.], [280., 280.]]]],
                dims=("time", "height1", "y", "x"), attrs={"units":"K", "grid_mapping":"projection_lambert"}),
            "relative_humidity_2m": xr.DataArray([[[[.75, .5], [.5, .5]]]],
                dims=("time", "height1", "y", "x"), attrs={"units":"1", "grid_mapping":"projection_lambert"}),
        }, coords={"time":np.array(["2021-10-23T12:00:00"],dtype="datetime64[ns]"),
                   "height1":[2.],"x":[x,x+5000],"y":[y,y+5000]})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"subset.nc"
            dataset.to_netcdf(path,engine="scipy")
            observations={("SN1","air_temperature"):dict(station_id="SN1",latitude=60.,longitude=10.,observation=9.),
                          ("SN1","relative_humidity"):dict(station_id="SN1",latitude=60.,longitude=10.,observation=70.)}
            forecasts=meps.nearest_grid_values(path.read_bytes(),observations,["temperature","humidity"])
        self.assertAlmostEqual(forecasts[("temperature","SN1")][0],10.)
        self.assertAlmostEqual(forecasts[("humidity","SN1")][0],75.)

    def test_grid_export_writes_arcgis_ready_celsius_geotiff(self):
        projection = dict(grid_mapping_name="lambert_conformal_conic", standard_parallel=[63.3,63.3],
            longitude_of_central_meridian=15., latitude_of_projection_origin=63.3,
            false_easting=0., false_northing=0., earth_radius=6371000.)
        dataset = xr.Dataset({
            "projection_lambert": xr.DataArray(0, attrs=projection),
            "air_temperature_2m": xr.DataArray([[[[273.15, 274.15], [275.15, 276.15]]]],
                dims=("time", "height1", "y", "x"),
                attrs={"units":"K", "grid_mapping":"projection_lambert"}),
        }, coords={"time":np.array(["2021-10-23T12:00:00"],dtype="datetime64[ns]"),
                   "height1":[2.], "x":[500., 1500.], "y":[1500., 500.]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meps_temperature.tif"
            meps.write_grid_geotiff(
                dataset, "temperature", path,
                datetime(2021,10,23,tzinfo=timezone.utc),
                datetime(2021,10,23,12,tzinfo=timezone.utc), 12,
            )
            with rasterio.open(path) as source:
                self.assertEqual(source.shape, (2, 2))
                self.assertIsNotNone(source.crs)
                self.assertLess(source.transform.e, 0)
                self.assertEqual(source.tags()["units"], "degC")
                np.testing.assert_allclose(source.read(1), [[0, 1], [2, 3]], atol=1e-5)

    def test_output_database_resume_tables_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path=meps.init_output(Path(directory)/"meps.sqlite")
            with closing(sqlite3.connect(path)) as connection:
                run=datetime(2021,10,23,tzinfo=timezone.utc)
                meps.mark_status(connection,run,"complete","samples=1")
                meps.mark_status(connection,run,"complete","samples=2")
                self.assertEqual(meps.run_status(connection,run),"complete")
                self.assertEqual(connection.execute("SELECT detail FROM run_status").fetchone()[0],"samples=2")

    def test_ncss_request_is_one_regional_subset_and_streamed(self):
        response = Mock(status_code=200, headers={"Content-Length":"4"})
        response.iter_content.return_value = [b"data"]
        session = Mock()
        session.get.return_value = response
        args = SimpleNamespace(retries=0, timeout=15, max_response_mb=1)
        run = datetime(2021,10,23,tzinfo=timezone.utc)
        valid = datetime(2021,10,23,12,tzinfo=timezone.utc)
        observations = {("SN1","air_temperature"):dict(station_id="SN1",latitude=60.,longitude=10.)}
        bbox = (10.3, 59.6, 11.2, 60.2)
        content, status = meps.request_forecast(
            session, run, valid, observations, ["temperature", "humidity"], args, bbox,
        )
        self.assertEqual((content,status),(b"data","ok"))
        call=session.get.call_args
        self.assertIn("meps_det_2_5km_20211023T00Z.nc",call.args[0])
        self.assertTrue(call.kwargs["stream"])
        self.assertEqual(call.kwargs["params"]["var"],"air_temperature_2m,relative_humidity_2m")
        self.assertEqual(call.kwargs["params"]["time"],"2021-10-23T12:00:00Z")
        self.assertEqual((call.kwargs["params"]["west"], call.kwargs["params"]["south"],
                  call.kwargs["params"]["east"], call.kwargs["params"]["north"]), bbox)
        response.close.assert_called_once()

    def test_ncss_retry_and_oversize_guard(self):
        unavailable=Mock(status_code=503,headers={"Retry-After":"0"})
        successful=Mock(status_code=200,headers={})
        successful.iter_content.return_value=[b"ok"]
        session=Mock();session.get.side_effect=[unavailable,successful]
        args=SimpleNamespace(retries=1,timeout=10,max_response_mb=1)
        run=datetime(2021,10,23,tzinfo=timezone.utc);valid=run+timedelta(hours=12)
        observations={("SN1","air_temperature"):dict(station_id="SN1",latitude=60.,longitude=10.)}
        with patch.object(meps.time_module,"sleep") as sleep:
            self.assertEqual(meps.request_forecast(session,run,valid,observations,["temperature"],args),
                             (b"ok","ok"))
        self.assertEqual(session.get.call_count,2)
        sleep.assert_called_once_with(0.)
        oversized=Mock(status_code=200,headers={"Content-Length":str(2*1024*1024)})
        oversized.iter_content.side_effect=AssertionError("Oversize body must not be read")
        session.get.side_effect=None
        session.get.return_value=oversized
        with self.assertRaisesRegex(ValueError,"declares"):
            meps.request_forecast(session,run,valid,observations,["temperature"],args)
        oversized.close.assert_called_once()

    def test_archive_outage_is_reported_without_traceback_and_is_resumable(self):
        error=meps.requests.HTTPError("503 Server Error")
        stderr=StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(meps,"collect",side_effect=error), \
                redirect_stderr(stderr):
            result=meps.main(["--start","2021-10-23","--end","2021-10-24",
                              "--database",str(Path(directory)/"samples.sqlite")])
        self.assertEqual(result,1)
        self.assertIn("archive request failed after retries",stderr.getvalue())
        self.assertIn("failed run will be retried",stderr.getvalue())
        self.assertNotIn("Traceback",stderr.getvalue())


if __name__ == "__main__":
    unittest.main()