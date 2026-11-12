"""Tester for værbasert søk etter lokale kuldeavvik."""
import json
from pathlib import Path
import tempfile
import unittest

import netCDF4
import numpy as np
import pandas as pd

import cold_spots as cold


class ColdSpotTests(unittest.TestCase):
    def test_ring_excludes_inner_block_and_recovers_exact_deficit(self):
        field = np.full((51, 51), 10.0)
        field[23:28, 23:28] = -20
        anomaly, support = cold.local_deviation(field, 2, 5)
        self.assertAlmostEqual(anomaly[25, 25], -30)
        self.assertAlmostEqual(support[25, 25], 1)
        self.assertTrue(np.isnan(anomaly[0, 0]))

    def test_missing_neighbors_not_zero_and_shared_sums_follow_changed_masks(self):
        field = np.full((51, 51), -10.0)
        field[25, 25] = -15
        field[20:23, 20:30] = np.nan
        engine = cold.ReferenceRings()
        for _ in range(2):
            for pair, (inner, outer) in zip(engine.evaluate(field), cold.RINGS):
                expected = cold.local_deviation(field, inner, outer)
                np.testing.assert_allclose(pair[0], expected[0], atol=1e-10)
                np.testing.assert_allclose(pair[1], expected[1], atol=1e-10)
            field[30:35] = np.nan
        self.assertAlmostEqual(engine.evaluate(field)[0][0][25, 25], -5)

    def test_event_thresholds_missing_breaks_runs_and_seasons(self):
        stats = np.zeros((len(cold.METRICS), 2))
        run = np.zeros(2, dtype=np.int16)
        for values, month in [([-2, 0], 1), ([-5, -1], 1), ([np.nan, -2], 7), ([-3, -2], 7)]:
            cold.accumulate(stats, run, np.asarray(values), month)
        np.testing.assert_array_equal(stats[0], [3, 4])
        np.testing.assert_array_equal(stats[2], [3, 2])
        np.testing.assert_array_equal(stats[3], [1, 0])
        np.testing.assert_array_equal(stats[6], [2, 2])
        np.testing.assert_array_equal(stats[8], [2, 0])
        np.testing.assert_array_equal(stats[10], [1, 2])

    def test_combination_stability_coverage_and_no_events(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            years = [1996, 1997]
            for year in years:
                stats = np.zeros((3, len(cold.METRICS), 4))
                stats[:, 0] = 366 if year == 1996 else 365
                stats[:, 2, 0] = 30
                stats[:, 2, 1] = 30 if year == 1996 else 0
                stats[:, 2, 3] = 30
                stats[:, 0, 3] = 10
                stats[:, 4] = stats[:, 2] * 4
                np.savez(cache / f"{year}.npz", stats=stats,
                         support=stats[:, 0], days=366 if year == 1996 else 365)
            results, _ = cold.combine(cache, years)
            r = results[0]
            np.testing.assert_array_equal(r["classification"], [2, 0, 0, 0])
            np.testing.assert_array_equal(r["usable"], [True, True, True, False])
            self.assertTrue(np.isnan(r["values"]["kuldestyrke_ved_hendelse_c"][2]))
            self.assertEqual(r["values"]["kuldestyrke_ved_hendelse_c"][0], 4)

    def test_two_year_netcdf_export_resume_and_gpkg_roundtrip(self):
        import geopandas as gpd
        import rasterio
        from shapely.geometry import box
        from rasterio.features import rasterize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "result"
            source.mkdir()
            polygon = root / "land.gpkg"
            gpd.GeoDataFrame({"navn": ["land"]}, geometry=[box(260000, 6650000, 275000, 6665000)],
                             crs=32633).to_file(polygon, driver="GPKG")
            records = []
            for year in [1996, 1997]:
                path = source / f"{year}.nc"
                n = len(pd.date_range(f"{year}-01-01", f"{year}-12-31"))
                with netCDF4.Dataset(path, "w") as ds:
                    for name, length in [("time", n), ("Y", 15), ("X", 15)]:
                        ds.createDimension(name, length)
                    ds.createVariable("X", "f8", ("X",))[:] = 260500 + np.arange(15) * 1000
                    ds.createVariable("Y", "f8", ("Y",))[:] = 6664500 - np.arange(15) * 1000
                    t = ds.createVariable("time", "f8", ("time",))
                    t.units = f"hours since {year}-01-01"
                    t[:] = np.arange(n) * 24 + 6
                    mapping = ds.createVariable("utm", "i4")
                    mapping.proj4 = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0"
                    var = ds.createVariable("tn", "f4", ("time", "Y", "X"), fill_value=-999.99)
                    var.units, var.grid_mapping = "Celsius", "utm"
                    var[:] = 0
                    var[:, 7, 7] = -6
                    var[:, 0, 0] = -999.99
                records.append(dict(year=year, netcdf=str(path), bytes=path.stat().st_size))
            (source / "manifest.json").write_text(json.dumps(dict(archives=records)))
            recs, grid, ids = cold.prepare(source, polygon, output, [1996, 1997])
            for record in recs:
                cold.extract_year(record, grid, ids, output / "cache")
            receipt = output / "cache" / "1996.npz"
            mtime = receipt.stat().st_mtime_ns
            cold.extract_year(recs[0], grid, ids, output / "cache")
            self.assertEqual(receipt.stat().st_mtime_ns, mtime)
            self.assertEqual(cold.verify_year_samples(recs[0], grid, ids, output / "cache")["status"], "passed")
            result, _ = cold.combine(output / "cache", [1996, 1997])
            cold.export_results(output, grid, ids, result, [1996, 1997])
            with rasterio.open(output / "mulige_kuldehull_ring_2_5km.tif") as ds:
                a = ds.read(1)
                self.assertEqual(a[7, 7], 2)
                self.assertEqual(a[0, 0], 255)
                frame = gpd.read_file(output / "mulige_kuldehull.gpkg", layer="mulige_kuldehull")
                restored = rasterize(zip(frame.geometry, frame.klasse), out_shape=ds.shape,
                                     transform=ds.transform, fill=0)
                np.testing.assert_array_equal(restored, np.where(a == 255, 0, a))
            with rasterio.open(output / "kalde_dogn_prosent_ring_2_5km.tif") as ds:
                self.assertEqual(ds.read(1)[7, 7], 100)
            with rasterio.open(output / "kuldestyrke_ved_hendelse_c_ring_2_5km.tif") as ds:
                self.assertAlmostEqual(ds.read(1)[7, 7], 6)
            with rasterio.open(output / "lengste_kuldeperiode_dogn_ring_2_5km.tif") as ds:
                self.assertEqual(ds.read(1)[7, 7], 366)
            self.assertEqual(frame.areal_km2.sum(), 1)
            self.assertEqual(frame.lengste_kuldeperiode_dogn.iloc[0], 366)
            self.assertAlmostEqual(frame.referansedekning_prosent.iloc[0], 100)
            empty_output = root / "no_candidates"
            empty_output.mkdir()
            for ring in result:
                ring["classification"][:] = 0
            cold.export_results(empty_output, grid, ids, result, [1996, 1997])
            self.assertTrue(gpd.read_file(empty_output / "mulige_kuldehull.gpkg").empty)
            with netCDF4.Dataset(recs[0]["netcdf"], "a") as ds:
                ds.history = "changed"
            with self.assertRaisesRegex(ValueError, "Endret filstørrelse|andre kilder"):
                cold.prepare(source, polygon, output, [1996, 1997])


if __name__ == "__main__":
    unittest.main()
