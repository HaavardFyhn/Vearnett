"""Tests for streamed thirty-year features and held-out weather validation."""
import json
from pathlib import Path
import tempfile
import unittest

import netCDF4
import numpy as np
import pandas as pd

import climate_regions as climate


class ClimateRegionTests(unittest.TestCase):
    def test_summer_winter_amplitude_and_frost_day_counts(self):
        monthly = np.zeros((12, len(climate.weather.MONTH_METRICS), 2), dtype=np.float32)
        monthly[[0, 1, 11], climate.weather.MI["temperature_mean"]] = -5
        monthly[[5, 6, 7], climate.weather.MI["temperature_mean"]] = 15
        monthly[0, climate.weather.MI["frost_fraction"]] = 1
        monthly[1, climate.weather.MI["frost_fraction"]] = 1 / 29
        monthly[11, climate.weather.MI["frost_fraction"]] = 2 / 31
        summer, winter, amplitude, frost = climate.seasonal_indicators(monthly, 2000)
        np.testing.assert_array_equal(summer, 15)
        np.testing.assert_array_equal(winter, -5)
        np.testing.assert_array_equal(amplitude, 20)
        np.testing.assert_allclose(frost, 34, atol=1e-5)
        monthly[1, climate.weather.MI["frost_fraction"]] = 1 / 28
        np.testing.assert_array_equal(climate.seasonal_indicators(monthly, 2001)[3], 34)
        monthly[1, climate.weather.MI["temperature_mean"]] = -10
        np.testing.assert_allclose(climate.seasonal_indicators(monthly, 2001)[1],
                                   (-5 * 62 - 10 * 28) / 90, atol=1e-6)

    def test_calendar_spell_duration_degree_days_and_rolling_rain(self):
        tg = np.array([[-2], [-1], [6], [21], [22], [23]], dtype=np.float32)
        rr = np.array([[0], [0], [1], [2], [3], [4]], dtype=np.float32)
        result = dict(zip(climate.ANNUAL_NAMES, climate.annual_indices(tg, rr)[:, 0]))
        self.assertEqual(result["longest_wet_spell_days"], 4)
        self.assertEqual(result["longest_dry_spell_days"], 2)
        self.assertEqual(result["longest_warm_spell_tg_ge20_days"], 3)
        self.assertEqual(result["longest_cold_spell_tg_lt0_days"], 2)
        self.assertEqual(result["growing_degree_days_base5"], 52)
        self.assertEqual(result["heating_degree_days_base17"], 48)
        self.assertEqual(result["freezing_degree_days_base0"], 3)
        self.assertEqual(result["days_tg_gt5"], 4)
        self.assertEqual(result["max_3day_precipitation_mm"], 9)
        self.assertEqual(result["max_5day_precipitation_mm"], 10)

    def test_sketch_is_reproducible_and_removes_monthly_levels_not_amplitude(self):
        values = np.array([[1, 2], [3, 6], [10, 20], [14, 28]], dtype=np.float32)
        months = np.array([1, 1, 2, 2])
        a = climate.timing_sketch(values, months, 2000)
        b = climate.timing_sketch(values + 100, months, 2000)
        np.testing.assert_allclose(a, b, atol=1e-5)
        np.testing.assert_allclose(a[1], a[0] * 2, atol=1e-5)
        np.testing.assert_array_equal(a, climate.timing_sketch(values, months, 2000))
        self.assertFalse(np.array_equal(a, climate.timing_sketch(values, months, 2001)))

    def test_exact_pair_errors_have_physical_units(self):
        daily = dict(tg=np.array([[0, 2], [2, 4]]), rr=np.array([[0, 2], [2, 0]]))
        error = climate.pair_errors(daily, np.array([0]), np.array([1]))
        self.assertEqual(error["temperature_rmse_c"], 2)
        self.assertEqual(error["precipitation_rmse_mm"], 2)
        self.assertEqual(error["wet_day_disagreement"], 1)

    def test_feature_scalers_and_pca_fit_training_cells_only(self):
        rng = np.random.default_rng(42)
        blocks = [rng.normal(size=(60, 16)).astype(np.float32) for _ in climate.FAMILIES]
        training = np.arange(40)
        first, fitted = climate.encode_blocks(blocks, training)
        for block in blocks:
            block[40:] += 1000
        second, refitted = climate.encode_blocks(blocks, training)
        np.testing.assert_array_equal(first[:40], second[:40])
        for a, b in zip(fitted, refitted):
            np.testing.assert_array_equal(a["model"].mean_, b["model"].mean_)

    def test_aggregate_excludes_heldout_year_from_every_feature_family(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for year, value in [(2000, 1), (2001, 100)]:
                np.save(cache / f"months_{year}.npy",
                        np.full((12, len(climate.weather.MONTH_METRICS), 3), value, dtype=np.float32))
                np.save(cache / f"annual_{year}.npy",
                        np.full((len(climate.ANNUAL_NAMES), 3), value, dtype=np.float32))
                for name in ("tg", "rr"):
                    np.save(cache / f"sketch_{name}_{year}.npy",
                            np.full((3, climate.SKETCH_SIZE), value, dtype=np.float32))
            blocks = climate.aggregate_blocks(cache, np.ones(3, dtype=bool), [2000])
            np.testing.assert_array_equal(blocks[0][:, :12], 1)
            np.testing.assert_array_equal(blocks[0][:, -1], 0)
            np.testing.assert_allclose(blocks[3][:, -2], np.log1p(366), atol=1e-6)
            np.testing.assert_array_equal(blocks[3][:, -1], 0)
            np.testing.assert_array_equal(blocks[5], 1)
            np.testing.assert_array_equal(blocks[4], 0)
            all_years = climate.aggregate_blocks(cache, np.ones(3, dtype=bool), [2000, 2001])
            self.assertTrue(np.all(all_years[0][:, :12] == 50.5))

    def test_full_thirty_year_fixture_and_saved_products(self):
        import geopandas as gpd
        import rasterio
        from shapely.geometry import box

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "result"
            source.mkdir()
            polygon = root / "norway.gpkg"
            gpd.GeoDataFrame({"name": ["land"]}, geometry=[box(260000, 6650000, 272000, 6750000)],
                             crs=32633).to_file(polygon, driver="GPKG")
            records = []
            rng = np.random.default_rng(42)
            for year in climate.YEARS:
                path = source / f"{year}.nc"
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
                with netCDF4.Dataset(path, "w") as ds:
                    for name, length in [("time", len(dates)), ("Y", 100), ("X", 13)]:
                        ds.createDimension(name, length)
                    ds.createVariable("X", "f8", ("X",))[:] = 260500 + np.arange(13) * 1000
                    ds.createVariable("Y", "f8", ("Y",))[:] = 6749500 - np.arange(100) * 1000
                    times = ds.createVariable("time", "f8", ("time",))
                    times.units = f"hours since {year}-01-01 00:00:00"
                    times[:] = np.arange(len(dates)) * 24 + 6
                    mapping = ds.createVariable("UTM_Zone_33", "i4")
                    mapping.proj4 = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"
                    offset = np.repeat(np.arange(13)[None, :], 100, axis=0)
                    t = (8 * np.sin(np.arange(len(dates))[:, None, None] * 2 * np.pi / 365) +
                         offset[None] + (year - 1996) * 0.03 +
                         rng.normal(0, 1, (len(dates), 100, 13)))
                    p = rng.gamma(0.7, 2, t.shape) * (1 + offset[None] / 4)
                    for name, values in (("tg", t), ("rr", p), ("tn", t - 3), ("tx", t + 3)):
                        var = ds.createVariable(name, "f4", ("time", "Y", "X"), fill_value=-999.99)
                        var[:] = values
                        if name == "rr" and year == 2018:
                            var[100, 0, 0] = -999.99
                        var.units = "kg/m^2" if name == "rr" else "Celsius"
                        var.grid_mapping = "UTM_Zone_33"
                records.append(dict(year=year, netcdf=str(path), bytes=path.stat().st_size,
                                    source=f"https://example.test/{year}.nc"))
            (source / "manifest.json").write_text(json.dumps(dict(status="complete", archives=records)))
            result = climate.main([
                "--source", str(source), "--polygon", str(polygon), "--output-dir", str(output),
                "--training-cells", "700", "--validation-cells", "400", "--candidates", "8", "32",
                "--threads", "1",
            ])
            self.assertEqual(result, 0)
            metadata = json.loads((output / "analysis.json").read_text())
            self.assertEqual(metadata["complete_cells"], 1199)
            self.assertEqual(metadata["excluded_incomplete_cells"], 1)
            self.assertEqual(metadata["days"], 10958)
            audit = json.loads((output / "timing_compression_audit.json").read_text())
            self.assertTrue(np.isfinite(audit["tg"]["projection_distance_correlation"]))
            self.assertLess(audit["tg"]["projection_median_relative_distance_error"], 0.15)
            comparison = pd.read_csv(output / "cluster_comparison.csv")
            self.assertTrue(np.isfinite(comparison.select_dtypes("number")).all().all())
            expected_amplitude, expected_frost = [], []
            for record in records:
                dates = pd.date_range(f"{record['year']}-01-01", f"{record['year']}-12-31")
                with netCDF4.Dataset(record["netcdf"]) as ds:
                    temperature = ds["tg"][:, 40, 4].astype(np.float64)
                    expected_amplitude.append(
                        temperature[np.isin(dates.month, [6, 7, 8])].mean() -
                        temperature[np.isin(dates.month, [12, 1, 2])].mean())
                    expected_frost.append(int(np.sum(ds["tn"][:, 40, 4] < 0)))
            for name, expected in (
                ("summer_winter_temperature_amplitude_c", np.mean(expected_amplitude)),
                ("frost_days_per_year", np.mean(expected_frost)),
            ):
                with rasterio.open(output / f"{name}_1km.tif") as raster:
                    self.assertAlmostEqual(raster.read(1)[40, 4], expected, places=5)
                    self.assertEqual(raster.read(1)[0, 0], raster.nodata)
            for k in (8, 32):
                with rasterio.open(output / f"k{k}" / "climate_regions_1km.tif") as raster:
                    data = raster.read(1)
                    self.assertTrue(np.all(data[:, -1] == 0))
                    self.assertEqual(data[0, 0], 0)
                    self.assertEqual(len(np.unique(data[data > 0])), k)
                    self.assertEqual(raster.tags()["period"], "1996-2025")
                    self.assertEqual(raster.res, (1000, 1000))
                reps = gpd.read_file(output / f"k{k}" / "representatives.gpkg")
                self.assertIn("frost_days_per_year", reps.columns)
                self.assertIn("summer_winter_temperature_amplitude_c", reps.columns)
                self.assertEqual(len(pd.read_csv(output / f"k{k}" / "annual_profiles.csv")), k * 30)
                self.assertEqual(len(pd.read_csv(output / f"k{k}" / "decade_profiles.csv")), k * 3)
                with rasterio.open(output / f"k{k}" / "temperature_rmse_c_1km.tif") as raster:
                    values = raster.read(1)
                    for rep in reps.itertuples():
                        row, col = raster.index(rep.geometry.x, rep.geometry.y)
                        self.assertEqual(values[row, col], 0)
                    row, col = 40, 4
                    group = data[row, col]
                    rep = reps[reps.region_id == group].iloc[0]
                    rep_row, rep_col = raster.index(rep.geometry.x, rep.geometry.y)
                    square, count = 0.0, 0
                    for record in records:
                        with netCDF4.Dataset(record["netcdf"]) as ds:
                            difference = ds["tg"][:, row, col] - ds["tg"][:, rep_row, rep_col]
                            square += np.square(difference.astype(np.float64)).sum()
                            count += len(difference)
                    self.assertAlmostEqual(values[row, col], np.sqrt(square / count), places=5)
            record = records[-1]
            receipt = output / "cache" / "2025.json"
            saved = json.loads(receipt.read_text())
            saved["source"]["mtime_ns"] += 1
            receipt.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                climate.extract_year(record, metadata["grid"],
                                     np.load(output / "cache" / "cell_ids.npy"),
                                     np.load(output / "cache" / "sample.npy"), output / "cache")


if __name__ == "__main__":
    unittest.main()
