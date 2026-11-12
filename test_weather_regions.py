"""Focused tests for weather-region features, representatives and map alignment."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import weather_regions as regions


class WeatherRegionTests(unittest.TestCase):
    def test_monthly_levels_and_thresholds_have_physical_meaning(self):
        tg = np.array([[-2, 5], [2, 9]], dtype=np.float32)
        rr = np.array([[0, 20], [2, 30]], dtype=np.float32)
        tn, tx = tg - 3, tg + 3
        stats = regions.describe_month(tg, rr, tn, tx)
        np.testing.assert_array_equal(stats[regions.MI["temperature_mean"]], [0, 7])
        np.testing.assert_array_equal(stats[regions.MI["precipitation_total"]], [2, 50])
        np.testing.assert_array_equal(stats[regions.MI["wet_fraction"]], [0.5, 1])
        np.testing.assert_array_equal(stats[regions.MI["heavy_fraction"]], [0, 1])
        np.testing.assert_array_equal(stats[regions.MI["freeze_thaw_fraction"]], [1, 0])

    def test_anomalies_remove_season_not_spatial_variability_amplitude(self):
        values = np.array([[1, 3, 10, 14], [12, 16, 20, 28]], dtype=np.float32)
        anomalies = regions.daily_anomalies(values, np.array([1, 1, 2, 2]))
        np.testing.assert_array_equal(anomalies, [[-1, 1, -2, 2], [-2, 2, -4, 4]])
        np.testing.assert_array_equal(values[0], [1, 3, 10, 14])

    def test_absolute_features_keep_local_temperature_and_rainfall_levels(self):
        monthly = np.zeros((5, 12, len(regions.MONTH_METRICS), 3), dtype=np.float32)
        monthly[:, :, regions.MI["temperature_mean"]] = [0, 10, 20]
        monthly[:, :, regions.MI["precipitation_total"]] = [10, 100, 1000]
        blocks = regions.summary_blocks(monthly, list(range(5)))
        np.testing.assert_array_equal(blocks[0][:, 0], [0, 10, 20])
        np.testing.assert_allclose(blocks[1][:, 0], np.log1p([10, 100, 1000]))
        np.testing.assert_array_equal(blocks[3], 0)

    def test_year_omission_excludes_omitted_year_from_summary(self):
        monthly = np.zeros((5, 12, len(regions.MONTH_METRICS), 2), dtype=np.float32)
        monthly[4, :, regions.MI["temperature_mean"]] = 100
        included = regions.summary_blocks(monthly, [0, 1, 2, 3])
        np.testing.assert_array_equal(included[0], 0)
        self.assertTrue(np.all(regions.summary_blocks(monthly, list(range(5)))[0] == 20))

    def test_representatives_are_real_members_of_their_own_cluster(self):
        features = np.array([[0, 0], [2, 0], [9, 0], [12, 0]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        reps = regions.representative_ids(features, labels, np.array([[0.5, 0], [10, 0]]))
        np.testing.assert_array_equal(reps, [0, 2])
        np.testing.assert_array_equal(labels[reps], [0, 1])
        with self.assertRaisesRegex(ValueError, "no members"):
            regions.representative_ids(features, labels, np.zeros((3, 2)))

    def test_daily_error_metrics_match_known_values(self):
        daily = dict(tg=np.array([[0, 2], [2, 4]], dtype=np.float32),
                     rr=np.array([[0, 2], [2, 0]], dtype=np.float32))
        monthly = np.zeros((5, 12, len(regions.MONTH_METRICS), 2), dtype=np.float32)
        monthly[:, :, regions.MI["precipitation_total"]] = [10, 20]
        errors = regions.representation_errors(daily, monthly, np.array([0, 1]), np.array([1, 1]))
        np.testing.assert_allclose(errors["temperature_rmse_c"], [2, 0])
        np.testing.assert_allclose(errors["precipitation_rmse_mm"], [2, 0])
        np.testing.assert_allclose(errors["wet_day_disagreement"], [1, 0])
        np.testing.assert_allclose(errors["monthly_precipitation_mae_mm"], [10, 0])
        np.testing.assert_allclose(errors["annual_precipitation_mae_mm"], [120, 0])

    def test_raster_is_north_up_at_exact_grid_cell_centres(self):
        import rasterio

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.tif"
            grid = dict(width=3, height=2, west=260000, north=6653000, crs="EPSG:32633")
            regions.write_raster(path, grid, np.array([0, 2, 4]), [1, 2, 3], "region_id",
                                 categorical=True)
            with rasterio.open(path) as source:
                np.testing.assert_array_equal(source.read(1), [[1, 0, 2], [0, 3, 0]])
                self.assertEqual(source.xy(0, 0), (260500, 6652500))
                self.assertEqual(source.res, (1000, 1000))
                self.assertEqual(source.nodata, 0)
                self.assertEqual(source.crs.to_epsg(), 32633)

    def test_group_weights_and_pca_fit_use_training_cells_only(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2021-01-01", "2025-12-31")
        n = 30
        monthly = rng.uniform(1, 20, (5, 12, len(regions.MONTH_METRICS), n)).astype(np.float32)
        daily = dict(tg=rng.normal(size=(n, len(dates))).astype(np.float32),
                     rr=rng.uniform(0, 10, size=(n, len(dates))).astype(np.float32))
        training = np.arange(20)
        features, models = regions.fit_features(monthly, daily, dates, training, list(range(5)))
        np.testing.assert_allclose(
            regions.transform_features(monthly, daily, dates, models), features, atol=1e-6,
        )
        self.assertEqual(features.shape[0], n)
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(sum(models["equal_block_weights"]), 1)
        monthly[:, :, regions.MI["temperature_mean"], 25] = 1000000
        daily["tg"][25] *= 10000
        _, changed = regions.fit_features(monthly, daily, dates, training, list(range(5)))
        for before, after in zip(models["blocks"], changed["blocks"]):
            if before["kind"] == "standard":
                np.testing.assert_array_equal(before["model"].mean_, after["model"].mean_)
            else:
                np.testing.assert_allclose(before["model"].components_, after["model"].components_)

    def test_selection_is_explicit_elbow_compromise(self):
        table = pd.DataFrame(dict(
            k=[8, 12, 16, 24, 32], normalized_representative_error=[1, 0.6, 0.5, 0.48, 0.47],
            silhouette=[0.3] * 5, year_omission_ari_mean=[0.8] * 5,
        ))
        self.assertIn(regions.select_candidate(table), [12, 16])
        self.assertIn("selection_score", table)

    def test_end_to_end_five_year_fixture_creates_norway_only_regions(self):
        import geopandas as gpd
        import netCDF4
        import rasterio
        from shapely.geometry import box

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "result"
            source.mkdir()
            polygon = root / "norway.gpkg"
            gpd.GeoDataFrame({"name": ["test land"]},
                             geometry=[box(260000, 6650000, 272000, 6660000)],
                             crs="EPSG:32633").to_file(polygon, driver="GPKG")
            records = []
            rng = np.random.default_rng(42)
            for year in range(2021, 2026):
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
                path = source / f"{year}.nc"
                with netCDF4.Dataset(path, "w") as ds:
                    for name, length in [("time", len(dates)), ("Y", 10), ("X", 13)]:
                        ds.createDimension(name, length)
                    ds.createVariable("X", "f8", ("X",))[:] = 260500 + np.arange(13) * 1000
                    ds.createVariable("Y", "f8", ("Y",))[:] = 6659500 - np.arange(10) * 1000
                    times = ds.createVariable("time", "f8", ("time",))
                    times.units = f"hours since {year}-01-01 00:00:00"
                    times[:] = np.arange(len(dates)) * 24 + 6
                    mapping = ds.createVariable("UTM_Zone_33", "i4")
                    mapping.proj4 = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"
                    group = np.repeat(np.arange(13)[None, :] // 3, 10, axis=0)
                    seasonal = 8 * np.sin(np.arange(len(dates))[:, None, None] * 2 * np.pi / 365)
                    t = seasonal + group[None] * 4 + rng.normal(0, 1, (len(dates), 10, 13))
                    rain = rng.gamma(0.7, 2, t.shape) * (group[None] + 1)
                    for name, values in (("tg", t), ("rr", rain), ("tn", t - 3), ("tx", t + 3)):
                        var = ds.createVariable(name, "f4", ("time", "Y", "X"), fill_value=-999.99)
                        var[:] = values
                        var.units = "kg/m^2" if name == "rr" else "Celsius"
                        var.grid_mapping = "UTM_Zone_33"
                records.append(dict(year=year, netcdf=str(path), bytes=path.stat().st_size))
            (source / "manifest.json").write_text(json.dumps(dict(status="complete", archives=records)))
            with threadpool_limits(limits=1):
                result = regions.main([
                    "--source", str(source), "--polygon", str(polygon), "--output-dir", str(output),
                    "--training-cells", "70", "--validation-cells", "40",
                    "--candidates", "2", "3", "4", "--threads", "1",
                ])
            self.assertEqual(result, 0)
            analysis = json.loads((output / "analysis.json").read_text())
            self.assertEqual(analysis["complete_cells"], 120)
            with rasterio.open(output / "weather_regions_1km.tif") as raster:
                data = raster.read(1)
                self.assertTrue(np.all(data[:, -1] == 0))
                self.assertTrue(np.all(data[:, :-1] > 0))
                self.assertEqual(len(np.unique(data[data > 0])), analysis["selected_k"])
            representatives = gpd.read_file(output / "representatives.gpkg")
            self.assertEqual(len(representatives), analysis["selected_k"])
            comparison = pd.read_csv(output / "cluster_comparison.csv")
            self.assertTrue(comparison["year_omission_ari_mean"].between(-1, 1).all())
            self.assertEqual(int(pd.read_csv(output / "region_summary.csv").cells.sum()), 120)
            self.assertEqual(regions.main([
                "--source", str(source), "--polygon", str(polygon), "--output-dir", str(output),
                "--additional-k", "4", "--threads", "1",
            ]), 0)
            alternative = json.loads((output / "k4" / "analysis.json").read_text())
            self.assertEqual(alternative["selected_k"], 4)
            self.assertEqual(alternative["overview_groups"], analysis["selected_k"])
            with rasterio.open(output / "k4" / "weather_regions_1km.tif") as raster:
                self.assertEqual(len(np.unique(raster.read(1)[raster.read(1) > 0])), 4)


if __name__ == "__main__":
    unittest.main()
