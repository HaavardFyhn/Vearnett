import argparse
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import train_senorge_grid_pairs as training
from prepare_senorge_multi_target import TARGETS, pair_labels, prepare as prepare_multi
import predict_senorge_grid_pairs as inference
import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
from shapely.geometry import box


class GridPairTrainingTests(unittest.TestCase):
    def test_multi_target_labels_distinguish_direct_difference_from_regression(self):
        x = np.arange(100, dtype=np.float32)
        fitting = np.stack([x, x + 5, np.full_like(x, np.nan)])
        evaluation = fitting[:, :30].copy()
        evaluation[1, 0] = np.nan
        evaluation[1, 1:4] += 10
        values = pair_labels(fitting, evaluation, np.array([[0, 1], [1, 0], [0, 2]]))
        delta = evaluation[1, 1:] - evaluation[0, 1:]
        residuals = delta - 5
        expected = [np.sqrt(np.mean(delta**2)), np.quantile(np.abs(delta), .95),
                    np.sqrt(np.mean(residuals**2)), np.quantile(np.abs(residuals), .95)]
        np.testing.assert_allclose(values[:2], [expected, expected], rtol=1e-6)
        self.assertTrue(np.isnan(values[2]).all())

    def test_whole_basin_volume_overlap_and_boundary_touch(self):
        grid = gpd.GeoDataFrame(dict(x=[500., 1500., 2500.], y=[500., 500., 500.]),
                                geometry=gpd.points_from_xy([500, 1500, 2500], [500]*3), crs=25833)
        basins = gpd.GeoDataFrame(dict(lake_id=[1, 2], volume_m3=[1000., 200.], max_depth_m=[8., 2.]),
                                 geometry=[box(900, 200, 1100, 800), box(1200, 200, 1800, 800)], crs=25833)
        result = training.attach_basins(grid, basins)
        np.testing.assert_array_equal(result.basin_count, [1, 2, 0])
        np.testing.assert_array_equal(result.basin_volume_m3, [1000, 1200, 0])
        np.testing.assert_array_equal(result.basin_max_depth_m, [8, 8, 0])
        np.testing.assert_array_equal(result.basin_present, [1, 1, 0])

    def test_streaming_moments_equal_direct_ols_with_missing_values(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, 4))
        y = 2*x + rng.normal(size=(100, 4))
        x[10, 0] = np.nan
        y[90, 1] = np.nan
        stats = np.zeros((2, 6, 4))
        for day in range(100):
            training.update_moments(stats[int(day >= 80)], x[day], y[day])
        actual, _ = training.regression_labels(stats[0], stats[1], 2, 2)
        expected = []
        for pair in range(4):
            errors = []
            for source, target in ((x, y), (y, x)):
                a, b = source[:, pair], target[:, pair]
                train = np.isfinite(a[:80]) & np.isfinite(b[:80])
                slope, intercept = np.polyfit(a[:80][train], b[:80][train], 1)
                keep = np.isfinite(a[80:]) & np.isfinite(b[80:])
                errors.append(np.sqrt(np.mean((b[80:][keep]-slope*a[80:][keep]-intercept)**2)))
            expected.append(np.mean(errors))
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_symmetric_percentile_label_uses_absolute_errors_in_both_directions(self):
        forward = np.array([[0., 1.], [1., 2.], [2., 4.], [8., 7.]])
        reverse = np.array([[1., 0.], [3., 1.], [5., 3.], [9., 9.]])
        actual = training.symmetric_percentile_labels(forward, reverse)
        expected = (np.quantile(np.abs(forward), .95, axis=0) +
                    np.quantile(np.abs(reverse), .95, axis=0))/2
        np.testing.assert_allclose(actual, expected)
        with self.assertRaisesRegex(ValueError, "Invalid directional residuals"):
            training.symmetric_percentile_labels(np.full((2, 1), np.nan), np.full((2, 1), np.nan))

    def fixture_grid(self):
        x = np.concatenate([260500 + block*110000 + np.arange(10)*1000 for block in range(8)])
        y = np.full(len(x), 6650500.)
        grid = gpd.GeoDataFrame(dict(grid_id=np.arange(len(x)), x=x, y=y, elevation=np.arange(len(x)),
                                    coast_dist_fjord=np.ones(len(x)), coast_dist_ocean=np.ones(len(x)),
                                    objtype=["land"]*len(x)),
                                geometry=gpd.points_from_xy(x,y), crs=25833)
        for field in training.TERRAIN_FIELDS:
            grid[field] = np.arange(len(x)) * .1
        for field in training.LAKE_FIELDS:
            grid[field] = np.arange(len(x)) % 3
        return grid

    def test_pairs_unique_station_free_symmetric_and_spatial_disjoint(self):
        grid = self.fixture_grid()
        pairs = training.sample_pairs(grid, 300, scale=4000)
        self.assertEqual(len(np.unique(pairs, axis=0)), 300)
        self.assertTrue((pairs[:,0] < pairs[:,1]).all())
        np.testing.assert_array_equal(training.pair_features(grid,pairs),
                                      training.pair_features(grid,pairs[:,::-1]))
        train,test = training.spatial_split(grid,pairs)
        self.assertFalse(set(pairs[train].ravel()) & set(pairs[test].ravel()))
        blocks = np.floor(grid[["x","y"]].to_numpy()/100000)
        self.assertFalse(set(map(tuple,blocks[pairs[train].ravel()])) &
                         set(map(tuple,blocks[pairs[test].ravel()])))

    def test_stratified_random_dates_cover_thirty_years_without_overlap(self):
        years = list(range(1996, 2026))
        plan = training.date_plan(years, days_per_month=10, temporal_split="monthly-random")
        self.assertEqual(len(plan), 3600)
        self.assertEqual(plan.date.nunique(), 3600)
        self.assertEqual(int((plan.partition == 0).sum()), 2880)
        self.assertEqual(int((plan.partition == 1).sum()), 720)
        grouped = plan.assign(month=pd.to_datetime(plan.date).dt.month).groupby(["year", "month", "partition"])
        self.assertTrue((grouped.size().unstack()[0] == 8).all())
        self.assertTrue((grouped.size().unstack()[1] == 2).all())
        self.assertEqual(len(grouped.size()), 720)
        pd.testing.assert_frame_equal(
            plan, training.date_plan(years, days_per_month=10, temporal_split="monthly-random"))
        changed = training.date_plan(years, days_per_month=10, temporal_split="monthly-random", seed=17)
        self.assertFalse(plan.equals(changed))
        for year in years:
            dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
            part = plan.loc[plan.year == year]
            np.testing.assert_array_equal(dates[part.day_index].strftime("%Y-%m-%d"), part.date)
        with self.assertRaises(ValueError):
            training.date_plan(years, days_per_month=10, date_step=2)

    def test_half_normal_distance_distribution_and_rare_long_pairs(self):
        x, y = np.meshgrid(np.arange(250)*1000., np.arange(250)*1000.)
        grid = pd.DataFrame(dict(x=x.ravel(), y=y.ravel()))
        pairs = training.sample_pairs(grid, 100000, maximum=100000, scale=25000,
                                      distribution="half-normal")
        distance = np.linalg.norm(grid.to_numpy()[pairs[:, 0]]-grid.to_numpy()[pairs[:, 1]], axis=1)
        self.assertEqual(len(np.unique(pairs, axis=0)), 100000)
        self.assertTrue((pairs[:, 0] < pairs[:, 1]).all())
        self.assertTrue(((distance >= 1000) & (distance <= 100000)).all())
        self.assertGreater((distance < 20000).mean(), .5)
        self.assertGreater((distance < 50000).mean(), .93)
        self.assertGreater((distance >= 90000).sum(), 0)
        self.assertLess((distance >= 90000).mean(), .002)
        # Equal-width near-distance intervals should carry more mass than distant ones.
        self.assertGreater(((distance >= 1000) & (distance < 10000)).sum(),
                           ((distance >= 20000) & (distance < 29000)).sum())

    def test_real_netcdf_to_model_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid = self.fixture_grid()
            grid_path = root/"grid.gpkg"
            grid.drop(columns=training.LAKE_FIELDS).to_file(grid_path,driver="GPKG")
            basins = gpd.GeoDataFrame(dict(lake_id=[1], volume_m3=[10000.], max_depth_m=[4.]),
                                     geometry=[box(260100,6650100,261000,6650900)],crs=25833)
            basin_path = root/"basins.gpkg"
            basins.to_file(basin_path,driver="GPKG")
            archive = root/"archive"
            archive.mkdir()
            rng = np.random.default_rng(42)
            for year in (2021,2022):
                with netCDF4.Dataset(archive/f"seNorge2018_{year}.nc","w") as ds:
                    for name,n in (("X",800),("Y",2),("time",365)):
                        ds.createDimension(name,n)
                    ds.createVariable("X","f8",("X",))[:] = 260500+np.arange(800)*1000
                    ds.createVariable("Y","f8",("Y",))[:] = [6650500,6649500]
                    times=ds.createVariable("time","f8",("time",))
                    times.units=f"hours since {year}-01-01"
                    times[:]=np.arange(365)*24+6
                    mapping=ds.createVariable("utm","i4")
                    mapping.proj4="+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs +towgs84=0,0,0"
                    var=ds.createVariable("tg","f4",("time","Y","X"),fill_value=-999.99)
                    var.units="Celsius"; var.grid_mapping="utm"
                    var[:]=rng.normal(size=(365,2,800))+np.sin(np.arange(365)[:,None,None]*.017)*10
            args=argparse.Namespace(grid=grid_path,basins=basin_path,archive_dir=archive,
                                    output_dir=root/"result",years=[2021,2022],pairs=250,
                                    date_step_days=3,trees=5,jobs=1)
            result=training.run(args)
            self.assertEqual(result["status"],"complete")
            self.assertEqual(result["accepted_pairs"],250)
            self.assertEqual(len(result["features"]),25)
            self.assertTrue(np.isfinite(result["validation"]["with_basins"]["rmse"]))
            labels=pd.read_csv(args.output_dir/"pair_labels.csv")
            predicted=inference.predict(args.output_dir,labels[["grid_id_1","grid_id_2"]].iloc[:3])
            self.assertTrue(predicted.predicted_rmse_c.ge(0).all())
            self.assertIn("basin_volume_m3_1",predicted)
            heldout=pd.read_csv(args.output_dir/"spatial_holdout_predictions.csv")
            self.assertAlmostEqual(np.sqrt(np.mean((heldout.target_error-heldout.predicted_with_basins)**2)),
                                   result["validation"]["with_basins"]["rmse"])
            with self.assertRaises(FileExistsError):
                training.run(args)
            args.output_dir = root / "random_result"
            args.date_step_days = 1
            args.days_per_month = 10
            args.temporal_split = "monthly-random"
            args.distance_distribution = "half-normal"
            args.distance_scale_m = 25000
            args.maximum_distance_m = 100000
            args.prepare_only = True
            prepared = training.run(args)
            self.assertEqual(prepared["status"], "prepared")
            self.assertFalse((args.output_dir / "model.joblib").exists())
            plan = pd.read_csv(args.output_dir / "selected_dates.csv")
            self.assertEqual(len(plan), 240)
            self.assertEqual(int((plan.partition == 0).sum()), 192)
            actual_grid = gpd.read_file(args.output_dir / "grid_features.gpkg")
            pairs = np.load(args.output_dir / "cache" / "pairs.npy")
            values = []
            for year in args.years:
                part = plan.loc[plan.year == year]
                with netCDF4.Dataset(archive / f"seNorge2018_{year}.nc") as ds:
                    values.extend([ds["tg"][int(day), actual_grid.nc_row.iloc[pairs[0]],
                                            actual_grid.nc_col.iloc[pairs[0]]].diagonal()
                                   for day in part.day_index])
            values = np.asarray(values)
            mask = plan.partition.to_numpy() == 0
            direct = []
            for a, b in ((values[:, 0], values[:, 1]), (values[:, 1], values[:, 0])):
                slope, intercept = np.polyfit(a[mask], b[mask], 1)
                direct.append(np.sqrt(np.mean((b[~mask]-slope*a[~mask]-intercept)**2)))
            from prepare_senorge_server import prepare
            import zipfile
            from remote_senorge.train_remote import load_data
            multi_zip = root / "multi.zip"
            prepare_multi(args.output_dir, root / "multi_cache", multi_zip, 250)
            multi_folder = root / "multi"
            with zipfile.ZipFile(multi_zip) as zipped:
                self.assertIsNone(zipped.testzip())
                zipped.extractall(multi_folder)
            previous = None
            for target in TARGETS:
                current, multi_meta = load_data(multi_folder, target)
                self.assertEqual(len(current["y"]), 250)
                self.assertEqual(multi_meta["target_metric"], target)
                if previous is not None:
                    for key in ("x", "train", "test", "grid_id_1", "grid_id_2"):
                        np.testing.assert_array_equal(current[key], previous[key])
                previous = current
            self.assertAlmostEqual(float(current["y"][0]), float(np.mean([
                np.quantile(np.abs(b[~mask] - np.polyval(np.polyfit(a[mask], b[mask], 1), a[~mask])), .95)
                for a, b in ((values[:, 0], values[:, 1]), (values[:, 1], values[:, 0]))
            ])), places=5)
            package = root / "package"
            prepare(args.output_dir, package)
            unpacked = root / "unpacked"
            with zipfile.ZipFile(root / "package.zip") as zipped:
                zipped.extractall(unpacked)
            data, meta = load_data(unpacked)
            self.assertAlmostEqual(float(data["y"][0]), float(np.mean(direct)), places=5)
            self.assertEqual(meta["train_days"], 192)
            self.assertEqual(meta["test_days"], 48)
            self.assertEqual(meta["maximum_distance_m"], 100000)
            self.assertEqual(meta["temporal_split"], "monthly-random")
            self.assertEqual(sum(b["spatial_test"] for b in meta["spatial_distance_bins"]),
                             int(data["test"].sum()))
            self.assertTrue((data["y"] >= 0).all())
            with self.assertRaises(FileExistsError):
                prepare(args.output_dir, package)
            percentile_rows = training.stream_labels(
                list(sorted(archive.glob("*.nc"))), actual_grid, pairs, args.output_dir / "cache",
                plan=plan, target_metric="absolute-error-p95")
            percentile_direct = []
            for a, b in ((values[:, 0], values[:, 1]), (values[:, 1], values[:, 0])):
                slope, intercept = np.polyfit(a[mask], b[mask], 1)
                percentile_direct.append(np.quantile(np.abs(b[~mask]-slope*a[~mask]-intercept), .95))
            self.assertAlmostEqual(percentile_rows.target_error.iloc[0],
                                   float(np.mean(percentile_direct)), places=5)
            p95_package = root / "p95_package"
            external_archives = root / "external_archives"
            external_archives.mkdir()
            for path in archive.glob("*.nc"):
                shutil.copyfile(path, external_archives / path.name)
            shutil.rmtree(args.output_dir / "cache" / "absolute_error_p95_v2")
            prepare(args.output_dir, p95_package, "absolute-error-p95",
                    archive_dir=external_archives)
            p95_cache_identity = json.loads(
                (args.output_dir / "cache" / "absolute_error_p95_v2" / "metadata.json").read_text())
            self.assertTrue(all(Path(item["path"]).parent == external_archives
                                for item in p95_cache_identity["archives"]))
            p95_unpacked = root / "p95_unpacked"
            with zipfile.ZipFile(root / "p95_package.zip") as zipped:
                zipped.extractall(p95_unpacked)
            p95_data, p95_meta = load_data(p95_unpacked)
            self.assertEqual(p95_meta["target_metric"], "absolute-error-p95")
            self.assertAlmostEqual(float(p95_data["y"][0]),
                                   float(np.mean(percentile_direct)), places=5)
            changed_plan = training.date_plan(args.years, days_per_month=10,
                                              temporal_split="monthly-random", seed=18)
            with self.assertRaisesRegex(ValueError, "Datoutvalg"):
                training.stream_labels(list(sorted(archive.glob("*.nc"))), actual_grid, pairs,
                                       args.output_dir / "cache", plan=changed_plan)


if __name__ == "__main__":
    unittest.main()
