from pathlib import Path
import argparse
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import prepare_footprint_training as prep
import train_senorge_grid_pairs as training
import netCDF4
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd


class PreparationTests(unittest.TestCase):
    def test_complete_portable_map_package_and_handle_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, terrain, fine = root / "source", root / "terrain", root / "fine"
            source.mkdir()
            (terrain / "cache").mkdir(parents=True)
            fine.mkdir()
            x = np.concatenate([np.arange(43000, 58000, 1000)+b*100000 for b in range(12)])
            frame = gpd.GeoDataFrame(dict(
                grid_id=np.arange(len(x)), x=x.astype(float), y=np.full(len(x), 6650000.),
                elevation=np.full(len(x), 500.), coast_dist_fjord=np.ones(len(x)),
                coast_dist_ocean=np.ones(len(x)), objtype=["land"]*len(x)),
                geometry=gpd.points_from_xy(x, np.full(len(x), 6650000)), crs=25833)
            for name in training.TERRAIN_FIELDS + training.LAKE_FIELDS:
                frame[name] = 0.
            frame.to_file(source / "grid_features.gpkg", driver="GPKG")
            plan = training.date_plan(list(range(1996, 2026)), days_per_month=10,
                                      temporal_split="monthly-random")
            plan.to_csv(source / "selected_dates.csv", index=False)
            records = []
            for year in range(1996, 2026):
                path = source / f"seNorge2018_{year}.nc"
                path.write_bytes(b"fixture")
                records.append(training.fingerprint(path))
            training.write_json(source / "config.json", dict(archives=records))
            geo = prep.geometry(frame, (2, 2, 2))
            ref = frame.iloc[int(geo["refs"][0])]
            for path, resolution, pixels in (
                    (terrain / "cache" / "dem_100m.tif", 100, 800),
                    (fine / "fine.tif", 10, 128)):
                with rasterio.open(path, "w", driver="GTiff", width=pixels, height=pixels,
                                   count=1, dtype="float32", crs=25833, nodata=-9999,
                                   transform=from_origin(ref.x-pixels*resolution/2,
                                                         ref.y+pixels*resolution/2, resolution, resolution)) as dst:
                    dst.write(np.full((pixels, pixels), 500., np.float32), 1)

            def label_fixture(archives, plan, grid, geometry_data, cache, labels, mask):
                for a, row, col in zip(geometry_data["anchor"], geometry_data["row"], geometry_data["col"]):
                    labels[a, :, row, col] = 1.
                    mask[a, :, row, col] = True
                labels.flush()
                mask.flush()

            args = argparse.Namespace(source=source, terrain=terrain, dtm10=fine,
                                      counts=[2, 2, 2], output=root / "package")
            with patch("prepare_footprint_training.label_maps", side_effect=label_fixture), \
                    patch("prepare_footprint_training.shutil.disk_usage",
                          return_value=argparse.Namespace(free=100*1024**3)):
                prep.prepare(args)
            with zipfile.ZipFile(root / "package.zip") as archive:
                self.assertIsNone(archive.testzip())
                manifest = json.loads(archive.read("footprint_manifest.json"))
                self.assertEqual(set(archive.namelist()), set(manifest["checksums"]) | {"footprint_manifest.json"})
                self.assertNotIn(str(source), json.dumps(manifest))
                self.assertEqual(manifest["targets"], prep.TARGETS)
            # Windows rename fails if preparation/validation retained any mmap handle.
            for name in ("scenes.npy", "local10.npy", "features.npy", "labels.npy", "label_mask.npy"):
                path = args.output / name
                path.rename(path.with_suffix(".released"))

    def test_reference_scenes_stay_in_disjoint_blocks_and_map_north_up(self):
        xx, yy = np.meshgrid(np.arange(43000, 58000, 1000), np.arange(6643000, 6658000, 1000))
        grid = pd.concat([pd.DataFrame(dict(x=xx.ravel()+b*100000, y=yy.ravel()))
                          for b in range(12)], ignore_index=True)
        geo = prep.geometry(grid, counts=(4, 3, 3))
        groups = [set(map(tuple, np.floor(geo["xy"][geo["split"] == i]/100000).astype(int)))
                  for i in range(3)]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for i, ref in enumerate(geo["refs"]):
            self.assertEqual(geo["mapping"][i, 32, 32], ref)
            for row, col in np.argwhere(geo["mapping"][i] >= 0):
                target = geo["mapping"][i, row, col]
                np.testing.assert_array_equal(grid.iloc[target][["x", "y"]].to_numpy(),
                                              grid.iloc[ref][["x", "y"]].to_numpy()+[(col-32)*1000, (32-row)*1000])

    def test_terrain_window_keeps_shape_nodata_and_reference_height(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dem.tif"
            heights = np.arange(10000, dtype=np.float32).reshape(100, 100)
            heights[0] = -9999
            with rasterio.open(path, "w", driver="GTiff", width=100, height=100, count=1,
                               dtype="float32", crs=25833, nodata=-9999,
                               transform=from_origin(0, 1000, 10, 10)) as dst:
                dst.write(heights, 1)
            with rasterio.open(path) as src:
                values = prep.terrain_window(src, 500, 500, 10, 100, 100)
            self.assertEqual(values.shape, (2, 100, 100))
            self.assertFalse(values[1, 0].any())
            self.assertTrue((values[0, 0] == 0).all())
            self.assertAlmostEqual(float(values[0, 10, 10]), float((heights[10, 10]-100)/1000), places=3)

    def test_six_dense_labels_match_direct_ols_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            plan = training.date_plan(list(range(2021, 2025)), days_per_month=10,
                                      temporal_split="monthly-random")
            grid = pd.DataFrame(dict(x=[500., 1500.], y=[6650500.]*2, nc_row=[0, 0], nc_col=[0, 1]))
            geo = dict(pairs=np.array([[0, 1]], np.int32), anchor=np.array([0]),
                       row=np.array([32]), col=np.array([33]))
            paths, samples = [], {"tg": [], "tn": []}
            rng = np.random.default_rng(42)
            for year in range(2021, 2025):
                path = root / f"seNorge2018_{year}.nc"
                paths.append(path)
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
                with netCDF4.Dataset(path, "w") as ds:
                    for name, size in (("X", 2), ("Y", 1), ("time", len(dates))):
                        ds.createDimension(name, size)
                    ds.createVariable("X", "f8", ("X",))[:] = [500, 1500]
                    ds.createVariable("Y", "f8", ("Y",))[:] = [6650500]
                    time = ds.createVariable("time", "f8", ("time",))
                    time.units = f"hours since {year}-01-01"
                    time[:] = np.arange(len(dates))*24
                    for variable in ("tg", "tn"):
                        values = rng.normal(0, 5, (len(dates), 2)).astype(np.float32)
                        var = ds.createVariable(variable, "f4", ("time", "Y", "X"))
                        var.units = "Celsius"
                        var[:] = values[:, None]
                        samples[variable].extend(values[plan.loc[plan.year == year].day_index])
            labels = np.lib.format.open_memmap(root / "labels.npy", mode="w+", dtype=np.float32,
                                               shape=(1, 6, 64, 64))
            mask = np.lib.format.open_memmap(root / "mask.npy", mode="w+", dtype=bool, shape=labels.shape)
            labels[:] = 0
            mask[:] = False
            try:
                prep.label_maps(paths, plan, grid, geo, cache, labels, mask)
                for i, target in enumerate(prep.TARGETS):
                    variable = target.rsplit("_", 1)[1]
                    months = pd.to_datetime(plan.date).dt.month
                    keep = np.ones(len(plan), bool) if i < 2 else months.isin(
                        (12, 1, 2) if i < 4 else (6, 7, 8)).to_numpy()
                    values = np.asarray(samples[variable])[keep]
                    fit = plan.partition.to_numpy()[keep] == 0
                    errors = []
                    for a, b in ((values[:, 0], values[:, 1]), (values[:, 1], values[:, 0])):
                        design = np.column_stack([np.ones(len(a)), a])
                        coef = np.linalg.lstsq(design[fit], b[fit], rcond=None)[0]
                        errors.append(np.sqrt(np.mean((design[~fit]@coef-b[~fit])**2)))
                    self.assertTrue(mask[0, i, 32, 33])
                    self.assertAlmostEqual(float(labels[0, i, 32, 33]), float(np.mean(errors)), places=5)
                saved = labels.copy()
                prep.label_maps(paths, plan, grid, geo, cache, labels, mask)
                np.testing.assert_array_equal(saved, labels)
            finally:
                labels._mmap.close()
                mask._mmap.close()


if __name__ == "__main__":
    unittest.main()
