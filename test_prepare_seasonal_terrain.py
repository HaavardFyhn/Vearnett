from pathlib import Path
import argparse
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import prepare_seasonal_terrain as preparation
import train_senorge_grid_pairs as training
import netCDF4
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS
import geopandas as gpd


class PreparationTests(unittest.TestCase):
    def test_resume_disk_budget_does_not_count_existing_allocations_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            cells = 20
            needed = preparation.additional_patch_bytes(p, cells)
            self.assertGreater(needed, cells*3*64*64*3)
            for name, dtype in (("terrain.npy", np.float16), ("terrain_valid.npy", np.uint8)):
                data = np.lib.format.open_memmap(p / name, mode="w+", dtype=dtype,
                                                shape=(cells, 3, 64, 64))
                data[:] = 0
                data.flush()
                data._mmap.close()
            self.assertEqual(preparation.additional_patch_bytes(p, cells), 0)

    def test_complete_portable_package_excludes_source_paths_and_verifies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, dem = root / "source", root / "dem"
            source.mkdir()
            dem.mkdir()
            (source / "cache").mkdir()
            x = np.concatenate([250000+block*100000+np.arange(3)*1000 for block in range(10)])
            y = np.full(len(x), 6650500.)
            grid = gpd.GeoDataFrame(dict(
                grid_id=np.arange(len(x)), x=x, y=y, elevation=np.full(len(x), 500),
                coast_dist_fjord=np.ones(len(x)), coast_dist_ocean=np.ones(len(x)), objtype=["land"]*len(x)),
                geometry=gpd.points_from_xy(x, y), crs=25833)
            for name in training.TERRAIN_FIELDS + training.LAKE_FIELDS:
                grid[name] = 0.
            grid.to_file(source / "grid_features.gpkg", driver="GPKG")
            pairs = np.array([[block*3+a, block*3+b] for block in range(10)
                              for a, b in ((0, 1), (0, 2), (1, 2))])
            np.save(source / "cache" / "pairs.npy", pairs)
            archives = []
            for year in range(1996, 2026):
                path = source / f"seNorge2018_{year}.nc"
                path.write_bytes(b"fixture")
                archives.append(training.fingerprint(path))
            training.write_json(source / "config.json", dict(archives=archives))
            training.date_plan(list(range(1996, 2026)), days_per_month=10,
                               temporal_split="monthly-random").to_csv(source / "selected_dates.csv", index=False)
            with rasterio.open(dem / "terrain.tif", "w", driver="GTiff", width=64, height=64,
                               count=1, dtype="float32", crs=25833, nodata=-9999,
                               transform=from_origin(249000, 6652000, 50, 50)) as dst:
                dst.write(np.full((64, 64), 500., dtype=np.float32), 1)

            def labels(archives, grid, pairs, cache, plan, variable):
                return pd.DataFrame(dict(target_error=np.full(len(pairs), 1 if variable == "tg" else 2)))

            args = argparse.Namespace(source=source, output=root / "package", dtm_dir=dem)
            with patch("prepare_seasonal_terrain.training.stream_labels", side_effect=labels), \
                    patch("prepare_seasonal_terrain.shutil.disk_usage",
                          return_value=argparse.Namespace(free=100*1024**3)):
                preparation.prepare(args)
            with zipfile.ZipFile(root / "package.zip") as zipped:
                self.assertIsNone(zipped.testzip())
                names = set(zipped.namelist())
                self.assertNotIn("cache/source_config.json", names)
                self.assertNotIn("config.json", names)
                meta = json.loads(zipped.read("seasonal_manifest.json"))
                self.assertNotIn(str(source), json.dumps(meta))
                self.assertEqual(names, set(meta["checksums"]) | {"seasonal_manifest.json"})
                self.assertEqual(meta["targets"], ["winter_tg", "summer_tg", "winter_tn", "summer_tn"])
            receipt = json.loads((args.output / "cache" / "complete.json").read_text())
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(sum(receipt["spatial_counts"].values()), len(pairs))

    def test_all_equivalent_crs_tiles_enter_mosaic_and_incomplete_vrt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tiles, cache = root / "tiles", root / "cache"
            tiles.mkdir()
            cache.mkdir()
            for i, crs in enumerate((25833, CRS.from_epsg(25833).to_wkt(version="WKT1_ESRI"))):
                with rasterio.open(tiles / f"dem_{i}.tif", "w", driver="GTiff", width=64, height=64,
                                   count=1, dtype="float32", crs=crs, nodata=-9999,
                                   transform=from_origin(300000+i*3200, 6650000, 50, 50)) as dst:
                    dst.write(np.full((64, 64), 500+i*100, dtype=np.float32), 1)
            rasters = preparation.terrain_rasters(tiles, cache)
            for path in rasters:
                with rasterio.open(path) as src:
                    self.assertAlmostEqual(float(next(src.sample([(301000, 6649000)]))[0]), 500)
                    self.assertAlmostEqual(float(next(src.sample([(304000, 6649000)]))[0]), 600)
            (cache / "terrain.vrt").write_text("<VRTDataset/>")
            with self.assertRaisesRegex(ValueError, "omitted DEM tiles"):
                preparation.terrain_rasters(tiles, cache)

    def test_winter_minimum_labels_match_direct_regressions_and_cache_cannot_mix_variables(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            rng = np.random.default_rng(42)
            archives, records = [], []
            plan = training.date_plan(list(range(2021, 2025)), days_per_month=10,
                                       temporal_split="monthly-random")
            plan = plan.loc[pd.to_datetime(plan.date).dt.month.isin([12, 1, 2])].reset_index(drop=True)
            for year in range(2021, 2025):
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
                path = root / f"seNorge2018_{year}.nc"
                signal = np.sin(np.arange(len(dates))*.017)*10
                tg = signal[:, None] + rng.normal(size=(len(dates), 3))
                tn = tg - 3 + rng.normal(size=(len(dates), 3))*[1, 2, 3]
                selected = plan.loc[plan.year == year]
                records.extend(tn[selected.day_index.to_numpy()])
                with netCDF4.Dataset(path, "w") as ds:
                    for name, count in (("X", 3), ("Y", 1), ("time", len(dates))):
                        ds.createDimension(name, count)
                    ds.createVariable("X", "f8", ("X",))[:] = [500, 1500, 2500]
                    ds.createVariable("Y", "f8", ("Y",))[:] = [6650500]
                    t = ds.createVariable("time", "f8", ("time",))
                    t.units = f"hours since {year}-01-01"
                    t[:] = np.arange(len(dates))*24 + 6
                    mapping = ds.createVariable("utm", "i4")
                    mapping.proj4 = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"
                    for name, values in (("tg", tg), ("tn", tn)):
                        var = ds.createVariable(name, "f4", ("time", "Y", "X"))
                        var.units, var.grid_mapping = "Celsius", "utm"
                        var[:] = values[:, None]
                archives.append(path)
            grid = pd.DataFrame(dict(nc_row=[0, 0, 0], nc_col=[0, 1, 2]))
            pairs = np.array([[0, 1], [1, 2]])
            actual = training.stream_labels(archives, grid, pairs, cache, plan=plan, variable="tn")
            # Compare using the float32 values actually written to NetCDF, not float64 originals.
            records = np.asarray(records, dtype=np.float32).astype(float)
            mask = plan.partition.to_numpy() == 0
            expected = []
            for first, second in pairs:
                errors = []
                for a, b in ((records[:, first], records[:, second]), (records[:, second], records[:, first])):
                    slope, intercept = np.polyfit(a[mask], b[mask], 1)
                    errors.append(np.sqrt(np.mean((b[~mask]-slope*a[~mask]-intercept)**2)))
                expected.append(np.mean(errors))
            np.testing.assert_allclose(actual.target_error, expected, rtol=1e-8)
            again = training.stream_labels(archives, grid, pairs, cache, plan=plan, variable="tn")
            pd.testing.assert_frame_equal(actual, again)
            with self.assertRaisesRegex(ValueError, "Datoutvalg"):
                training.stream_labels(archives, grid, pairs, cache, plan=plan, variable="tg")

    def test_four_targets_use_exact_seasonal_dates(self):
        plan = training.date_plan(list(range(1996, 2026)), days_per_month=10, temporal_split="monthly-random")
        for _, months, _ in preparation.TARGETS:
            selected = plan.loc[pd.to_datetime(plan.date).dt.month.isin(months)]
            self.assertEqual(len(selected), 900)
            self.assertEqual(int((selected.partition == 0).sum()), 720)
            self.assertEqual(int((selected.partition == 1).sum()), 180)

    def test_buffered_pairs_do_not_share_blocks_or_terrain_support(self):
        xy = np.array([[block*100000 + offset, 50000] for block in range(20)
                       for offset in (1000, 40000, 41000, 99000)], dtype=float)
        pairs = np.array([[block*4+1, block*4+2] for block in range(20)] +
                         [[block*4, block*4+1] for block in range(20)])
        split = preparation.buffered_split(xy, pairs)
        self.assertTrue((split[20:] == -1).all())
        sets = [set(map(tuple, np.floor(xy[np.unique(pairs[split == i])]/100000)))
                for i in range(3)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for a in range(3):
            for b in range(a+1, 3):
                delta = np.abs(xy[np.unique(pairs[split == a])][:, None]-xy[np.unique(pairs[split == b])])
                self.assertTrue((delta.max(axis=2) >= 33000).all())

    def test_patch_orientation_resolution_quantization_and_nodata_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dem.tif"
            raw = (np.arange(128)[:, None]*10 + np.arange(128)[None, :]).astype(np.float32)
            raw[0:64, 0] = -9999
            with rasterio.open(path, "w", driver="GTiff", width=128, height=128, count=1,
                               dtype="float32", crs=25833, nodata=-9999,
                               transform=from_origin(0, 6400, 50, 50)) as dst:
                dst.write(raw, 1)
            with rasterio.open(path) as src:
                h, valid = preparation.patch(src, 1600, 4800, 50, 200)
                self.assertEqual(h.shape, (64, 64))
                self.assertEqual(h.dtype, np.float16)
                self.assertTrue((valid[:, 0] == 0).all())
                self.assertTrue((h[:, 0] == 0).all())
                np.testing.assert_allclose(h[:, 1:].astype(float)*1000+200, raw[:64, 1:64], atol=.5)
                self.assertGreater(float(h[63, 63]), float(h[0, 63]))
                self.assertGreater(float(h[10, 63]), float(h[10, 1]))
                _, missing = preparation.patch(src, -10000, 20000, 500, 200)
                self.assertLess(float(missing.mean()), .5)


if __name__ == "__main__":
    unittest.main()
