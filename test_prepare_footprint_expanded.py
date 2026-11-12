import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import prepare_footprint_expanded as prep
import numpy as np
import pandas as pd
import netCDF4
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd

from remote_senorge.footprint_atlas import pair_features, FEATURE_FIELDS


class ExpandedPreparationTests(unittest.TestCase):
    def grid(self):
        xx, yy = np.meshgrid(np.arange(41000, 110000, 1000), np.arange(6641000, 6710000, 1000))
        return pd.concat([pd.DataFrame(dict(x=xx.ravel()+b*150000, y=yy.ravel()))
                          for b in range(12)], ignore_index=True)

    def test_twelve_thousand_unique_scene_safe_references(self):
        grid = self.grid()
        geo, report = prep.expanded_geometry(grid)
        self.assertEqual([report[key]["references"] for key in ("train", "validation", "test")],
                         [10000, 1000, 1000])
        self.assertEqual(len(np.unique(geo["refs"])), 12000)
        regions = [set(map(tuple, np.floor(geo["xy"][geo["split"] == i]/150000).astype(int)))
                   for i in range(3)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(regions[a] & regions[b])
            # Exact input support envelopes, including half-cell NW offset.
            aa, bb = geo["xy"][geo["split"] == a], geo["xy"][geo["split"] == b]
            for start in range(0, len(aa), 128):
                delta = np.abs(aa[start:start+128, None]-bb[None])
                self.assertFalse((delta < 80000).all(axis=2).any())
        for i in range(0, len(geo["refs"]), 137):
            self.assertEqual(geo["mapping"][i, 32, 32], geo["refs"][i])
            row, col = np.where(geo["mapping"][i] >= 0)
            expected = geo["xy"][i]+np.column_stack([col-32, 32-row])*1000
            np.testing.assert_array_equal(grid[["x", "y"]].to_numpy()[geo["mapping"][i, row, col]], expected)
        with self.assertRaisesRegex(ValueError, "Only"):
            prep.expanded_geometry(grid, [1000000, 1000, 1000])

    def test_plane_ridge_orientation_and_nodata_descriptors(self):
        rr, cc = np.indices((100, 100))
        plane = np.ma.array(500+cc*2+rr*3, mask=False)
        values = prep.fine_descriptors(plane)
        self.assertEqual(values[0], 1)
        self.assertAlmostEqual(values[4], .2, places=6)
        self.assertAlmostEqual(values[5], .3, places=6)
        self.assertAlmostEqual(values[3], np.hypot(.2, .3), places=6)
        self.assertLess(values[6], values[7])
        self.assertLess(values[7], values[9])
        ridge = np.ma.array(np.broadcast_to(500-np.abs(cc[0]-50)*2, (100, 100)), mask=False)
        ridge_values = prep.fine_descriptors(ridge)
        self.assertGreater(ridge_values[4], 0)
        self.assertEqual(ridge_values[5], 0)
        plane.mask[:50] = True
        self.assertEqual(prep.fine_descriptors(plane)[0], .5)
        plane.mask[:] = True
        np.testing.assert_array_equal(prep.fine_descriptors(plane), 0)

    def test_portable_features_exactly_match_existing_pair_builder(self):
        rng = np.random.default_rng(42)
        table = pd.DataFrame(rng.normal(size=(20, 16)), columns=FEATURE_FIELDS)
        table["objtype_code"] = np.arange(20) % 3
        table["objtype"] = table.objtype_code.astype(str)
        pairs = np.column_stack([np.full(20, 4), np.arange(20)])
        expected = prep.training.pair_features(table, pairs).to_numpy()
        actual = pair_features(table[FEATURE_FIELDS].to_numpy(), 4, np.arange(20))
        np.testing.assert_array_equal(actual, expected)

    def test_six_labels_direct_ols_missing_values_and_resume(self):
        plan = prep.training.date_plan(list(range(1996, 2026)), days_per_month=10,
                                       temporal_split="monthly-random")
        rng = np.random.default_rng(52)
        weather = rng.normal(3, 5, (2, 3, 3600)).astype(np.float32)
        weather[0, 1, :8] = np.nan
        weather[:, 2, :1000] = np.nan
        output, masks = np.zeros((6, 2), np.float32), np.zeros((6, 2), bool)
        prep.label_chunk(weather, 0, np.array([1, 2]), plan, output, masks)
        for i in range(6):
            season = np.ones(3600, bool) if i < 2 else pd.to_datetime(plan.date).dt.month.isin(
                (12, 1, 2) if i < 4 else (6, 7, 8)).to_numpy()
            fit = season & (plan.partition == 0)
            test = season & (plan.partition == 1)
            a, b = weather[i % 2, 0], weather[i % 2, 1]
            valid = np.isfinite(a) & np.isfinite(b)
            train, evaluation = fit & valid, test & valid
            errors = []
            for x, y in ((a, b), (b, a)):
                design = np.column_stack([x[train], np.ones(train.sum())])
                slope, intercept = np.linalg.lstsq(design, y[train], rcond=None)[0]
                errors.append(np.sqrt(np.mean((slope*x[evaluation]+intercept-y[evaluation])**2)))
            self.assertTrue(masks[i, 0])
            self.assertFalse(masks[i, 1])
            self.assertAlmostEqual(output[i, 0], np.mean(errors), places=5)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            np.save(cache / "sampled_weather.npy", weather)
            geo = dict(refs=np.array([0]), mapping=np.full((1, 64, 64), -1, np.int32))
            geo["mapping"][0, 32, 33] = 1
            geo["mapping"][0, 32, 34] = 2
            prep.build_labels(geo, plan, cache / "sampled_weather.npy", root, cache)
            before = (root / "labels.npy").read_bytes()
            with patch.object(prep, "label_chunk", side_effect=AssertionError("Resume recomputed a finished scene")):
                prep.build_labels(geo, plan, cache / "sampled_weather.npy", root, cache)
            self.assertEqual(before, (root / "labels.npy").read_bytes())

    def test_weather_atlas_decodes_sources_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = prep.training.date_plan([2021], days_per_month=10, temporal_split="monthly-random")
            grid = pd.DataFrame(dict(x=[500., 1500.], y=[6650500., 6650500.], nc_row=[0, 0], nc_col=[0, 1]))
            path = root / "seNorge2018_2021.nc"
            with netCDF4.Dataset(path, "w") as ds:
                for dim, count in (("X", 2), ("Y", 1), ("time", 365)):
                    ds.createDimension(dim, count)
                ds.createVariable("X", "f8", ("X",))[:] = [500, 1500]
                ds.createVariable("Y", "f8", ("Y",))[:] = [6650500]
                time = ds.createVariable("time", "f8", ("time",))
                time.units = "days since 2021-01-01"
                time[:] = np.arange(365)
                for offset, variable in enumerate(("tg", "tn")):
                    value = ds.createVariable(variable, "f4", ("time", "Y", "X"), fill_value=-9999)
                    value.units = "Celsius"
                    value[:] = np.arange(365)[:, None, None]+np.array([0., .5])[None, None]+offset
            atlas = prep.build_weather(grid, plan, [path], root)
            result = np.load(atlas)
            np.testing.assert_allclose(result[0, 0], plan.day_index)
            np.testing.assert_allclose(result[1, 1], plan.day_index+1.5)
            with patch.object(prep.netCDF4, "Dataset", side_effect=AssertionError("Reopened finished source")):
                self.assertEqual(atlas, prep.build_weather(grid, plan, [path], root))

    def test_complete_portable_package_without_sensitive_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, terrain, fine = root / "source", root / "terrain", root / "fine"
            source.mkdir()
            (terrain / "cache").mkdir(parents=True)
            fine.mkdir()
            points = []
            for block in range(12):
                points.extend([[75000+block*150000, 6675000], [76000+block*150000, 6675000]])
            frame = gpd.GeoDataFrame(dict(x=np.array(points)[:, 0], y=np.array(points)[:, 1],
                                         elevation=500., coast_dist_fjord=1000., coast_dist_ocean=2000.,
                                         objtype="land"),
                                     geometry=gpd.points_from_xy(*np.array(points).T), crs=25833)
            for name in prep.training.TERRAIN_FIELDS + prep.training.LAKE_FIELDS:
                frame[name] = 0.
            frame.to_file(source / "grid_features.gpkg", driver="GPKG")
            plan = prep.training.date_plan(list(range(1996, 2026)), days_per_month=10,
                                          temporal_split="monthly-random")
            plan.to_csv(source / "selected_dates.csv", index=False)
            archives = []
            for year in range(1996, 2026):
                path = source / f"seNorge2018_{year}.nc"
                path.write_bytes(b"fixture")
                archives.append(prep.training.fingerprint(path))
            prep.write_json(source / "config.json", dict(archives=archives))
            for path, res, pixels in ((terrain / "cache" / "dem_100m.tif", 100, 800),
                                      (fine / "fine.tif", 10, 100)):
                with rasterio.open(path, "w", driver="GTiff", width=pixels, height=pixels, count=1,
                                   dtype="float32", crs=25833, nodata=-9999,
                                   transform=from_origin(74500 if res == 10 else 35000,
                                                         6675500 if res == 10 else 6715000, res, res)) as dst:
                    dst.write(np.full((pixels, pixels), 500., np.float32), 1)

            def label_fixture(geo, plan, weather, folder, cache):
                labels = np.ones((len(geo["refs"]), 6, 64, 64), np.float32)
                mask = np.repeat((geo["mapping"] >= 0)[:, None], 6, axis=1)
                np.save(folder / "labels.npy", labels)
                np.save(folder / "label_mask.npy", mask)
                return {name: int(mask[:, i].sum()) for i, name in enumerate(prep.TARGETS)}

            args = argparse.Namespace(source=source, terrain=terrain, dtm10=fine, counts=[2, 2, 2],
                                      output=root / "package")
            with patch.object(prep, "build_weather", return_value=None), \
                    patch.object(prep, "build_labels", side_effect=label_fixture), \
                    patch.object(prep.shutil, "disk_usage", return_value=argparse.Namespace(free=100*1024**3)):
                prep.prepare(args)
            with zipfile.ZipFile(root / "package.zip") as archive:
                self.assertIsNone(archive.testzip())
                manifest = json.loads(archive.read("footprint_manifest.json"))
                self.assertEqual(set(archive.namelist()), set(prep.FILES) |
                                 {"footprint_manifest.json", "geography_report.json"})
                self.assertNotIn(str(source), json.dumps(manifest))
                self.assertNotIn("pipeline_config.py", archive.namelist())
                self.assertEqual(manifest["features"][-6:], prep.ABSOLUTE_FIELDS)
                self.assertEqual(len(manifest["features"]), 31)
            # Emulate the completed relative-only v2 package, then upgrade just its inputs.
            manifest["features"] = prep.training.FEATURES
            manifest.pop("absolute_features")
            manifest["architecture"] = "reference-resunet-spatial10-v2"
            prep.write_json(args.output / "footprint_manifest.json", manifest)
            with zipfile.ZipFile(root / "package.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name in prep.FILES+["footprint_manifest.json", "geography_report.json"]:
                    archive.write(args.output / name, name)
            prep.write_json(args.output / "cache" / "complete.json",
                            dict(sha256=prep.sha256(root / "package.zip")))
            scientific = {name: prep.sha256(args.output / name) for name in prep.FILES[:8]}
            with patch.object(prep, "build_terrain", side_effect=AssertionError("Recomputed terrain")), \
                    patch.object(prep, "build_weather", side_effect=AssertionError("Recomputed weather")), \
                    patch.object(prep, "build_labels", side_effect=AssertionError("Recomputed labels")), \
                    patch.object(prep.shutil, "disk_usage", return_value=argparse.Namespace(free=100*1024**3)):
                prep.repackage(args.output)
            self.assertEqual(scientific, {name: prep.sha256(args.output / name) for name in scientific})
            with zipfile.ZipFile(root / "package.zip") as archive:
                self.assertIsNone(archive.testzip())
                updated = json.loads(archive.read("footprint_manifest.json"))
                self.assertEqual(updated["absolute_features"], prep.ABSOLUTE_FIELDS)
                self.assertEqual(updated["architecture"], "reference-resunet-spatial10-absolute-v2")
            receipt = json.loads((args.output / "cache" / "complete.json").read_text())
            self.assertEqual(receipt["sha256"], prep.sha256(root / "package.zip"))
            for name in ("terrain_height.npy", "terrain_valid.npy", "spatial10.npy", "labels.npy", "label_mask.npy"):
                (args.output / name).rename(args.output / (name+".released"))


if __name__ == "__main__":
    unittest.main()
