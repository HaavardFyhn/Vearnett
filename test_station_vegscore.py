from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import geopandas as gpd
import pyogrio
from shapely.geometry import LineString

from station_roads import RoadNetwork, SCORE_COMPONENTS
from test_station_app import FakeEngine


class VegScoreTests(unittest.TestCase):
    def test_municipal_roads_are_excluded_from_cached_dataset_without_changing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = gpd.GeoDataFrame(dict(
                linkid=["a", "b", "c"], vegnokkel=["E6", "Kv12", "F17"],
                roadcategory=["Europaveg", "KommunalVeg", "Fylkesveg"],
                kommunenavn=["A"]*3, fylkesnr=["03"]*3, istunnel=["0"]*3, isferry=["0"]*3,
                VS_Total=[10., 20., 30.], VS_Total_klasse=[1., 2., 3.], unused=["unused"]*3,
                **{field: [1.]*3 for field, _, _ in SCORE_COMPONENTS}),
                geometry=[LineString([(x, -100), (x, 100)]) for x in [0, 1000, 2000]], crs=25833)
            for suffix, driver in ((".gdb", "OpenFileGDB"), (".gpkg", "GPKG")):
                with self.subTest(driver=driver):
                    source = root / ("roads" + suffix)
                    pyogrio.write_dataframe(frame, source, layer="Vegnett_VegScore_full", driver=driver)
                    cache = root / ("cache_" + driver)
                    roads = RoadNetwork(source, cache_dir=cache)
                    self.assertEqual(pyogrio.read_info(source)["features"], 3)
                    self.assertEqual(pyogrio.read_info(roads.path)["features"], 2)
                    self.assertNotIn("unused", pyogrio.read_info(roads.path)["fields"])
                    self.assertEqual(set(pyogrio.read_dataframe(roads.path).roadcategory), {"Europaveg", "Fylkesveg"})
                    self.assertNotIn("K", [item["code"] for item in roads.metadata["categories"]])
                    with patch("station_roads.pyogrio.write_dataframe", side_effect=AssertionError("Must reuse cache")):
                        self.assertEqual(RoadNetwork(source, cache_dir=cache).path, roads.path)

    def test_existing_non_municipal_geopackage_is_compacted_without_altering_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roads.gpkg"
            frame = gpd.GeoDataFrame(dict(
                linkid=["a"], vegnokkel=["E6"], roadcategory=["E"], kommunenavn=["A"],
                istunnel=["0"], isferry=["0"], VS_Total=[10.], VS_Total_klasse=[1.],
                unused=["not needed"]),
                geometry=[LineString([(0, 0), (100, 100)])], crs=25833)
            pyogrio.write_dataframe(frame, path, layer="roads", driver="GPKG")
            roads = RoadNetwork(path)
            self.assertIn("unused", pyogrio.read_info(path)["fields"])
            self.assertNotIn("unused", pyogrio.read_info(roads.path)["fields"])
            self.assertEqual(roads.metadata["features"], 1)

    def test_geodatabase_layer_uses_compact_versioned_spatial_index_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "ProssExport.gdb"
            frame = gpd.GeoDataFrame(dict(
                linkid=["a", "b"], vegnokkel=["E6", "F17"], roadcategory=["E", "F"],
                kommunenavn=["A"]*2, fylkesnr=["03", "11"], istunnel=["0"]*2, isferry=["0"]*2,
                VS_Total=[10., 50.], VS_Total_klasse=[1., 5.], unused=["unused"]*2,
                **{field: [float(i+1)]*2 for i, (field, _, _) in enumerate(SCORE_COMPONENTS)}),
                geometry=[LineString([(x, -100, 10), (x, 100, 10)]) for x in [0, 1000]], crs=25833)
            pyogrio.write_dataframe(frame, source, layer="Vegnett_VegScore_full", driver="OpenFileGDB")
            other = frame.iloc[:1].copy()
            pyogrio.write_dataframe(other, source, layer="Vegnett_VegScore", driver="OpenFileGDB")
            cache = Path(folder) / "cache"
            road = RoadNetwork(source, cache_dir=cache)
            self.assertEqual(road.source_path, source)
            self.assertEqual(road.path.suffix, ".gpkg")
            self.assertEqual(road.metadata["features"], 2)
            self.assertNotIn("unused", pyogrio.read_info(road.path)["fields"])
            self.assertEqual(pyogrio.list_layers(road.path)[:, 0].tolist(), ["Vegnett_VegScore_full"])
            self.assertFalse(pyogrio.read_dataframe(road.path).geometry.has_z.any())
            fid = int(road.attributes["fid"][0])
            self.assertAlmostEqual(road.components(dict(fid=fid))["components"][0]["share"], 1/21)
            with patch("station_roads.pyogrio.write_dataframe", side_effect=AssertionError("Must reuse cache")):
                cached = RoadNetwork(source, cache_dir=cache)
            self.assertEqual(cached.path, road.path)
            self.assertEqual(set(cached.county_rows), {"03", "11"})
            import os
            datafile = next(source.glob("*.gdbtable"))
            stat = datafile.stat()
            os.utime(datafile, ns=(stat.st_atime_ns, stat.st_mtime_ns+10000000))
            refreshed = RoadNetwork(source, cache_dir=cache)
            self.assertNotEqual(refreshed.path, road.path)

    def test_portable_roads_refresh_old_stage_and_use_full_layer(self):
        from build_portable import prepare_roads
        import zipfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "root"
            source = root / "data" / "ProssExport.gdb"
            source.parent.mkdir(parents=True)
            frame = gpd.GeoDataFrame(dict(
                linkid=["full", "municipal"], vegnokkel=["E6", "Kv1"], roadcategory=["E", "KommunalVeg"],
                kommunenavn=["A"]*2, fylkesnr=["03"]*2, istunnel=["0"]*2, isferry=["0"]*2,
                VS_Total=[50.]*2, VS_Total_klasse=[5.]*2,
                **{field: [1.]*2 for field, _, _ in SCORE_COMPONENTS}),
                geometry=[LineString([(0, 0), (100, 100)]), LineString([(200, 0), (300, 100)])], crs=25833)
            pyogrio.write_dataframe(frame, source, layer="Vegnett_VegScore_full", driver="OpenFileGDB")
            stage = Path(folder) / "stage"
            stage.mkdir()
            with patch("build_portable.ROOT", root):
                manifest = prepare_roads(stage)
            frame["VS_Total"] = 70.
            pyogrio.write_dataframe(frame, source, layer="Vegnett_VegScore_full", driver="OpenFileGDB")
            with patch("build_portable.ROOT", root):
                manifest = prepare_roads(stage)
            self.assertEqual(manifest["layer"], "Vegnett_VegScore_full")
            self.assertEqual((manifest["features"], manifest["source_features"]), (1, 2))
            self.assertFalse((stage / "data" / "ProssExport.gdb").exists())
            self.assertEqual(pyogrio.read_info(source)["features"], 2)
            archive = Path(folder) / "portable.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for path in stage.rglob("*"):
                    if path.is_file():
                        bundle.write(path, path.relative_to(stage))
            extracted = Path(folder) / "extracted"
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)
            roads = RoadNetwork(extracted / manifest["source"])
            self.assertEqual(roads.metadata["layer"], "Vegnett_VegScore_full")
            self.assertEqual(roads.metadata["score_max"], 70.)

    def test_components_are_read_on_demand_with_relative_shares(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.gpkg"
            frame = gpd.GeoDataFrame(dict(
                linkid=["a", "b"], vegnokkel=["E6", "F17"], roadcategory=["Europaveg", "Fylkesveg"],
                kommunenavn=["A"]*2, istunnel=["0"]*2, isferry=["0"]*2,
                VS_Total=[10., 50.], VS_Total_klasse=[1., 5.],
                **{field: [float(i + 1), np.nan if i == 0 else 0.]
                   for i, (field, _, _) in enumerate(SCORE_COMPONENTS)}),
                geometry=[LineString([(x, -100), (x, 100)]) for x in [0, 1000]], crs=25833)
            pyogrio.write_dataframe(frame, path, layer="scores", driver="GPKG")
            road = RoadNetwork(path)
            fid = int(road.attributes["fid"][0])
            with patch.object(road, "read", side_effect=AssertionError("No map geometry reload")):
                result = road.components(dict(fid=fid))
            self.assertEqual(result["roadid"], "E6")
            self.assertEqual([item["label"] for item in result["components"]],
                             [label for _, label, _ in SCORE_COMPONENTS])
            self.assertAlmostEqual(sum(item["share"] for item in result["components"]), 1)
            self.assertAlmostEqual(result["components"][0]["share"], 1 / 21)
            self.assertTrue(all("value" not in item for item in result["components"]))
            self.assertTrue(all(field not in road.read_fields for field, _, _ in SCORE_COMPONENTS))
            from station_app import create_app
            client = create_app(FakeEngine(), road_network=road).test_client()
            response = client.post("/api/roads/components", json=dict(fid=fid))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), result)
            self.assertEqual(client.post("/api/roads/components", json=[]).status_code, 400)
            self.assertEqual(client.post("/api/roads/components", json=dict(fid=fid),
                                         headers={"Origin": "https://example.org"}).status_code, 403)
            lon, lat = road.to_lonlat.transform(0, 0)
            nearest = road.components(dict(lon=lon, lat=lat, categories=["E"], classes=["1"]))
            self.assertEqual(nearest["fid"], fid)
            empty = road.components(dict(fid=int(road.attributes["fid"][1])))
            self.assertIsNone(empty["components"][0]["share"])
            self.assertTrue(all(item["share"] == 0 for item in empty["components"][1:]))
            with self.assertRaisesRegex(ValueError, "Ingen veglenke"):
                road.components(dict(fid=999999))
            with self.assertRaisesRegex(ValueError, "Ugyldig kartposisjon"):
                road.components(dict(lon=float("nan"), lat=60))

    def test_500m_portable_stage_preserves_remote_road_cache_after_relocation(self):
        import hashlib
        import json
        import shutil
        from build_portable import prepare_roads

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "root"
            source = root / "data" / "ProssExport.gdb"
            source.parent.mkdir(parents=True)
            frame = gpd.GeoDataFrame(dict(
                linkid=["a"], vegnokkel=["E6"], roadcategory=["E"],
                kommunenavn=["A"], fylkesnr=["03"], istunnel=["0"], isferry=["0"],
                VS_Total=[50.], VS_Total_klasse=[5.]),
                geometry=[LineString([(0, 0), (100, 100)])], crs=25833)
            pyogrio.write_dataframe(frame, source, layer="Vegnett_VegScore_full", driver="OpenFileGDB")
            (source / "fingerprint.Z").write_bytes(b"uppercase")
            (source / "fingerprint.a").write_bytes(b"lowercase")
            stage = Path(folder) / "stage"
            with patch("build_portable.ROOT", root):
                manifest = prepare_roads(stage, cell_m=500)
            from remote_senorge.train_remote import sha256
            entries = [(path.name, sha256(path)) for path in source.iterdir()
                       if path.is_file() and not path.name.lower().endswith(".lock")]
            canonical = hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()
            windows = hashlib.sha256(json.dumps(sorted(entries, key=lambda entry: entry[0].lower())).encode()).hexdigest()
            self.assertNotEqual(canonical, windows)
            self.assertEqual(manifest["source_sha256"], canonical)
            (stage / "outputs" / "portable_weather.json").write_text(json.dumps(dict(roads=manifest)))
            relocated = Path(folder) / "returned"
            shutil.copytree(stage, relocated)
            with patch("build_portable.ROOT", root), \
                    patch("station_roads.pyogrio.write_dataframe", side_effect=AssertionError("Must preserve remote caches")):
                self.assertEqual(prepare_roads(relocated, cell_m=500), manifest)
                legacy = dict(manifest, source_sha256=windows)
                (relocated / "outputs" / "portable_weather.json").write_text(json.dumps(dict(roads=legacy)))
                self.assertEqual(prepare_roads(relocated, cell_m=500), manifest)
                (source / "fingerprint.Z").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Road inputs changed"):
                    prepare_roads(relocated, cell_m=500)
            original = RoadNetwork(stage / manifest["indexed_source"],
                cache_dir=stage / "outputs" / "station_app_cache" / "roads", portable_cache=True)
            returned = RoadNetwork(relocated / manifest["indexed_source"],
                cache_dir=relocated / "outputs" / "station_app_cache" / "roads", portable_cache=True)
            self.assertEqual(original.index_path.name, returned.index_path.name)
            self.assertEqual(original.grid_link_key(FakeEngine()), returned.grid_link_key(FakeEngine()))

    def test_old_500m_stage_is_filtered_for_packaging_without_removing_originals(self):
        import json
        import shutil
        from build_portable import copy_runtime_inputs, prepare_roads, replaced_road_input

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "root"
            source = root / "data" / "ProssExport.gdb"
            source.parent.mkdir(parents=True)
            frame = gpd.GeoDataFrame(dict(
                linkid=["a", "b"], vegnokkel=["E6", "K1"], roadcategory=["E", "K"],
                kommunenavn=["A"]*2, fylkesnr=["03"]*2, istunnel=["0"]*2, isferry=["0"]*2,
                VS_Total=[50.]*2, VS_Total_klasse=[5.]*2),
                geometry=[LineString([(0, 0), (100, 100)]), LineString([(200, 0), (300, 100)])], crs=25833)
            pyogrio.write_dataframe(frame, source, layer="Vegnett_VegScore_full", driver="OpenFileGDB")
            stage = Path(folder) / "stage"
            with patch("build_portable.ROOT", root):
                manifest = prepare_roads(stage, cell_m=500)
            raw = stage / "data" / "Vegnett_VegScore_full.gpkg"
            pyogrio.write_dataframe(frame, raw, layer="Vegnett_VegScore_full", driver="GPKG")
            shutil.copytree(source, stage / "data" / "ProssExport.gdb")
            old = dict(manifest, indexed_source=raw.relative_to(stage).as_posix(), features=2)
            (stage / "outputs" / "portable_weather.json").write_text(json.dumps(dict(roads=old)))
            with patch("build_portable.ROOT", root):
                updated = prepare_roads(stage, cell_m=500)
            self.assertEqual(updated["features"], 1)
            self.assertEqual(pyogrio.read_info(raw)["features"], 2)
            self.assertEqual(pyogrio.read_info(stage / updated["indexed_source"])["features"], 1)
            network = RoadNetwork(stage / updated["indexed_source"],
                                  cache_dir=stage / "outputs" / "station_app_cache" / "roads", portable_cache=True)
            network.prepare_nearest_distances(FakeEngine(), 2000)
            transient_lock = stage / "outputs" / "placement_direct_p95_work" / "precompute_frost.lock"
            transient_lock.parent.mkdir(parents=True, exist_ok=True)
            transient_lock.write_bytes(b"0")
            packaged = Path(folder) / "packaged"
            copy_runtime_inputs(stage, packaged)
            self.assertTrue(transient_lock.is_file())
            self.assertFalse((packaged / transient_lock.relative_to(stage)).exists())
            self.assertTrue((packaged / updated["indexed_source"]).is_file())
            copied = RoadNetwork(packaged / updated["indexed_source"],
                                 cache_dir=packaged / "outputs" / "station_app_cache" / "roads", portable_cache=True)
            self.assertIsNotNone(copied.load_nearest_distances(FakeEngine(), 500))
            self.assertFalse((packaged / "data" / "ProssExport.gdb").exists())
            self.assertFalse((packaged / "data" / raw.name).exists())
            self.assertTrue((stage / "data" / "ProssExport.gdb").exists())
            self.assertTrue(replaced_road_input(Path("data") / "ProssExport.gdb" / "file.gdbtable"))
            self.assertTrue(replaced_road_input(Path("outputs") / "station_app_cache" / "roads" / "old.gpkg"))
            self.assertFalse(replaced_road_input(updated["indexed_source"]))

    def test_county_hash_selects_attributes_and_clips_prepared_overview(self):
        from shapely.geometry import box
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.gpkg"
            frame = gpd.GeoDataFrame(dict(linkid=["a", "b", "c"], vegnokkel=["E6"]*3,
                roadcategory=["Europaveg"]*3, kommunenavn=["A"]*3, fylkesnr=["03", "11", "03"],
                istunnel=["0"]*3, isferry=["0"]*3, VS_Total=[10., 50., 20.],
                VS_Total_klasse=[1., 1., 1.], unused=["unused"]*3),
                geometry=[LineString([(x, -100), (x, 100)]) for x in [0, 1000, 2000]], crs=25833)
            pyogrio.write_dataframe(frame, path, layer="scores", driver="GPKG")
            road = RoadNetwork(path)
            road.prepare_display(FakeEngine())
            bounds = [road.geographic_bounds[0]-.01, road.geographic_bounds[1]-.01,
                      road.geographic_bounds[2]+.01, road.geographic_bounds[3]+.01]
            settings = dict(categories=["E"], classes=["1"], colour="score",
                            _county_key=("03",), _county_geometry=box(-100, 0, 2100, 200))
            np.testing.assert_array_equal(road.matching_rows(bounds, settings), [0, 2])
            self.assertNotIn("unused", road.read_fields)
            with patch.object(road, "read", side_effect=AssertionError("Prepared overview must avoid source reads")):
                result = road.prepared_overview(FakeEngine(), None, bounds, settings, road.matching_rows(bounds, settings))
            self.assertEqual(result["summary"]["links"], 2)
            self.assertAlmostEqual(result["summary"]["displayed_km"], .2)
            from shapely.geometry import shape
            self.assertTrue(all(not shape(feature["geometry"]).is_empty for feature in result["features"]))
            cached = RoadNetwork(path)
            np.testing.assert_array_equal(cached.county_rows["03"], [0, 2])
            self.assertEqual(len(cached.matching_rows(bounds, {**settings, "_county_key": ("03", "11")})), 3)
            self.assertEqual(len(cached.matching_rows(bounds, {**settings, "_county_key": ()})), 0)

    def test_score_classes_values_nulls_cache_and_corridor(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.gpkg"
            frame = gpd.GeoDataFrame(dict(linkid=["a", "b", "c"], vegnokkel=["E6", "R3", "F17"],
                roadcategory=["Europaveg", "Riksveg", "Fylkesveg"], kommunenavn=["A"]*3,
                istunnel=["0", "1", "0"], isferry=["0", "0", "1"],
                VS_Total=[10., 50., np.nan], VS_Total_klasse=[1., 5., np.nan]),
                geometry=[LineString([(x, -100), (x, 100)]) for x in [0, 1000, 2000]], crs=25833)
            pyogrio.write_dataframe(frame, path, layer="scores", driver="GPKG")
            road = RoadNetwork(path)
            bounds = list(road.geographic_bounds)
            bounds = [bounds[0]-.01, bounds[1]-.01, bounds[2]+.01, bounds[3]+.01]
            settings = dict(categories=["E", "R", "F"], classes=["1", "5", "U"], colour="score")
            self.assertTrue(road.metadata["score_network"])
            self.assertEqual(road.metadata["classes"], ["1", "5", "U"])
            self.assertEqual(road.metadata["class_ranges"]["5"], [50, 50])
            roads, _ = road.read(bounds, settings)
            self.assertEqual(len(roads), 3)
            self.assertFalse(roads.iloc[0].istunnel)
            self.assertTrue(roads.loc[roads.edge_id == "b", "istunnel"].iloc[0])
            result = road.view(FakeEngine(), np.array([3., 4., 5.]), None, .01, bounds, settings, zoom=12)
            self.assertEqual({f["properties"]["VS_Total"] for f in result["features"]}, {10., 50., None})
            selected = {**settings, "classes": ["5"]}
            mask = road.corridor_mask(FakeEngine(), bounds, selected, 0)
            np.testing.assert_array_equal(mask, [False, True, False])
            road.prepare_display(FakeEngine())
            cached = RoadNetwork(path)
            np.testing.assert_allclose(cached.attributes["vs_total"], road.attributes["vs_total"], equal_nan=True)
            rows = cached.matching_rows(bounds, settings)
            overview = cached.prepared_overview(FakeEngine(), None, bounds, settings, rows)
            self.assertEqual(overview["summary"]["links"], 3)
            self.assertEqual({f["properties"]["VS_Total_klasse"] for f in overview["features"]}, {1, 5, None})
            self.assertTrue(all(f["properties"]["mean_error"] is None for f in overview["features"]))
            self.assertEqual(len(cached.read(bounds, {**settings, "classes": []})[0]), 0)
            from shapely.geometry import box
            county_settings = {**settings, "_county_key": ("test",),
                               "_county_geometry": box(950, 0, 1050, 200)}
            county_roads, _ = cached.read(bounds, county_settings)
            self.assertEqual(len(county_roads), 1)
            self.assertAlmostEqual(county_roads.geometry.length.iloc[0], 100)
            result = cached.view(FakeEngine(), np.array([3., 4., 5.]), None, .01,
                                 bounds, county_settings, zoom=12)
            self.assertEqual(result["summary"]["links"], 1)
            self.assertAlmostEqual(result["summary"]["displayed_km"], .1)
            # The unmasked cache must not be polluted by the county selection.
            self.assertEqual(len(cached.read(bounds, settings)[0]), 3)


if __name__ == "__main__":
    unittest.main()
