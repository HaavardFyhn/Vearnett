from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import train_senorge_grid_pairs
import numpy as np
from flask import Flask
from pyproj import Transformer
from rasterio.io import MemoryFile

from remote_senorge.footprint_atlas import AtlasDataset
from remote_senorge.footprint_model import TARGETS, create_model
from remote_senorge import test_footprint_atlas as atlas_tests
from remote_senorge.train_remote import sha256, write_json
from station_footprint import FootprintBackend, raster_png, register_footprint_view


class StationFootprintTests(unittest.TestCase):
    def test_heatmap_blue_to_red_and_nodata_for_continuous_and_binary_scales(self):
        import matplotlib
        from PIL import Image
        description = dict(web_bounds=[0, 0, 3000, 750])
        for maximum in (3., 1.):
            values = np.array([[0, maximum/2, maximum, -9999]], np.float32)
            with patch("rasterio.warp.reproject",
                       side_effect=lambda source, destination, **kwargs: np.copyto(destination, values)):
                image = np.asarray(Image.open(raster_png(values, None, description, maximum)))
            expected = (matplotlib.colormaps["RdBu_r"](np.array([0., .5, 1.]))*255).astype(np.uint8)
            np.testing.assert_array_equal(image[0, :3, :3], expected[:, :3])
            self.assertGreater(image[0, 0, 2], image[0, 0, 0])
            self.assertGreater(image[0, 2, 0], image[0, 2, 2])
            np.testing.assert_array_equal(image[0, :, 3], [220, 220, 220, 0])

    def fixture(self, root):
        import torch
        prepared = root / "outputs" / "senorge_footprint_expanded"
        prepared.mkdir(parents=True)
        atlas_tests.AtlasTests().fixture(prepared)
        folder = root / "outputs" / "senorge_footprint_station_model"
        folder.mkdir()
        torch.manual_seed(42)
        model = create_model(torch, 8, True, 64, 11, 31).eval()
        torch.save(model.state_dict(), folder / "selected_model.pt")
        np.savez(folder / "normalization.npz", mean=np.zeros(31, np.float32), std=np.full(31, 1000, np.float32))
        (folder / "footprint_manifest.json").write_bytes((prepared / "footprint_manifest.json").read_bytes())
        write_json(folder / "summary.json", dict(status="complete"))
        scores = {target: dict(rmse=.2, bias=.05) for target in TARGETS}
        write_json(folder / "metadata.json", dict(
            architecture="reference-resunet-spatial10-absolute-v2",
            manifest_sha256=sha256(prepared / "footprint_manifest.json"),
            selected=dict(width=8, name="synthetic", fine=True), test=scores, baseline_test=scores,
            checksums={name: sha256(folder / name) for name in
                       ("selected_model.pt", "normalization.npz", "footprint_manifest.json", "summary.json")}))
        return FootprintBackend(root)

    def test_reference_input_parity_all_six_targets_and_cache(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            backend = self.fixture(Path(temp))
            try:
                atlas = backend.atlas
                arrays = atlas.arrays()
                with torch.inference_mode():
                    direct = backend.model(torch.from_numpy(arrays["scenes"][0][None]),
                                           torch.from_numpy(arrays["local10"][0][None]),
                                           torch.from_numpy((arrays["features"][0]/1000)[None])).numpy()[0]
                direct[:, 32, 32] = 0
                values, mapping = backend.predict_reference(0)
                np.testing.assert_allclose(values, direct, atol=1e-7)
                np.testing.assert_array_equal(mapping, atlas.refs["mapping"][0])
                self.assertIs(values, backend.predict_reference(0)[0])
                self.assertFalse(values.flags.writeable)
                self.assertEqual(backend.reference_at(75400, 6675100), 0)
                with self.assertRaisesRegex(ValueError, "utenfor"):
                    backend.reference_at(99999999, 0)
                for target in TARGETS:
                    result = backend.describe(0, target)
                    self.assertEqual(result["cell_m"], 1000)
                    self.assertEqual(result["shape"], [64, 64])
                    self.assertEqual(result["valid_cells"], 2)
                    self.assertEqual(backend.point(0, target, 75000, 6675000)["value_c"], 0)
                    self.assertTrue(backend.point(0, target, 76000, 6675000)["supported"])
                    self.assertFalse(backend.point(0, target, 77000, 6675000)["supported"])
                    outside = backend.point(0, target, 108000, 6675000)
                    self.assertFalse(outside["supported"])
                    self.assertIsNone(outside["value_c"])
            finally:
                backend.close()

    def test_exact_native_geotiff_and_transparent_png(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            backend = self.fixture(Path(temp))
            try:
                png = backend.png(0, "winter_tn")
                image = np.asarray(Image.open(png))
                self.assertTrue((image[:, :, 3] == 0).any())
                self.assertTrue((image[:, :, 3] > 0).any())
                with ZipFile(backend.export(0, "winter_tn")) as archive:
                    self.assertEqual(set(archive.namelist()), {"footprint_winter_tn_1000m.tif", "footprint.json"})
                    metadata = json.loads(archive.read("footprint.json"))
                    self.assertEqual(metadata["cell_m"], 1000)
                    with MemoryFile(archive.read("footprint_winter_tn_1000m.tif")) as memory:
                        with memory.open() as src:
                            self.assertEqual(src.shape, (64, 64))
                            self.assertEqual(src.res, (1000, 1000))
                            self.assertEqual(src.crs.to_epsg(), 25833)
                            self.assertEqual(src.nodata, -9999)
                            self.assertEqual(src.xy(32, 32), (75000, 6675000))
                            self.assertEqual(src.xy(31, 33), (76000, 6676000))
                            data = src.read(1)
                            self.assertEqual(data[32, 32], 0)
                            self.assertEqual(data[0, 0], -9999)
                            self.assertAlmostEqual(float(data[32, 33]),
                                                   backend.predict_reference(0)[0][3, 32, 33], places=7)
                            self.assertEqual(src.tags()["model_sha256"], metadata["model_sha256"])
            finally:
                backend.close()

    def test_api_station_selection_custom_point_targets_errors_and_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backend = self.fixture(root)
            try:
                stations = [dict(station_id="SN1", name="fixture", supported=True, reference=0),
                            dict(station_id="SN2", name="outside", supported=False, reference=None)]
                app = Flask(__name__)
                register_footprint_view(app, root, backend, stations)
                client = app.test_client()
                self.assertTrue(client.get("/api/footprint/available").get_json()["available"])
                options = client.get("/api/footprint/options").get_json()
                self.assertEqual(options["cell_m"], 1000)
                self.assertEqual([target["id"] for target in options["targets"]], TARGETS)
                for target in TARGETS:
                    result = client.get(f"/api/footprint?station=SN1&target={target}&cell_m=1000")
                    self.assertEqual(result.status_code, 200)
                    self.assertEqual(result.get_json()["target"], target)
                for query in ("station=SN2", "station=UNKNOWN", "reference=-1", "reference=6",
                              "station=SN1&target=bad", "station=SN1&cell_m=5000",
                              "lon=nan&lat=60", "lon=190&lat=60"):
                    result = client.get(f"/api/footprint?{query}")
                    self.assertEqual(result.status_code, 400, query)
                    self.assertIn("error", result.get_json())
                lon, lat = Transformer.from_crs(25833, 4326, always_xy=True).transform(75000, 6675000)
                result = client.get(f"/api/footprint?lon={lon}&lat={lat}").get_json()
                self.assertEqual(result["reference"], 0)
                result = client.get(f"/api/footprint/point?reference=0&point_lon={lon}&point_lat={lat}").get_json()
                self.assertEqual(result["value_c"], 0)
                response = client.get("/api/footprint.png?reference=0&target=summer_tg")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["X-Cell-Size-M"], "1000")
                response = client.get("/api/footprint/export?reference=0&target=summer_tg")
                with ZipFile(BytesIO(response.data)) as archive:
                    self.assertIn("footprint_summer_tg_1000m.tif", archive.namelist())
            finally:
                backend.close()

    def test_missing_optional_install_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Flask(__name__)
            register_footprint_view(app, Path(temp))
            client = app.test_client()
            self.assertFalse(client.get("/api/footprint/available").get_json()["available"])
            response = client.get("/api/footprint/options")
            self.assertEqual(response.status_code, 503)
            self.assertIn("error", response.get_json())

    def test_corrupt_returned_model_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backend = self.fixture(root)
            backend.close()
            with (root / "outputs" / "senorge_footprint_station_model" / "selected_model.pt").open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                FootprintBackend(root)


if __name__ == "__main__":
    unittest.main()
