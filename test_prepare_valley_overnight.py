import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

from prepare_valley_overnight import prepare
from remote_senorge.train_valley_neural import load_data


class PackageTests(unittest.TestCase):
    def test_allowlist_verified_portable_shape_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            data, output = Path(temp) / "data", Path(temp) / "portable"
            data.mkdir()
            for name in ("config.json", "terrain.npz", "stations.csv",
                         "regimes_progress.json", "observations_progress.json"):
                (data / name).write_text("synthetic input")
            from remote_senorge.test_valley_neural import NeuralTests
            arrays, _ = NeuralTests().fixture(Path(temp) / "fixture")
            # Exporter requires30validation nights; extend independent selection/calibration days.
            samples = []
            matrix = []
            for split, year, nights in ((0, 2022, 12), (1, 2023, 30), (2, 2024, 12)):
                for stamp in pd.date_range(f"{year}-12-01", periods=nights, tz="UTC"):
                    samples.append(dict(
                        source="SOURCE", target=("TRAIN", "VALIDATION", "TEST")[split],
                        split=split, source_c=-2., target_c=-3., forecast_delta_c=-.5,
                        calm_clear=False, night=stamp.strftime("%Y-%m-%d"),
                        time=(stamp+pd.Timedelta(hours=3)).isoformat()))
                    matrix.append(arrays["x"][0])
            samples = pd.DataFrame(samples)
            x = np.array(matrix)
            features = dict(graph_regime=x, terrain=x[:, :16], regime=x[:, :28])
            names = {name: [f"feature{i}" for i in range(a.shape[1])] for name, a in features.items()}
            raw = pd.DataFrame(dict(accepted=[True], rejection=[None]))
            inputs = (dict(bbox=[5, 60, 7, 62], groups="synthetic", winters=[2022, 2023, 2024]),
                      samples, features, np.full(len(samples), -1., np.float32), names,
                      pd.DataFrame(dict(station_id=["SOURCE"], sensor_id=[0])), raw)
            with patch("prepare_valley_overnight.load_prepared_samples", return_value=inputs), patch("builtins.print"):
                zipped = prepare(data, output)
            with ZipFile(zipped) as archive:
                self.assertEqual(set(archive.namelist()), {
                    "training.npz", "manifest.json", "train_valley_neural.py", "test_valley_neural.py",
                    "README.md", "package_checksums.json", "requirements.txt"})
                self.assertNotIn("pipeline_config", "\n".join(archive.namelist()))
                extracted = Path(temp) / "extracted"
                archive.extractall(extracted)
                restored, manifest = load_data(extracted)
                np.testing.assert_array_equal(restored["x"], x)
                self.assertEqual(manifest["rows"]["selection"], 15)
                self.assertEqual(manifest["rows"]["calibration"], 15)
                receipt = json.loads((output / "package_receipt.json").read_text())
                self.assertGreater(receipt["bytes"], 0)
            with self.assertRaises(FileExistsError):
                prepare(data, output)


if __name__ == "__main__":
    unittest.main()
