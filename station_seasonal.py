"""CPU seasonal CNN inference with shared, cached terrain endpoint encodings."""
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from remote_senorge.train_remote import sha256
from remote_senorge.train_seasonal_terrain import create_model
from station_senorge import BasinLookup, SeNorgePairModel


TARGET_LABELS = {
    "winter_tg": "Vinter · døgnmiddel",
    "summer_tg": "Sommer · døgnmiddel",
    "winter_tn": "Vinter · døgnminimum",
    "summer_tn": "Sommer · døgnminimum",
}
WARNING = (
    "Eksperimentell terreng-CNN 1996–2025: DJF-vinter / JJA-sommer. "
    "Kartet viser anslått representasjonsfeil i seNorge, ikke temperatur eller "
    "nøyaktighet mot uavhengige målestasjoner. Terreng: 50/100/500 m, knyttet til "
    "nærmeste 1 km-rutesenter; samme rute får feil 0. "
    "Trent på 1–100 km; større avstander ekstrapoleres uten dokumentert nøyaktighet. "
    "Romlig test med 16,5 km buffer; få eller ingen testpar ved store avstander. "
    "Valgt etter resultat på samme testsett, uten en ekstra uavhengig modellvalgstest. "
    "Sommermiddel har underestimering på ca. 0,09 °C i testen."
    " Utenfor 1 km-landrutenettet brukes terrengvindu ved avrundet rutesenter "
    "med endepunktets høyde som referanse; dette er utenfor treningsstøtten."
)


class SeasonalBackend:
    def __init__(self, root, folder):
        import torch
        self.torch = torch
        torch.set_num_threads(4)
        self.lock = threading.RLock()
        self.root, self.folder = Path(root), Path(folder)
        self.metadata = json.loads((self.folder / "metadata.json").read_text())
        for name, field in (("combined.pt", "model_sha256"),
                            ("combined_normalization.npz", "normalization_sha256"),
                            ("seasonal_manifest.json", "manifest_sha256")):
            if sha256(self.folder / name) != self.metadata[field]:
                raise ValueError(f"Seasonal model checksum mismatch: {name}")
        if self.metadata["architecture"] != "symmetric-multiscale-cnn-v1":
            raise ValueError("Unsupported seasonal model architecture")
        self.model = create_model("combined", torch)
        self.model.load_state_dict(torch.load(self.folder / "combined.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        with np.load(self.folder / "combined_normalization.npz", allow_pickle=False) as data:
            self.mean, self.std = data["mean"].copy(), data["std"].copy()
        if (self.mean.shape != (25,) or self.std.shape != (25,) or
                not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any()):
            raise ValueError("Invalid seasonal normalization")
        self.prepared = self.root / "outputs" / "senorge_seasonal_terrain"
        manifest = json.loads((self.folder / "seasonal_manifest.json").read_text())
        if sha256(self.prepared / "seasonal_manifest.json") != self.metadata["manifest_sha256"]:
            raise ValueError("Local terrain does not belong to returned model")
        self.manifest = manifest
        for name in ("terrain.npy", "terrain_valid.npy"):
            if sha256(self.prepared / name) != manifest["checksums"][name]:
                raise ValueError(f"Local terrain checksum mismatch: {name}")
        grid_path = self.root / "outputs" / "senorge_30yr_station_model" / "grid_features.csv.gz"
        frame = pd.read_csv(grid_path, float_precision="round_trip")
        with np.load(self.prepared / "seasonal_training.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(frame[["x", "y"]].to_numpy(), data["xy"])
        keys = BasinLookup.cell_keys(frame.x.to_numpy(), frame.y.to_numpy())
        self.cells = {tuple(key): i for i, key in enumerate(keys)}
        self.height = np.load(self.prepared / "terrain.npy", mmap_mode="r", allow_pickle=False)
        self.valid = np.load(self.prepared / "terrain_valid.npy", mmap_mode="r", allow_pickle=False)
        if self.height.shape != (len(frame), 3, 64, 64) or self.valid.shape != self.height.shape:
            raise ValueError("Invalid local terrain arrays")
        self.embeddings = {}
        self.fallback = None

    def close(self):
        self.height._mmap.close()
        self.valid._mmap.close()

    def terrain_input(self, keys):
        values = np.empty((len(keys), 3, 2, 64, 64), dtype=np.float32)
        for i, key in enumerate(keys):
            index = self.cells.get(tuple(key[:2]))
            if index is not None:
                values[i, :, 0] = self.height[index]
                values[i, :, 1] = self.valid[index]
            else:
                self.missing_patch(values[i], key)
        if not np.isfinite(values).all() or not np.isin(values[:, :, 1], [0, 1]).all():
            raise ValueError("Invalid seasonal terrain values/mask")
        return values

    def missing_patch(self, output, key):
        import pyogrio
        import rasterio
        from terrain_patches import patch
        if self.fallback is None:
            frame = pyogrio.read_dataframe(
                self.root / "outputs" / "norway_grid_1000m.gpkg",
                columns=["x", "y", "elevation"], read_geometry=False)
            self.fallback = {tuple(k): float(h) for k, h in zip(
                BasinLookup.cell_keys(frame.x.to_numpy(), frame.y.to_numpy()), frame.elevation)}
        elevation = self.fallback.get(tuple(key[:2]))
        if elevation is None and len(key) == 3:
            elevation = key[2]
        if elevation is None or not np.isfinite(elevation):
            raise ValueError(f"No 1 km terrain cell for seasonal inference: {tuple(key)}")
        cache = self.prepared / "cache"
        for scale, (resolution, name) in enumerate(((50, "terrain.vrt"), (100, "dem_100m.tif"),
                                                  (500, "dem_500m.tif"))):
            with rasterio.open(cache / name) as source:
                h, mask = patch(source, key[0]*1000, key[1]*1000, resolution, elevation)
                output[scale, 0], output[scale, 1] = h, mask

    def encode(self, x, y, elevation=None):
        keys = BasinLookup.cell_keys(x, y)
        if elevation is not None:
            keys = [(*key, 0. if tuple(key) in self.cells else float(h))
                    for key, h in zip(keys, elevation)]
        with self.lock, self.torch.inference_mode():
            missing = sorted({tuple(key) for key in keys if tuple(key) not in self.embeddings})
            for start in range(0, len(missing), 128):
                batch = missing[start:start+128]
                patches = self.torch.from_numpy(self.terrain_input(batch))
                encoded = self.model.encode(patches).numpy()
                if not np.isfinite(encoded).all():
                    raise ValueError("Invalid seasonal terrain embedding")
                self.embeddings.update(zip(batch, encoded.copy()))
            return np.array([self.embeddings[tuple(key)] for key in keys], dtype=np.float32)

    def predict(self, x, coordinates, elevations=None):
        sx, sy, tx, ty = coordinates
        source_h, target_h = (None, None) if elevations is None else elevations
        a, b = self.encode(sx, sy, source_h), self.encode(tx, ty, target_h)
        with self.lock, self.torch.inference_mode():
            result = []
            for start in range(0, len(x), 8192):
                stop = start+8192
                inputs = np.concatenate([a[start:stop]+b[start:stop],
                                         np.abs(a[start:stop]-b[start:stop]),
                                         (x[start:stop]-self.mean)/self.std], axis=1)
                result.append(self.model.head(self.torch.from_numpy(inputs)).numpy())
            predicted = np.concatenate(result)
        if not np.isfinite(predicted).all() or (predicted < 0).any():
            raise ValueError("Invalid seasonal predictions")
        return predicted


class SeasonalPairModel(SeNorgePairModel):
    def __init__(self, backend, basins, target):
        self.backend, self.basins = backend, basins
        self.target = backend.metadata["targets"].index(target)

    def predict_pairs(self, source, target, source_ids, target_ids):
        x, identical, coordinates = self.pair_inputs(source, target, source_ids, target_ids)
        if len(x) == 0:
            return np.empty(0, dtype=np.float32)
        source_h = source.get("senorge_elevation", source["elevation"])[source_ids]
        target_h = target.get("senorge_elevation", target["elevation"])[target_ids]
        result = self.backend.predict(x, coordinates, (source_h, target_h))[:, self.target].copy()
        result[identical] = 0
        return result
