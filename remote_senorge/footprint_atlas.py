"""Shared-atlas map construction; portable NumPy only, no GIS dependency."""
import json
from pathlib import Path

import numpy as np

if __package__:
    from .train_remote import sha256
else:
    from train_remote import sha256


SPATIAL_FIELDS = [
    "valid_fraction", "elevation_std_km", "relief_km", "mean_slope",
    "mean_abs_east_gradient", "mean_abs_north_gradient",
    "nw_minus_mean_km", "ne_minus_mean_km", "sw_minus_mean_km", "se_minus_mean_km",
    "low_fraction_10m_below_mean",
]
FEATURE_FIELDS = [
    "x", "y", "elevation", "coast_dist_fjord", "coast_dist_ocean", "objtype_code",
    "relief_10km", "rugstd_10km", "tpi_10km", "relief_25km", "rugstd_25km", "tpi_25km",
    "basin_present", "basin_count", "basin_max_depth_m", "basin_volume_m3",
]
ABSOLUTE_FIELDS = [
    "reference_elevation_m", "target_elevation_m",
    "reference_coast_ocean_distance_m", "target_coast_ocean_distance_m",
    "reference_coast_fjord_distance_m", "target_coast_fjord_distance_m",
]


def pair_features(table, source, targets, absolute=False):
    a, b = table[source], table[targets]
    dx, dy = np.abs(a[0]-b[:, 0]), np.abs(a[1]-b[:, 1])
    columns = [np.hypot(dx, dy), np.abs(a[2]-b[:, 2]), np.abs(a[3]-b[:, 3]),
               np.abs(a[4]-b[:, 4]), (a[5] == b[:, 5]).astype(float), dx, dy]
    columns.extend(np.abs(a[i]-b[:, i]) for i in range(6, 12))
    for i in range(12, 16):
        columns.extend([np.minimum(a[i], b[:, i]), np.maximum(a[i], b[:, i]), np.abs(a[i]-b[:, i])])
    if absolute:
        for i in (2, 4, 3):
            columns.extend([np.full(len(b), a[i]), b[:, i]])
    return np.column_stack(columns).astype(np.float32)


class SceneArray:
    """Lazily reconstruct one scene; atlas files remain shared between references."""
    def __init__(self, dataset, kind):
        self.dataset, self.kind = dataset, kind

    def __getitem__(self, index):
        d = self.dataset
        ref = int(d.refs["refs"][index])
        return self.for_reference(ref, d.refs["mapping"][index] if self.kind == "features" else None)

    def for_reference(self, ref, mapping=None):
        d = self.dataset
        cx, cy = d.table[ref, :2]
        x0, y0 = d.manifest["atlas_origin"]
        row, col = int(round((y0-cy)/1000)), int(round((cx-x0)/1000))
        if self.kind == "scenes":
            # Atlas edge is half a cell NW of its first 1km centre.
            r, c = (row-40)*10, (col-40)*10
            height = d.height[r:r+800, c:c+800].astype(np.float32)
            valid = d.valid[r:r+800, c:c+800].astype(np.float32)
            if height.shape != (800, 800):
                raise ValueError("Reference scene outside terrain atlas")
            return np.stack([np.where(valid, height-d.table[ref, 2]/1000, 0), valid])
        if self.kind == "local10":
            result = d.spatial[:, row-40:row+40, col-40:col+40].astype(np.float32)
            if result.shape != (len(SPATIAL_FIELDS), 80, 80):
                raise ValueError("Fine terrain scene outside descriptor atlas")
            return result
        if self.kind == "features":
            if mapping is None or mapping.shape != (64, 64):
                raise ValueError("Expected64x64 output cell mapping")
            mask = mapping >= 0
            result = np.zeros((d.feature_count, 64, 64), np.float32)
            result[:, mask] = pair_features(d.table, ref, mapping[mask], d.absolute).T
            return result
        raise ValueError(f"Unknown scene kind: {self.kind}")


class AtlasDataset:
    def __init__(self, folder, verify=True):
        folder = Path(folder)
        self.manifest = json.loads((folder / "footprint_manifest.json").read_text())
        if self.manifest.get("schema") != 2 or self.manifest.get("spatial_fields") != SPATIAL_FIELDS:
            raise ValueError("Invalid spatial footprint schema")
        fields = self.manifest.get("absolute_features", [])
        if fields not in ([], ABSOLUTE_FIELDS):
            raise ValueError("Invalid absolute endpoint feature schema")
        self.absolute = bool(fields)
        self.feature_count = 25+len(fields)
        if "features" in self.manifest and (
                len(self.manifest["features"]) != self.feature_count or
                (self.absolute and self.manifest["features"][-len(fields):] != fields)):
            raise ValueError("Feature names do not match absolute endpoint channels")
        if verify:
            for name, digest in self.manifest["checksums"].items():
                if Path(name).name != name or sha256(folder / name) != digest:
                    raise ValueError(f"Expanded dataset checksum mismatch: {name}")
        self.maps = []
        try:
            with np.load(folder / "references.npz", allow_pickle=False) as stored:
                self.refs = {key: stored[key] for key in stored.files}
            self.table = self.open(folder / "grid_table.npy")
            self.height = self.open(folder / "terrain_height.npy")
            self.valid = self.open(folder / "terrain_valid.npy")
            self.spatial = self.open(folder / "spatial10.npy")
            self.labels = self.open(folder / "labels.npy")
            self.mask = self.open(folder / "label_mask.npy")
            self.validate()
        except Exception:
            self.close()
            raise

    def open(self, path):
        result = np.load(path, mmap_mode="r", allow_pickle=False)
        self.maps.append(result)
        return result

    def validate(self):
        refs, manifest = self.refs, self.manifest
        n = len(refs["refs"])
        if (refs["split"].shape != (n,) or not np.isin(refs["split"], [0, 1, 2]).all() or
                not np.issubdtype(refs["refs"].dtype, np.integer) or
                len(np.unique(refs["refs"])) != n or refs["refs"].min() < 0 or
                refs["refs"].max() >= len(self.table)):
            raise ValueError("Invalid or duplicate reference locations")
        if self.table.shape[1:] != (16,) or not np.isfinite(self.table).all():
            raise ValueError("Invalid endpoint feature table")
        if refs["mapping"].shape != (n, 64, 64) or not np.issubdtype(refs["mapping"].dtype, np.integer):
            raise ValueError("Invalid output cell map")
        if refs["mapping"].min() < -1 or refs["mapping"].max() >= len(self.table):
            raise ValueError("Output cell outside table")
        if self.labels.shape != (n, 6, 64, 64) or self.mask.shape != self.labels.shape or self.mask.dtype != bool:
            raise ValueError("Invalid six-target map shapes")
        if self.height.shape != self.valid.shape or self.valid.dtype != np.uint8:
            raise ValueError("Invalid terrain atlas")
        if self.spatial.shape != (len(SPATIAL_FIELDS), self.height.shape[0]//10, self.height.shape[1]//10):
            raise ValueError("Invalid spatial terrain channel count")
        for start in range(0, self.height.shape[0], 256):
            if not np.isfinite(self.height[start:start+256]).all() or not np.isin(self.valid[start:start+256], [0, 1]).all():
                raise ValueError("Invalid terrain heights/coverage")
        if not np.isfinite(self.spatial).all() or ((self.spatial[0] < 0) | (self.spatial[0] > 1)).any():
            raise ValueError("Invalid spatial terrain descriptors/coverage")
        xy = self.table[refs["refs"], :2]
        if not np.array_equal(xy, refs["xy"]):
            raise ValueError("Reference coordinates differ from grid table")
        if not np.allclose(self.table[:, :2], np.round(self.table[:, :2]/1000)*1000):
            raise ValueError("Expected integer-kilometre terrain centres")
        x0, y0 = manifest["atlas_origin"]
        rows = np.round((y0-xy[:, 1])/1000).astype(int)
        cols = np.round((xy[:, 0]-x0)/1000).astype(int)
        if ((rows < 40) | (cols < 40) | (rows+40 > self.spatial.shape[1]) |
                (cols+40 > self.spatial.shape[2])).any():
            raise ValueError("Input scene exceeds atlas")
        if not np.array_equal(refs["mapping"][:, 32, 32], refs["refs"]):
            raise ValueError("Reference must be output pixel32,32")
        rr, cc = np.indices((64, 64))
        delta = np.stack([(cc-32)*1000, (32-rr)*1000], axis=-1)
        size = manifest["block_size_m"]
        if size < 82000:
            raise ValueError("Spatial blocks smaller than input support")
        pos = xy % size
        if ((pos < 41000) | (pos > size-41000)).any():
            raise ValueError("Terrain support crosses a partition block")
        memberships = []
        for part in range(3):
            ids = np.flatnonzero(refs["split"] == part)
            if len(ids) < 2:
                raise ValueError("Empty partition")
            memberships.append(set(map(tuple, np.floor(xy[ids]/size).astype(int))))
            count = np.zeros(6, np.int64)
            for i in ids:
                if not np.isfinite(self.labels[i]).all() or (self.labels[i][self.mask[i]] < 0).any():
                    raise ValueError("Invalid labels")
                if (self.mask[i][:, refs["mapping"][i] < 0]).any():
                    raise ValueError("Labels on missing output cells")
                keep = refs["mapping"][i] >= 0
                if not np.array_equal(self.table[refs["mapping"][i][keep], :2], (xy[i]+delta)[keep]):
                    raise ValueError("Output cells have wrong location/orientation")
                count += self.mask[i].sum(axis=(1, 2))
            if (count == 0).any():
                raise ValueError("No labels for one or more targets in partition")
        if any(memberships[a] & memberships[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Geographic partition leakage")

    def arrays(self):
        return dict(scenes=SceneArray(self, "scenes"), local10=SceneArray(self, "local10"),
                    features=SceneArray(self, "features"), labels=self.labels, label_mask=self.mask)

    def close(self):
        for value in self.maps:
            value._mmap.close()
        self.maps.clear()
