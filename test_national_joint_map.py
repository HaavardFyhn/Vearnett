from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.io import MemoryFile
from types import SimpleNamespace

from national_joint_map import NationalJointMap, connections
from station_temperature_map import map_geometry
from national_temperature_features import temperature_features
from remote_senorge.train_national_transfer import fusion_model


class NationalJointMapTests(unittest.TestCase):
    def test_graph_matches_existing_feature_geometry_and_rejects_disconnected(self):
        origin = np.array([200000., 6800000.])
        rows, cols = np.indices((80, 80))
        height = (rows*2+cols).astype(np.float32)
        valid = np.ones_like(height, bool)
        valid[:, 42] = False
        table = np.zeros((height.size, 16))
        table[:, :2] = np.column_stack([origin[0]+cols.ravel()*1000, origin[1]-rows.ravel()*1000])
        table[:, 2] = height.ravel()
        station = table[[40*80+40]]
        positions = np.array([[40, 40]])
        result = connections(table, station, positions, height, valid, origin, lambda _:None)
        expected, _, graph, _ = map_geometry(table, positions, height, valid, origin, 0)
        lookup = dict(zip(result["cell"], result["graph"][:, 0]))
        for cell, features in zip(expected, graph):
            np.testing.assert_array_equal(lookup[cell], features)
        self.assertTrue((cols.ravel()[result["cell"]] < 42).all())
        self.assertTrue((result["source"][:, 0] == 0).all())
        self.assertTrue((result["source"][:, 1:] == -1).all())

    def test_shortlist_and_six_source_limit_and_100km_threshold(self):
        height = np.zeros((250,250),np.float32)
        valid = np.ones_like(height,bool)
        origin = np.array([200000.,6800000.])
        station_positions = np.array([[125,125+i] for i in range(14)])
        stations = np.zeros((14,16))
        stations[:,:2] = origin+station_positions[:,::-1]*[1000,-1000]
        table = np.zeros((3,16))
        table[:,:2] = origin+np.array([[120,125],[25,125],[24,125]])*[1000,-1000]
        links = connections(table,stations,station_positions,height,valid,origin,lambda _:None)
        self.assertIn(0,links["cell"])
        self.assertIn(1,links["cell"])
        self.assertNotIn(2,links["cell"])
        first = np.flatnonzero(links["cell"] == 0)[0]
        np.testing.assert_array_equal(links["source"][first],[0,1,2,3,4,5])
        self.assertLessEqual((links["source"]>=0).sum(1).max(),6)

    def backend(self):
        backend=NationalJointMap.__new__(NationalJointMap)
        table=np.zeros((4,16))
        table[:,:2]=[[200000,6800000],[201000,6800000],[200000,6799000],[201000,6799000]]
        stations=pd.DataFrame(dict(station_id=["A","B"],name=["Station A","Station B"],
                                  lon=[6.,6.1],lat=[61.,61.1],group=["train","test"]))
        backend.base=SimpleNamespace(table=table,origin=np.array([200000.,6800000.]),
            height=np.zeros((2,2)),stations=stations,
            to_lonlat=Transformer.from_crs(25833,4326,always_xy=True))
        backend.context=dict(cell=np.array([0,3]),source=np.array([[0,1,-1,-1,-1,-1],[0,-1,-1,-1,-1,-1]]),
            temperature=np.array([-5.,-10.]),radius=np.array([4.,6.]),
            removed_radius=np.array([[1.,-2.,np.nan,np.nan,np.nan,np.nan],[np.nan]*6]),
            removed_temperature=np.array([[.5,-1.,np.nan,np.nan,np.nan,np.nan],[np.nan]*6]),
            station_ids=np.array([0,1]),temperatures=np.array([-2.,-3.]),
            metadata=dict(uncertainty_map_validated=False,time="2024-12-15T03:00:00+00:00"))
        return backend

    def test_signed_cell_contributions_and_sole_source_nodata(self):
        backend=self.backend()
        result=backend.detail(0)
        self.assertEqual([s["radius_change_c"] for s in result["sources"]],[1.,-2.])
        self.assertTrue(result["sources"][1]["extrapolation"])
        self.assertEqual(result["lower_c"],-9)
        result=backend.detail(3)
        self.assertIsNone(result["sources"][0]["radius_change_c"])
        self.assertFalse(result["sources"][0]["deletion_prediction_available"])
        with self.assertRaises(ValueError):backend.detail(2)

    def test_exact1km_geojson_and_national_geotiff_nodata(self):
        backend=self.backend()
        geo=backend.geojson()
        self.assertFalse(geo["overview"])
        self.assertEqual(len(geo["features"]),2)
        ring=np.array(geo["features"][0]["geometry"]["coordinates"][0])
        x,y=Transformer.from_crs(4326,25833,always_xy=True).transform(ring[:,0],ring[:,1])
        np.testing.assert_allclose(x,[199500,200500,200500,199500,199500],atol=.001)
        np.testing.assert_allclose(y,[6799500,6799500,6800500,6800500,6799500],atol=.001)
        with MemoryFile(backend.geotiff()) as memory,memory.open() as raster:
            self.assertEqual(raster.count,5)
            self.assertEqual(raster.res,(1000.,1000.))
            self.assertEqual(raster.crs.to_epsg(),25833)
            self.assertEqual(raster.read(1,masked=True).count(),2)
            self.assertEqual(raster.read(2)[0,0],4)
            self.assertEqual(raster.read(3)[1,1],-16)
            self.assertEqual(raster.read(5)[0,0],2)
            self.assertEqual(raster.read(5)[1,1],1)
            self.assertEqual(raster.read(1)[0,1],-9999)
        with self.assertRaises(ValueError):backend.geojson([10,60,5,65])

    def test_end_to_end_native_cell_prediction_deletion_and_persistent_resume(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            backend=self.backend()
            backend.folder=Path(tmp)
            backend.fingerprint="test"
            backend.torch=torch
            backend.model=fusion_model(torch,8).eval()
            backend.mean,backend.std=np.zeros(39,np.float32),np.ones(39,np.float32)
            backend.base.days=["2024-12-15"]
            backend.base.station_table=backend.base.table[:2].copy()
            backend.base.positions=np.array([[0,0],[0,1]])
            backend.base.valid=np.ones((2,2),bool)
            backend.context=None
            backend.observations=lambda day,hour:(np.array([0,1]),np.array([-2.,-3.],np.float32))
            backend.uncertainty=SimpleNamespace(radii=lambda x,a,m,p:m.sum(1).astype(float)+np.abs(p))
            links=dict(cell=np.array([2,3]),source=np.array([[0,1,-1,-1,-1,-1],[0,-1,-1,-1,-1,-1]]),
                       graph=np.zeros((2,6,5),np.float32))
            fields=np.zeros((3,4,4),np.float32)
            forecast=dict(fields=fields,projection_xy=np.zeros((4,2)),url=np.array("test"),run_time=np.array("test"))
            backend.cache=lambda name,build:links if name.endswith("_links") else forecast
            torch.set_num_threads(1)
            meta=backend.prepare("2024-12-15",3,lambda _:None)
            self.assertEqual(meta["cells"],2)
            self.assertEqual(meta["sources"],2)
            combined=np.vstack([backend.base.station_table,backend.base.table[[2,3]]])
            features,_=temperature_features(combined,0,np.array([2]),np.zeros((1,5)),-2.,
                pd.Timestamp("2024-12-15T03:00:00Z"),fields[1],forecast["projection_xy"])
            with torch.inference_mode():
                expected=backend.model(torch.from_numpy(features[None]),torch.tensor([[-2.]]),torch.tensor([[True]])).item()
            self.assertAlmostEqual(float(backend.context["temperature"][1]),expected,places=5)
            first=backend.detail(2)
            without=[first["temperature_c"]+source["temperature_change_c"] for source in first["sources"]]
            with torch.inference_mode():
                features_b,_=temperature_features(combined,1,np.array([2]),np.zeros((1,5)),-3.,
                    pd.Timestamp("2024-12-15T03:00:00Z"),fields[1],forecast["projection_xy"])
                expected_b=backend.model(torch.from_numpy(features_b[None]),torch.tensor([[-3.]]),torch.tensor([[True]])).item()
            self.assertAlmostEqual(without[0],expected_b,places=5)
            self.assertTrue(np.isnan(backend.context["removed_radius"][1]).all())
            original=backend.context["temperature"].copy()
            backend.context=None
            with patch.object(backend,"cache",side_effect=AssertionError("Cache must restore finished prediction")):
                again=backend.prepare("2024-12-15",3,lambda _:None)
            np.testing.assert_array_equal(backend.context["temperature"],original)
            self.assertEqual(meta,again)


if __name__=="__main__":
    unittest.main()
