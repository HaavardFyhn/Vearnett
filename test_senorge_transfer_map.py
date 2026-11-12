import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from senorge_transfer_map import SeNorgeTransferMap, register_senorge_transfer_map
from senorge_corrected_transfer import features
from rasterio.io import MemoryFile


class SeNorgeTransferMapTests(unittest.TestCase):
    def backend(self):
        backend=SeNorgeTransferMap.__new__(SeNorgeTransferMap)
        backend.table=np.zeros((4,16))
        backend.table[:,:2]=[[200000,6800000],[201000,6800000],[202000,6800000],[247000,6800000]]
        backend.table[:,2]=[100,200,300,400]
        backend.weather=np.array([[[-2,-1],[-3,-2],[-4,-3],[-5,-4]],
                                  [[-6,-5],[-7,-6],[-8,-7],[-9,-8]]],dtype=np.float32)
        backend.days=["2024-12-15","2024-12-16"]
        backend.plan=pd.DataFrame(dict(date=backend.days))
        backend.to_xy=Transformer.from_crs(4326,25833,always_xy=True)
        backend.to_lonlat=Transformer.from_crs(25833,4326,always_xy=True)
        backend.tree=cKDTree(backend.table[:,:2])
        backend.origin=np.array([200000,6800000])
        backend.ij=np.array([[0,0],[0,1],[0,2],[0,47]])
        backend.shape=(1,48)
        backend.context=None
        backend.key=0
        backend.model=SimpleNamespace(
            predict=lambda x,c:(np.full(len(x),c)+x[:,7]*.001,np.full(len(x),1.25)),
            selected=dict(mean="test_mean",error="test_error"))
        return backend

    def prepare(self,backend,**kwargs):
        lon,lat=backend.to_lonlat.transform(*backend.table[0,:2])
        return backend.prepare(lon,lat,"2024-12-15",0,**kwargs)

    def test_reference_is_any_grid_cell_no_station_catalogue(self):
        backend=self.backend()
        lon,lat=backend.to_lonlat.transform(*backend.table[1,:2])
        reference=backend.reference(lon,lat,"2024-12-15",1)
        self.assertEqual(reference["cell"],1)
        self.assertEqual(reference["archive_c"],-7)
        with self.assertRaises(ValueError):backend.reference(0,0,"2024-12-15",0)
        with self.assertRaises(ValueError):backend.reference(lon,lat,"2020-01-01",0)
        with self.assertRaises(ValueError):backend.reference(lon,lat,"2024-12-15",2)

    def test_exact_training_features_archive_and_override_identity_and_46km(self):
        backend=self.backend()
        with patch.object(backend.model,"predict",wraps=backend.model.predict) as predict:
            data=self.prepare(backend)
            expected=features(backend.table,np.array([0,0,0]),np.array([0,1,2]),
                np.full(3,-2),np.repeat("2024-12-15",3),np.zeros(3))
            np.testing.assert_array_equal(predict.call_args.args[0],expected)
        self.assertEqual(data["cells"],3)
        self.assertEqual(data["radius_km"],46)
        self.assertEqual(backend.context["targets"].tolist(),[0,1,2])
        self.assertEqual(backend.context["temperature"][0],-2)
        self.assertEqual(backend.context["rmse"][0],0)
        np.testing.assert_array_equal(backend.context["rmse"][1:],[1.25,1.25])
        data=self.prepare(backend,override=10)
        self.assertTrue(data["reference"]["manual_override"])
        self.assertEqual(backend.context["temperature"][0],10)
        np.testing.assert_array_equal(backend.context["observed"],[-2,-3,-4])
        with self.assertRaises(ValueError):self.prepare(backend,override=np.nan)

    def test_target_archive_never_enters_inference_and_stale_keys_fail(self):
        backend=self.backend()
        first=self.prepare(backend)
        prediction=backend.context["temperature"].copy()
        backend.weather[0,1:,0]=99
        second=self.prepare(backend)
        np.testing.assert_array_equal(backend.context["temperature"],prediction)
        lon,lat=backend.to_lonlat.transform(*backend.table[1,:2])
        detail=backend.detail(lon,lat,second["key"])
        self.assertEqual(detail["archived_target_c"],99)
        self.assertEqual(detail["predicted_rmse_c"],1.25)
        with self.assertRaises(ValueError):backend.detail(lon,lat,first["key"])
        lon,lat=backend.to_lonlat.transform(*backend.table[3,:2])
        with self.assertRaises(ValueError):backend.detail(lon,lat,second["key"])

    def test_geotiff_native_resolution_temperature_rmse_nodata(self):
        backend=self.backend()
        data=self.prepare(backend)
        with MemoryFile(backend.geotiff(data["key"])) as memory,memory.open() as image:
            self.assertEqual(image.count,2)
            self.assertEqual(image.res,(1000,1000))
            self.assertEqual(image.crs.to_epsg(),25833)
            self.assertEqual(image.descriptions,("temperature_c","rmse_c"))
            self.assertEqual(image.read(1,masked=True).count(),3)
            self.assertEqual(image.read(1)[0,0],-2)
            self.assertEqual(image.read(2)[0,0],0)
            self.assertEqual(image.read(2)[0,1],1.25)
            self.assertEqual(image.read(2)[0,47],-9999)

    def test_routes_reference_prepare_detail_and_tile_png(self):
        from flask import Flask
        backend=self.backend()
        app=Flask(__name__)
        with patch("senorge_transfer_map.SeNorgeTransferMap",return_value=backend):
            register_senorge_transfer_map(app,"unused")
            client=app.test_client()
            self.assertEqual(client.get("/api/senorge-transfer/options").status_code,200)
        lon,lat=backend.to_lonlat.transform(*backend.table[0,:2])
        query=dict(lon=lon,lat=lat,day="2024-12-15",variable=0)
        reference=client.get("/api/senorge-transfer/reference",query_string=query)
        self.assertEqual(reference.json["archive_c"],-2)
        result=client.post("/api/senorge-transfer/prepare",query_string=dict(query,temperature=10))
        self.assertEqual(result.status_code,200,result.data)
        key=result.json["key"]
        detail=client.get("/api/senorge-transfer/cell",query_string=dict(lon=lon,lat=lat,key=key))
        self.assertEqual(detail.json["temperature_c"],10)
        self.assertTrue(detail.json["identity_rule"])
        x,y=Transformer.from_crs(25833,3857,always_xy=True).transform(*backend.table[1,:2])
        limit=20037508.342789244
        for z in (12,14):
            size=2*limit/2**z
            col,row=int((x+limit)//size),int((limit-y)//size)
            response=client.get(f"/api/senorge-transfer/tile/{z}/{col}/{row}.png",
                                query_string=dict(key=key,metric="rmse_c"))
            self.assertEqual(response.status_code,200)
            with MemoryFile(response.data) as memory,memory.open() as image:
                i,j=int((limit-y-row*size)/size*256),int((x+limit-col*size)/size*256)
                expected=np.rint(np.array([255,247,236])+
                    (np.array([127,0,0])-np.array([255,247,236]))*1.25/5).astype(np.uint8)
                np.testing.assert_array_equal(image.read()[:,i,j],np.r_[expected,200])
        response=client.get("/api/senorge-transfer/export",query_string=dict(key=key))
        self.assertEqual(response.mimetype,"image/tiff")
        self.assertEqual(client.get("/api/senorge-transfer/cell",
            query_string=dict(lon=lon,lat=lat,key=999)).status_code,400)


if __name__=="__main__":
    unittest.main()
