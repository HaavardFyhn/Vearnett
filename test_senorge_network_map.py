from collections import OrderedDict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from senorge_network_map import SeNorgeNetworkMap, network_scenario, register_senorge_network_map
import test_senorge_transfer_map as fixtures
from rasterio.io import MemoryFile


class SeNorgeNetworkMapTests(unittest.TestCase):
    def backend(self, folder):
        base=fixtures.SeNorgeTransferMapTests().backend()
        backend=SeNorgeNetworkMap.__new__(SeNorgeNetworkMap)
        backend.__dict__.update(base.__dict__)
        lon,lat=backend.to_lonlat.transform(backend.table[[0,1],0],backend.table[[0,1],1])
        backend.stations=pd.DataFrame(dict(station_id=["A","B","OFF"],name=["A","B","Off grid"],
            lon=[lon[0],lon[1],0],lat=[lat[0],lat[1],0]))
        backend.root=Path(folder)
        model_folder=backend.root/"outputs"/"senorge_transfer_gpu_returned"
        model_folder.mkdir(parents=True)
        (model_folder/"portable_ridge.json").write_text("{}")
        (backend.root/"senorge_transfer_map.py").write_text("test adapter")
        backend.folder=backend.root/"cache"
        backend.folder.mkdir()
        backend.fingerprint="test"
        backend.catalogue_sha="test-catalogue"
        backend.raster_cache=OrderedDict()
        backend.model.manifest=dict(provenance={"test":"test"})
        backend.model.predict=lambda x,c:(c+x[:,4],np.abs(x[:,4])+.5)
        return backend

    def test_original_minimum_next_best_ties_unaffected_and_lost(self):
        context=dict(best=np.array([1.,2.,3.,np.inf,0.],np.float32),
            second=np.array([2.,2.,np.inf,np.inf,0.],np.float32),
            winner=np.array([0,1,0,-1,0]),
            temperature=np.array([10,20,30,np.nan,0.],np.float32),
            second_temperature=np.array([11,21,np.nan,np.nan,0.],np.float32))
        removed=network_scenario(context,0)
        np.testing.assert_allclose(removed["radius"][:2],[2,2])
        np.testing.assert_array_equal(removed["lost"],[False,False,True,False,False])
        np.testing.assert_allclose(removed["change"][[0,1,4]],[1,0,0])
        self.assertTrue(np.isnan(removed["change"][2]))
        self.assertTrue(np.isnan(removed["change"][3]))
        np.testing.assert_array_equal(context["best"][:3],[1,2,3])
        context["second"][0]=.5
        with self.assertRaises(ValueError):network_scenario(context,0)

    def test_full_station_inference_matches_bruteforce_and_saved_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            backend=self.backend(folder)
            result=backend.prepare_network("2024-12-15",0)
            self.assertEqual(result["metadata"]["stations"],3)
            self.assertEqual(result["metadata"]["eligible_stations"],2)
            self.assertEqual(result["metadata"]["cells"],4)
            self.assertFalse(result["catalogue"][2]["available"])
            np.testing.assert_allclose(backend.context["best"],[0,0,1.5,46.5])
            np.testing.assert_allclose(backend.context["second"],[1.5,1.5,2.5,np.inf])
            np.testing.assert_array_equal(backend.context["winner"],[0,1,1,1])
            removed=backend.summary(backend.key,"B")
            self.assertEqual(removed["lost_cells"],1)
            self.assertEqual(removed["increased_cells"],2)
            self.assertEqual(removed["unchanged_cells"],1)
            off=backend.summary(backend.key,"OFF")
            self.assertEqual(off["affected_cells"],0)
            with patch.object(backend.model,"predict",side_effect=AssertionError("Must restore saved prediction")):
                restored=backend.prepare_network("2024-12-15",0)
                self.assertEqual(result["metadata"],restored["metadata"])
            with self.assertRaises(ValueError):backend.summary("bad","A")
            with self.assertRaises(ValueError):backend.summary(backend.key,"bad")

    def test_signed_delta_no_mutation_export_and_lost_grey_png(self):
        from pyproj import Transformer
        with tempfile.TemporaryDirectory() as folder:
            backend=self.backend(folder)
            backend.prepare_network("2024-12-15",0)
            key=backend.key
            backend.scenario(key,"B","change")
            self.assertEqual(backend.rasters["rmse_c"][0,0],0)
            self.assertEqual(backend.rasters["rmse_c"][0,1],1.5)
            self.assertEqual(backend.rasters["rmse_c"][0,2],1)
            self.assertTrue(np.isnan(backend.rasters["rmse_c"][0,47]))
            with MemoryFile(backend.network_export(key,"B","without")) as memory,memory.open() as image:
                self.assertEqual(image.res,(1000,1000))
                np.testing.assert_allclose(image.read(2)[0,:3],[0,1.5,2.5])
                self.assertEqual(image.read(2)[0,47],-9999)
            backend.scenario(key,"B","change")
            x,y=Transformer.from_crs(25833,3857,always_xy=True).transform(*backend.table[3,:2])
            z=14
            limit=20037508.342789244
            size=2*limit/2**z
            col,row=int((x+limit)//size),int((limit-y)//size)
            png=backend.tile(z,col,row,"rmse_c",key,lost=backend.loss_raster)
            with MemoryFile(png) as memory,memory.open() as image:
                i,j=int((limit-y-row*size)/size*256),int((x+limit-col*size)/size*256)
                np.testing.assert_array_equal(image.read()[:,i,j],[85,85,85,200])
            np.testing.assert_array_equal(backend.context["best"],[0,0,1.5,46.5])

    def test_network_routes_prepare_summary_lookup_export_and_invalid_requests(self):
        from flask import Flask
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            backend=self.backend(folder)
            app=Flask(__name__)
            with patch("senorge_network_map.SeNorgeNetworkMap",return_value=backend):
                register_senorge_network_map(app,folder)
                client=app.test_client()
                self.assertEqual(client.get("/api/senorge-network/status").json["state"],"idle")
                with patch("senorge_network_map.threading.Thread",
                    side_effect=lambda target,**kwargs:SimpleNamespace(start=target)):
                    response=client.post("/api/senorge-network/prepare",
                        query_string=dict(day="2024-12-15",variable=0))
                    self.assertEqual(response.status_code,202)
            state=client.get("/api/senorge-network/status").json
            self.assertEqual(state["state"],"ready")
            self.assertEqual(len(state["catalogue"]),3)
            key=state["metadata"]["key"]
            query=dict(key=key,station="B")
            summary=client.get("/api/senorge-network/summary",query_string=query)
            self.assertEqual(summary.json["increased_cells"],2)
            lon,lat=backend.to_lonlat.transform(*backend.table[3,:2])
            detail=client.get("/api/senorge-network/cell",query_string=dict(query,lon=lon,lat=lat))
            self.assertTrue(detail.json["lost_prediction"])
            self.assertIsNone(detail.json["increase_c"])
            export=client.get("/api/senorge-network/export",query_string=dict(query,mode="without"))
            self.assertEqual(export.mimetype,"image/tiff")
            self.assertEqual(client.get("/api/senorge-network/summary",
                query_string=dict(key="bad",station="B")).status_code,400)
            self.assertEqual(client.get("/api/senorge-network/summary",
                query_string=dict(key=key,station="bad")).status_code,400)


if __name__=="__main__":
    unittest.main()
