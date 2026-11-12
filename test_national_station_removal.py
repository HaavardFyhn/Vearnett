import unittest
from unittest.mock import patch

import numpy as np
from national_station_removal import removal_arrays, scenario_backend, station_catalogue, register_station_removal
import test_national_joint_map as fixtures
from rasterio.io import MemoryFile


class StationRemovalTests(unittest.TestCase):
    def client(self):
        from flask import Flask
        backend=fixtures.NationalJointMapTests().backend()
        backend.context["metadata"].update(day="2024-12-15",hour=3,key="test")
        app=Flask(__name__)
        with patch("national_station_removal.NationalJointMap",return_value=backend):
            register_station_removal(app,"unused")
            client=app.test_client()
            self.assertEqual(client.get("/api/national-removal/options").status_code,200)
        return client,backend

    def test_native_tiles_same_cell_values_across_zoom_and_removal_nodata(self):
        from pyproj import Transformer
        client,backend=self.client()
        projection=Transformer.from_crs(25833,3857,always_xy=True)

        def pixel(cell,z,**query):
            x,y=projection.transform(*backend.base.table[cell,:2])
            limit=20037508.342789244
            size=2*limit/2**z
            column,row=int((x+limit)//size),int((limit-y)//size)
            response=client.get(f"/api/national-removal/tile/{z}/{column}/{row}.png",
                                query_string=query)
            self.assertEqual(response.status_code,200,response.data)
            self.assertEqual(response.mimetype,"image/png")
            with MemoryFile(response.data) as memory,memory.open() as image:
                self.assertEqual((image.width,image.height,image.count),(256,256,4))
                col=int((x+limit-column*size)/size*256)
                line=int((limit-y-row*size)/size*256)
                return image.read()[:,line,col]

        for z in (12,14):
            expected=np.rint(np.array([255,247,236])+
                (np.array([127,0,0])-np.array([255,247,236]))*4/25).astype(np.uint8)
            np.testing.assert_array_equal(pixel(0,z),np.r_[expected,200])
            np.testing.assert_array_equal(pixel(0,z,station="B",mode="change"),[33,102,172,200])
            np.testing.assert_array_equal(pixel(3,z,station="A",mode="without"),[85,85,85,200])
            self.assertEqual(pixel(1,z)[3],0)
        self.assertEqual(client.get("/api/national-removal/tile/20/0/0.png").status_code,400)
        self.assertEqual(client.get("/api/national-removal/tile/4/16/0.png").status_code,400)
        self.assertEqual(client.get("/api/national-removal/tile/12/2000/1000.png",
            query_string=dict(station="B",mode="change",metric="source_count")).status_code,400)

    def test_native_coordinate_lookup_and_summary_without_aggregation(self):
        client,backend=self.client()
        lon,lat=backend.base.to_lonlat.transform(*backend.base.table[0,:2])
        response=client.get("/api/national-removal/cell",query_string=dict(lon=lon,lat=lat))
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json,backend.detail(0))
        self.assertEqual(client.get("/api/national-removal/cell",
            query_string=dict(lon=0,lat=0)).status_code,400)
        with patch.object(backend,"geojson",side_effect=AssertionError("Must not aggregate")):
            full=client.get("/api/national-removal/summary")
            self.assertIsNone(full.json["removal"])
            changed=client.get("/api/national-removal/summary",
                query_string=dict(station="A",mode="change"))
            self.assertEqual(changed.status_code,200)
            self.assertEqual(changed.json["removal"]["affected_cells"],2)
            self.assertEqual(changed.json["removal"]["lost_cells"],1)
            self.assertNotIn("features",changed.json)

    def test_all_catalogue_stations_visible_with_explicit_reading_availability(self):
        backend=fixtures.NationalJointMapTests().backend()
        backend.context["station_ids"]=np.array([0])
        backend.context["temperatures"]=np.array([-2.])
        stations=station_catalogue(backend)
        self.assertEqual(len(stations),2)
        self.assertEqual(stations[0]["station_id"],"A")
        self.assertTrue(stations[0]["available"])
        self.assertEqual(stations[0]["source_c"],-2)
        self.assertFalse(stations[1]["available"])
        self.assertIsNone(stations[1]["source_c"])
        self.assertTrue(stations[1]["extrapolation"])

    def test_signed_changes_unaffected_cells_and_sole_source_loss(self):
        backend=fixtures.NationalJointMapTests().backend()
        c=backend.context
        changes=removal_arrays(c,0)
        np.testing.assert_array_equal(changes["used"],[True,True])
        np.testing.assert_array_equal(changes["lost"],[False,True])
        self.assertEqual(changes["radius_change"][0],1)
        self.assertTrue(np.isnan(changes["radius_change"][1]))
        other=removal_arrays(c,1)
        np.testing.assert_array_equal(other["radius_change"],[-2.,0.])
        np.testing.assert_array_equal(other["temperature_change"],[-1.,0.])
        self.assertFalse(other["lost"].any())

    def test_without_prediction_matches_saved_delta_and_does_not_mutate_full(self):
        backend=fixtures.NationalJointMapTests().backend()
        changes=removal_arrays(backend.context,1)
        scenario=scenario_backend(backend,changes,"without")
        np.testing.assert_array_equal(scenario.context["radius"],[2.,6.])
        np.testing.assert_array_equal(scenario.context["temperature"],[-6.,-10.])
        np.testing.assert_array_equal(backend.context["radius"],[4.,6.])
        delta=scenario_backend(backend,changes,"change")
        np.testing.assert_array_equal(delta.context["radius"],[-2.,0.])
        full=scenario_backend(backend,changes,"full")
        np.testing.assert_array_equal(full.context["radius"],backend.context["radius"])
        with self.assertRaises(ValueError):scenario_backend(backend,changes,"bad")

    def test_sole_source_without_geotiff_is_nodata_not_zero_radius(self):
        backend=fixtures.NationalJointMapTests().backend()
        without=scenario_backend(backend,removal_arrays(backend.context,0),"without")
        self.assertEqual(without.context["cell"].tolist(),[0])
        without.context["source"][without.context["source"]==0]=-1
        with MemoryFile(without.geotiff()) as memory,memory.open() as raster:
            self.assertEqual(raster.read(1,masked=True).count(),1)
            self.assertEqual(raster.read(2)[0,0],5.)
            self.assertEqual(raster.read(2)[1,1],-9999.)
            self.assertEqual(raster.read(5)[0,0],1)


if __name__=="__main__":
    unittest.main()
