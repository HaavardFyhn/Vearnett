from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from station_senorge_gpu import SeNorgeGPUPairModel, load_profile
from station_weather import WeatherRegistry, exact_network
from greedy_400_stations_pruned import TERRAIN, make_cache
import test_senorge_network_map as fixtures
from test_station_app import Scenario, FakeWeatherEngine, FakeWeather


class SeNorgeGPUStationTests(unittest.TestCase):
    def registry(self, backend):
        grid=pd.DataFrame(dict(x=backend.table[:,0],y=backend.table[:,1],
            elevation=backend.table[:,2],coast_dist_fjord=0.,coast_dist_ocean=0.,
            objtype="land",objtype_code=0))
        for field in TERRAIN:
            grid[field]=0.
        return SimpleNamespace(root=backend.root,models={"senorge-gpu-network-backend":backend},
            grid=grid,target_cache=make_cache(grid),cell_m=1000)

    def config(self):
        return dict(id="temperature:senorge_gpu_tg",kind="senorge_gpu_tg",phenomenon="temperature",
            label="New seNorge GPU",metric="RMSE",unit="C",prefix="unused",day="2024-12-15",variable=0)

    def test_profile_uses_exact_saved_network_and_preserves_all_station_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            backend=fixtures.SeNorgeNetworkMapTests().backend(folder)
            registry=self.registry(backend)
            profile=load_profile(registry,self.config())
            np.testing.assert_array_equal(profile["baseline"],[0,0,1.5,46.5])
            np.testing.assert_array_equal(profile["second"],[1.5,1.5,2.5,np.inf])
            np.testing.assert_array_equal(profile["winner"],[0,1,1,1])
            self.assertEqual(profile["stations"].station_id.tolist(),["A","B","OFF"])
            self.assertTrue(profile["allow_nodata"])
            self.assertEqual(profile["status"]["state"],"ready")
            self.assertIn("2024-12-15",profile["model_warning"])

    def test_proposed_moved_source_and_multi_removal_use_exact_pair_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            backend=fixtures.SeNorgeNetworkMapTests().backend(folder)
            registry=self.registry(backend)
            adapter=SeNorgeGPUPairModel(backend,"2024-12-15",0)
            source=make_cache(registry.grid.iloc[[0,1]])
            target=registry.target_cache
            result=adapter.predict_pairs(source,target,np.array([0,0,1,1]),np.array([0,3,2,3]))
            np.testing.assert_allclose(result,[0,np.inf,1.5,46.5])
            best,second,winner=exact_network(adapter,source,target,False,source_batch=2)
            np.testing.assert_array_equal(best,[0,0,1.5,46.5])
            np.testing.assert_array_equal(second,[1.5,1.5,2.5,np.inf])
            np.testing.assert_array_equal(winner,[0,1,1,1])
            single=exact_network(adapter,make_cache(registry.grid.iloc[[0]]),target,False)[0]
            np.testing.assert_array_equal(single,[0,1.5,2.5,np.inf])
            bad=SimpleNamespace(predict_pairs=lambda *args:np.full(4,np.inf))
            with self.assertRaises(ValueError):
                exact_network(bad,make_cache(registry.grid.iloc[[0]]),target,False)

    def test_registry_adds_only_new_choices_when_assets_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            gpu=root/"outputs"/"senorge_transfer_gpu_returned"
            for filename in ("verification.json","portable_ridge.json","portable_ridge.npz",
                             "senorge_transfer_gpu_results/report.json"):
                path=gpu/filename
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text("{}")
            registry=WeatherRegistry(root,pd.DataFrame(),{},[],models={})
            ids={row["id"] for row in registry.available}
            self.assertEqual(ids,{"temperature:senorge_gpu_tg","temperature:senorge_gpu_tn"})
            with patch("station_senorge_gpu.load_profile",return_value={"status":{"state":"ready"}}) as load:
                registry.profile("temperature:senorge_gpu_tg")
                self.assertEqual(load.call_args.args[1]["variable"],0)

    def test_original_scenario_nodata_statistics_and_removal_are_gated(self):
        weather=FakeWeather()
        original=weather.profile("temperature:continuous")
        gpu=dict(original,allow_nodata=True,baseline=np.array([1.,2.,np.inf]),
                 second=np.array([3.,np.inf,np.inf]),winner=np.array([0,1,-1]))
        engine=FakeWeatherEngine()
        engine.weather=SimpleNamespace(profile=lambda _:gpu,available=FakeWeather.available)
        scenario=Scenario(engine)
        state=scenario.state()
        self.assertEqual(state["mean_rmse"],1.5)
        scenario.frost_edits[1]={"removed":True}
        np.testing.assert_array_equal(scenario.surface(),[1,np.inf,np.inf])
        self.assertEqual(scenario.state()["mean_rmse"],1)
        scenario.comparison_scope="selected"
        scenario.selected={"kind":"frost","index":1}
        with np.errstate(invalid="ignore"):
            _,_,reduction=scenario.change()
        self.assertEqual(reduction[0],0)
        self.assertTrue(np.isnan(reduction[1:]).all())
        scenario.frost_edits.clear()
        scenario.engine.weather=FakeWeather()
        self.assertEqual(scenario.state()["mean_rmse"],float(original["baseline"].mean()))


if __name__=="__main__":
    unittest.main()
