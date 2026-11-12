"""Kontroller norsk eksport mot et lite raster med adskilte soneområder."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd
import export_climate_norwegian as norwegian

import rasterio
from rasterio.transform import from_origin
import geopandas as gpd

class NorwegianClimateTests(unittest.TestCase):
    def create_fixture(self, root: Path) -> np.ndarray:
        folder = root / "k2"
        folder.mkdir(parents=True)
        labels = np.array([[1, 1, 0, 2], [1, 0, 2, 2], [0, 2, 1, 0]], dtype=np.uint8)
        with rasterio.open(
            folder / "climate_regions_1km.tif", "w", driver="GTiff", width=4, height=3,
            count=1, dtype="uint8", crs=32633, transform=from_origin(260000, 6660000, 1000, 1000),
            nodata=0,
        ) as ds:
            ds.write(labels, 1)
        rows = []
        for zone, x in ((1, 260500), (2, 263500)):
            row = dict(region_id=zone, cells=4, representative_x=x, representative_y=6659500,
                       mean_temperature_c=-1 if zone == 1 else 5,
                       mean_annual_precipitation_mm=600 if zone == 1 else 2500,
                       summer_temperature_c=10, winter_temperature_c=-10,
                       summer_winter_temperature_amplitude_c=20, frost_days_per_year=200)
            for name in ("summer_temperature_c", "winter_temperature_c",
                         "summer_winter_temperature_amplitude_c", "frost_days_per_year"):
                row["representative_" + name] = row[name] + 1
            for name in ("temperature_rmse_c", "precipitation_rmse_mm", "wet_day_disagreement",
                         "monthly_precipitation_mae_mm", "annual_precipitation_mae_mm"):
                row[name + "_mean"] = 0.1
                row[name + "_p95"] = 0.2
            rows.append(row)
        pd.DataFrame(rows).to_csv(folder / "region_summary.csv", index=False)
        annual = []
        for zone in (1, 2):
            for year in range(1996, 2026):
                row = dict(year=year, region_id=zone, temperature_c=(year - 1996) / 10,
                           precipitation_mm=800 + year - 1996,
                           representative_temperature_c=1, representative_precipitation_mm=900)
                for name in norwegian.FIELDS:
                    if name.startswith(("longest_", "growing_", "heating_", "freezing_", "days_tg_", "max_")):
                        row[name] = 3
                        row["representative_" + name] = 4
                annual.append(row)
        annual = pd.DataFrame(annual)
        annual.to_csv(folder / "annual_profiles.csv", index=False)
        annual["decade_start"] = 1996 + ((annual.year - 1996) // 10) * 10
        annual.drop(columns="year").groupby(["region_id", "decade_start"]).mean().to_csv(
            folder / "decade_profiles.csv")
        pd.DataFrame([
            dict(region_id=zone, month=month, temperature_c=1, precipitation_mm=50,
                 representative_temperature_c=2, representative_precipitation_mm=60)
            for zone in (1, 2) for month in range(1, 13)
        ]).to_csv(folder / "seasonal_profiles.csv", index=False)
        (folder / "analysis.json").write_text(json.dumps(dict(
            status="complete", period="1996-2025", groups=2, complete_cells=8)))
        return labels

    def test_translation_keeps_representation_and_error_statistics_distinct(self):
        self.assertEqual(norwegian.field_info("representative_frost_days_per_year")[0],
                         "rep_frostdager_aar")
        self.assertEqual(norwegian.field_info("temperature_rmse_c_p95")[0], "temp_rmse_c_p95")
        with self.assertRaisesRegex(ValueError, "Mangler norsk"):
            norwegian.field_info("unrecognised_metric")

    def test_full_export_roundtrip_profiles_and_fact_based_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_fixture(root / "source")
            output = root / "norsk"
            norwegian.export(root / "source", output, [2])
            package = output / "klimasoner_norge_1996_2025.gpkg"
            zones = gpd.read_file(package, layer="klimasoner_2")
            reps = gpd.read_file(package, layer="representanter_2")
            self.assertEqual(zones.sone_id.tolist(), [1, 2])
            self.assertTrue((zones.geometry.geom_type == "MultiPolygon").all())
            self.assertTrue((zones.geometry.area == 4e6).all())
            self.assertEqual(zones.frostdager_aar.tolist(), [200, 200])
            self.assertIn("200,0 frostdager", zones.kjennetegn.iloc[0])
            self.assertIn("ikke nøyaktighet", zones.representasjon.iloc[0])
            self.assertEqual(reps.rep_frostdager_aar.tolist(), [201, 201])
            with closing(sqlite3.connect(package)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM aarsprofiler_2").fetchone()[0], 60)
                self.assertEqual(db.execute(
                    "SELECT COUNT(*) FROM gpkg_contents WHERE data_type='attributes'").fetchone()[0], 4)
                documented = {row[0] for row in db.execute("SELECT feltnavn FROM feltforklaring")}
            for frame in (zones, reps):
                self.assertTrue(set(frame.columns) - {"geometry"} <= documented)
            text = (output / "KLIMASONER_FORKLART.md").read_text(encoding="utf-8")
            self.assertIn("### Sone 1: Kaldt, nedbørfattig, stor", text)
            self.assertIn("### Sone 2: Mildt, svært nedbørrikt, stor", text)
            self.assertIn("200,0 frostdager", text)
            self.assertIn("201,0 per år", text)
            self.assertIn("2,00 °C i årsmiddeltemperatur", text)
            self.assertTrue((output / "klimasoner_2.png").is_file())
            self.assertEqual(pd.read_csv(output / "soneoversikt_2.csv", sep=";",
                                         decimal=",").frostdager_aar.tolist(), [200, 200])
            with self.assertRaises(FileExistsError):
                norwegian.export(root / "source", output, [2])

    def test_corrupt_count_fails_without_publishing_partial_deliverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_fixture(root / "source")
            summary_path = root / "source" / "k2" / "region_summary.csv"
            summary = pd.read_csv(summary_path)
            summary.loc[0, "cells"] = 5
            summary.to_csv(summary_path, index=False)
            with self.assertRaisesRegex(ValueError, "Antall ruter"):
                norwegian.export(root / "source", root / "norsk", [2])
            self.assertFalse((root / "norsk").exists())
            self.assertEqual(list(root.glob("norsk_arbeid_*")), [])


if __name__ == "__main__":
    unittest.main()
