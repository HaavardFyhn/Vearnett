"""Lag norske klimasoner, GeoPackage, kart og sonebeskrivelser uten ny analyse."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
import pandas as pd
from pyproj import CRS
import rasterio
from rasterio.features import rasterize, shapes
import geopandas as gpd
from shapely.geometry import MultiPolygon, shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent
PERIOD = "1996–2025"
MONTHS = ("januar", "februar", "mars", "april", "mai", "juni",
          "juli", "august", "september", "oktober", "november", "desember")
FIELDS = {
    "region_id": ("sone_id", "", "Sonenummer innen denne inndelingen, sortert fra kaldere til varmere årsmiddel."),
    "cells": ("antall_ruter", "ruter", "Antall komplette 1 km-ruter i sonen."),
    "representative_x": ("representant_ost", "m", "Representantens østkoordinat i EPSG:32633."),
    "representative_y": ("representant_nord", "m", "Representantens nordkoordinat i EPSG:32633."),
    "mean_temperature_c": ("aarsmiddel_temp_c", "°C", "Gruppemiddel av døgnmiddeltemperatur over alle 10 958 døgn."),
    "mean_annual_precipitation_mm": ("aarsnedbor_mm", "mm/år", "Gjennomsnittlig årsnedbør, midlet over ruter og år."),
    "temperature_rmse_c": ("temp_rmse_c", "°C", "RMSE mellom rutens og representantens døgnmiddeltemperatur."),
    "precipitation_rmse_mm": ("nedbor_rmse_mm", "mm/døgn", "RMSE mellom rutens og representantens døgnnedbør."),
    "wet_day_disagreement": ("vatdag_avvik", "andel", "Andel døgn der rute og representant er uenige om nedbør ≥1 mm."),
    "monthly_precipitation_mae_mm": ("maanedsnedbor_mae_mm", "mm/måned", "Midlere absolutt forskjell i månedssummer mot representanten."),
    "annual_precipitation_mae_mm": ("aarsnedbor_mae_mm", "mm/år", "Midlere absolutt forskjell i årssummer mot representanten."),
    "summer_temperature_c": ("sommertemp_c", "°C", "Gjennomsnittlig døgnmiddeltemperatur juni–august, dagvektet."),
    "winter_temperature_c": ("vintertemp_c", "°C", "Gjennomsnittlig døgnmiddeltemperatur januar, februar og desember, dagvektet."),
    "summer_winter_temperature_amplitude_c": ("sommer_vinter_amplitude_c", "°C", "Sommertemperatur minus vintertemperatur."),
    "frost_days_per_year": ("frostdager_aar", "dager/år", "Antall døgn med minimumstemperatur under 0 °C, årlig gjennomsnitt."),
    "year": ("aar", "år", "Kalenderår."),
    "month": ("maaned", "", "Kalendermåned 1–12."),
    "decade_start": ("tiaar_start", "år", "Startår for tiårsperioden 1996–2005, 2006–2015 eller 2016–2025."),
    "temperature_c": ("middeltemp_c", "°C", "Gjennomsnittlig døgnmiddeltemperatur i tabellens periode."),
    "precipitation_mm": ("nedborsum_mm", "mm", "Nedbørsum i tabellens periode, midlet over år der relevant."),
    "longest_wet_spell_days": ("lengste_vatperiode_dager", "dager", "Lengste sammenhengende periode med nedbør ≥1 mm/døgn innen kalenderåret."),
    "longest_dry_spell_days": ("lengste_torrperiode_dager", "dager", "Lengste sammenhengende periode med nedbør <1 mm/døgn innen kalenderåret."),
    "longest_warm_spell_tg_ge20_days": ("lengste_varmperiode_dager", "dager", "Lengste periode med døgnmiddeltemperatur ≥20 °C innen kalenderåret."),
    "longest_cold_spell_tg_lt0_days": ("lengste_kuldeperiode_dager", "dager", "Lengste periode med døgnmiddeltemperatur <0 °C innen kalenderåret."),
    "growing_degree_days_base5": ("vekstgraddager_base5", "°C·døgn", "Sum av maks(døgnmiddeltemperatur −5 °C, 0) per år."),
    "heating_degree_days_base17": ("oppvarmingsgraddager_base17", "°C·døgn", "Sum av maks(17 °C − døgnmiddeltemperatur, 0) per år."),
    "freezing_degree_days_base0": ("frostgraddager_base0", "°C·døgn", "Sum av maks(−døgnmiddeltemperatur, 0) per år."),
    "days_tg_gt5": ("dager_middeltemp_over5", "dager", "Antall døgn med døgnmiddeltemperatur >5 °C per år."),
    "max_3day_precipitation_mm": ("maks_3dogn_nedbor_mm", "mm", "Største sum over tre påfølgende døgn innen hvert kalenderår."),
    "max_5day_precipitation_mm": ("maks_5dogn_nedbor_mm", "mm", "Største sum over fem påfølgende døgn innen hvert kalenderår."),
}


def field_info(source: str) -> tuple[str, str, str]:
    if source.startswith("representative_") and source not in ("representative_x", "representative_y"):
        name, unit, description = field_info(source.removeprefix("representative_"))
        return "rep_" + name, unit, "Representantpunkt: " + description
    for suffix, norwegian, description in (
        ("_mean", "_middel", " Gjennomsnitt over sonens ruter."),
        ("_p95", "_p95", " 95-persentil over sonens ruter; ikke konfidensgrense."),
    ):
        if source.endswith(suffix):
            name, unit, base = field_info(source[:-len(suffix)])
            return name + norwegian, unit, base + description
    if source not in FIELDS:
        raise ValueError(f"Mangler norsk feltforklaring: {source}")
    return FIELDS[source]


def translate(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={column: field_info(column)[0] for column in frame.columns})


def zone_name(row: pd.Series) -> str:
    t, p, a = row.mean_temperature_c, row.mean_annual_precipitation_mm, row.summer_winter_temperature_amplitude_c
    thermal = "Svært kaldt" if t < -2 else "Kaldt" if t < 0 else "Kjølig" if t < 3 else "Mildt"
    rain = "nedbørfattig" if p < 800 else "moderat nedbørrikt" if p < 1400 else "nedbørrikt" if p < 2200 else "svært nedbørrikt"
    contrast = "liten" if a < 15 else "middels" if a < 20 else "stor"
    return f"{thermal}, {rain}, {contrast} sommer–vinter-kontrast"


def number(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def characteristics(row: pd.Series) -> str:
    return (
        f"Årsmiddel {number(row.mean_temperature_c)} °C; sommer "
        f"{number(row.summer_temperature_c)} °C; vinter {number(row.winter_temperature_c)} °C. "
        f"Sommer–vinter-amplitude {number(row.summer_winter_temperature_amplitude_c)} °C. "
        f"Årsnedbør {number(row.mean_annual_precipitation_mm, 0)} mm; "
        f"{number(row.frost_days_per_year)} frostdager per år (minimum <0 °C). "
        "Verdiene er gjennomsnitt for gruppen i 1996–2025, ikke verdier fra én værstasjon."
    )


def representation(row: pd.Series) -> str:
    return (
        f"Midlere temperatur-RMSE mot representantpunktet: "
        f"{number(row.temperature_rmse_c_mean, 2)} °C; nedbør-RMSE: "
        f"{number(row.precipitation_rmse_mm_mean, 2)} mm/døgn. "
        f"Uenighet om våt/tørr dag: {number(100 * row.wet_day_disagreement_mean)} %. "
        "Dette er likhet i seNorge-data, ikke nøyaktighet mot uavhengige målestasjoner."
    )


def read_partition(source: Path, k: int) -> dict:
    folder = source / f"k{k}"
    meta = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
    if meta["status"] != "complete" or meta["period"] != "1996-2025" or meta["groups"] != k:
        raise ValueError(f"Ufullstendig eller uventet analyse for {k} soner")
    summary = pd.read_csv(folder / "region_summary.csv").sort_values("region_id")
    if summary.region_id.tolist() != list(range(1, k + 1)):
        raise ValueError("Soneoversikt må inneholde hver sone nøyaktig én gang")
    tables = {name: pd.read_csv(folder / filename) for name, filename in (
        ("maanedsprofiler", "seasonal_profiles.csv"),
        ("aarsprofiler", "annual_profiles.csv"), ("tiaarsprofiler", "decade_profiles.csv"),
    )}
    for name, frame in tables.items():
        if set(frame.region_id) != set(summary.region_id) or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Ugyldige verdier eller soner i {name}")
    with rasterio.open(folder / "climate_regions_1km.tif") as raster:
        labels, transform, crs = raster.read(1), raster.transform, raster.crs
        if CRS.from_user_input(crs).to_epsg() != 32633 or raster.res != (1000, 1000) or raster.nodata != 0:
            raise ValueError("Forventet 1 km-raster i EPSG:32633 med NoData=0")
    if not np.array_equal(np.unique(labels[labels > 0]), np.arange(1, k + 1)):
        raise ValueError("Rasterets soner stemmer ikke med soneoversikten")
    counts = np.bincount(labels.ravel(), minlength=k + 1)[1:]
    if not np.array_equal(counts, summary.cells) or int(counts.sum()) != meta["complete_cells"]:
        raise ValueError("Antall ruter stemmer ikke med analysen")
    parts = {zone: [] for zone in range(1, k + 1)}
    for geometry, value in shapes(labels, mask=labels > 0, transform=transform, connectivity=4):
        parts[int(value)].append(shape(geometry))
    geometries = []
    for zone in range(1, k + 1):
        geometry = unary_union(parts[zone])
        geometries.append(MultiPolygon([geometry]) if geometry.geom_type == "Polygon" else geometry)
    norwegian = translate(summary)
    norwegian["sonenavn"] = [zone_name(row) for _, row in summary.iterrows()]
    norwegian["kjennetegn"] = [characteristics(row) for _, row in summary.iterrows()]
    norwegian["representasjon"] = [representation(row) for _, row in summary.iterrows()]
    norwegian["periode"] = PERIOD
    norwegian["ruteareal_km2"] = summary.cells.to_numpy(dtype=float)
    norwegian["kilde"] = "MET Norge, seNorge_2018"
    annual_mean = tables["aarsprofiler"].groupby("region_id").mean()
    extra = [key for key in FIELDS if key.startswith(("longest_", "growing_", "heating_", "freezing_", "days_tg_", "max_"))]
    for key in extra:
        norwegian[field_info(key)[0]] = annual_mean.loc[summary.region_id, key].to_numpy()
        rep_key = "representative_" + key
        norwegian[field_info(rep_key)[0]] = annual_mean.loc[summary.region_id, rep_key].to_numpy()
    zones = gpd.GeoDataFrame(norwegian, geometry=geometries, crs=crs)
    representatives = gpd.GeoDataFrame(
        norwegian.copy(), geometry=gpd.points_from_xy(summary.representative_x, summary.representative_y), crs=crs,
    )
    representatives["rolle"] = "Faktisk 1 km-rute nær gruppens sentrum i standardisert egenskapsrom"
    return dict(k=k, summary=summary, tables=tables, zones=zones, representatives=representatives,
                labels=labels, transform=transform, crs=crs, metadata=meta)


def verify_partition(partition: dict, zones: gpd.GeoDataFrame, representatives: gpd.GeoDataFrame) -> None:
    if not zones.is_valid.all() or zones.geometry.is_empty.any() or zones.crs != partition["crs"]:
        raise ValueError("Ugyldig geometri eller koordinatsystem etter eksport")
    expected = partition["labels"]
    actual = rasterize(
        zip(zones.geometry, zones.sone_id), out_shape=expected.shape,
        transform=partition["transform"], fill=0, dtype=expected.dtype,
    )
    if not np.array_equal(expected, actual):
        raise ValueError("Polygonene gjengir ikke det opprinnelige rasteret nøyaktig")
    if not np.allclose(zones.area / 1e6, zones.ruteareal_km2, rtol=0, atol=1e-8):
        raise ValueError("Polygonareal stemmer ikke med antall 1 km-ruter")
    if len(representatives) != partition["k"]:
        raise ValueError("Feil antall representanter")
    by_id = zones.set_index("sone_id")
    for point in representatives.itertuples():
        if not by_id.loc[point.sone_id].geometry.contains(point.geometry):
            raise ValueError(f"Representanten ligger utenfor sone {point.sone_id}")


def plot_map(partition: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    k, labels, tr = partition["k"], partition["labels"], partition["transform"]
    cmap = ListedColormap(plt.get_cmap("tab20" if k <= 20 else "gist_ncar")(np.linspace(0, 1, k)))
    extent = (tr.c, tr.c + tr.a * labels.shape[1], tr.f + tr.e * labels.shape[0], tr.f)
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set_facecolor("#e6e6e6")
    image = ax.imshow(np.ma.masked_equal(labels, 0), extent=extent, cmap=cmap,
                      norm=BoundaryNorm(np.arange(0.5, k + 1.5), k), interpolation="nearest")
    reps = partition["representatives"]
    ax.scatter(reps.geometry.x, reps.geometry.y, c="black", marker="x", s=18, label="Representantpunkt")
    ax.set(title=f"Klimasoner i Norge {PERIOD} – {k} grupper",
           xlabel="Østkoordinat, UTM 33N (m)", ylabel="Nordkoordinat, UTM 33N (m)")
    ax.legend(loc="lower right")
    fig.colorbar(image, ax=ax, ticks=np.arange(1, k + 1), label="Sonenummer (fra kaldere til varmere årsmiddel)")
    fig.tight_layout()
    fig.savefig(output / f"klimasoner_{k}.png", dpi=150)
    plt.close(fig)


def report(partitions: list[dict]) -> str:
    text = [
        f"# Klimasoner i Norge {PERIOD}",
        "",
        "## Hva viser inndelingen?",
        "",
        "Dette er en datadrevet gruppering av 1 km-ruter med lignende temperatur- og "
        "nedbørforhold i seNorge_2018 fra MET Norge. Analysen bruker absolutt værnivå, "
        "årstidsvariasjon, ekstremverdier, frost, våte/tørre perioder, graddager, "
        "variasjon og endringer mellom år samt samvariasjon i det daglige været. "
        "Det er ikke en offisiell Köppen-inndeling, og 1996–2025 er ikke standardnormalen 1991–2020.",
        "",
        "GeoPackage-filen `klimasoner_norge_1996_2025.gpkg` inneholder polygonlagene "
        "`klimasoner_8` og `klimasoner_32`, representantpunkter og norske profiltabeller. "
        "Hver sone er én multipolygon med alle sine adskilte områder; områdene er ikke "
        "bundet sammen med kunstige forbindelser. Grensene følger de opprinnelige "
        "1 km-rutene nøyaktig, uten glatting. Koordinatsystem: WGS 84 / UTM sone 33N, EPSG:32633. "
        "Areal er ruteareal i kartplanet, ikke et detaljert landareal fra eiendomsgrenser.",
        "",
        "Åpne GeoPackage-filen i ArcGIS Pro og legg til ønsket polygonlag. "
        "Bruk **Unike verdier / Unique Values** på `sone_id`; bruk `sonenavn` som etikett. "
        "Klikk på en sone for norske egenskaper. Felter med `rep_` gjelder representanten, "
        "ikke gjennomsnittet av sonen. `feltforklaring` beskriver navn, enheter og kildefelt. "
        "CSV-filene er UTF-8 med BOM, semikolon og desimalkomma, egnet for norsk Excel.",
        "",
        "## Hvordan skal en representant forstås?",
        "",
        "Representanten er en faktisk rute som ligger nær K-means-senteret i det "
        "standardiserte egenskapsrommet. Den er ikke nødvendigvis geografisk sentral "
        "eller lik gruppemiddelet på alle egenskaper, og er ikke en målt værstasjon "
        "eller et anbefalt sted å plassere en stasjon. Koordinatene viser rutens sentrum. "
        "Ulike ruter i samme sone kan fremdeles få svært ulikt vær på en bestemt dag.",
        "",
        "Sommer betyr juni–august. Vinter betyr januar, februar og desember i samme "
        "kalenderår. Temperaturene er dagvektet, også i skuddår. Amplitude er "
        "sommermiddel minus vintermiddel, ikke varmeste minus kaldeste døgn. "
        "En frostdag har minimumstemperatur under 0 °C; kuldeperioder bruker derimot "
        "døgnmiddel under 0 °C. Nedbør ≥1 mm definerer en våt dag. "
        "Lengste perioder og maksimale 3-/5-døgnssummer nullstilles ved årsskiftet.",
        "",
        "Sonenavnene er beskrivende etiketter, ikke nye beregnede klasser. "
        "Årsmiddel under −2 °C kalles svært kaldt, −2 til under 0 kaldt, "
        "0 til under 3 kjølig og minst 3 mildt. Årsnedbør under 800 mm kalles "
        "nedbørfattig, 800 til under 1400 moderat nedbørrikt, 1400 til under 2200 "
        "nedbørrikt og minst 2200 svært nedbørrikt. Sommer–vinter-kontrast under "
        "15 °C kalles liten, 15 til under 20 middels og minst 20 stor. "
        "Disse ordvalgene påvirker ikke grupperingen; flere soner kan få samme beskrivende navn.",
        "",
        "## Usikkerhet og begrensninger",
        "",
        "RMSE er avvik mot representanten i samme griddede produkt, ikke målefeil "
        "mot uavhengige stasjoner. Oppgitte RMSE-tall er gjennomsnitt av rutenes "
        "RMSE over alle døgn. P95 er 95-persentilen over ruter, ikke en "
        "konfidensgrense. Uenighet om våt/tørr dag er en andel døgn, ikke en "
        "sannsynlighet for at sonen er feil. Fine soner gir mindre avvik, men grensene "
        "er mindre stabile når femårsblokker utelates. Daglig samvariasjon er "
        "komprimert med tilfeldig projeksjon og PCA; sluttavvikene beregnes fra originaldata.",
        "",
        "Bare komplette landruter på fastlandet og tilhørende øyer inngår. "
        "Svalbard, Jan Mayen, naboland og ruter med manglende data er ikke klassifisert. "
        "Det inngår ikke vind, luftfuktighet eller et eget snødatasett. "
        "Døgnmiddeltemperatur og nedbør gjelder 06–06 UTC; minimum/maksimum gjelder "
        "18–18 UTC. Tiårsendringene nedenfor er observerte forskjeller mellom "
        "1996–2005 og 2016–2025, ikke statistisk signifikante trender eller prognoser. "
        "Nedbør er generelt vanskeligere å representere med ett punkt enn temperatur.",
        "",
        "Kilde og kreditering: **MET Norge, seNorge_2018**. Behold kildehenvisning "
        "og vilkårene i originalarkivene ved videre bruk: "
        "https://www.met.no/en/free-meteorological-data/Licensing-and-crediting",
    ]
    for part in partitions:
        k, summary = part["k"], part["summary"]
        text += ["", f"## Inndeling i {k} klimasoner", "",
                 f"{int(summary.cells.sum()):_} ruter er klassifisert. ".replace("_", " ") +
                 "Sonenummer er sortert etter årsmiddeltemperatur innen denne inndelingen. "
                 "8 og 32 soner er uavhengige inndelinger, ikke et hierarki; "
                 "samme sonenummer betyr ikke samme klima i de to kartene.",
                 "", f"![Klimakart med {k} soner](klimasoner_{k}.png)", "",
                 "| Sone | Beskrivende navn | År °C | Sommer °C | Vinter °C | Amplitude °C | Nedbør mm/år | Frostdager/år |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for _, row in summary.iterrows():
            text.append(f"| {int(row.region_id)} | {zone_name(row)} | " + " | ".join(
                number(row[key]) for key in ("mean_temperature_c", "summer_temperature_c",
                                            "winter_temperature_c", "summer_winter_temperature_amplitude_c",
                                            "mean_annual_precipitation_mm", "frost_days_per_year")) + " |")
        for _, row in summary.iterrows():
            zone = int(row.region_id)
            annual = part["tables"]["aarsprofiler"].query("region_id == @zone").mean(numeric_only=True)
            decades = part["tables"]["tiaarsprofiler"].query("region_id == @zone").set_index("decade_start")
            delta_t = decades.loc[2016, "temperature_c"] - decades.loc[1996, "temperature_c"]
            delta_p = decades.loc[2016, "precipitation_mm"] - decades.loc[1996, "precipitation_mm"]
            text += ["", f"### Sone {zone}: {zone_name(row)}", "",
                     f"**Hovedtrekk:** {int(row.cells):_} ruter, tilsvarende "
                     f"{int(row.cells):_} km² ruteareal. ".replace("_", " ") +
                     "Årsmiddeltemperaturen er "
                     f"{number(row.mean_temperature_c)} °C, sommermiddelet "
                     f"{number(row.summer_temperature_c)} °C og vintermiddelet "
                     f"{number(row.winter_temperature_c)} °C. Sommer–vinter-amplituden er "
                     f"{number(row.summer_winter_temperature_amplitude_c)} °C. "
                     f"Sonen har i gjennomsnitt {number(row.frost_days_per_year)} frostdager "
                     f"og {number(row.mean_annual_precipitation_mm, 0)} mm nedbør per år.",
                     "",
                     f"**Perioder og kraftig nedbør:** Gjennomsnittet av hver rutes årlige "
                     f"lengste våtperiode er {number(annual.longest_wet_spell_days)} døgn; "
                     f"lengste tørrperiode {number(annual.longest_dry_spell_days)} døgn og "
                     f"lengste kuldeperiode {number(annual.longest_cold_spell_tg_lt0_days)} døgn. "
                     f"Gjennomsnittlig årlig maksimum for 3-døgnsnedbør er "
                     f"{number(annual.max_3day_precipitation_mm)} mm. Dette er gjennomsnitt "
                     "av lokale årlige maksimum, ikke maksimum av en utjevnet sonetidsserie.",
                     "",
                     f"**Representant:** Øst {int(row.representative_x)}, nord "
                     f"{int(row.representative_y)} (UTM 33N). Representantens sommer/vinter er "
                     f"{number(row.representative_summer_temperature_c)} / "
                     f"{number(row.representative_winter_temperature_c)} °C, amplituden "
                     f"{number(row.representative_summer_winter_temperature_amplitude_c)} °C, "
                     f"antall frostdager {number(row.representative_frost_days_per_year)} per år "
                     f"og årsnedbøren {number(annual.representative_precipitation_mm, 0)} mm. "
                     "Punktets faktiske verdier er altså ikke identiske med sonemiddelet.",
                     "",
                     f"**Hvor godt representeres gruppen?** Midlere daglig temperatur-RMSE "
                     f"mot punktet er {number(row.temperature_rmse_c_mean, 2)} °C "
                     f"(P95 {number(row.temperature_rmse_c_p95, 2)} °C). "
                     f"Nedbør-RMSE er {number(row.precipitation_rmse_mm_mean, 2)} mm/døgn "
                     f"(P95 {number(row.precipitation_rmse_mm_p95, 2)}). "
                     f"Rute og punkt er uenige om våt/tørr dag i gjennomsnitt "
                     f"{number(100 * row.wet_day_disagreement_mean)} % av døgnene.",
                     "",
                     f"**Forskjell mellom første og siste tiår:** "
                     f"{number(delta_t, 2)} °C i årsmiddeltemperatur og "
                     f"{number(delta_p, 0)} mm i årsnedbør "
                     "(2016–2025 minus 1996–2005).",
                     ]
    return "\n".join(text) + "\n"


def export(source: Path, output: Path, groups: list[int]) -> None:
    if output.exists():
        raise FileExistsError(f"Resultatmappen finnes allerede: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + "_arbeid_", dir=output.parent) as temp:
        work = Path(temp)
        package = work / "klimasoner_norge_1996_2025.gpkg"
        partitions = []
        used_fields = set()
        for k in groups:
            print(f"Lager polygoner og norske egenskaper: {k} klimasoner", flush=True)
            part = read_partition(source, k)
            part["zones"].to_file(package, layer=f"klimasoner_{k}", driver="GPKG")
            part["representatives"].to_file(package, layer=f"representanter_{k}", driver="GPKG")
            zones = gpd.read_file(package, layer=f"klimasoner_{k}")
            reps = gpd.read_file(package, layer=f"representanter_{k}")
            verify_partition(part, zones, reps)
            used_fields.update(part["summary"].columns)
            part["zones"].drop(columns="geometry").to_csv(
                work / f"soneoversikt_{k}.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
            with closing(sqlite3.connect(package)) as db:
                for name, frame in part["tables"].items():
                    used_fields.update(frame.columns)
                    norwegian = translate(frame)
                    norwegian.to_sql(f"{name}_{k}", db, index=False)
                    norwegian.to_csv(work / f"{name}_{k}.csv", index=False, sep=";",
                                     decimal=",", encoding="utf-8-sig")
            plot_map(part, work)
            partitions.append(part)
            print(f"Kontrollert {k} soner: nøyaktig rasterlikhet, areal og representanter", flush=True)
        dictionary = [dict(kildefelt=key, feltnavn=field_info(key)[0], enhet=field_info(key)[1],
                           forklaring=field_info(key)[2]) for key in sorted(used_fields)]
        for key in FIELDS:
            if key.startswith(("longest_", "growing_", "heating_", "freezing_", "days_tg_", "max_")):
                for source_key in (key, "representative_" + key):
                    if source_key not in used_fields:
                        dictionary.append(dict(kildefelt=source_key, feltnavn=field_info(source_key)[0],
                                               enhet=field_info(source_key)[1], forklaring=field_info(source_key)[2]))
        for name, unit, explanation in (
            ("sonenavn", "", "Beskrivende etikett etter oppgitte temperatur-, nedbør- og amplitudeterskler; ikke en offisiell klimaklasse."),
            ("kjennetegn", "", "Norsk oppsummering av gruppens temperatur, årstidskontrast, nedbør og frost."),
            ("representasjon", "", "Norsk oppsummering av avvik mot representantpunktet og hva avvikene betyr."),
            ("periode", "år", "Analyseperioden 1996–2025."),
            ("ruteareal_km2", "km²", "Antall ruter × 1 km² i kartplanet, ikke detaljert landareal."),
            ("kilde", "", "MET Norge, seNorge_2018. Se kildevilkår i rapporten og originalene."),
            ("rolle", "", "Representanten er en faktisk rute nær sentroiden i egenskapsrommet, ikke en værstasjon."),
        ):
            dictionary.append(dict(kildefelt="", feltnavn=name, enhet=unit, forklaring=explanation))
        guide = pd.DataFrame(dictionary).drop_duplicates("feltnavn")
        guide.to_csv(work / "feltforklaring.csv", sep=";", index=False, encoding="utf-8-sig")
        with closing(sqlite3.connect(package)) as db:
            guide.to_sql("feltforklaring", db, index=False)
            tables = ["feltforklaring"] + [
                f"{name}_{k}" for k in groups for name in ("maanedsprofiler", "aarsprofiler", "tiaarsprofiler")]
            for name in tables:
                db.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, description) "
                           "VALUES (?, 'attributes', ?, ?)", (name, name, "Norsk klimabeskrivelse og profiler"))
            db.commit()
        (work / "KLIMASONER_FORKLART.md").write_text(report(partitions), encoding="utf-8")
        metadata = dict(
            status="ferdig", periode=PERIOD, antall_soner=groups, kilde="MET Norge, seNorge_2018",
            kildeanalyse=str(source.resolve()), koordinatsystem="EPSG:32633",
            kontroll="Polygonene rasterisert tilbake gir nøyaktig de opprinnelige soneverdiene. "
                     "Areal er kontrollert mot antall ruter. Alle representanter ligger i riktig sone.",
            avgrensning="Ingen ny klynging. Norske beskrivelser og eksakt vektoreksport av eksisterende resultater.",
        )
        (work / "leveranse.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.move(str(work), str(output))
    print(f"Ferdig: {output}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "climate_regions_1996_2025")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "klimasoner_1996_2025_norsk")
    args = parser.parse_args(argv)
    export(args.source.resolve(), args.output_dir.resolve(), [8, 32])


if __name__ == "__main__":
    main()
