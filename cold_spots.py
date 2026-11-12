"""Finn mulige kuldehull fra lokale minimumstemperaturavvik, uten terreng eller klimasoner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import netCDF4
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from pyproj import CRS
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import shape

from download_nve_grid import write_json


ROOT = Path(__file__).resolve().parent
YEARS = list(range(1996, 2026))
RINGS = ((2, 5), (5, 10), (10, 20))
METRICS = (
    "gyldige", "sum_avvik", "kalde", "sterke", "sum_kulde", "maks_kulde",
    "lengste", "vinter_gyldige", "vinter_kalde", "sommer_gyldige", "sommer_kalde",
)


def square_sum(values: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return uniform_filter(values.astype(np.float64), size=size, mode="constant", cval=0) * size**2


def local_deviation(field: np.ndarray, inner: int, outer: int) -> tuple[np.ndarray, np.ndarray]:
    """Kvadratisk ring: inner < maks(|dx|, |dy|) <= outer kilometer."""
    if not 0 <= inner < outer:
        raise ValueError("Ugyldige ringradier")
    finite = np.isfinite(field)
    count = square_sum(finite, outer) - square_sum(finite, inner)
    total = square_sum(np.where(finite, field, 0), outer) - square_sum(np.where(finite, field, 0), inner)
    potential = (2 * outer + 1)**2 - (2 * inner + 1)**2
    support = (count >= max(20, potential * 0.5) - 1e-6) & finite
    reference = np.divide(total, count, out=np.full(field.shape, np.nan), where=support)
    return np.where(support, field - reference, np.nan), count / potential


class ReferenceRings:
    """Del vindussummer mellom ringene; gjenbruk dekning bare ved identisk datamaske."""

    def __init__(self) -> None:
        self.mask: np.ndarray | None = None
        self.counts: dict[int, np.ndarray] = {}

    def evaluate(self, field: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        finite = np.isfinite(field)
        radii = sorted({radius for ring in RINGS for radius in ring})
        if self.mask is None or not np.array_equal(finite, self.mask):
            self.mask = finite.copy()
            self.counts = {radius: square_sum(finite, radius) for radius in radii}
        sums = {radius: square_sum(np.where(finite, field, 0), radius) for radius in radii}
        results = []
        for inner, outer in RINGS:
            count = self.counts[outer] - self.counts[inner]
            potential = (2 * outer + 1)**2 - (2 * inner + 1)**2
            usable = finite & (count >= max(20, potential * 0.5) - 1e-6)
            reference = np.divide(sums[outer] - sums[inner], count,
                                  out=np.full(field.shape, np.nan), where=usable)
            results.append((np.where(usable, field - reference, np.nan), count / potential))
        return results


def accumulate(stats: np.ndarray, run: np.ndarray, anomaly: np.ndarray, month: int) -> None:
    valid = np.isfinite(anomaly)
    cold = valid & (anomaly <= -2)
    strong = valid & (anomaly <= -5)
    deficit = np.where(cold, -anomaly, 0)
    stats[0] += valid
    stats[1] += np.where(valid, anomaly, 0)
    stats[2] += cold
    stats[3] += strong
    stats[4] += deficit
    stats[5] = np.maximum(stats[5], np.where(valid, -anomaly, 0))
    run[:] = np.where(cold, run + 1, 0)
    stats[6] = np.maximum(stats[6], run)
    if month in (12, 1, 2):
        stats[7] += valid
        stats[8] += cold
    if month in (6, 7, 8):
        stats[9] += valid
        stats[10] += cold


def prepare(source: Path, polygon: Path, output: Path, years: list[int]) -> tuple[list[dict], dict, np.ndarray]:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    records = [r for r in manifest["archives"] if r["year"] in years]
    if [r["year"] for r in records] != years:
        raise ValueError("Mangler verifiserte årsarkiver")
    fingerprints = []
    for record in records:
        path = Path(record["netcdf"]).resolve()
        stat = path.stat()
        if stat.st_size != record["bytes"]:
            raise ValueError(f"Endret filstørrelse: {path}")
        fingerprints.append(dict(year=record["year"], path=str(path), bytes=stat.st_size,
                                 mtime_ns=stat.st_mtime_ns))
    land = gpd.read_file(polygon)
    if land.empty or land.crs is None or not land.is_valid.all():
        raise ValueError("Ugyldig norsk landavgrensning")
    with netCDF4.Dataset(records[0]["netcdf"]) as ds:
        xs, ys = np.asarray(ds["X"][:]), np.asarray(ds["Y"][:])
        if not (np.allclose(np.diff(xs), 1000, rtol=0, atol=1e-6) and
                np.allclose(np.diff(ys), -1000, rtol=0, atol=1e-6)):
            raise ValueError("Forventet nordvendt 1 km-raster")
        crs = rasterio.crs.CRS.from_string(ds[ds["tn"].grid_mapping].proj4)
        if crs.to_epsg() != 32633:
            raise ValueError("Forventet EPSG:32633")
        transform = from_origin(xs[0] - 500, ys[0] + 500, 1000, 1000)
        mask = rasterize(((g, 1) for g in land.to_crs(CRS.from_epsg(32633)).geometry), out_shape=(len(ys), len(xs)),
                         transform=transform, fill=0, dtype="uint8").astype(bool)
        ids = np.flatnonzero(mask.ravel())
        grid = dict(width=len(xs), height=len(ys), west=float(xs[0] - 500),
                    north=float(ys[0] + 500), crs="EPSG:32633", cell_m=1000)
    pstat = polygon.stat()
    config = dict(schema=1, years=years, sources=fingerprints, polygon=str(polygon.resolve()),
                  polygon_bytes=pstat.st_size, polygon_mtime_ns=pstat.st_mtime_ns,
                  rings=RINGS, grid=grid, min_ring_coverage=0.5, threshold_c=2, strong_threshold_c=5)
    config = json.loads(json.dumps(config))
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError("Cache tilhører andre kilder/innstillinger; bruk ny resultatmappe")
    write_json(path, config)
    np.save(cache / "norske_ruter.npy", ids)
    return records, grid, ids


def extract_year(record: dict, grid: dict, ids: np.ndarray, cache: Path) -> None:
    year = record["year"]
    path = cache / f"{year}.npz"
    if path.exists():
        with np.load(path) as saved:
            if saved["stats"].shape != (len(RINGS), len(METRICS), len(ids)):
                raise ValueError(f"Feil form på årsresultat {year}")
        print(f"CACHE {year}: ferdig årsberegning", flush=True)
        return
    if shutil.disk_usage(cache).free < 3 * 1024**3:
        raise OSError("Mindre enn 3 GiB ledig; originalene beholdes")
    print(f"ÅR {year}: leser døgnminimum og beregner tre lokale referanser", flush=True)
    stats = np.zeros((len(RINGS), len(METRICS), len(ids)), dtype=np.float64)
    runs = np.zeros((len(RINGS), len(ids)), dtype=np.int16)
    support_sum = np.zeros((len(RINGS), len(ids)), dtype=np.float64)
    references = ReferenceRings()
    with netCDF4.Dataset(record["netcdf"]) as ds:
        var = ds["tn"]
        if var.units != "Celsius" or var.dimensions != ("time", "Y", "X"):
            raise ValueError(f"{year}: uventet minimumstemperatur")
        if rasterio.crs.CRS.from_string(ds[var.grid_mapping].proj4).to_epsg() != 32633:
            raise ValueError(f"{year}: feil koordinatsystem")
        if not (np.allclose(ds["X"][:], grid["west"] + 500 + np.arange(grid["width"]) * 1000,
                            rtol=0, atol=1e-6) and
                np.allclose(ds["Y"][:], grid["north"] - 500 - np.arange(grid["height"]) * 1000,
                            rtol=0, atol=1e-6)):
            raise ValueError(f"{year}: feil koordinater")
        t = ds["time"]
        decoded = netCDF4.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
        dates = pd.DatetimeIndex([pd.Timestamp(x.year, x.month, x.day) for x in decoded])
        if not dates.equals(pd.date_range(f"{year}-01-01", f"{year}-12-31")):
            raise ValueError(f"{year}: manglende/dupliserte datoer")
        if any((x.hour, x.minute, x.second) != (6, 0, 0) for x in decoded):
            raise ValueError(f"{year}: uventet tidsstempel")
        for day, date in enumerate(dates):
            field = np.ma.masked_invalid(var[day]).filled(np.nan).astype(np.float64)
            for ring, (anomaly, support) in enumerate(references.evaluate(field)):
                values = anomaly.ravel()[ids]
                accumulate(stats[ring], runs[ring], values, date.month)
                support_sum[ring] += np.where(np.isfinite(values), support.ravel()[ids], 0)
            if (day + 1) % 60 == 0:
                print(f"{year}: {day + 1}/{len(dates)} døgn", flush=True)
    pending = cache / f"{year}.pending.npz"
    np.savez_compressed(pending, stats=stats, support=support_sum, days=len(dates))
    pending.replace(path)
    print(f"FERDIG ÅR {year}", flush=True)


def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)


def verify_year_samples(record: dict, grid: dict, ids: np.ndarray, cache: Path) -> dict:
    """Kontroller et helt års utvalgte ruter med direkte nabouttrekk uten filter."""
    with np.load(cache / f"{record['year']}.npz") as saved:
        expected = saved["stats"]
    rng = np.random.default_rng(1729)
    sample = np.unique(np.concatenate([
        rng.choice(len(ids), min(4, len(ids)), replace=False),
        [int(np.argmax(expected[0, 2]))],
    ]))
    result = np.zeros((len(RINGS), len(METRICS), len(sample)), dtype=float)
    runs = np.zeros((len(RINGS), len(sample)), dtype=np.int16)
    with netCDF4.Dataset(record["netcdf"]) as ds:
        t = ds["time"]
        decoded = netCDF4.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
        for day, timestamp in enumerate(decoded):
            field = np.ma.masked_invalid(ds["tn"][day]).filled(np.nan).astype(float)
            for ring, (inner, outer) in enumerate(RINGS):
                values = np.full(len(sample), np.nan)
                potential = (2 * outer + 1)**2 - (2 * inner + 1)**2
                for j, position in enumerate(sample):
                    row, col = divmod(int(ids[position]), grid["width"])
                    r0, r1 = max(0, row - outer), min(grid["height"], row + outer + 1)
                    c0, c1 = max(0, col - outer), min(grid["width"], col + outer + 1)
                    distance = np.maximum(np.abs(np.arange(r0, r1)[:, None] - row),
                                          np.abs(np.arange(c0, c1)[None, :] - col))
                    neighbors = field[r0:r1, c0:c1][(distance > inner) & (distance <= outer)]
                    finite = neighbors[np.isfinite(neighbors)]
                    if len(finite) >= max(20, potential * 0.5) and np.isfinite(field[row, col]):
                        values[j] = field[row, col] - finite.mean()
                accumulate(result[ring], runs[ring], values, timestamp.month)
    if not np.allclose(result, expected[:, :, sample], rtol=1e-8, atol=1e-7):
        raise ValueError("Direkte nabouttrekk samsvarer ikke med årsberegningen")
    return dict(year=record["year"], sample_cell_ids=ids[sample].tolist(),
                days=len(decoded), status="passed",
                method="Direkte nabouttrekk for alle døgn og tre ringer, uten uniform_filter.")


def combine(cache: Path, years: list[int]) -> tuple[list[dict], np.ndarray]:
    totals = None
    annual_frequency, annual_eligible, support = [], [], []
    days = 0
    decade_counts = {}
    for year in years:
        with np.load(cache / f"{year}.npz") as saved:
            stats = saved["stats"]
            n = int(saved["days"])
            if totals is None:
                totals = np.zeros_like(stats)
            for j in range(len(METRICS)):
                if j in (5, 6):
                    totals[:, j] = np.maximum(totals[:, j], stats[:, j])
                else:
                    totals[:, j] += stats[:, j]
            eligible = stats[:, 0] >= n * 0.9
            frequency = divide(stats[:, 2], stats[:, 0])
            annual_frequency.append(np.where(eligible, frequency, np.nan))
            annual_eligible.append(eligible)
            support.append(saved["support"])
            decade = 1996 + ((year - 1996) // 10) * 10
            if decade not in decade_counts:
                decade_counts[decade] = np.zeros((len(RINGS), 2, stats.shape[-1]))
            decade_counts[decade][:, 0] += stats[:, 2]
            decade_counts[decade][:, 1] += stats[:, 0]
            days += n
    frequencies = np.asarray(annual_frequency)
    eligible_count = np.asarray(annual_eligible).sum(axis=0)
    recurrent_years = np.sum(frequencies >= 0.05, axis=0)
    decade_freq = np.stack([divide(value[:, 0], value[:, 1]) for value in decade_counts.values()])
    results = []
    for ring in range(len(RINGS)):
        s = totals[ring]
        usable = (s[0] >= days * 0.9) & (eligible_count[ring] >= np.ceil(len(years) * 0.9))
        freq = divide(s[2], s[0])
        # Operational screening criteria, not statistical significance or proof of cold-air pooling.
        recurrent = usable & (freq >= 0.05)
        stable = recurrent & (recurrent_years[ring] >= np.ceil(len(years) * 0.8)) & (
            np.all(decade_freq[:, ring] >= 0.05, axis=0))
        classification = np.where(stable, 2, np.where(recurrent, 1, 0)).astype(np.uint8)
        values = dict(
            middelavvik_c=divide(s[1], s[0]), kalde_dogn_prosent=freq * 100,
            sterke_avvik_prosent=divide(s[3], s[0]) * 100,
            kuldestyrke_ved_hendelse_c=divide(s[4], s[2]),
            storste_kuldeavvik_c=s[5], lengste_kuldeperiode_dogn=s[6],
            vinter_kalde_dogn_prosent=divide(s[8], s[7]) * 100,
            sommer_kalde_dogn_prosent=divide(s[10], s[9]) * 100,
            gjentakende_aar=recurrent_years[ring].astype(float),
            gyldige_dogn=s[0], gyldige_aar=eligible_count[ring].astype(float),
            referansedekning_prosent=divide(np.sum(support, axis=0)[ring], s[0]) * 100,
        )
        # Annual variation, computed without treating missing years as zero.
        f = frequencies[:, ring]
        mean = divide(np.nansum(f, axis=0), eligible_count[ring])
        values["aarlig_frekvens_std_prosentpoeng"] = np.sqrt(divide(
            np.nansum((f - mean)**2, axis=0), eligible_count[ring])) * 100
        for index, decade in enumerate(decade_counts):
            values[f"kalde_dogn_{decade}_{decade+9}_prosent"] = decade_freq[index, ring] * 100
        results.append(dict(values=values, usable=usable, classification=classification))
    return results, frequencies


def write_raster(path: Path, grid: dict, ids: np.ndarray, values: np.ndarray,
                 valid: np.ndarray, categorical: bool = False) -> None:
    nodata = 255 if categorical else -9999
    array = np.full(grid["height"] * grid["width"], nodata, dtype="uint8" if categorical else "float32")
    keep = valid & np.isfinite(values)
    array[ids[keep]] = values[keep]
    with rasterio.open(path, "w", driver="GTiff", width=grid["width"], height=grid["height"],
                       count=1, dtype=array.dtype, crs=grid["crs"], nodata=nodata,
                       transform=from_origin(grid["west"], grid["north"], 1000, 1000),
                       compress="deflate") as target:
        target.write(array.reshape(grid["height"], grid["width"]), 1)
        target.update_tags(kilde="MET Norge, seNorge_2018", periode="1996-2025",
                           metode="Lokalt tn-avvik uten terreng eller klimasoner")


def export_results(output: Path, grid: dict, ids: np.ndarray, results: list[dict], years: list[int]) -> None:
    summaries = []
    for index, ((inner, outer), result) in enumerate(zip(RINGS, results)):
        suffix = f"ring_{inner}_{outer}km"
        for name, values in result["values"].items():
            write_raster(output / f"{name}_{suffix}.tif", grid, ids, values, result["usable"])
        write_raster(output / f"mulige_kuldehull_{suffix}.tif", grid, ids, result["classification"],
                     result["usable"], categorical=True)
        summaries.append(dict(ring=suffix, vurderbare_ruter=int(result["usable"].sum()),
                              gjentakende_ruter=int(np.sum(result["usable"] & (result["classification"] >= 1))),
                              stabile_ruter=int(np.sum(result["usable"] & (result["classification"] == 2)))))
        if index != 0:
            continue
        path = output / "mulige_kuldehull.gpkg"
        with rasterio.open(output / f"mulige_kuldehull_{suffix}.tif") as raster:
            classes = raster.read(1)
            components = [(shape(geometry), int(value)) for geometry, value in shapes(
                classes, mask=(classes > 0) & (classes < 255), transform=raster.transform, connectivity=4)]
            geometries, rows = [], []
            # Aggregate polygon metrics by rasterized component IDs in one pass.
            if components:
                component_map = rasterize(((g, i + 1) for i, (g, _) in enumerate(components)),
                                          out_shape=classes.shape, transform=raster.transform, dtype="int32")
                membership = component_map.ravel()[ids]
                counts = np.bincount(membership, minlength=len(components) + 1)
                metrics = {}
                for name in ("kalde_dogn_prosent", "sterke_avvik_prosent",
                             "kuldestyrke_ved_hendelse_c", "vinter_kalde_dogn_prosent",
                             "sommer_kalde_dogn_prosent", "gjentakende_aar",
                             "lengste_kuldeperiode_dogn", "referansedekning_prosent"):
                    values = result["values"][name]
                    finite = np.isfinite(values)
                    total = np.bincount(membership[finite], weights=values[finite], minlength=len(counts))
                    count = np.bincount(membership[finite], minlength=len(counts))
                    metrics[name] = divide(total, count)
                for i, (geometry, classification) in enumerate(components, 1):
                    geometries.append(geometry)
                    rows.append(dict(omraade_id=i, klasse=classification,
                                     vurdering="Stabilt lokalt kuldeavvik" if classification == 2 else "Gjentakende lokalt kuldeavvik",
                                     areal_km2=int(counts[i]),
                                     **{name: float(values[i]) for name, values in metrics.items()}))
                frame = gpd.GeoDataFrame(rows, geometry=geometries, crs=grid["crs"])
                frame.to_file(path, layer="mulige_kuldehull", driver="GPKG")
                restored = rasterize(((g, c) for g, c in zip(frame.geometry, frame.klasse)),
                                     out_shape=classes.shape, transform=raster.transform, fill=0, dtype="uint8")
                if not np.array_equal(restored, np.where(classes == 255, 0, classes)) or not frame.is_valid.all():
                    raise ValueError("GeoPackage samsvarer ikke med kandidatrasteret")
            else:
                gpd.GeoDataFrame({"omraade_id": pd.Series(dtype=int)}, geometry=[], crs=grid["crs"]).to_file(
                    path, layer="mulige_kuldehull", driver="GPKG", geometry_type="Polygon")
        plot(output, grid, ids, result)
    pd.DataFrame(summaries).to_csv(output / "oppsummering.csv", index=False, sep=";", encoding="utf-8-sig")
    (output / "LES_MEG.md").write_text(
        "# Mulige kuldehull fra temperaturdata\n\n"
        f"Analyseår: {years[0]}–{years[-1]}. Kilde: MET Norge, seNorge_2018. "
        "Dette er en uavhengig analyse av døgnminimum (`tn`, 18–18 UTC), uten terreng, høydejustering eller klimasoner. "
        "Et lokalt kuldeavvik er ikke bevis på kaldluftsopphopning, inversjon eller en faktisk frostlomme.\n\n"
        "## Sammenligning med omgivelsene\n\n"
        "For hvert døgn trekkes gjennomsnittet av tilgjengelige nabocellers minimumstemperatur fra rutens minimum. "
        "Negativt avvik betyr at ruten er kaldere. Referansen omfatter hele kildedomenet, også naboland, "
        "mens resultatene bare omfatter norske landruter. Hav og manglende data inngår ikke som nulltemperatur. "
        "Referansecellene kan ha minimum på ulike tidspunkt i samme døgn; dette er ikke en samtidig målt inversjon.\n\n"
        "Ringene er kvadratiske, definert ved inner < maks(|østforskyvning|, |nordforskyvning|) ≤ outer km: "
        "2–5, 5–10 og 10–20. Hjørnene ligger lenger unna enn den ytre radiusen i rett linje. "
        "Den sentrale blokken utelates. Hver referanse krever minst 50 % av ringens mulige celler og minst 20 celler. "
        "Kartet over referansedekning viser støtten der sammenligningen er gyldig. "
        "Kyst og datagrenser kan derfor ha færre vurderbare ruter.\n\n"
        "## Klasser – praktiske søkekriterier, ikke statistisk signifikans\n\n"
        "- 0: Ikke kandidat etter valgte terskler, men tilstrekkelig datadekning.\n"
        "- 1: Minst 2 °C kaldere enn ringreferansen på minst 5 % av gyldige døgn.\n"
        "- 2: Som klasse 1, og minst 5 % slike døgn i minst 80 % av analyseårene, "
        "samt minst 5 % i hver av de tre tiårsperiodene 1996–2005, 2006–2015 og 2016–2025.\n"
        "- NoData=255 i klassekartet: utenfor norsk land eller utilstrekkelig datagrunnlag.\n\n"
        "En vurderbar rute krever minst 90 % gyldige døgn totalt og minst 90 % av årene med "
        "minst 90 % gyldige døgn. Årsfrekvenser med svakere dekning utelates, ikke settes til null. "
        "Sterke avvik betyr minst 5 °C kaldere. Verken p-verdier eller formell signifikans er beregnet. "
        "Stabilitet mellom år er et gjentakelseskriterium; årene er ikke nødvendigvis uavhengige.\n\n"
        "## Bruk i ArcGIS og terrengsammenligning\n\n"
        "Start med `mulige_kuldehull.gpkg`, laget `mulige_kuldehull`, eller "
        "`mulige_kuldehull_ring_2_5km.tif`. GeoPackage inneholder sammenhengende kandidatpolygoner "
        "per klasse, med fire-nabosammenheng og gjennomsnittlige indikatorer per område. "
        "Bruk unike verdier på `klasse`. Klassegrensene skiller også klasse 1 fra klasse 2 innen samme dalføre. "
        "Ingen grenser er glattet; ett ruteareal er 1 km² i kartplanet. "
        "Alle rastrene har EPSG:32633, 1000 m oppløsning og original seNorge-justering. "
        "Sammenlign senere med terreng på samme koordinatsystem og bevar også de kontinuerlige avvikene, "
        "ikke bare kandidatklassene. Ingen topografisk forklaring er lagt inn på forhånd.\n\n"
        "## Indikatorer\n\n"
        "`middelavvik_c`: middel over alle gyldige døgn, negativt er kaldere. "
        "`kalde_dogn_prosent` / `sterke_avvik_prosent`: frekvens av avvik ≤−2/−5 °C. "
        "`kuldestyrke_ved_hendelse_c`: gjennomsnittlig positiv kuldeforskjell bare på døgn ≤−2 °C, "
        "NoData dersom ingen slike døgn. `storste_kuldeavvik_c`: største positive kuldeforskjell, "
        "0 dersom ingen negative avvik. "
        "`lengste_kuldeperiode_dogn`: lengste sammenhengende rekke med avvik ≤−2 °C innen ett kalenderår; "
        "rekker brytes ved manglende data og årsskiftet. Vinter er januar/februar/desember, sommer juni–august. "
        "Frekvenser for årstidene bruker bare gyldige døgn i den årstiden. "
        "`aarlig_frekvens_std_prosentpoeng`: spredning mellom gyldige års frekvenser, ikke en usikkerhetsmargin. "
        "`gjentakende_aar`: antall tilstrekkelig dekkede år med frekvens ≥5 %. "
        "GeoPackage-verdier er arealvektede rutemiddel av indikatorene, ikke temperaturavvik "
        "beregnet fra en egen områdetidsserie.\n\n"
        "## Begrensninger\n\n"
        "Høyere terreng kan være kaldere av andre årsaker enn kuldehull. "
        "1 km-interpolerte temperaturer kan glatte bort små søkk og uobserverte lokale inversjoner. "
        "Manglende signal utelukker derfor ikke et kuldehull. Relative kuldeavvik kan også forekomme "
        "ved temperatur over null; kartet er ikke et rent frostfarekart. Vind, skyer, nattlengde og "
        "stasjonstetthet er ikke kontrollert. Funnet bør tolkes som kandidater til videre terreng- "
        "og måleundersøkelse, ikke som dokumentasjon av en fysisk mekanisme.\n\n"
        "Behold MET Norge-kreditering og kildevilkår: "
        "https://www.met.no/en/free-meteorological-data/Licensing-and-crediting\n",
        encoding="utf-8")
    with (output / "LES_MEG.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Resultatoversikt\n\n"
                     "| Referansering | Vurderbare ruter | Kandidatruter (inkludert stabile) | Stabile ruter |\n"
                     "|---|---:|---:|---:|\n")
        for summary in summaries:
            handle.write(f"| {summary['ring']} | {summary['vurderbare_ruter']} | "
                         f"{summary['gjentakende_ruter']} | {summary['stabile_ruter']} |\n")
        handle.write("\nDette er antall 1 km-ruter, ikke antall påviste fysiske kuldehull. "
                     "Kandidatene inkluderer også områder som kan være kaldere på grunn av høyde.\n")
    write_json(output / "analysis.json", dict(status="complete", years=years, rings=RINGS,
               grid=grid, results=summaries, source="MET Norge seNorge_2018", terrain_used=False,
               climate_regions_used=False, formal_significance_test=False))


def plot(output: Path, grid: dict, ids: np.ndarray, result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    fig, axes = plt.subplots(1, 2, figsize=(15, 13))
    extent = (grid["west"], grid["west"] + grid["width"] * 1000,
              grid["north"] - grid["height"] * 1000, grid["north"])
    for index, (ax, values, title) in enumerate(zip(
        axes, (result["classification"], result["values"]["kalde_dogn_prosent"]),
        ("Mulige kuldehull – praktiske kandidatklasser", "Døgn minst 2 °C kaldere enn omgivelsene"),
    )):
        array = np.full(grid["height"] * grid["width"], np.nan)
        array[ids[result["usable"]]] = values[result["usable"]]
        options = dict(cmap="magma", vmin=0, vmax=30) if index else dict(
            cmap=ListedColormap(["#dedede", "#79b5e3", "#31226b"]),
            norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
        image = ax.imshow(array.reshape(grid["height"], grid["width"]), extent=extent, **options)
        ax.set_title(title, fontsize=11)
        bar = fig.colorbar(image, ax=ax, shrink=0.7, extend="max" if index else "neither")
        if index:
            bar.set_label("Andel gyldige døgn (%)")
        else:
            bar.set_ticks([0, 1, 2], labels=["Ikke kandidat", "Gjentakende", "Stabilt"])
    fig.suptitle("Lokale minimumstemperaturavvik 1996–2025 · kvadratisk referansering 2–5 km\n"
                 "Uten terreng/høydejustering – ikke bevis på kaldluftsopphopning")
    fig.tight_layout()
    fig.savefig(output / "mulige_kuldehull.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "senorge_norway_1996_2025")
    parser.add_argument("--polygon", type=Path, default=ROOT / "outputs" / "norway_polygon.gpkg")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mulige_kuldehull_1996_2025")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "analysis.json").exists():
        parser.error("Analysen er allerede ferdig; bruk ny resultatmappe")
    started = time.monotonic()
    records, grid, ids = prepare(args.source, args.polygon, args.output_dir, YEARS)
    for record in records:
        write_json(args.output_dir / "status.json", dict(status="running", year=record["year"]))
        extract_year(record, grid, ids, args.output_dir / "cache")
    print("KONTROLL: direkte nabouttrekk for et helt år og utvalgte ruter", flush=True)
    verification = verify_year_samples(records[0], grid, ids, args.output_dir / "cache")
    write_json(args.output_dir / "uavhengig_kontroll.json", verification)
    results, _ = combine(args.output_dir / "cache", YEARS)
    export_results(args.output_dir, grid, ids, results, YEARS)
    write_json(args.output_dir / "status.json", dict(status="complete",
               elapsed_minutes=(time.monotonic() - started) / 60))
    print(f"FERDIG: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
