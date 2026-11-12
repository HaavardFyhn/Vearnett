# Nattkjøring

## Vegbanetemperatur: Fem År
`road-temperature-priority` laster ned `mean(road_surface_temperature PT1M)` og
trener bare kontinuerlig RMSE (°C) på 5000 m, akkurat som vind. Minuttverdiene
lagres som ett døgnmiddel per stasjon i `frost_history.db`. Resultat:
`outputs/road_temperature_5y_5000m/`.
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u overnight_run.py --plan road-temperature-priority
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u precompute_frost.py --phenomenon road_temperature --kind continuous --cell-m 5000 2500 1000 --prepare-maps
```

## Vind: Fem År
`wind-priority` laster ned manglende femårs vindobservasjoner og trener deretter
bare vind-RMSE på 5000 m. Temperatur, luftfuktighet, nedbør og snø kjøres ikke på
nytt. Vind bruker Frost-elementet `wind_speed`, aggregert som døgnmiddel.

Offline-baselinen bruker minimum forventet feil fra én stasjon. Dette er
monotont når stasjoner legges til og samsvarer med den eksisterende
Værnett-definisjonen; det modellerer ikke en sammenslått flerstasjonsprognose.
Vindresultatet skrives til `outputs/wind_5y_5000m/`.

Kontroller køen uten nedlasting eller trening:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" overnight_run.py --plan wind-priority --dry-run
```

Start femårsjobben:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u overnight_run.py --plan wind-priority
```

Kontroller køen uten nedlasting eller trening:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" overnight_run.py --plan wind-priority --dry-run
```

Start femårsjobben:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u overnight_run.py --plan wind-priority
```

Bruk `--skip-download` seulement si vindobservasjonene allerede ligger i
`frost_history.db`. Planen er fast på 5000 m, stopper ved første feil og beholder
ferdige trinn. Den krever `FROST_CLIENT_ID` for vindnedlasting, eksisterende
5000 m-rutenett og tilstrekkelig ledig diskplass. Den legger ikke til de ennå
uferdige potensielle innsjø-egenskapene i terrengmodellen.

## Prioritert Temperatur og Luftfuktighet
Den prioriterte planen kjører femårs **kontinuerlig RMSE** i denne rekkefølgen:
1. Luftfuktighet, 5000 m.
2. Temperatur, 2500 m.
3. Luftfuktighet, 2500 m.
4. Temperatur, 1000 m.
5. Luftfuktighet, 1000 m.

Kontroller først uten å starte arbeid:
```powershell
python overnight_run.py --plan temperature-priority --dry-run
```
Start med det eksisterende ArcGIS-miljøet:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u overnight_run.py --plan temperature-priority
```
Planen henter manglende luftfuktighetsdata før første steg og manglende
temperaturdata før temperatur 2500 m. Hvert fenomen lastes ned bare én gang;
ferdige måneder gjenbrukes. `--skip-download` utelater begge nedlastingene.
`--years` (standard 5) og `--force` støttes. Planen har fast rekkefølge og brukes
ikke sammen med `--phenomenon` eller et endret rutenettutvalg.

Resultatmapper er `outputs/humidity_5y_5000m/`,
`outputs/temperature_5y_2500m/`, `outputs/humidity_5y_2500m/`,
`outputs/temperature_5y_1000m/` og `outputs/humidity_5y_1000m/`.
Ingen temperatur-5000 m-analyse, klasseanalyse, nedbør eller snø legges til denne
planen. Loggen heter `overnight_temperature-priority_YYYYMMDD_HHMMSS.log`.
Ved feil stoppes planen før senere steg. Allerede ferdige resultater beholdes.

Forutsetningene om `FROST_CLIENT_ID`, diskplass, strømtilkobling og kjøretid
beskrevet under gjelder også denne planen. Full analyse på 1000 m kan ta
betydelig mer enn én natt. Jobben startes ikke av en tørrkjøring.

## Luftfuktighet: Fem År
Luftfuktighet bruker Frost-elementet **`relative_humidity`**, døgnmiddel og
kontinuerlig RMSE. Feilen måles i **prosentpoeng relativ luftfuktighet**, ikke
prosentvis relativ feil. Klassegrenser for luftfuktighet er ikke definert;
temperatur-, nedbør- og snøklassene kan ikke brukes på luftfuktighet.

Bruk det eksisterende `arcgis-nvdb`-miljøet og kontroller køen først:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" overnight_run.py --phenomenon humidity --dry-run
```
Start når forutsetningene under er oppfylt:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u overnight_run.py --phenomenon humidity --years 5 --grid-spacing-m 5000 2500 1000
```

Den avgrensede køen gjør bare dette:
1. Syntakssjekker de relevante Python-filene.
2. Laster ned manglende luftfuktighetsmåneder og gjenopptar uferdige partier.
3. Bygger stasjonspar og feilmodell for 5000 m, uten kandidatanalyse.
4. Gjør deretter det samme for 2500 m.
5. Kjører til slutt 1000 m.

Rekkefølgen **5000 → 2500 → 1000 m** er standard for et valgt fenomen, også
uten `--grid-spacing-m`. Nedlastingen skjer bare én gang. Resultatene holdes
adskilt per oppløsning, også stasjonsvariabler, par og modeller.

Ingen andre værfenomener kjøres, og ingen nye terrengrutenett bygges. Fullstendige
databaserapporter over alle eksisterende observasjoner kjøres heller ikke i denne
køen, siden de kan være kostbare for en stor database. En feil stopper den
avgrensede køen før avhengige steg. Prosessen returnerer en feilkode ved feil.

## Før Du Starter
   `outputs/norway_polygon.gpkg` og alle valgte rutenett må finnes:
   `outputs/norway_grid_5000m.gpkg`, `outputs/norway_grid_2500m.gpkg` og
   `outputs/norway_grid_1000m.gpkg`. Hele utvalget kontrolleres før nedlasting.
   tilstedeværelse uten å skrive ut verdien:
   ```powershell
   [bool]$env:FROST_CLIENT_ID
   ```
   Dersom den mangler, setter du din egen klient-ID lokalt i terminalen. Verken
   tørrkjøringen eller testene gjennomfører autentisert nedlasting.
   midlertidige data, stasjonspar og modeller kan kreve mange GB. Programmet
   varsler under 30 GiB ledig; dette er et varsel, ikke en garanti for at 30 GiB
   er nok. Ved oppsettskontrollen var bare omtrent 19 GiB ledig.
   observasjoner/stasjoner. Det er ingen garanti for at hele jobben blir ferdig
   innen et bestemt antall timer.
   modell-/Frost-forhåndsberegninger kan konkurrere om RAM, CPU og disk.

## Gjenopptakelse og Omkjøring
Kjør samme kommando igjen etter avbrudd. Ferdige nedlastingsmåneder og partier
gjenbrukes. Uttømte nettverksforsøk gir feil og skal ikke merke et uferdig parti
eller en uferdig måned som komplett. Frost-svar uten passende tidsserie er et
gyldig tomt resultat.

Bruk bare eksisterende observasjoner:
```powershell
python -u overnight_run.py --phenomenon humidity --skip-download
```
Bygg analysen på nytt uten tvungen ny nedlasting:
```powershell
python -u overnight_run.py --phenomenon humidity --force
```
`--force` gjelder bare analysen. Bruk dette ved senere kjøringer dersom nye
observasjoner er lagt til og eksisterende par/modell skal oppdateres. Uten
`--force` gjenbrukes kompatible analysefiler. `--force-download` brukes aldri av
denne køen.

For bare én oppløsning, velg for eksempel `--grid-spacing-m 5000`. Et delutvalg
kan angis med `--grid-spacing-m 5000 2500`. Aliaset `--grid-spacing` støttes
også. Duplikater fjernes og rekkefølgen sorteres fra grovt til fint.

Plasseringsgevinst og kandidat–mål-analyse er fjernet fra pipeline-kjøringen.
Analysesteget avsluttes etter trening av modellen. Kartforberedelse kjøres separat.

Nedbør med 10-minutters fallback, ny trening og oppdaterte kart:
```powershell
python -u overnight_run.py --plan precipitation-priority
```
Køen laster bare 10-minutters observasjoner for stasjoner uten døgnserie i
perioden. Døgn med minst 90 % gyldige intervaller lagres (minst 130 av 144,
slutt kl. 06 UTC). Manglende intervaller regnes som null for godkjente døgn;
døgn under grensen utelates. Summen skaleres ikke opp. Endret regel utløser
ny behandling av tidligere fallback-måneder, uten tvungen nedlasting av døgnserier.
Deretter trenes både forekomst- og mengdemodellen på 5000 m-grunnlaget.
Femårskjøringen forbereder forekomstkart på 5000, 2500 og 1000 m for begge nett.

## Logg og Resultater
Loggen heter `overnight_humidity_YYYYMMDD_HHMMSS.log` i prosjektmappen.
Resultatene lagres i separate mapper:

Hver mappe inneholder sine egne filer:

Dette forhindrer at et finere steg overskriver eller gjenbruker analyser fra et
annet rutenett. Tidligere filer direkte i `outputs/` beholdes, men gjenbrukes
ikke automatisk av denne oppløsningsdelte køen. Ved manuell kjøring av `main.py`
kan samme isolering velges med `--output-dir`; uten argumentet beholdes gamle
filstier for kompatibilitet.

Analysen krever minst to stasjoner med tilstrekkelig antall døgn og gyldig
overlapp for trening/validering. Koden gir en tydelig feil dersom dataene ikke
oppfyller kravene. Trenings-/valideringsopplegget er det samme som i den
eksisterende kontinuerlige pipelinen; dette er ikke en ny uavhengig sertifisering
av luftfuktighetsmodellens kvalitet. Værnett finner ferdige
luftfuktighetsmodeller i oppløsningsmappene og viser kontinuerlig RMSE i
prosentpoeng. Eksakte grunnflater for observasjonsnettet beregnes separat av
appen; se appveiledningen i `README.md`.

Kjør isolerte tester uten HTTP eller produksjonsskriving:
```powershell
python -m unittest test_overnight_run -v
```

## Eksisterende Samlekø
`python -u overnight_run.py` uten `--phenomenon` eller `--plan` beholder den eksisterende
samlekøen med femårs temperatur-, snø- og nedbørskjøringer og databaserapporter.
Den inkluderer ikke luftfuktighet og fortsetter historisk etter feil i et steg.
Bruk `--phenomenon humidity` for den avgrensede, feilstoppende luftfuktighetskøen.
