# seNorge-trening på server

Pakken inneholder ferdige egenskaper og etiketter for én million unike
1 km-rutepar, ikke Frost-treningspar. Perioden, dato- og avstandsutvalget
er beskrevet i `manifest.json`; de faktiske datoene ligger i `selected_dates.csv`.
Den nye 30-årspakken bruker døgnmiddeltemperatur 1996–2025 og 10 tilfeldig valgte
dager per måned per år: 3 600 datoer. Av disse brukes 8 per måned til lineær
tilpasning og 2 til evaluering (2 880/720 totalt), uten datooverlapp.
Etiketten er gjennomsnittet av prediksjons-RMSE i begge retninger.
Dette beskriver historisk likhet, ikke prediksjon i en fremtidig periode.
Den opprinnelige femårspakken bruker i stedet 2021–2025, alle dager og
kronologisk 80/20-deling. Ikke bland pakker eller gjenbruk resultatmapper.
Dette er ikke direkte temperatur-RMSE mellom rutene og ikke korrelasjon alene.

30-årspakken trekker avstand fra en halvnormalfordeling med skala 25 km,
tilfeldig retning og avstandsgrenser 1–100 km. Landdekning, nærmeste rute
og fjerning av duplikater endrer den teoretiske fordelingen. Se de faktisk
observerte avstandsintervallene og kvantilene i manifestets `sampling`.
Modellrapporten har også testfeil og antall testpar per avstandsintervall:
svært få lange par gir svakere dokumentasjon ved 90–100 km.
Tilfeldige enkeltdager gir ikke grunnlag for sammenhengende værperioder.

Bassengegenskapene er tilstedeværelse, antall, maksimal hel-bassengdybde
og sum av hele volumene for individuelle lake_id-bassenger med positivt
arealoverlapp med ruten. For hvert par brukes min, maks og absolutt forskjell,
slik at rekkefølgen på rutene ikke endrer prediksjonen.
Dette er potensielle terrengforsenkninger, ikke nødvendigvis vann.
En forsenknings volum kan gjentas i flere ruter og må ikke summeres til et
nasjonalt volum. Interpolerte 1 km-temperaturer kan glatte bort lokale effekter.

## 1. Koble til og overfør fra Windows PowerShell

Bruk VPN/institusjonens nett hvis maskinen krever det. Bytt vertsnavnet med
fullt DNS-navn eller eksisterende SSH-alias dersom kortnavnet ikke virker.

```powershell
ssh -p 1959 havfyh@its.npra.io
```
hav
Kontroller serverens vertsnøkkelfingeravtrykk mot en betrodd kilde første gang.
Ikke deaktiver host-key-kontroll, og ikke del passord/private nøkler.
På serveren:

```bash
mkdir -p ~/senorge-training
exit
```

Fra prosjektmappen på Windows:

```powershell
scp -P 1959 .\outputs\senorge_server_package.zip havfyh@its.npra.io:senorge-training/
ssh -p 1959 havfyh@its.npra.io
```

Konfigurer `my-server` som et SSH-alias for serverens faktiske vertsnavn og
port, eller erstatt aliaset med en host som virker fra ditt nettverk.
Ingen tilkobling eller
installasjon på serveren er utført automatisk av denne pakken.

For den nye pakken brukes `senorge_server_package_1996_2025.zip` i stedet
for `senorge_server_package.zip` i alle kommandoer. Bruk en egen servermappe,
for eksempel `~/senorge-training-30yr`, for å bevare femårskjøringen.

## Separat 95-prosentilforsøk

Dette forsøket erstatter hver par-RMSE-etikett med gjennomsnittet av
95-persentilen for absolutte OLS-residualer i hver retning på de samme
evalueringsdatoene. Datoer, par, egenskaper, romlig holdout og CPU/GPU-modeller
er uendret. Modellen predikerer dermed et empirisk parvis feilnivå; dette er
ikke et kalibrert 95 % prediksjonsintervall for en temperaturmåling.
Standardpakken og installert Station Lab-modell blir ikke overskrevet.

Lag en separat pakke lokalt fra den eksisterende 30-årsdatapakken. Først
koble til den eksterne disken; bytt `F:` med dens faktiske stasjonsbokstav.
Disken bør ha minst 100 GiB ledig. Koden kopierer de 25 årsfilene som allerede
ligger lokalt, og laster ned og verifiserer de manglende årene 2021–2025:

```powershell
$external = 'F:\senorge-p95'
$raw = Join-Path $external 'senorge_norway_1996_2025'
robocopy .\outputs\senorge_norway_1996_2025\netcdf (Join-Path $raw 'netcdf') /E /COPY:DAT /DCOPY:DAT /R:2 /W:2
if ($LASTEXITCODE -ge 8) { throw 'Copying seNorge archives failed' }
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u .\download_senorge_norway.py --annual --start 1996-01-01 --end 2025-12-31 --output-dir $raw
if ($LASTEXITCODE -ne 0) { throw 'Downloading/verifying the full seNorge archive set failed' }
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" .\prepare_senorge_server.py --source .\outputs\senorge_grid_pairs_1m_1996_2025_random --archive-dir (Join-Path $raw 'netcdf') --output .\outputs\senorge_server_package_1996_2025_p95 --target-metric absolute-error-p95
```

The prepared one-million-pair dataset remains in the project outputs folder.
Pakkelagingen beregner de nye etikettene fra de samme seNorge-filene og kan
bruke lang tid. Only the resulting training ZIP is uploaded to the server; the
raw NetCDF weather archives stay local. Keep the originals until verification
and package creation both complete.

If the external drive is unavailable but C: has at least 12 GiB free, download
just the missing years into the configured second archive folder instead:

```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u .\download_senorge_norway.py --annual --start 2021-01-01 --end 2025-12-31 --output-dir .\outputs\senorge_norway_2021_2025
```

In this case, package with the existing configured archive paths (omit
`--archive-dir`); the 30-year source configuration already points 2021–2025
to this folder:

```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" .\prepare_senorge_server.py --source .\outputs\senorge_grid_pairs_1m_1996_2025_random --output .\outputs\senorge_server_package_1996_2025_p95 --target-metric absolute-error-p95
```

```powershell
ssh -p 1959 havfyh@its.npra.io "mkdir -p ~/senorge-training-30yr-p95"
scp -P 1959 .\outputs\senorge_server_package_1996_2025_p95.zip havfyh@its.npra.io:senorge-training-30yr-p95/
ssh -p 1959 havfyh@its.npra.io
```

On the server, keep the p95 setup in its own folder. The following extracts
the ZIP; if it is already extracted, start with `python3 -m venv .venv`:

```bash
cd ~/senorge-training-30yr-p95
python3 -m zipfile -e senorge_server_package_1996_2025_p95.zip .
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze > installed_versions.txt
python -m unittest test_train_remote -q
```

Then run the GPU test and training steps below from this same directory.
Run one GPU at a time; the four unchanged CPU/GPU experiments
sammenlignes fortsatt mot holdout. Enhetlig RMSE/MAE i rapporten er feil i
prediksjonen av p95-etiketten, ikke dekning eller temperaturfeil.

## 2. Pakk ut og opprett separat miljø på Ubuntu

The following commands are for the standard RMSE package in
`~/senorge-training`. For p95, use the separate `~/senorge-training-30yr-p95`
commands above; do not switch back to `~/senorge-training` during its setup.

```bash
cd ~/senorge-training
python3 -m zipfile -e senorge_server_package.zip .
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze > installed_versions.txt
python -m unittest test_train_remote -q
```

Hvis `venv` mangler: be administrator installere `python3-venv`, eller bruk
et separat Conda-miljø med Python 3.10/3.11. Ikke endre systemets Python
eller NVIDIA-driver. Pakken har ingen GIS-avhengigheter.

XGBoost er bevisst låst til **2.1.4**. Ikke erstatt dette ukritisk med
nyeste XGBoost: nyere versjoner kan kreve nyere CUDA/driver enn maskinens
550.163.01 / CUDA 12.4. `nvidia-smi` viser driverens CUDA-støtte, ikke at
et separat CUDA-toolkit er installert. Standard Linux-pakken for XGBoost
2.1 har GPU-støtte; ikke installer `xgboost-cpu`.

Kontroller overføringen:

```bash
python -c 'import json,pathlib; from train_remote import sha256; m=json.loads(pathlib.Path("package_checksums.json").read_text()); assert all(sha256(pathlib.Path(k))==v for k,v in m.items()); print("Alle SHA256-kontroller bestod")'
nvidia-smi
python -c 'import xgboost; print(xgboost.__version__); print(xgboost.build_info())'
```

Sjekk lokal ressurs-/køpolicy før du starter. Ikke anta at en ledig GPU
er reservert til deg. Bruk jobbplanlegger dersom institusjonen krever det.

## 3. Test faktisk GPU-trening

```bash
python -c 'import numpy as np; from xgboost import XGBRegressor; from train_remote import verify_device; x=np.random.default_rng(42).normal(size=(1000,25)).astype("float32"); m=XGBRegressor(n_estimators=5,tree_method="hist",device="cuda:0"); m.fit(x,x[:,0]); print("GPU bekreftet:",verify_device(m,"xgboost","cuda:0"))'
```

Testen skal skrive `cuda:0`. Manglende GPU, feil bibliotek/driver eller
CPU-fallback skal undersøkes før full kjøring. Treneren avviser GPU-fallback;
en CPU-kjøring skal ikke fremstå som en vellykket GPU-test.
Denne pakken er testet lokalt på CPU; maskinens GPU er ikke fjernverifisert.

## 4. Start i bakgrunnen

Bruk én GPU og et moderat antall CPU-tråder først:

```bash
cd ~/senorge-training-30yr-p95
source .venv/bin/activate
mkdir -p results
nohup python -u train_remote.py --device cuda:0 --jobs 8 --output results > training.log 2>&1 < /dev/null &
echo $! > training.pid
cat training.pid
sleep 5
ps -p "$(cat training.pid)" -o pid,etime,cmd
tail -n 20 training.log
```

Du kan koble fra SSH. `nohup` beskytter mot vanlig terminalfrakobling,
ikke omstart, administratorstopp, jobbkøpolicy eller automatisk prosessrydding
ved utlogging på enkelte systemer. Bruk institusjonens køsystem ved behov.

```bash
tail -f training.log
cat results/status.json
watch -n 2 nvidia-smi
```

Ctrl+C ved loggvisning stopper bare visningen.
`kill "$(cat training.pid)"` stopper jobben; kontroller PID/kommando med `ps`
først for å unngå å stoppe en annen prosess etter at PID er gjenbrukt.
Avbrutte eksperimenter beregnes på nytt. Fullførte eksperimenter gjenbrukes
fra kontrollerte modellkvitteringer ved samme kommando. Fil-lås hindrer to
trenere mot samme resultatmappe. `summary.json` skrives først når alt er ferdig.

For GPU 1 brukes `--device cuda:1`. Ved `CUDA_VISIBLE_DEVICES=1` blir dette
kortet normalt synlig som `cuda:0` inne i prosessen. Ikke start to
fullstendige sammenligninger samtidig. To kort gir ikke automatisk 48 GB
sammenhengende GPU-minne, og fler-GPU-distribusjon inngår ikke her.

## 5. Hva trenes og sammenlignes?

Fire eksperimenter bruker samme faste romlige holdout:

1. ExtraTrees på CPU uten bassenger.
2. ExtraTrees på CPU med bassenger.
3. XGBoost på GPU uten bassenger.
4. XGBoost på GPU med bassenger.

Ingen treningspar har endepunkt i holdout-blokkene (100 km).
Par som krysser trenings-/testgrensen utelates fra valideringen.
Etter måling av valideringsresultatene tilpasses hver sluttmodell til alle par.
Sluttmodellens avvik på disse samme parene er derfor ikke en ny holdout-test.
Ingen automatisk hyperparametertilpasning eller early stopping bruker testen.

RMSE/MAE i rapporten måler hvor godt modellen forutsier parets seNorge-baserte
etikett, ikke selve dagstemperaturen mot målinger. For standardpakken er
etiketten RMSE; for p95-pakken er den gjennomsnittet av de retningsvise
95-persentilene av absolutte OLS-residualer.
Forbedring med bassenger dokumenteres bare dersom holdout-tallene støtter det.

`results/summary.json` inneholder sammenligningen. Hvert eksperiment har
en endelig `.joblib`-modell, holdout-prediksjoner i `.npz` og en kvittering.
XGBoost lagres også som `.ubj`. `grid_features.csv.gz` inneholder rutenes
egenskaper for senere prediksjon/integrasjon. Eksisterende Station Lab er
ikke koblet til denne nye modellen; dette krever kontrollert integrasjon.
Sammendragets `target_metric` angir om etiketten er RMSE eller p95.

## 6. Hent resultatene tilbake

På serveren når `results/status.json` sier `complete`:

```bash
cd ~/senorge-training-30yr-p95
tar -czf senorge_results_1996_2025_p95.tar.gz results installed_versions.txt training.log manifest.json
```

Fra Windows-prosjektmappen:

```powershell
scp -P 1959 havfyh@its.npra.io:senorge-training-30yr-p95/senorge_results_1996_2025_p95.tar.gz .\outputs\
```

Hold modellfilene sammen med miljøversjoner og manifest. Joblib-modeller
skal bare lastes fra betrodde kilder og kan kreve samme bibliotekversjoner.
Pakken inneholder kun eksplisitt utvalgte filer: ingen SSH-nøkler,
database, Frost-legitimasjon eller lokal prosjektkonfigurasjon.
Behold MET Norge/seNorge_2018-kreditering og originaldataenes bruksvilkår.

### Vis det separate P95-laget i Værnett

Pakk kun ut `manifest.json`, `grid_features.csv.gz` og den vinnende XGBoost-
modellens `.ubj` og `.json` (behold mappenavnet `results_jobs14`) under
`outputs/senorge_30yr_p95_overlay`. Start Værnett på nytt og bruk
**Værfenomen → Sammenlign temperaturmodeller** for å bytte mellom RMSE og
P95. Begge kartene bruker samme blå–røde logaritmiske fargeskala, 1 km-ruter
og referansestasjon per rute. Hvert kart har egne fargegrenser (0,5- og
99,5-persentilen av sine positive verdier), slik at rødt viser relativt
høy usikkerhet i hvert kart, ikke samme verdi i grader. Velg **Vanlig kart** for å gå tilbake til
scenariovisningen. Sammenligningen følger ikke scenarioredigeringer eller
vegfiltre. Ruter utenfor treningsavstanden vises ikke. P95 viser et
modellert feilnivå, ikke temperatur eller et validert 95 % prediksjonsintervall.
Velg **Forskjell · P95 − 1,96 × RMSE** for å se differansen i °C i hver rute.
Faktoren 1,96 gir en referanse basert på normalfordelte feil med forventning
null, ikke et validert 95 % prediksjonsintervall.
**Skjul farger under** beholder samme persentil når du bytter kart. Ved
90. persentil vises omtrent de høyeste 10 %; for differansen brukes
absoluttverdien, slik at store forskjeller i begge retninger kan vises.

## 10 millioner par: direkte temperaturforskjell og OLS

Bruk en separat pakke og resultatmappe; de installerte Værnett-modellene
endres ikke. De fire målene deler par, egenskaper og romlig holdout. Alle
etiketter bruker de samme historiske evalueringsdatoene. OLS-modellene
bruker de øvrige datoene til å tilpasse regresjonen i begge retninger.

| `--target` | Etikett (°C) |
|---|---|
| `direct-rmse` | Kvadratroten av gjennomsnittlig kvadrert forskjell mellom døgnmiddeltemperaturene. |
| `direct-p95` | 95-persentilen av den absolutte forskjellen mellom døgnmiddeltemperaturene. |
| `rmse` | Gjennomsnittet av de to retningsvise OLS-residualenes RMSE. |
| `absolute-error-p95` | Gjennomsnittet av de to retningsvise OLS-residualenes absolutte 95-persentiler. |

De direkte målene bruker temperaturen ved det andre stedet uendret. De
måler ikke Pearson-korrelasjon. P95 er en historisk griddet etikett, ikke
validert dekning for en temperaturmåling.

Fra prosjektmappen i PowerShell:

```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u .\prepare_senorge_multi_target.py --pairs 10000000 --cache .\outputs\senorge_multi_10m_cache --output .\outputs\senorge_multi_10m.zip
```

Temperaturmatrisen og ferdige etikettblokker beholdes for gjenopptakelse
med samme kommando. Ikke kjør to pakkebygg mot samme cache. Pakken har
én felles egenskapsmatrise; rå NetCDF-filer overføres ikke.

```powershell
ssh -p 1959 havfyh@its.npra.io "mkdir -p ~/senorge-training-10m"
scp -P 1959 .\outputs\senorge_multi_10m.zip havfyh@its.npra.io:senorge-training-10m/
```

På SSH-serveren, bruk det eksisterende p95-miljøet. Kjør først XGBoost
sekvensielt på én tildelt GPU, med og uten bassengegenskaper for hvert mål.
Bruk `--jobs 14` hvis serverens ressursregler tillater det; kjør første mål
for å måle tids- og minnebehov før resten. ExtraTrees er valgfritt og kan
kreve vesentlig mer minne og disk på 10 millioner par.

```bash
cd ~/senorge-training-10m
source ~/senorge-training-30yr-p95/.venv/bin/activate
python -m zipfile -e senorge_multi_10m.zip .
python -m unittest test_train_remote -q
nvidia-smi
python -u train_remote.py --target direct-rmse --backends xgboost --device cuda:0 --jobs 14 --output results/direct-rmse
```

Når første mål er ferdig, kjør:

```bash
for target in direct-p95 rmse absolute-error-p95; do
    python -u train_remote.py --target "$target" --backends xgboost --device cuda:0 --jobs 14 --output "results/$target" || break
done
```

Hvert mål får sin egen `summary.json`. Sammenlign valideringsfeil på samme
etikett når du vurderer en modell; RMSE og MAE i rapporten er feil i
prediksjonen av etiketten, ikke den direkte temperaturforskjellen.

Installer manifestet, `grid_features.csv.gz`, og `summary.json`,
`xgboost_with_basins.json` og `xgboost_with_basins.ubj` for hvert mål under
`outputs/senorge_results_10m`, med `results/<mål>/` bevart. Etter omstart
vises de fire modellene under **Værfenomen → Sammenlign temperaturmodeller
→ 10 millioner rutepar**. De bruker samme referansestasjon per 1 km-rute
som den eksisterende sammenligningen, ikke et nytt stasjonsutvalg.
**Skjul farger under** beholder valgt persentil når du bytter kart.
