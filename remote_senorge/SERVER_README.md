# seNorge-trening på mathmagic-land

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
ssh havfyh@mathmagic-land
```

Kontroller serverens vertsnøkkelfingeravtrykk mot en betrodd kilde første gang.
Ikke deaktiver host-key-kontroll, og ikke del passord/private nøkler.
På serveren:

```bash
mkdir -p ~/senorge-training
exit
```

Fra prosjektmappen på Windows:

```powershell
scp .\outputs\senorge_server_package.zip havfyh@mathmagic-land:senorge-training/
ssh havfyh@mathmagic-land
```

scp bruker SSH-port 22 som standard. For annen port, bruk `scp -P PORT`
og `ssh -p PORT`, eller konfigurer et SSH-alias. Ingen tilkobling eller
installasjon på serveren er utført automatisk av denne pakken.

For den nye pakken brukes `senorge_server_package_1996_2025.zip` i stedet
for `senorge_server_package.zip` i alle kommandoer. Bruk en egen servermappe,
for eksempel `~/senorge-training-30yr`, for å bevare femårskjøringen.

## 2. Pakk ut og opprett separat miljø på Ubuntu

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
source ~/senorge-training/.venv/bin/activate
cd ~/senorge-training
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
RMSE-etikett, ikke selve dagstemperaturen mot målinger.
Forbedring med bassenger dokumenteres bare dersom holdout-tallene støtter det.

`results/summary.json` inneholder sammenligningen. Hvert eksperiment har
en endelig `.joblib`-modell, holdout-prediksjoner i `.npz` og en kvittering.
XGBoost lagres også som `.ubj`. `grid_features.csv.gz` inneholder rutenes
egenskaper for senere prediksjon/integrasjon. Eksisterende Station Lab er
ikke koblet til denne nye modellen; dette krever kontrollert integrasjon.

## 6. Hent resultatene tilbake

På serveren når `results/status.json` sier `complete`:

```bash
cd ~/senorge-training
tar -czf senorge_results.tar.gz results installed_versions.txt training.log
```

Fra Windows-prosjektmappen:

```powershell
scp havfyh@mathmagic-land:senorge-training/senorge_results.tar.gz .\outputs\
```

Hold modellfilene sammen med miljøversjoner og manifest. Joblib-modeller
skal bare lastes fra betrodde kilder og kan kreve samme bibliotekversjoner.
Pakken inneholder kun eksplisitt utvalgte filer: ingen SSH-nøkler,
database, Frost-legitimasjon eller lokal prosjektkonfigurasjon.
Behold MET Norge/seNorge_2018-kreditering og originaldataenes bruksvilkår.
