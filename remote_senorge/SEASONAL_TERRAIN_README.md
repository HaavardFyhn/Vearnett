# Vinter/sommer og terreng-CNN

Dette er et separat eksperiment. Den fungerende Værnett-modellen endres
ikke, og nye resultater kobles ikke automatisk til appen.

## Lokal forberedelse og fremdrift

Fra prosjektmappen i Windows:

```powershell
python -u prepare_seasonal_terrain.py
```

Bruk prosjektets fungerende GIS-Python-miljø. Kommandoen kan gjenopptas
etter avbrudd uten å endre kilder eller innstillinger. Fullførte årsdata
og blokker med terrengvinduer gjenbrukes. Terrengforberedelsen kan ta timer;
hold Windows våken og tilkoblet strøm.

Dersom jobben allerede kjører i bakgrunnen, start ikke en ny. Følg den med:

```powershell
Get-Content .\outputs\seasonal_terrain_preparation.log -Tail 15 -Wait
```

Filen `outputs/senorge_seasonal_terrain/cache/patch_progress.json` viser
antall ferdige terrengruter. `status.json` i resultatmappen viser
running/failed/complete. ZIP-en publiseres først etter fullført pakking.

## Data og rettferdig sammenligning

- Samme opprinnelige million unike par, 1–100 km, fra 30-årsutvalget.
  Par uten komplette nok etiketter for alle fire mål fjernes.
- Vinter: desember/januar/februar. Sommer: juni/juli/august.
  Dette er kalenderbaserte årstidsutvalg, ikke sammenhengende vinterår.
- Fire mål: `winter_tg`, `summer_tg` (døgnmiddel), `winter_tn`, `summer_tn`
  (døgnminimum). Hvert mål bruker 900 datoer: 720 til lineær tilpasning,
  180 til evaluering. De opprinnelige tilfeldige 8/2-datoene per måned beholdes.
- Hver etikett er gjennomsnittet av lineær prediksjons-RMSE i begge
  retninger. Dette er ikke dagstemperaturen og ikke direkte korrelasjon.
- Tre kvadratiske, nordorienterte terrengvinduer per rute:
  64×64 piksler ved 50 m (3,2 km), 100 m (6,4 km), 500 m (32 km).
  DTM50 aggregeres med arealgjennomsnitt ved grovere oppløsning.
  Ingen 10 m-detaljer oppfinnes. Terrengformen er bevart ved disse
  oppløsningene, ikke som en tapsfri kopi av terrenget.
- Høyde lagres som float16 i kilometer relativt til endepunktets terrenghøyde.
  Gyldighetsmasken er et eget uint8-array. Manglende terreng er ikke havnivå.
  Hav, landegrenser og hull kan gi ufullstendige vinduer; nettverket ser masken.
- Geografisk test: 20 % av 100 km-blokkene med seed42. Validering:
  20 % av de øvrige blokkene. Resten er trening. Begge endepunkter må
  tilhøre samme delmengde, og alle endepunkter må ligge minst 16,5 km
  fra blokkgrensen. Dette utelukker overlappende terrengvinduer på tvers
  av delmengdene, også med en margin for rasterresampling.
- XGBoost og CNN vurderes på nøyaktig samme par og de samme fire målene.
  De nye testtallene er ikke direkte sammenlignbare med gamle testtall:
  både sesongmål og romlig buffer er forskjellige.
- Kun valideringsdelen brukes til early stopping. Testdelen brukes ikke
  til valg av epoke eller normalisering. Det lages ingen sluttmodell
  tilpasset alle testpar; beholdte modeller er de faktisk evaluerte modellene.

## Overfør fra Windows PowerShell

Vent til lokal forberedelse har skrevet `cache/complete.json` og ZIP-en er
ferdig. Bruk en ny servermappe; ikke overskriv tidligere treningsdata.

```powershell
ssh -p 1959 havfyh@its.npra.io "mkdir -p ~/senorge-seasonal-terrain"
scp -P 1959 .\outputs\senorge_seasonal_terrain.zip havfyh@its.npra.io:senorge-seasonal-terrain/
ssh -p 1959 havfyh@its.npra.io
```

Kontroller SSH-fingeravtrykk ved behov. Ikke del nøkler/passord.
Terrengpakken er mye større enn tabellpakken: flere GB.

## På Ubuntu-serveren

```bash
cd ~/senorge-seasonal-terrain
python3 -m zipfile -e senorge_seasonal_terrain.zip .
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip freeze > installed_versions.txt
python -m unittest test_seasonal_terrain -q
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(8,device="cuda:0"); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),x.sum().item())'
```

PyTorch 2.5.1 CUDA 12.4 brukes med den oppgitte driver550, ikke siste
tilfeldige PyTorch/CUDA-versjon. Toolkit-/driverinstallasjon kreves ikke
av denne veiledningen. Hvis GPU-testen feiler, stopp og undersøk feilen.
Ved manglende `venv`, be administrator hjelpe; ikke endre system-Python.
Bruk serverens køsystem hvis det er påkrevd.

```bash
nohup python -u train_seasonal_terrain.py --device cuda:0 --jobs 8 \
  --batch-size 128 --epochs 50 --patience 8 \
  > seasonal_training.log 2>&1 < /dev/null &
echo $! > seasonal_training.pid
sleep 10
ps -p "$(cat seasonal_training.pid)" -o pid,etime,cmd
tail -n 25 seasonal_training.log
```

GPU-minne og gjennomstrømning må måles på serveren. Ved minnefeil kan du
starte med `--batch-size 32` og en ny `--output seasonal_results_batch32`.
Bruk én GPU først; to kort gir ikke automatisk samlet minne.
Nohup tåler normalt SSH-frakobling, ikke omstart eller serverens prosessrydding.

## De tre eksperimentene

1. `xgboost`: fire separate modeller på de 25 tabellvariablene.
2. `terrain`: symmetrisk CNN med terrengvinduer og bare avstand samt
   absolutte øst-/nordforskjeller mellom endepunktene.
3. `combined`: samme CNN med alle 25 tabellvariabler i tillegg.

Hver skala har en egen liten CNN som deles mellom endepunktene.
Summen og absolutt forskjell av endepunktsrepresentasjonene kombineres.
Rekkefølgen på paret påvirker dermed ikke prediksjonen.
Nettverket har fire positive utganger og likevektet MSE over de fire målene.
Alle er i grader Celsius; dette er ikke nødvendigvis beste oppgavevekting.
XGBoost-prediksjoner klippes ved null før evaluering fordi målverdien er RMSE;
samme regel må brukes ved senere prediksjon fra de lagrede UBJ-modellene.

Programmet kontrollerer kontrollsummer, romlige buffere og GPU-tilgjengelighet.
Fullførte eksperimenter gjenbrukes med samme innstillinger og kontrollerte
resultatfiler. Et avbrutt CNN-eksperiment trenes på nytt fra begynnelsen;
den beste epoken er lagret, men optimizer/batchposisjon gjenopptas ikke.
`seasonal_results/status.json` viser running/failed/complete.
`summary.json` skrives først når alle valgte eksperimenter er fullført.

```bash
tail -f seasonal_training.log
cat seasonal_results/status.json
```

Hent tilbake resultatene (ingen terrengkopi nødvendig):

```bash
tar -czf seasonal_terrain_results.tar.gz seasonal_results seasonal_manifest.json \
  installed_versions.txt seasonal_training.log
```

Fra Windows-prosjektmappen:

```powershell
scp -P 1959 havfyh@its.npra.io:senorge-seasonal-terrain/seasonal_terrain_results.tar.gz .\outputs\
```

## Tolkning

Lavere test-RMSE betyr bedre prediksjon av sesongens seNorge-feiletikett,
ikke dokumentert temperaturprognose eller nøyaktighet mot uavhengige stasjoner.
Terrengdetaljer kan ikke gjenopprette lokale kuldehull som mangler i interpolerte
1 km-etiketter. Derfor er faktisk gevinst fra CNN et åpent spørsmål.
Se antall og feil per avstandsintervall; tomme intervaller har ingen testskår.
Nye arkitekturer bør ikke velges ved gjentatt tilpasning til dette testutvalget.
Krediter MET Norge/seNorge_2018 og Kartverket DTM50; behold kildenes bruksvilkår.

## Resultater i Værnett

Etter tilbakeføring kontrollerer `install_seasonal_station_model.py` manifest,
kontrollsummer, testpar, fasit og rapportert RMSE. Den kombinerte modellen er
valgt for alle fire sesongmål og tilbys som fire egne temperaturknapper.
Den gamle helårsmodellen overskrives ikke. Se prosjektets README, avsnittet
«Sesongmodeller og terrengbasert dyp læring», for isolert CPU-appmiljø,
forhåndsberegning og begrensninger. Serverens treningspakke og manifest
beholdes uendret som proveniens for eksperimentet.
