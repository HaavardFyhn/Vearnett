# Expanded spatial-terrain footprints (v2)

This is a new experiment, not a replacement for the current Station Lab models.
Its target is historical seNorge 1 km representation-error RMSE in Celsius,
not independently calibrated station uncertainty or a temperature forecast.

## What changed

- **10,000 training / 1,000 validation / 1,000 test unique reference cells**.
  Selection fails explicitly if the requested counts are unavailable.
- New deterministic **150 km regional blocks, seed 314159**, stratified
  south-to-north. All reference centres are at least 41 km inside their block.
  The full 80 km terrain input, descriptor halo and 64 km output remain inside
  that block. Different partitions therefore share no scene support.
- References cycle through occupied **5 km strata** before reusing a stratum.
  References are unique but nearby scenes still overlap within each partition.
  See `geography_report.json` for actual region counts, output-cell reuse and
  geographic coverage. Millions of overlapping labels are not millions of
  independent geographic examples.
- Native 10 m terrain is summarized at **every available land-grid cell**:
  coverage, elevation standard deviation, relief, mean gradient magnitude,
  absolute east/north gradients, four quadrant height contrasts and low-terrain
  fraction. Height descriptors use kilometres, gradients use m/m; the low
  fraction counts terrain more than 10 m below its cell mean.
- These 11 spatial channels cover the full **80 x 80 cell input**, including
  the halo. A spatial convolutional encoder preserves their arrangement.
  Unlike v1, there is no globally pooled reference-only 10 m branch.
  Outside the land-grid atlas or where terrain is unavailable, descriptors are
  zero and their explicit coverage channel is zero. Missing is not flat terrain.
- The residual U-Net still sees 100 m terrain shape throughout the scene,
  the reference marker and signed position, and **31 reference-to-cell features**:
  the original 25 features plus reference and target absolute elevation,
  absolute ocean distance and absolute fjord distance. These six new values
  are in metres (elevation above sea level, not an absolute-value transform).
  Reference values are broadcast over valid output cells; target values vary
  across the map. Existing relative terrain and distance differences remain.
  It outputs six nonnegative 64 x 64 maps at 1 km:
  annual/winter/summer daily mean (`tg`) and daily minimum (`tn`) RMSE.
  Annual uses all 12 months; winter is DJF and summer JJA.

Shared atlases reconstruct scenes on demand. They avoid storing 12,000 copies
of overlapping 800 x 800 terrain patches and pair-feature maps. Preparation
temporarily retains a roughly 8.7 GiB sampled-weather atlas, reading the original
1996-2025 weather once. Labels are computed one reference at a time with bounded
moment arrays. Weather preparation checkpoints yearly, native terrain every
256 cells, and labels after each reference. A partially completed unit is
recomputed on restart. Never transfer an unfinished package.
Atomic JSON checkpoint replacement retries Windows access/sharing-lock errors
for up to about nine seconds, logging each retry. Persistent locks and other I/O
errors still stop preparation explicitly; the previous checkpoint remains intact.

Native fine descriptors summarize terrain shape; they are not a learned raw
10 m encoder and cannot turn 1 km seNorge into measured 10 m weather truth.
The map is a **64 km square**, about 45 km maximum corner distance, not a
100 km radius. Scene-safe partitions do not themselves prove cross-valley
generalization. The new split overlaps geography inspected in earlier
experiments, so it is not a newly untouched final test or directly comparable
to the old test scores.

## Prepare on Windows

Use the isolated environment; do not install PyTorch into the ArcGIS environment.
From the project directory:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u prepare_footprint_expanded.py
```

Default destination: `outputs\senorge_footprint_expanded`.
Original weather, v1 inputs/results and existing application models stay intact.
Preparation requires at least 20 GiB free and retains its own resumable cache.
The final ZIP is built with an explicit allowlist, CRC check and SHA-256 receipt.
Local source configuration/fingerprints are deliberately excluded from the ZIP.

### Upgrade an already completed package without repeating preparation

The absolute values already exist in `grid_table.npy`. This command verifies
the completed package, updates the portable code and manifest, and rebuilds
the same ZIP with all terrain, weather labels, reference locations and splits
unchanged:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u prepare_footprint_expanded.py --repackage
```

No download, terrain processing or label computation is performed. All 31
features use training-only normalization; the CNN and XGBoost receive the same
updated features. The manifest records the ordered feature names and architecture
`reference-resunet-spatial10-absolute-v2`. Older relative-only packages remain
readable, but old 25-input checkpoints cannot be resumed as 31-input models.
Retransfer the rebuilt ZIP if you uploaded the earlier version. Use a fresh
server directory/results output below; do not overwrite a running experiment.

Monitor:

```powershell
Get-Content .\outputs\senorge_footprint_expanded\status.json
Get-Content .\outputs\senorge_footprint_expanded\cache\fine_progress.json
Get-Content .\outputs\senorge_footprint_expanded\cache\weather_progress.json
Get-Content .\outputs\senorge_footprint_expanded\cache\label_progress.json
Get-Content .\outputs\senorge_footprint_expanded\cache\complete.json
```

Progress files appear when the corresponding stage starts. Transfer only after
`status.json` says `complete` and `cache\complete.json` exists.

```powershell
scp -P 1959 .\outputs\senorge_footprint_expanded.zip havfyh@its.npra.io:~/
ssh -p 1959 havfyh@its.npra.io
```

## Train on the GPU server

The existing server environment with torch 2.5.1+cu124 works; a new environment
can be installed explicitly as follows. No CUDA-unavailable CPU fallback occurs.

```bash
mkdir -p ~/senorge-footprint-v2-absolute
cd ~/senorge-footprint-v2-absolute
unzip -q ~/senorge_footprint_expanded.zip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m unittest test_footprint test_footprint_atlas -v
python -m pip freeze > installed_versions.txt
nvidia-smi

nohup python -u train_footprint.py --device cuda:0 --hours 40 \
  --epochs 300 --patience 25 --batch-size 2 --jobs 8 \
  --output footprint_v2_absolute_results \
  > footprint_v2_absolute_training.log 2>&1 < /dev/null &
echo $! > footprint_v2_absolute_training.pid
sleep 10
ps -p "$(cat footprint_v2_absolute_training.pid)" -o pid,etime,cmd
tail -n 25 footprint_v2_absolute_training.log
```

V2 trains the spatial-descriptor model (widths 16/24, three seeds), not the old
reference-only fine branch. Validation alone selects a model. Test is scored
once for the selected CNN, with per-regional-block metrics in addition to
pooled cell-weighted metrics. Identity pixels are excluded from losses/scores.
Feature normalization uses training supervised pixels only. The six XGBoost
baselines use the identical labels/splits, with at most 1,000,000 training and
100,000 validation pixels sampled deterministically for v2; all test pixels
are scored. This sampling is recorded in each baseline result.

The 40-hour limit is a **maximum CNN budget**, not a minimum runtime; early
stopping may finish sooner. Budget is checked between full epochs; normalization,
hash checks, final evaluation and baselines add time. Completed epochs preserve
optimizer/scheduler state. Restart with the identical command/output to resume
an interrupted epoch sequence. Completed trial receipts, including trials that
hit the budget, are not extended on restart. Do not select settings from test.

After completion:

```bash
cat footprint_v2_absolute_results/status.json
tail -n 30 footprint_v2_absolute_training.log
tar -czf footprint_v2_absolute_results.tar.gz footprint_v2_absolute_results \
  footprint_manifest.json installed_versions.txt footprint_v2_absolute_training.log
```

Back on Windows:

```powershell
scp -P 1959 havfyh@its.npra.io:senorge-footprint-v2-absolute/footprint_v2_absolute_results.tar.gz .\outputs\
```

Return the results for verification before app integration, especially winter
scores and the neighbouring-valley case. Lower average error alone does not
establish that a station represents the adjacent valley.

## Test the returned model in Station Lab at native 1 km

The returned absolute-input model can be installed as a **separate experimental
station-by-station footprint view**. Existing annual/seasonal app models and
national network scenarios are not replaced. Do not apply this finite scene
model as an unrestricted national pair predictor.

```powershell
.\.venv-seasonal-app\Scripts\python.exe install_footprint_station_model.py
.\.venv-seasonal-app\Scripts\python.exe -u station_app.py --footprint-only --cell-m 1000 --port 8914
```

Open `http://127.0.0.1:8914/footprint` (the command reports a different port
if 8914 is occupied). The installer verifies archive allowlists/checksums,
local dataset provenance, returned test labels/masks, six CNN and baseline
scores, validation-only model selection and CPU/GPU map parity. It refuses to
overwrite an existing installation at `outputs\senorge_footprint_station_model`.
Use the isolated environment, not the ArcGIS environment.

In a newly started full Station Lab, the **Footprint-CNN · 1 km** header link
opens the same view. `--footprint-only` avoids loading national scenario
surfaces and leaves an already-running app untouched.

- Choose a Frost station by ID/name or click the map to set a new reference.
  The default is SN56960 (HAUKEDAL) when available; also inspect SN56895.
- Choose any of the six annual/winter/summer `tg`/`tn` targets.
- Select **Les RMSE i 1 km-ruten** to inspect individual predicted cell values.
- Reference coordinates snap to the nearest stored seNorge 1 km land-cell
  centre. Absolute input values belong to this grid cell, not a separately
  measured sensor elevation.
- Predictions are fixed **64 x 64 cells at 1,000 m** in EPSG:25833, with
  reference at pixel(32,32). Web display uses nearest-neighbour reprojection,
  not bilinear smoothing or a 5 km prediction grid.
- Outside the output square or missing land-grid cells, the map is transparent
  and point inspection reports no prediction. A missing reference cell is
  rejected explicitly; no nearest-distant-land fallback is used.
- **Eksporter native 1 km GeoTIFF + metadata** downloads a ZIP with a64x64
  float32 GeoTIFF,1,000m resolution,EPSG:25833 and NoData=-9999, plus the
  reference, target, bounds, warnings and model/dataset provenance in JSON.

Returned absolute-input v2 test scores (same1,000 held-out reference maps,
3,493,981 nonidentity cells per target across four regions):

| Target | CNN RMSE (°C) | XGBoost RMSE (°C) | CNN bias (°C) |
|---|---:|---:|---:|
| Annual tg | 0.2472 | 0.2419 | +0.1260 |
| Annual tn | 0.3430 | 0.3428 | +0.1973 |
| Winter tg | 0.3650 | 0.3417 | +0.1657 |
| Winter tn | 0.4605 | 0.4330 | +0.2383 |
| Summer tg | 0.1666 | 0.1896 | +0.0340 |
| Summer tn | 0.2061 | 0.2441 | +0.1192 |

The selected model is `resunet_w24_spatial10_s42`. It improves both summer
targets relative to the matching baseline, but not annual or winter targets.
These new-region results are not directly comparable to earlier experiments.
The single-reference view shows a directional representation-error map, not
independent station accuracy. The Frost network view takes the minimum of
these finite maps; it is not a joint station-network assimilation.
The returned CPU verification receipt records three full maps checked against
GPU predictions, maximum difference0.00034833°C (acceptance threshold0.002°C).

### Frost network, station contribution and point ranking

Use **Kartvisning** to choose **Hele Frost-nettet**, **Valgt stasjons bidrag**
or **Dekningstap uten valgt stasjon**. Preparation starts on demand, reports
progress and resumes from atomic checkpoints. To prepare before opening the
viewer:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u station_footprint_network.py
```

Only supported Frost references participate; custom points do not. Each
station contributes its trained64x64km square at native1km, not a64km radius.
The union has NoData wherever no footprint covers a stored land cell.
All six targets are computed together; changing target requires no retraining.

At each cell, full-network RMSE is the best single station's predicted RMSE.
The selected station's removal contribution is the minimum without it minus
the full-network minimum. This is zero for redundant stations and tied
winners, including stations sharing a snapped reference cell. If it is the
only covering station, removal loses coverage: the numeric contribution is
NoData, not infinity or a fabricated Celsius value. A separate binary map
shows1where coverage would be lost and0where it remains, with NoData outside
the original network coverage.

Click a cell to rank covering stations by RMSE and inspect their removal
contributions. The list shows up to30best stations plus the selected station
if it covers the cell; click an entry to display that station's contribution
map. Network exports retain1000m EPSG:25833 cells and include coverage counts,
best-station indices and the Frost-ID index catalogue; reduction exports
also contain the coverage-loss raster. PNG previews may subsample the large
national display to2200pixels per dimension using nearest neighbour;
predictions, point inspections and GeoTIFF exports remain native1km.
All four footprint views share a blue-to-red heatmap and legend: low values
are blue and high values red. The binary loss map uses blue for retained
coverage and red for lost coverage. Background tiles are grayscale without
desaturating the heatmap or station markers. Numeric scales and exports
are unchanged.
