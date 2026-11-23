# Weather-station pipeline v3 — weather-station models

This version replaces correlation similarity with symmetric held-out prediction RMSE. Roads are not included.

## Temperature workflow
Værnett uses **direct P95** when the recalculated baselines are available. For example, 2 °C means
the estimated 95th percentile of the absolute difference in daily mean temperature is 2 °C,
without correcting the reading from the reference station. This is based on seNorge 1996–2025,
not observed coverage of a 95% prediction interval.
The standard Direct P95 uncertainty and baseline maps use a fixed color scale: blue at 0 °C,
white at 1 °C, and saturated red at 3 °C or above. The original values are still used for filtering and export.
Below the **Skjul farger under** slider, the app shows the cutoff in map units and the percentage of
finite grid cells strictly above it (respecting county and road-corridor selection for single layers).

To prepare the staged 500 m app with the existing 10-million-pair model and all four placement maps:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u .\prepare_direct_p95.py --data-root .\outputs\portable_build\Remote_500m\data --device cuda --placement-workers 8
```

Omit `--placement-workers` for baselines only, or use `--device cpu` without CUDA. These are
resumable national calculations. Restart the app once the Direct P95 baselines are published;
each placement map becomes available after its separate calculation. The previous RMSE files are
kept. For **Tomt nett**, the interactive **Endring i usikkerhet** view for Direct P95
uses a **chosen ±1 °C calculation reference**, not a requirement: add a station to see where its estimated 95% threshold
is strictly below 1 °C. This is not measured uncertainty without stations. The precomputed
**Plasseringsgevinst** map uses the reference recorded in its result file (1 °C
for the updated empty-network run). The legacy RMSE view
retains its 0.5 °C reference.

For a non-staged 1 km launch, prepare the project cache with
`python precompute_frost.py --phenomenon temperature --kind direct_p95 --cell-m 1000 --station-network both`.

To run the four Direct P95 placement maps in one pass on the existing SSH machine, build the bundle:

```powershell
.\.venv-seasonal-app\Scripts\python.exe .\prepare_placement_remote.py --direct-p95-stage .\outputs\portable_build\Remote_500m\data --output .\outputs\placement_direct_p95_ssh.zip
scp -P 1959 .\outputs\placement_direct_p95_ssh.zip havfyh@its.npra.io:~/vaernett_500m/
ssh -p 1959 havfyh@its.npra.io
```

In the SSH shell:

```bash
cd ~/vaernett_500m
unzip -n placement_direct_p95_ssh.zip -d placement_direct_p95
tmux new -s placement-p95
```

Inside that tmux session, run:

```bash
cd ~/vaernett_500m
.venv/bin/python -u placement_direct_p95/run_direct_p95_placement.py > placement_direct_p95/run.log 2>&1
```

This uses 14 CPU workers and one prediction per candidate–target pair for all four
networks. It verifies the existing terrain files before reusing them. Rerun the same
command in the same folder to resume; copying the old local single-network checkpoints
is not supported. The four NPZ and four GeoTIFF files are written under
`placement_direct_p95/stage/outputs` on the server. Copy the `placement_gain_500m_road200_direct_p95*`
results to `outputs\portable_build\Remote_500m\data\outputs` on Windows and restart Værnett.

If **Tomt nett** is already finished, upload the current `placement_gain.py` as
`placement_direct_p95/placement_gain_existing.py` alongside the original on SSH.
From `~/vaernett_500m`, calculate only SVV, SVV + MET, and all stations in one
pass using the staged inputs from the completed empty-network run:

```bash
.venv/bin/python -u placement_direct_p95/placement_gain_existing.py \
  --dataset temperature:direct_p95 --station-network existing-networks \
  --data-root placement_direct_p95/stage --work-dir placement_direct_p95/work_existing \
  --workers 14 --radius-km 100 --overwrite
```

The three existing networks use their own Direct P95 baselines, not the 1 °C
empty-network target. This leaves the empty map and the original checkpoints
untouched. Repeat the same command with the same script and work directory to resume.

Download observations → daily aggregation → station/grid terrain features → station pairs → chronological 80/20 station-to-station regressions → symmetric validation RMSE → ExtraTrees spatial RMSE model.
The pipeline no longer evaluates placement gain or generates candidate-ranking surfaces.

Station Lab's seNorge 30-year temperature model (and experimental seasonal models)
uses the nearest valid 1 km terrain cell for every existing, moved or proposed
station: coordinates, cell elevation, land category and terrain features all come
from that cell. Real station coordinates remain visible on the map. This avoids
mixing legacy 5 km station terrain with 1 km placement inputs; it does not change
the station-based Frost models, whose features must match their training.
Under `Værstasjoner`, `Vis beregningspunkter (debug)` shows a small hollow dot at
each calculation point and a thin dashed line from the displayed station.
The checkbox can hide these indicators. Moved/proposed stations also expose their
1 km calculation point for other weather models.
After changing this source-cell policy, old temperature surface caches are not reused.
Recalculate both national networks without retraining the seNorge model:
```bash
python precompute_frost.py --phenomenon temperature --kind senorge30 --cell-m 1000 2500 5000 --station-network both --prepare-maps
```
Restart the app afterwards; the source cells are always 1 km even at coarser display resolutions.

For the other five weather phenomena, rebuild training features, models and both
networks' 1 km maps with:
```bash
python retrain_weather_1km.py
```
Use `--dry-run` to inspect commands. The queue uses existing observations only;
it does not download data or calculate placement gain. New models are built under
`outputs/weather_1km/<phenomenon>/` with cell coordinates, elevation and terrain,
matching proposed stations. Each model is published only after both network maps
and PNGs finish. Existing app processes keep their loaded models; restarting while
training still uses the previous model for unpublished phenomena. Already
published models are skipped, not overwritten, when restarting this command.
Under `Værstasjoner`, offline calculation status shows the current step, phase,
completed work and a measured remaining-time estimate **for that phase**, not a
promise for the whole queue. Data loading/model fitting initially show no estimate;
an interrupted/stale phase is labelled explicitly. Progress is also printed in
the calculation log.

All known model stations in the chosen MET/Vegvær network stay visible regardless
of the selected weather sensors. The weather selector offers only temperature;
the station markers and popups still show all known sensors. A white overlay marks
stations without a selected sensor; these markers do not contribute to that layer.
Only the selected temperature profile is retained at startup. Other phenomena
contribute station and sensor metadata, not their weather models or map surfaces.
When using an older stage, metadata is extracted from its existing profiles
without loading the corresponding models.
New portable ZIPs contain the temperature model and compact metadata for the
other sensors, but omit their models and weather profiles. An existing stage and
previous ZIPs are not deleted; rebuild the portable ZIP to apply this change.
New stations and sensor edits offer all six sensor types, independently of
available weather models. Multiple sensors can be selected; sensors without a
weather layer are saved as station metadata and do not trigger model loading.
**Vegværstasjoner + MET-stasjoner** includes road-weather stations plus stations
whose recorded owner list contains exactly **MET.NO**, including co-owned stations.
Unknown ownership does not exclude a recognized road-weather station. Other
non-road stations are excluded. The option requires the owner catalog and is
preserved in saved scenarios; older scenarios retain their original network.
Without a prepared combined-network baseline cache, its surface is calculated
on demand. The SSH job below prepares a transferable cache to avoid repeating
this work locally.
The popup no longer offers the `Legg til sensor` and `Fjern sensor` buttons.
Existing proposals can still be edited under `Egne forslag`.
New-station popups also offer `Fjern stasjon` beside `Endre sensorer`.
Adding sensors keeps one marker in its original circle/MET or square/Vegvær shape,
showing both original and proposed sensors instead of stacking markers.
Clicking an upgraded marker selects its proposed sensors when they contribute
to a selected weather layer. `Endring i usikkerhet` then compares the scenario
with and without those proposed sensors, retaining the original station and
other proposals. With no active proposed sensor, selection uses the original
station as before.
A purple rim marks altered stations; stations without a selected sensor have no
visible rim or shadow. The popup labels added sensors
`Lagt til (forslag)` and allows editing/removing just those sensors. Dragging
a combined marker in `Flyttemodus` moves the whole station, including its original
and added sensors. Original observations and station metadata remain unchanged:
all moves and sensor removals apply only to the scenario.
`Gjenopprett stasjon` restores original sensors in an existing scenario.
Climate-zone legends and map popups show zone names without numeric prefixes.

### Roads lacking representation in a climate occurrence

The app's **Dekningsgrad** panel shows the saved status analysis for E/R/F
roads. Enable **Vis** and select any of the five coloured categories.
Select only **Svært interessant kandidatområde** to highlight the highest
category. Hover over a
category for its national road length and occurrence count. Click the map for
the status of its 250 m road cell; station markers still open their normal
popups. If the cell has no selected coverage category, the click uses the
normal map action instead, such as climate-zone information on Klimasoner.
Clicks on road features likewise fall back to their normal road popup.
The layer follows the county filter; category tooltips describe Norway.
This is a fixed analysis of the original 849 eligible stations, **not** a
live coverage calculation for the edited scenario. Missing a station inside
an occurrence is not the same as being far from a station.

The display uses `status.tif` generated alongside the analysis. It retains the
original 250 m road cells, not detailed road LineStrings; at overview scale
each image pixel shows the highest selected category within it. Zoom in to
see individual cells. Scenario JSON retains these display choices. HTML
exports include **Dekningsgrad** as a separately switchable overlay, with the
same legend font as Klimasoner. All 250 m road cells in the selected counties
and selected categories are embedded, including areas outside the current
viewport. With all counties selected this covers Norway. A raster overview
is used below zoom 10; zooming in shows full-detail cell polygons with status
popups, without Python. The initial layer visibility follows **Vis** in the app.
New portable builds copy the small status raster and summary automatically;
the existing portable executable/ZIP must be rebuilt to include this feature.

Run `python road_climate_coverage.py` to reproduce the supplied Dekningsgrad
notebook using the union of the **six full model station lists**, deduplicated
by Frost station ID and restricted to holders **STATENS VEGVESEN** or **MET.NO**.
It uses original station longitude/latitude, not model-cell centres, the SVV-only
profile, or unsaved app scenario edits. No extra operational/recency rule is
introduced beyond membership in those model lists.

Defaults:
- Inputs: `data/Dekningsgrad/klimasoner_v1_0.tif` and `vegnett_ERF_250m.tif`.
- Models/road lines: `outputs/portable_build/Remote_500m/data`.
- Output: `outputs/road_climate_coverage` (must not already exist).

Use `--inputs`, `--data-root` and `--output` to override these paths.
Owner metadata must already be downloaded with `python station_owners.py`.
The 250 m analysis grid is intentional even with the 500 m app: changing it
would change the notebook method. Climate is resampled by nearest neighbour;
occurrences are 8-connected components within each of the eight climate zones.
Only occurrences with roads receive IDs. The boundary uses eight neighbours
and a two-cell square dilation (500 m), **not station distance buffers**.
Relevant occurrences have at least 20 road cells. The exact score combines
coverage deficit (0–50), missing transition coverage (0–30), and the average-tie
percentile rank of road-cell count (0–20), then rounds to one decimal place.
The notebook's zero-global-station-density convention is retained.
“Unique” here means a separate geographic occurrence of a climate class;
there is no additional climate-rarity score.

To change the score classes, edit `SCORE_THRESHOLDS` and, if desired, `STATUS`
near the top of `road_climate_coverage.py`. `klassifiser_score` uses inclusive
upper limits, as in the notebook: a score equal to a limit stays in the lower
class. The four limits must be strictly increasing and between 0 and below 100.
These settings affect **new analyses only**, not the saved analysis already in
the app. For example, save a new result without overwriting the old one:

```powershell
.\.venv-seasonal-app\Scripts\python.exe road_climate_coverage.py --output .\outputs\road_climate_coverage_new
```

This repeats the national analysis and line export. To use the result in the
app, stop the app, back up or rename the existing `outputs\road_climate_coverage`
folder, then rename the new folder to `road_climate_coverage` and restart the
app. The new manifest records the limits and labels so the map and exported
lines stay consistent. Rebuild a portable executable and re-export HTML to
share the new classification; existing executables and HTML do not update.

`road_climate_coverage.gpkg` has two spatially indexed EPSG:25833 layers:
- `scored_roads`: all E/R/F road segments in relevant, scored 250 m road cells.
- `roads_without_station`: the subset whose **entire climate occurrence** has
  zero qualifying stations, not simply roads far from a station.

Both contain actual full-detail road **LineStrings**, not raster-cell outlines
or generated centre lines. Original road geometry is split at scored raster-cell
boundaries and occurrence boundaries; MultiLineStrings are exploded. Each
segment retains the original road identifiers, occurrence/climate IDs, station
counts, component scores, total `Kandidatscore`, class/status and `UtenStasjon`.
Road-cell counts are not road lengths. The reference raster determines the
scoring; the full `Vegnett_VegScore_full` network supplies output geometry.
Any mismatch in source dates can leave scored cells without intersecting lines.

Audit outputs include all input stations and assignments (`stations.csv`),
all road-bearing occurrences (`all_occurrences.csv`), the relevant scored
occurrences (`candidates.csv`), exact occurrence/transition rasters, and a
provenance manifest. Results are staged and published only on successful
completion, and existing result folders and original notebook files are preserved.
The GeoPackage can be loaded in GIS or used as a precomputed line layer in the
station app without repeating the national analysis.

### Save and share a scenario

Station popups in the app and HTML show owners from Frost's
`data[].stationHolders`. **Stasjonseiere** lists the number of unique active Frost
stations per owner in the current station selection, across sensors. Removed
stations and user-created proposals are excluded; moved stations keep their
owners. A jointly held station counts once for each owner, so totals can exceed
the number of stations. Missing holders are explicitly labelled **Ikke oppgitt**.

Run `python station_owners.py` with the configured Frost authentication to fetch
all Norwegian SensorSystem holders. This writes `data/station_owners.json` and
an Excel-friendly `data/station_owners.csv` count list for the entire downloaded
Norwegian Frost catalogue (including stations outside the app's selection).
The command replaces owner metadata atomically and does not change observation
history, model caches or saved weather surfaces. An empty or failed response
keeps previous metadata. Refreshing owners requires network access; viewing
cached owners and HTML snapshots does not. Portable builds include the metadata
when available, including builds that reuse a remote stage.

The header provides three save/open buttons:

- **Lagre kart som HTML** downloads an interactive, read-only snapshot of the
  current scenario. Open the HTML directly in a modern Chrome, Edge or Firefox
  browser, or send it to someone else. Weather images, station symbols, county
  boundaries and compressed vector data are embedded; no Python or Værnett
  server is required.
  Internet access is needed for the topographic background.
  The background has no layer-panel entry. Tile failures show an explicit message.
  Map tiles are not embedded.
  Recipients can pan, zoom, adjust overlay transparency, toggle saved layers and
  click stations for colour-coded sensors and scenario changes.
  Calculation points and move indicators follow station visibility and have no
  separate layer-panel entry, regardless of the app's debug checkbox.
  The complete road selection is saved, including links outside the current
  viewport, using the category, class, road-reference and county filters.
  The road visibility checkbox only controls initial visibility, not inclusion.
  Road links expose their saved attributes.
  The **Vis usikkerhet langs veg** checkbox switches both uncertainty and
  placement gain between selected counties and the road corridor, using the
  corridor distance set in the app when saving. Both extents are embedded even when the road corridor
  or road overlay is hidden. The weather corridor is independent of the road
  overlay's category, class and road-reference filters. The saved distance is
  shown as **Viser usikkerhet inntil … m fra veg** and cannot
  be changed without exporting again.
  **Usikkerhet** and **Grunnett** switch
  between the edited scenario and the original, unedited station network for
  temperature. Both use the app's identical uncertainty legend labels
  (**Mindre usikkert** / **Mer usikkert**) and colour ramp.
  The temperature legend title is simply **Temperatur**, without units;
  the reduction legend title is **Reduksjon**.
  **Reduksjon** shows all changes in the saved scenario relative to its original
  network, regardless of the selected station or **Alle endringer** button.
  It respects the selected county/road extent, station network and station filters;
  reduced uncertainty is blue and increased uncertainty is red.
  Terrain and **Lagret visning** are not exported. Saving from terrain views
  initially opens uncertainty instead.
  The uncertainty maps cover the selected counties without the app's
  minimum-RMSE or cold-air-basin-only display filters.
  **Plasseringsgevinst** embeds the existing precomputed placement map for the
  selected base network and selected counties; it does not change with proposed
  stations or the uncertainty road-corridor checkbox. If that network's placement
  result is unavailable, the export reports an error instead of using another network.
  **Usikkerhet**, **Grunnett**, **Reduksjon**, **Plasseringsgevinst**, **Klimasoner** and **Bare bakgrunnskart** share
  one exclusive radio selector. The last option hides the weather/climate raster;
  station, road and basin overlays can also be unchecked for background only.
  Changing the display extent does not enable a hidden raster.
  The climate overview includes its own legend.
  Clicking it shows the zone and whole-zone summary statistics. Zone identification
  uses the saved 1600-pixel-high overview, not the native raster; boundary cells
  can differ and statistics are not measurements at the clicked point.
  Climate uses the saved county filter and does not apply the road corridor filter.
  Recipients cannot recalculate or edit.
  Map clicks show a climate summary and the nearest visible station rather than
  exact coordinates. Full-detail cold-air basin polygons and depth/area/volume
  attributes are always embedded for the selected counties (or all Norway),
  even when hidden in the app, using the current minimum depth filter.
  Their initial visibility still follows the app's **Vis** checkbox.
  Roads and basin details share a canvas above weather images. Road strokes
  take click priority, while basin interiors remain clickable away from roads.
  Changing weather/climate layers cannot cover their click targets. County borders and
  calculation indicators do not intercept clicks.
  Moved station markers and their original-position markers retain the same
  station identity, sensors and Vegværstasjon status.
  Missing basin data causes an explicit export error, not an incomplete snapshot.
  To keep the browser responsive, vectors are decoded and rendered only for the
  current viewport at zoom level 10 or closer. A county-filtered national basin
  raster and any currently rendered roads provide a low-zoom preview.
  If a viewport exceeds 30,000
  features, a message requests a closer zoom; no features are dropped from the file.
  National exports can be large and take several minutes to prepare. The file
  uses the browser's built-in gzip decompression; older browsers are unsupported.
  Vector exports keep every geometry vertex, rounding geographic coordinates to
  seven decimal places (less than 1 cm displacement) to reduce file size.
  Wait for map layers to finish loading before saving.
- **Lagre scenario** downloads a versioned JSON file containing proposed stations,
  moved/removed stations, added/removed sensors, calculation settings, panel
  settings, selected map layer, station visibility and map view.
- **Åpne scenario** restores that JSON in Værnett after confirmation, replacing
  the current browser session's scenario. The matching weather models and terrain
  grid must be available. Station positions are checked against the saved terrain
  cells rather than trusting candidate indices from another installation.
  Invalid scenario files leave the existing calculation scenario unchanged.

JSON files contain scenario settings, not models or preprocessing data. HTML is
a frozen map, not an editable scenario; keep the JSON as well to continue working.
Both files include station locations and should only be shared as intended.

The white overlay alpha is set in `station_app.html` by
`--inactive-station-alpha:0.60`: 0 means no white overlay, 1 means fully white.
Edit this value and refresh the browser; no model/server restart is needed.
The final marker-settings `<style>` block in `station_app.html` also controls
`--station-size` (total size including border), `--station-border-width`,
`--station-border-style` (`solid`, `dashed`, `dotted`, `double`),
`--station-border-colour`, `--station-radius` (`50%` circle, `0` square),
`--station-shadow`, `--station-opacity`, `--station-outline` and
`--station-outline-offset`. Base values are in `:root`; the following rules
override them for Vegvær, manual, selected, altered and inactive stations.
Add any of these variables to a state rule to customise that state. Later rules
win if several states apply (inactive styling is last). Marker dimensions and
centred anchors follow these CSS settings, not the legacy Leaflet inline sizes.

## Setup
### Offline 500 m workflow on the SSH server

The 500 m workflow is opt-in: the working 1 km models, caches and default app
startup are preserved. `run_500m.py` rebuilds terrain from the original N100
land-cover, contour, height-point and maritime data, not by subdividing the 1 km
feature table. Existing/moved/manual station inputs use the same nearest 500 m
terrain cell. Five Frost weather models are retrained into `outputs/weather_500m`;
temperature reuses the fitted seNorge30 model with new 500 m source/target terrain.
Both all-station and Vegvær maps, road/grid links, and a 500 m portable data stage
are prepared.

This is finer **terrain and calculation spacing**, not new 500 m weather
observations. The seNorge temperature model retains its original 1 km weather
training, basin lookup and native-cell identity rule. Consequently two 500 m
points in one native weather cell may still have zero modelled temperature error.
Existing climate-zone and cold-basin geometries retain their native resolution.
The new grid/model combination is not an independent scientific validation.

No SSH password or key is bundled. Preparation does not start remote jobs.
On Windows, with observation downloads/writers stopped:

```powershell
.\.venv-seasonal-app\Scripts\python.exe prepare_remote_500m.py
& .\outputs\upload_500m.ps1
ssh -p 1959 havfyh@its.npra.io
```

The preparation hashes inputs and exports a small code ZIP with credential-free
configuration and pinned dependencies. Large inputs are uploaded in place with
`scp`, **not copied into another local archive**. The observation database alone
is about 49 GiB; check remote storage before uploading. Keep inputs unchanged
through preparation/upload. A nonempty SQLite WAL/journal blocks preparation:
stop writers and checkpoint the database before retrying; do not delete the WAL.
The remote runner verifies all input checksums before computing.
If the code ZIP already exists, use `prepare_remote_500m.py --force` to regenerate
it atomically; without this flag preparation stops before expensive hashing.
If PowerShell script execution is disabled, use the individual `ssh`/`scp`
commands from the generated upload script interactively, subject to your
organization's policy. Changing machine-wide execution policy is not required.
For a failed transfer, re-upload only the affected file or geodatabase rather
than the entire observation database. Re-extract the updated source ZIP with
`unzip -o remote_500m_code.zip`, then run `python run_500m.py --verify-only`.
Verification lists every missing file, size mismatch and checksum mismatch;
do not bypass it. A reset SSH connection does not explain the exact file
mismatch by itself.

On the server (Linux shell, Python 3.11):

```bash
cd ~/vaernett_500m
df -h .
free -h
nproc
unzip remote_500m_code.zip
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-500m.txt
python run_500m.py --jobs 2 --threads 4 --dry-run
nohup python -u run_500m.py --jobs 2 --threads 4 > run_500m.log 2>&1 < /dev/null &
tail -f run_500m.log
```

Allow substantial storage beyond the uploaded inputs for five models, caches and
the portable stage. Leave a safety margin; the exact size depends on station
availability. Start with two jobs and monitor RAM/storage before increasing
`--jobs`. `jobs * threads` must fit the available logical CPUs. These ExtraTrees
models use CPU training, not CUDA; having a GPU does not accelerate them without
changing the model family. Terrain runs first and alone. Independent weather
training runs in parallel; exact maps and roads then run sequentially to avoid
cache-lock conflicts and peak memory multiplication.

Logs and progress are under `outputs/weather_500m`. The identical command resumes
completed phases without retraining; interrupted training restarts only its
unfinished model. Inputs/code changes require a fresh remote work directory.
`--phase terrain|train|maps|roads|stage` allows separate invocations with prerequisite
checks. A single lock prevents competing workflow invocations. Models become
available only after both network maps succeed. For an interrupted incomplete
portable stage, choose a new `--stage-name` rather than reusing its partial folder.

After completion, return the **portable data stage** (no database required):

```bash
tar -czf remote_500m_stage.tar.gz -C outputs/portable_build/Remote_500m data
```

Back on Windows:

```powershell
scp -P 1959 havfyh@its.npra.io:~/vaernett_500m/remote_500m_stage.tar.gz .\outputs\
New-Item -ItemType Directory -Force .\outputs\portable_build\Remote_500m
tar -xzf .\outputs\remote_500m_stage.tar.gz -C .\outputs\portable_build\Remote_500m
.\.venv-portable-build\Scripts\python.exe build_portable.py --cell-m 500 --name Vaernett_500m --reuse-stage .\outputs\portable_build\Remote_500m\data
```

Create the extraction destination directory first. Use the same updated source
files for the final executable build. The portable launcher reads its bundled
resolution automatically. The 500 m stage includes a compact
`Vegnett_VegScore_full` source and portable road-link caches. Existing ZIPs and
running apps are not modified.

Extract the complete ZIP before launching `Vaernett.exe`; the adjacent `_internal`
folder contains required projection databases, models and runtime libraries.
Portable map caches fingerprint the executable rather than loose Python source
files, which PyInstaller stores inside the executable. Normal Python runs still
fingerprint the rendering source files.
First startup can take several minutes while preparing cold-air-basin layers.
Repack releases with `build_portable.py` or Python's `zipfile`, not .NET ZIP update
mode, which corrupted compressed empty entries in a sensor-fix archive. Verify
archive CRCs, complete Windows extraction (including empty-file sizes), and normal
executable startup before distributing a repacked ZIP.

To use the returned 500 m data in the current Python app as well (without copying
the stage into the main outputs folder), run:

```powershell
.\.venv-seasonal-app\Scripts\python.exe station_app.py --data-root .\outputs\portable_build\Remote_500m\data --port 8767
```

`--data-root` reads the resolution from the stage manifest, uses its terrain,
models and road caches, but serves the current project's HTML/JavaScript.
With the returned stage at `outputs/portable_build/Remote_500m/data`, plain
`python station_app.py` now selects that stage and its 500 m grid automatically.
Startup prints the selected data folder and resolution. Use
`python station_app.py --cell-m 1000` to explicitly use the project's 1 km inputs.
If no returned stage exists, the default remains 1 km. Explicit `--data-root`
takes precedence; bundled executables keep using their own data.
An occupied port selects the next free port; use the URL printed at startup.
Runtime content hashes do not require the remote-training package. After updating
the source to fix a failed executable smoke test, rerun the same build command
with `--reuse-stage`, not `--reuse-executable`; this rebuilds the executable
without retraining or transferring the stage again.
If Windows denies access when PyInstaller replaces files from a previous failed
build, use a fresh `--name` (for example `Vaernett_500m_ready`) with the same
`--reuse-stage` path. Do not remove the returned data stage.

The packager collects XGBoost runtime data and native libraries explicitly,
not every submodule. Scanning `xgboost.testing` with `--collect-all xgboost`
can abort PyInstaller with a missing `hypothesis` test dependency. Updating
the packager and rebuilding with `--reuse-stage` fixes this without installing
XGBoost's testing dependencies.

Road-source checksums use platform-independent filename ordering. Older Linux
and Windows stage fingerprints are both accepted only when they match the current
file contents. If an older Windows packager reports `Road inputs changed` after
returning a Linux stage, update `build_portable.py` and retry the same command;
no recalculation is needed for an ordering-only mismatch. Actual content or layer
changes still require rebuilding the remote stage.

`Værfenomen` is collapsed by default and expands using the catalogue arrow.
`Flyttemodus` under `Værstasjoner` is off by default; enable it to drag existing
stations and your own proposals. Proposal moves retain their sensors and list
order, snap calculations to the new 1 km cell and keep the displayed drop position.
Occupied proposal/network cells are rejected without changing the scenario.
Turning it off locks dragging without undoing station moves. Resetting
view defaults turns it off again. Station selection and sensor proposals remain
available while movement is locked.

1. Put `frost_history.db` beside `main.py`.
2. Put `norway_grid_coast_elevation.gpkg` in `outputs/`, or edit `pipeline_config.py`.
3. Set environment variable `FROST_CLIENT_ID` when downloading.

## Commands
Full temperature run:
```bash
python main.py --phenomenon temperature --years 1
```
Use existing observations and terrain:
```bash
python main.py --phenomenon temperature --years 1 --from-step information --skip-download
```
Rebuild cached pairs and model:
```bash
python main.py --phenomenon temperature --years 1 --from-step information --force
```

## Outputs
Names include phenomenon, period, and `prediction_rmse` to prevent cache mixing.
- Station terrain features, station-pair targets, model metadata and fitted model.

## Ten-minute precipitation fallback
Daily-reporting stations retain their native `sum(precipitation_amount P1D)` series.
Frost series availability is checked over the requested period; stations with any
daily series in that period are never requested for ten-minute observations.
For other stations, the primary `:0` / time-series `0` ten-minute precipitation
series is accumulated from the previous day at 06:00 UTC to the reporting day at
06:00 UTC, matching Frost's daily precipitation convention.
Only days with at least 90% of the 144 distinct ten-minute intervals are saved:
130 valid readings are accepted; 129 are rejected. Missing, negative,
nonfinite, very uncertain or erroneous readings are excluded; quality codes
0, 1, 2 and 4 are accepted. For accepted days, missing intervals count as zero;
the observed sum is not scaled up. Days below 90% remain missing. The full
06:00-to-06:00 period must have elapsed before a daily total can be published.
This zero-rain assumption can underestimate rainfall if missing intervals were wet.
Duplicate intervals are not summed twice; conflicting duplicates stop the batch.
Native daily observations take priority over derived totals.
Only daily totals are stored, with separate provenance and versioned per-station/month
resume status. A coverage-policy change reprocesses fallback observations, without
forcing native daily downloads. The existing daily-download status cannot skip the new fallback.
Updated observation markers invalidate cached training pairs and models.
The interactive app reuses completed observation snapshots only when their full
model, terrain, station-order and reporting-period signature matches a saved
surface. Downloads therefore do not force database scans or map recalculation
in the app. Offline precalculation still reads the updated observations.
An already running app keeps its loaded models; restart it after the update
queue finishes to load the new models and prepared maps.
Availability dates accept ISO timestamps both with and without fractional seconds,
so native Frost observations and derived daily precipitation can coexist during downloads.

The weather legend uses qualitative endpoints, `Mindre usikkert` (blue) and
`Mer usikkert` (red), without model metrics, units or numeric limits.
Terrain popups and station terrain comparisons show only elevation, distance
to fjord and distance to open sea. Road ranking rows appear on weather maps,
not climate-zone or terrain maps. The experimental Footprint/status header
items are omitted; road filters and map popups remain.
Road popups show a segmented bar for Vegbase, Eksponering, Kjøreforhold,
Naturfare, Skred and Værutsatt. Segment widths are relative shares of these six
raw subscores, not claimed additive contributions to the weighted VS_Total.
No absolute component values are displayed. Missing components are labelled;
zero components have no segment. Scores are fetched only for the clicked road,
without expanding the national road geometry caches. At overview zoom the
nearest road matching the active filters is inspected.

Download the five-year history, retrain occurrence and amount models, then prepare
the operational precipitation maps for all-MET and SVV networks at all resolutions:
```bash
python -u overnight_run.py --plan precipitation-priority
```
This queue stops on failure. Use `--dry-run` to inspect it without executing.

## Scope and caveats
Precipitation and snow have operational occurrence models; their continuous
amount/depth models remain experimental. Final scientific validation should hold
out complete station IDs from both sides of every pair. No road or traffic
weighting is included in training.

## Station-network maps and affected area
Re-evaluate the saved greedy network on the existing 1,000 m terrain grid:
```bash
python make_uncertainty_map_gif.py --cell-m 1000 --min-improvement 0.01
```
The station coordinates, order, and source features remain those selected on the
5,000 m grid. Only the evaluation targets change; this does not rerun station
selection or retrain the RMSE model. The finer-grid result is not expected to
equal the original checkpoint surface or be optimal on the finer grid.

The GIF highlights in green the cells whose predicted RMSE decreases by more
than `--min-improvement` degrees Celsius when each station is added, and marks
the new station with a star. Set the threshold to `0` to count every strict
improvement. Affected area is the improved-cell count times the squared grid
spacing: each 1,000 m cell contributes 1,000,000 m². Boundary cells count as full
squares, so this is a grid-area estimate, not exact clipped land area. The first
station establishes the baseline and is excluded from the incremental-area plot.

Outputs are `uncertainty_map_final_1000m.png`, `uncertainty_growth_1000m.gif`,
`station_affected_area_1000m.csv`, and `station_affected_area_1000m.png` in
`outputs/`. Replay surfaces use a unique directory in the system temporary
folder to limit RAM use and avoid Windows file-mapping locks between runs.
Once predictions complete, a uniquely named `uncertainty_surfaces_1000m_*.npy`
is copied to `outputs/` before rendering, so plotting failures do not require
another prediction run. The saved array uses about 442 MB for 341 stations.
The mapping is closed and the temporary directory removed on normal completion
or a Python exception. Use `--max-stations 2` for a short preview with separate output
names, or `--cell-m 5000` to reproduce the original grid and check the full
replay against the saved checkpoint.

### Replot without predictions
The affected-area versus station-rank chart uses a logarithmic Y-axis. If any
areas are zero, it uses a symmetric log scale with a linear region up to one
cell's area so zeros remain visible. The uncertainty map colour scale was
already logarithmic; this chart does not represent distance versus RMSE.

Redraw only the chart from the existing CSV:
```bash
python make_uncertainty_map_gif.py --cell-m 1000 --replot-only
```
Regenerate maps and the GIF from the existing complete replay array:
```bash
python make_uncertainty_map_gif.py --cell-m 1000 --reuse-surfaces outputs/uncertainty_surfaces_1000m.npy
```
Use the uniquely named saved `.npy` path printed by newer runs instead when
appropriate. Arrays must come from the same grid, station locations, and station
order. Dimensions, finite values, and non-increasing uncertainty are checked,
but an old array without metadata cannot prove its originating grid/model.
`--skip-gif` skips animation rendering, and `--min-improvement` can change the
affected-area threshold without predicting again. CSV-only replotting uses the
threshold already stored in that CSV. A preview array cannot reconstruct a full
animation; it contains only the preview stations. Without a complete saved array,
the CSV is sufficient for the area chart but not spatial maps or the GIF.

### ArcGIS Pro
Export from the saved predictions (requires Rasterio, available in the tested
`arcgis-nvdb` environment):
```bash
python make_uncertainty_map_gif.py --cell-m 1000 --reuse-surfaces outputs/uncertainty_surfaces_1000m.npy --export-arcgis --skip-gif
```
Add the following from `outputs/` using ArcGIS Pro's **Add Data**:
- `station_network_1000m_rank341_uncertainty.tif`: best-network predicted RMSE in degrees Celsius.
- `station_network_1000m_rank341_rmse_reduction.tif`: RMSE decrease from adding station 341.
- `station_network_1000m_rank341_affected.tif`: 1 for improved cells above the threshold, 0 otherwise.
- `station_network_1000m.gpkg`, layer `stations`: coordinates, rank, and affected area in square metres.

The rasters use EPSG:25833, exact target-cell centres and spacing, and NoData
outside evaluated cells. File rank suffixes reflect the selected station count;
use `--export-rank 50` to export uncertainty and improvement at station 50 instead
of the final station. The first station establishes the baseline, so its reduction
and affected rasters are zero. These are georeferenced data layers, not an ArcGIS
animation or a georeferenced GIF. In the station layer, a definition query such
as `rank <= 50` displays the network up to that rank.

## Værnett
Start den lokale nettapplikasjonen med ArcGIS-miljøet:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" station_app.py
```
Åpne nettadressen programmet skriver ut. Standardporten er 8765; dersom den er i
bruk, velges neste ledige port. Bruk også `arcgis-nvdb` som Python-tolk i VS Code
hvis editoren melder at Rasterio mangler. Flask, SciPy, Pillow, PyProj, Rasterio
og de eksisterende modellpakkene er testet i dette miljøet. Kartbibliotek, ikoner,
skrifter og OpenStreetMap-kartfliser krever internett. Observasjoner, modeller og
scenarier behandles lokalt og sendes ikke til kartleverandøren.

### Kuldehull I Kartet
**Kuldehull** viser terrengbassengene fra
`outputs/potential_lakes_norway_DTM10_sill20_tiled/potential_lakes.gpkg`.
Velg **Vis** for et eget kartlag. **Dypere enn** velger hele bassenger
med `max_depth_m` strengt større enn terskelen. Dette er DTM-dybde, ikke målt
temperatur eller bekreftede kaldluftsområder.
**Bare usikkerhet i kuldehull** klipper værkart og forhåndsvisning til de valgte
polygonene ved kartbildets pikseloppløsning. Filteret endrer ikke modellen,
stasjonsnettet, nøkkeltallene eller ArcGIS-eksporten. Vegkorridor og
Kuldehull-filter kan kombineres. Visningen lastes bare for gjeldende kartutsnitt.
Projeksjon, fem detaljnivåer og en landsoversikt lagres permanent i
`.kuldehull_cache` ved GeoPackage-filen. Forbered dette før appen startes:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" .\station_kuldehull.py
```
Appen laster alle detaljnivåer og bygger søkeindekser før HTTP-serveren starter.
Mangler cachen, bygges den under oppstart, ikke ved første kartklikk. Endret
GeoPackage gir ny cache automatisk. Nattkjøringen for seNorge inkluderer også
dette forberedelsestrinnet. Ukjente kartutsnitt må fortsatt tegnes ved visning,
men krever ingen ny reprojisering eller forenkling. På landsnivå brukes en
forhåndsberegnet 1 km oversikt; zoom inn for detaljerte bassenggrenser.
Originaldata endres ikke. **Ctrl + klikk** (eller Cmd + klikk) i kartet viser basseng-ID,
maks-/middeldybde, spillhøyde, areal og volum fra originalpolygonet og CSV-en.
Inspeksjonen følger valgt dybdegrense og legger ikke til en stasjon.

### Nett, Stasjonsvalg og Areal
- Velg **Frost-observasjoner** eller **Tomt nett** som grunnnett.
  Det automatiske foreslåtte stasjonsnettet er fjernet fra appens nettvelger.
- **Vindhastighet** bruker den lagrede femårsmodellen for kontinuerlig RMSE og vises i
  m/s. Kjør den eksakte Frost-forhåndsberegningen for vind separat før dette nettet
  kan vises eller redigeres.
- Klikk en Frost-stasjon eller et eget forslag i kartet/listen. **Reduksjon** viser den
  valgte sammenligningen. Frost-markører viser også ID, navn og datoperiode.
- For Frost-nettet slår knappen **Alle endringer** mellom to sammenligninger:
  - Når knappen er aktiv, sammenlignes hele scenariet med opprinnelig
    **Grunnett**, inkludert alle flyttinger, fjerninger og egne forslag.
    Valg av stasjon endrer ikke denne sammenligningen.
  - Når knappen er av (standard), vises bare den valgte stasjonens effekt, med alle andre
    scenarioendringer beholdt. For en uendret stasjon sammenlignes nettet med/uten
    stasjonen. For en flyttet stasjon sammenlignes dens nye plassering med dens
    opprinnelige plassering. For en fjernet, grå stasjon vises økningen i
    usikkerhet som fjerningen medfører, sammenlignet med å ha stasjonen tilbake
    på opprinnelig plass. Alle andre endringer beholdes i begge nett.
    Velger du en annen stasjon, forsvinner den forrige stasjonens effekt.
  Knappen endrer ikke stasjonsplasseringer eller valgt stasjon.
- Bidraget er forskjellen mellom dagens nett **med og uten den valgte stasjonen**.
  Klikk på stasjon 5 i et nett med 12 stasjoner sammenligner altså alle 12 med de
  øvrige 11, ikke bare historisk forbedring da stasjon 5 ble satt inn.
  Dette gjelder uendrede stasjoner når **Alle endringer** er av, og stasjoner i foreslåtte nett.
  Ved sammenligning mot grunnnett eller ved valgt flytting viser **Reduksjon**
  økt usikkerhet i rødt og redusert usikkerhet i blått.
  Fjerning av en stasjon kan bare øke eller bevare usikkerheten; økningen vises i rødt.
  Positive reduksjonsverdier betyr forbedring; negative verdier betyr økt usikkerhet.
  Eksport bruker aktiv sammenligning. Peke-forhåndsvisning og nøkkeltallpanelet er fjernet.
- API-ets påvirkede areal teller ruter der forskjellen er større enn standardterskelen.
  Ved sammenligning mot grunnnett eller valgt flytting telles begge retninger
  (absolutt endring over terskelen);
  rødt viser økning og blått viser reduksjon.
  Arealverdier er i **km²**: hver rute teller 1 km² ved 1000 m, 6,25 km² ved 2500 m,
  og 25 km² ved 5000 m. Rutene er ikke klippet mot kystlinjen.
- Dersom nettet bare har én stasjon, finnes det ikke noe nett uten den valgte
  stasjonen å sammenligne med. Bidrag og påvirket areal vises derfor som «—».
- Klikk på et ledig kartpunkt for å legge til et eget forslag. Angre, fjern
  enkeltforslag eller fjern alle egne forslag. Bytte av grunnnett eller antall
  foreslåtte stasjoner fjerner egne forslag etter bekreftelse.
- **Alt + klikk** i kartet (også på terrenglag eller veglenker) viser hvilke
  aktive stasjoner som passer best for stedet du klikket på.
  Dette endrer ikke scenariet. Flyttede og egne stasjoner tas med; fjernede
  stasjoner utelates. For standardmodellen viser popupen et anslått 95 %-intervall
  (± °C rundt stasjonens døgnmiddel), samt avstand til beste stasjon og
  avviksgrensene for de andre stasjonene i listen.
  Beste stasjon merkes **1** med gul ring, og nest beste merkes **2** med lilla ring.
  Navn og stiplede linjer viser forbindelsene til stedet; kartet viser alle tre punktene.
  **Sammenlign terreng** åpner høyde, avstand til kyst/fjord, relieff,
  terrengruhet og TPI for stedet, beste og nest beste stasjon side om side.
  Dette er geografisk kontekst, ikke en forklaring av hvor mye hver variabel
  bidro til den beregnede feilen. Ved lik feil kan flere stasjoner være like gode.
  Dra popupen i overskriften for å flytte den uten å flytte kartet eller
  inspeksjonspunktene. Overskriften kan også fokuseres og flyttes med piltastene.
  Utvidelse av terrengtabellen beholder boksens valgte plassering.
  Tekniske modellnotater er fjernet fra katalogen; modellmetadata beholdes i API og eksport.
  Sammenligningen gjelder gjeldende scenario og værmodell, uavhengig av valgt
  sammenligning. Det er beste enkeltstasjon, ikke nødvendigvis den nærmeste.
- I et ferdig beregnet Frost-nett kan du dra en Frost-markør til en annen terrengrute,
  eller velge stasjonen og fjerne den fra scenariet. Dette endrer ikke Frost-databasen.
  Du kan flytte eller fjerne flere Frost-stasjoner. Usikkerhetsflaten beregnes på nytt
  eksakt for rutene der de redigerte stasjonene var best. En enkelt stasjon kan
  gjenopprettes fra popupen. Tilbakestillingsknappen øverst nullstiller hele scenariet.
  Flyttede stasjoner har en grå markør på opprinnelig plass og en stiplet pil
  mot den nye plasseringen. Begge markørene velger samme stasjon; bare den nye
  kan dras. Ved gjentatt flytting peker pilen alltid fra opprinnelig til nyeste
  plassering. **Skjul stasjoner** skjuler også gamle plasseringer og piler.
  Fjernede stasjoner blir stående som grå, klikkbare markører på sin opprinnelige plass.
  De teller ikke i det aktive nettet. I **Alle endringer** vises endringen fra
  grunnnettet; når **Alle endringer** er av, viser en valgt grå markør usikkerhetsøkningen ved fjerning.

### Hva Usikkerheten Betyr
For hver rute brukes **laveste forventede feil fra én stasjon**:

$$E(r)=\min_{s\in S}\widehat{e}(s,r).$$

Dette gjelder både Værnettens eksakte Frost-grunnflate og den frakoblede
`information_gain.py`-analysen: per rute velges den laveste forventede feilen
blant tilgjengelige enkeltstasjoner. Det er ikke en sammenslått
flerstasjonsværprediksjon.

Stasjoner kombineres ikke i en felles prediksjon. Det er ikke kriging, et
flerstasjonsregresjonsnett eller en modell som lar flere svake stasjoner sammen
gi et bedre resultat enn den beste enkeltstasjonen. «Usikkerhet» er modellert
prediksjonsfeil, ikke en observert feil eller et konfidensintervall. Gjennomsnittet
er et uvektet gjennomsnitt over evalueringsrutene.

Frost-nettet bruker **alle kvalifiserte observasjonsstasjoner**, uten avstandsgrense
eller utvalg av nærmeste stasjoner. Tidligere tilnærmede flater brukes ikke av denne
versjonen. Argumentet `--frost-neighbors 0` aksepteres for kompatibilitet, men
andre verdier avvises. Foreslått nett og egne forslag evalueres også mot alle
mål-ruter med den valgte værmodellen.

Usikkerhetskartet bruker blått for lav feil og rødt for høy feil. Temperaturens
logaritmiske fargeskala bruker 0,5- og 99,5-persentilen av positive verdier i den
lagrede nasjonale **SVV-grunnflaten** i ruter som berører en 200 m korridor
rundt vegnettet uten kommunale veger. Nullverdier vises under nedre fargegrense.
Dette bruker alle veger i datasettet uavhengig av vegpanelets filtre.
Endre `ROAD_COLOUR_CALIBRATION_ENABLED` og
`ROAD_COLOUR_CALIBRATION_DISTANCE_M` øverst i `station_app.py` for å slå
vegkalibreringen av eller på og endre dens avstand. Start appen på nytt etterpå.
Når kalibreringen er av, brukes SVV-grunnflatens nasjonale fargegrenser.
Vegbanetemperatur bruker temperaturmodellens grenser ved samme oppløsning, slik
at samme RMSE i °C har samme farge i de to kartene. For de andre fenomenene brukes
deres egne grenser,
slik at lave RMSE-verdier ikke gir et ensfarget kart. Skalaen står
fast for samme modell og oppløsning når egne stasjoner legges til eller fjernes.
Begge grensene er de samme for alle MET-stasjoner, SVV-stasjoner og tomt nett.
Grensene hentes fra det uredigerte SVV-grunnnettet uten datofilter, også når en
vegkorridor, et fylkesutvalg eller en RMSE-terskel vises.
Raster, tegnforklaring og feilfargede veger bruker samme faste grenser.
Zoom, panorering, vegfiltre og egne stasjoner endrer aldri fargeskalaen.
Tallene i tegnforklaringen viser grensene; samme farge er derfor ikke nødvendigvis
samme feil på tvers av modeller eller oppløsninger. Feilverdier, arealer og
eksporterte rasterverdier endres ikke av fargeskalaen.
Tallene vises automatisk mellom kartlagets overskrift og fargelinjen og følger
kartlagets skala. Temperaturkartene for Direct P95 bruker de faste verdiene 0, 1 og 3 °C.
**Plasseringsgevinst** viser tre tall, avrundet til hele enheter.
**Endring i usikkerhet** viser i stedet bare endepunktene med enhet, samt 0
i midten når sammenligningen kan gi både mer (minus) og mindre (pluss) usikkerhet.
For Direct P95 viser **Tomt nett** det anslåtte intervallet på skalaen, fra ≥1 til 0 °C.
Områder som ligger på eller over beregningsreferansen, får ingen farge.
Endepunktene merkes med ≤ og ≥ der verdier utenfor fargegrensene
bruker samme ytterfarge.
Kategoriske tegnforklaringer, som klimasoner, har ikke en numerisk skala.
For **Usikkerhet** og **Grunnett** lastes pikselverdier og farger sammen.
**Skjul farger under** skjuler deretter piksler lokalt, uten nye kartforespørsler,
og raske skyveendringer samles til siste verdi før neste opptegning.
Grensen gjelder verdien i det viste rasterpikselet; modell- og eksportdata
endres ikke. Ufiltrerte Frost-kart klargjøres med tallverdier før oppstart og
gjenbrukes fra `outputs/station_app_cache/weather_maps` ved senere oppstarter.
Forbered manglende SVV-grunnflater separat med
`python precompute_frost.py --all-weather-maps --station-network both`.
Gamle forhåndsberegnede bilder med andre fargegrenser brukes ikke.

### Responsiv Oppstart
Appen starter som standard på **5000 m**. Finere evalueringsrutenett lastes først
når de velges. Plasseringsrutenettet på 1000 m lastes først ved første plassering;
senere brukes det samme rutenettet på nytt. De fysiske
stasjonsplasseringene endres ikke. Første plassering kan derfor gi en kort ekstra
pause, mens vanlig oppstart unngår lesing av hele 1000 m-geometrien.

Det automatiske foreslåtte nettet tilbys ikke lenger i brukergrensesnittet.
Egne forslag kan fortsatt legges til manuelt.

Resultatet med/uten valgt stasjon mellomlagres mens scenarioet er uendret, slik
at nøkkeltall, raster og vegvisning ikke starter samme beregning flere ganger.
Endret nett, modell, oppløsning, stasjonsvalg eller ferdig Frost-grunnflate
oppdaterer resultatet. Et terskelbytte kan gjenbruke de samme feilverdiene.

Usikkerhetskartet for Frost uten egne stasjoner leser den ferdig lagrede flaten
direkte, uten modellprediksjon eller beregning av stasjonsbidrag for kartbildet.
Ferdige bilder gjenbrukes mellom forespørsler. Første oppstart må fortsatt lese
terreng, modell og data; første bilde må omprojiseres og fargelegges.
Ved zoomnivå 5 eller lavere brukes høyst 400 bildepiksler langs lengste side,
ved nivå 6–7 høyst 800, ellers høyst 1600. Grovere usikkerhetsbilder bruker
gjennomsnitt ved nedskalering. Dette gjelder bare visningen, ikke
beregningsrutenett, nøkkeltall eller ArcGIS-eksport.

Verifisert Frost-stasjonsutvalg mellomlagres også på disk som
`availability_*.json` i `outputs/station_app_cache/`, med kontroll mot
database-/WAL-/rapport-signaturen. Det gjenbrukes ved serveromstart og forkastes
når datakildene endres. Dette erstatter ikke selve den eksakte Frost-beregningen.

### Eksakt Forhåndsberegning
**Kartappen beregner aldri eksakte Frost-grunnflater**, verken ved oppstart,
modellbytte, oppløsningsbytte eller i bakgrunnen. Den leser bare ferdige filer.
Kjør forhåndsberegningen separat før appen skal brukes. For eksempel luftfuktighet
på 5000 m:
```powershell
python -u precompute_frost.py --phenomenon humidity --cell-m 5000
python station_app.py --phenomenon humidity --cell-m 5000
```

Alle tilgjengelige prediksjonsmål for alle fem fenomener og tre rutenett:
```powershell
python -u precompute_frost.py
```
Dette kan omfatte **24 grunnflater**, inkludert vindhastighet. Vindmodellen er
lagret på 5000 m og kan evalueres på de andre rutenettene. For bare vind på
5000 m:
```powershell
python -u precompute_frost.py --phenomenon wind --cell-m 5000
python station_app.py --phenomenon wind --cell-m 5000
```
Kjøringen går i forgrunnen, én jobb om
gangen, fra 5000 m via 2500 m til 1000 m. Fremdrift skrives til terminalen.
Første fullstendige kjøring kan ta lang tid; gyldige ferdige filer hoppes over.
En fillås hindrer at to separate forhåndsberegninger kjører samtidig.

Begrens kjøringen med for eksempel `--phenomenon temperature humidity`,
`--cell-m 5000 2500` og `--kind continuous` (`class` og `all` støttes også).
Vis oppdagede modeller uten beregning eller skriving med:
```powershell
python precompute_frost.py --dry-run
```
Valgt fenomen uten ferdig modell for ønsket prediksjonsmål avvises.
De tidligere appvalgene `--precompute-all` og `--precompute-phenomenon` er fjernet.

For hver rute lagres laveste forventede feil, nest laveste feil fra en annen
stasjon og indeksen til beste stasjon. Dette gir eksakt stasjonsbidrag uten ny
beregning av hele nettet: når beste stasjon tas ut, brukes nest beste. Dersom to
stasjoner har samme laveste feil, gir fjerning av én av dem ingen forbedringsforskjell.
Egne forslag tas med i sammenligningen med/uten den valgte Frost-stasjonen.

Ferdige filer heter `frost_exact_<fingeravtrykk>.npz` i
`outputs/station_app_cache/`. Sjekkpunkter med `.partial.npz` lagres etter hver
stasjonsgruppe og gjenopptas etter omstart. Fingeravtrykket omfatter blant annet
modell, kildeegenskaper, stasjonsutvalg, datovindu og terrengrutenett. Gamle
12-nærmeste-filer er ikke kompatible og blir aldri brukt som erstatning.
Midlertidige fil-låser ved lagring (for eksempel fra OneDrive eller antivirus)
prøves på nytt opptil åtte ganger. Ved vedvarende lås beholdes både forrige
sjekkpunkt og den ferdigskrevne reservefilen (`.pending_*.npz`). Neste separate
kjøring bruker automatisk den gyldige filen med flest ferdige stasjoner;
ufullstendige reservefiler ignoreres. Kjør samme kommando igjen etter en feil.
Grunnflaten trenger bare beregnes én gang for en uendret kombinasjon av disse
inngangsdataene. Oppdatert modell, datovindu eller rutenett krever en ny flate;
den gamle filen overskrives ikke.

Hvis den valgte Frost-flaten mangler, vises ikke feilanslag eller påvirket
areal for den, og eksport av Frost-nettet er blokkert.
Appen viser den separate kommandoen som mangler.
**Tomt nett** og terrenglag kan fortsatt brukes. Statuskontrollen leser bare fra
disk og kan laste en kompatibel fil som den separate kjøringen har ferdigstilt.
Den starter aldri beregning. Nye modeller eller endrede datakilder krever
serveromstart. Avbrutt forhåndsberegning fortsetter fra sjekkpunkt neste gang
`precompute_frost.py` kjøres; appen bruker aldri uferdige sjekkpunkter.

### Oppløsning og Faste Plasseringer
**Oppløsning** bytter mellom de eksisterende terrengrutene på **1000, 2500 og
5000 m**. Beregningen bruker terrengverdiene og geometrien fra det valgte
rutenettet; det er ikke bare en forstørrelse av bildet. Nett, valgt stasjon og
egne forslag beholdes. Arealet følger rutestørrelsen.

Fysiske stasjonsplasseringer flyttes ikke ved byttet. Egne forslag knyttes alltid
til terrengpunktene på 1000 m, også når evalueringsrutene er grovere. Klikk mer
enn 1500 m fra gyldige plasseringer avvises. De lagrede foreslåtte stasjonene
beholder koordinater og kildeegenskaper fra den opprinnelige utvelgelsen på
5000 m. Den eksisterende 1000 m-replayfilen gjenbrukes; for 2500 og 5000 m
beregnes bare den delen av stasjonsrekken som etterspørres. Den lagrede replayfilen
gjenbrukes bare for den opprinnelige temperatur-RMSE-modellen. For andre modeller
evalueres de samme kildestasjonene på nytt med riktig modell og modellens enhet.
Første beregning av mange foreslåtte stasjoner på fint rutenett kan derfor ta tid;
beregnede prefikser og prediksjoner mellomlagres i appens minne per modell og oppløsning.

### Plasseringsgevinst på 500 m langs veg

`placement_gain.py` prøver én ny temperaturstasjon i hvert 500 m-terrengpunkt.
Det eksisterende SVV-nettet er uendret mellom kandidatene; dette er ikke en
sekvensiell, grådig utplassering. Gevinsten er
`sum(max(SVV_RMSE - kandidat_RMSE, 0) * korridorareal_km2)`, med enhet **°C·km²**.
Bare arealet innen **200 m** fra det ikke-kommunale vegnettet teller.
Overlappende vegbuffere telles bare én gang. Bufferbuene polygoniseres med
64 segmenter per kvadrant; arealvektet integrasjon er fortsatt en tilnærming
der RMSE er konstant i hver 500 m-rute.

Plasseringskartet tegnes på nytt for kartutsnittet ved panorering og zoom.
Kystmasken bruker en romlig indeks og klippes til utsnittet før rasterisering,
slik at ikke hele Norges detaljerte kystlinje behandles ved hver panorering.
Dette endrer ikke gevinstverdiene, 500 m-rutenettet eller kystavgrensningen.
Vegvær og Vegvær + MET bruker samme fargeskala, med maksimum fra det lagrede
Vegvær-plasseringskartet. Tegnforklaringen viser «Mindre gevinst» og «Mer gevinst».
Det opprinnelige Vegvær-resultatet må derfor også ligge i datamappen.

**Vis** i Vegnett-panelet slår vegnettet av og på. Med **Vis poeng** av vises
svarte veger over fargekartene og like over fylkesgrensene, uten klikkfunksjon.
Med **Vis poeng** på brukes det opprinnelige, poengfargede vegnettet, med
klikkfunksjon og GeoJSON-nedlasting. Bare ett av vegnettene vises om gangen.
Standard er svarte veger på og poeng av; begge valg lagres i scenariofilen.
Eldre scenariofiler uten poengvalget beholder den tidligere poengvisningen.

Svarte veger bruker bare E-, R- og F-referanser fra
`data/komprimert_nettverk_med_blindveg_10.gpkg`, inkludert blindveger. Vegkategori, vegnavn (for
eksempel E6, Rv3 eller Fv17) og valgte fylker filtrerer også dette nettet.
Scoreklassene finnes ikke i denne datakilden og er deaktivert uten **Vis poeng**;
valgene beholdes til poengvisningen slås på igjen. Andre vegkategorier er bare
tilgjengelige i poengvisningen. Visningsfiltrene endrer ikke værkartets vegkorridor.
E/R/F-kandidater filtreres i GeoPackage-spørringen før geometrien lastes,
og vegreferanser tolkes én gang per unik verdi. Geometrien leses én gang,
romlig indekseres og tegnes som
256-pikslers kartfliser med omtrent 1,5 pikslers linjebredde. Flisene forenkles
etter zoom, klippes til flisens rektangel og rasteriseres samlet som flerlinjer
når geometritypene tillater det. Flisene mellomlagres per filtervalg i serverminnet og nettleseren; panorering utløser
verken scenario- eller værberegninger. Første visning bruker noen sekunder på
innlesing. Filen tas med ved neste portable bygg.

Standard påvirkningsradius er **100 km** rundt hver kandidat. Ruter med sentrum
akkurat 100 km unna tas med. Utenfor radiusen settes kandidatens forbedring til
null, uten å kjøre modellen. Det eksisterende SVV-grunnnettet begrenses ikke.
Bruk `--radius-km 0` for beregning uten avstandskutt. Dette er en tilnærming;
den kan redusere gevinsten og endre rangeringen av kandidater.

Lag en kodepakke på Windows, uten å kopiere datasettet på nytt:

```powershell
.\.venv-seasonal-app\Scripts\python.exe prepare_placement_remote.py
scp -P 1959 .\outputs\placement_gain_code.zip havfyh@its.npra.io:~/vaernett_500m/
```

På SSH-maskinen, med det tidligere ferdige 500 m-datastadiet og miljøet:

```bash
cd ~/vaernett_500m
unzip -o placement_gain_code.zip
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u placement_gain.py --data-root outputs/portable_build/Remote_500m/data --work-dir outputs/placement_road200_radius100 --workers 14 --benchmark 28
```

Forberedelse av korridorareal og bassengegenskaper skjer først og mellomlagres.
Benchmarken skriver ingen endelig resultatflate. Prøv eventuelt `--workers 28`
og sammenlign faktisk tid; 28 logiske CPU-er er ikke 28 fysiske kjerner.
14 prosesser med én modelltråd hver er utgangspunktet for 14 fysiske kjerner.
Deretter kan hele kjøringen startes i en eksisterende `tmux`-økt:

```bash
python -u placement_gain.py --data-root outputs/portable_build/Remote_500m/data --work-dir outputs/placement_road200_radius100 --workers 14 > outputs/placement_road200_radius100.log 2>&1
```

Samme kommando gjenopptar ferdige kandidatblokker etter avbrudd. Endrede
inngangsdata, kode, radius eller relevante pakkeversjoner krever ny arbeidsmappe.
Ingen kandidat–mål-matrise for hele landet lagres: inndata minnekartlegges,
mål søkes med romlig indeks, og prediksjoner kjøres i avgrensede blokker.
Eksisterende resultatfiler overskrives ikke uten `--overwrite`.

Resultater i datastadiets `outputs`:
- `placement_gain_500m_road200.npz`: appens resultatflate og beregningsmetadata.
- `placement_gain_500m_road200.tif`: GeoTIFF i EPSG:25833 til ArcGIS.

Kopier resultatene tilbake til samme lokale 500 m-datastadium:

```powershell
scp -P 1959 'havfyh@its.npra.io:~/vaernett_500m/outputs/portable_build/Remote_500m/data/outputs/placement_gain_500m_road200.*' .\outputs\portable_build\Remote_500m\data\outputs\
```

Etter oppstart/oppfrisking av appen vises **Plasseringsgevinst · 200 m
vegkorridor** under **Kartlag**. Kartutsnitt gjengis ved panorering/zoom.
Vanlig klikk foreslår en ny stasjon, akkurat som i usikkerhetskartet.
Plasseringskartet er et fast beregningsresultat for det valgte grunnnettet;
scenarioendringer og appens visningskorridor endrer det ikke.

### Ny SSH-beregning med Vegværstasjoner + MET-stasjoner

`--station-network svv-met` bruker unionen av vegværstasjoner og stasjoner med
**MET.NO** i eierlisten som fast, uredigert grunnnett. Standardvalget `svv`
beholder den opprinnelige beregningen. Kandidater, temperaturmodell, faktisk
200 m vegkorridorareal, 100 km påvirkningsradius og parallelle kandidatberegninger
er ellers uendret. Det nåværende datagrunnlaget har 453 vegværstasjoner og
265 MET.NO-stasjoner, til sammen 718 temperaturstasjoner.

Lag en separat kodepakke med eierdata fra Windows:

```powershell
.\.venv-seasonal-app\Scripts\python.exe prepare_placement_remote.py --include-owners --output .\outputs\placement_gain_svv_met_code.zip
scp -P 1959 .\outputs\placement_gain_svv_met_code.zip havfyh@its.npra.io:~/vaernett_500m/
```

**Ikke pakk den ut over koden til den gamle jobben.** Bruk en egen kodemappe og
arbeidsmappe, men gjenbruk det store datasettet og det eksisterende Python-miljøet:

```bash
cd ~/vaernett_500m
mkdir -p placement_svv_met_code
unzip -o placement_gain_svv_met_code.zip -d placement_svv_met_code

nohup sh -c '
.venv/bin/python -u placement_svv_met_code/placement_gain.py \
  --data-root outputs/portable_build/Remote_500m/data \
  --work-dir outputs/placement_road200_radius100_svv_met \
  --station-network svv-met \
  --radius-km 100 \
  --workers 14
status=$?
printf "\nExit status: %s\n" "$status"
exit "$status"
' >> outputs/placement_road200_radius100_svv_met.log 2>&1 < /dev/null &

tail -n 30 -f outputs/placement_road200_radius100_svv_met.log
```

Den nye grunnflaten beregnes først og sjekkpunktlagres med den eksisterende,
eksakte delnettberegningen. XGBoost bruker opptil 14 tråder i denne fasen.
Kandidatfasen bruker 14 prosesser med én modelltråd hver, som tidligere.
Vent gjerne til den gamle jobben er ferdig for å unngå at to store jobber
konkurrerer om samme CPU. Avbrudd kan gjenopptas med samme kommando og uendret
kode, eierdata og parametere; ikke start to kopier av den nye jobben.

Resultatene lagres separat i datamappens `outputs`:

- `placement_gain_500m_road200_svv_met.npz`
- `placement_gain_500m_road200_svv_met.tif`

Eierdata, nettvalg og grunnflate inngår i fingeravtrykket. Gamle delresultater
kan ikke brukes med det nye grunnnettet. Loggen viser `Baseline cache: ...`;
kopier også denne lille grunnflatefilen tilbake til samme relative plassering
i den lokale datamappen for å gjenbruke grunnflaten i appen uten ny beregning.
Kopier NPZ/TIFF tilbake til den lokale datamappens `outputs`. Appen bruker det
tilsvarende plasseringskartet når **Vegværstasjoner + MET-stasjoner** er valgt,
og viser en tydelig feil hvis dette kartet ennå ikke finnes. Hvert nettvalg
bruker sitt eget resultat; se beregningene for alle stasjoner og tomt nett nedenfor.

### SSH: tomt nett først, deretter alle temperaturstasjoner

De fire nettvalgene er knyttet til separate, uredigerte plasseringskart:

| Nettvalg i appen | CLI-verdi | Resultatnavn i datamappens `outputs` |
| --- | --- | --- |
| Vegværstasjoner | `svv` | `placement_gain_500m_road200.npz/.tif` |
| Vegværstasjoner + MET-stasjoner | `svv-met` | `placement_gain_500m_road200_svv_met.npz/.tif` |
| Alle værstasjoner | `all` | `placement_gain_500m_road200_all.npz/.tif` |
| Tomt nett | `empty` | `placement_gain_500m_road200_empty.npz/.tif` |

`all` bruker den lagrede grunnflaten fra alle temperaturstasjoner, uten SVV-,
eier- eller datofilter. Kandidater, 500 m rutenett, faktisk 200 m vegkorridorareal,
100 km påvirkningsradius og 14 enkelttrådede prosesser er ellers som tidligere.
Stasjoner uten temperaturmåling inngår ikke i temperaturgrunnflaten.

Et tomt nett har ingen temperaturprediksjon og derfor ingen målt, endelig RMSE.
For å kunne rangere den første stasjonen bruker `empty` en **antatt referansefeil
på 0,5 °C** ved hvert målpunkt. Gevinsten er
`sum(max(0.5 - kandidat_RMSE, 0) * korridorareal_km2)` innen 100 km fra kandidaten;
utenfor radiusen gis ingen gevinst. Dette er en eksplisitt planleggingsantakelse,
ikke en målt feil eller en ny modell for temperatur uten stasjoner.
`--empty-reference-rmse` kan endre referansen i en separat beregning med ny
arbeidsmappe; verdien kan påvirke rangeringen. Den lagres i sjekkpunktets
fingeravtrykk, NPZ og GeoTIFF. Usikkerhetskartets behandling av tomt nett endres ikke.

For å se denne reduksjonen i appen: velg **Tomt nett**,
legg til én eller flere temperaturstasjoner og velg **Reduksjon**.
Referansen er fast på **0,5 °C** og kan ikke endres i appen eller ved å
åpne eldre scenariofiler med en annen referanse. Reduksjonen er
`max(0.5 - scenarioets temperatur-RMSE, 0)`. Bare steder med RMSE
strengt lavere enn referansen får farge; manglende prediksjoner og områder
utenfor stasjonenes påvirkningsradius er gjennomsiktige.
Eksempel: 0,5 °C referanse og 0,2 °C RMSE gir 0,3 °C reduksjon.
Alle temperaturforslag i det tomme nettet inngår, uavhengig av valgt stasjon.
Dette endrer ikke selve usikkerhetsverdiene eller referansen i en allerede
lagret plasseringsgevinstfil. SSH-skriptet bruker også 0,5 °C og en egen
arbeidsmappe, slik at gamle sjekkpunkter med 5 °C ikke blandes inn.

Nett med eksisterende stasjoner bruker samme faste SVV-fargeskala.
Tomt nett bruker en egen, fast skala fra sitt lagrede resultat, ellers ville
de langt større gevinstene stort sett få maksimal rødfarge. Fargene i det tomme
nettet kan derfor ikke sammenlignes direkte med fargene i de andre nettene.
Tegnforklaringen beholder «Mindre gevinst» og «Mer gevinst»; hold over tittelen
for å se referanseantakelsen for tomt nett.

Lag og last opp en ny, separat kodepakke fra Windows:

```powershell
.\.venv-seasonal-app\Scripts\python.exe prepare_placement_remote.py --output .\outputs\placement_empty_all_ref05_20261115_code.zip
scp -P 1959 .\outputs\placement_empty_all_ref05_20261115_code.zip havfyh@its.npra.io:~/vaernett_500m/
ssh -p 1959 havfyh@its.npra.io
```

Kjør på SSH-maskinen:

```bash
cd ~/vaernett_500m
mkdir -p placement_empty_all_ref05_20261115_code
unzip -o placement_empty_all_ref05_20261115_code.zip -d placement_empty_all_ref05_20261115_code
nohup sh placement_empty_all_ref05_20261115_code/run_placement_empty_all.sh \
  >> outputs/placement_empty_all_ref05_20261115.log 2>&1 < /dev/null &
echo "Started batch: $!"
tail -n 30 -f outputs/placement_empty_all_ref05_20261115.log
```

Skriptet kjører **tomt nett først**, og starter **alle stasjoner straks det
første er fullført**. Ved feil stopper det før neste jobb og skriver
`Batch exit status`. De kjøres ikke parallelt. Arbeidsmappene er
`outputs/placement_road200_radius100_empty_ref05` og
`outputs/placement_road200_radius100_all`; tidligere SVV/MET-jobber endres ikke.
Vent til eventuelle andre store jobber er ferdige for å unngå CPU-konkurranse.
Start aldri to kopier av batchen samtidig.

Samme uendrede skript kan gjenopptas etter et avbrudd. `--overwrite` i skriptet
lar en allerede ferdig første jobb sette sammen sine sjekkpunkter og skrive
sine egne resultatfiler igjen før den fortsetter til jobb to. Det omgår ikke
kontrollen av endret kode, data eller referansefeil. Ikke erstatt kode i en
mappe mens en jobb kjører, og ikke gjenbruk sjekkpunkter fra et annet nett.

Kopier begge resultatpar tilbake fra Windows når de er ferdige:

```powershell
scp -P 1959 'havfyh@its.npra.io:~/vaernett_500m/outputs/portable_build/Remote_500m/data/outputs/placement_gain_500m_road200_empty.*' .\outputs\portable_build\Remote_500m\data\outputs\
scp -P 1959 'havfyh@its.npra.io:~/vaernett_500m/outputs/portable_build/Remote_500m/data/outputs/placement_gain_500m_road200_all.*' .\outputs\portable_build\Remote_500m\data\outputs\
```

Ingen ekstra grunnflatefil trengs for disse to nettene. Start den oppdaterte
Python-appen og velg ønsket **Stasjonsnett**. Manglende resultat gir en tydelig
feil i stedet for å vise et annet netts plasseringsgevinst. Det lagrede kartet
viser grunnnettet, ikke en ny optimalisering etter hvert stasjonsforslag eller
etter dato-/sensorendringer i scenariet.
**Skjul farger under** kan også brukes på plasseringsgevinst for å skjule
områder med lav gevinst. Filteret nullstilles ved bytte mellom gevinstkartet og
usikkerhetskartene.
**Kartlag langs veger** begrenser også gevinstkartet til valgt avstand fra veg,
både i appen og i nye lagrede HTML-kart.
Plasseringsgevinst for ett værfenomen vises som fargede 256-pikslers kartfliser,
uavhengig av bakgrunnskartets gråtoner. Før serveren starter, klargjøres
ufiltrerte oversiktsfliser på zoomnivå 0–6 for alle tilgjengelige grunnnett.
Disse lagres under `outputs/station_app_cache/placement_tiles` og gjenbrukes
ved senere oppstarter. Første klargjøring kan ta noen minutter; fremdriften
vises i terminalen. Endret kartgrunnlag eller tegningskode gir nye fliser.
Panorering gjenbruker flisene; bare nye områder lastes. Serveren beholder opptil
512 ferdige fliser og ett klargjort kilderaster i minnet. Nettleseren kan lagre
flisene i én time. Flisene inneholder både gevinstverdier og opprinnelige farger.
**Skjul farger under** endrer derfor bare gjennomsiktigheten lokalt i nettleseren,
uten nye flisforespørsler. Raske skyveendringer samles til siste verdi før neste
opptegning. Stasjonsnett, dataversjon, fylker og vegkorridor har separate
hurtigbuffernøkler. Nye data krever at siden lastes på nytt.
Detaljer over zoomnivå 6 og nye fylkes-/vegfilter tegnes ved behov og mellomlagres
i minnet. HTML-eksport bruker fortsatt innebygde bilder og trenger ikke flistjeneren.

### Prøve 100 km-radius i stasjonsappen

Vanlig appoppstart og nye portable bygg bruker som standard 100 km radiuskutt
for nye og flyttede stasjoner. Bruk `--station-radius-km 0` for å slå det av.
Direkte opprettelse av `StationEngine`, blant annet i `host_station_app.py`,
har fortsatt standard uten radiuskutt. Start en separat testinstans
(ikke stopp en app med et ulagret scenario):

```powershell
.\.venv-seasonal-app\Scripts\python.exe station_app.py --station-radius-km 100 --port 8790
```

Dette gjelder bare **nye/flyttede** stasjoner. Eksisterende grunnflater,
stasjonsmarkører og sensorinformasjon beholdes. Utenfor 100 km konkurrerer
ikke den nye stasjonen med grunnnettet. I et tomt nett er disse rutene uten
beregnet verdi, ikke kunstig null usikkerhet. En merknad under Værstasjoner
viser at forsøket er aktivt. Start uten flagget for opprinnelig oppførsel.
Samme lagrede scenario kan åpnes i to separate instanser for visuell sammenligning.

For målte prediksjonstider og resultatsammenligning på et lite utvalg:

```powershell
.\.venv-seasonal-app\Scripts\python.exe compare_station_radius.py --data-root .\outputs\portable_build\Remote_500m\data --candidates 3 --output .\outputs\radius_comparison.csv
```

CSV-en sammenligner ukachede prediksjoner og samme SVV-baseline med/uten kutt:
tid, antall predikerte ruter, gjennomsnittlig/største RMSE-økning og endringer
i rutene som berører 200 m-vegkorridoren. Dette er ikke arealvektet
gevinstintegrasjon, og tre kandidatpunkter er ikke en nasjonal validering.
Modellens prediksjoner utover 100 km er ekstrapolasjoner.

### Eldre lagret Plasseringsgevinst (API)
Uten det nye 500 m-resultatet viser det eksisterende API-et fortsatt
`total_error_reduction` fra det lagrede
femårskartet på **5000 m** for valgt værfenomen og prediksjonsmål. Hver rute
angir total forventet feilreduksjon ved en ny stasjon på dette kandidatpunktet,
summert over rutene i den lagrede beregningen. Det er ikke punktets lokale RMSE
og ikke bidraget fra den valgte stasjonen i det nåværende scenarioet.

Kartet leses fra `<modellprefiks>_error_reduction.gpkg`, lag `error_reduction`.
Luftfuktighet bruker sin ferdige `humidity_5y_5000m`-mappe. Koordinatene må
samsvare med kandidatgridet på 5000 m. Høyere gevinst vises rødere, med en fast
logaritmisk skala for den lagrede filen. Verdier og ferdige bilder mellomlagres;
ingen prediksjoner eller ny Frost-beregning startes for å vise laget.

Det lagrede kartet forblir på 5000 m også når det aktive evalueringsrutenettet
er finere. Egne stasjoner, nettbytte og vegkorridor endrer ikke de lagrede
gevinstverdiene eller bildet. Det oppdateres ikke til en ny optimal plassering
etter hvert eget forslag. API-et viser den opprinnelige beregningens resultat;
scenarioets ordinære ArcGIS-eksport gjelder fortsatt det aktive
stasjonsnettet. En manglende lagret flate meldes som feil, uten ny beregning.

### NVE weather grids: native 1 km data for ArcGIS Pro

Use `download_nve_grid.py` for NVE GridTimeSeries data, not the MEPS forecast
downloader. The MEPS NCSS service returned HTTP 503 again on 2026-10-28, even
for a tiny subset; NVE point requests succeeded. MEPS has a native 2.5 km grid
and cannot provide genuine 1 km detail by resampling.

Run with the existing `arcgis-nvdb` environment (the editor's Anaconda environment
does not have Rasterio). Start with one Oslo cell:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" .\download_nve_grid.py --start 2025-01-15 --end 2025-01-15 --point 262500 6650500 --themes tm rr --output-dir .\outputs\nve_point_oslo_20250115
```

Download a 5 x 5 km Oslo pilot, with 25 native 1 km cells:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" .\download_nve_grid.py --start 2025-01-15 --end 2025-01-15 --bbox 260000 6648000 265000 6653000 --themes tm rr --output-dir .\outputs\nve_grid_oslo_20250115
```

Both point and bounding-box coordinates are **UTM zone 33N metres, not
longitude/latitude**. Bounds expand outward to the native kilometre grid;
requests sample cell centres, and raster rows run north to south.
Unlike the MEPS command, **both dates are inclusive**. `tm` is daily
temperature, `rr` daily precipitation, `sd` snow depth and `swe` snow-water
equivalent. For three-hour data, use `tm3h` or `rr3h` and explicit hours, e.g.
`--start 2025-01-15T03 --end 2025-01-15T09`. Wind speed/direction themes are
also available through `--themes`; units come from the API.

The downloader calls `https://gts.nve.no/api/GridTimeSeries` sequentially,
retries transient HTTP errors, caches validated JSON per cell, and checks
response coordinates, units and time axes. Rerun the same command to reuse
downloaded cells; use a different output directory for a different query.
`--dry-run` prints the grid/request count without network or output files.
The default guard is 10,000 cells; long queries should be split into smaller
areas/date ranges. NVE returns a point time series per cell, not a bulk
national raster, so a national multi-year download is not a suitable pilot.

**View in ArcGIS Pro:** Map > Add Data, browse to the output directory and
select `nve_tm_20250115T0600_1km.tif` or `nve_rr_20250115T0600_1km.tif`.
Right-click the layer > Zoom To Layer. Use a Stretch renderer for temperature
and precipitation. A constant field (e.g. zero rain) will have one colour.
Each theme/timestamp has a separate single-band float GeoTIFF with 1,000 m
pixels, EPSG:32633 (WGS 84 / UTM 33N), units and timestamp metadata. Source
NoData is masked, not interpreted as weather. `manifest.json` records ranges,
valid/missing counts and raster paths; `responses/` retains original API values.
NVE's explicit HTTP 400 "No cell exists" response is also recorded and masked
(for example over the Oslo fjord); other HTTP 400 errors remain fatal.
The API's daily timestamps are at 06:00; timestamps are preserved exactly,
without inventing a timezone or treating daily values as instantaneous readings.

Create styled layers, PNG previews and a standalone ArcGIS Pro project, and
open it (ArcPy required; no changes to any existing project):
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" .\export_nve_arcgis.py .\outputs\nve_grid_oslo_20250115 --open
```
This creates `nve_weather.aprx` with separate maps for each theme, stretch
symbology and an extent fitted to the download. Only the first timestamp is
visible initially in each map. In the opened project, expand **Catalog >
Project > Maps** and double-click **NVE tm [Celcius] - 1 km** to open the
temperature map (or the `rr` map for precipitation). Alternatively, add the
generated `.lyrx` files to your own map. The exporter refuses to overwrite an existing project.
If the installation does not include Esri's `Blank.aprx` template, pass
`--template` with a blank project you created in ArcGIS Pro.

These are **NVE gridded products, not interchangeable with MEPS forecast
training data**. They are not written into the MEPS/Frost training database.
In particular, daily `tm` is not a +12-hour temperature forecast, and this
endpoint does not supply the MEPS relative-humidity field.
API documentation: https://api.nve.no/doc/gridtimeseries-data-gts/

### Whole-country seNorge bulk download (MET Norway)

For mainland Norway, use `download_senorge_norway.py` instead of hundreds
of thousands of NVE point requests. It downloads one original daily NetCDF
per date from MET Norway's seNorge_2018 `Latest` collection, retains it, and
exports native 1 km daily temperature (`tg`) and precipitation (`rr`) GeoTIFFs:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" .\download_senorge_norway.py --start 2025-01-13 --end 2025-01-17 --output-dir .\outputs\senorge_norway_20250113_20250117
```
Dates are inclusive: this example downloads five files, about 85 MB total,
and creates ten rasters, without creating an ArcGIS project. Dependencies
are Requests, NumPy, netCDF4 and Rasterio, available in `arcgis-nvdb`.
Completed NetCDF files are reused on restart. The script validates dates,
native kilometre spacing, projection and units, preserves missing values,
and reports failures with a nonzero exit code.

The full seNorge domain covers **mainland Norway and neighbouring areas**;
it is not clipped to Norway's border and excludes Svalbard and Jan Mayen.
Daily mean temperature is exported in degrees Celsius; precipitation in
kg/m² is numerically equivalent to mm of water. Original UTC timestamps and
daily aggregation descriptions are retained in raster metadata.
`manifest.json` records per-day extents, valid/missing cell counts, ranges,
source URLs and original NetCDF paths. Credit **MET Norway**; the original
licensing URL is included in each raster.

This is the separately versioned MET Norway **seNorge_2018** bulk product,
not an assertion that it is identical to NVE GTS `tm`/`rr`. File names use
`senorge_tg`/`senorge_rr`, not `nve_tm`, to keep provenance explicit.
The `Latest` collection's historical coverage is server-controlled; a
missing daily file produces an explicit HTTP error, not substitution from
another product or date.

#### Five complete years: annual archives
Use the annual `Archive` collection for 2021 through 2025:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u .\download_senorge_norway.py --annual --start 2021-01-01 --end 2025-12-31 --output-dir .\outputs\senorge_norway_2021_2025
```
This retains five original NetCDFs, about 11.4 GB total, rather than downloading
1,826 separate daily files. Originals also retain the archive's minimum and
maximum temperature variables where present. No ArcGIS project or GeoTIFFs
are created by default. Each file covers its entire calendar year even if
the requested range selects only part of that year.

Annual transfers are sequential, report progress every 128 MiB, and preserve
`.pending.nc` files on network failures. Rerun the same command to resume
with HTTP Range and If-Range using the server's ETag/Last-Modified validator.
If the server ignores a range or the source changed, that year's download
restarts safely rather than mixing revisions. Files without a validator also
restart from zero. Completed annual files are reused and revalidated.

Validation checks the full calendar (including leap days), daily spacing,
native 1 km coordinates, projection and units, and reads every temperature
and precipitation slab to detect missing days or corrupt compressed chunks.
`manifest.json` is updated after each verified year with coverage, per-variable
ranges, valid-cell counts, sizes and source URLs. Do not treat a partial
manifest as a completed five-year collection; check `status`.

To additionally export all 3,652 daily temperature/precipitation GeoTIFFs,
rerun the same command with `--export-geotiffs`. It reuses the annual files,
processes one day at a time, writes per-day completion records, and skips
completed exports on restart. Allow roughly another 6 GB for GeoTIFFs
(seasonal compression varies). The default output is in OneDrive: check
cloud quota as well as local free space before large runs.

#### Short-range virtual-pair pilot
`senorge_short_range_pilot.py` samples 1–50 km terrain-grid pairs, uses
2021–2023 seNorge temperature to label their two-direction prediction RMSE,
and tests whether adding those virtual pairs helps the Frost pair-error model.
Frost remains the evaluation target; seNorge is a gridded teacher, not
independent ground truth. Training virtual pairs are excluded from each
held-out 100 km station block.

Run the bounded pilot with its default 5,000 virtual pairs, 5,000 Frost pairs,
every twelfth seNorge day, and three spatial holdouts:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u ".\senorge_short_range_pilot.py"
```
The initial screening run improved mean Frost-test MAE by about 0.75% but
worsened RMSE by about 0.9%; sub-50 km test bins were small and mixed. Treat
this as inconclusive and keep it separate from the production model. The
summary and sampled virtual pairs are written under
`outputs/senorge_short_range_temperature_pilot/`.

To also build the optional Station Lab profile, rerun with
`--export-stationlab-profile`. It writes a separate
`temperature_5y_prediction_rmse_senorge_*` bundle; the Grunnett files are
untouched. Precompute its 5 km baseline explicitly, then start a separate app:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u ".\senorge_short_range_pilot.py" --export-stationlab-profile
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u ".\precompute_frost.py" --phenomenon temperature --cell-m 5000 --kind senorge
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u ".\station_app.py" --port 8910 --cell-m 5000 --phenomenon temperature
```
Select **seNorge-pilot** under the temperature model choices. This is an
experimental comparison profile; use it to inspect station-change patterns,
not as a calibrated absolute-error replacement.

For the detailed 1 km run, start the overnight workflow from PowerShell. It
samples 20,000 virtual pairs from daily seNorge 2021–2023, uses five spatial
validation repeats, exports the optional model, precomputes Grunnett and
seNorge surfaces at 1 km, then launches Station Lab on a free port starting at
8911:
```powershell
$python = "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe"
$pythonArgs = @('-u', '.\overnight_senorge_stationlab.py')
$process = Start-Process -FilePath $python -ArgumentList $pythonArgs -WorkingDirectory $PWD -RedirectStandardOutput '.\outputs\senorge_overnight_console.log' -RedirectStandardError '.\outputs\senorge_overnight_errors.log' -PassThru
$process.Id
```
The detailed log is saved under `outputs/`. If interrupted after model training,
rerun with `--skip-training`; this preserves the model fingerprint so the
1 km precompute can resume from its checkpoint:
```powershell
& $python -u .\overnight_senorge_stationlab.py --skip-training
```

### Five-year weather regions over mainland Norway

`weather_regions.py` classifies native 1 km cells using all 1,826 days of
2021-2025 from the verified annual seNorge archives. It is a **five-year
weather-region analysis, not a 30-year climate normal**, a Koppen
classification or a forecast. Wind, humidity and measured snow are not
available in these files and are not represented.

Run the complete workflow:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u .\weather_regions.py --output-dir .\outputs\weather_regions_2021_2025
```
The input defaults to `outputs/senorge_norway_2021_2025`. The existing
`outputs/norway_polygon.gpkg` land polygon (N100 non-ocean union) masks
Norway using cell centres, transformed into the source CRS. Coastal partial
cells count as complete square pixels: reported cell areas are nominal,
not exact clipped land area. The domain excludes Svalbard and Jan Mayen.
Only cells with complete finite `tg`, `rr`, `tn` and `tx` for the whole
period are classified; no missing weather is filled or silently imputed.
Exclusion counts and source-file fingerprints are recorded.

#### Balanced features
Six families receive equal expected squared contribution after training:
1. Twelve monthly mean temperatures, retaining absolute Celsius differences.
2. Twelve monthly precipitation totals, with `log1p` for skewed rainfall.
3. Monthly variability/extreme statistics: daily temperature standard
   deviation and 5th/95th percentiles; wet-day fraction (>= 1 mm); heavy-day
   fraction (>= 20 mm); precipitation 95th percentile and maximum;
   frost frequency (`tn < 0`), freeze-thaw frequency (`tn < 0 < tx`) and
   mean diurnal temperature range (`tx - tn`). Rainfall magnitude statistics
   use `log1p`; frequencies remain fractions.
4. Year-by-month departures of temperature and log monthly rainfall from
   the cell's multi-year monthly pattern. These describe differences between
   the five observed years, not a defensible long-term trend estimate.
5. Twelve PCA components of daily temperature anomalies.
6. Twelve PCA components of daily `log1p` precipitation anomalies.

Source aggregation windows differ: `tg`/`rr` use 06-06 UTC, while `tn`/`tx`
use 18-18 UTC. Frost/freeze-thaw and diurnal-range statistics use the
min/max product's own window; they are not hourly crossing counts.

Daily anomalies subtract the cell's multi-year calendar-month mean. They
are **not divided by that cell's own standard deviation**: variability
amplitude remains meaningful. Absolute levels remain in the first two
families. For families 1-4, each feature is spatially standardized using
training cells, then the family is normalized by its nonconstant feature
count. PCA score families are normalized by their total training variance,
not component-wise whitening, so larger components retain more influence.
PCA variance retention is reported in `analysis.json`; some fine-scale
daily behaviour is inevitably lost in this compression.

#### Comparing group counts and stability
The defaults compare 8, 12, 16, 24 and 32 MiniBatchKMeans groups with fixed
seed 1729. Preprocessing/PCA is fitted on 12,000 randomly sampled land cells;
4,000 disjoint spatial sample cells evaluate the candidate models.
This is **not independent weather-station validation**.

For each K, the nearest training member to each feature-space centroid is
a real-cell representative. Candidate quality uses daily temperature RMSE,
daily precipitation RMSE, wet/dry disagreement, and monthly precipitation
total MAE against those representatives. Each metric is divided by its
single-representative baseline and the four are averaged. Silhouette
measures separation in the balanced feature space.

For each omitted year, seasonal features, interannual departures, PCA,
scalers and candidate cluster models are all recomputed without that year.
The adjusted Rand index (ARI) compares validation-cell partitions to the
five-year partition. This tests sensitivity to the observed years, **not
prediction skill on future weather**.

Selection is an explicit heuristic:
`error_elbow + 0.2 * silhouette + 0.2 * mean_year_omission_ARI`.
The elbow is normalized representation-error reduction minus normalized
`log(K)` between the smallest and largest candidates. Highest score wins;
there is no claim of a uniquely optimal number of climate regions.
`cluster_comparison.csv` preserves every metric so the compromise can be
reviewed. After selection, MiniBatchKMeans is refitted on **all** complete
Norway cells. Thus candidate-validation scores describe the sampled models;
the final descriptive error rasters describe the full-data refit.

The national 2021-2025 comparison selected **8 broad groups** with this
stability-weighted heuristic, but representation errors kept improving up
to **32 groups**, without a clear plateau. Both scales are useful; a coarse
map is not a guarantee of matching daily local weather. Export a tested
detailed alternative without repeating extraction or year-omission tests:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u .\weather_regions.py --output-dir .\outputs\weather_regions_2021_2025 --additional-k 32
```
This creates `k32/` with its own map, representatives, error rasters and
profiles, while preserving the original overview. It reuses the fitted
scalers/PCA exactly, then fits 32 clusters to all cells. The alternative
counts must already be in the original comparison; existing result
directories are not overwritten. The maps are independently fitted
partitions, **not nested administrative-style subdivisions** of each other.

#### Outputs and interpretation
The output directory contains:
- `weather_regions_1km.tif`: categorical region IDs, NoData=0. IDs are
  ordered by increasing region-wide daily mean temperature, not quality.
- `weather_regions.png`: map preview with representative locations.
- `representatives.gpkg`, layer `representatives`: real grid-cell centres,
  region IDs, nominal areas, average conditions and error summaries.
- `temperature_rmse_c_1km.tif`: daily temperature RMSE against the region's
  representative, in Celsius.
- `precipitation_rmse_mm_1km.tif`: daily precipitation RMSE, in mm/day.
- `wet_day_disagreement_1km.tif`: fraction of days when local and
  representative wet/dry status differ (0=perfect agreement).
- `monthly_precipitation_mae_mm_1km.tif`,
  `annual_precipitation_mae_mm_1km.tif`: absolute differences in observed
  monthly/yearly rainfall totals.
- `temperature_extreme_mae_c_1km.tif`: difference in monthly daily-mean
  temperature 5th/95th percentiles, not record min/max temperature.
- `heavy_day_frequency_error_1km.tif`: absolute difference in >=20 mm
  day frequency; `feature_distance_1km.tif`: distance to the assigned
  centroid, not a probability.
- `region_summary.csv`: counts, mean conditions, mean/p95 representation
  errors; `seasonal_profiles.csv` compares each group's mean seasonal
  conditions with its representative; `year_month_profiles.csv` retains
  each region's observed month/year conditions.
- `representation_errors.png`, `seasonal_profiles.png`,
  `cluster_comparison.png`: readable diagnostics.
- `analysis.json`, `selection.json`: parameters, caveats, metrics and
  selection formula; `feature_preprocessing.joblib`/`kmeans.joblib`: fitted
  models for this feature definition.

All rasters use EPSG:32633 with 1,000 m pixels and identical alignment.
Add the categorical map in ArcGIS Pro with **Unique Values**, and error
rasters with **Stretch**. No ArcGIS project is created.
Regions need not be connected; coordinates/elevation are deliberately
not clustering features. A geographic barrier does not itself force a new
weather type. A group can still have large day-to-day rainfall differences:
use the error layers rather than interpreting a shared colour as identical
weather everywhere. Representatives approximate group weather; they are
not proposed instrument sites or independently validated observations.

The workflow caches extracted annual weather in `cache/` (roughly 6 GB)
and uses several GB of RAM; four numerical threads are the default.
Interrupted extraction can reuse completed years with the same source
fingerprints. A completed analysis is not overwritten; choose another
output directory to compare parameters. A small preview can use
`--max-cells 2000 --training-cells 1200 --validation-cells 400`; these are
sampled cells, not a complete national raster.

### Thirty-year Norway climate regions: 1996-2025

#### Run unattended overnight from a terminal

From this project folder in PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_climate_overnight.ps1
```

The launcher returns immediately, printing a background worker PID, progress
log, error log and status-file path. The independent Windows process continues
when the terminal or VS Code closes. **Keep the computer plugged in and disable
automatic sleep for the night**; it cannot work while Windows is sleeping or
shut down. Signing out or rebooting can terminate the job. The launcher does
not change system power settings or create a scheduled task.

It downloads/resumes missing 1996-2025 originals, then runs the classification.
Completed archives with matching sizes are reused; a completed analysis is
not rerun. The current national results are already complete, so the default
command checks them and finishes quickly rather than spending another night
recalculating. To intentionally make a new analysis:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_climate_overnight.ps1 -OutputDir .\outputs\climate_regions_1996_2025_new
```

Check progress (use the actual timestamped log printed by the launcher):

```powershell
Get-Content .\outputs\climate_regions_1996_2025\overnight_status.json
Get-Content .\outputs\climate_regions_1996_2025\overnight_YYYYMMDD_HHMMSS_fff.log -Tail 30 -Wait
```

Ctrl+C stops **log watching only**, not the background job. To stop the job,
first identify its Python child with
`Get-CimInstance Win32_Process | Where-Object ParentProcessId -eq <worker-PID>`,
then use `Stop-Process -Id <python-PID>,<worker-PID>`. Use only the specific
IDs belonging to this job. An abruptly stopped worker can leave status
`running`; check the PID before interpreting that status. Rerunning the same
command resumes available yearly caches; incomplete comparison/export stages
are recomputed. It is not a checkpoint of every calculation.

Additional options: `-Threads 4`, `-Python 'C:\path\to\python.exe'`,
`-SourceDir 'D:\weather\senorge_norway_1996_2025'`, and `-Foreground` to run
in the current terminal with an actual exit code. The background launch exit
code only confirms process creation: inspect status/logs for success or failure.
An exclusive output-directory lock prevents two launcher workers writing the
same results; do not run the Python analysis manually against that directory
at the same time. Failures are recorded explicitly with the stage and message.
Source paths in existing manifests remain unchanged when directories move.

`climate_regions.py` is a separate, bounded-memory extension of the five-year
analysis. It processes all **10,958 days in 1996-2025**, retaining the native
1 km grid and Norway land-centre mask. This is a recent 30-year descriptive
classification, **not an official climate classification or the standard
1991-2020 climate normal**. Existing five-year outputs remain unchanged.

Download and verify the originals, referencing the existing 2021-2025 files
without copying them:

```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u .\download_senorge_norway.py --annual --start 1996-01-01 --end 2025-12-31 --output-dir .\outputs\senorge_norway_1996_2025 --reuse-dir .\outputs\senorge_norway_2021_2025\netcdf
```

The full source collection is about 65.6 GB (decimal), including about
11.45 GB already downloaded for 2021-2025. Feature caches need approximately
17 GB more, plus small maps/models. Original files are retained. The approved
local-drive run started with about 129.6 GiB free; extraction stops explicitly
if less than 5 GiB remains. Annual HTTP transfers and completed yearly
extractions can resume. Do not move reused originals: their actual paths are
stored in the source manifest.

Run the analysis in another terminal. `--wait-for-download` allows extraction
of each verified year while later years are still downloading:

```powershell
$env:LOKY_MAX_CPU_COUNT='4'
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u .\climate_regions.py --output-dir .\outputs\climate_regions_1996_2025 --wait-for-download
```

The workflow reads one year of daily mean temperature and precipitation at
a time, and monthly slices of minimum/maximum temperature. It uses seven
equally weighted feature families:

1. Monthly mean temperature, preserving absolute differences, plus an
   explicit summer-minus-winter temperature amplitude.
2. Monthly precipitation totals, log-transformed to limit outlier dominance.
3. Monthly variability/extremes: temperature spread and percentiles, rain
   intensity/extremes, wet/heavy-day frequency, frost, freeze-thaw days and
   daily temperature range.
4. Mean and interannual spread of annual longest wet/dry/warm/cold spells,
   growing/heating/freezing degree-days, days above 5 C, and maximum 3-/5-day
   rain totals. Annual frost-day counts and their interannual standard
   deviation are explicitly included too, in addition to monthly frost
   frequency in the variability family.
5. Interannual standard deviations and linear per-decade trends of each
   calendar month's mean temperature and log precipitation.
6. Shared daily temperature timing.
7. Shared daily precipitation timing, using log1p precipitation.

Wet means at least 1 mm/day; warm means daily mean at least 20 C; cold means
daily mean below 0 C. Degree-day bases are respectively 5, 17 and 0 C.
Spells and rolling-rain maxima **reset at calendar-year boundaries**; fixed
warm/cold thresholds are not percentile-defined heatwaves/coldwaves.
Freeze-thaw indicates daily minimum below zero and maximum above zero,
not observed hourly crossings. Source daily intervals differ: mean temperature
and precipitation cover 06-06 UTC, while minimum/maximum cover 18-18 UTC.

**Summer-winter amplitude** means June-August mean daily temperature minus
the mean for January, February and December. Monthly means are weighted by
the actual number of days (including leap February), then annual contrasts
are averaged across the retained years. Winter here uses those three months
of each calendar year, not a continuous cross-year winter; no unobserved
December 1995 or January-February 2026 is required. This is seasonal contrast,
not the hottest day minus the coldest day.

**Frost days** means days with daily minimum temperature (`tn`) below 0 C,
not days with mean temperature below zero. Monthly frost frequencies are
converted back to exact integer daily counts before annual aggregation.
The map gives mean frost days per year; clustering also includes their
year-to-year standard deviation. Cold spells using daily mean below zero
remain a separate feature.

Daily timing is compressed using a reproducible 128-dimensional signed
random projection, then 12-component PCA for each variable. Anomalies remove
each cell's **that-year calendar-month mean**, leaving monthly levels and
trends to the other families. This differs from the five-year feature
definition. It preserves amplitude information but is approximate, not
exact full-series PCA. `timing_compression_audit.json` reports sampled
distance distortion/correlations and retained sketch-PCA variance.
Low correlations can occur for near-identical pairwise distances, so inspect
relative distortion as well. Scalars/PCA are fitted only to training cells.

Candidate K values 8, 12, 16, 24 and 32 use 8,000 disjoint training and
2,000 validation cells. Six held-out five-year blocks each refit feature
aggregation, scaling, PCA and clustering **without those years**. Daily
representative errors are then evaluated on the held-out years.
`cluster_comparison.csv` separates these errors from full-period descriptive
errors and reports adjusted-Rand partition stability. These tests are not
independent station validation or future forecasts.

Both `k8/` and `k32/` are fitted independently to all complete Norway cells;
they are not nested or automatically declared optimal. Each contains:

- `climate_regions_1km.tif`: categorical region IDs, NoData=0, ordered from
  colder to warmer group means. Numeric IDs are not comparable with the
  five-year run.
- `climate_regions.png`, `representatives.gpkg`, `region_summary.csv`.
- `seasonal_profiles.csv`, `annual_profiles.csv`, `decade_profiles.csv`
  (1996-2005, 2006-2015 and 2016-2025) showing group and representative
  conditions, including the added annual indices.
- Five error rasters: daily temperature/precipitation RMSE, wet/dry
  disagreement, and monthly/annual precipitation MAE. Final errors use
  every original daily field, not the timing projection.
- `representation_errors.png`, `model.joblib`, `analysis.json`.

The output root additionally contains continuous 1 km maps of summer mean,
winter mean, `summer_winter_temperature_amplitude_c_1km.tif` and
`frost_days_per_year_1km.tif`. These indicators also appear in each region
summary and annual/decade profile, for both the group and its representative.
They use the same complete-cell mask as the categorical regions.

Add the categorical GeoTIFF to ArcGIS Pro with **Unique Values** and error
rasters with **Stretch**. No ArcGIS project is created. All rasters are
EPSG:32633 with 1,000 m pixels. Coverage excludes Svalbard, Jan Mayen,
neighbouring countries and incomplete source cells. Classification is based
on the MET Norway seNorge_2018 gridded product, not independent observations.
No wind, humidity or snow dataset has been added; those require separately
verified sources and harmonised availability. Preserve MET Norway attribution
and original source licensing/provenance when distributing outputs.

Extraction checks source size/mtime, dates, projection, units, masks and cache
array shapes. Rerunning the same command can reuse completed yearly feature
caches; parameter/source mismatches fail explicitly. A completed analysis
is not overwritten. Use a new output directory for another configuration.

#### Completed 1996-2025 national run

The retained originals total 65,568,496,708 bytes. Both partitions cover
324,188 complete Norway land-centre cells; 148 land cells with missing
source data are excluded. All 10,958 days contribute to the final errors.
The following are means of cellwise errors against each group's real
representative, not errors against independent weather stations:

| Groups | Daily temperature RMSE | Daily rain RMSE | Wet/dry disagreement |
| --- | --- | --- | --- |
| 8 | 2.068 C | 6.356 mm | 20.90% |
| 32 | 1.379 C | 5.189 mm | 16.12% |

Across the six held-out five-year blocks, corresponding mean validation
errors were 2.069 C / 6.140 mm / 20.64% for 8 groups, and
1.395 C / 5.104 mm / 16.09% for 32 groups. Mean partition stability
(adjusted Rand index after omitting a block) was 0.766 for 8 groups and
0.559 for 32 groups: finer detail improves representative similarity but
has less stable boundaries. Candidate comparisons do not establish that
32 is an optimal number of groups.

The 512-pair timing audit found median relative projection-distance errors
of 3.95% for temperature and 4.31% for log precipitation; exact-versus-PCA
distance correlations were 0.991 and 0.961. PCA retained 93.3% and 74.5%
of sampled sketch variance, respectively. These diagnostics quantify the
compression approximation, not station accuracy.

### Norsk leveranse: klimasoner som GeoPackage og sonebeskrivelser

De ferdige resultatene for 1996–2025 kan eksporteres uten å laste ned eller
beregne klimainndelingen på nytt:

```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" .\export_climate_norwegian.py
```

Resultatene ligger i `outputs\klimasoner_1996_2025_norsk`:

- `klimasoner_norge_1996_2025.gpkg`: polygonlagene `klimasoner_8` og
  `klimasoner_32`, punktlagene `representanter_8` og `representanter_32`,
  måneds-/års-/tiårsprofiler og tabellen `feltforklaring`.
- `KLIMASONER_FORKLART.md`: norsk forklaring av metoden, representanter,
  feilmål og begrensninger, og en egen beskrivelse av hver av de 40 sonene
  i de to uavhengige inndelingene.
- `klimasoner_8.png` og `klimasoner_32.png`: norske kart.
- Norske soneoversikter og profiltabeller som CSV med UTF-8 BOM, semikolon
  og desimalkomma for norsk Excel.

I ArcGIS Pro: legg til ønsket polygonlag fra GeoPackage-filen og velg
**Unike verdier / Unique Values** på `sone_id`. `sonenavn` er en kort
beskrivende etikett. `kjennetegn` og `representasjon` gir norske tekster
som kan leses ved å klikke på sonen. Feltnavn bruker ASCII av hensyn til
GIS-verktøy; norske tegn beholdes i tekstverdier og rapport.

Felter som `sommertemp_c`, `vintertemp_c`,
`sommer_vinter_amplitude_c`, `frostdager_aar` og `aarsnedbor_mm` beskriver
gruppemiddelet. `rep_` foran et feltnavn betyr verdien for den faktiske
representantruten, ikke gruppemiddelet. Polygonene inneholder også
årlige periode-/graddags-/nedbørindekser, midlet over ruter og år.

Polygonene følger nøyaktig det klassifiserte 1 km-rasteret i EPSG:32633,
uten glatting eller kunstige forbindelser mellom adskilte områder.
Eksporten kontrollerer gyldig geometri, areal mot antall ruter, alle
representanters sonetilhørighet, og at tilbakerasterisering gjengir alle
originalverdier og NoData-hull nøyaktig.

Sonenavnene er forklarende etiketter etter dokumenterte terskler, ikke en
offisiell klimaklassifikasjon og ikke nye K-means-resultater. Flere soner
kan ha samme navn; sonenummeret er den unike identifikatoren innen hvert lag.
De opprinnelige analysene bevares uendret. Eksisterende norsk resultatmappe
overskrives ikke; bruk `--output-dir` for en ny leveransemappe og eventuelt
`--source` for en annen ferdig kildeanalyse med samme periode og struktur.

Eksporttest:

```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -m unittest test_export_climate_norwegian -q
```

### Mulige kuldehull fra værdata – uavhengig av terreng og klimasoner

`cold_spots.py` undersøker lokale avvik i døgnminimum (`tn`) for 1996–2025.
Ingen klimasoner, høyder eller terrengindekser brukes. De allerede nedlastede
originalene leses lokalt; analysen gjør ingen nye nettforespørsler.

```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u .\cold_spots.py
```

Resultatmappe: `outputs\mulige_kuldehull_1996_2025`. Beregningen kan ta lang
tid. Hvert ferdig kalenderår lagres i `cache` og gjenbrukes ved omstart med
samme kommando og uendrede kilder. Avbrutte år beregnes på nytt.
`status.json` viser sist påbegynte år og til slutt fullført status. Ved feil
viser Python feilmelding i terminalen; `running` etter et avbrudd er ikke bevis
på at en prosess fortsatt kjører. Sluttresultater regnes som ferdige bare når
`analysis.json` med `status=complete` er skrevet.

For en separat Windows-bakgrunnsprosess, start dette fra prosjektmappen
(ikke samtidig som en annen kuldehullsanalyse mot samme resultatmappe):

```powershell
$python = Join-Path $env:LOCALAPPDATA 'ESRI\conda\envs\arcgis-nvdb\python.exe'
$jobb = Start-Process -FilePath $python -ArgumentList '-u', '.\cold_spots.py' -WorkingDirectory (Get-Location).Path -RedirectStandardOutput .\outputs\kuldehull_fremdrift.log -RedirectStandardError .\outputs\kuldehull_feil.log -PassThru
$jobb.Id
Get-Content .\outputs\kuldehull_fremdrift.log -Tail 20 -Wait
```

Prosessen fortsetter når terminal/VS Code lukkes. Hold PC-en våken og
tilkoblet strøm; avlogging eller omstart kan stoppe den. Ctrl+C ved
loggvisning stopper bare visningen. Bruk `Stop-Process -Id <jobbens-PID>`
hvis akkurat denne jobben skal stoppes. En prosess-ID er ikke bevis på
vellykket fullføring: kontroller sluttkvitteringen og feilloggen.

For hvert døgn beregnes rutens minimum minus gjennomsnittet av tilgjengelige
minimumsverdier i en kvadratisk referansering rundt ruten. Tre skalaer brukes:
2–5, 5–10 og 10–20 km, definert som
`indre < maks(|dx|, |dy|) <= ytre` for rutesentrenes forskyvning i kilometer.
Det er **kvadratiske ringer**, ikke sirkulære avstandssoner; hjørnene ligger
lenger unna i rett linje. Den indre blokken, inkludert ruten selv, utelates.
Referansen må ha minst 50 % av ringens potensielle ruter og minst 20 gyldige
ruter. Manglende data/hav inngår ikke som nulltemperatur. Kildedata fra
naboland kan inngå i referansen nær riksgrensen, men bare norske landruter
eksporteres.

Start med:

- `mulige_kuldehull.gpkg`: kandidatpolygoner fra 2–5 km-skalaen med norske
  felter for styrke, frekvens, årstider og gjentakelse. Bruk unike verdier
  på `klasse` i ArcGIS. Polygonene følger 1 km-rutene uten glatting.
- `mulige_kuldehull_ring_2_5km.tif`: kandidatklasse på 1 km-raster.
- `kalde_dogn_prosent_ring_2_5km.tif`: prosent døgn minst 2 °C kaldere enn
  omgivelsene. Behold dette kontinuerlige kartet til terrengsammenligningen.
- `middelavvik_c_ring_2_5km.tif`: gjennomsnittlig lokalt temperaturavvik
  over alle gyldige døgn; negativt betyr kaldere.
- `kuldestyrke_ved_hendelse_c_ring_2_5km.tif`: gjennomsnittlig positiv
  kuldeforskjell bare på døgn minst 2 °C kaldere; NoData uten slike hendelser.
- `mulige_kuldehull.png`, `LES_MEG.md` og `oppsummering.csv`.

Tilsvarende rasterindikatorer følger også for 5–10 og 10–20 km, inkludert
avvik på minst 5 °C, vinter-/sommerfrekvens, største avvik, lengste
sammenhengende kuldeperiode innen ett kalenderår, gjentakende år,
tiårsfrekvenser, spredning mellom år og datadekning.

Kandidatklassene er eksplisitte **praktiske søkekriterier**, ikke
statistisk signifikans:

- 0: Ikke kandidat etter tersklene, men vurderbar.
- 1: Minst 2 °C kaldere enn referansen på minst 5 % av gyldige døgn.
- 2: Som klasse 1, også minst 5 % i minst 24 av 30 tilstrekkelig dekkede
  år, og minst 5 % i hvert av tiårene 1996–2005, 2006–2015 og 2016–2025.
- NoData=255 i klassekartet: utenfor norsk land eller utilstrekkelig dekning.
  Kontinuerlige raster bruker NoData=−9999.

Vurderbare ruter krever minst 90 % gyldige døgn totalt og minst 27 av 30
år med minst 90 % gyldige døgn. Ingen p-verdier eller formell
signifikanstest beregnes. Vinter er januar/februar/desember, sommer juni–august;
årstidsfrekvenser bruker bare gyldige døgn i den aktuelle årstiden.
Lengste perioder brytes ved manglende data og årsskiftet.

Alle raster og polygoner bruker EPSG:32633 og original seNorge-justering.
Dette gjør dem egnet til senere sammenligning med terreng, men høydeeffekter
er **ikke** fjernet. Kandidater kan skyldes vanlig høydeavkjøling, ikke
kaldluftsopphopning. Døgnminimum er ikke en samtidig temperaturmåling i
nabocellene. Relativt kalde dager kan også ha temperatur over null.
1 km-interpolasjonen kan glatte bort små frostlommer, så fravær av signal
utelukker ikke et fysisk kuldehull. Vind/skyer og stasjonsdekning er ikke
kontrollert. Kartet bør brukes som kandidatsøk, ikke som bevis på mekanisme
eller et komplett frostfarekart.

Test:

```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -m unittest test_cold_spots -q
```

Den fullførte nasjonale beregningen leste alle 10 958 døgn på omtrent 65
minutter på utviklingsmaskinen. Ved 2–5 km-skalaen var 324 154 ruter
vurderbare; 49 234 oppfylte kandidatterskelen, hvorav 33 089 oppfylte det
strengere gjentakelseskriteriet. Dette er **ruter, ikke påviste fysiske
kuldehull**. GeoPackage-filen inneholder 15 369 sammenhengende polygoner
etter oppdeling på klasse. Høydeeffekter er ikke fjernet.

47 tester bestod sammen med eksisterende klima-/GIS-/nedlastingstester.
Sluttkontrollen bekreftet eksakt polygon/raster-likhet, gyldige arealer og
geometrier, alle masker, og uavhengig omberegning av klassene for utvalgte
ruter fra alle 30 års mellomresultater. Fem utvalgte ruter ble også
kontrollert med direkte nabouttrekk for samtlige 366 døgn i 1996 og alle tre
referanseskalaene, uten vindusfilteret brukt i hovedberegningen.
`sluttkontroll.json` og `uavhengig_kontroll.json` lagrer kontrollresultatene.

### MEPS og Frost treningsdata
`download_meps_training.py` lager en separat, lokal SQLite-tabell med MEPS-anslag
og Frost-observasjoner ved stasjoner og samme gyldige tidspunkt. Den bruker
MEPS `air_temperature_2m` og `relative_humidity_2m`, og lagrer også biasen
`observasjon − MEPS`. Standard er ett kjør per dag: 00Z med +12 timers ledetid,
matchet mot nærmeste Frost-observasjon innen ±60 minutter. Standardfenomenene
er temperatur og luftfuktighet. Dette er treningsrader for en ny
værkorreksjonsmodell, ikke en ny Frost-stasjonsfeilmodell.

Start med en uke som tørrkjøring:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u ".\download_meps_training.py" --start 2021-10-23 --end 2021-10-30 --dry-run
```
Fjern `--dry-run` for å hente perioden. Fem år, med sluttdato eksklusiv:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u ".\download_meps_training.py" --start 2021-10-23 --end 2026-10-23 --cycles 0 --lead-hours 12
```
Standardkjøringen gjør omtrent én sekvensiell, romlig avgrenset NCSS-forespørsel
per dag; den laster ikke ned de fulle MEPS-filene på titalls gigabyte. Resultat
og fremdrift lagres i `outputs/meps_training_<start>_<slutt>.sqlite`, og fullførte
datoer hoppes over ved omstart. HTTP-feil får begrensede nye forsøk, og svar over
250 MB avvises før de leses inn. NCSS returnerer et begrenset NetCDF3-utdrag;
bare temperatur- og fuktighetsfeltene i stasjonsutsnittet leses. Ikke start parallelle nedlastinger mot METs
delte tjeneste. En enkelt punktforespørsel ga HTTP 503 under utvikling; vent og
kjør samme kommando igjen hvis tjenesten fortsatt er utilgjengelig.

For å inspisere det første vellykkede temperaturutsnittet i ArcGIS, legg til
`--save-first-grid-dir .\outputs\meps_grid_inspection`. Da lagres originalt
NetCDF-utsnitt og en georeferert GeoTIFF med prognosetid, gyldighetstid,
ledetid og enhet som rastermetadata. GeoTIFF-en bruker MEPS sitt opprinnelige
projiserte koordinatsystem; ArcGIS kan projisere laget ved visning. Bruk en ny
mappe for hvert uttrekk. De sist testede arkivforespørslene svarte HTTP 503;
ingen raster opprettes før en forespørsel lykkes.

En liten Oslo-test kan begrenses til stasjoner og MEPS-grid innenfor dette
området (vest, sør, øst, nord i grader):
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u ".\download_meps_training.py" --start 2025-01-15 --end 2025-01-16 --phenomena temperature --cycles 0 --lead-hours 12 --bbox 10.3 59.6 11.2 60.2 --database outputs/meps_training_oslo_20250115.sqlite --save-first-grid-dir outputs/meps_grid_oslo_20250115
```
Dette tidsrommet har Frost-stasjoner i boksen. `--bbox` begrenser både
stasjonsutvalget og NCSS-gridforespørselen. Om arkivet svarer HTTP 503, kjør
samme kommando igjen senere.

Temperaturens stasjonsparpilot kan kjøres separat fra produksjonsmodellen:
```powershell
& "$env:LOCALAPPDATA\ESRI\conda\envs\arcgis-nvdb\python.exe" -u ".\temperature_corridor_pilot.py" --max-pairs 10000 --distance-scale-km 300
```
Par trekkes tilfeldig uten tilbakelegging, med halvnormal vekting etter avstand;
300 km er skalaen som styrer hvor fort vekten avtar. Piloten lagrer både parets
avstand og terrengmål langs korridoren, og bruker romlige holdout-blokker. For
direkte DTM10-profiler kan `--dtm-dir .\data\DTM10` legges til. DTM10-kjøringen
er tregere; øk `--sample-step-m` for færre profiler per par. Resultater lagres
under `outputs/temperature_corridor_pilot/` og endrer ikke den aktive
stasjonsmodellen.

Frost-stasjonene trengs som fasit for å lære forecast-bias. Modellen testes på
holdte-utenfor stasjoner og datoer for å måle overføring til steder uten
observasjoner. MEPS-verdier på ikke-stasjonspunkter kan brukes ved prediksjon,
men uten observasjoner der er de ikke uavhengige treningsetiketter. Scriptet
henter foreløpig ikke nedbør eller snø; akkumulering og målevinduer må kobles
riktig før disse treningsmålene legges til.

### Værmodeller og Forhåndsvisning
Velg **Temperatur**, **Nedbør**, **Snø** eller **Luftfuktighet**, og deretter et
tilgjengelig prediksjonsmål. Luftfuktighet støtter bare **Kontinuerlig**; ved
bytte fra en klassemodell velges kontinuerlig automatisk. De eksisterende femårsmodellene gjenbrukes. Egne
stasjonsplasseringer bevares ved modellbytte. Måleenhetene er:
- Temperatur, kontinuerlig: RMSE i °C.
- Nedbør, kontinuerlig: RMSE for døgnnedbør i mm/døgn.
- Snø, kontinuerlig: RMSE for snødybde i cm.
- Luftfuktighet, kontinuerlig: RMSE i **prosentpoeng relativ luftfuktighet**,
  ikke selve luftfuktighetsverdien eller prosentvis relativ feil.
- Temperaturklasser: feilandel for `< −4`, `−4 til 4` og `> 4` °C.
- Nedbørforekomst: Brier-skår for vått/tørt, der vått er minst 0,1 mm/døgn.
- Snøforekomst: Brier-skår for snø/snøfritt, der snø er minst 1 cm.

Luftfuktighetsmodellen leses fra nattkjøringens ferdige mappe
`outputs/humidity_5y_<oppløsning>m/`. Appen krever modell, metadata og
stasjonsvariabler i samme mappe. Finnes en ferdig modell for kartets oppløsning,
foretrekkes den. Ellers brukes 5000 m-modellen, og til slutt eventuelle eldre
filer direkte i `outputs/`. **Modellgrunnlag** viser kildens ruteoppløsning;
modeller oppdages ved serveroppstart. Start serveren på nytt for å ta i bruk en
ny 2500/1000 m-modell som blir ferdig etter at appen er startet.
en modell bygget med 5000 m-egenskaper som evalueres på 1000 m-ruter er ikke
en ny modell trent på 1000 m. Modell, stasjonsutvalg og mellomlager holdes
separat fra de andre fenomenene. Eksakte Frost-flater for luftfuktighet må
fortsatt forhåndsberegnes; ferdig modell betyr ikke automatisk ferdig grunnflate.

Åpne direkte på luftfuktighet og 5000 m:
```powershell
python station_app.py --phenomenon humidity --cell-m 5000
```
Forhåndsberegn bare luftfuktighet på alle tre rutenett før bruk:
```powershell
python -u precompute_frost.py --phenomenon humidity
```
Dette er helt uavhengig av kartserveren. Ferdige gyldige flater gjenbrukes,
og uferdige flater gjenopptas. Det er fortsatt ingen avstandstilnærming.

Peke-forhåndsvisning er fjernet. Musebevegelser starter ingen beregning.

### Kartlag: vær, høyde og klimasoner

Værnett har et forenklet brukergrensesnitt for vegeiere. Temperaturmodellen
vises som **kontinuerlig**; treningsperioder vises ikke som egne etiketter.
Verktøylinjen har **Usikkerhet**, **Grunnett** og **Reduksjon**.
Fanene for påvirket areal og plasseringsgevinst, forbedringsterskelen,
nøkkeltallpanelet og vegstatistikken er fjernet fra grensesnittet.
Vegfilter, vegfarger, datobekreftelse og eksport fungerer som før. Vegfeil og
varsler om avkortede utvalg vises fortsatt. Metodeinformasjon ligger under
modellmetadata i API og eksport. Underliggende modeller, terskelstandarder, API-er og
eksportdata er uendret; navnene er enklere, ikke nye beregninger.

Kartlagvelgeren viser bare **Værkart**, **Høyde over havet** og **Klimasoner**.
Andre terrengverdier kan fortsatt inspiseres i terrengpopupen; de tilbys ikke
som egne kartlag i velgeren.

Klimasoner leses fra `data/klimasoner.tif` (EPSG:32633, 1000 m, åtte klasser).
Navn og KMeans-metode leses fra rasterets metadata. Disklagrede PNG-er på
400/800/1600 piksler reprojiseres med nærmeste nabo, aldri interpolerte
klasseverdier. NoData (0) er gjennomsiktig. Ingen polygonkonvertering er
nødvendig for denne kompakte rasterfilen. Endret kildefil gir ny bildebuffer.
Fargeforklaringen viser alle sonenavn; vanlig klikk viser originalcellens sone
og `sone_statistikk` fra rastermetadataene uten å legge til en stasjon.
Statistikken er merket som gjennomsnittsverdier for hele sonen, ikke målinger
ved klikkpunktet. Fylkesmasken og aktiv vegkorridor gjelder også her.
Klimasoner er kun et bakgrunnslag: stasjonsnett, RMSE og de to RMSE-lagene i
GeoPackage-eksporten endres ikke.

### Hjelpetekst i appen
**Hjelp**, til venstre for **Skjul stasjoner**, åpner en hvit, rullbar boks.
Introduksjonen står øverst, fulgt av avsnitt for mappene i venstrepanelet.
Lukk boksen med **Lukk**, Escape eller klikk utenfor.

Rediger teksten direkte i `station_app.html`: søk etter `id="help-content"`.
Bruk `<h3>` for seksjonsoverskrifter og `<p>` for avsnitt, og legg til flere
`<section>`-blokker ved behov. Oppfrisk nettleseren etter lagring.
Teksten er felles for appen; den redigeres ikke eller lagres som en del av et scenario.

### Stasjonsmarkører og Terrenglag
Kartknappen **Skjul stasjoner** skjuler Frost-stasjoner og
egne stasjonsmarkører. Knappen skifter deretter til **Vis stasjoner**. Stasjonene
er fortsatt med i beregningen; nett, valgt stasjon, usikkerhet og veglag endres
ikke av synlighetsvalget. Vis stasjonene igjen for å velge dem i kartet.

Velg et felt under **Terrenglag** for å erstatte værflaten i kartet:
- **Høyde over havet**, i meter.
- **TPI**, med lagrede 10 km- og 25 km-egenskaper, i meter.
- **Relieff**, 10 km og 25 km, i meter.
- **Terrengruhet** (`rugstd`), 10 km og 25 km, i meter.
- **Avstand til fjord** og **Avstand til åpent hav**, vist i kilometer.

Terrenglagene bruker data fra valgt **1000, 2500 eller 5000 m** rutenett. Ingen
værprediksjoner eller Frost-forhåndsberegninger kreves for å tegne terrenget.
Høyde, relieff, terrengruhet og kystavstand bruker lineær skala tilpasset lagets
verdier. TPI bruker en symmetrisk, divergerende skala med null i midten.
Kartlegenden viser riktig navn, enhet og verdiområde.

Et klikk på terrengkartet åpner **Terrengverdier** for den evaluerte ruten,
inkludert de øvrige terrengfeltene, rutesenter og oppløsning. Klikket legger
ikke til en stasjon. **Foreslå stasjon her** er en separat handling i popupen.
Samme terrenginspeksjon finnes fra en veglenkes popup.

Dette er eksisterende ruteegenskaper, ikke en ny detaljert høydemodell eller
punktmåling på vegbanen. Det ekstrapoleres ikke til punkter utenfor evaluerte
ruter. Veglag, gjennomsiktighet og valgfri vegkorridor kan brukes sammen med terrenget.
**Vegfarge: Usikkerhet**, vegstatistikk, nettets nøkkeltall og ordinær
ArcGIS-eksport gjelder fortsatt værscenariet, ikke det viste terrengfeltet.
Velg **Værkart** eller et av værkartlagene for å gå tilbake til værflaten.

### Vegnett og Usikkerhet Langs Veg
Det svarte veglaget bruker `data/komprimert_nettverk_med_blindveg_10.gpkg`, lag
`komprimert_nettverk_med_blindveg_10`, skrivebeskyttet. Filen har 2 797 839
lenker i EPSG:25833; 452 498 E/R/F-lenker brukes i bakgrunnsvisningen.
Dette endrer ikke datakilden for poengvisning eller værkartets vegkorridor.
Geometri og romfilter leses med GeoPackage-driveren og
et separat mellomlager; det kreves ingen ombygging av kildedata.

Vegklasse (`roadclass`, 0–10) er forskjellig fra vegkategori (E/R/F/K/P/S).
**Klasse 0–4 er valgt som standard**, med 0 som høyeste prioritet, deretter 1
og videre. Klasse 5 og høyere er fortsatt tilgjengelige i klassefilteret.
Antall lenker per klasse vises når pekeren holdes over klassevalget.

En kildeversjonert indeks lagres i `outputs/station_app_cache/roads/`.
Den inneholder lenke-ID, utstrekning, vegklasse, kategori og normaliserte
vegnummer. Disse leses og tolkes én gang, ikke på nytt for hvert kartutsnitt
eller hver serveroppstart. Filtrering skjer mot lagrede verdier før geometrien
hentes, og lenkene behandles med klasse 0 først. Endret filstørrelse eller
endringstid i kildefilen gir en ny indeks. En skadet indeks bygges opp igjen.
Selve GeoPackage-filen endres ikke, og skjult vegnett laster heller ikke indeksen.

Under **Vegnett** kan du:
- Slå veglaget av og på. Laget er av ved oppstart; metadata og vegfil leses først
  når **Vis** aktiveres. Skjult vegnett starter ingen vegforespørsler.
  Standardutvalget er alle E-, R- og F-veger.
  Innen disse kategoriene er klasse 0–4 valgt; bruk klassefilteret for øvrige klasser.
- Filtrere **vegkategori** og **vegklasse** uavhengig. Kategori leses fra første
  bokstav i hvert vegnummer i `roadid`: `{Rv3}` gir R, og en sammensatt verdi
  som `{E6,Rv3}` matcher både E og R. Kategorien er ikke det samme som
  `roadclass`; klassene 0–10 vises med sine faktiske kildeverdier.
- Avgrense et vegnummer som E6, Rv3 eller F17. `Rv3` og `R3` normaliseres til
  samme referanse. Nummerfilteret gjelder fortsatt gjeldende kartutsnitt.
- Velge nøytral vegfarge eller farge etter **usikkerhet**. Ferjelenker vises stiplet.
- Klikke på en lenke for å se nummer, navn, klasse, type og modellert feil,
  samt velge **Foreslå stasjon her**. Et vanlig vegklikk legger ikke til en stasjon.
- Under **Værfenomen** kan værkartet begrenses til ruter som berører en
  vegkorridor rundt hele vegnettet uten kommunale veger. Standardavstanden er
  2000 m. Valget er uavhengig av om veglaget vises og av vegpanelets filtre.
  Dette er et visningsfilter; selve værmodellen endres ikke.
- Laste ned viste lenker som `vegnett_utsnitt.geojson` med geometri i WGS84,
  originale lenkeegenskaper og beregnede feilverdier. Hver lenke får også
  `weather_dataset`, `weather_unit` og `target_cell_m` for bruk i GIS.

Vegpanelet viser ikke lenger sum lenkelengde, gjennomsnittsfeil, 90-persentil
eller km veg med bidrag. Beregningene beholdes i API- og nedlastingsdata.
Lengder summerer nettverkslenker; parallelle eller overlappende
lenker telles separat og er ikke slått sammen til unik fysisk veglengde.

Feilverdier hentes fra den **eksisterende valgte evalueringsruten**, ikke fra
nye prediksjoner eller direkte fra Frost-observasjonene. Dette er forventet
prediksjonsfeil, ikke lufttemperatur, nedbørsmengde, snødybde eller et varsel
for selve vegbanen. Vegpunkter utenfor en evaluert rute får ingen verdi; de
ekstrapoleres ikke inn fra en fjern rute. Uferdige eksakte Frost-flater gir bare
et vanlig veglag, uten oppdiktede værverdier.

Statistikken er et prøvetakingsbasert anslag: hver synlig lenke deles i jevne
lengdeintervaller og prøvetas midt i intervallet. Intervallet følger
ruteoppløsningen og tilpasses større utsnitt; faktisk prøvetakingsavstand vises
i panelet. Vegfarge og popup bruker lengdevektet gjennomsnittsfeil per lenke,
ikke en kontinuerlig værberegning langs linjen. **Veg med bidrag** gjelder den
lagrede valgte stasjonen, ikke en flyktig peke-forhåndsvisning.

Visningskorridorens avstand er uavhengig av avstanden som brukes til å kalibrere
fargeskalaen. Vegnummer, kategori og klasse i vegpanelet påvirker bare veglaget,
ikke værkartets korridor.
**Vis vegscoreklasse 4** (grønn) og **Vis vegscoreklasse 5** (magenta) kan
slås på uavhengig av hverandre og av det svarte vegnettet. Begge er av ved oppstart
og etter tilbakestilling til standard.

Merkingen **Veg · N ruter** angir at peke-tallene gjelder akkurat dette lokale
utvalget, ikke hele Norge. Gjennomsnittsfeil og reduksjon er rutevektede
gjennomsnitt i utvalget; påvirket areal i km² teller hele ruter der reduksjonen
overstiger terskelen. Ingen matchende veg-ruter gir et tomt kart og «—», og starter
ingen modellprediksjon. Opplysninger i vegpanelet gjelder fortsatt den valgte,
lagrede stasjonen og er lengdevektede; de skal ikke blandes med lokale peke-tall.

Kandidaten og eventuelle egne stasjoner evalueres bare for de etterspurte
ruteindeksene. Lokale prediksjoner har et separat, avgrenset mellomlager med
modell, stasjon og ruteutvalg i nøkkelen. De kan ikke brukes som en full flate.
Når en stasjon faktisk legges til, beregnes den fortsatt for hele nettet, og
scenarioets ordinære nøkkeltall og eksport forblir landsdekkende.

En rute beholdes i korridorvisningen når den kvadratiske ruten berører
vegkorridoren. Ved 5000 m oppløsning blir et smalt vegbånd derfor fortsatt
vist med store ruter; korridorvalget forbedrer ikke modellens romlige oppløsning.
Nettets ordinære nøkkeltall og ArcGIS-rastereksport gjelder fortsatt hele
scenariet. Vegpanelets tall og GeoJSON-nedlasting gjelder bare filtrerte lenker
i kartutsnittet.

Kommunale veger fjernes fra appens indekserte vegdatakopi. Den opprinnelige
datakilden beholdes uendret. Kopien inneholder bare feltene som vegvisningen,
filtreringen og VegScore-komponentene trenger. Første oppstart med en eksisterende datakilde kan
ta lengre tid fordi den filtrerte kopien og den nasjonale vegmasken må bygges;
de gjenbrukes ved senere oppstarter. Nye portable ZIP-filer inneholder bare den
filtrerte vegdatakopien, ikke den opprinnelige geodatabasen eller den gamle
ufiltrerte GeoPackage-filen. Ved gjenbruk av et 500 m stage beholdes originalene
i stage-mappen, men de tas ikke med i ZIP-filen.

Når det ikke finnes forhåndsberegnede veg-rutekoblinger for valgt avstand,
undersøker korridorberegningen nærmeste veg for hver rute i stedet for å
beregne alle veg–rute-par. Ruter som berører korridorgrensen er fortsatt med.

Den tidligere grensen på 12 000 lenker er fjernet. Alle filtrerte lenker i
kartutsnittet tas med. Under zoomnivå 8, når utvalget har mer enn 2000 lenker,
vises et forenklet oversiktslag gruppert etter vegkategori. Alle lenkene beholdes,
men detaljert vegstatistikk utelates. Ved **Usikkerhet** fargelegges også
oversiktslaget: lengdevektede prøver fra den lagrede værflaten samles i opptil
16 feilgrupper per kategori, med blått for lav og rødt for høy feil. Manglende
rutedekning er grå, uten ekstrapolering. Grupperingen forenkler bare visningen;
innzooming viser individuelle lenker. GeoJSON-nedlasting av oversikten inneholder gruppene, ikke
individuelle lenkeegenskaper. Ved innzooming brukes den detaljerte visningen.
Fargegruppene har faste grenser fra hele grunnnettets skala; de tilpasses ikke
lokalt min/maks i kartutsnittet eller feilverdiene etter nye stasjoner.
Nøytrale oversikter mellomlagres uavhengig av værscenario, mens feilfargede
oversikter oppdateres når værflaten endres. Prøvenes kobling til terrengruter
gjenbrukes. Store vegresponser
komprimeres. Første bygging av et nytt landsdekkende vegutsnitt kan fortsatt ta
tid. Visningsgeometrien forenkles etter zoomnivå, mens prøvetaking og korridorvalg
bruker uforenklet geometri.
Vegkilder og modeller endres ikke. Om GeoJSON-lagring ikke støttes i den
integrerte VS Code-nettleseren, bruk samme lokale appadresse i vanlig nettleser.

### Forhåndsberegn Vegkorridorer
For værkartets ufiltrerte korridor kan du lagre én eksakt avstand til nærmeste
veg per terrengrute, for avstander opptil 2000 m:

```powershell
python -u precompute_road_grid.py --cell-m 1000 --nearest
```

For den returnerte 500 m-datamappen:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u precompute_road_grid.py --data-root ".\outputs\portable_build\Remote_500m\data" --cell-m 500 --nearest
```

Avstanden måles fra den kvadratiske ruten til nærmeste veg uten kommunale veger.
En lagret avstand kan sammenlignes direkte med for eksempel 500 eller 2000 m.
Ruter langs kanten av kartutsnittet beregnes separat for å bevare eksakt
avgrensning; fylkes- og vegfiltre bruker også den ordinære beregningen.
Avstander over den lagrede grensen faller tilbake til ordinær beregning. Filen
er knyttet til vegdatakilden og rutenettet, og må beregnes på nytt hvis de endres.
500 m-cachen følger med ved portable bygg som gjenbruker den samme datamappen.

For en mer detaljert kobling mellom **alle** aktuelle veglenker og terrengruter,
som også støtter vegfiltre, bruk:
```powershell
python -u precompute_road_grid.py
```
Standard er eksakt avstand 0–1000 m mellom veggeometrien og hver kvadratiske
terrengrute. Alle vegklasser og kategorier i vegnettet uten kommunale veger tas med. Lenke-ID knytter resultatet
til den lagrede vegklasse-/kategori-/vegnummerindeksen, slik at filtrene kan
brukes uten ny romlig søking over hele vegnettet. Dette krever ingen værmodell
eller Frost-beregning. Ferdige filer gjenbrukes og ligger i
`outputs/station_app_cache/roads/road_grid_*.npz`.

Kommandoen lagrer også gjenbrukbar oversiktsgeometri og faste, lengdevektede
prøvepunkters kobling til hvert terrengrutenett. Nye oversiktsutsnitt slipper
dermed ny GeoPackage-lesing, forenkling og romlig prøvetaking. Grenselenker
kontrolleres mot originalgeometrien for korrekt lengde i utsnittet. Feilfargen
bruker faste prøver på hele kildelenken og grunnnettets faste fargeintervaller,
ikke en ny lokal skala ved hvert zoomnivå. Nærbilder bruker fortsatt originaldata.
Nettleseren har høyst én vegforespørsel i arbeid; ved raske endringer etterspørres
deretter bare det siste valget, uten opphopning av avbrutte beregninger.

Bruk `--cell-m 5000 2500` for bestemte rutenett, `--distance-m 2000` for større
avstander, eller `--dry-run` for å vise kjøringen uten å skrive. Store avstander
kan produsere svært store filer fordi alle relevante veg–rute-par lagres. Kilden,
rutesentrene, radrekkefølgen og rutestørrelsen inngår i nøkkelen.
Mindre korridoravstander kan filtreres fra en større forhåndsberegning.
Manglende filer eller avstander utover lagret maksimum bruker den opprinnelige
romlige beregningen; appen starter aldri denne forhåndskjøringen selv.
Ved kanten av kartutsnittet kontrolleres klipte veger separat, slik at resultatet
beholder samme avgrensning som før. Panorering og bytte av vegfarge alene
forårsaker ikke ny nasjonal kobling av veger til terrengruter.

### seNorge-rutepar på GPU-server

`train_senorge_grid_pairs.py` lager stasjonsuavhengige par fra hele terrengnettet.
Standardinnstillingene beholder den opprinnelige femårskjøringen med 1–50 km
avstand. `prepare_senorge_server.py` eksporterer ferdige etiketter
og 25 symmetriske paregenskaper fra de fullførte årssjekkpunktene:

```powershell
python prepare_senorge_server.py
```

Standardpakken `outputs/senorge_server_package.zip` inneholder én million par
fra 2021–2025, ikke de store NetCDF-arkivene eller lokal konfigurasjon.
Eksisterende ZIP overskrives ikke; velg `--output outputs\et_nytt_pakkenavn`
for en ny eksport. Pakken har kontrollsummer og separat Python-miljø.
Se [serverveiledningen](remote_senorge/SERVER_README.md) for SSH, overføring,
installasjon, faktisk GPU-test, bakgrunnskjøring og tilbakeføring av resultater.

For det nye 30-årsutvalget, kjør:

```powershell
python -u prepare_senorge_server.py --prepare-30yr
```

Dette lager separat `outputs/senorge_server_package_1996_2025.zip`.
Kommandoen bruker eksisterende arkiver (inkludert gjenbruk av 2021–2025),
beregner etiketter uten lokal modelltrening og pakker deretter serverfilene.
La terminalen stå åpen til den er ferdig. Fullførte årssjekkpunkter gjenbrukes
ved samme kommando etter avbrudd. Ingen værdata lastes ned eller kopieres.

- 10 tilfeldig valgte datoer uten tilbakelegging per måned i 1996–2025:
  3 600 datoer, 8 til lineær tilpasning og 2 til evaluering per måned.
- Faste tilfeldighetsfrø gir reproduserbart utvalg. Alle par bruker samme
  datoer og tidsdeling. Eksakt utvalg lagres i `selected_dates.csv`.
- Én million unike, uordnede par. Avstand trekkes fra en halvnormalfordeling
  med skala 25 km og tilfeldig retning, begrenset til 1–100 km.
  Mål utenfor dekkede ruter avvises fremfor å hope seg opp ved kysten.
- Landdekning, nærmeste rute og duplikatfjerning påvirker fordelingen.
  `sampling.json` dokumenterer faktiske antall og kvantiler.
- Tilfeldig månedsdeling måler historisk likhet, ikke fremtidig ytelse.
  Enkeltdager kan overse ekstremvær og kan ikke brukes til sammenhengende
  værperioder. Fordelingen over år og måneder gir bredere, men ikke nødvendigvis
  bedre, dekning enn alle dager i den opprinnelige femårsperioden.

Den portable treneren sammenligner ExtraTrees på CPU og XGBoost på GPU,
begge med/uten potensielle bassenger og med samme romlige holdout.
I den opprinnelige femårspakken inngår 904 766 av 1 000 000 gyldige par i romlig
trening og 56 867 i test; antall for hvert nytt utvalg står i manifestet.
Grensekryssende par utelates fra validering, men brukes i endelig tilpasning.
Serverrapporten viser også feil og antall testpar per avstandsintervall;
sjeldne lange par gir svakt grunnlag for å vurdere ytelse nær 100 km.
Etiketten er symmetrisk lineær prediksjons-RMSE i seNorge, ikke uavhengig
stasjonsnøyaktighet. Bassenger beskrives med tilstedeværelse, antall, maksimal
hel-bassengdybde og sum av hele volumene for individuelle overlappende bassenger.
Disse er potensielle terrengforsenkninger, ikke nødvendigvis vann.
Gjengitte helvolumer kan ikke summeres på tvers av ruter.
Eksisterende Station Lab-modeller og profiler endres ikke.

For et separat eksperiment kan 30-årspakken bruke målet
`absolute-error-p95` med `--target-metric`. Da predikeres gjennomsnittet av de
to retningenes 95-persentiler for absolutte evalueringsresidualer i stedet for
RMSE-etiketten.
Alle datoer, egenskaper, romlige delinger og CPU/GPU-modeller beholdes.
Dette er ikke i seg selv et kalibrert 95 % prediksjonsintervall for temperatur,
og den eksisterende modellen/installeringen skal ikke overskrives med forsøket.
Se [serverveiledningen](remote_senorge/SERVER_README.md) for separat pakking
og kjøring.

#### Nytt eksperimentelt valg i Værnett

Etter servertrening kopieres `senorge_results_30yr.tar.gz` til `outputs`.
Installer den portable modellen og kontroller resultatene:

```powershell
python install_senorge_station_model.py
python -m pip install -r requirements-station-senorge.txt
python precompute_frost.py --phenomenon temperature --kind senorge30 --cell-m 5000 2500 1000 --prepare-maps
```

Installeringen trekker bare ut eksplisitt valgte UBJ-/JSON-/holdout-filer,
kontrollerer datasett, modell og prediksjoner, og lar serverens store
joblib-modeller ligge i arkivet. XGBoost kjører prediksjon på CPU i appen.
Gamle profiler beholdes; det nye valget er `temperature:senorge30`,
«Temperatur (kontinuerlig)». Start appen på nytt etter installering.
Forhåndsberegning kreves separat ved hver kartoppløsning; manglende flater
vises som manglende, de erstattes ikke med gamle modellresultater.
Under første klargjøring kan 5 000/2 500 m-visning brukes mens 1 000 m
beregnes separat. Dette er faktisk evaluering på de grovere terrengrutene,
ikke oppskalering av et gammelt kart. Følg fremdriften i
`outputs/senorge_30yr_app_precompute.log`; velg 1 000 m når loggen viser
`PNG KLAR 1000 m` og `Alle valgte eksakte Frost-grunnflater er lagret`.
La maskinen være våken. Ved avbrudd gjenbruker samme kommando de fullførte
flatene og atomiske del-sjekkpunktene. Appen og forhåndsberegningen kan
kjøre samtidig.

Romlig test av de fire modellene (45 757 par):

| Modell | RMSE for parets feiletikett (°C) | MAE (°C) | R² |
| --- | ---: | ---: | ---: |
| ExtraTrees uten basseng | 0,27096 | 0,20402 | 0,5373 |
| ExtraTrees med basseng | 0,26977 | 0,20185 | 0,5413 |
| XGBoost uten basseng | 0,26853 | 0,20215 | 0,5455 |
| XGBoost med basseng | 0,26652 | 0,19967 | 0,5523 |

Valget av XGBoost med bassenger er basert på lavest RMSE på dette ene
romlige testutvalget, ikke en ny uavhengig modellutvalgstest.
Bassengforbedringen er liten (ca. 0,75 % lavere RMSE for XGBoost);
den dokumenterer ikke en kausal bassengeffekt. R² gjelder prediksjon
av feiletiketten, ikke forklart variasjon i dagstemperaturen.
Ved 75–90 km finnes bare 15 romlige testpar, og ved 90–100 km ingen.
`outputs/senorge_30yr_station_model/inspection.json` inneholder kontrollert
RMSE, MAE, R², systematisk avvik og 95-persentilen av absolutt avvik.

Appen bruker stasjonsterreng som endepunktsegenskaper, men modellen ble
trent på terrengruter, ikke faktiske stasjonsmålinger. Overføringen til
stasjoner har derfor ikke uavhengig stasjonsvalidering.
Bassenger beskriver den nærmeste faste 1 km-ruten rundt hvert endepunkt,
også ved visning på 2 500 eller 5 000 m; det gjøres ingen grov volumaggregering.
Der værtreningens rutetabell ikke dekker et endepunkt, beregnes samme positive
arealoverlapp fra bassengpolygonene, ikke en standardverdi for manglende data.

Etter eksplisitt brukervalg ekstrapolerer dette eksperimentelle appvalget
også utover 100 km. En vedvarende advarsel vises og følger ArcGIS-/vegeksport.
Samme 1 km-rute gir feil 0 som en separat identitetsregel, ikke en målt
modellprestasjon; dette gjelder ikke avstandsekstrapolasjon ellers.
De numeriske paregenskapene beregnes i treningspresisjon før float32-
konvertering. De gamle modellene bruker fortsatt sine opprinnelige
float32-inndata og ingen nye bassengvariabler.
Plasseringene i det foreslåtte nettet er ikke reoptimalisert for denne modellen.

### Sesongmodeller og terrengbasert dyp læring

Et separat eksperiment lager fire etiketter fra de samme datoene og parene:
vinter (desember–februar) og sommer (juni–august), hver for døgnmiddel (`tg`)
og døgnminimum (`tn`). Det beholdes 10 datoer per måned og år i 1996–2025:
720 til lineær tilpasning og 180 til evaluering for hvert sesongmål.
RMSE-etiketten beregnes på nytt innen sesongen; den gamle helårsetiketten
kan ikke deles opp i ettertid fra samlede årsmomenter.

```powershell
python -u prepare_seasonal_terrain.py
```

Kommandoen gjenbruker `outputs/senorge_grid_pairs_1m_1996_2025_random`,
leser de eksisterende værarkivene og DTM50 i nabomappen `..\dtm50\data`,
og lager separat `outputs/senorge_seasonal_terrain.zip`.
Ingen av de fungerende appmodellene eller kartene overskrives.
Forberedelsen krever flere GB og kan ta lang tid; diskbudsjettet kontrolleres
før generering. Fullførte årsdata og terrengblokker gjenbrukes etter avbrudd.
Ikke overfør ZIP før `outputs/senorge_seasonal_terrain/cache/complete.json`
er skrevet; selve ZIP-navnet publiseres først etter fullført pakking.
Status ligger i resultatmappens `status.json`.

Terreng lagres én gang per rute: tre nordorienterte 64×64-vinduer ved 50,
100 og 500 m pikselstørrelse (3,2/6,4/32 km). Høyde relativt til endepunktet
lagres som float16 i kilometer, sammen med en separat gyldighetsmaske.
Dette er rasterrepresentasjoner ved angitt oppløsning, ikke eksakt 10 m-
terreng eller bevis for lokale temperaturforskjeller.
`terrain_coverage` i manifestet rapporterer faktisk dekning og helt tomme vinduer.

Serveren sammenligner fire XGBoost-modeller med en symmetrisk terreng-CNN
og en CNN som også får de 25 tabellvariablene. Samme par brukes i alle
eksperimenter. Geografisk trening, validering og test er separate.
Alle endepunkter nærmere enn 16,5 km fra en 100 km-blokkgrense utelates,
slik at 32 km-terrengvinduer ikke overlapper på tvers av delmengdene.
Antall gjenværende par står i manifestet; det er ikke én million treningspar.
Early stopping og normalisering bruker ikke testdelen.
Denne strengere testen kan ikke direkte sammenlignes med den gamle testen
uten buffer. Nettverket har ingen garanti for å slå XGBoost.

Se [egen norsk serverveiledning](remote_senorge/SEASONAL_TERRAIN_README.md)
for overføring via `its.npra.io:1959`, separat miljø med PyTorch 2.5.1 CUDA12.4,
GPU-kontroll, bakgrunnstrening og tilbakeføring av resultater.
CNN er testet på CPU i et isolert miljø, ikke installert i appens GIS-miljø.
Serverresultatene er nå kontrollert mot det opprinnelige manifestet, testparene,
fasitverdiene, modellkontrollsummer og ny beregning av test-RMSE.
Kombinert CNN har lavest test-RMSE for alle fire mål:

| Mål | XGBoost | Terreng-CNN | Kombinert CNN |
|---|---:|---:|---:|
| Vinter døgnmiddel | 0,2630 | 0,2615 | 0,2449 |
| Sommer døgnmiddel | 0,3940 | 0,3616 | 0,3325 |
| Vinter døgnminimum | 0,3127 | 0,3096 | 0,2915 |
| Sommer døgnminimum | 0,2587 | 0,2516 | 0,2338 |

Tallene er feil i prediksjon av representasjons-RMSE, ikke feil i en
temperaturprognose. Modellvalg er gjort etter denne ene romlige testen;
den er ikke lenger en helt uavhengig test av modellvalget.
Sommermiddel underestimeres med ca. 0,09 grader i gjennomsnitt.

Installer fra den returnerte `outputs/seasonal_terrain_results.tar.gz`:

```powershell
python install_seasonal_station_model.py
python -m venv --system-site-packages .venv-seasonal-app
.\.venv-seasonal-app\Scripts\python.exe -m pip install -r requirements-station-seasonal.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv-seasonal-app\Scripts\python.exe precompute_frost.py --phenomenon temperature --kind seasonal --cell-m 5000 --prepare-maps
.\.venv-seasonal-app\Scripts\python.exe station_app.py --port 8912 --cell-m 5000
```

Opprett miljøet med prosjektets GIS-Python, slik at eksisterende GIS-pakker
arves. PyTorch og nyere SymPy installeres bare i det separate miljøet.
Fire egne temperaturknapper i appen velger vinter/sommer og døgnmiddel/minimum.
Den gamle helårsmodellen er bevart. `--kind seasonal` velger alle fire
sesongmål for forhåndsberegning; bruk `--cell-m 2500 1000` for finere kart.
1 km er terrengrutenettets tilknytning, mens 5/2,5/1 km er kartets
evalueringsavstand. Et manglende finoppløst mellomlager rapporteres eksplisitt.

Appen bruker CPU, delte terrengkodinger mellom sesongmodellene og samme
symmetriske nettverk som serveren. Den trenger de lokale terrengarrayene i
`outputs/senorge_seasonal_terrain`, samt terrengrastrene i undermappen `cache`
for endepunkter utenfor treningsrutenettet. Behold disse filene.
For landruter brukes de ferdige treningsvinduene; manglende treningsruter
genereres fra samme DEM med 1 km-rutens høyde. Utenfor hele landrutenettet
brukes avrundet 1 km-senter og endepunktets høyde som referanse; dette er
utenfor treningsstøtten og står i modelladvarselen. Manglende DEM bruker
alltid gyldighetsmaske, ikke en antakelse om havnivå.
Numerisk kontroll mot direkte CPU-nettverk gir identiske resultater i
kontrollutvalget; største avvik fra returnerte GPU-prediksjoner var 0,000423
grader over alle 13 340 testpar (fire mål). Dette er ikke et nytt mål på modellens
prediksjonsnøyaktighet.

### Reference-conditioned uncertainty footprints (annual + seasonal)

`prepare_footprint_training.py` prepares a separate map-prediction experiment.
One reference location conditions six 64 x 64 output maps at 1 km:
annual mean/minimum, winter mean/minimum and summer mean/minimum.
Annual targets use all 12 months; they are not an average of the two seasons.
Each reference is labelled against every available terrain/weather cell in its
64 km square using the original 3,600-date seNorge plan. Default preparation
uses 256 train / 64 validation / 64 test reference scenes, about 1.07 million
reference-to-cell comparisons per target before missing-data filtering.

The residual U-Net sees an 80 km terrain scene at 100 m, signed relative
coordinates and a reference marker. Unlike independently pooled endpoint
encoders, it preserves the spatial relation between valleys and intervening
ridges. A native 10 m branch covers 1.28 km around the reference; this is NOT
10 m coverage of the entire scene or 10 m weather truth. A spatial decoder
also receives the existing 25 pair attributes, including whole-basin features.
Validity masks are explicit. The square covers up to approximately 45.3 km
from its reference, not the old model's full 100 km distance range.

All reference centres lie at least 41 km inside their assigned 100 km block.
Entire input scenes, not just output pairs, stay inside their geographic
partition. These block groups have been used in prior experiments and are
not newly untouched test geography.

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u prepare_footprint_training.py
```

Default output: `outputs/senorge_footprint_training.zip`. Preparation has
yearly atomic label checkpoints and resumable terrain scenes. Wait for
`cache/complete.json` before transferring; archive allowlist/CRC/checksums
are verified. Old weather originals, training packages and app models remain
unchanged.

The portable `remote_senorge/train_footprint.py` compares residual U-Nets of
width16/24 with/without reference10m inputs over three seeds, validation-only
model selection, learning-rate reduction, early stopping and six XGBoost
baselines. The default CNN budget is 40 hours, checked between epochs;
evaluation/baselines add time. Completed epochs save optimizer/scheduler
state for interrupted-run resume. No GPU fallback is accepted.
Only the selected CNN is evaluated on test maps, not every candidate.
Centre identity pixels are excluded from learned-error comparison metrics.

See the [footprint GPU guide](remote_senorge/FOOTPRINT_README.md) for transfer
via port1959, server installation, weekend execution and result retrieval.
This new experiment is not yet a Station Lab profile. App integration needs
evaluated returned results and a map-inference adapter, with explicit handling
outside the finite footprint. A best-single-station footprint minimum is not
joint assimilation of multiple station observations.

#### Expanded geographic footprint experiment (v2)

`prepare_footprint_expanded.py` preserves the first experiment and prepares
**10,000 / 1,000 / 1,000 unique training/validation/test reference locations**.
It uses new 150 km regional blocks with full-scene interior buffers and
5 km reference-selection strata. The output remains 64 km square at 1 km
resolution, with 80 km terrain context and the same six annual/seasonal targets.

Native 10 m-derived relief, slope, quadrant structure, low-terrain and coverage
descriptors are spatial inputs at every available land-grid cell throughout
each scene and halo, not just at the reference. Shared atlases reconstruct
scenes on demand instead of duplicating terrain for all references. Yearly
sampled-weather and per-reference label checkpoints bound local disk/RAM use.
The 31 reference-to-cell inputs retain the original 25 features and add
reference/target absolute elevation, ocean distance and fjord distance in
metres, with training-only normalization. Completed v2 packages can be upgraded
using `prepare_footprint_expanded.py --repackage` without recalculating labels
or terrain. Retransfer the rebuilt ZIP and start a fresh training output:
25-input checkpoints are not compatible with the new 31-input network.
No application model is changed automatically.

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u prepare_footprint_expanded.py
```

Wait for `outputs/senorge_footprint_expanded/cache/complete.json` and complete
status before transferring `outputs/senorge_footprint_expanded.zip`.
The new partition is not an untouched test relative to earlier experiments;
compare models on these exact splits, and inspect regional scores rather than
treating overlapping cells as independent examples. See the
[expanded footprint guide](remote_senorge/FOOTPRINT_EXPANDED_README.md)
for progress checks, transfer, GPU training and return commands.

The returned absolute-input footprint CNN is available in a separate
experimental **Footprint-CNN · 1 km** station-by-station view. It does not
replace national network scenarios. Install the verified returned archive,
then start only this view while leaving existing apps running:

```powershell
.\.venv-seasonal-app\Scripts\python.exe install_footprint_station_model.py
.\.venv-seasonal-app\Scripts\python.exe station_app.py --footprint-only --cell-m 1000 --port 8914
```

Open `http://127.0.0.1:8914/footprint`. Select a station (including SN56960
and SN56895), change between six targets, or click a custom reference.
Prediction/export resolution is fixed1,000m,64x64 cells,EPSG:25833; areas
outside the footprint or missing land cells are NoData, not zero uncertainty.
The GeoTIFF+JSON export preserves native cell values and model provenance.
The full app also exposes this view via its header after restarting.
The **Kartvisning** selector also offers a Frost-only full-network map,
selected-station RMSE reduction and a separate loss-of-coverage map. All use
the same six targets and native1km predictions. The full-network value is
the minimum representation RMSE of the stations whose64x64km square covers
that land cell, not a joint assimilation. Outside this union: NoData. The
square is not a64km radius.

For a selected station, contribution is **minimum RMSE without that station
minus minimum RMSE with all stations**. Tied/redundant stations have zero
contribution. Where removing the only covering station leaves no prediction,
the numeric contribution is NoData, with coverage loss shown separately.
Click a cell in a network view to see covering Frost stations ranked by RMSE
and each station's removal contribution. Click a ranked station to display
its contribution map. Custom references remain available only in single-map
mode and are never added to the Frost network.

Network preparation is resumable and computes all six targets once. The UI
starts it when needed and displays progress; alternatively prepare first:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u station_footprint_network.py
```

Only Frost stations whose snapped reference is a supported seNorge land cell
are included. Model/dataset/catalogue fingerprints invalidate stale caches.
Exports include native1km GeoTIFFs for the chosen view, station coverage
counts and best-station indices, plus an index-to-Frost-ID catalogue in JSON.
Reduction exports also include a separate loss-of-coverage raster.
National PNG previews use nearest-neighbour sampling capped at2200pixels
per dimension; this does not reduce prediction or GeoTIFF resolution.
All footprint heatmaps and legends use a blue-to-red scale (blue = low,
red = high), including the binary coverage-loss view. The background tiles
are grayscale; heatmaps and station markers retain their colors.
Returned v2 improves both summer targets against its matching XGBoost baseline
but not annual/winter targets; app warnings and the
[expanded guide](remote_senorge/FOOTPRINT_EXPANDED_README.md) report these results.
Never interpret the view as a temperature forecast, independently calibrated
sensor accuracy or a joint assimilation of the whole station network.

### Weather-regime and terrain-connectivity valley pilot

This separate experiment tests **winter nighttime station transfer** before
attempting a national learned covariance/uncertainty model. It never modifies
the original Frost database, trained footprint models or either running app.

- Western-Norway bounding box:5.4–7.3E,60.9–61.9N. The prepared catalogue
  currently contains65terrain-supported stations.
- Three DJF winters:2022–23 training,2023–24 validation,2024–25 final test.
  Observations are sampled at00,03,06UTC: **not daily minimum temperature**.
- Geographic challenge groups are assigned to the nearest anchor within6km:
  SN56960(Haukedal) is test, SN55820(Fjærland–Bremuseet) is validation; other stations
  train. These are geographic groups, not verified whole-valley boundaries.
  Accepted QC data may further
  reduce availability; preparation never fabricates missing observations.
- All training endpoints exclude validation/test stations. Only training
  stations act as observed sources, including during validation/test.
  The nearest-six source shortlist is formed only from stations with accepted
  training-winter QC readings, then limited to connected land paths within45km.
  Validation/test availability never qualifies a new source station.
- Downloaded Frost observations retain source/sensor ID, time definition,
  level, quality code and rejection reason. OnlyqualityCode0 observations,
  degC,PT0H,PT1H and compatible2m metadata pass. Code1 is excluded because it
  includes corrected/interpolated values. Sensor choice uses only metadata
  coverage from the training winter; ambiguous duplicates are rejected.
  The old value-only Frost database is not a replacement for these data.
  Gaularfjell(SN56895) returned10minute qualityCode6 observations on the
  diagnostic night and was therefore replaced as validation anchor, not
  admitted by relaxing the quality requirement. It remains a prediction-only
  diagnostic where no accepted observations exist.
  The initial replacement SN57340(Jølster–Kvamsfjellet) had no accepted
  training-winter readings and failed the training-only sensor-selection rule.
  The user-approved SN55820 replacement has accepted data in all three winters.
  Anchor changes used availability/QC only, before fitting or examining test
  prediction errors; this remains an exploratory pilot, not a pristine final
  certification test.
- MEPS forecasts use the preceding12UTC run, giving12/15/18h leads. Only
  bounded regional temperature, grid-relative wind components and total
  cloud fraction are downloaded. Snow state is **not** included in v1.
  MEPS may assimilate *earlier* readings from held-out stations: this avoids
  contemporaneous target leakage, not all historical background dependence.
- Terrain connectivity uses a four-neighbour1km land graph. A minimax path
  finds the lowest possible maximum terrain height between two cells. Heights
  are25th-percentile summaries of100m DEM samples. A15kmhalo allows detours;
  connections outside this region are unknown. Sea/NoData is not traversed.
  Unsupported straight paths have an explicit indicator, not an invented
  ridge height. This is a terrain-graph feature, **not a GNN or10m weather truth**.

Preparation is resumable with per-day checksum receipts, bounded response
sizes, retries and separate provider locks. Do not run two copies of the same
provider stage. The two providers may run independently.

```powershell
# Real-data single-night smoke tests:
.\.venv-seasonal-app\Scripts\python.exe prepare_valley_pilot.py --stage regimes --day 2024-01-15

# Existing pilot: update geographic split labels without repeating downloads:
.\.venv-seasonal-app\Scripts\python.exe prepare_valley_pilot.py --stage groups

# Set the client ID in your own terminal; never paste it into chat or source.
$frostId = Read-Host "Frost client ID" -AsSecureString
$env:FROST_CLIENT_ID = [System.Net.NetworkCredential]::new("", $frostId).Password
.\.venv-seasonal-app\Scripts\python.exe prepare_valley_pilot.py --stage observations --day 2024-01-15

# Full preparation (271winter nights across the three winters):
.\.venv-seasonal-app\Scripts\python.exe -u prepare_valley_pilot.py --stage regimes
.\.venv-seasonal-app\Scripts\python.exe -u prepare_valley_pilot.py --stage observations
Remove-Item Env:\FROST_CLIENT_ID

# Run only when both provider stages report ready:
.\.venv-seasonal-app\Scripts\python.exe -u train_valley_pilot.py
```

Data: `outputs\valley_regime_pilot`. Each provider has its own `status.json`.
The trainer refuses incomplete or corrupted data. It compares:

1. Unfitted **forecast-difference transfer**: source observation plus target
   forecast minus source forecast.
2. **Terrain**: observed source temperature, calendar and static endpoint terrain.
3. **Regime**: same inputs plus source/target forecast temperature, wind, cloud
   and source forecast residual.
4. **Graph+regime**: same inputs plus terrain pass/ridge features and wind,
   cloud and residual interactions.

The three fitted baselines use the same ExtraTrees capacity and fixed settings.
This isolates information-content changes before expensive neural training.
Selection uses the first half of validation nights; interval calibration uses
the other half. A90% radius uses the **maximum error per calibration night**,
not thousands of correlated pair rows as independent samples. Distribution
shift between valleys means nominal coverage is not guaranteed; empirical
pair and night-simultaneous coverage are reported.

Results: `outputs\valley_regime_pilot_results\report.json`, reloadable models,
QC rejection counts, sensor choices and test predictions. Scores include
per-station error, below−5C temperatures and forecast calm-clear conditions
(wind<=2m/s,cloud<=0.3), an inversion-risk proxy, not proof of inversions.
Do not promote a model just because a synthetic test passes or because it
reproduces the1km seNorge labels better. The pilot must first show improvement
on accepted held-out Frost observations and difficult nights. A learned
positive-semidefinite covariance/network-information footprint is a later
stage, not implemented or claimed by this pilot.

#### First real valley-pilot results

Both providers completed271winter nights. The QC-selected data yielded3714
training,1072validation and877test station-transfer rows. The final test is
Haukedal on78nights, not877independent weather events.

| Approach | Selection-validation RMSE (C) | Haukedal test RMSE (C) | Test RMSE below -5C (C) |
|---|---:|---:|---:|
| Forecast-difference transfer | 4.885 | 2.781 | 3.653 |
| Terrain | 5.646 | 2.599 | 3.378 |
| Weather regime | 5.123 | 2.216 | 2.772 |
| Graph + regime | 4.368 | 2.382 | 3.365 |

Validation selected **graph+regime**, although weather-regime alone had the
lowest test error. Do not retrospectively select on this test. The regime
model's test RMSE is14.7%lower than terrain-only, supporting further investigation
of weather inputs. Graph features did not improve this held-out location over
regime-only. On the42forecast-calm-clear pair rows, forecast-difference transfer
RMSE was2.293C versus3.059C for regime-only and3.805C for graph+regime:
the targeted inversion-risk case is not solved.

Night-max calibrated90%interval radii were approximately8.56–10.16C, with
97.4–100%night-simultaneous test coverage. These are too broad to establish
useful calibrated local uncertainty; the validation/test distribution shift
needs investigation. These observation-transfer scores are not comparable
to the CNN's error in reproducing seNorge RMSE labels. Results and checksummed
models are saved separately, and neither app was replaced.

#### Visualize the saved pilot in Station Lab

Open `http://127.0.0.1:8914/valley-pilot`, or select **Dalpilot · testresultater**
in the footprint header after restarting the footprint server. The view is
read-only and does not replace the existing1kmfootprint heatmaps.

- Select a UTC timestamp or move the time slider. The map shows the held-out
  target observation and the contributing source stations. Lines and source
  markers use blue–white–red for **prediction minus target observation**:
  blue means too cold, red too warm, saturated at±6C. The basemap is grayscale.
  No raster, interpolation or national prediction is implied.
- Switch between the four methods. Each line is a separate station transfer,
  not a combined-network prediction. Select a source marker or table entry
  to inspect its winter timeline against the observed target temperature.
- The timeline compares all methods at identical available source/target
  timestamps, with no lines drawn across missing nights. Click a point to
  inspect that timestamp on the map.
- Filter forecast-calm-clear conditions or target temperatures<=−5C.
  Calm-clear eligibility depends on both source and target forecasts, so a
  timestamp may have fewer source transfers in that filter. Summary metrics
  are recomputed for the displayed target/filter; source metrics cover only
  that source. The validation-selected model remains explicitly identified.
- CSV export contains all saved test transfers, not just the active filter.
  The backend checks input/model provenance and recomputes saved test scores
  before displaying results. Wide calibrated interval radii and limitations
  are visible separately; they are not rendered as local uncertainty maps.

#### Portable overnight neural experiment

`prepare_valley_overnight.py` verifies the completed QC/forecast pilot and
exports its fixed paired samples with an allowlisted standalone GPU trainer.
No credentials, local configuration, GIS dependencies or raw archives are
packaged; existing data, results and apps remain unchanged.

```powershell
& .\.venv-seasonal-app\Scripts\python.exe .\prepare_valley_overnight.py
scp -P 1959 .\outputs\valley_neural_training.zip havfyh@its.npra.io:~/
```

See [GPU training and resume commands](remote_senorge/VALLEY_NEURAL_README.md).
The63trial sweep compares terrain MLP, forecast-residual correction, three
weather-gated experts and a graph-feature gated variant, across three widths,
three seeds and optional training-only emphasis on cold/calm-clear examples.
Normalization uses training rows only. Every trial uses the same untouched
selection-validation nights; calibration and test do not select settings.
The forecast-transfer baseline can win if neural trials are worse.

The requested40hours is a **maximum accumulated training budget**, not a
minimum; the3714training rows may finish much sooner. Every completed epoch
saves optimizer, RNG, current/best weights and time for exact-command resume.
Completed runs are not extended. Package/result checksums and a process lock
guard against incompatible data or competing workers.

This is an exploratory reuse of the already-inspected Haukedal test, not a
new final certification test. Weather-gated experts are learned statistical
regimes, not physical inversion labels; graph descriptors are not a GNN or
learned covariance. No national uncertainty map or automatic app replacement
is claimed. Return the separate results archive for verification and comparison,
especially cold/calm-clear cases and usefulness of calibrated intervals.

#### Synthetic western temperature sandbox

Open `/valley-sandbox` in the footprint app (port8914), also linked from the
observed valley-results page. This separate what-if viewer loads the returned
validation-selected `regime_gate_w64_s2026_risk1` signed-transfer model.
Select one of11QC-supported pilot stations, an archived00/03/06UTC weather
timestamp, a source-temperature slider from-20to+20C, and one to three hops.

The rooted breadth-first tree uses connected land pairs within45km. Children
are visited nearest-first with deterministic ties; each station receives one
prediction and never feeds back into an already visited station. Each next
hop uses its parent's **synthetic** temperature and forecast-residual features,
not that parent's observed reading. This is path-dependent, not assimilation
or a nationally optimized information network.

Markers show absolute synthetic temperature in blue-white-red; the basemap is
grayscale. Dashed edges denote later hops. Unreached stations remain listed;
there is no interpolation,64km raster, invented numerical domino uncertainty,
or automatic model replacement. Model outputs are not clipped to the slider
range; only display colors saturate. Errors are shown explicitly.

Only direct transfers to Haukedal have held-out validation evidence. Reversed
transfers, held-out sources, slider values and synthetic multi-hop sources can
all be outside the fitted distribution. Do not interpret these temperatures
as observations, operational forecasts or calibrated uncertainty. The earlier
CNN learned precomputed seasonal RMSE labels directly; it never contained a
signed temperature predictor.

#### National winter station-transfer dataset

`prepare_national_dataset.py` prepares a **separate national dataset**, not a
replacement for the western pilot or either app. It uses the existing terrain
atlas and Frost station catalogue; the full QC observation and MEPS downloads
must run in your own terminal. No credential is written to the dataset.

Prepared scope:

- All existing Frost `inside_norway` catalogue entries in57.8–71.3N/4–32E,
  including terrain-supported coastal islands but excluding Arctic territories.
  This is a snapshot of our existing catalogue, not a newly refreshed Frost
  source inventory.
- Five complete DJF winters:2020–21 through2024–25,451nights,00/03/06UTC.
- Actual point temperatures with strict Frost qualityCode0, hourly PT1H/PT0H,
  degC and2m/unspecified height. The old database's QC-less readings are not used.
- Existing absolute elevation, ocean/fjord distance, relief and TPI descriptors;
  local1km terrain-connectivity graph derived from100m DEM summaries.
- Previous-day12UTC MEPS runs,12/15/18h leads;2m temperature,10m wind and cloud.
  Adaptive station bounding boxes contain<=4096forecast cells per request.
  MEPS remains2.5km weather input, not1km observations or a1km accuracy claim.

Initialization in this workspace found1342terrain-supported catalogue entries
and115terrain exclusions. Accepted QC coverage will be lower; the final number
is deliberately not guessed. Sample MEPS metadata dates in all five winters and
bounded real2020/2024weather requests at the southern/northern catalogue ends
were verified. This does not establish availability of every archive day.

**Run the complete resumable preparation from PowerShell:**

```powershell
& .\.venv-seasonal-app\Scripts\python.exe .\prepare_national_dataset.py --stage all
```

If `FROST_CLIENT_ID` is not already set in this terminal, set it without echoing
or saving it first:

```powershell
$secret = Read-Host "Frost client ID" -AsSecureString
$env:FROST_CLIENT_ID = [System.Net.NetworkCredential]::new("", $secret).Password
Remove-Variable secret
```

Keep this terminal running; this is not a detached download service. Stop with
Ctrl+C and rerun the same command to resume. Completed station batches, forecast
metadata/subsets/days and daily sample shards are reused only after checksum
verification. A process lock prevents competing national writers. A transient
provider outage fails visibly with preserved progress; missing weather/QC is
never replaced with invented values. Documented Frost404/412 on the initial
observation page are explicitly logged/cached as no-data batches and counted
in the QC audit; authentication, server and pagination errors still fail.
Allow hours to days depending on provider
latency/retries, and reserve several GB of free disk space.

Missing MEPS runs: a404 for the expected run's `.das` is excluded only after
the official, populated daily catalogue confirms that exact run is absent.
The checksum-protected daily receipt records the catalogue URL, content hash,
listed paths and exclusion reason. Catalogue/server failures or a run still
listed in the catalogue fail visibly rather than becoming an exclusion.
Confirmed missing nights produce empty `(0,39)` feature shards, never substituted
weather or an earlier forecast run. The final manifest reports available/
excluded night counts and full exclusion evidence; the451requested nights
therefore need not all contain observations paired with forecasts. Geographic
splits, forecast lead times and minimum final coverage checks remain unchanged.
Rerun the same command after an interruption; existing downloads are preserved.

The stage order is terrain -> archive metadata preflight -> observations ->
QC audit and frozen regional splits -> forecasts -> graph/pairs/daily samples.
For a bounded Frost smoke before the full run:

```powershell
& .\.venv-seasonal-app\Scripts\python.exe .\prepare_national_dataset.py --stage observations --day 2020-12-15
```

This requests all supported stations in64-station batches for only one night.
Individual stages are `terrain`, `preflight`, `observations`, `audit`, `freeze`,
`regimes`, `build`; forecasting requires frozen splits, and building requires
all forecast nights. Do not run multiple provider copies for this dataset.

**Evaluation is fixed before fitting any model:**

- One sensor per station is selected by accepted metadata timestamp coverage in
  the three training winters only, with deterministic ties. At least30training
  nights are required for eligibility; no temperature-error selection.
-100km geographic blocks, not claimed physical valley regions. Two southern/
  northern-stratified blocks each are selected for validation and final test
  using deterministic QC availability, with>=2well-supported stations per
  candidate block. Blocks overlapping the inspected western pilot are not
  candidates.10km surrounding stations are excluded from training.
- Training uses only training-group stations in2020–21,2021–22,2022–23.
  Validation-group2023–24nights alternate between selection and calibration.
  Test-group2024–25is reserved for final evaluation.
- Held-out stations never act as observed sources. Up to six nearest connected
  training sources are kept from a12-station QC-supported shortlist within
  **100km**. This intentionally expands the pilot's45km pair limit; report
  distance-stratified accuracy when fitting. Paths are restricted to a local
  source-radius square plus15km halo, not globally optimal national routes.
- Unsupported/QC/buffer/no-source stations remain in explicit exclusion reports.
  Insufficient station/night coverage in any held-out block fails preparation;
  splits are not silently relaxed or chosen by temperature performance.

Inspect progress:

```powershell
Get-Content .\outputs\national_station_dataset\status.json
```

Final output is `outputs\national_station_dataset`:
`qc_audit.csv`, `station_splits.csv`, `station_coverage.geojson` (WGS84 points,
loadable in ArcGIS), exclusions, provider receipts, `pairs.npz`, and451
checksum-protected `samples\YYYY-MM-DD.npz` shards. Each shard contains39
terrain/weather/graph features, target-minus-source labels, observed
temperatures, source/target IDs, timestamps, split and calm-clear metadata.
No normalization or model fitting happens during preparation.

Only `dataset_manifest.json` plus status `dataset_ready: true` marks the complete
verified dataset ready. The manifest reports actual rows, nights, target stations
and held-out block coverage. Earlier `stage_complete` statuses are not readiness.
The national sharded schema is **not compatible with tonight's three-winter
valley trainer**; a national trainer with region/target-night balancing and
leave-region-out comparisons is the next step. Changed sample-builder code or
inputs are rejected rather than mixed with already-built shards.

Earlier pilot results already exposed2024–25weather in western Norway. Holding
out new blocks improves spatial evaluation but does not make that weather period
entirely untouched. MEPS may assimilate earlier readings from held-out stations.
This dataset does not yet include snow/history inputs, multi-station fusion,
newly computed drainage features, or national uncertainty rasters.

#### National signed-transfer and multi-station training

The ready national dataset can now be exported with:

```powershell
& .\.venv-seasonal-app\Scripts\python.exe .\prepare_national_training.py
```

The exporter verifies terrain/split/forecast provenance and every daily sample
checksum, then creates `outputs\national_transfer_training.zip` without raw
provider archives,local configuration,credentials or GIS dependencies.
It refuses overwrites. The excluded2022-01-16night remains in provenance.

See [remote setup,40-hour training,resume and result return](remote_senorge/NATIONAL_TRANSFER_README.md).
This new trainer predicts signed target temperatures and evaluates residual
RMSE after correction. A perfectly fixed-10C difference has zero corrected
error,unlike the old uncorrected RMS-difference footprint.

The first experiment includes copy-source,pooled constant/elevation-offset,
forecast transfer and a bounded weighted weather-tree control,plus16neural
trials: residual MLP,weather-gated experts,graph-feature gated experts and
joint masked multi-station fusion,at widths64/128and seeds42/2026. Fusion uses
up to6simultaneous source readings per target timestamp,without target
observations in its inputs. Equal-weight classical fusion controls can win.
Separate pair/fusion winners are chosen by geographic-block/target-night-
balanced selection RMSE. Calibration/test are opened only after selection.

40h is maximum accumulated fitting time,not minimum runtime or strict wall
time. Checkpoints preserve optimizer,current/best weights,RNG,history and
elapsed time after each epoch; exact-command resume rejects changed settings.
No app/model is automatically replaced. This is a fixed regional-split first
experiment,not leave-region-out CV or a national covariance/uncertainty map.
Fusion weights are not validated removal importance or independent precision
weights. National1km deployment requires further independent validation.

#### Residual uncertainty before continuous temperature maps

The next requested display is a64km-square1km land-cell temperature map for
an archived winter time,with an optional-20to+20C source slider. The user chose
to train a separate residual-error model **before** displaying uncertainty.
The existing temperature models/apps are therefore not replaced at this stage.

`prepare_national_uncertainty.py` creates
`outputs\national_uncertainty_training_v2.zip` using verified national examples
and the frozen validation-selected single-station residual MLP.
See [uncertainty training and result return](remote_senorge/NATIONAL_UNCERTAINTY_README.md).

The revised v2 design uses five unseen-target rounds **plus** five geographic
rounds. Unseen targets are excluded as fitting targets AND observed sources,
while nearby nonheld sources remain. Only these primary out-of-fold residuals
fit the error-scale model. Separate geographic rounds exclude whole100km
blocks and10km buffers as both targets and observed sources,providing novel-
region diagnostics rather than biasing the scale-training sample toward
long-distance boundary transfers. Coverage fractions,missing targets,distance
bands and per-block/per-winter errors are explicit. The selected architecture is
refitted for its original best validation epoch count,not selected using OOF/
test errors. Only geographic out-of-fold training residuals fit the absolute
error-scale model. Calibration uses maximum normalized error per entire
calibration night,targeting90%simultaneous evaluated station-transfer coverage.
No future-temperature forecast or fusion-error calibration is claimed.
Test interval widths/coverage are reported by block,target,distance and weather
subset. The existing two calibration/two test blocks remain limited; training
winter error breakdowns are descriptive,not new independent temporal coverage
evidence. Use v2,not the superseded regional-only training ZIP.

The final test was already inspected,so this residual experiment is exploratory.
Nightly station-pair calibration does not establish simultaneous coverage over
4096unobserved cells,synthetic slider values or domino propagation. The report
explicitly marks continuous-map uncertainty unvalidated. Return and inspect
coverage/widths before app integration; no calibrated map is enabled merely
because training completes.

#### Experimental continuous temperature map

Open `/temperature-map` in the footprint app on port8914,linked from the
footprint,pilot and station-domino views. This uses the **completed national
single-source temperature model**,not the older raw-difference footprint CNN.
Select a source from all1342terrain-supported stations,an archived DJF date and00/03/06UTC.
All stations are persistent clickable markers,including outside the current
temperature window. **Vis hele stasjonsnettet** zooms to the national catalogue;
**Vis alle stasjoner** toggles markers. Clicking a marker selects it,checks its
selected QC sensor observation and prepares its map if a source value is available.
Missing selected-sensor observations are explicit; no zero is fabricated.
Only392stations were original training sources. Other sources are flagged as
additional extrapolation,not represented as validated extensions of that network.
Click **Last og vis64 x64km** to prepare bounded cell weather and land paths.
The default source is Njos(SN55770),2024-12-15at03UTC.

Each supported cell in the64km square receives a direct temperature transfer.
Cells are exactly1000m in EPSG:25833. Local115km graph context,1km DEM
quantiles,sea barriers and39feature ordering match national training.
The new bounded MEPS downloads use the same previous12UTC run,units,forecast
leads,nearest-neighbour2.5km background and grid-relative wind bearings.
Source weather is checked against the original station archive; saved
weather/terrain caches in `outputs\national_temperature_map` are checksum-
verified and reused across sliders and all three hours of a date.

By default only the selected sensor's accepted QC observation supplies the
source temperature,including dates where that station had no paired examples.
Missing observations produce an explicit error,never a fabricated zero or
silent switch to synthetic mode. Explicitly enable the synthetic source slider
to try-20through+20C; source-residual features are recomputed for every value.
Sea,unsupported land and disconnected cells remain NoData. Click a colored
cell for temperature. The basemap is grayscale,with a fixed blue-white-red
temperature ramp saturating outside-20to+20C.

GeoTIFF download preserves the exact64x64grid,UTM projection,1000m cells,
NoData and model/source/time/experimental provenance. It can be added directly
to ArcGIS. Displayed polygons follow transformed UTM cell corners; there is no
map interpolation or smoothing. All cells are experimental extrapolations.
The1km output is not a claim that the2.5km MEPS background or predictions have
1km accuracy. This is archived contemporaneous transfer,not future weather.
The returned v2residual-error model now supplies **experimental intervals**:
128error trees using39features plus predicted temperature,a0.1C scale floor,
and the original45-night simultaneous calibration multiplier5.360000748.
The verified exploratory test reached81.1%night coverage,below the90%target,
with median radius8.28C and90th-percentile radius15.49C. These are radii,
not RMSE or standard deviations. No calibrated grid coverage,fusion or domino
is claimed; synthetic sliders and nontraining sources are extrapolations too.

**Vis eksperimentelle intervaller** includes cell radius and lower/upper bounds
in popups; **Kartfarger** switches between temperature and radius. The radius
ramp saturates at25C. The UI always shows the coverage limitation. When enabled,
GeoTIFF exports four bands: temperature,radius,lower and upper bound,all in C
with identical NoData masks. Disabling intervals preserves the one-band export.
Model/load/checksum errors are explicit,never replaced with guessed intervals.

Results live in `outputs\national_uncertainty_returned`. The original GPU
sklearn version differs from the GIS environment. Rather than changing GIS
dependencies or running unsupported pickle inference in the app,the verified
scalar trees are persisted as portable numeric arrays. Conversion matched
all22852returned test scales/intervals and original tree inference; CPU mean
predictions and the calibration factor were also verified. Runtime inference
uses original float32tree comparisons without loading a pickle. Original
model/report hashes and array checksums are checked before use. This is
prediction/provenance verification,not new scientific validation.
Original apps/model selections remain unchanged.

#### Joint national uncertainty and station contribution prerequisite

The national joint map is a separate requested experiment,not an aggregation
of the single-source intervals above. `prepare_joint_uncertainty.py` builds
`outputs\joint_uncertainty_training_v1.zip`. Follow
`remote_senorge\JOINT_UNCERTAINTY_README.md` for the40hmaximum GPU run.
It reuses the frozen selected fusion temperature model and trains a separate
error-scale model on unseen-target joint residuals. Five primary and five
regional diagnostic refits remove held station observations from all fitting
source slots. Full configurations and valid leave-one-reading-out predictions
are evaluated explicitly. The calibration target includes all joint locations
and deletions in each evaluated night; no source-independence shortcut is used.

Station contribution is the signed change in modeled interval radius and
temperature after removing that reading without replacement,not the fusion
attention weight or a causal/additive station value. Removing the last available
source gives NoData. The first requested national timestamp is2024-12-15at03UTC.
Future map inference may offer all1342supported catalogue sources with an
available selected-sensor QC observation,but still combines at most six nearby
connected readings per cell. Sources outside the original training network,
unobserved cells and synthetic readings remain flagged as extrapolations.
No joint nationwide layer is enabled until returned joint results are verified.

The returned joint run is now independently reproduced by
`national_joint_uncertainty.py`: original package/code/data hashes, 45-night
calibration factor, all 27,425 test configurations, all station-removal
temperature/radius changes and reported coverage. Numeric portable tree
traversal matches the trusted forest on all 300,000 OOF scale samples.
`outputs\joint_uncertainty_returned\verification.json` records this verification;
the runtime `JointUncertainty` uses checksummed arrays, not sklearn pickles.
No GIS dependency upgrade is required.

Returned full-group temperature RMSE is 2.511 C. Full joint night coverage is
87.8% (79/90), while simultaneous full-and-deletion coverage is 73.3% (66/90),
below the 90% calibration target. Median interval radii are respectively
7.744 C and 8.160 C, not RMSE estimates. Of 22,842 valid deletions, 11,347
reduce the predicted radius and 11,481 increase it; negative changes are
retained, not relabelled as beneficial station contributions. These are model
sensitivities, not causal station values or evidence of independent sources.
Model/result reproduction does not certify national cell coverage. Existing
prediction layers remain unchanged. A separate `/model-evaluation` page on the
experimental app displays the verified held-out results for both the joint
station and seNorge GPU models, with explicit warnings.

The evaluation page uses `station_model_evaluation.py` and
`station_model_evaluation.html` / `.js`. It offers model/date/hour or daily
variable filters, observed and predicted temperatures, actual signed errors
and RMSE, predicted seNorge RMSE or joint interval radius/observed coverage,
regional diagnostics and filtered CSV export. Clicking a joint target reveals
all valid single-source-removal changes and source markers; negative changes
remain visible. Removing the only source is explicitly NoData.

seNorge cells use exact 1 km UTM33N corners. When more than one source-target
evaluation exists at a cell, temperature is the mean of those separate
evaluation predictions, not a fusion-model result. Actual RMSE and predicted
RMSE are pooled using square-root mean-square; per-source predictions are
available on click or through the source filter. Blank areas have no evaluated
samples; no interpolation or live prediction occurs. The two model RMSEs are
not head-to-head scores because datasets/targets differ.

### Nationwide predicted joint uncertainty (not the evaluation samples)

`/national-uncertainty` on the experimental app is a separate inference map
built by `national_joint_map.py`. The default timestamp is 2024-12-15 at 03UTC.
All 426 available selected-sensor QC readings are used as candidates, including
65 sources outside the frozen training network, explicitly flagged as
extrapolation. For each native 1 km atlas cell, it shortlists the nearest
12 available readings within 100 km, computes the same local 115 km minimax
land graph and straight-route features used in training, and retains at most
six nearest terrain-connected sources. This available-catalogue selection is
an inference extension beyond the frozen training-source selection.

The verified joint temperature mean consumes the same 39 signed pair features,
source-temperature plus archived MEPS delta anchors, and frozen normalization.
Its verified joint error-scale forest then predicts the interval radius.
Every valid single-source deletion is also predicted without replacement,
preserving signed changes in temperature and radius. Removing the only source
is NoData. No independent-source variance reduction or radius averaging is used.

The whole national native grid is processed, not just test locations or a
station-centred 64 km patch. Sea, unsupported terrain and cells with no
connected observed source remain NoData. Zoomed maps show exact 1 km UTM33N
cell corners; the national overview aggregates predictions into 20 km display
tiles to bound browser load. Overview tiles can also cover unpredicted/sea
areas, so they must not be interpreted as continuous full-tile predictions.
The GeoTIFF retains the exact 1 km predictions and NoData, with five bands:
temperature, experimental radius, lower bound, upper bound, source count.

Preparation status is visible in the app; checksummed geometry, archived
weather, predictions and deletion effects persist under
`outputs\national_joint_map`. The default completed result is restored after
app restart. This is a nationwide **prediction** of experimental intervals,
not a nationwide coverage validation. Test nightly coverage remains 87.8%
for full inputs and 73.3% when deletion scenarios are included, below 90%.
This map uses the joint Frost/MEPS model, not the seNorge RMSE model or
held-out evaluation CSVs.

The national page also has a **Remove one observed station reading** selector.
Choose a source to colour the whole map by signed uncertainty/temperature
change (without minus full), switch to the prediction without that station,
or compare with the full network. Blue/red denote negative/positive changes;
unused-source cells have exactly zero change. Grey cells lose the sole source
and become NoData, not zero uncertainty. Clicked-cell source rows include a
"Remove across map" button. National counts describe affected, unchanged,
widened, narrowed and lost cells. `national_station_removal.py` uses the saved
no-replacement predictions, not a retraining or nearest-source replacement.
The separate download exports the five-band 1 km without-station scenario.
The national overview uses median signed radius changes and mean temperature
changes per display tile; zoom for exact cell changes.

All 1,342 terrain-supported catalogue stations are shown as persistent,
clickable markers above the grid. Green markers have an available selected-
sensor QC reading; clicking removes it and selects the signed radius-change
map. Grey markers have no reading at the prediction timestamp: clicking
explicitly explains that they were not used and restores the full network.
No station-name knowledge is required. A station toggle and restore-all button
are provided; marker availability is refreshed with the prepared timestamp.
The radius model is not monotonic in source count, so a negative removal
change is a predicted sensitivity, not evidence of better accuracy or a bad
station. This limitation is explained in the interface.

The interactive national map now uses Web Mercator PNG tiles rendered directly
from the native 1 km UTM prediction raster with nearest-cell resampling at
every zoom. It no longer switches between 20 km overview blocks and 1 km
polygons. Clicking land resolves the exact native cell and opens its details
without changing zoom; clicking a station selects removal without panning.
Resize preserves the current view. The legacy GeoJSON overview endpoint remains
available, but is not drawn by the interactive map. The native GeoTIFF export
is unchanged.

### New seNorge GPU virtual-reference map

On the experimental app (port 8914), open `/senorge-transfer`. This is the
returned **seNorge-only** two-stage GPU training, not the Frost/MEPS national
joint model and not the original raw temperature-similarity model. The original
port 8912 app is unchanged.

Choose one of the 3,600 cached historical dates (1996--2025), daily mean `tg`
or daily minimum `tn`, then click any supported land cell to place a virtual
reference. Its archived seNorge value supplies the reference temperature; an
optional manual override is available. No station catalogue or live station
reading is involved. Moving the virtual reference or changing date/variable
recomputes the map. Uncheck placement to inspect exact target-cell predictions.

The adapter uses the exact training feature builder, verified portable ridge,
selected `temperature_w64_s2026` network and `error_w128_s2026` squared-error
network. It first predicts signed temperature transfer, then returns
`sqrt(predicted MSE)` as **corrected predicted RMSE**, not a calibrated interval.
Target archive values are used only for the optional cell comparison, never
as inference features. A single displayed signed error is not measured RMSE.
The known reference cell copies its supplied temperature and has zero transfer
error as an explicit identity rule, not a forced floor on other predictions.

Native 1 km raster tiles and a two-band GeoTIFF (temperature, corrected RMSE)
cover supported land within **46 km**, the new training sampler's range.
This is deliberately not the old model's 100 km range. There is no automatic
zoom on reference/cell click and no 20 km overview aggregation. Selecting a
different reference invalidates old prediction keys and exports explicitly.
Map startup verifies the returned model/report and historical input checksums;
the large historical cache check can take time on the first request.

Held-out test actual temperature RMSE was 1.2120 C and pooled predicted RMSE
1.2515 C. These are historical seNorge grid-transfer diagnostics, not certified
real-station accuracy, a weather forecast, or nationwide uncertainty coverage.

### New seNorge GPU model over the Frost station network

Open `/senorge-network` on port 8914 for the **original app's network method
with the new seNorge temperature/error models**. All 1,342 Frost catalogue
locations are visible and clickable. A reference uses the archived seNorge
`tg` or `tn` value in the supported native land cell containing the station,
not an actual Frost sensor observation. Off-grid or missing-archive stations
remain visible with their explicit ineligibility reason.

The network result is the **minimum corrected predicted RMSE over all eligible
station references within 46 km** of each target. This reproduces the original
best-source coverage method; it is not a jointly trained fusion of readings.
The displayed temperature comes from the selected best-error source. The
calculation retains the best and second-best station/error/temperature at
every native cell, including ties. Clicking a station displays without-minus-full
RMSE: it is nonnegative where both predictions exist, zero where another source
already wins (or ties), and NoData where removal loses the sole eligible source.
Lost-source cells are grey on the removal map and NoData in the GeoTIFF.

Norway-wide means the entire supported atlas is evaluated; cells with no
eligible reference within the new training's 46 km range remain transparent
NoData, with counts shown explicitly. There is no extrapolated fill. This map
is not certified real-station uncertainty or an interval radius. Dates and
variables are selectable; native PNG tiles preserve 1 km predictions at every
zoom and clicks do not move the view. Full/without/change exports include
scenario and removed-station metadata.

Checksummed network predictions persist in `outputs/senorge_network_map`.
The first calculation verifies the model/archive and evaluates every eligible
station footprint; subsequent identical date/variable/network/model requests
restore the saved best/second-best arrays rather than refitting or rescoring.

The original `station_app.py` interface offers **seNorge GPU · langtids-RMSE
døgnmiddel** and **seNorge GPU · langtids-RMSE døgnminimum** alongside its existing
models. These replace its earlier single-day GPU entries: they are **not fixed
to 2024-12-15**. For every station-target pair, evaluate the trained models
on all **3,600 cached sampled dates, 1996--2025**, square each predicted RMSE,
average those squared errors in float64, then take the square root.
Dates are equally weighted, not weighted by calendar gaps or repeated to represent
every historical day. Daily mean and minimum are pooled independently.
Only temperature-sensor stations eligible in the original app's continuous
temperature setting participate: `air_temperature` observation availability
over its existing five-year window, merged with usable temperature terrain
features. The broader national catalogue is not used as the station set.
The sensor list/coordinates, availability window and terrain checksum are
recorded in the network receipt and cache identity.
Only after pooling are stations ranked at each target: the best and second-best
**fixed long-term** reference stations are saved, with no daily switching.
The same reference temperature and seasonal features used by training are
evaluated per date; target temperatures are never inference inputs.
No retraining, date-specific fitting or interval recalibration occurs.
This historical model-based estimate includes previously used training/test
dates and is not a new held-out validation or certified station accuracy.
Existing station selection, removal/restoration, moving/adding proposed
stations, comparison scopes and ArcGIS export are reused. Pair predictions for
edited/proposed sources use that same all-date pooling and exact native terrain
within 46 km. Multiple removals search the remaining references using pooled
scores. Missing any reference date excludes that reference from all-date
coverage rather than silently pooling a shorter period.

Run `.venv-seasonal-app\Scripts\python.exe -u senorge_long_term.py` to prepare
both variables. This is inference over billions of historical pair/date
combinations and can take many hours on CPU. Each complete unique source-cell
footprint is atomically saved with checksums in `outputs/senorge_long_term`;
restarting the same command restores completed footprints. An OS file lock
prevents concurrent writers. The final national best/second-best arrays and
date-count/aggregation receipts are saved separately for each variable.
The original app reports queued/running/failed/missing status in its existing
precomputation panel and does not substitute a single-day map while pending.
It automatically loads a finished network on its next status refresh.
Moving/adding a previously uncached source requires its complete all-date
footprint calculation; cached sources and deletions reuse saved scores.

For GPU inference without retraining, run `prepare_long_term_gpu.py`. It creates
`outputs/senorge_long_term_gpu_v1.zip`: only the eligible source-cell historical
temperatures, native terrain/footprint indices, frozen portable model weights
and checksums, not the full weather archive or Frost observation database.
See [GPU calculation instructions](remote_senorge/LONG_TERM_GPU_README.md).
The remote runner disables TF32 and preserves float64 normalization, ridge
prediction and pooled-error accumulation. It can resume at complete source
footprints, including completed CPU footprints shipped with the package.
`import_long_term_gpu.py` verifies all output files and independently reconstructs
one nonidentity pair over every date for every source/variable on CPU before
publishing any cache receipts. It rebuilds station rankings locally in original
station order. Benchmark-only output cannot be imported as national results.
The importer uses the same exclusive computation lock as CPU precomputation;
stop the CPU precompute process before importing. Keep the local package
directory unchanged until import; its identity must match the running app.

To test a completed variable before the entire GPU job finishes, import an
immutable snapshot with `import_long_term_gpu.py <archive> --checkpoint --variable 0`
(`0` = daily mean; `1` = daily minimum). The archive must contain
`long_term_results/run_config.json`, `long_term_results/status.json` and every
completed `.npz` plus `.json` source receipt for the selected variable. The final
`report.json` is not required in explicit checkpoint mode. The importer still
requires every eligible source, every sampled date, valid identity/checksums,
and independent all-date CPU pair verification. It never fills missing sources
by launching CPU inference. Only the selected variable's national map is
published; the other remains pending. Copy only completed immutable source files,
not `.pending` files, while the other variable continues running. A subsequent
normal full import verifies/reuses the first variable and publishes the second.
Do not modify/repackage the running GPU package to create a checkpoint.

For these new entries only, unsupported source/target pairs carry NoData,
finite-cell summaries exclude unsupported cells, exact-cell source diagnostics
exclude ineligible pairs, and exported rasters write explicit NoData. Existing
model behaviour is unchanged. Restart the original app to load the new registry
entries; there is no dependency upgrade or change to existing model files.

The original app's **Vis bare RMSE over** slider is a display-only filter:
zero disables it; a positive threshold shows only cells with finite RMSE
strictly greater than that value. It intersects the selected road-corridor
mask. The slider maximum is the actual finite maximum of the active baseline
or scenario surface within that road mask, not the colour scale. Changes to
the scenario, model or road selection refresh this limit and clamp the selected
threshold; empty/pending surfaces disable the slider. It also intersects the
existing cold-basin mask. Baseline maps use baseline RMSE;
scenario/change maps use current scenario RMSE, and hover previews use the
preview scenario's RMSE, not the magnitude of the change. Terrain and placement
layers are unaffected. The filter does not change station calculations,
summary statistics or road vector overlays.

**Eksporter til ArcGIS** downloads `norway_rmse.gpkg` with two EPSG:25833
polygon layers: `rmse_norway` contains every finite cell of the active national
RMSE surface; `rmse_visible` applies the current strict RMSE threshold, selected
road-corridor cell mask and, when enabled, the depth-filtered Kuldehull polygon
mask (clipped to its displayed viewport). Each polygon retains `cell_id` and
`rmse`. Unsupported/NoData cells are omitted, not encoded as zero. The active
evaluation resolution is preserved. Baseline view exports baseline RMSE;
uncertainty/change views export current scenario RMSE, never change magnitudes.
Transient hover previews, terrain and placement layers are not exported.
Filter and model identifiers are stored as layer metadata. Even when the
filtered selection is empty, both layers exist. The legacy raster/station ZIP
remains available separately at `/api/export`.

`verify_senorge_transfer_results.py` reproduces all 300,000 returned assessment
and test temperature/RMSE predictions. Returned checkpoints contain NumPy2
state incompatible with the GIS runtime; the trusted ridge parameters and
neural best-state tensors were converted in an isolated matched-version
environment and verified before the unchanged app uses the evaluated CSVs.
The missing remote environment-version file is recorded in
`outputs\senorge_transfer_gpu_returned\verification.json`. All result/data/code
checksums matched the original package. Test actual RMSE is 1.212 C, predicted
pooled RMSE 1.251 C; regional and error-bin miscalibration remains visible.

#### seNorge-only corrected temperature transfer and learned RMSE

The original Station Lab has a collapsible **Fylker** selector using
`data/Basisdata_0000_Norge_25833_Fylker_FGDB.gdb`, layer `fylke`.
All 15 counties are selected by default. Compact two-column checkboxes,
All/None buttons and the selected-count summary support multiple selection.
County selection does not remove source stations or hide station markers:
stations outside the selected area still contribute to its uncertainty.
Thin county boundaries are always drawn above map/road overlays and below
station markers, including when no counties are selected. The shared borders
are merged and simplified by 100 m for display only, then served as cached,
compressed GeoJSON; precise county masks and exports use unchanged polygons.
They do not intercept map clicks.
Weather, combined, terrain, placement and Kuldehull overlays are clipped to the
selected polygons. Roads are geometrically clipped before their displayed
length/error statistics and GeoJSON download are produced. RMSE limits and map
summary metrics respect the county selection. Empty selection shows no data
overlays but leaves station markers visible. The GeoPackage's `rmse_visible`
layer intersects county boundaries with the other masks; `rmse_norway`
deliberately retains the full national surface as its unfiltered comparison.
Legacy raster ZIP exports use the county cell mask too.

**Stasjonsdata siden** is optional (blank by default). For each active
phenomenon, only stations with a known `last_time` on or after the selected
date are eligible. Unknown dates, including unverified ProssStasjon reference
locations, are excluded when this filter is enabled. This means at least one
observation since the cutoff, not continuous reporting or a completeness test.
The existing observation-availability history is used; no live Frost fetch is
performed. Uncertainty and station-removal comparisons use the filtered
network; clearing the date restores eligibility. It combines with the SVV
station restriction and applies independently to each checked weather layer.
Changing the date first calls a metadata-only count endpoint. The app shows,
per checked layer, how many stations the date excludes, how many remain, how
many have unknown dates, and removals/restorations relative to the current
filter. Nothing changes until **Bekreft og beregn** is pressed; **Avbryt**
keeps the existing network and map. Confirmation is rejected if the scenario
has changed since the count was obtained.
The **Værstasjoner** dropdown starts on **SSV-stasjoner** in the production app.
It replaces the separate SVV checkbox; the other choices are **Alle MET-stasjoner**
(the original eligible Frost network, without the SVV restriction) and **Tomt nett**.
The selection changes the network and SVV restriction in one action.
Hypothetical
user-added stations remain scenario proposals, not historical observations.

Kuldehull, Fylker, Vegnett and Værstasjoner use matching collapsible headings.
Kuldehull depth controls are initially collapsed; the Vis controls for
Kuldehull and Vegnett remain accessible on their heading rows. Collapsing a
section does not reset its filters. The technical information block has been
removed, while the map, map buttons and legend remain. **Gjennomsiktighet**
runs from 0% (opaque) to 100% (transparent); the initial 15% preserves the
previous 85% map opacity.

Precalculation now prepares **both all-station and SVV-only networks by default**:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u precompute_frost.py --all-weather-maps
```

Use `--station-network all` to retain the previous all-station-only workload,
or `--station-network svv` to prepare just the SVV variant (the underlying
all-station profile is still required). Model, grid and experimental-model
selection flags work as before. For the ProssStasjon model specifically:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u precompute_frost.py --phenomenon temperature --kind senorge30_pross --prepare-maps --station-network svv
```

SVV eligibility uses the same road-name prefix rule as the app. Each subset
stores exact best/second-best errors and winning original station indices,
with resumable checkpoints and 400/800/1600 px PNGs when maps are requested.
The app loads these caches for SVV-only baseline and single-station removal;
no subset prediction is needed after a compatible cache has been prepared.
`--cached-only` never computes missing SVV surfaces. Arbitrary date cutoffs
are not precomputed: only confirming a new date can trigger that calculation.

The app and `precompute_road_grid.py` use `data/ProssExport.gdb`, layer
`Vegnett_VegScore_full` (read-only). The selected layer is streamed once into a compact
indexed GeoPackage under `outputs/station_app_cache/roads/`, retaining only map,
county and score-component fields. This preserves fast county filtering, spatial
index queries and on-demand component reads without loading other GDB layers.
The source path and its file sizes/mtimes identify the cache; changed GDB inputs
get new caches rather than reusing the former GeoPackage source's road/grid links.
Existing road-category/route filters, corridor masks, RMSE previews and filtered
GeoPackage exports use this source. Class checkboxes select the supplied
`VS_Total_klasse` (1--5), not NVDB road class. Their labels show the actual
VS_Total range in the file; missing classes/scores have an explicit unknown
option and grey colour. All score classes are selected initially; road categories
initially remain E/R/F. Detailed roads retain exact `VS_Total` and
`VS_Total_klasse` in popups and downloaded GeoJSON. Choose yellow--red,
purple--yellow or light--dark blue sequential score palettes, neutral or
modelled RMSE colouring. Score palettes use the full source's fixed numeric
VS_Total range so filtering does not rescale colours. At overview zoom,
roads are grouped by category and score class with length-weighted scores;
zoom in for individual link scores. VS_Total is a separate prioritization input,
not predicted RMSE and does not change model inference or station ranking.
Existing non-score RoadNetwork inputs remain supported.
`build_portable.py` validates the current `ProssExport.gdb` and bundles an
indexed copy of this layer without municipal roads, including when `--reuse-stage`
is used. The original source is preserved. Its smoke test checks the
running portable app's selected road layer. Rebuild the executable after runtime
code changes; `--reuse-executable` rejects outdated road/app/launcher code.
Previously built ZIPs are not changed automatically.

SVV/Vegvaer stations now come exclusively from the MET/Frost network. A name
must start with a Norwegian road reference (for example `E18`, `FV307`, `RV13`
or `KV5738`); matching ignores case and accepts spacing before the number.
Embedded references and place names without a leading road number do not match.
The same rule controls network eligibility, markers, offline precalculation and
the fixed colour-scale reference, for every phenomenon. No station-location file
or distance radius is used. Observation dates and sensor eligibility remain the
MET/Frost values. The previous `temperature:senorge30_pross` ID redirects to
`temperature:senorge30`; there is no separate location-based model in the UI.
Rebuild compatible subsets with `precompute_frost.py --all-weather-maps`.
If precipitation has no eligible road-prefixed stations, its Vegvaer network
is empty; as explicitly chosen, its fixed colour reference then uses the
all-MET baseline instead, with a visible notice. Downloading and retraining
with complete ten-minute totals can add genuine Vegvaer sources. No stations are
silently reclassified or given precipitation sensors.

Road attributes are indexed by `fylkesnr` in the persistent lookup cache.
County selections query only the matching row groups, and low-zoom views can
read compact per-county prepared geometry instead of the national geometry file.
Only the fields needed for filtering, VS_Total and popups are read from the
source. Exact county clipping remains in effect for roads, corridors and exports.
Build these display caches with `precompute_road_grid.py --help`.
Climate-zone legends and zone-average popup statistics are collapsed initially;
expand their summaries when the full information is needed.

The expanded PyTorch GPU successor is prepared with
`prepare_senorge_transfer_gpu.py`. Its verified portable archive is
`outputs\senorge_transfer_gpu_v1.zip`. See
[GPU instructions](remote_senorge/SENORGE_TRANSFER_GPU_README.md) for setup,
model selection, cross-fitting, budget/resume semantics and result return.
Defaults use 2,400 reference cells, 16 targets/source and 1.4 million sampled
rows across the same frozen geographic/year policy. Signed-temperature neural
models compete against offset/ridge baselines; a separate neural error model
learns squared held-geography residuals and returns their square root as RMSE.
Baselines remain eligible so exact offsets need not have a positive error floor.
The 40-hour accumulated neural fitting cap is not a required runtime.
Use the second GPU only after checking it is free; do not interrupt the joint
uncertainty run. Neither existing apps nor returned station models are replaced.

`senorge_corrected_transfer.py` is a separate bounded national experiment using
the existing raw `tg` (daily mean) and `tn` (daily minimum) temperature atlas.
It does not use MEPS,Frost temperatures,old precomputed RMSE labels or app
predictions. The old `regression_labels` path already measured an OLS-corrected
relationship in one experiment; this new path explicitly saves the signed
temperature predictor and learns its remaining squared prediction errors.

Default sampling uses120training,30validation and30test reference cells,
covering every original150km geographic block represented by those splits.
Eight distinct nearby target cells per reference are sampled across distance
bands up to the original64km square. Endpoints are disjoint between geographic
splits. It samples100000training and25000rows per evaluation split,including
both variables,from the existing3600-date1996-2025plan. These are sampled
dates,not all days of thirty years.

Training temperatures are1996-2015. Validation-only mean selection uses
2016-18in held validation geography.2019-20in that geography are diagnostic
error-model assessment,with no fitted rescaling.2021-25in final held geography
evaluate both temperature and predicted RMSE. Historical footprint geography
and years were already inspected; results remain exploratory.

Candidates are a learned constant signed offset,a signed terrain/source-
temperature/calendar ridge correction (fixed penalty10on training-standardized
inputs),and ExtraTrees corrections on that
linear predictor. This first pilot uses existing CPU sklearn tooling,not a new
GPU/CNN training search. The validation winner is frozen. Five training-block
cross-fits remove both source and target endpoints in held blocks; all training
rows get genuinely held-geography temperature predictions. A second ExtraTrees
model fits their **squared errors** using inference features plus predicted
temperature. It never sees target temperature as an input. There is no training-
residual error-model accuracy claim.

Predicted RMSE is `sqrt(predicted mean squared error)`,not absolute error,
standard deviation or an interval radius. When pooling cell/date RMSE estimates,
take the square root of their mean square,not their arithmetic mean. Reports
compare observed and predicted pooled RMSE by variable,season,block,distance
and prediction bins. No positive error floor is added: a learned exact
`T_B=T_A-10C` relationship must yield zero corrected RMSE. The focused test
verifies zero signed-prediction/error-model RMSE while raw difference RMSE is10C.
That does not guarantee any flexible model automatically learns every offset
or generalizes perfectly to unseen geography.

Run from the project environment:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -m unittest test_senorge_corrected_transfer -q
.\.venv-seasonal-app\Scripts\python.exe -u senorge_corrected_transfer.py --jobs 4
```

Results are in `outputs\senorge_corrected_transfer_pilot_v2`: selected signed
model,error model,OOF temperatures/squared errors,calibration/test CSVs and a
checksummed report. Existing output is never overwritten. Raw source hashes,
settings and missing-row counts are recorded. Missing weather samples are
counted; excessive missingness is an explicit failure. This experiment measures
**seNorge reproduction**,not independent real-station accuracy,future weather,
calibrated intervals or station-network contributions. It does not modify the
running joint GPU experiment or either app.
The initial unregularized output `outputs\senorge_corrected_transfer_pilot`
is superseded: rank-deficient/collinear terrain inputs produced unstable OOF
linear extrapolations. V2uses ridge regularization without capping weather
errors or silently discarding the failing fold. Both evaluations are exploratory,
not fresh untouched test certifications.

Verified V2pilot results (25000test examples,238targets,four held geographic
blocks,600sampled dates in2021-25): the validation-selected ridge mean predictor
has1.4073C corrected test RMSE,compared with1.8790C for copying the source
(raw difference). The squared-error model predicts pooled RMSE1.4150C.
Daily-mean observed/predicted RMSE is1.2824/1.3731C; daily-minimum is
1.5229/1.4559C. Winter observed/predicted RMSE is1.7580/1.6678C.
The agreement in pooled RMSE is not universal calibration: individual
geographic blocks and distance bands still over/underestimate error.
The lowest-to-highest predicted-error quintiles have observed RMSE
0.7610,1.1616,1.2343,1.3978,2.1205C,respectively.

All50000calibration/test source and target values were checked against the
raw sampled weather atlas; saved temperature/error-model predictions and
squared-error definitions were reproduced exactly within numerical tolerance.
Six focused tests include perfect-offset zero error,unstable collinear-feature
regression,signed-direction inputs and geographic endpoint disjointness.

Validation: `python -m unittest test_station_temperature_map
test_prepare_national_dataset test_station_valley_sandbox -q`.
This covers39feature and saved-prediction parity,graph barriers,source sensor
selection,missing-observation/slider behavior,NoData,GeoTIFF georeferencing
and explicit failed preparation. First-time map preparation accesses the
public MEPS archive; errors are displayed and logged rather than replaced
with guessed weather.

### Frost-Data og Eksport
Frost-databasen åpnes skrivebeskyttet. En fersk tilgjengelighetsrapport brukes
for å unngå full gjennomlesing av databasen, men hver aktuell stasjon kontrolleres
for ikke-null observasjoner i SQLite. Utvalget krever `inside_norway = 1`, minst
én brukbar observasjon av riktig element i femårsvinduet og komplette
terrengvariabler. Dette er ikke et krav om fem komplette år. Verifisert utvalg
for 14.10.2021–14.10.2026: **1062 temperatur-, 726 nedbør- og 469 snøstasjoner**.

**Eksporter til ArcGIS** laster ned GeoTIFF-er, stasjoner som GeoPackage/CSV og
`scenario.json`. `selected_reduction.tif` og `selected_affected.tif` gjelder valgt
stasjon, ikke nødvendigvis sist tillagte stasjon. Påvirkningsrasteret er en binær
0/1-maske; JSON-filen oppgir påvirket areal i km², valgt stasjon og rutestørrelse.
Eksport bruker **EPSG:25833** og valgt oppløsning. Nettleserkartet omprojiseres til
Web Mercator for å passe bakgrunnskartet.

Scenarier lagres per nettleserøkt i minnet og forsvinner når serveren startes på
nytt. Eksporter ønskede resultater før avslutning. Hvert scenario støtter opptil
40 egne stasjoner. Serveren lytter bare på `127.0.0.1`; verktøyet er lokalt og
ikke en produksjonstjeneste. Originale observasjoner, modeller og sjekkpunkter
endres ikke.

Standardfilen for det lagrede 1000 m-nettet er
`outputs/uncertainty_surfaces_1000m.npy`. En annen fullstendig fil fra samme
stasjonsrekke og rutenett kan velges slik:
```powershell
python station_app.py --surfaces outputs/uncertainty_surfaces_1000m_saved.npy
```
Kjør regresjonstestene med samme Python-miljø:
```powershell
python -m unittest test_station_app -v
```



& ".\.venv-portable-build\Scripts\python.exe" -u build_portable.py --name Vaernett_Update_01 --reuse-stage "outputs/portable_build/Vaernett_Windows_portable/data"

& ".\.venv-portable-build\Scripts\python.exe" -u build_portable.py


& ".\.venv-portable-build\Scripts\python.exe" -u build_portable.py `
  --cell-m 500 `
  --name Vaernett_500m_20261112 `
  --reuse-stage ".\outputs\portable_build\Remote_500m\data"

& ".\.venv-portable-build\Scripts\python.exe" -u build_portable.py --cell-m 500 --name Vaernett_500m_temperature_20261113 --reuse-stage ".\outputs\portable_build\Remote_500m\data"