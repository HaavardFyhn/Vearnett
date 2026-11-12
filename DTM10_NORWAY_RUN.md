# Nationwide DTM10 potential-lake run

`potential_lakes_dtm10_norway.py` computes one global Priority-Flood surface using tile-local queues and iterative spill-level exchange across chunk boundaries. It does not flood chunks independently and stitch their edges: a tile is revisited whenever a neighboring spill level improves, until no boundary can lower. Large raster arrays remain disk-backed, while the priority queue is bounded by `--flood-chunk-size` (default 512 cells per side). The national raster extent is about 19.2 billion cells; budget at least 400 GiB of free scratch space, plus space for outputs. The current tiles occupy about 7.1 GiB locally.

The runner writes depth, raw-basin-ID, and sill-group-ID GeoTIFFs, plus raw and grouped CSV tables and `summary.json`. It deliberately does not polygonize the nationwide 10 m labels; use the rasters in ArcGIS for national inspection. Sill-group volumes are sums of member-basin volumes, not a reflooded connected-reservoir simulation.

## Stage files from Windows (local PowerShell)

First leave the Ubuntu SSH session by typing `exit`. Wait until the prompt is back in Windows PowerShell (for example, `PS H:\>`), then change to the local repository root and run these commands there. Do not run them at the `havfyh@mathmagic-land` Ubuntu prompt: the `scp` source paths are Windows files, and `ssh` from the server back to `its.npra.io:1959` is not the file-transfer route.

```powershell
Set-Location -LiteralPath 'C:\Users\havfyh\OneDrive - Statens vegvesen\værstasjoner\pipeline_v3'
ssh -p 1959 havfyh@its.npra.io "mkdir -p ~/potential_lakes/data ~/potential_lakes/outputs"
scp -P 1959 .\potential_lakes.py .\potential_lakes_dtm10_norway.py havfyh@its.npra.io:~/potential_lakes/
scp -P 1959 .\outputs\norway_polygon.gpkg havfyh@its.npra.io:~/potential_lakes/outputs/
scp -P 1959 -r .\data\DTM10 havfyh@its.npra.io:~/potential_lakes/data/
```

If the tiles and boundary are already on the server and only the runner needs updating, from local Windows PowerShell use:

```powershell
Set-Location -LiteralPath 'C:\Users\havfyh\OneDrive - Statens vegvesen\værstasjoner\pipeline_v3'
scp -P 1959 .\potential_lakes_dtm10_norway.py havfyh@its.npra.io:~/potential_lakes/
```

## Prepare and launch on Ubuntu

```bash
ssh -p 1959 havfyh@its.npra.io
python3 -m venv ~/venvs/potential-lakes
source ~/venvs/potential-lakes/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy numba pandas rasterio geopandas pyogrio shapely
```

Check that all 251 tiles arrived and that the server volume still has ample free space:

```bash
find ~/potential_lakes/data/DTM10 -type f -name '*_10m_z33.tif' | wc -l
du -sh ~/potential_lakes/data/DTM10
df -h /tmp
```

Before launching, run the CRS preflight from the project directory so Python can import the runner and its helper module:

```bash
cd ~/potential_lakes
~/venvs/potential-lakes/bin/python - <<'PY'
from pathlib import Path
import rasterio
from potential_lakes_dtm10_norway import is_supported_dtm_crs

tile = sorted(Path.home().joinpath("potential_lakes/data/DTM10").glob("*.tif"))[0]
with rasterio.open(tile) as source:
  print("authority:", source.crs.to_epsg())
  print("normalized:", source.crs.to_dict())
  print("accepted:", is_supported_dtm_crs(source.crs))
  assert is_supported_dtm_crs(source.crs)
PY
```

Launch in the background. The work directory should be on the server's large local NVMe volume; `/tmp` is on that volume on the checked host.

```bash
mkdir -p ~/potential_lakes/outputs
nohup ~/venvs/potential-lakes/bin/python -u ~/potential_lakes/potential_lakes_dtm10_norway.py \
  --dtm-dir ~/potential_lakes/data/DTM10 \
  --norway ~/potential_lakes/outputs/norway_polygon.gpkg \
  --out-dir ~/potential_lakes/outputs/potential_lakes_norway_DTM10_sill20 \
  --work-dir /tmp/$USER/potential_lakes_norway_DTM10_work \
  --flood-chunk-size 512 \
  --merge-sill-m 20 \
  > ~/potential_lakes/outputs/dtm10_norway.log 2>&1 < /dev/null &
echo $!
```

Follow progress with:

```bash
tail -f ~/potential_lakes/outputs/dtm10_norway.log
```

On success, the output directory contains `potential_lake_depth.tif`, `potential_lake_id.tif`, `potential_lake_group_id.tif`, `potential_lakes.csv`, `potential_lake_groups.csv`, and `summary.json`. The flood is CPU-intensive and single-process; expected runtime should be measured on the server rather than inferred from the 28-core count.

The runner does not resume a partial flood: boundary spill state and raster arrays are mutated in place. A completed `dtm10_mosaic.tif` can be reused, however, so a retry need not mosaic all 251 source tiles again. Locate a completed mosaic and use a fresh work directory:

```bash
find /tmp/$USER -name dtm10_mosaic.tif -printf '%p %s bytes\n'
```

After selecting the complete mosaic path, launch a clean retry (replace the `MOSAIC` value with the path printed above):

```bash
MOSAIC=/tmp/$USER/potential_lakes_norway_DTM10_work/dtm10_mosaic.tif
WORK=/tmp/$USER/potential_lakes_norway_DTM10_work_retry_20261028
LOG=~/potential_lakes/outputs/dtm10_norway_retry_20261028.log
nohup ~/venvs/potential-lakes/bin/python -u ~/potential_lakes/potential_lakes_dtm10_norway.py \
  --mosaic "$MOSAIC" \
  --norway ~/potential_lakes/outputs/norway_polygon.gpkg \
  --out-dir ~/potential_lakes/outputs/potential_lakes_norway_DTM10_sill20 \
  --work-dir "$WORK" \
  --flood-chunk-size 512 \
  --merge-sill-m 20 \
  > "$LOG" 2>&1 < /dev/null &
echo $!
```

This reruns the outlet scan and flood from clean elevation/state arrays but skips mosaicking. Preserve all old scratch directories and logs until the new run succeeds; do not remove scratch files while a process is active.