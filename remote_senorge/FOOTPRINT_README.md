# Reference-conditioned footprint maps: annual, winter and summer

This is a new experiment, not a replacement of the working Station Lab model.
One forward pass produces six 64 x 64 maps at 1 km resolution. Each map
describes the historical representation error relative to its marked reference
location, not temperature itself or a calibrated uncertainty distribution.

## What is different

- Six outputs: annual mean/minimum, DJF mean/minimum and JJA mean/minimum.
  Annual means all twelve months, not the average of winter and summer.
- Labels are symmetric two-direction linear-regression evaluation RMSE in
  seNorge 1996-2025. All 3,600 original random dates are retained: 8 fitting
  and 2 evaluation dates per month per year. No station-location restriction.
- Each reference is supervised against every available seNorge-mapped terrain
  cell in its 64 km square, not only the old sparse million pairs.
- A residual U-Net sees the entire terrain scene, preserving spatial layout,
  ridges and the relationship between the reference and neighbouring valleys.
  A reference marker and signed relative coordinates condition the map.
- An 80 km scene at 100 m includes an 8 km halo around the 64 km output.
  The reference is output pixel row32/column32 (north-up). Terrain heights
  are float16 kilometre offsets relative to the reference terrain elevation.
- The fine branch sees native 10 m terrain in a 1.28 km square around the
  reference. This first experiment does NOT encode the entire 80 km scene
  at 10 m, nor give every target pixel its own 10 m encoder. Wider terrain
  context remains 100 m. Validity masks distinguish missing DEM from sea level.
- Existing 25 pair attributes, including whole-basin attributes, are spatial
  decoder inputs. Their normalization uses training cells only.
- Geographic groups reuse the earlier seed42 100 km block assignment.
  Every reference is at least 41 km inside its block, so the complete scene
  stays within its partition. Scenes can overlap within a partition, never
  between partitions. Prior experiments already inspected these geographic
  groups: this is not a newly untouched final test.

## Local preparation: Windows, project directory

Use the isolated GIS/PyTorch environment already prepared:

```powershell
.\.venv-seasonal-app\Scripts\python.exe -u prepare_footprint_training.py
```

Default reference counts: 256 training, 64 validation and 64 test. Smaller
counts can be specified with `--counts TRAIN VALIDATION TEST` in a NEW output
directory, not mixed into an existing checkpoint.

Preparation reads existing NetCDFs, DTM50-derived 100 m terrain and native DTM10.
It can take hours. Keep Windows awake and plugged in. Completed terrain scenes
and annual sufficient-statistics checkpoints are reused on restart. A partially
processed year is recomputed. Source fingerprints protect against mixed inputs.
No weather originals or old models are removed.

If running in the background, do not launch a second copy:

```powershell
Get-Content .\outputs\footprint_preparation.log -Tail 15 -Wait
Get-Content .\outputs\senorge_footprint_training\status.json
Get-Content .\outputs\senorge_footprint_training\cache\label_progress.json
```

Wait for `cache/complete.json` and status complete. The final ZIP is published
only after allowlist, CRC and SHA256 checks. Raw local paths, credentials,
GIS configuration, VRTs and weather originals are NOT packaged.

## Transfer: Windows PowerShell, not the SSH terminal

```powershell
ssh -p 1959 havfyh@its.npra.io "mkdir -p ~/senorge-footprint"
scp -P 1959 .\outputs\senorge_footprint_training.zip havfyh@its.npra.io:senorge-footprint/
ssh -p 1959 havfyh@its.npra.io
```

## Ubuntu: prepare a separate environment

```bash
cd ~/senorge-footprint
python3 -m zipfile -e senorge_footprint_training.zip .
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip freeze > installed_versions.txt
python -m unittest test_footprint -q
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name(0)); print(torch.ones(8,device="cuda:0").sum().item())'
```

Stop if tests or GPU checks fail. Follow server queue/allocation rules.

## Weekend experiment

```bash
nohup python -u train_footprint.py --device cuda:0 --hours 40 \
  --epochs 300 --patience 25 --batch-size 2 --jobs 8 \
  > footprint_training.log 2>&1 < /dev/null &
echo $! > footprint_training.pid
sleep 10
ps -p "$(cat footprint_training.pid)" -o pid,etime,cmd
tail -n 25 footprint_training.log
```

This is a 40-hour CNN experiment budget, NOT a promise that training will take
40 hours or improve with time. Early stopping can finish earlier. The budget
is checked between epochs; a final epoch, selection/evaluation and the six
XGBoost baselines can extend past it. There are 12 candidate runs:
width16/24, with/without native10m reference branch, seeds42/123/2026.
Order pairs context-only and fine10m runs for each width/seed, starting with
the smaller width. If budget runs out, not all
comparisons will be completed; completed candidates and stopped-by-budget flags
are recorded. Choose a smaller suite via `--widths 16 --seeds 42` to benchmark.

AdamW, weight decay, validation-driven learning-rate reduction, gradient
clipping and early stopping are enabled. Loss is equally weighted MSE across
the six targets over valid supervised pixels. Each completed epoch saves model,
optimizer, scheduler and epoch state. Restart with the EXACT same arguments
to continue an interrupted run. Completed trials are checksum-verified.

The winner is chosen by validation MSE, not the test map. Only that CNN is
scored on test maps after model selection. XGBoost uses the same reference
splits and target labels. The centre identity cell is set to zero for CNN
inference and excluded from comparative metrics for both models.

To monitor:

```bash
tail -f footprint_training.log
cat footprint_results/status.json
nvidia-smi
```

Ctrl+C stops `tail`, not the background trainer. Use one A5000 first; the
second GPU stays available and does not automatically add VRAM. CUDA failure
does not silently switch to CPU. GPU memory/throughput must be measured on
the server. On OOM, reduce batch size using a NEW output directory.

## Return results

```bash
tar -czf footprint_results.tar.gz footprint_results footprint_manifest.json \
  installed_versions.txt footprint_training.log
```

From Windows:

```powershell
scp -P 1959 havfyh@its.npra.io:senorge-footprint/footprint_results.tar.gz .\outputs\
```

Keep local `outputs/senorge_footprint_training` for future map inference.
Returned weights alone are not sufficient for app integration.

## Interpretation and limitations

The output square extends 32 km along each axis (maximum corner distance
approximately 45.3 km); it is NOT a trained 100 km-radius footprint.
No high-resolution temperature claims are justified by 1 km seNorge labels.
A model can only learn cross-valley differences present in those labels.
Symmetric labels do not enforce symmetric maps after swapping references.
The best-single-station minimum across footprints does not model joint
assimilation of several stations.

Test maps overlap within a geographic partition. Cell counts are not counts
of independent samples. Report geographic-block variability and same-valley/
cross-ridge examples, including SN56960/SN56895, before app replacement.
Local observations for SN56895 contain suspicious extreme summer readings;
quality-control them before using them as independent station truth.

Credit MET Norway/seNorge_2018 and Kartverket DTM10/DTM50 and retain their terms.
