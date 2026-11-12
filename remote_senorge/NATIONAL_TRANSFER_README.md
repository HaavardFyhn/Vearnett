# National corrected transfer and multi-station fusion

This package fits signed temperature transfer, not the old raw-difference RMSE
footprints. Source observations and weather/terrain features predict the target
temperature; held-out residual errors measure what remains after correction.

## Fixed first experiment

- Single-source controls: copy source, pooled training-only constant offset,
  pooled linear signed-height correction, forecast transfer, weather ExtraTrees
  residual correction (128trees,<=300000uniform training pairs with geographic/
  target-night weights).
- Neural single-source residual MLP, weather-gated3experts, graph-feature gated
  experts; widths64/128,seeds42/2026.
- Joint fusion network sees up to6simultaneous sources, learns masked softmax
  weights and per-source forecast-residual corrections. Sources are processed
  jointly; padding is masked and order is permutation-invariant. Baselines are
  equal-weight means of each classical single-source method.
-16neural trials in total, including4fusion trials. Interleaved family order,
  AdamW.001/decay.01,Huber2C,dropout.1,clip5;200epochs/patience25.
- Training and selection weight each100km geographic block equally, then each
  target-night equally inside the block; multiple source/time rows share that
  mass. Normalization uses those training weights only.
- Separate validation-selected winners for pair and fusion tasks. Fusion is
  evaluated once per target/time, not per pair. Do not directly compare pooled
  pair RMSE with fusion RMSE as if populations/weights were identical.
- Calibration/test do not fit corrections, normalization or choose trials.
  No per-heldout-station offset or station ID feature is fitted. Test predictions
  are produced only after both winners are fixed.

Sources are contemporaneous observations. This is neither future forecasting,
daily-minimum prediction,domino propagation,national1km mapping,nor a learned
covariance/GNN. Fusion weights are not station-removal importance. Calibrated
90%radii use max error per calibration night and may be very broad. Regional and
temporal distribution shifts invalidate coverage guarantees.

This run uses two frozen validation/two test blocks,not leave-region-out CV.
The earlier western pilot used2024-25weather too. MEPS may assimilate earlier
heldout station readings. All missing-run exclusions remain recorded.

## Transfer and setup

The verified ZIP is already prepared locally. In project PowerShell:

```powershell
scp -P 1959 .\outputs\national_transfer_training.zip havfyh@its.npra.io:~/
ssh -p 1959 havfyh@its.npra.io
```

On the server, use a new directory; do not overwrite valley runs:

```bash
mkdir ~/national-transfer-v1
cd ~/national-transfer-v1
unzip -q ~/national_transfer_training.zip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m unittest test_national_transfer -q
python -m pip freeze > installed_versions.txt
nvidia-smi
nohup python -u train_national_transfer.py --device cuda:0 --hours 40 \
  --epochs 200 --patience 25 --batch-size 2048 --jobs 4 \
  --output national_transfer_results > national_transfer_training.log 2>&1 < /dev/null &
echo $! > training.pid
sleep 15
ps -p "$(cat training.pid)" -o pid,etime,cmd
tail -n 20 national_transfer_training.log
```

Initial checksum/loading/tree-fitting may take time before neural epoch messages.
40h is accumulated measured fitting budget,not a strict wall-clock deadline:
the classical fit and an in-flight neural epoch finish before budget checks.
Normalization,selection scoring,checkpoint writes and final inference add time.
Budget may end before every trial; summary lists attempted trials. Do not extend
the budget after inspecting test scores to select a preferred result.

Every complete neural epoch saves current/best weights,optimizer,RNG/history/
elapsed time. Rerun exactly the same command to resume after interruption. A
partial epoch restarts from its last checkpoint. Finished results verify hashes
and do not retrain. Changed code/data/settings/device/budget are rejected. The
output has a process lock; never run two writers. Only load your own trusted
PyTorch/joblib checkpoints; they are not safe arbitrary download formats.

```bash
cat national_transfer_results/status.json
tail -n 30 national_transfer_training.log
```

Summary contains separate winners,validation tables,test errors by block and
cold/calm-clear cases,calibrated intervals and night coverage. Artifacts include
training normalization,offset coefficients,tree model,trial checkpoints,
selected neural weights where applicable,and pair/fusion test CSVs with
baseline predictions and fusion source identities/counts.

## Return results

```bash
tar -czf national_transfer_results.tar.gz national_transfer_results \
  manifest.json package_checksums.json installed_versions.txt national_transfer_training.log
exit
```

In project PowerShell:

```powershell
scp -P 1959 havfyh@its.npra.io:national-transfer-v1/national_transfer_results.tar.gz .\outputs\
```

Do not automatically replace existing apps/models. Verify calibration/test
predictions and compare geography,weather and source count before deployment.
