# Overnight weather-regime station-transfer experiment

This package reuses the verified western-Norway pilot. It does not replace
either Station Lab app and does not yet create a covariance/uncertainty surface.
It contains no Frost credentials, local configuration, GIS libraries or raw
weather archives. No additional download is needed on the GPU machine.

## What is different

63 neural trials compare four families:

- **Terrain MLP control** predicts target-minus-source temperature.
- **Forecast-residual MLP** starts at source observation plus the MEPS
  target-minus-source difference, then learns a signed correction.
- **Regime-gated experts** learn three softmax-weighted residual corrections.
  The gate sees source/target forecast temperature, cloud and grid-relative
  wind, plus source forecast residual and speed. Experts are learned statistical
  regimes, not proven physical inversion categories.
- **Graph-gated experts** add minimax terrain-pass/ridge descriptors and their
  interactions with wind, cloud and source forecast residual to the experts.
  This is a connectivity-feature ablation, not a graph neural network.

Each expert/MLP has two hidden Linear+SiLU layers, LayerNorm after the first
activation,10%dropout and a scalar signed output. Residual output layers start
at zero, so they begin at the forecast-transfer baseline. Widths32/64/128 and
seeds42/123/2026 are compared. Non-terrain families also compare risk weight1
versus3 for training calm-clear or below-5C examples. AdamW uses lr0.001 and
weight_decay0.01, Huber loss(delta2C) and gradient clipping.

Each training night gets equal mass before risk weighting. The same
selection-validation RMSE chooses every epoch/trial, regardless of risk weight.
The unfitted forecast-transfer baseline is eligible to win. Calibration and
test do not influence normalization, weights, early stopping or selection.
The test is scored only after selection, not after every trial.

The sample remains small:3714training rows over90nights,523selection-validation
rows over45nights,549calibration rows over46nights and877test rows over78nights.
Pairs share stations and timestamps; these are not independent geographic
examples. Haukedal's test was already inspected in earlier work: this is an
**exploratory comparison, not a fresh untouched test**. The model needs later
evaluation on new regions/years. Longer training cannot create new information.

## Transfer from Windows

Prepare locally once:

```powershell
& .\.venv-seasonal-app\Scripts\python.exe .\prepare_valley_overnight.py
scp -P 1959 .\outputs\valley_neural_training.zip havfyh@its.npra.io:~/
ssh -p 1959 havfyh@its.npra.io
```

## GPU setup and launch

Use a new directory, not the previous footprint/results directory. The existing
GPU virtual environment may be reused without altering it, or make a fresh one:

```bash
mkdir -p ~/valley-neural-v1
cd ~/valley-neural-v1
unzip -q ~/valley_neural_training.zip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m unittest test_valley_neural -q
python -m pip freeze > installed_versions.txt
nvidia-smi

nohup python -u train_valley_neural.py --device cuda:0 --hours 40 \
  --epochs 1000 --patience 100 --batch-size 256 --jobs 4 \
  --output valley_neural_results > valley_neural_training.log 2>&1 < /dev/null &
echo $! > valley_neural_training.pid
sleep 10
ps -p "$(cat valley_neural_training.pid)" -o pid,etime,cmd
tail -n 20 valley_neural_training.log
```

There is no CUDA-to-CPU fallback. If the process is already complete when
checking its PID, inspect the log and status rather than assuming a launch
failure. The dataset is small enough that all trials **may finish much earlier
than40hours**.40hours is a maximum accumulated measured training budget, not a
minimum runtime. Checksums, initialization, final evaluation and checkpoint
overhead add time outside that measured budget; an in-progress epoch can finish
after the deadline.

## Resume and inspect

Every completed epoch atomically saves current/best model states, optimizer,
RNG states, validation history and elapsed training time. An interrupted epoch
is rerun from its last durable checkpoint. Use exactly the same command/output
to resume. Changed code, data, trial settings, device or budget are rejected.
An output directory is process-locked to prevent simultaneous writers.

```bash
cat valley_neural_results/status.json
tail -n 30 valley_neural_training.log
```

Completed runs verify final artifact checksums rather than retraining/rescoring.
Budget-completed runs are final; extending the budget requires a separate new
output and must not be chosen using the inspected test. Checkpoints use PyTorch
serialization: load only files you generated/trust, never arbitrary downloads.

`summary.json` reports validation-selected configuration, all validation trials,
actual epochs/time, selected model and forecast baseline test errors, cold and
calm-clear errors, and calibrated intervals. A90%radius is calibrated from the
maximum absolute error per calibration night. Broad intervals or low empirical
coverage remain failures, not success simply because overall RMSE improves.
There is no guarantee of conformal coverage under spatial/weather shifts.

Artifacts include training-only `normalization.npz`, `run_config.json`,
`selected_model.pt`(only when a neural trial beats the baseline),
`selected_test_predictions.csv`, trial checkpoints and `summary.json`.
No automatic app installation occurs.

Return results:

```bash
tar -czf valley_neural_results.tar.gz valley_neural_results \
  manifest.json package_checksums.json installed_versions.txt valley_neural_training.log
exit
```

```powershell
scp -P 1959 havfyh@its.npra.io:valley-neural-v1/valley_neural_results.tar.gz .\outputs\
```

Inspect cold/calm-clear errors and interval usefulness before considering a
larger neural graph/covariance experiment. MEPS can assimilate earlier held-out
station observations;12–18hour leads do not prove total background independence.
These are contemporaneous nighttime station transfers, not daily minima or
future temperature forecasts.
