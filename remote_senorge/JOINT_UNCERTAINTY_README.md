# Joint station uncertainty experiment

This is the prerequisite for the requested national **joint** uncertainty map.
It does not install a national map or replace existing apps. Existing single-
source intervals are not joint intervals and must not be merged by taking the
minimum,averaging them or dividing by the square root of station count.

The original validation-selected fusion network combines at most six nearby
connected station readings per target/time. Its original best epoch is frozen
(one epoch for the returned winner). Five unseen-target and five buffered
whole-region refits generate genuinely held-out joint predictions. Held
observations are excluded from every fitting source slot as well as targets.
Nearby nonheld sources remain in the primary scheme. Region rounds diagnose
novel geography and are not pooled into error-scale fitting.

For each target/time the experiment evaluates the full group and every valid
single-reading deletion,without replacing the deleted source. Sole-source
deletion has no prediction and must later be NoData. A128-tree error-scale
model learns absolute joint errors from primary OOF configurations only.
Its120permutation-invariant features are39input-feature means,minima,maxima,
joint prediction,anchor spread and source count. No target observation enters
the inference features. Sources are not assumed independent.

Real-data preflight: primary rounds cover289175joint target/time groups across
all392training targets,with1619032full/deletion configurations. Regional
rounds cover100687groups. These are correlated configurations,not independent
samples. Per-fold full errors are broken down by block,winter,distance and
cold/calm conditions; test coverage is additionally broken down by source count.

Up to300000OOF configurations are sampled equally across target folds,
then weighted by geographic block and target-night. All groups/configurations
remain in fold error diagnostics. This is not an independently validated
coverage result for the scale model's fitting residuals.

The45original calibration nights fit the finite-sample90%multiplier using
the maximum normalized error across **all evaluated full groups and valid
deletions in each night**. The already-inspected test reports both full-group
coverage and the stricter full-and-deletions coverage. No test-based selection
or tuning occurs. Existing two calibration/two test areas are not sufficient
national certification; no automatic grid-coverage guarantee is made.

`test_contributions.csv` records signed changes in predicted temperature and
interval radius when a source is removed. Positive radius change means this
model predicts a wider interval without that reading; negative change is
possible and is not hidden. This is a model sensitivity diagnostic,not a
causal,additive station value or a fusion-weight interpretation.

Fitting/evaluation reuse the frozen national source network,not new stations.
The later map may offer all1342terrain-supported catalogue stations,using only
available selected-sensor QC observations. At2024-12-15at03UTC426had such
readings,361from original training sources. Other map sources must be flagged
as extrapolation. Each cell will still use at most six sources,not all426at
once. National cell weather and map assembly follow result assessment.

## Run on GPU

Project PowerShell:

```powershell
scp -P 1959 .\outputs\joint_uncertainty_training_v1.zip havfyh@its.npra.io:~/
ssh -p 1959 havfyh@its.npra.io
```

GPU server:

```bash
mkdir ~/joint-uncertainty-v1
cd ~/joint-uncertainty-v1
unzip -q ~/joint_uncertainty_training_v1.zip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m unittest test_joint_uncertainty -q
python -m pip freeze > installed_versions.txt
nohup python -u train_joint_uncertainty.py --device cuda:0 --hours 40 \
  --folds 5 --batch-size 2048 --jobs 4 --tree-rows 300000 \
  --output joint_uncertainty_results > joint_uncertainty_training.log 2>&1 < /dev/null &
echo $! > training.pid
sleep 15
ps -p "$(cat training.pid)" -o pid,etime,cmd
tail -n 20 joint_uncertainty_training.log
```

40hours is an accumulated fitting cap,not a required duration; epoch/tree fits
can overrun it. Prediction/I/O adds wall time. Budget exhaustion emits no ready
model. Exact-command resume preserves epochs,optimizer and RNG. Changed
settings/code/data require a fresh output. Completed outputs verify checksums
without rescoring. Only load your own trusted torch/joblib files.

After completion:

```bash
cat joint_uncertainty_results/status.json
tar -czf joint_uncertainty_results.tar.gz joint_uncertainty_results \
  joint_spec.json manifest.json package_checksums.json installed_versions.txt \
  joint_uncertainty_training.log
exit
```

Project PowerShell:

```powershell
scp -P 1959 havfyh@its.npra.io:joint-uncertainty-v1/joint_uncertainty_results.tar.gz .\outputs\
```

Return the results for verification before national map integration.
