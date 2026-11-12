# Residual uncertainty v2 before experimental1km mapping

The temperature mean model is frozen at the validation-selected national
single-source residual MLP. This experiment does not replace it or the apps.
There are **ten refits in two distinct five-round schemes**,not five locations:

- **Unseen-target rounds (primary):** station targets are distributed across
  geography into five held-out sets. Held station observations are excluded
  from fitting both as targets AND as sources. Nearby nonheld source stations
  remain available. This matches missing-thermometer transfer better than
  holding out entire regions. Sources in the same held set are unavailable;
  those pairs are explicitly counted rather than silently retained.
- **Whole-region rounds (diagnostic):** groups of100km blocks with10km buffers
  are excluded as training targets AND observed sources. Only outside-region
  sources generate residuals. These are harder boundary/novel-region tests,
  not mixed into the primary residual-error fitting sample.

Both retrain the selected architecture for its original best epoch count (one
epoch in this run),with fixed settings and no new model/epoch selection.
The error-scale model fits **unseen-target out-of-fold residuals only**.
Reports show evaluated/original pair coverage,distance bands,missing target
stations,errors by training-winter and every evaluated geographic block,
plus cold/calm-clear errors,separately for both schemes.

Verified preparation counts with five rounds per scheme:
unseen-target residuals cover1,332,844pairs (85.4%of training pairs),all392
training targets and51blocks. Buffered regional residuals cover216,790pairs,
138targets and44blocks. The primary sample includes660,102transfers in the
0-25km band versus21,291in the regional sample. These are correlated pairs,
not that many independent observations. Reports recompute counts at runtime.

An ExtraTrees error-scale model fits absolute unseen-target residuals with
training-only geographic/target-night weighting. Features are the same39
inference features plus predicted temperature,not target observations.
Up to300000uniform cross-fit pairs fit128trees; scale is floored at0.1C.

The45calibration nights in the existing two validation blocks fit a multiplier to the maximum normalized absolute
error of all evaluated station transfers in each night. This targets90%
simultaneous **station-pair** coverage,as requested. The already-inspected
test in the existing two test blocks is evaluated once and marked exploratory.
No test-based selection/tuning. Coverage and interval width are reported per
block,target,distance band and cold/calm-clear subset as well as across all
evaluated predictions per night. Descriptive training-winter OOF errors are
not a fresh temporal holdout or validation of the error-scale model itself.
This revision improves residual representativeness and geographic diagnostics;
it does not manufacture new calibration/test regions or winters. New independent
regions/years are still needed for national certification.

This is NOT a proof of simultaneous coverage over4096unobserved grid cells.
It is not a fusion-error model,domino uncertainty,or valid for arbitrary slider
temperatures. Distribution shift can invalidate the intervals. The report
always leaves uncertainty_map_validated=false; return results before any
map integration. No raster will be labelled calibrated automatically.

## Remote launch

In project PowerShell:

```powershell
scp -P 1959 .\outputs\national_uncertainty_training_v2.zip havfyh@its.npra.io:~/
ssh -p 1959 havfyh@its.npra.io
```

On GPU server:

```bash
mkdir ~/national-uncertainty-v2
cd ~/national-uncertainty-v2
unzip -q ~/national_uncertainty_training_v2.zip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m unittest test_national_uncertainty -q
python -m pip freeze > installed_versions.txt
nohup python -u train_national_uncertainty.py --device cuda:0 --hours 40 \
  --folds 5 --batch-size 2048 --jobs 4 --tree-rows 300000 \
  --output national_uncertainty_results > uncertainty_training.log 2>&1 < /dev/null &
echo $! > training.pid
sleep 15
ps -p "$(cat training.pid)" -o pid,etime,cmd
tail -n 20 uncertainty_training.log
```

Do not use the earlier `national_uncertainty_training.zip`; v2 uses different
fold semantics/checkpoints and requires a new directory/output.

40h is a maximum accumulated fitting budget; loading,prediction and
tree fit/calibration overhead add wall time. An epoch/tree fit can overrun.
Incomplete cross-fit never emits a ready uncertainty model. Rerun the exact
command after interruption; checkpoints preserve optimizer,RNG and epochs.
Completed outputs verify hashes without rescoring. Changed settings/code/data
are rejected. Load only your own trusted torch/joblib checkpoints.

```bash
cat national_uncertainty_results/status.json
tail -n 30 uncertainty_training.log
tar -czf national_uncertainty_results.tar.gz national_uncertainty_results \
  uncertainty_spec.json manifest.json package_checksums.json \
  installed_versions.txt uncertainty_training.log
exit
```

In project PowerShell:

```powershell
scp -P 1959 havfyh@its.npra.io:national-uncertainty-v2/national_uncertainty_results.tar.gz .\outputs\
```

Inspect nightly coverage,interval widths and geographic failures before maps.
Next integration needs separate bounded MEPS weather at1km land-cell centres:
the national training archive contains station weather only. Intervals over
new cells remain experimental even if held-out station coverage improves.
