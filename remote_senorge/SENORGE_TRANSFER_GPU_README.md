# Expanded seNorge signed-temperature and RMSE GPU experiment

This package trains actual PyTorch models,not CPU ExtraTrees with a device flag.
It preserves the separate signed prediction / residual-error stages. The
tagged pilot test CSV is not used as training input. Raw samples come directly
from the existing seNorge `tg` and `tn` archive at native1km cells.

Defaults expand from180to2400reference sources (2000train,200validation,
200test),with16targets each. Target sampling covers distance bands within the
64km square and retains the original18training/four validation/four test
150km geographic blocks. Sample sizes are1000000training,100000selection,
100000assessment and200000test rows. Exact prepared counts/geography are in
`manifest.json`; missing samples are counted. This is not every national cell
or every day: the source has3600sampled dates over1996-2025.

Mean fitting uses1996-2015,validation-only selection2016-18,diagnostic error
assessment2019-20,and final test2021-25. Geographic endpoints remain disjoint
between train,validation and test. Selection and assessment share validation
geography but use disjoint years. These regions/years were already inspected
in past experiments,so the final comparison is exploratory,not fresh validation.

## Models and selection

- Learned constant offset and ridge baselines are eligible to win.
- Four PyTorch signed-temperature trials: widths64/128,seeds42/2026.
  Each adds a learned residual to a training-only standardized ridge anchor.
  Two hidden layers,SiLU,LayerNorm,dropout0.1,zero-initialized output,
  AdamW0.001/weight decay0.01,Huber temperature loss,gradient clipping5.
- The mean winner is chosen solely by geographically balanced selection RMSE.
- Five geographic training-block cross-fits refit the frozen winner for its
  best validation epoch count. Source AND target endpoints in held blocks
  cannot enter fitting. Ridge/normalization are independently refitted inside
  each fold. Every training row receives an OOF temperature prediction.
- The error labels are **squared OOF temperature errors**,not raw differences,
  in-sample errors or absolute errors. Four PyTorch positive-MSE trials use the
  same width/seed grid. Inputs are49inference features plus predicted temperature.
  A pooled weighted MSE baseline remains eligible to win.
- Whole OOFtraining blocks (every fifth sorted block) are reserved for error
  trial selection using squared-error-target MSE. Other OOFblocks fit the error
  network. This residual reserve is a model-selection diagnostic,not fully
  independent error certification: OOFtemperature fits share other blocks.
  No calibration/test row chooses the error model or rescales its output.
- Predicted RMSE is the square root of predicted MSE. No error floor is imposed.
  An exactly learned10Coffset yields zero corrected error,within floating-point
  tolerance. Softplus neural outputs alone do not guarantee exact zero; the
  zero-error pooled baseline can win. More epochs do not guarantee learning.

Results compare actual/predicted pooled RMSE by variable,season,block,distance
and predicted-error bins,alongside raw difference,offset and ridge RMSE.
No calibrated intervals or real-station accuracy claims follow from this.

## GPU launch

Project PowerShell:

```powershell
scp -P 1959 .\outputs\senorge_transfer_gpu_v1.zip havfyh@its.npra.io:~/
ssh -p 1959 havfyh@its.npra.io
```

The other joint experiment uses `cuda:0`. Check that the second GPU is free;
do not stop the other job. On the GPU server:

```bash
nvidia-smi
mkdir ~/senorge-transfer-gpu-v1
cd ~/senorge-transfer-gpu-v1
unzip -q ~/senorge_transfer_gpu_v1.zip
source ~/national-uncertainty-v2/.venv/bin/activate
python -m unittest test_senorge_transfer_gpu -q
python -m pip freeze > installed_versions.txt
nohup python -u train_senorge_transfer_gpu.py \
  --device cuda:1 --hours 40 --epochs 200 --patience 25 \
  --batch-size 2048 --jobs 4 --folds 5 \
  --output senorge_transfer_gpu_results \
  > senorge_transfer_gpu_training.log 2>&1 < /dev/null &
echo $! > training.pid
sleep 15
ps -p "$(cat training.pid)" -o pid,etime,cmd
tail -n 20 senorge_transfer_gpu_training.log
```

If that environment is unavailable,create a local venv,install
`requirements.txt`,and torch2.5.1from the official cu124wheel index,as for prior
packages. Do not upgrade an environment used by an active job.

40hours is a maximum accumulated **neural** fitting budget,not a mandatory
runtime or wall-clock deadline. CPU baseline fits,normalization,prediction/
I/O add wall time; one epoch can overrun. An incomplete mean search,cross-fit
or error search never emits completed deployable results. Exact-command
resume preserves model/optimizer/RNG and complete epochs,not an interrupted
partial epoch. Changed code,data,settings,device or budget require a fresh
output. Completed results verify checksums without re-evaluating test data.
Load only your own trusted checkpoint/joblib files.

Portable smoke checks use `--device cpu --epochs 1 --widths 8 --seeds 42
--validation-only --output smoke_results`,with bounded data. Validation-only
mode never predicts or scores calibration/test and cannot be changed into a
production run by resuming the same output.

## Return results

After `status.json` says `complete`:

```bash
cat senorge_transfer_gpu_results/status.json
python -m pip freeze > installed_versions.txt
tar -czf senorge_transfer_gpu_results.tar.gz senorge_transfer_gpu_results \
  manifest.json package_checksums.json installed_versions.txt \
  senorge_transfer_gpu_training.log
exit
```

Project PowerShell:

```powershell
scp -P 1959 havfyh@its.npra.io:senorge-transfer-gpu-v1/senorge_transfer_gpu_results.tar.gz .\outputs\
```

Return results for verification before changing any app or promoting models.

If `tar` reports a missing `installed_versions.txt`, it can still have created
an archive containing the other files. Do not assume it is complete: verify
the archive and model/result checksums. The returned first run omitted this
file, but all 300,000 assessment/test predictions have been reproduced locally
from the checksummed models and original data. The experimental app now exposes
held-out evaluation maps at `/model-evaluation`, not live seNorge forecasts.
