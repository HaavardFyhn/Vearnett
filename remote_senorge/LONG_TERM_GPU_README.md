# Long-term corrected RMSE: GPU inference only

No training or model selection. Uses the original app's temperature-sensor
eligibility, all 3,600 sampled dates (1996--2025), and frozen returned models.
Each station-target pair pools predicted MSE over dates before ranking sources.
Source temperatures are historical seNorge, not actual Frost observations.

The package contains numeric terrain, temperatures only at eligible source
cells, date/footprint indices, frozen portable weights and checksums. No
credentials, Frost observation database or full national weather archive.

Use the existing working PyTorch CUDA environment. On the current server this is
`~/national-transfer-v1/.venv`, not `~/senorge-transfer-gpu-v1/.venv`.
Required Python packages: torch, numpy, pandas, scikit-learn, joblib.
Checksumming uses streamed reads and does not require Python 3.11's
`hashlib.file_digest`.
No GIS libraries are required. Do not install/upgrade CUDA packages unnecessarily.

```bash
mkdir -p ~/senorge-long-term-gpu-v1
cd ~/senorge-long-term-gpu-v1
source ~/national-transfer-v1/.venv/bin/activate
python -m zipfile -e ~/senorge_long_term_gpu_v1.zip .
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Optional bounded benchmark: one unique source for each variable.
python -u compute_long_term_gpu.py --device cuda:0 --benchmark --output benchmark_results

# Full resumable calculation (not training).
nohup python -u compute_long_term_gpu.py --device cuda:0 \
  --output long_term_results > long_term_gpu.log 2>&1 < /dev/null &
tail -n 10 long_term_gpu.log
```

Default batch size 65,536. If CUDA runs out of memory, resume with
`--batch-size 16384`; completed source outputs are checksummed and reused.
Use a free GPU (for example `--device cuda:1`) if GPU 0 is occupied.
Do not run two processes into the same output directory.
The benchmark output is marked `benchmark`, never a completed national result.
Completed CPU source footprints shipped in the package are reused automatically.

```bash
cat long_term_results/status.json
# Only after state is complete:
tar -czf senorge_long_term_gpu_results.tar.gz long_term_results long_term_gpu.log
```

Copy that archive back to the local workspace's `outputs` directory. The local
importer verifies package identity, every file and all-date sample predictions
before placing results into the exact cache used by the original app:

```powershell
.\.venv-seasonal-app\Scripts\python.exe .\import_long_term_gpu.py `
  .\outputs\senorge_long_term_gpu_results.tar.gz
```

It rebuilds best/second-best station rankings locally using the app's original
station order. Exact ties remain ties; removal never forces a positive change.
Unsupported cells retain NoData. Both original model entries load completed
results through the existing app progress refresh, without retraining.
