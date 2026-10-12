# Observed h20-9-57 environment

Observed on 2026-10-12 after the run-0 interruption:

| Component | Version |
|---|---|
| GPU | NVIDIA H20-3e |
| NVIDIA driver | 595.58.03 |
| Python | 3.10.18 |
| PyTorch | 2.9.0 |
| Transformers | 4.57.1 |
| Triton | 3.5.0 |
| flash-attn | 2.8.3 |
| flashinfer-python | 0.5.3 |
| datasets | 4.8.5 |
| math-verify | 0.8.0 |
| accelerate | 1.11.0 |
| NumPy | 1.26.4 |

The launchers explicitly set `CUDA_HOME=/usr/local/cuda-12.4`, disable the
Triton fallback, run Hugging Face in offline mode, and use the environment at
`/DATA/disk0/qyl/envs/retroinfer`. That absolute environment is not included in
this source archive.
