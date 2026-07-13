# Environment setup

The repository provides two isolated Conda environments. Do not combine their PyTorch or CUDA packages.

## Blackwell GPUs

Use this environment for GPUs with compute capability `sm_120`, including the NVIDIA RTX PRO 6000 Blackwell. PyTorch 2.1.1 with CUDA 11.8 cannot execute kernels on these devices.

```bash
conda env create -f environment-blackwell.yml
conda activate future-localization-blackwell
```

The file pins the Hugging Face stack used by this project and installs the verified CUDA 13.0 PyTorch wheel. If that wheel index is unavailable on another machine, install the newest PyTorch CUDA wheel supported by the local NVIDIA driver and keep the remaining package pins unchanged.

Verify the runtime before launching an experiment:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul:", (x @ x).shape)
PY
```

For `sm_120`, this project rejects PyTorch builds whose CUDA version is older than 12.8. A traceback containing an interpreter from an older environment means that VS Code is launching the wrong Python executable.

## Pre-Blackwell GPUs

Use the legacy environment on GPUs supported by CUDA 11.8:

```bash
conda env create -f environment-legacy.yml
conda activate future-localization-legacy
```

Do not run `pip install torch`, `pip install triton`, or install pip `nvidia-*-cu12` packages in this environment. Those packages can overwrite the Conda PyTorch 2.1.1 installation and produce binary or linker errors.

## Version constraints

The following combinations are intentional:

- `numpy==1.26.4` avoids NumPy 2.x ABI issues in the legacy stack.
- `mkl==2023.1.0` avoids the `iJIT_NotifyEvent` linker issue with PyTorch 2.1.1.
- `transformers==4.37.2`, `peft==0.10.0`, and `accelerate==0.26.1` are kept together. Upgrading only one of these packages can break Trainer or PEFT APIs.
- Graph rendering requires the `dot` executable supplied by the Conda `graphviz` package.

Generated model checkpoints and Hugging Face caches are intentionally excluded from Git.
