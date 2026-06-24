# CSAT environment setup

## Recommended setup for Blackwell GPUs

Use this setup on GPUs such as NVIDIA RTX PRO 6000 Blackwell (`sm_120`). The old PyTorch 2.1.1/CUDA 11.8 stack does not support this GPU architecture.

```bash
conda create -n LLMSFT_BW python=3.11 -y
conda activate LLMSFT_BW

python -m pip install --upgrade pip setuptools wheel

# The current machine driver reports CUDA 13.2 support. The verified PyTorch wheel here is CUDA 13.0.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Keep the Hugging Face stack pinned to the versions verified with this project.
python -m pip install numpy==1.26.4 datasets wandb transformers==4.37.2 peft==0.10.0 accelerate==0.26.1 sentencepiece sentence-transformers==2.6.1
python -m pip install git+https://github.com/jinghanjia/fastargs
python -m pip install terminaltables sacrebleu rouge_score matplotlib seaborn scikit-learn pandas scipy tqdm huggingface_hub

cd lm-evaluation-harness
python -m pip install -e .
cd ..
```

Verify that PyTorch sees Blackwell without an `sm_120 is not compatible` warning:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('torch cuda', torch.version.cuda)
print('available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
print((x @ x).shape)
PY
```

If the `cu130` wheel index is unavailable on a different machine, check https://pytorch.org/get-started/locally/ and choose the newest CUDA wheel supported by the driver. With driver `595.58.03`, CUDA 13.0 wheels are supported.

When using VS Code's run button, confirm the traceback path and startup print use the Blackwell environment. A traceback containing `/home/chenhang/.conda/envs/LLMSFT/lib/python3.9/` means the old incompatible environment is still being used. The Blackwell environment should show `/home/chenhang/.conda/envs/LLMSFT_BW/bin/python`, Python 3.11, and `torch 2.12.0+cu130`.

## Legacy setup for older GPUs

```bash
conda create -n [Name] python=3.9
conda activate [Name]

# Install the PyTorch stack with conda. Do not install or upgrade torch with pip later.
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Keep NumPy/MKL compatible with PyTorch 2.1.1.
# This avoids the NumPy 2.x warning and the MKL 2025 iJIT_NotifyEvent linker error.
conda install numpy==1.26.4 mkl==2023.1.0 -c defaults

pip install datasets wandb transformers==4.37.2 peft==0.10.0 accelerate==0.26.1 sentencepiece sentence-transformers==2.6.1
pip install git+https://github.com/jinghanjia/fastargs
pip install terminaltables sacrebleu rouge_score matplotlib seaborn scikit-learn

cd lm-evaluation-harness
pip install -e .
```

Notes:

- Do not run `pip install torch`, `pip install triton`, or install `nvidia-*-cu12` packages in this environment.
- Keep `peft==0.10.0` with `transformers==4.37.2`. Newer PEFT versions, such as `0.17.1`, expect newer Transformers cache classes like `EncoderDecoderCache` and will fail to import with Transformers 4.37.2.
- Keep `accelerate==0.26.1` with `transformers==4.37.2`. Newer Accelerate 1.x versions remove the `dispatch_batches` argument that this Transformers Trainer still passes to `Accelerator`.
- The repaired PyTorch 2.1.1/CUDA 11.8 stack does not support Blackwell GPUs such as RTX PRO 6000 with compute capability `sm_120`. On those GPUs, CUDA kernels may fail with low-level device-side asserts even when CPU-side tensor ranges look valid. Use a PyTorch/CUDA build that supports `sm_120`, or run with `overall.use_cpu=1` only for debugging.
- If rebuilding from an exported environment file, remove pip entries such as `torch==2.8.0`, `triton==3.4.0`, and `nvidia-*-cu12`; they can overwrite the conda PyTorch 2.1.1/CUDA 11.8 files and cause `ModuleNotFoundError: No module named 'torch._strobelight'`.
- A verified working combination is `torch==2.1.1`, `torchvision==0.16.1`, `torchaudio==2.1.1`, `pytorch-cuda=11.8`, `numpy==1.26.4`, `mkl==2023.1.0`, `transformers==4.37.2`, `peft==0.10.0`, and `accelerate==0.26.1`.