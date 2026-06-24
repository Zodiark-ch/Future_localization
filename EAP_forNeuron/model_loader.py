from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def load_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
    cache_dir: str | None = None,
    device: str = "cuda:0",
    use_bfloat16: bool = True,
    use_cpu: bool = False,
):
    ensure_src_on_path()
    from modeling_patches import patch_mistral_rotary_embedding

    patch_mistral_rotary_embedding()
    tokenizer_source = tokenizer_name_or_path or model_name_or_path
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    load_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
    }
    if use_cpu or device == "cpu" or not torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"
        resolved_device = torch.device("cpu")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if use_bfloat16 else torch.float16
        cuda_index = _cuda_index(device)
        load_kwargs["device_map"] = {"": cuda_index}
        resolved_device = torch.device(f"cuda:{cuda_index}")

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=cache_dir,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    return model, tokenizer, resolved_device


def _cuda_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0
