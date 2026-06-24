import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.lora_utils import ComponentWiseLoRALinear, apply_lora_to_model, has_component_wise_lora


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q) + torch.cat([k, k], dim=-1) + torch.cat([v, v], dim=-1)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()

    def forward(self, x):
        return self.self_attn(x) + self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=1,
            model_type="tiny",
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])

    def forward(self, x=None, input_ids=None, **_kwargs):
        if x is None:
            x = input_ids.float()
        for layer in self.model.layers:
            x = layer(x)
        return x

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs


def test_projection_matrix_lora_uses_peft_rank_pattern(tmp_path):
    info_dir = _write_lora_info_dir(tmp_path)
    model, report = apply_lora_to_model(
        TinyModel(),
        mode="projection_matrix",
        info_dir=str(info_dir),
        target_modules="auto",
        default_rank=1,
        alpha=4,
        dropout=0.0,
    )
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert report["lora_backend"] == "peft"
    assert report["rank_pattern_count"] == 2
    assert any("lora_A" in name for name in trainable_names)


def test_head_lora_wraps_slices_and_exports_merged_weight(tmp_path):
    info_dir = _write_lora_info_dir(tmp_path)
    model, report = apply_lora_to_model(
        TinyModel(),
        mode="head",
        info_dir=str(info_dir),
        target_modules="auto",
        alpha=4,
        dropout=0.0,
        head_min_rank=0,
        head_max_rank=4,
    )
    assert report["lora_backend"] == "component_wise"
    assert has_component_wise_lora(model)
    assert isinstance(model.model.layers[0].self_attn.q_proj, ComponentWiseLoRALinear)
    loss = model(torch.randn(2, 3, 4)).sum()
    loss.backward()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for _name, parameter in trainable)
    state_dict = model.state_dict()
    assert "model.layers.0.self_attn.q_proj.weight" in state_dict
    assert not any("lora_A" in key or "lora_B" in key for key in state_dict)


def _write_lora_info_dir(tmp_path):
    info_dir = tmp_path / "lora_info"
    info_dir.mkdir()
    (info_dir / "rank_pattern.json").write_text(
        json.dumps(
            {
                "model.layers.0.self_attn.q_proj": 2,
                "model.layers.0.self_attn.v_proj": 1,
            }
        ),
        encoding="utf-8",
    )
    component_scores = [
        _score("q_proj", "q_proj.weight.head_0", row_slice=[0, 2], rank_score=1.0),
        _score("q_proj", "q_proj.weight.head_1", row_slice=[2, 4], rank_score=0.2),
        _score("o_proj", "o_proj.weight.head_0", col_slice=[0, 2], rank_score=0.8),
    ]
    (info_dir / "component_scores.json").write_text(json.dumps(component_scores), encoding="utf-8")
    (info_dir / "summary.json").write_text(
        json.dumps({"attention_granularity": "head"}),
        encoding="utf-8",
    )
    return info_dir


def _score(component_type, component_name, row_slice=None, col_slice=None, rank_score=1.0):
    module_name = f"model.layers.0.self_attn.{component_type}"
    return {
        "parameter_name": f"{module_name}.weight",
        "module_name": module_name,
        "component_type": component_type,
        "component_name": f"model.layers.0.self_attn.{component_name}",
        "head_idx": 0,
        "row_slice": row_slice,
        "col_slice": col_slice,
        "rank_score": rank_score,
        "raw_score": rank_score,
        "mean_score": rank_score,
        "sqrt_numel_score": rank_score,
    }