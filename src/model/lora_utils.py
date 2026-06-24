import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn


DEFAULT_EAP_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_STANDARD_LORA_TARGET_MODULES = ("q_proj", "v_proj")
LORA_SCORE_SOURCES = {
    "rank_score",
    "normalized_abs",
    "raw_abs",
    "sum_abs",
    "mean_abs",
    "sqrt_numel_abs",
}


@dataclass
class LoraComponentSpec:
    component_name: str
    module_name: str
    component_type: str
    rank: int
    alpha: int
    row_slice: tuple[int, int] | None = None
    col_slice: tuple[int, int] | None = None
    head_idx: int | None = None
    a_name: str | None = None
    b_name: str | None = None

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)


class ComponentWiseLoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        specs: list[LoraComponentSpec],
        dropout: float = 0.0,
    ):
        super().__init__()
        if not specs:
            raise ValueError("ComponentWiseLoRALinear requires at least one non-zero-rank spec")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.dropout = nn.Dropout(p=dropout)
        self.specs = []
        for idx, spec in enumerate(specs):
            normalized = LoraComponentSpec(**{**spec.__dict__})
            normalized.a_name = f"lora_A_{idx}"
            normalized.b_name = f"lora_B_{idx}"
            a_shape, b_shape = self._lora_shapes(normalized)
            lora_a = torch.empty(
                a_shape,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            lora_b = torch.empty(
                b_shape,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
            nn.init.zeros_(lora_b)
            self.register_parameter(normalized.a_name, nn.Parameter(lora_a))
            self.register_parameter(normalized.b_name, nn.Parameter(lora_b))
            self.specs.append(normalized)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = F.linear(inputs, self.weight, self.bias)
        delta_total = torch.zeros_like(result)
        for spec in self.specs:
            lora_a = getattr(self, spec.a_name)
            lora_b = getattr(self, spec.b_name)
            lora_inputs = self._select_inputs(inputs, spec)
            low_rank = F.linear(self.dropout(lora_inputs), lora_a)
            delta = F.linear(low_rank, lora_b) * spec.scaling
            if spec.row_slice is None:
                delta_total = delta_total + delta
            else:
                start, end = spec.row_slice
                delta_total[..., start:end] = delta_total[..., start:end] + delta
        return result + delta_total

    def merged_weight(self) -> torch.Tensor:
        merged = self.weight.detach().clone()
        for spec in self.specs:
            lora_a = getattr(self, spec.a_name).detach()
            lora_b = getattr(self, spec.b_name).detach()
            delta = lora_b.matmul(lora_a) * spec.scaling
            if spec.col_slice is not None:
                start, end = spec.col_slice
                merged[:, start:end] = merged[:, start:end] + delta.to(merged.dtype)
            elif spec.row_slice is not None:
                start, end = spec.row_slice
                merged[start:end, :] = merged[start:end, :] + delta.to(merged.dtype)
            else:
                merged = merged + delta.to(merged.dtype)
        return merged

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        destination[prefix + "weight"] = self.merged_weight() if not keep_vars else self.merged_weight().detach()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def _lora_shapes(self, spec: LoraComponentSpec) -> tuple[tuple[int, int], tuple[int, int]]:
        if spec.col_slice is not None:
            start, end = spec.col_slice
            return (spec.rank, end - start), (self.out_features, spec.rank)
        if spec.row_slice is not None:
            start, end = spec.row_slice
            return (spec.rank, self.in_features), (end - start, spec.rank)
        return (spec.rank, self.in_features), (self.out_features, spec.rank)

    def _select_inputs(self, inputs: torch.Tensor, spec: LoraComponentSpec) -> torch.Tensor:
        if spec.col_slice is None:
            return inputs
        start, end = spec.col_slice
        return inputs[..., start:end]

    def extra_repr(self) -> str:
        ranks = ",".join(str(spec.rank) for spec in self.specs)
        return f"in_features={self.in_features}, out_features={self.out_features}, component_ranks=[{ranks}]"


def apply_lora_to_model(
    model: nn.Module,
    mode: str,
    info_dir: str | None = None,
    target_modules: str | list[str] | tuple[str, ...] = "auto",
    default_rank: int = 8,
    alpha: int = 32,
    dropout: float = 0.05,
    alpha_strategy: str = "constant",
    rank_pattern_path: str | None = None,
    alpha_pattern_path: str | None = None,
    component_scores_path: str | None = None,
    head_min_rank: int = 0,
    head_max_rank: int = 32,
    head_rank_multiple: int = 1,
    head_rank_score_source: str = "rank_score",
) -> tuple[nn.Module, dict]:
    mode = (mode or "standard").strip()
    if mode not in {"standard", "projection_matrix", "head"}:
        raise ValueError("lora_mode must be one of: standard, projection_matrix, head")
    if not info_dir or str(info_dir).lower() == "none":
        info_dir = None
    if mode == "head":
        return apply_headwise_lora_to_model(
            model=model,
            info_dir=info_dir,
            target_modules=target_modules,
            alpha=alpha,
            dropout=dropout,
            alpha_strategy=alpha_strategy,
            component_scores_path=component_scores_path,
            min_rank=head_min_rank,
            max_rank=head_max_rank,
            rank_multiple=head_rank_multiple,
            rank_score_source=head_rank_score_source,
        )
    return apply_peft_lora_to_model(
        model=model,
        mode=mode,
        info_dir=info_dir,
        target_modules=target_modules,
        default_rank=default_rank,
        alpha=alpha,
        dropout=dropout,
        alpha_strategy=alpha_strategy,
        rank_pattern_path=rank_pattern_path,
        alpha_pattern_path=alpha_pattern_path,
    )


def apply_peft_lora_to_model(
    model: nn.Module,
    mode: str,
    info_dir: str | None,
    target_modules: str | list[str] | tuple[str, ...],
    default_rank: int,
    alpha: int,
    dropout: float,
    alpha_strategy: str,
    rank_pattern_path: str | None = None,
    alpha_pattern_path: str | None = None,
) -> tuple[nn.Module, dict]:
    rank_pattern = {}
    alpha_pattern = {}
    if mode == "projection_matrix":
        rank_pattern_path = _resolve_info_file(info_dir, rank_pattern_path, "rank_pattern.json")
        rank_pattern = _load_rank_pattern(rank_pattern_path)
        alpha_pattern_path = _resolve_optional_info_file(info_dir, alpha_pattern_path, "alpha_pattern.json")
        alpha_pattern = _load_rank_pattern(alpha_pattern_path) if alpha_pattern_path else {}
    if alpha_strategy == "twice_rank" and rank_pattern and not alpha_pattern:
        alpha_pattern = {module_name: max(1, int(rank) * 2) for module_name, rank in rank_pattern.items()}
    parsed_targets = parse_target_modules(target_modules)
    if parsed_targets == ["auto"]:
        parsed_targets = list(rank_pattern.keys()) if rank_pattern else list(DEFAULT_STANDARD_LORA_TARGET_MODULES)
    peft_rank = max(1, int(default_rank))
    if rank_pattern and all(rank > 0 for rank in rank_pattern.values()):
        peft_rank = max(1, min(rank_pattern.values()))
    peft_config = LoraConfig(
        r=peft_rank,
        lora_alpha=int(alpha),
        target_modules=parsed_targets,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    report = {
        "lora_backend": "peft",
        "lora_mode": mode,
        "target_modules": parsed_targets,
        "default_rank": peft_rank,
        "rank_pattern_path": rank_pattern_path,
        "rank_pattern_count": len(rank_pattern),
        "alpha_pattern_count": len(alpha_pattern),
        "trainable_parameters": trainable_parameter_summary(model),
    }
    return model, report


def apply_headwise_lora_to_model(
    model: nn.Module,
    info_dir: str | None,
    target_modules: str | list[str] | tuple[str, ...],
    alpha: int,
    dropout: float,
    alpha_strategy: str,
    component_scores_path: str | None = None,
    min_rank: int = 0,
    max_rank: int = 32,
    rank_multiple: int = 1,
    rank_score_source: str = "rank_score",
) -> tuple[nn.Module, dict]:
    component_scores_path = _resolve_info_file(info_dir, component_scores_path, "component_scores.json")
    scores = _load_component_scores(component_scores_path)
    summary_path = _resolve_optional_info_file(info_dir, None, "summary.json")
    summary = _load_json(summary_path) if summary_path else {}
    if summary.get("attention_granularity") not in {None, "head"}:
        print(
            "[LoRA] Warning: lora_mode=head is using component scores whose "
            f"summary attention_granularity={summary.get('attention_granularity')!r}."
        )
    parsed_targets = parse_target_modules(target_modules)
    if parsed_targets == ["auto"]:
        parsed_targets = sorted({score["component_type"] for score in scores})
    specs_by_module = build_component_lora_specs(
        scores=scores,
        target_modules=parsed_targets,
        alpha=alpha,
        alpha_strategy=alpha_strategy,
        min_rank=min_rank,
        max_rank=max_rank,
        rank_multiple=rank_multiple,
        rank_score_source=rank_score_source,
    )
    if not specs_by_module:
        raise ValueError("No non-zero-rank component scores matched the requested head-wise LoRA configuration")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replaced_modules = []
    named_modules = dict(model.named_modules())
    for module_name, specs in specs_by_module.items():
        module = named_modules.get(module_name)
        if module is None:
            raise KeyError(f"LoRA component score references missing module: {module_name}")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Head-wise LoRA can only wrap nn.Linear modules, got {type(module)} for {module_name}")
        parent, child_name = _get_parent_module(model, module_name)
        setattr(parent, child_name, ComponentWiseLoRALinear(module, specs, dropout=dropout))
        replaced_modules.append(module_name)
    report = {
        "lora_backend": "component_wise",
        "lora_mode": "head",
        "component_scores_path": component_scores_path,
        "summary_path": summary_path,
        "target_modules": parsed_targets,
        "wrapped_module_count": len(replaced_modules),
        "component_count": sum(len(specs) for specs in specs_by_module.values()),
        "wrapped_modules": replaced_modules,
        "trainable_parameters": trainable_parameter_summary(model),
    }
    return model, report


def build_component_lora_specs(
    scores: list[dict],
    target_modules: list[str],
    alpha: int,
    alpha_strategy: str,
    min_rank: int,
    max_rank: int,
    rank_multiple: int,
    rank_score_source: str,
) -> dict[str, list[LoraComponentSpec]]:
    if rank_score_source not in LORA_SCORE_SOURCES:
        raise ValueError(f"Unsupported lora_head_rank_score_source: {rank_score_source}")
    target_set = set(target_modules)
    filtered = [score for score in scores if score.get("component_type") in target_set]
    if not filtered:
        return {}
    values = [_score_value(score, rank_score_source) for score in filtered]
    ranks = _allocate_ranks(values, min_rank=min_rank, max_rank=max_rank, rank_multiple=rank_multiple)
    specs_by_module: dict[str, list[LoraComponentSpec]] = {}
    for score, rank in zip(filtered, ranks):
        if rank <= 0:
            continue
        module_name = score.get("module_name") or str(score["parameter_name"]).removesuffix(".weight")
        component_alpha = _component_alpha(rank, alpha=alpha, alpha_strategy=alpha_strategy)
        spec = LoraComponentSpec(
            component_name=score.get("component_name") or _component_name(score),
            module_name=module_name,
            component_type=score.get("component_type", module_name.rsplit(".", 1)[-1]),
            rank=int(rank),
            alpha=int(component_alpha),
            row_slice=_optional_slice(score.get("row_slice")),
            col_slice=_optional_slice(score.get("col_slice")),
            head_idx=score.get("head_idx"),
        )
        specs_by_module.setdefault(module_name, []).append(spec)
    return specs_by_module


def parse_target_modules(target_modules: str | list[str] | tuple[str, ...]) -> list[str]:
    if target_modules is None:
        return ["auto"]
    if isinstance(target_modules, str):
        values = [item.strip() for item in target_modules.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in target_modules if str(item).strip()]
    return values or ["auto"]


def trainable_parameter_summary(model: nn.Module) -> dict:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": int(trainable),
        "total": int(total),
        "percent": float(trainable / total * 100) if total else 0.0,
    }


def has_component_wise_lora(model: nn.Module) -> bool:
    return any(isinstance(module, ComponentWiseLoRALinear) for module in model.modules())


def _load_rank_pattern(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    pattern = _load_json(path)
    return {str(key): int(value) for key, value in pattern.items() if int(value) > 0}


def _load_component_scores(path: str) -> list[dict]:
    scores = _load_json(path)
    if not isinstance(scores, list):
        raise ValueError(f"Expected component_scores.json to contain a list, got {type(scores)}")
    return scores


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_info_file(info_dir: str | None, explicit_path: str | None, filename: str) -> str:
    path = explicit_path if explicit_path and str(explicit_path).lower() != "none" else None
    if path is None and info_dir:
        path = os.path.join(info_dir, filename)
    if path is None:
        raise ValueError(f"Missing LoRA info file: pass --lora_info_dir or explicit {filename} path")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA info file does not exist: {path}")
    return path


def _resolve_optional_info_file(info_dir: str | None, explicit_path: str | None, filename: str) -> str | None:
    path = explicit_path if explicit_path and str(explicit_path).lower() != "none" else None
    if path is None and info_dir:
        candidate = os.path.join(info_dir, filename)
        path = candidate if os.path.exists(candidate) else None
    if path and not os.path.exists(path):
        raise FileNotFoundError(f"LoRA info file does not exist: {path}")
    return path


def _allocate_ranks(values: list[float], min_rank: int, max_rank: int, rank_multiple: int) -> list[int]:
    if min_rank < 0 or max_rank < min_rank:
        raise ValueError("Expected 0 <= lora_head_min_rank <= lora_head_max_rank")
    if rank_multiple < 1:
        raise ValueError("lora_head_rank_multiple must be >= 1")
    if not values:
        return []
    min_score = min(values)
    max_score = max(values)
    ranks = []
    for value in values:
        if math.isclose(max_score, min_score):
            rank = max_rank if value > 0 else min_rank
        else:
            normalized = (value - min_score) / (max_score - min_score)
            rank = min_rank + normalized * (max_rank - min_rank)
        ranks.append(_round_rank(rank, min_rank=min_rank, max_rank=max_rank, rank_multiple=rank_multiple))
    return ranks


def _round_rank(rank: float, min_rank: int, max_rank: int, rank_multiple: int) -> int:
    rounded = int(round(rank / rank_multiple) * rank_multiple)
    return max(min_rank, min(max_rank, rounded))


def _component_alpha(rank: int, alpha: int, alpha_strategy: str) -> int:
    if alpha_strategy == "constant":
        return int(alpha)
    if alpha_strategy == "twice_rank":
        return max(1, int(rank) * 2)
    raise ValueError("lora_alpha_strategy must be constant or twice_rank")


def _score_value(score: dict, source: str) -> float:
    if source in {"rank_score", "normalized_abs"}:
        return float(score.get("rank_score", 0.0))
    if source in {"raw_abs", "sum_abs"}:
        return abs(float(score.get("raw_score", score.get("abs_score", 0.0))))
    if source == "mean_abs":
        return abs(float(score.get("mean_score", 0.0)))
    if source == "sqrt_numel_abs":
        return abs(float(score.get("sqrt_numel_score", 0.0)))
    raise ValueError(f"Unsupported score source: {source}")


def _optional_slice(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"Expected slice with two values, got {value}")
    return int(value[0]), int(value[1])


def _component_name(score: dict) -> str:
    parameter_name = score.get("parameter_name", "")
    head_idx = score.get("head_idx")
    if head_idx is None:
        return parameter_name
    return f"{parameter_name}.head_{head_idx}"


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]