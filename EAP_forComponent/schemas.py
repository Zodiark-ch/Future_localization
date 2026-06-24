from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
DEFAULT_CACHE_DIR = "/home/chenhang/CSAT/.cache"


@dataclass
class PairExample:
    clean: str
    corrupted: str
    correct_idx: int
    incorrect_idx: int


@dataclass
class PairBatch:
    clean_input_ids: torch.Tensor
    clean_attention_mask: torch.Tensor
    corrupted_input_ids: torch.Tensor
    corrupted_attention_mask: torch.Tensor
    labels: torch.Tensor
    correct_idx: torch.Tensor
    incorrect_idx: torch.Tensor
    label_positions: torch.Tensor

    def to(self, device: torch.device | str) -> "PairBatch":
        return PairBatch(
            clean_input_ids=self.clean_input_ids.to(device),
            clean_attention_mask=self.clean_attention_mask.to(device),
            corrupted_input_ids=self.corrupted_input_ids.to(device),
            corrupted_attention_mask=self.corrupted_attention_mask.to(device),
            labels=self.labels.to(device),
            correct_idx=self.correct_idx.to(device),
            incorrect_idx=self.incorrect_idx.to(device),
            label_positions=self.label_positions.to(device),
        )


@dataclass
class ComponentTarget:
    parameter_name: str
    module_name: str
    layer_idx: int
    component_type: str
    granularity: str
    head_idx: int | None
    head_kind: str | None
    row_slice: tuple[int, int] | None
    col_slice: tuple[int, int] | None
    module: nn.Linear
    shape: torch.Size
    numel: int

    @property
    def component_name(self) -> str:
        if self.head_idx is None:
            return self.parameter_name
        return f"{self.parameter_name}.head_{self.head_idx}"

    @property
    def rank_pattern_key(self) -> str:
        return self.parameter_name.removesuffix(".weight")


@dataclass
class ComponentScore:
    parameter_name: str
    module_name: str
    layer_idx: int
    component_type: str
    granularity: str
    score_token_mode: str
    head_idx: int | None
    head_kind: str | None
    row_slice: tuple[int, int] | None
    col_slice: tuple[int, int] | None
    raw_score: float
    abs_score: float
    mean_score: float
    sqrt_numel_score: float
    rank_score: float
    shape: tuple[int, ...]
    numel: int
    token_count: int
    element_count: int
    localization_mode: str = "current"
    current_score: float | None = None
    future_directional_score_theta: float | None = None
    future_directional_score_theta_hat: float | None = None
    future_correction: float | None = None
    future_step_k: float | None = None
    mean_raw_rank: float | None = None

    @property
    def component_name(self) -> str:
        if self.head_idx is None:
            return self.parameter_name
        return f"{self.parameter_name}.head_{self.head_idx}"

    @property
    def rank_pattern_key(self) -> str:
        return self.parameter_name.removesuffix(".weight")


@dataclass
class EAPComponentConfig:
    model_name_or_path: str = "mistralai/Mistral-7B-v0.1"
    tokenizer_name_or_path: str | None = None
    output_dir: str = "files/component_scores/ioi_mistral"
    dataset_name: str = "ioi_mistral"
    data_path: str | None = None
    corruption_column: str = "corrupted"
    input_format: str = "auto"
    metric: str = "task_loss"
    max_samples: int | None = 128
    batch_size: int = 1
    max_length: int | None = None
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    attention_granularity: str = "projection_matrix"
    localization_mode: str = "current"
    future_model_name_or_path: str | None = None
    future_model_cache_dir: str | None = None
    future_step_k: float = 1.0
    future_step_k_min: float = 0.0
    future_step_k_max: float = 10.0
    future_step_k_samples: int = 10
    future_step_k_seed: int = 0
    future_delta_parameter_filter: str | None = None
    future_hvp_strategy: str = "finite_difference"
    future_finite_difference_epsilon: float = 1e-3
    score_token_mode: str = "all_active"
    score_normalization: str = "sum"
    rank_score_source: str = "normalized_abs"
    min_rank: int = 0
    max_rank: int = 32
    rank_budget: int | None = None
    rank_multiple: int = 1
    head_to_matrix_aggregation: str = "mean"
    mask_fill_strategy: str = "random"
    mask_seed: int = 0
    mask_min_keep_ratio: float = 0.0
    mask_max_keep_ratio: float = 1.0
    mask_all_parameters: bool = True
    device: str = "cuda:0"
    use_bfloat16: bool = True
    use_cpu: bool = False
    cache_dir: str | None = DEFAULT_CACHE_DIR
    capture_device: str = "cpu"
    metadata: dict[str, Any] = field(default_factory=dict)
