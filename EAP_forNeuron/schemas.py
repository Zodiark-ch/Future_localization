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
class NeuronTarget:
    parameter_name: str
    module_name: str
    module: nn.Linear
    weight: torch.Tensor
    shape: torch.Size
    flat_indices: torch.Tensor
    weight_values: torch.Tensor

    @property
    def candidate_count(self) -> int:
        return int(self.flat_indices.numel())

    @property
    def total_count(self) -> int:
        return int(self.weight.numel())


@dataclass
class ScoreShard:
    parameter_name: str
    shape: torch.Size
    flat_indices: torch.Tensor
    scores: torch.Tensor

    @property
    def candidate_count(self) -> int:
        return int(self.flat_indices.numel())


@dataclass
class EAPNeuronConfig:
    model_name_or_path: str = "mistralai/Mistral-7B-v0.1"
    tokenizer_name_or_path: str | None = None
    mask_path: str | None = None
    output_dir: str | None = None
    dataset_name: str = "ioi_mistral"
    data_path: str | None = None
    corruption_column: str = "corrupted"
    metric: str = "task_loss"
    output_ratio: float = 0.1
    ratio_base: str = "all"
    max_samples: int | None = 128
    batch_size: int = 1
    max_length: int | None = None
    score_abs: bool = True
    score_token_mode: str = "label_position"
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    include_lm_head: bool = False
    include_embed_tokens: bool = False
    unsupported_policy: str = "drop"
    score_dtype: str = "float32"
    device: str = "cuda:0"
    use_bfloat16: bool = True
    use_cpu: bool = False
    cache_dir: str | None = DEFAULT_CACHE_DIR
    save_scores: bool = False
    row_chunk_size: int = 256
    max_concat_candidates: int = 50_000_000
    metadata: dict[str, Any] = field(default_factory=dict)
