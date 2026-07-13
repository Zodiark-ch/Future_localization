from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from EAP_forComponent.schemas import DEFAULT_CACHE_DIR, DEFAULT_TARGET_MODULES


@dataclass
class EdgeScore:
    edge_id: str
    source_node: str
    destination_module: str
    layer_idx: int
    component_type: str
    score_token_mode: str
    localization_mode: str
    raw_score: float
    abs_score: float
    mean_score: float
    sqrt_numel_score: float
    rank_score: float
    token_count: int
    element_count: int
    numel: int
    current_score: float | None = None
    future_directional_score_theta: float | None = None
    future_directional_score_theta_hat: float | None = None
    future_correction: float | None = None
    future_step_k: float | None = None
    mean_raw_rank: float | None = None


@dataclass
class EAPLogicalCircuitConfig:
    model_name_or_path: str = "mistralai/Mistral-7B-v0.1"
    tokenizer_name_or_path: str | None = None
    output_dir: str = "files/logical_circuit/ioi_mistral"
    dataset_name: str = "ioi_mistral"
    data_path: str | None = None
    corruption_column: str = "corrupted"
    input_format: str = "auto"
    metric: str = "task_loss"
    max_samples: int | None = 128
    batch_size: int = 1
    max_length: int | None = None
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
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
    circuit_construction: str = "node_induced"
    node_topn: int = 500
    edge_topn: int | None = None
    edge_threshold: float | None = None
    edge_score_abs: bool = True
    graph: bool = False
    graph_node_topn: int = 25
    graph_edge_threshold_ratio: float = 0.1
    graph_edge_budget_multiplier: float = 3.0
    graph_input_edge_limit_ratio: float = 0.3
    component_granularity: str = "head"
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
