from __future__ import annotations

import argparse

from EAP_forLogicalCircuit.runner import EAPForLogicalCircuitRunner
from EAP_forLogicalCircuit.datasets import SUPPORTED_DATASET_NAMES
from EAP_forLogicalCircuit.schemas import EAPLogicalCircuitConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute edge-level current attribution for logical circuit construction.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tokenizer_name_or_path")
    parser.add_argument("--cache_dir")
    parser.add_argument("--dataset_name", choices=SUPPORTED_DATASET_NAMES, default="ioi_mistral")
    parser.add_argument("--data_path")
    parser.add_argument("--corruption_column", default="corrupted")
    parser.add_argument("--input_format", choices=["auto", "prompt", "raw"], default="auto")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metric", choices=["task_loss", "logit_diff"], default="task_loss")
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--localization_mode", choices=["current", "future"], default="current")
    parser.add_argument("--future_model_name_or_path")
    parser.add_argument("--future_model_cache_dir")
    parser.add_argument("--future_step_k", type=float, default=1, help="Fixed K used when --future_step_k_samples <= 1.")
    parser.add_argument("--future_step_k_min", type=float, default=0)
    parser.add_argument("--future_step_k_max", type=float, default=8)
    parser.add_argument("--future_step_k_samples", type=int, default=1)
    parser.add_argument("--future_step_k_seed", type=int, default=0)
    parser.add_argument("--future_delta_parameter_filter", default=None)
    parser.add_argument("--future_hvp_strategy", choices=["hvp", "finite_difference"], default="finite_difference")
    parser.add_argument("--future_finite_difference_epsilon", type=float, default=1e-3)
    parser.add_argument("--score_token_mode", choices=["all_active", "label_position"], default="all_active")
    parser.add_argument("--score_normalization", choices=["sum", "mean", "sqrt_numel"], default="sum")
    parser.add_argument("--edge_topn", type=int, default=20000)
    parser.add_argument("--edge_threshold", type=float, default=None)
    parser.add_argument("--circuit_construction", choices=["node_induced", "edge_attribution"], default="node_induced")
    parser.add_argument("--node_topn", type=int, default=500)
    parser.add_argument("--edge_score_abs", type=_str_to_bool, default=True)
    parser.add_argument("--graph", type=_str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--graph_node_topn", type=int, default=60)
    parser.add_argument(
        "--graph_edge_threshold_ratio",
        type=float,
        default=0.1,
        help="Deprecated for graph export; selection now uses connectivity requirements plus edge budget.",
    )
    parser.add_argument(
        "--graph_edge_budget_multiplier",
        type=float,
        default=3.0,
        help="Graph edge budget multiplier: ceil(multiplier * graph node count) after required connectivity edges.",
    )
    parser.add_argument(
        "--graph_input_edge_limit_ratio",
        type=float,
        default=0.3,
        help="Maximum input outgoing graph edges as floor(ratio * graph node count).",
    )
    parser.add_argument("--component_granularity", choices=["projection_matrix", "head"], default="head")
    parser.add_argument("--rank_score_source", default="normalized_abs")
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--max_rank", type=int, default=32)
    parser.add_argument("--rank_budget", type=int, default=None)
    parser.add_argument("--rank_multiple", type=int, default=1)
    parser.add_argument("--head_to_matrix_aggregation", choices=["mean", "max", "sum"], default="mean")
    parser.add_argument("--mask_fill_strategy", choices=["random", "magnitude", "first"], default="random")
    parser.add_argument("--mask_seed", type=int, default=0)
    parser.add_argument("--mask_min_keep_ratio", type=float, default=0.1, help="Lowest-ranked component keep ratio in [0, 1].")
    parser.add_argument("--mask_max_keep_ratio", type=float, default=0.9, help="Highest-ranked component keep ratio in [0, 1].")
    parser.add_argument("--mask_target_only", action="store_true")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_bfloat16", type=_str_to_bool, default=True)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--capture_device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EAPLogicalCircuitConfig(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        corruption_column=args.corruption_column,
        input_format=args.input_format,
        metric=args.metric,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        target_modules=tuple(item.strip() for item in args.target_modules.split(",") if item.strip()),
        localization_mode=args.localization_mode,
        future_model_name_or_path=args.future_model_name_or_path,
        future_model_cache_dir=args.future_model_cache_dir,
        future_step_k=args.future_step_k,
        future_step_k_min=args.future_step_k_min,
        future_step_k_max=args.future_step_k_max,
        future_step_k_samples=args.future_step_k_samples,
        future_step_k_seed=args.future_step_k_seed,
        future_delta_parameter_filter=args.future_delta_parameter_filter,
        future_hvp_strategy=args.future_hvp_strategy,
        future_finite_difference_epsilon=args.future_finite_difference_epsilon,
        score_token_mode=args.score_token_mode,
        score_normalization=args.score_normalization,
        edge_topn=args.edge_topn,
        circuit_construction=args.circuit_construction,
        node_topn=args.node_topn,
        edge_threshold=args.edge_threshold,
        edge_score_abs=args.edge_score_abs,
        graph=args.graph,
        graph_node_topn=args.graph_node_topn,
        graph_edge_threshold_ratio=args.graph_edge_threshold_ratio,
        graph_edge_budget_multiplier=args.graph_edge_budget_multiplier,
        graph_input_edge_limit_ratio=args.graph_input_edge_limit_ratio,
        component_granularity=args.component_granularity,
        rank_score_source=args.rank_score_source,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        rank_budget=args.rank_budget,
        rank_multiple=args.rank_multiple,
        head_to_matrix_aggregation=args.head_to_matrix_aggregation,
        mask_fill_strategy=args.mask_fill_strategy,
        mask_seed=args.mask_seed,
        mask_min_keep_ratio=args.mask_min_keep_ratio,
        mask_max_keep_ratio=args.mask_max_keep_ratio,
        mask_all_parameters=not args.mask_target_only,
        device=args.device,
        use_bfloat16=args.use_bfloat16,
        use_cpu=args.use_cpu,
        cache_dir=args.cache_dir,
        capture_device=args.capture_device,
    )
    EAPForLogicalCircuitRunner(config).run()


def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


if __name__ == "__main__":
    main()
