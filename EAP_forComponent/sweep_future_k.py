from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EAP_forComponent.runner import EAPForComponentRunner
from EAP_forComponent.datasets import SUPPORTED_DATASET_NAMES
from EAP_forComponent.schemas import DEFAULT_TARGET_MODULES, EAPComponentConfig


SCORE_KEYS = ("raw_score", "abs_score", "mean_score", "sqrt_numel_score", "rank_score")
TOPK_VALUES = (50, 100, 200, 500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep future_step_k and compare future EAP scores/ranks to current EAP.")
    parser.add_argument("--current_model_name_or_path")
    parser.add_argument("--future_base_model_name_or_path")
    parser.add_argument("--future_model_name_or_path", required=True)
    parser.add_argument(
        "--model_name_or_path",
        help="Compatibility fallback used for both current_model_name_or_path and future_base_model_name_or_path.",
    )
    parser.add_argument("--current_tokenizer_name_or_path")
    parser.add_argument("--future_tokenizer_name_or_path")
    parser.add_argument("--tokenizer_name_or_path")
    parser.add_argument("--cache_dir")
    parser.add_argument("--dataset_name", choices=SUPPORTED_DATASET_NAMES, default="5_digit_arithmetic")
    parser.add_argument("--data_path")
    parser.add_argument("--corruption_column", default="corrupted")
    parser.add_argument("--input_format", choices=["auto", "prompt", "raw"], default="auto")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--work_dir")
    parser.add_argument("--metric", choices=["task_loss", "logit_diff"], default="task_loss")
    parser.add_argument("--target_modules", default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--attention_granularity", choices=["projection_matrix", "head"], default="head")
    parser.add_argument("--future_model_cache_dir")
    parser.add_argument("--future_delta_parameter_filter")
    parser.add_argument("--future_hvp_strategy", choices=["hvp", "finite_difference"], default="finite_difference")
    parser.add_argument("--future_finite_difference_epsilon", type=float, default=1e-3)
    parser.add_argument("--score_token_mode", choices=["all_active", "label_position"], default="all_active")
    parser.add_argument("--score_normalization", choices=["sum", "mean", "sqrt_numel"], default="sum")
    parser.add_argument(
        "--rank_score_source",
        choices=["normalized_abs", "rank_score", "raw_abs", "sum_abs", "mean_abs", "sqrt_numel_abs"],
        default="sum_abs",
    )
    parser.add_argument("--score_key", choices=SCORE_KEYS, default="abs_score")
    parser.add_argument("--head_to_matrix_aggregation", choices=["mean", "max", "sum"], default="mean")
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--max_rank", type=int, default=32)
    parser.add_argument("--rank_budget", type=int, default=None)
    parser.add_argument("--rank_multiple", type=int, default=1)
    parser.add_argument("--mask_fill_strategy", choices=["random", "magnitude", "first"], default="magnitude")
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
    parser.add_argument("--k_start", default="0")
    parser.add_argument("--k_end", default="100")
    parser.add_argument("--k_step", default="5")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--keep_intermediate", action="store_true")
    args = parser.parse_args()
    _resolve_model_args(args, parser)
    return args


def _resolve_model_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.current_model_name_or_path is None:
        args.current_model_name_or_path = args.model_name_or_path
    if args.future_base_model_name_or_path is None:
        args.future_base_model_name_or_path = args.model_name_or_path
    if args.current_tokenizer_name_or_path is None:
        args.current_tokenizer_name_or_path = args.tokenizer_name_or_path
    if args.future_tokenizer_name_or_path is None:
        args.future_tokenizer_name_or_path = args.tokenizer_name_or_path
    if not args.current_model_name_or_path:
        parser.error("--current_model_name_or_path is required unless --model_name_or_path is provided as a fallback.")
    if not args.future_base_model_name_or_path:
        parser.error("--future_base_model_name_or_path is required unless --model_name_or_path is provided as a fallback.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir / "_intermediate"
    work_dir.mkdir(parents=True, exist_ok=True)

    k_values = list(_decimal_range(args.k_start, args.k_end, args.k_step))
    if not k_values:
        raise ValueError("No k values generated. Check --k_start, --k_end, and --k_step.")

    current_output_dir = work_dir / "current"
    _reset_intermediate_dir(current_output_dir, work_dir)
    current_config = _build_config(
        args=args,
        output_dir=current_output_dir,
        localization_mode="current",
        future_step_k=None,
    )
    print(f"[K sweep] Running current localization: {current_output_dir}")
    try:
        current_scores = _run_and_load_scores(current_config)
    finally:
        if not args.keep_intermediate:
            _remove_intermediate_dir(current_output_dir, work_dir)

    current_score_map = _score_map(current_scores, args.score_key)
    current_rank_map = _rank_map(current_scores, args.score_key)
    current_ranking = _ranking_list(current_scores, args.score_key)
    score_results: list[dict[str, Any]] = []
    rank_results: list[dict[str, Any]] = []
    topk_accuracy_results: list[dict[str, Any]] = []

    for index, k_value in enumerate(k_values, start=1):
        k_label = _k_label(k_value)
        future_output_dir = work_dir / f"future_k_{k_label}"
        _reset_intermediate_dir(future_output_dir, work_dir)
        future_config = _build_config(
            args=args,
            output_dir=future_output_dir,
            localization_mode="future",
            future_step_k=float(k_value),
        )
        print(f"[K sweep] Running future localization {index}/{len(k_values)}: k={k_value}")
        try:
            future_scores = _run_and_load_scores(future_config)
            future_score_map = _score_map(future_scores, args.score_key)
            future_rank_map = _rank_map(future_scores, args.score_key)
            future_ranking = _ranking_list(future_scores, args.score_key)
            _validate_component_sets(current_score_map, future_score_map, k_value)
            score_difference = sum(
                abs(future_score_map[component_name] - current_score)
                for component_name, current_score in current_score_map.items()
            )
            rank_difference = sum(
                abs(future_rank_map[component_name] - current_rank)
                for component_name, current_rank in current_rank_map.items()
            )
            score_results.append(
                {
                    "k": float(k_value),
                    "k_label": str(k_value),
                    "score_difference": float(score_difference),
                }
            )
            rank_results.append(
                {
                    "k": float(k_value),
                    "k_label": str(k_value),
                    "rank_difference": int(rank_difference),
                }
            )
            topk_accuracy = _topk_overlap_accuracies(current_ranking, future_ranking, TOPK_VALUES)
            topk_accuracy_results.append(
                {
                    "k": float(k_value),
                    "k_label": str(k_value),
                    "score_difference": float(score_difference),
                    "rank_difference": int(rank_difference),
                    **topk_accuracy,
                }
            )
            print(
                f"[K sweep] k={k_value}: score_difference={score_difference:.8g}, "
                f"rank_difference={rank_difference}"
            )
            print(f"[K sweep] k={k_value}: {_format_topk_accuracies(topk_accuracy, TOPK_VALUES)}")
            _print_current_top50_future_breakdown(
                k_value=k_value,
                current_ranking=current_ranking,
                current_scores=current_scores,
                future_scores=future_scores,
                future_rank_map=future_rank_map,
                score_key=args.score_key,
            )
        finally:
            if not args.keep_intermediate:
                _remove_intermediate_dir(future_output_dir, work_dir)

    if not args.keep_intermediate:
        _remove_empty_work_dir(work_dir)

    top_score_results = sorted(score_results, key=lambda item: item["score_difference"])[: args.top_n]
    top_rank_results = sorted(rank_results, key=lambda item: item["rank_difference"])[: args.top_n]
    top_top50_accuracy_results = sorted(
        topk_accuracy_results,
        key=lambda item: (-item["top50_accuracy"], -item["top50_overlap"], item["score_difference"]),
    )[: args.top_n]
    _write_json(output_dir / "score_differences.json", score_results)
    _write_json(output_dir / "rank_differences.json", rank_results)
    _write_json(output_dir / "topk_accuracies.json", topk_accuracy_results)
    _write_json(
        output_dir / "k_sweep_summary.json",
        {
            "current_model_name_or_path": args.current_model_name_or_path,
            "future_base_model_name_or_path": args.future_base_model_name_or_path,
            "future_model_name_or_path": args.future_model_name_or_path,
            "current_tokenizer_name_or_path": args.current_tokenizer_name_or_path,
            "future_tokenizer_name_or_path": args.future_tokenizer_name_or_path,
            "dataset_name": args.dataset_name,
            "data_path": args.data_path,
            "score_key": args.score_key,
            "k_start": args.k_start,
            "k_end": args.k_end,
            "k_step": args.k_step,
            "component_count": len(current_score_map),
            "current_top_components": current_ranking[: args.top_n],
            "top_score_differences": top_score_results,
            "top_rank_differences": top_rank_results,
            "top_top50_accuracies": top_top50_accuracy_results,
            "kept_intermediate": bool(args.keep_intermediate),
        },
    )

    print(f"[K sweep] Saved score differences: {output_dir / 'score_differences.json'}")
    print(f"[K sweep] Saved rank differences: {output_dir / 'rank_differences.json'}")
    print(f"[K sweep] Saved top-k accuracies: {output_dir / 'topk_accuracies.json'}")
    print(f"[K sweep] Saved summary: {output_dir / 'k_sweep_summary.json'}")
    _print_top_results("Top score-difference k values", top_score_results, "score_difference")
    _print_top_results("Top rank-difference k values", top_rank_results, "rank_difference")
    _print_top_results("Top top50-accuracy k values", top_top50_accuracy_results, "top50_accuracy")


def _build_config(
    args: argparse.Namespace,
    output_dir: Path,
    localization_mode: str,
    future_step_k: float | None,
) -> EAPComponentConfig:
    if localization_mode == "current":
        model_name_or_path = args.current_model_name_or_path
        tokenizer_name_or_path = args.current_tokenizer_name_or_path
    elif localization_mode == "future":
        model_name_or_path = args.future_base_model_name_or_path
        tokenizer_name_or_path = args.future_tokenizer_name_or_path
    else:
        raise ValueError(f"Unsupported localization_mode: {localization_mode}")
    return EAPComponentConfig(
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        output_dir=str(output_dir),
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        corruption_column=args.corruption_column,
        input_format=args.input_format,
        metric=args.metric,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        target_modules=tuple(item.strip() for item in args.target_modules.split(",") if item.strip()),
        attention_granularity=args.attention_granularity,
        localization_mode=localization_mode,
        future_model_name_or_path=args.future_model_name_or_path if localization_mode == "future" else None,
        future_model_cache_dir=args.future_model_cache_dir,
        future_step_k=1.0 if future_step_k is None else future_step_k,
        future_step_k_min=0.0 if future_step_k is None else future_step_k,
        future_step_k_max=0.0 if future_step_k is None else future_step_k,
        future_step_k_samples=1,
        future_step_k_seed=0,
        future_delta_parameter_filter=args.future_delta_parameter_filter,
        future_hvp_strategy=args.future_hvp_strategy,
        future_finite_difference_epsilon=args.future_finite_difference_epsilon,
        score_token_mode=args.score_token_mode,
        score_normalization=args.score_normalization,
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
        metadata={
            "k_sweep": {
                "score_key": args.score_key,
                "stage": localization_mode,
                "future_step_k": future_step_k,
            }
        },
    )


def _run_and_load_scores(config: EAPComponentConfig) -> list[dict[str, Any]]:
    runner: EAPForComponentRunner | None = None
    try:
        runner = EAPForComponentRunner(config)
        paths = runner.run()
        score_path = paths["component_scores_json"]
        with open(score_path, "r", encoding="utf-8") as file:
            return json.load(file)
    finally:
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _score_map(scores: list[dict[str, Any]], score_key: str) -> dict[str, float]:
    score_map: dict[str, float] = {}
    for score in scores:
        component_name = str(score["component_name"])
        score_map[component_name] = _score_value(score, score_key)
    return score_map


def _rank_map(scores: list[dict[str, Any]], score_key: str) -> dict[str, int]:
    return {item["component_name"]: rank for rank, item in enumerate(_ranking_list(scores, score_key), start=1)}


def _ranking_list(scores: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "component_name": str(score["component_name"]),
                "score": _score_value(score, score_key),
            }
            for score in scores
        ),
        key=lambda item: (-item["score"], item["component_name"]),
    )


def _topk_overlap_accuracies(
    current_ranking: list[dict[str, Any]],
    future_ranking: list[dict[str, Any]],
    topk_values: tuple[int, ...],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    component_count = min(len(current_ranking), len(future_ranking))
    for top_k in topk_values:
        denominator = min(top_k, component_count)
        if denominator <= 0:
            overlap = 0
            accuracy = 0.0
        else:
            current_top = {item["component_name"] for item in current_ranking[:denominator]}
            future_top = {item["component_name"] for item in future_ranking[:denominator]}
            overlap = len(current_top & future_top)
            accuracy = overlap / denominator
        results[f"top{top_k}_overlap"] = int(overlap)
        results[f"top{top_k}_denominator"] = int(denominator)
        results[f"top{top_k}_accuracy"] = float(accuracy)
    return results


def _format_topk_accuracies(topk_accuracy: dict[str, Any], topk_values: tuple[int, ...]) -> str:
    parts = []
    for top_k in topk_values:
        overlap = topk_accuracy[f"top{top_k}_overlap"]
        denominator = topk_accuracy[f"top{top_k}_denominator"]
        accuracy = topk_accuracy[f"top{top_k}_accuracy"]
        parts.append(f"top{top_k}={overlap}/{denominator} ({accuracy:.4f})")
    return ", ".join(parts)


def _print_current_top50_future_breakdown(
    k_value: Decimal,
    current_ranking: list[dict[str, Any]],
    current_scores: list[dict[str, Any]],
    future_scores: list[dict[str, Any]],
    future_rank_map: dict[str, int],
    score_key: str,
) -> None:
    current_by_component = _score_entry_map(current_scores)
    future_by_component = _score_entry_map(future_scores)
    top_components = [item["component_name"] for item in current_ranking[:50]]
    component_width = max(14, min(96, max((len(component_name) for component_name in top_components), default=14)))
    print(
        f"[K sweep] k={k_value}: current top50 component score table "
        f"(top50 selected by {score_key}; raw future = activation_delta + correction)"
    )
    header = (
        f"{'rank':>4} | {'component':<{component_width}} | {'current_raw':>14} | "
        f"{'future_raw':>14} | {'future_rank':>11} | {'activation_delta':>16} | {'correction':>14}"
    )
    print(header)
    print("-" * len(header))
    for rank, component_name in enumerate(top_components, start=1):
        current_entry = current_by_component[component_name]
        future_entry = future_by_component[component_name]
        current_raw = _score_value(current_entry, "raw_score")
        future_raw = _score_value(future_entry, "raw_score")
        future_rank = future_rank_map[component_name]
        activation_delta = _optional_score_value(future_entry, "current_score")
        correction = _optional_score_value(future_entry, "future_correction")
        print(
            f"{rank:4d} | {component_name:<{component_width}} | {_format_float(current_raw):>14} | "
            f"{_format_float(future_raw):>14} | {future_rank:11d} | {_format_float(activation_delta):>16} | "
            f"{_format_float(correction):>14}"
        )


def _score_value(score: dict[str, Any], score_key: str) -> float:
    value = score.get(score_key)
    if value is None:
        raise ValueError(f"Score entry for {score.get('component_name')!r} is missing {score_key!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Score entry for {score.get('component_name')!r} has non-finite {score_key}: {value}.")
    return value


def _optional_score_value(score: dict[str, Any], score_key: str) -> float | None:
    value = score.get(score_key)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Score entry for {score.get('component_name')!r} has non-finite {score_key}: {value}.")
    return value


def _score_entry_map(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(score["component_name"]): score for score in scores}


def _format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6e}"


def _validate_component_sets(
    current_scores: dict[str, float],
    future_scores: dict[str, float],
    k_value: Decimal,
) -> None:
    current_components = set(current_scores)
    future_components = set(future_scores)
    if current_components == future_components:
        return
    missing = sorted(current_components - future_components)[:10]
    extra = sorted(future_components - current_components)[:10]
    raise ValueError(
        f"Component set mismatch at k={k_value}: missing_from_future={missing}, extra_in_future={extra}"
    )


def _decimal_range(start: str, end: str, step: str) -> list[Decimal]:
    current = Decimal(str(start))
    stop = Decimal(str(end))
    increment = Decimal(str(step))
    if increment <= 0:
        raise ValueError("--k_step must be positive.")
    values: list[Decimal] = []
    while current <= stop:
        values.append(current)
        current += increment
    return values


def _k_label(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text.replace("-", "minus_").replace(".", "p")


def _reset_intermediate_dir(path: Path, work_dir: Path) -> None:
    if path.exists():
        _remove_intermediate_dir(path, work_dir)
    path.mkdir(parents=True, exist_ok=True)


def _remove_intermediate_dir(path: Path, work_dir: Path) -> None:
    resolved_path = path.resolve()
    resolved_work_dir = work_dir.resolve()
    try:
        resolved_path.relative_to(resolved_work_dir)
    except ValueError as exc:
        raise ValueError(f"Refusing to remove path outside work_dir: {resolved_path}") from exc
    if resolved_path == resolved_work_dir:
        raise ValueError(f"Refusing to remove work_dir itself: {resolved_path}")
    if resolved_path.is_dir():
        shutil.rmtree(resolved_path)
    elif resolved_path.exists():
        resolved_path.unlink()


def _remove_empty_work_dir(work_dir: Path) -> None:
    try:
        work_dir.rmdir()
    except OSError:
        pass


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_top_results(title: str, rows: list[dict[str, Any]], difference_key: str) -> None:
    print(f"[K sweep] {title}:")
    for index, row in enumerate(rows, start=1):
        print(f"  {index}. k={row['k_label']} {difference_key}={row[difference_key]}")


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