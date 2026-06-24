from __future__ import annotations

import json
import random
import re
import time
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from EAP_forComponent.components import ComponentRegistry
from EAP_forComponent.data import load_pair_dataset
from EAP_forComponent.future_localization import FutureLocalizationScorer, _load_future_state_dict
from EAP_forComponent.hooks import ComponentActivationCache
from EAP_forComponent.mask_builder import ComponentMaskBuilder
from EAP_forComponent.model_loader import ensure_src_on_path, load_model_and_tokenizer
from EAP_forComponent.outputs import save_outputs
from EAP_forComponent.rank_allocator import LoraRankAllocator
from EAP_forComponent.schemas import ComponentScore, ComponentTarget, EAPComponentConfig, PairBatch
from EAP_forComponent.scorer import ComponentAttributionScorer, normalize_localization_mode


class EAPForComponentRunner:
    def __init__(self, config: EAPComponentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.registry: ComponentRegistry | None = None

    def run(self) -> dict[str, Path]:
        started = time.time()
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(
            model_name_or_path=self.config.model_name_or_path,
            tokenizer_name_or_path=self.config.tokenizer_name_or_path,
            cache_dir=self.config.cache_dir,
            device=self.config.device,
            use_bfloat16=self.config.use_bfloat16,
            use_cpu=self.config.use_cpu,
        )
        dataset, collator = load_pair_dataset(
            dataset_name=self.config.dataset_name,
            tokenizer=self.tokenizer,
            data_path=self.config.data_path,
            corruption_column=self.config.corruption_column,
            max_samples=self.config.max_samples,
            max_length=self.config.max_length,
            input_format=self.config.input_format,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        self.registry = ComponentRegistry.from_model(
            model=self.model,
            target_modules=self.config.target_modules,
            attention_granularity=self.config.attention_granularity,
        )
        scores = self.run_attribution(dataloader)
        allocator = LoraRankAllocator(
            min_rank=self.config.min_rank,
            max_rank=self.config.max_rank,
            rank_multiple=self.config.rank_multiple,
            rank_budget=self.config.rank_budget,
            head_to_matrix_aggregation=self.config.head_to_matrix_aggregation,
            rank_score_source=self.config.rank_score_source,
        )
        ranks = allocator.allocate(scores)
        rank_pattern = allocator.to_rank_pattern(ranks)
        mask_builder = ComponentMaskBuilder(
            mask_fill_strategy=self.config.mask_fill_strategy,
            seed=self.config.mask_seed,
            include_all_parameters=self.config.mask_all_parameters,
            min_keep_ratio=self.config.mask_min_keep_ratio,
            max_keep_ratio=self.config.mask_max_keep_ratio,
        )
        component_mask = mask_builder.build(self.model, scores)
        elapsed = time.time() - started
        summary = {
            "model_name_or_path": self.config.model_name_or_path,
            "dataset_name": self.config.dataset_name,
            "data_path": str(self.config.data_path) if self.config.data_path else None,
            "metric": self.config.metric,
            "attention_granularity": self.config.attention_granularity,
            "localization_mode": normalize_localization_mode(self.config.localization_mode),
            "future_model_name_or_path": self.config.future_model_name_or_path,
            "future_step_k": self.config.future_step_k,
            "future_step_k_min": self.config.future_step_k_min,
            "future_step_k_max": self.config.future_step_k_max,
            "future_step_k_samples": self.config.future_step_k_samples,
            "future_step_k_seed": self.config.future_step_k_seed,
            "future_hvp_strategy": self.config.future_hvp_strategy,
            "future_delta_parameter_filter": self.config.future_delta_parameter_filter,
            "delta_activation_convention": "clean_minus_corrupted",
            "gradient_source": "corrupted_activation",
            "score_token_mode": self.config.score_token_mode,
            "score_normalization": self.config.score_normalization,
            "rank_score_source": self.config.rank_score_source,
            "target_modules": list(self.config.target_modules),
            "max_samples": self.config.max_samples,
            "batch_size": self.config.batch_size,
            "registry": self.registry.metadata if self.registry else {},
            "lora_allocation": allocator.last_allocation,
            "mask_builder": mask_builder.last_summary,
            "elapsed_seconds": elapsed,
        }
        if hasattr(self, "_future_localization_summary"):
            summary["future_localization"] = self._future_localization_summary
        summary.update(self.config.metadata)
        paths = save_outputs(
            output_dir=self.config.output_dir,
            scores=scores,
            rank_pattern=rank_pattern,
            lora_allocation=allocator.last_allocation,
            component_mask=component_mask,
            summary=summary,
        )
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return paths

    def run_attribution(self, dataloader: DataLoader) -> list:
        assert self.model is not None
        assert self.registry is not None
        assert self.device is not None
        targets = self.registry.targets()
        localization_mode = normalize_localization_mode(self.config.localization_mode)
        if localization_mode == "future":
            k_values = _future_step_k_values(self.config)
            future_state_dict = (
                _load_sampled_future_state_dict(model=self.model, targets=targets, config=self.config)
                if len(k_values) > 1
                else None
            )
            score_runs: list[list[ComponentScore]] = []
            per_k_summaries = []
            for sample_idx, future_step_k in enumerate(k_values, start=1):
                print(
                    f"[Future K sampling] Running sample {sample_idx}/{len(k_values)} "
                    f"with future_step_k={future_step_k:.4f}"
                )
                sample_config = replace(self.config, future_step_k=future_step_k)
                scorer = FutureLocalizationScorer(
                    model=self.model,
                    targets=targets,
                    config=sample_config,
                    device=self.device,
                    future_state_dict=future_state_dict,
                )
                scores = scorer.score(dataloader)
                score_runs.append(scores)
                per_k_summaries.append(scorer.last_summary)
            if len(score_runs) == 1:
                self._future_localization_summary = per_k_summaries[0]
                return score_runs[0]
            aggregated_scores = _aggregate_future_k_scores(score_runs=score_runs, k_values=k_values)
            self._future_localization_summary = {
                "aggregation": "mean_scores_and_mean_abs_raw_ranks",
                "future_step_k_values": k_values,
                "future_step_k_min": self.config.future_step_k_min,
                "future_step_k_max": self.config.future_step_k_max,
                "future_step_k_samples": len(k_values),
                "future_step_k_seed": self.config.future_step_k_seed,
                "score_definition": "mean of per-K attribution scores",
                "rank_definition": "mean rank from per-K abs(raw_score) rankings; lower mean_raw_rank is better",
                "rank_score_definition": "component_count + 1 - mean_raw_rank",
                "per_k": per_k_summaries,
            }
            return aggregated_scores
        scorer = ComponentAttributionScorer(
            targets=targets,
            score_token_mode=self.config.score_token_mode,
            score_normalization=self.config.score_normalization,
            localization_mode=localization_mode,
        )
        cache = ComponentActivationCache(targets, capture_device=self.config.capture_device)
        cache.register()
        try:
            for batch in tqdm(dataloader, desc="EAP_forComponent attribution"):
                batch = batch.to(self.device)
                cache.clear_batch()
                with torch.no_grad(), cache.capture("clean"):
                    ensure_src_on_path()
                    from modeling_patches import sequential_position_ids

                    self.model(
                        input_ids=batch.clean_input_ids,
                        attention_mask=batch.clean_attention_mask,
                        position_ids=sequential_position_ids(batch.clean_input_ids),
                        use_cache=False,
                    )
                self.model.zero_grad(set_to_none=True)
                with cache.capture("corrupted"):
                    loss = self._compute_loss(batch, input_kind="corrupted")
                loss.backward()
                scorer.score_batch(
                    clean_outputs=cache.clean_outputs,
                    corrupted_outputs=cache.corrupted_outputs,
                    output_grads=cache.output_grads,
                    attention_mask=batch.corrupted_attention_mask.detach().cpu(),
                    label_positions=self._label_positions(batch, "corrupted").detach().cpu(),
                    clean_inputs=cache.clean_inputs,
                    corrupted_inputs=cache.corrupted_inputs,
                )
                self.model.zero_grad(set_to_none=True)
                cache.clear_batch()
        finally:
            cache.remove()
        return scorer.finalize()

    def _compute_loss(self, batch: PairBatch, input_kind: str = "clean") -> torch.Tensor:
        if self.config.metric == "task_loss":
            ensure_src_on_path()
            from training_losses import task_loss

            input_ids, attention_mask, labels = self._loss_inputs(batch, input_kind)
            loss, _outputs = task_loss(
                self.model,
                (input_ids, attention_mask, labels),
            )
            return loss
        if self.config.metric == "logit_diff":
            return self._logit_diff_loss(batch, input_kind=input_kind)
        raise ValueError(f"Unsupported metric: {self.config.metric}")

    def _loss_inputs(self, batch: PairBatch, input_kind: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_kind == "clean":
            return batch.clean_input_ids, batch.clean_attention_mask, batch.labels
        if input_kind != "corrupted":
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        labels = torch.full_like(batch.corrupted_input_ids, -100)
        positions = self._label_positions(batch, "corrupted")
        rows = torch.arange(batch.corrupted_input_ids.size(0), device=batch.corrupted_input_ids.device)
        labels[rows, positions] = batch.correct_idx.to(labels.device)
        return batch.corrupted_input_ids, batch.corrupted_attention_mask, labels

    def _label_positions(self, batch: PairBatch, input_kind: str) -> torch.Tensor:
        if input_kind == "clean":
            return batch.label_positions
        if input_kind != "corrupted":
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        return batch.corrupted_attention_mask.long().sum(dim=1).sub(1).clamp_min(0)

    def _logit_diff_loss(self, batch: PairBatch, input_kind: str = "clean") -> torch.Tensor:
        ensure_src_on_path()
        from modeling_patches import sequential_position_ids

        if input_kind == "clean":
            input_ids = batch.clean_input_ids
            attention_mask = batch.clean_attention_mask
            positions = batch.label_positions.to(input_ids.device)
        elif input_kind == "corrupted":
            input_ids = batch.corrupted_input_ids
            attention_mask = batch.corrupted_attention_mask
            positions = self._label_positions(batch, "corrupted").to(input_ids.device)
        else:
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=sequential_position_ids(input_ids),
            use_cache=False,
        )
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        logits = outputs.logits[rows, positions, :].float()
        vocab_size = logits.size(-1)
        valid = (
            (batch.correct_idx >= 0)
            & (batch.correct_idx < vocab_size)
            & (batch.incorrect_idx >= 0)
            & (batch.incorrect_idx < vocab_size)
        )
        if not valid.any():
            return logits.sum() * 0.0
        valid_logits = logits[valid]
        correct = batch.correct_idx[valid].to(logits.device)
        incorrect = batch.incorrect_idx[valid].to(logits.device)
        logit_diff = valid_logits.gather(1, correct[:, None]).squeeze(1) - valid_logits.gather(
            1, incorrect[:, None]
        ).squeeze(1)
        return -logit_diff.mean()


def _future_step_k_values(config: EAPComponentConfig) -> list[float]:
    sample_count = int(config.future_step_k_samples)
    if sample_count < 1:
        raise ValueError("future_step_k_samples must be >= 1")
    if sample_count == 1:
        return [round(float(config.future_step_k), 4)]
    min_units = _decimal_units(config.future_step_k_min, ROUND_CEILING)
    max_units = _decimal_units(config.future_step_k_max, ROUND_FLOOR)
    if min_units > max_units:
        raise ValueError("future_step_k_min must be <= future_step_k_max after 4-decimal rounding")
    capacity = max_units - min_units + 1
    if sample_count > capacity:
        raise ValueError(
            f"Cannot sample {sample_count} unique K values from the 4-decimal grid in "
            f"[{config.future_step_k_min}, {config.future_step_k_max}] with capacity {capacity}."
        )
    rng = random.Random(int(config.future_step_k_seed))
    sampled_units = sorted(rng.sample(range(min_units, max_units + 1), sample_count))
    return [unit / 10000.0 for unit in sampled_units]


def _load_sampled_future_state_dict(
    model: torch.nn.Module,
    targets: list[ComponentTarget],
    config: EAPComponentConfig,
) -> dict[str, torch.Tensor]:
    if not config.future_model_name_or_path:
        raise ValueError("future_model_name_or_path is required when localization_mode='future'")
    parameter_names = _future_delta_parameter_names(model=model, targets=targets, config=config)
    return _load_future_state_dict(
        model_name_or_path=str(config.future_model_name_or_path),
        cache_dir=config.future_model_cache_dir or config.cache_dir,
        parameter_names=parameter_names,
    )


def _future_delta_parameter_names(
    model: torch.nn.Module,
    targets: list[ComponentTarget],
    config: EAPComponentConfig,
) -> list[str]:
    base_parameters = dict(model.named_parameters())
    if config.future_delta_parameter_filter:
        try:
            pattern = re.compile(config.future_delta_parameter_filter)
        except re.error as error:
            raise ValueError(f"Invalid future_delta_parameter_filter regex: {error}") from error
        return [name for name in base_parameters if pattern.search(name)]
    target_names = sorted({target.parameter_name for target in targets})
    return [name for name in target_names if name in base_parameters]


def _decimal_units(value: float, rounding) -> int:
    return int((Decimal(str(value)) * Decimal("10000")).to_integral_value(rounding=rounding))


def _aggregate_future_k_scores(
    score_runs: list[list[ComponentScore]],
    k_values: list[float],
) -> list[ComponentScore]:
    if not score_runs:
        return []
    if len(score_runs) != len(k_values):
        raise ValueError(f"Expected {len(k_values)} score runs, got {len(score_runs)}")
    first_names = [score.component_name for score in score_runs[0]]
    first_name_set = set(first_names)
    score_maps = []
    for run_idx, scores in enumerate(score_runs, start=1):
        score_map = {score.component_name: score for score in scores}
        if set(score_map) != first_name_set:
            missing = sorted(first_name_set - set(score_map))[:10]
            extra = sorted(set(score_map) - first_name_set)[:10]
            raise ValueError(
                f"Future K sample {run_idx} component set mismatch: missing={missing}, extra={extra}"
            )
        score_maps.append(score_map)
    rank_maps = [_abs_raw_rank_map(scores) for scores in score_runs]
    component_count = len(first_names)
    aggregated_scores = []
    for base_score in score_runs[0]:
        component_name = base_score.component_name
        component_scores = [score_map[component_name] for score_map in score_maps]
        mean_raw_rank = mean(rank_map[component_name] for rank_map in rank_maps)
        aggregated_scores.append(
            replace(
                base_score,
                raw_score=_mean_score_field(component_scores, "raw_score"),
                abs_score=_mean_score_field(component_scores, "abs_score"),
                mean_score=_mean_score_field(component_scores, "mean_score"),
                sqrt_numel_score=_mean_score_field(component_scores, "sqrt_numel_score"),
                rank_score=float(component_count + 1 - mean_raw_rank),
                current_score=_mean_optional_score_field(component_scores, "current_score"),
                future_directional_score_theta=_mean_optional_score_field(
                    component_scores, "future_directional_score_theta"
                ),
                future_directional_score_theta_hat=_mean_optional_score_field(
                    component_scores, "future_directional_score_theta_hat"
                ),
                future_correction=_mean_optional_score_field(component_scores, "future_correction"),
                future_step_k=None,
                mean_raw_rank=float(mean_raw_rank),
            )
        )
    return aggregated_scores


def _abs_raw_rank_map(scores: list[ComponentScore]) -> dict[str, int]:
    ranked = sorted(scores, key=lambda score: (-abs(float(score.raw_score)), score.component_name))
    return {score.component_name: rank for rank, score in enumerate(ranked, start=1)}


def _mean_score_field(scores: list[ComponentScore], field_name: str) -> float:
    return float(mean(float(getattr(score, field_name)) for score in scores))


def _mean_optional_score_field(scores: list[ComponentScore], field_name: str) -> float | None:
    values = [getattr(score, field_name) for score in scores]
    if any(value is None for value in values):
        return None
    return float(mean(float(value) for value in values))
