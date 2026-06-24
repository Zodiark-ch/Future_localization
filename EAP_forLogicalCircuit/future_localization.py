from __future__ import annotations

import gc
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from EAP_forComponent.future_localization import _load_future_state_dict
from EAP_forComponent.schemas import PairBatch
from EAP_forLogicalCircuit.current_localization import destination_gradient, select_token_rows
from EAP_forLogicalCircuit.edge_hooks import DestinationInputCache
from EAP_forLogicalCircuit.graph_registry import EdgeTarget
from EAP_forLogicalCircuit.model_loader import ensure_src_on_path
from EAP_forLogicalCircuit.schemas import EAPLogicalCircuitConfig, EdgeScore


@dataclass
class _DirectionResult:
    raw_sums: dict[str, float]
    direction_sums: dict[str, float]
    token_counts: dict[str, int]
    element_counts: dict[str, int]
    sample_count: int


def future_trapezoid_raw_score(current_score: float, direction_theta: float, direction_theta_hat: float) -> float:
    return float(current_score) + 0.5 * (float(direction_theta) + float(direction_theta_hat))


class FutureEdgeLocalizationScorer:
    def __init__(
        self,
        model: nn.Module,
        edge_targets: list[EdgeTarget],
        config: EAPLogicalCircuitConfig,
        device: torch.device | str,
        future_state_dict: dict[str, torch.Tensor] | None = None,
    ):
        if str(config.localization_mode).lower() != "future":
            raise ValueError("FutureEdgeLocalizationScorer requires localization_mode='future'")
        if config.future_hvp_strategy not in {"hvp", "finite_difference"}:
            raise ValueError("future_hvp_strategy must be 'hvp' or 'finite_difference'")
        if future_state_dict is None and not config.future_model_name_or_path:
            raise ValueError("future_model_name_or_path is required when localization_mode='future'")
        self.model = model
        self.edge_targets = edge_targets
        self.config = config
        self.device = torch.device(device)
        self.future_state_dict = future_state_dict
        self.last_summary: dict = {}
        self.timings: dict[str, float] = {}

    def score(self, dataloader) -> list[EdgeScore]:
        with self._timed("load_delta_tensors"):
            delta_tensors = self._load_delta_tensors()
        if self.config.future_hvp_strategy == "finite_difference":
            with self._timed("score_current_theta"):
                current = self._score_values(dataloader)
            with self._timed("score_theta_epsilon"):
                with _temporary_parameter_delta(
                    self.model,
                    delta_tensors,
                    self.config.future_finite_difference_epsilon * self.config.future_step_k,
                ):
                    shifted_theta = self._score_values(dataloader)
            direction_theta = self._finite_difference_from_values(current, shifted_theta)
            with _temporary_parameter_delta(self.model, delta_tensors, self.config.future_step_k):
                with self._timed("score_theta_hat"):
                    theta_hat = self._score_values(dataloader)
                with self._timed("score_theta_hat_epsilon"):
                    with _temporary_parameter_delta(
                        self.model,
                        delta_tensors,
                        self.config.future_finite_difference_epsilon * self.config.future_step_k,
                    ):
                        shifted_theta_hat = self._score_values(dataloader)
                direction_theta_hat = self._finite_difference_from_values(theta_hat, shifted_theta_hat)
        else:
            with self._timed("hvp_theta"):
                current = self._autograd_direction(dataloader, delta_tensors)
            direction_theta = current.direction_sums
            with _temporary_parameter_delta(self.model, delta_tensors, self.config.future_step_k):
                with self._timed("hvp_theta_hat"):
                    direction_theta_hat = self._autograd_direction(dataloader, delta_tensors).direction_sums
        self.last_summary = {
            "future_hvp_strategy": self.config.future_hvp_strategy,
            "future_step_k": self.config.future_step_k,
            "future_finite_difference_epsilon": self.config.future_finite_difference_epsilon,
            "delta_parameter_filter": self.config.future_delta_parameter_filter,
            "delta_parameter_count": len(delta_tensors),
            "timings_seconds": {key: round(value, 6) for key, value in sorted(self.timings.items())},
        }
        return self._finalize(current, direction_theta, direction_theta_hat)

    def _autograd_direction(self, dataloader, delta_tensors: dict[str, torch.Tensor]) -> _DirectionResult:
        raw_sums = self._zero_float_dict()
        direction_sums = self._zero_float_dict()
        token_counts = self._zero_int_dict()
        element_counts = self._zero_int_dict()
        sample_count = 0
        delta_items = self._delta_parameter_items(delta_tensors)
        delta_names = [name for name, _ in delta_items]
        delta_parameters = [parameter for _, parameter in delta_items]
        for batch in tqdm(dataloader, desc="EAP_forLogicalCircuit future HVP"):
            batch = batch.to(self.device)
            sample_count += int(batch.corrupted_input_ids.size(0))
            self.model.zero_grad(set_to_none=True)
            with _math_sdp_kernel_context():
                batch_scores = self._edge_score_tensors(batch, create_graph=True)
                for target in self.edge_targets:
                    score_info = batch_scores.get(target.edge_id)
                    if score_info is None:
                        continue
                    score_tensor, token_count, element_count = score_info
                    raw_sums[target.edge_id] += float(score_tensor.detach().float().item())
                    token_counts[target.edge_id] += int(token_count)
                    element_counts[target.edge_id] += int(element_count)
                    if not score_tensor.requires_grad:
                        continue
                    grads = torch.autograd.grad(
                        score_tensor,
                        delta_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    direction_sums[target.edge_id] += _dot_grads_with_delta(
                        grads=grads,
                        parameter_names=delta_names,
                        delta_tensors=delta_tensors,
                        scale=self.config.future_step_k,
                    )
            self.model.zero_grad(set_to_none=True)
            del batch_scores
            gc.collect()
        return _DirectionResult(
            raw_sums=raw_sums,
            direction_sums=direction_sums,
            token_counts=token_counts,
            element_counts=element_counts,
            sample_count=sample_count,
        )

    def _score_values(self, dataloader) -> _DirectionResult:
        raw_sums = self._zero_float_dict()
        token_counts = self._zero_int_dict()
        element_counts = self._zero_int_dict()
        sample_count = 0
        for batch in tqdm(dataloader, desc="EAP_forLogicalCircuit current values"):
            batch = batch.to(self.device)
            sample_count += int(batch.corrupted_input_ids.size(0))
            self.model.zero_grad(set_to_none=True)
            batch_scores = self._edge_score_tensors(batch, create_graph=False)
            for target in self.edge_targets:
                score_info = batch_scores.get(target.edge_id)
                if score_info is None:
                    continue
                score_tensor, token_count, element_count = score_info
                raw_sums[target.edge_id] += float(score_tensor.detach().float().item())
                token_counts[target.edge_id] += int(token_count)
                element_counts[target.edge_id] += int(element_count)
            self.model.zero_grad(set_to_none=True)
            del batch_scores
            gc.collect()
        return _DirectionResult(
            raw_sums=raw_sums,
            direction_sums=self._zero_float_dict(),
            token_counts=token_counts,
            element_counts=element_counts,
            sample_count=sample_count,
        )

    def _finite_difference_from_values(self, base: _DirectionResult, shifted: _DirectionResult) -> dict[str, float]:
        epsilon = float(self.config.future_finite_difference_epsilon)
        if epsilon <= 0:
            raise ValueError("future_finite_difference_epsilon must be positive")
        return {
            target.edge_id: (shifted.raw_sums[target.edge_id] - base.raw_sums[target.edge_id]) / epsilon
            for target in self.edge_targets
        }

    def _edge_score_tensors(self, batch: PairBatch, create_graph: bool) -> dict[str, tuple[torch.Tensor, int, int]]:
        scoring_on_cpu = not create_graph
        cache = DestinationInputCache(
            edge_targets=self.edge_targets,
            capture_device=self.device,
            detach_tensors=False,
            source_capture_device=self.config.capture_device if scoring_on_cpu else self.device,
            detach_source_tensors=scoring_on_cpu,
            output_grad_capture_device=self.config.capture_device if scoring_on_cpu else self.device,
            detach_output_grads=scoring_on_cpu,
        )
        cache.register()
        try:
            with self._timed("clean_forward"):
                with torch.no_grad(), cache.capture("clean"):
                    _forward_model(self.model, batch.clean_input_ids, batch.clean_attention_mask)
            with self._timed("corrupted_loss_forward"):
                with cache.capture("corrupted"):
                    loss = _compute_loss(self.model, batch, self.config.metric, input_kind="corrupted")

            corrupted_items = [
                (param_name, tensor)
                for param_name, tensor in cache.corrupted_inputs.items()
                if tensor.requires_grad
            ]
            input_grads: dict[str, torch.Tensor] = {}
            if corrupted_items:
                with self._timed("destination_input_grads"):
                    grads = torch.autograd.grad(
                        loss,
                        [tensor for _, tensor in corrupted_items],
                        retain_graph=create_graph,
                        create_graph=create_graph,
                        allow_unused=True,
                    )
                input_grads = {
                    param_name: grad
                    for (param_name, _), grad in zip(corrupted_items, grads, strict=True)
                    if grad is not None
                }
            if not create_graph:
                input_grads = {
                    name: grad.detach().to(self.config.capture_device)
                    for name, grad in input_grads.items()
                }

            attention_mask = batch.corrupted_attention_mask.detach().cpu() if scoring_on_cpu else batch.corrupted_attention_mask
            label_positions = _label_positions(batch, "corrupted")
            if scoring_on_cpu:
                label_positions = label_positions.detach().cpu()
            scores: dict[str, tuple[torch.Tensor, int, int]] = {}
            with self._timed("edge_score_tensors"):
                for target in self.edge_targets:
                    clean_input = cache.clean_source_outputs.get(target.source_node)
                    corrupted_input = cache.corrupted_source_outputs.get(target.source_node)
                    if clean_input is None:
                        clean_input = _first_available(cache.clean_inputs, _destination_input_keys(target))
                    if corrupted_input is None:
                        corrupted_input = _first_available(cache.corrupted_inputs, _destination_input_keys(target))
                    grad_input = destination_gradient(
                        target,
                        input_grads=input_grads,
                        output_grads=cache.output_grads,
                    )
                    if clean_input is None or corrupted_input is None or grad_input is None:
                        continue
                    selected_clean, selected_corrupted, selected_grad = select_token_rows(
                        clean_activation=clean_input,
                        corrupted_activation=corrupted_input,
                        output_grad=grad_input,
                        attention_mask=attention_mask,
                        label_positions=label_positions,
                        token_mode=self.config.score_token_mode,
                    )
                    if selected_clean.numel() == 0:
                        scores[target.edge_id] = (clean_input.sum() * 0.0, 0, 0)
                        continue
                    score = ((selected_clean.float() - selected_corrupted.float()) * selected_grad.float()).sum()
                    scores[target.edge_id] = (
                        score,
                        int(selected_clean.size(0)),
                        int(selected_clean.numel()),
                    )
            return scores
        finally:
            cache.remove()

    def _finalize(
        self,
        current: _DirectionResult,
        direction_theta: dict[str, float],
        direction_theta_hat: dict[str, float],
    ) -> list[EdgeScore]:
        denominator = max(1, current.sample_count)
        scores: list[EdgeScore] = []
        for target in self.edge_targets:
            edge_id = target.edge_id
            current_score = current.raw_sums[edge_id] / denominator
            theta_score = direction_theta[edge_id] / denominator
            theta_hat_score = direction_theta_hat[edge_id] / denominator
            correction = 0.5 * (theta_score + theta_hat_score)
            raw_score = future_trapezoid_raw_score(current_score, theta_score, theta_hat_score)
            raw_sum = current.raw_sums[edge_id] + 0.5 * (
                direction_theta[edge_id] + direction_theta_hat[edge_id]
            )
            element_count = max(1, current.element_counts[edge_id])
            mean_score = raw_sum / element_count
            sqrt_numel_score = raw_score / max(1.0, math.sqrt(float(target.num_features)))
            if self.config.score_normalization == "sum":
                rank_score = abs(raw_score)
            elif self.config.score_normalization == "mean":
                rank_score = abs(mean_score)
            elif self.config.score_normalization == "sqrt_numel":
                rank_score = abs(sqrt_numel_score)
            else:
                raise ValueError(f"Unsupported score_normalization: {self.config.score_normalization}")
            scores.append(
                EdgeScore(
                    edge_id=edge_id,
                    source_node=target.source_node,
                    destination_module=target.destination_module,
                    layer_idx=target.layer_idx,
                    component_type=target.component_type,
                    score_token_mode=self.config.score_token_mode,
                    localization_mode="future",
                    raw_score=float(raw_score),
                    abs_score=float(abs(raw_score)),
                    mean_score=float(mean_score),
                    sqrt_numel_score=float(sqrt_numel_score),
                    rank_score=float(rank_score),
                    token_count=int(current.token_counts[edge_id]),
                    element_count=int(current.element_counts[edge_id]),
                    numel=int(target.num_features),
                    current_score=float(current_score),
                    future_directional_score_theta=float(theta_score),
                    future_directional_score_theta_hat=float(theta_hat_score),
                    future_correction=float(correction),
                    future_step_k=float(self.config.future_step_k),
                )
            )
        return scores

    def _load_delta_tensors(self) -> dict[str, torch.Tensor]:
        base_parameters = dict(self.model.named_parameters())
        parameter_names = self._delta_parameter_names(base_parameters)
        owns_future_state = self.future_state_dict is None
        future_state = self.future_state_dict or _load_future_state_dict(
            model_name_or_path=str(self.config.future_model_name_or_path),
            cache_dir=self.config.future_model_cache_dir or self.config.cache_dir,
            parameter_names=parameter_names,
        )
        deltas: dict[str, torch.Tensor] = {}
        for name in parameter_names:
            if name not in future_state:
                raise ValueError(f"Future model is missing parameter {name!r}. Pass a merged full checkpoint.")
            base_parameter = base_parameters[name]
            future_parameter = future_state.pop(name) if owns_future_state else future_state[name]
            if tuple(future_parameter.shape) != tuple(base_parameter.shape):
                raise ValueError(
                    f"Future parameter {name!r} shape {tuple(future_parameter.shape)} does not match "
                    f"base shape {tuple(base_parameter.shape)}."
                )
            if not base_parameter.requires_grad:
                base_parameter.requires_grad_(True)
            base_cpu = base_parameter.detach().cpu()
            delta = future_parameter.detach().cpu().float() - base_cpu.float()
            if self.config.future_hvp_strategy == "finite_difference":
                delta = delta.to(dtype=base_cpu.dtype)
            deltas[name] = delta
            del future_parameter, base_cpu, delta
        if owns_future_state:
            future_state.clear()
        if not deltas:
            raise ValueError("No parameters matched future_delta_parameter_filter.")
        return deltas

    def _delta_parameter_names(self, base_parameters: dict[str, nn.Parameter]) -> list[str]:
        if self.config.future_delta_parameter_filter:
            try:
                pattern = re.compile(self.config.future_delta_parameter_filter)
            except re.error as error:
                raise ValueError(f"Invalid future_delta_parameter_filter regex: {error}") from error
            return [name for name in base_parameters if pattern.search(name)]
        edge_parameter_names = sorted(
            {
                parameter_name
                for target in self.edge_targets
                for parameter_name in target.delta_parameter_names
            }
        )
        return [name for name in edge_parameter_names if name in base_parameters]

    def _delta_parameter_items(self, delta_tensors: dict[str, torch.Tensor]) -> list[tuple[str, nn.Parameter]]:
        base_parameters = dict(self.model.named_parameters())
        return [(name, base_parameters[name]) for name in delta_tensors]

    def _zero_float_dict(self) -> dict[str, float]:
        return {target.edge_id: 0.0 for target in self.edge_targets}

    def _zero_int_dict(self) -> dict[str, int]:
        return {target.edge_id: 0 for target in self.edge_targets}

    @contextmanager
    def _timed(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - started)


@contextmanager
def _math_sdp_kernel_context():
    if not torch.cuda.is_available():
        yield
        return
    attention = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attention, "sdpa_kernel", None) if attention is not None else None
    sdp_backend = getattr(attention, "SDPBackend", None) if attention is not None else None
    if sdpa_kernel is not None and sdp_backend is not None:
        with sdpa_kernel(sdp_backend.MATH, set_priority=True):
            yield
        return
    legacy_sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
    if legacy_sdp_kernel is not None:
        with legacy_sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            yield
        return
    previous_flash = torch.backends.cuda.flash_sdp_enabled()
    previous_math = torch.backends.cuda.math_sdp_enabled()
    previous_mem_efficient = torch.backends.cuda.mem_efficient_sdp_enabled()
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(previous_flash)
        torch.backends.cuda.enable_math_sdp(previous_math)
        torch.backends.cuda.enable_mem_efficient_sdp(previous_mem_efficient)


@contextmanager
def _temporary_parameter_delta(model: nn.Module, delta_tensors: dict[str, torch.Tensor], scale: float):
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, delta in delta_tensors.items():
            parameter = parameters[name]
            parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype), alpha=float(scale))
    try:
        yield
    finally:
        with torch.no_grad():
            for name, delta in delta_tensors.items():
                parameter = parameters[name]
                parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype), alpha=-float(scale))


def _dot_grads_with_delta(
    grads: tuple[torch.Tensor | None, ...],
    parameter_names: list[str],
    delta_tensors: dict[str, torch.Tensor],
    scale: float,
) -> float:
    total = 0.0
    for grad, name in zip(grads, parameter_names, strict=True):
        if grad is None:
            continue
        delta = delta_tensors[name].to(device=grad.device, dtype=torch.float32)
        total += float((grad.float() * delta).sum().detach().cpu().item()) * float(scale)
    return total


def _compute_loss(model: nn.Module, batch: PairBatch, metric: str, input_kind: str) -> torch.Tensor:
    if metric == "task_loss":
        ensure_src_on_path()
        from training_losses import task_loss

        input_ids, attention_mask, labels = _loss_inputs(batch, input_kind)
        loss, _outputs = task_loss(model, (input_ids, attention_mask, labels))
        return loss
    if metric == "logit_diff":
        return _logit_diff_loss(model, batch, input_kind)
    raise ValueError(f"Unsupported metric: {metric}")


def _loss_inputs(batch: PairBatch, input_kind: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_kind == "clean":
        return batch.clean_input_ids, batch.clean_attention_mask, batch.labels
    if input_kind != "corrupted":
        raise ValueError(f"Unsupported input_kind: {input_kind}")
    labels = torch.full_like(batch.corrupted_input_ids, -100)
    positions = _label_positions(batch, "corrupted")
    rows = torch.arange(batch.corrupted_input_ids.size(0), device=batch.corrupted_input_ids.device)
    labels[rows, positions] = batch.correct_idx.to(labels.device)
    return batch.corrupted_input_ids, batch.corrupted_attention_mask, labels


def _label_positions(batch: PairBatch, input_kind: str) -> torch.Tensor:
    if input_kind == "clean":
        return batch.label_positions
    if input_kind != "corrupted":
        raise ValueError(f"Unsupported input_kind: {input_kind}")
    return batch.corrupted_attention_mask.long().sum(dim=1).sub(1).clamp_min(0)


def _logit_diff_loss(model: nn.Module, batch: PairBatch, input_kind: str) -> torch.Tensor:
    if input_kind == "clean":
        input_ids = batch.clean_input_ids
        attention_mask = batch.clean_attention_mask
        positions = batch.label_positions.to(input_ids.device)
    elif input_kind == "corrupted":
        input_ids = batch.corrupted_input_ids
        attention_mask = batch.corrupted_attention_mask
        positions = _label_positions(batch, "corrupted").to(input_ids.device)
    else:
        raise ValueError(f"Unsupported input_kind: {input_kind}")
    outputs = _forward_model(model, input_ids, attention_mask)
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


def _forward_model(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    ensure_src_on_path()
    from modeling_patches import sequential_position_ids

    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=sequential_position_ids(input_ids),
        use_cache=False,
    )


def _first_available(values: dict[str, torch.Tensor], keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return value
    return None


def _destination_input_keys(target: EdgeTarget) -> tuple[str, ...]:
    keys = getattr(target, "destination_input_keys", tuple())
    return keys or (target.destination_parameter_name,)
