from __future__ import annotations

import gc
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from EAP_forComponent.model_loader import ensure_src_on_path
from EAP_forComponent.schemas import ComponentScore, ComponentTarget, EAPComponentConfig, PairBatch
from EAP_forComponent.scorer import normalize_localization_mode, select_token_rows


@dataclass
class _DirectionResult:
    raw_sums: dict[str, float]
    direction_sums: dict[str, float]
    token_counts: dict[str, int]
    element_counts: dict[str, int]
    sample_count: int


class FutureLocalizationScorer:
    def __init__(
        self,
        model: nn.Module,
        targets: list[ComponentTarget],
        config: EAPComponentConfig,
        device: torch.device | str,
        future_state_dict: dict[str, torch.Tensor] | None = None,
    ):
        if normalize_localization_mode(config.localization_mode) != "future":
            raise ValueError("FutureLocalizationScorer requires localization_mode='future'")
        if config.future_hvp_strategy not in {"hvp", "finite_difference"}:
            raise ValueError("future_hvp_strategy must be 'hvp' or 'finite_difference'")
        if future_state_dict is None and not config.future_model_name_or_path:
            raise ValueError("future_model_name_or_path is required when localization_mode='future'")
        self.model = model
        self.targets = targets
        self.config = config
        self.device = torch.device(device)
        self.future_state_dict = future_state_dict
        self.last_summary: dict = {}
        self.timings: dict[str, float] = {}
        self.counters: dict[str, int] = {
            "component_count": len(targets),
            "score_value_passes": 0,
            "score_value_batches": 0,
            "activation_grad_calls": 0,
            "activation_grad_tensor_count": 0,
            "component_direction_grad_calls": 0,
        }
        self.delta_storage_dtype: str | None = None

    def score(self, dataloader: Iterable[PairBatch]) -> list[ComponentScore]:
        with self._timed("load_delta_tensors"):
            delta_tensors = self._load_delta_tensors()
        self.counters["delta_parameter_count"] = len(delta_tensors)
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
            "delta_parameter_count": len(delta_tensors),
            "future_hvp_strategy": self.config.future_hvp_strategy,
            "future_attention_kernel": "math_sdp" if self.config.future_hvp_strategy == "hvp" else "default",
            "future_step_k": self.config.future_step_k,
            "future_finite_difference_epsilon": self.config.future_finite_difference_epsilon,
            "delta_parameter_filter": self.config.future_delta_parameter_filter,
            "delta_storage_dtype": self.delta_storage_dtype,
            "timings_seconds": {key: round(value, 6) for key, value in sorted(self.timings.items())},
            "counters": dict(sorted(self.counters.items())),
        }
        return self._finalize(
            current=current,
            direction_theta=direction_theta,
            direction_theta_hat=direction_theta_hat,
        )

    def _autograd_direction(
        self,
        dataloader: Iterable[PairBatch],
        delta_tensors: dict[str, torch.Tensor],
    ) -> _DirectionResult:
        raw_sums = self._zero_float_dict()
        direction_sums = self._zero_float_dict()
        token_counts = self._zero_int_dict()
        element_counts = self._zero_int_dict()
        sample_count = 0
        delta_items = self._delta_parameter_items(delta_tensors)
        delta_names = [name for name, _parameter in delta_items]
        delta_parameters = [parameter for _name, parameter in delta_items]
        for batch in tqdm(dataloader, desc="EAP_forComponent future HVP"):
            batch = batch.to(self.device)
            sample_count += int(batch.corrupted_input_ids.size(0))
            self.model.zero_grad(set_to_none=True)
            with _math_sdp_kernel_context():
                batch_scores = self._component_score_tensors(batch, create_graph=True)
                for target in self.targets:
                    score_info = batch_scores.get(target.component_name)
                    if score_info is None:
                        continue
                    score_tensor, token_count, element_count = score_info
                    raw_sums[target.component_name] += float(score_tensor.detach().float().item())
                    token_counts[target.component_name] += int(token_count)
                    element_counts[target.component_name] += int(element_count)
                    if not score_tensor.requires_grad:
                        continue
                    self.counters["component_direction_grad_calls"] += 1
                    grads = torch.autograd.grad(
                        score_tensor,
                        delta_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    direction_sums[target.component_name] += _dot_grads_with_delta(
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

    def _score_values(self, dataloader: Iterable[PairBatch]) -> _DirectionResult:
        raw_sums = self._zero_float_dict()
        token_counts = self._zero_int_dict()
        element_counts = self._zero_int_dict()
        sample_count = 0
        self.counters["score_value_passes"] += 1
        for batch in tqdm(dataloader, desc="EAP_forComponent current values"):
            batch = batch.to(self.device)
            sample_count += int(batch.corrupted_input_ids.size(0))
            self.counters["score_value_batches"] += 1
            self.model.zero_grad(set_to_none=True)
            batch_scores = self._component_score_tensors(batch, create_graph=False)
            for target in self.targets:
                score_info = batch_scores.get(target.component_name)
                if score_info is None:
                    continue
                score_tensor, token_count, element_count = score_info
                raw_sums[target.component_name] += float(score_tensor.detach().float().item())
                token_counts[target.component_name] += int(token_count)
                element_counts[target.component_name] += int(element_count)
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

    def _finite_difference_from_values(
        self,
        base: _DirectionResult,
        shifted: _DirectionResult,
    ) -> dict[str, float]:
        epsilon = float(self.config.future_finite_difference_epsilon)
        if epsilon <= 0:
            raise ValueError("future_finite_difference_epsilon must be positive")
        return {
            target.component_name: (shifted.raw_sums[target.component_name] - base.raw_sums[target.component_name])
            / epsilon
            for target in self.targets
        }

    def _component_score_tensors(
        self,
        batch: PairBatch,
        create_graph: bool,
    ) -> dict[str, tuple[torch.Tensor, int, int]]:
        cache = _TensorActivationCache(self.targets)
        cache.register()
        try:
            with self._timed("clean_forward"):
                with cache.capture("clean"):
                    _forward_model(self.model, batch.clean_input_ids, batch.clean_attention_mask)
            with self._timed("corrupted_loss_forward"):
                with cache.capture("corrupted"):
                    loss = _compute_loss(self.model, batch, self.config.metric, input_kind="corrupted")
            output_grads: dict[str, torch.Tensor] = {}
            output_items = [
                (parameter_name, corrupted_output)
                for parameter_name, corrupted_output in cache.corrupted_outputs.items()
                if corrupted_output.requires_grad
            ]
            if output_items:
                self.counters["activation_grad_calls"] += 1
                self.counters["activation_grad_tensor_count"] += len(output_items)
                with self._timed("activation_grads"):
                    grads = torch.autograd.grad(
                        loss,
                        [corrupted_output for _parameter_name, corrupted_output in output_items],
                        retain_graph=True,
                        create_graph=create_graph,
                        allow_unused=True,
                    )
                output_grads = {
                    parameter_name: grad
                    for (parameter_name, _corrupted_output), grad in zip(output_items, grads, strict=True)
                    if grad is not None
                }
            if not create_graph:
                output_grads = {parameter_name: grad.detach() for parameter_name, grad in output_grads.items()}
            attention_mask = batch.corrupted_attention_mask
            label_positions = _label_positions(batch, "corrupted")
            scores: dict[str, tuple[torch.Tensor, int, int]] = {}
            with self._timed("component_score_tensors"):
                for target in self.targets:
                    output_grad = output_grads.get(target.parameter_name)
                    if output_grad is None:
                        continue
                    if target.component_type == "o_proj" and target.col_slice is not None:
                        clean_input = _maybe_detach(cache.clean_inputs.get(target.parameter_name), create_graph)
                        corrupted_input = _maybe_detach(cache.corrupted_inputs.get(target.parameter_name), create_graph)
                        if clean_input is None or corrupted_input is None:
                            continue
                        scores[target.component_name] = _score_o_proj_head_input_slice_tensor(
                            clean_input=clean_input,
                            corrupted_input=corrupted_input,
                            weight=target.module.weight if create_graph else target.module.weight.detach(),
                            output_grad=output_grad,
                            col_slice=target.col_slice,
                            attention_mask=attention_mask,
                            label_positions=label_positions,
                            token_mode=self.config.score_token_mode,
                        )
                        continue
                    clean_output = _maybe_detach(cache.clean_outputs.get(target.parameter_name), create_graph)
                    corrupted_output = _maybe_detach(cache.corrupted_outputs.get(target.parameter_name), create_graph)
                    if clean_output is None or corrupted_output is None:
                        continue
                    scores[target.component_name] = _score_component_output_tensor(
                        clean_output=clean_output,
                        corrupted_output=corrupted_output,
                        output_grad=output_grad,
                        attention_mask=attention_mask,
                        label_positions=label_positions,
                        token_mode=self.config.score_token_mode,
                        feature_slice=target.row_slice,
                    )
            return scores
        finally:
            cache.remove()

    def _finalize(
        self,
        current: _DirectionResult,
        direction_theta: dict[str, float],
        direction_theta_hat: dict[str, float],
    ) -> list[ComponentScore]:
        denominator = max(1, current.sample_count)
        scores: list[ComponentScore] = []
        for target in self.targets:
            component_name = target.component_name
            current_score = current.raw_sums[component_name] / denominator
            theta_score = direction_theta[component_name] / denominator
            theta_hat_score = direction_theta_hat[component_name] / denominator
            correction = 0.5 * (theta_score + theta_hat_score)
            raw_score = current_score + correction
            raw_sum = current.raw_sums[component_name] + 0.5 * (
                direction_theta[component_name] + direction_theta_hat[component_name]
            )
            element_count = max(1, current.element_counts[component_name])
            mean_score = raw_sum / element_count
            sqrt_numel_score = raw_score / max(1.0, math.sqrt(float(target.numel)))
            if self.config.score_normalization == "sum":
                rank_score = abs(raw_score)
            elif self.config.score_normalization == "mean":
                rank_score = abs(mean_score)
            elif self.config.score_normalization == "sqrt_numel":
                rank_score = abs(sqrt_numel_score)
            else:
                raise ValueError(f"Unsupported score_normalization: {self.config.score_normalization}")
            scores.append(
                ComponentScore(
                    parameter_name=target.parameter_name,
                    module_name=target.module_name,
                    layer_idx=target.layer_idx,
                    component_type=target.component_type,
                    granularity=target.granularity,
                    score_token_mode=self.config.score_token_mode,
                    head_idx=target.head_idx,
                    head_kind=target.head_kind,
                    row_slice=target.row_slice,
                    col_slice=target.col_slice,
                    raw_score=float(raw_score),
                    abs_score=float(abs(raw_score)),
                    mean_score=float(mean_score),
                    sqrt_numel_score=float(sqrt_numel_score),
                    rank_score=float(rank_score),
                    shape=tuple(int(value) for value in target.shape),
                    numel=int(target.numel),
                    token_count=int(current.token_counts[component_name]),
                    element_count=int(current.element_counts[component_name]),
                    localization_mode="future",
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
        delta_storage_dtype: torch.dtype | None = None
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
            delta_storage_dtype = delta.dtype
            deltas[name] = delta
            del future_parameter, base_cpu, delta
        if owns_future_state:
            future_state.clear()
        if not deltas:
            raise ValueError("No parameters matched future_delta_parameter_filter.")
        self.delta_storage_dtype = str(delta_storage_dtype) if delta_storage_dtype is not None else None
        return deltas

    def _delta_parameter_names(self, base_parameters: dict[str, nn.Parameter]) -> list[str]:
        if self.config.future_delta_parameter_filter:
            try:
                pattern = re.compile(self.config.future_delta_parameter_filter)
            except re.error as error:
                raise ValueError(f"Invalid future_delta_parameter_filter regex: {error}") from error
            return [name for name in base_parameters if pattern.search(name)]
        target_names = sorted({target.parameter_name for target in self.targets})
        return [name for name in target_names if name in base_parameters]

    def _delta_parameter_items(self, delta_tensors: dict[str, torch.Tensor]) -> list[tuple[str, nn.Parameter]]:
        base_parameters = dict(self.model.named_parameters())
        return [(name, base_parameters[name]) for name in delta_tensors]

    def _zero_float_dict(self) -> dict[str, float]:
        return {target.component_name: 0.0 for target in self.targets}

    def _zero_int_dict(self) -> dict[str, int]:
        return {target.component_name: 0 for target in self.targets}

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


class _TensorActivationCache:
    def __init__(self, targets: list[ComponentTarget]):
        self.targets = targets
        self.mode = "idle"
        self.clean_outputs: dict[str, torch.Tensor] = {}
        self.corrupted_outputs: dict[str, torch.Tensor] = {}
        self.clean_inputs: dict[str, torch.Tensor] = {}
        self.corrupted_inputs: dict[str, torch.Tensor] = {}
        self._handles = []

    def register(self) -> None:
        seen_modules = set()
        for target in self.targets:
            module_id = id(target.module)
            if module_id in seen_modules:
                continue
            seen_modules.add(module_id)
            self._handles.append(target.module.register_forward_hook(self._make_forward_hook(target)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @contextmanager
    def capture(self, mode: str):
        previous_mode = self.mode
        self.mode = mode
        try:
            yield self
        finally:
            self.mode = previous_mode

    def _make_forward_hook(self, target: ComponentTarget):
        parameter_name = target.parameter_name

        def hook(_module, inputs, output):
            if self.mode == "idle" or not torch.is_tensor(output):
                return None
            input_activation = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            if self.mode == "clean":
                self.clean_outputs[parameter_name] = output
                if input_activation is not None:
                    self.clean_inputs[parameter_name] = input_activation
                return None
            self.corrupted_outputs[parameter_name] = output
            if input_activation is not None:
                self.corrupted_inputs[parameter_name] = input_activation
            return None

        return hook


def _score_component_output_tensor(
    clean_output: torch.Tensor,
    corrupted_output: torch.Tensor,
    output_grad: torch.Tensor,
    attention_mask: torch.Tensor,
    label_positions: torch.Tensor,
    token_mode: str,
    feature_slice: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, int, int]:
    selected_clean, selected_corrupted, selected_grad = select_token_rows(
        clean_activation=clean_output,
        corrupted_activation=corrupted_output,
        output_grad=output_grad,
        attention_mask=attention_mask,
        label_positions=label_positions,
        token_mode=token_mode,
    )
    if selected_clean.numel() == 0:
        return clean_output.sum() * 0.0, 0, 0
    if feature_slice is not None:
        start, end = feature_slice
        selected_clean = selected_clean[:, start:end]
        selected_corrupted = selected_corrupted[:, start:end]
        selected_grad = selected_grad[:, start:end]
    score = ((selected_clean.float() - selected_corrupted.float()) * selected_grad.float()).sum()
    return score, int(selected_clean.size(0)), int(selected_clean.numel())


def _maybe_detach(tensor: torch.Tensor | None, keep_graph: bool) -> torch.Tensor | None:
    if tensor is None or keep_graph:
        return tensor
    return tensor.detach()


def _score_o_proj_head_input_slice_tensor(
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    weight: torch.Tensor,
    output_grad: torch.Tensor,
    col_slice: tuple[int, int],
    attention_mask: torch.Tensor,
    label_positions: torch.Tensor,
    token_mode: str,
) -> tuple[torch.Tensor, int, int]:
    selected_clean, selected_corrupted, selected_grad = select_token_rows(
        clean_activation=clean_input,
        corrupted_activation=corrupted_input,
        output_grad=output_grad,
        attention_mask=attention_mask,
        label_positions=label_positions,
        token_mode=token_mode,
    )
    if selected_clean.numel() == 0:
        return clean_input.sum() * 0.0, 0, 0
    start, end = col_slice
    selected_clean = selected_clean[:, start:end].float()
    selected_corrupted = selected_corrupted[:, start:end].float()
    selected_grad = selected_grad.float()
    weight_slice = weight[:, start:end].float()
    clean_contrib = selected_clean.matmul(weight_slice.t())
    corrupted_contrib = selected_corrupted.matmul(weight_slice.t())
    score = ((clean_contrib - corrupted_contrib) * selected_grad).sum()
    return score, int(selected_clean.size(0)), int(selected_grad.numel())


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


@contextmanager
def _temporary_parameter_delta(
    model: nn.Module,
    delta_tensors: dict[str, torch.Tensor],
    scale: float,
):
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


def _load_future_state_dict(
    model_name_or_path: str,
    cache_dir: str | None,
    parameter_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    path = Path(model_name_or_path)
    if path.exists() and path.is_file():
        if path.suffix == ".safetensors":
            return _load_safetensors_file(path, parameter_names=parameter_names)
        return _normalize_state_dict(torch.load(path, map_location="cpu"), parameter_names=parameter_names)
    if path.exists() and path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if index_path.exists():
            return _load_indexed_safetensors(path, index_path, parameter_names=parameter_names)
        safetensors_path = path / "model.safetensors"
        if safetensors_path.exists():
            return _load_safetensors_file(safetensors_path, parameter_names=parameter_names)
    ensure_src_on_path()
    from modeling_patches import patch_mistral_rotary_embedding
    from transformers import AutoModelForCausalLM

    patch_mistral_rotary_embedding()
    load_kwargs = {
        "pretrained_model_name_or_path": model_name_or_path,
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float32,
        "device_map": "cpu",
    }
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    del model
    gc.collect()
    return _filter_state_dict(state_dict, parameter_names=parameter_names)


def _normalize_state_dict(value, parameter_names: list[str] | None = None) -> dict[str, torch.Tensor]:
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict", "module"):
            nested = value.get(key)
            if isinstance(nested, dict):
                value = nested
                break
    if not isinstance(value, dict):
        raise ValueError("Future checkpoint must contain a state dict.")
    normalized = {}
    for name, tensor in value.items():
        if not torch.is_tensor(tensor):
            continue
        normalized[name.removeprefix("module.")] = tensor.detach().cpu()
    return _filter_state_dict(normalized, parameter_names=parameter_names)


def _filter_state_dict(
    state_dict: dict[str, torch.Tensor],
    parameter_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    if parameter_names is None:
        return state_dict
    return {name: state_dict[name] for name in parameter_names if name in state_dict}


def _load_safetensors_file(
    path: Path,
    parameter_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        available_names = set(handle.keys())
        names = parameter_names or sorted(available_names)
        for name in names:
            if name in available_names:
                state_dict[name] = handle.get_tensor(name).detach().cpu()
    return state_dict


def _load_indexed_safetensors(
    model_dir: Path,
    index_path: Path,
    parameter_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_data.get("weight_map", {})
    names = parameter_names or sorted(weight_map)
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name in names:
        shard_name = weight_map.get(name)
        if shard_name is not None:
            names_by_file[shard_name].append(name)
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name, shard_tensor_names in names_by_file.items():
        with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
            for name in shard_tensor_names:
                state_dict[name] = handle.get_tensor(name).detach().cpu()
    return state_dict