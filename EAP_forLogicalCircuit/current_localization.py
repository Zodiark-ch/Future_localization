from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from EAP_forLogicalCircuit.graph_registry import EdgeTarget
from EAP_forLogicalCircuit.schemas import EdgeScore


class CurrentEdgeAttributionScorer:
    def __init__(
        self,
        edge_targets: list[EdgeTarget],
        score_token_mode: str = "all_active",
        score_normalization: str = "sum",
    ):
        if score_token_mode not in {"all_active", "label_position"}:
            raise ValueError("score_token_mode must be 'all_active' or 'label_position'")
        if score_normalization not in {"sum", "mean", "sqrt_numel"}:
            raise ValueError("score_normalization must be 'sum', 'mean', or 'sqrt_numel'")
        self.edge_targets = edge_targets
        self.score_token_mode = score_token_mode
        self.score_normalization = score_normalization
        self.raw_sums = {target.edge_id: 0.0 for target in edge_targets}
        self.token_counts = {target.edge_id: 0 for target in edge_targets}
        self.element_counts = {target.edge_id: 0 for target in edge_targets}
        self.sample_count = 0

    def score_batch(
        self,
        clean_inputs: dict[str, torch.Tensor],
        corrupted_inputs: dict[str, torch.Tensor],
        input_grads: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        label_positions: torch.Tensor,
        source_clean_outputs: dict[str, torch.Tensor] | None = None,
        source_corrupted_outputs: dict[str, torch.Tensor] | None = None,
        output_grads: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.sample_count += int(attention_mask.size(0))
        source_clean_outputs = source_clean_outputs or {}
        source_corrupted_outputs = source_corrupted_outputs or {}
        output_grads = output_grads or {}
        for target in self.edge_targets:
            clean_source = source_clean_outputs.get(target.source_node)
            corrupted_source = source_corrupted_outputs.get(target.source_node)
            if clean_source is None:
                clean_source = _first_available(clean_inputs, _destination_input_keys(target))
            if corrupted_source is None:
                corrupted_source = _first_available(corrupted_inputs, _destination_input_keys(target))
            grad_input = destination_gradient(target, input_grads=input_grads, output_grads=output_grads)
            if clean_source is None or corrupted_source is None or grad_input is None:
                continue
            selected_clean, selected_corrupted, selected_grad = select_token_rows(
                clean_activation=clean_source,
                corrupted_activation=corrupted_source,
                output_grad=grad_input,
                attention_mask=attention_mask,
                label_positions=label_positions,
                token_mode=self.score_token_mode,
            )
            if selected_clean.numel() == 0:
                continue
            edge_delta = selected_clean.float() - selected_corrupted.float()
            raw_score = (edge_delta * selected_grad.float()).sum().item()
            self.raw_sums[target.edge_id] += float(raw_score)
            self.token_counts[target.edge_id] += int(selected_clean.size(0))
            self.element_counts[target.edge_id] += int(selected_clean.numel())

    def finalize(self) -> list[EdgeScore]:
        denominator = max(1, self.sample_count)
        scores: list[EdgeScore] = []
        for target in self.edge_targets:
            raw_sum = self.raw_sums[target.edge_id]
            raw_score = raw_sum / denominator
            element_count = max(1, self.element_counts[target.edge_id])
            mean_score = raw_sum / element_count
            sqrt_numel_score = raw_score / max(1.0, math.sqrt(float(target.num_features)))
            if self.score_normalization == "sum":
                rank_score = abs(raw_score)
            elif self.score_normalization == "mean":
                rank_score = abs(mean_score)
            else:
                rank_score = abs(sqrt_numel_score)
            scores.append(
                EdgeScore(
                    edge_id=target.edge_id,
                    source_node=target.source_node,
                    destination_module=target.destination_module,
                    layer_idx=target.layer_idx,
                    component_type=target.component_type,
                    score_token_mode=self.score_token_mode,
                    localization_mode="current",
                    raw_score=float(raw_score),
                    abs_score=float(abs(raw_score)),
                    mean_score=float(mean_score),
                    sqrt_numel_score=float(sqrt_numel_score),
                    rank_score=float(rank_score),
                    token_count=int(self.token_counts[target.edge_id]),
                    element_count=int(self.element_counts[target.edge_id]),
                    numel=int(target.num_features),
                )
            )
        return scores


def select_token_rows(
    clean_activation: torch.Tensor,
    corrupted_activation: torch.Tensor,
    output_grad: torch.Tensor,
    attention_mask: torch.Tensor,
    label_positions: torch.Tensor,
    token_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_seq_len = min(
        clean_activation.size(1),
        corrupted_activation.size(1),
        output_grad.size(1),
        attention_mask.size(1),
    )
    clean_activation = clean_activation[:, :min_seq_len]
    corrupted_activation = corrupted_activation[:, :min_seq_len]
    output_grad = output_grad[:, :min_seq_len].to(clean_activation.device)
    min_hidden_size = min(
        clean_activation.size(-1),
        corrupted_activation.size(-1),
        output_grad.size(-1),
    )
    clean_activation = clean_activation[..., :min_hidden_size]
    corrupted_activation = corrupted_activation[..., :min_hidden_size]
    output_grad = output_grad[..., :min_hidden_size]
    attention_mask = attention_mask[:, :min_seq_len].to(clean_activation.device)
    label_positions = label_positions.to(clean_activation.device)
    if token_mode == "label_position":
        rows = torch.arange(clean_activation.size(0), device=clean_activation.device)
        positions = label_positions.clamp_min(0).clamp_max(min_seq_len - 1)
        return clean_activation[rows, positions], corrupted_activation[rows, positions], output_grad[rows, positions]
    if token_mode != "all_active":
        raise ValueError(f"Unsupported token_mode: {token_mode}")
    active = attention_mask.bool()
    return clean_activation[active], corrupted_activation[active], output_grad[active]


def destination_gradient(
    target: EdgeTarget,
    input_grads: dict[str, torch.Tensor],
    output_grads: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if target.destination_kind == "attention":
        key = target.destination_output_key or target.destination_parameter_name
        output_grad = output_grads.get(key)
        if output_grad is None or target.destination_head_slice is None:
            return None
        start, end = target.destination_head_slice
        if output_grad.size(-1) < end:
            return None
        grad_head = output_grad[..., start:end]
        weight = target.module.weight[start:end]
        if not grad_head.requires_grad:
            grad_head = grad_head.detach().float()
            weight = weight.detach().to(device=grad_head.device, dtype=torch.float32)
        elif weight.device != grad_head.device or weight.dtype != grad_head.dtype:
            weight = weight.to(device=grad_head.device, dtype=grad_head.dtype)
        return F.linear(grad_head, weight.transpose(0, 1))

    grads = [input_grads[key] for key in _destination_input_keys(target) if key in input_grads]
    if not grads:
        return None
    total = grads[0]
    for grad in grads[1:]:
        total = _aligned_sum(total, grad)
    return total


def _aligned_sum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    min_seq_len = min(left.size(1), right.size(1))
    min_hidden = min(left.size(-1), right.size(-1))
    return left[:, :min_seq_len, :min_hidden] + right[:, :min_seq_len, :min_hidden].to(left.device)


def _first_available(values: dict[str, torch.Tensor], keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = values.get(key)
        if value is not None:
            return value
    return None


def _destination_input_keys(target: EdgeTarget) -> tuple[str, ...]:
    keys = getattr(target, "destination_input_keys", tuple())
    return keys or (target.destination_parameter_name,)
