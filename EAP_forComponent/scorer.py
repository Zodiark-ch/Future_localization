from __future__ import annotations

import math

import torch

from EAP_forComponent.schemas import ComponentScore, ComponentTarget


class ComponentAttributionScorer:
    def __init__(
        self,
        targets: list[ComponentTarget],
        score_token_mode: str = "all_active",
        score_normalization: str = "sum",
        localization_mode: str = "current",
    ):
        if score_token_mode not in {"all_active", "label_position"}:
            raise ValueError("score_token_mode must be 'all_active' or 'label_position'")
        if score_normalization not in {"sum", "mean", "sqrt_numel"}:
            raise ValueError("score_normalization must be 'sum', 'mean', or 'sqrt_numel'")
        localization_mode = normalize_localization_mode(localization_mode)
        if localization_mode != "current":
            raise ValueError("ComponentAttributionScorer only implements current localization")
        self.targets = targets
        self.score_token_mode = score_token_mode
        self.score_normalization = score_normalization
        self.localization_mode = localization_mode
        self.raw_sums = {target.component_name: 0.0 for target in targets}
        self.token_counts = {target.component_name: 0 for target in targets}
        self.element_counts = {target.component_name: 0 for target in targets}
        self.sample_count = 0

    def score_batch(
        self,
        clean_outputs: dict[str, torch.Tensor],
        corrupted_outputs: dict[str, torch.Tensor],
        output_grads: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        label_positions: torch.Tensor,
        clean_inputs: dict[str, torch.Tensor] | None = None,
        corrupted_inputs: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.sample_count += int(attention_mask.size(0))
        clean_inputs = clean_inputs or {}
        corrupted_inputs = corrupted_inputs or {}
        for target in self.targets:
            output_grad = output_grads.get(target.parameter_name)
            if output_grad is None:
                continue
            if target.component_type == "o_proj" and target.col_slice is not None:
                clean_input = clean_inputs.get(target.parameter_name)
                corrupted_input = corrupted_inputs.get(target.parameter_name)
                if clean_input is None or corrupted_input is None:
                    continue
                raw_score, token_count, element_count = score_o_proj_head_input_slice(
                    clean_input=clean_input,
                    corrupted_input=corrupted_input,
                    weight=target.module.weight.detach(),
                    output_grad=output_grad,
                    col_slice=target.col_slice,
                    attention_mask=attention_mask,
                    label_positions=label_positions,
                    token_mode=self.score_token_mode,
                )
            else:
                clean_output = clean_outputs.get(target.parameter_name)
                corrupted_output = corrupted_outputs.get(target.parameter_name)
                if clean_output is None or corrupted_output is None:
                    continue
                raw_score, token_count, element_count = score_component_output(
                    clean_output=clean_output,
                    corrupted_output=corrupted_output,
                    output_grad=output_grad,
                    attention_mask=attention_mask,
                    label_positions=label_positions,
                    token_mode=self.score_token_mode,
                    feature_slice=target.row_slice,
                )
            self.raw_sums[target.component_name] += float(raw_score)
            self.token_counts[target.component_name] += int(token_count)
            self.element_counts[target.component_name] += int(element_count)

    def finalize(self) -> list[ComponentScore]:
        denominator = max(1, self.sample_count)
        scores: list[ComponentScore] = []
        for target in self.targets:
            raw_sum = self.raw_sums[target.component_name]
            raw_score = raw_sum / denominator
            element_count = max(1, self.element_counts[target.component_name])
            mean_score = raw_sum / element_count
            sqrt_numel_score = raw_score / max(1.0, math.sqrt(float(target.numel)))
            if self.score_normalization == "sum":
                rank_score = abs(raw_score)
            elif self.score_normalization == "mean":
                rank_score = abs(mean_score)
            else:
                rank_score = abs(sqrt_numel_score)
            scores.append(
                ComponentScore(
                    parameter_name=target.parameter_name,
                    module_name=target.module_name,
                    layer_idx=target.layer_idx,
                    component_type=target.component_type,
                    granularity=target.granularity,
                    score_token_mode=self.score_token_mode,
                    localization_mode=self.localization_mode,
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
                    token_count=int(self.token_counts[target.component_name]),
                    element_count=int(self.element_counts[target.component_name]),
                    current_score=float(raw_score),
                )
            )
        return scores


def normalize_localization_mode(localization_mode: str) -> str:
    normalized = str(localization_mode or "current").strip().lower()
    aliases = {
        "current_localization": "current",
        "future_localization": "future",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"current", "future"}:
        raise ValueError("localization_mode must be 'current' or 'future'")
    return normalized


def score_component_output(
    clean_output: torch.Tensor,
    corrupted_output: torch.Tensor,
    output_grad: torch.Tensor,
    attention_mask: torch.Tensor,
    label_positions: torch.Tensor,
    token_mode: str,
    feature_slice: tuple[int, int] | None = None,
) -> tuple[float, int, int]:
    selected_clean, selected_corrupted, selected_grad = select_token_rows(
        clean_activation=clean_output,
        corrupted_activation=corrupted_output,
        output_grad=output_grad,
        attention_mask=attention_mask,
        label_positions=label_positions,
        token_mode=token_mode,
    )
    if selected_clean.numel() == 0:
        return 0.0, 0, 0
    if feature_slice is not None:
        start, end = feature_slice
        selected_clean = selected_clean[:, start:end]
        selected_corrupted = selected_corrupted[:, start:end]
        selected_grad = selected_grad[:, start:end]
    delta = selected_clean.float() - selected_corrupted.float()
    raw_score = (delta * selected_grad.float()).sum().item()
    token_count = int(selected_clean.size(0))
    element_count = int(selected_clean.numel())
    return float(raw_score), token_count, element_count


def score_o_proj_head_input_slice(
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    weight: torch.Tensor,
    output_grad: torch.Tensor,
    col_slice: tuple[int, int],
    attention_mask: torch.Tensor,
    label_positions: torch.Tensor,
    token_mode: str,
) -> tuple[float, int, int]:
    selected_clean, selected_corrupted, selected_grad = select_token_rows(
        clean_activation=clean_input,
        corrupted_activation=corrupted_input,
        output_grad=output_grad,
        attention_mask=attention_mask,
        label_positions=label_positions,
        token_mode=token_mode,
    )
    if selected_clean.numel() == 0:
        return 0.0, 0, 0
    start, end = col_slice
    device = selected_clean.device
    selected_clean = selected_clean[:, start:end].float()
    selected_corrupted = selected_corrupted[:, start:end].float()
    selected_grad = selected_grad.float()
    weight_slice = weight[:, start:end].detach().to(device=device, dtype=torch.float32)
    clean_contrib = selected_clean.matmul(weight_slice.t())
    corrupted_contrib = selected_corrupted.matmul(weight_slice.t())
    raw_score = ((clean_contrib - corrupted_contrib) * selected_grad).sum().item()
    token_count = int(selected_clean.size(0))
    element_count = int(selected_grad.numel())
    return float(raw_score), token_count, element_count


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
    output_grad = output_grad[:, :min_seq_len]
    attention_mask = attention_mask[:, :min_seq_len]
    output_grad = output_grad.to(clean_activation.device)
    attention_mask = attention_mask.to(clean_activation.device)
    label_positions = label_positions.to(clean_activation.device)
    if token_mode == "label_position":
        rows = torch.arange(clean_activation.size(0), device=clean_activation.device)
        positions = label_positions.clamp_min(0).clamp_max(min_seq_len - 1)
        return clean_activation[rows, positions], corrupted_activation[rows, positions], output_grad[rows, positions]
    if token_mode != "all_active":
        raise ValueError(f"Unsupported token_mode: {token_mode}")
    active = attention_mask.bool()
    return clean_activation[active], corrupted_activation[active], output_grad[active]
