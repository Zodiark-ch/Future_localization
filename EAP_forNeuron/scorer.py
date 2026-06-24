from __future__ import annotations

import torch

from EAP_forNeuron.schemas import NeuronTarget, ScoreShard


DTYPES = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class ParameterNeuronScorer:
    def __init__(
        self,
        targets: list[NeuronTarget],
        score_dtype: str = "float32",
        row_chunk_size: int = 256,
        score_token_mode: str = "label_position",
    ):
        self.targets = {target.parameter_name: target for target in targets}
        self.score_dtype = DTYPES.get(score_dtype, torch.float32)
        self.row_chunk_size = row_chunk_size
        self.score_token_mode = score_token_mode
        self.scores: dict[str, torch.Tensor] = {
            target.parameter_name: torch.zeros(target.candidate_count, dtype=torch.float32)
            for target in targets
        }
        self.normalizer = 0

    def score_batch(
        self,
        clean_inputs: dict[str, torch.Tensor],
        corrupted_inputs: dict[str, torch.Tensor],
        output_grads: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        label_positions: torch.Tensor,
    ) -> None:
        self.normalizer += int(attention_mask.size(0))
        for parameter_name, target in self.targets.items():
            clean_input = clean_inputs.get(parameter_name)
            corrupted_input = corrupted_inputs.get(parameter_name)
            output_grad = output_grads.get(parameter_name)
            if clean_input is None or corrupted_input is None or output_grad is None:
                continue
            score_delta = score_linear_weight_candidates(
                weight=target.weight.detach(),
                flat_indices=target.flat_indices,
                candidate_weights=target.weight_values,
                clean_input=clean_input,
                corrupted_input=corrupted_input,
                output_grad=output_grad,
                attention_mask=attention_mask,
                label_positions=label_positions,
                score_token_mode=self.score_token_mode,
                row_chunk_size=self.row_chunk_size,
            )
            self.scores[parameter_name] += score_delta.cpu().float()

    def finalize(self) -> dict[str, ScoreShard]:
        denominator = max(1, self.normalizer)
        finalized: dict[str, ScoreShard] = {}
        for parameter_name, target in self.targets.items():
            finalized[parameter_name] = ScoreShard(
                parameter_name=parameter_name,
                shape=target.shape,
                flat_indices=target.flat_indices.clone().cpu(),
                scores=(self.scores[parameter_name] / denominator).to(self.score_dtype).cpu(),
            )
        return finalized


def score_linear_weight_candidates(
    weight: torch.Tensor,
    flat_indices: torch.Tensor,
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    output_grad: torch.Tensor,
    attention_mask: torch.Tensor,
    label_positions: torch.Tensor,
    score_token_mode: str = "label_position",
    row_chunk_size: int = 256,
    candidate_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if flat_indices.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    device = output_grad.device
    flat_indices = flat_indices.to(device=device, dtype=torch.long)
    out_features, in_features = weight.shape
    rows = torch.div(flat_indices, in_features, rounding_mode="floor")
    cols = flat_indices.remainder(in_features)
    if candidate_weights is None:
        candidate_weights = weight.flatten()[flat_indices.to(weight.device)].detach().float().cpu()
    candidate_weights = candidate_weights.to(device=device, dtype=torch.float32)
    selected_clean, selected_corrupted, selected_grad = _select_token_rows(
        clean_input=clean_input.to(device),
        corrupted_input=corrupted_input.to(device),
        output_grad=output_grad,
        attention_mask=attention_mask.to(device),
        label_positions=label_positions.to(device),
        score_token_mode=score_token_mode,
    )
    if selected_clean.numel() == 0:
        return torch.zeros(flat_indices.numel(), dtype=torch.float32, device=device)
    result = torch.empty(flat_indices.numel(), dtype=torch.float32, device=device)
    unique_rows = torch.unique(rows, sorted=True)
    for start in range(0, unique_rows.numel(), row_chunk_size):
        row_chunk = unique_rows[start : start + row_chunk_size]
        row_mask = torch.isin(rows, row_chunk)
        candidate_positions = row_mask.nonzero(as_tuple=False).flatten()
        chunk_rows = rows[candidate_positions]
        for row in row_chunk.tolist():
            positions = candidate_positions[chunk_rows == row]
            if positions.numel() == 0:
                continue
            row_cols = cols[positions]
            delta_x = selected_corrupted[:, row_cols] - selected_clean[:, row_cols]
            grad = selected_grad[:, row].unsqueeze(1)
            row_weight = candidate_weights[positions]
            result[positions] = (delta_x.float() * grad.float()).sum(dim=0) * row_weight
    return result


def _select_token_rows(
    clean_input: torch.Tensor,
    corrupted_input: torch.Tensor,
    output_grad: torch.Tensor,
    attention_mask: torch.Tensor,
    label_positions: torch.Tensor,
    score_token_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_seq_len = min(clean_input.size(1), corrupted_input.size(1), output_grad.size(1), attention_mask.size(1))
    clean_input = clean_input[:, :min_seq_len]
    corrupted_input = corrupted_input[:, :min_seq_len]
    output_grad = output_grad[:, :min_seq_len]
    attention_mask = attention_mask[:, :min_seq_len]
    if score_token_mode == "label_position":
        rows = torch.arange(clean_input.size(0), device=clean_input.device)
        positions = label_positions.clamp_min(0).clamp_max(min_seq_len - 1)
        return clean_input[rows, positions], corrupted_input[rows, positions], output_grad[rows, positions]
    if score_token_mode != "all_active":
        raise ValueError(f"Unsupported score_token_mode: {score_token_mode}")
    active = attention_mask.bool()
    return clean_input[active], corrupted_input[active], output_grad[active]
