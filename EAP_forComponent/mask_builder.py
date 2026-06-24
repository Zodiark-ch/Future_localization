from __future__ import annotations

import torch
from torch import nn

from EAP_forComponent.schemas import ComponentScore


class ComponentMaskBuilder:
    def __init__(
        self,
        mask_fill_strategy: str = "random",
        seed: int = 0,
        include_all_parameters: bool = True,
        min_keep_ratio: float = 0.0,
        max_keep_ratio: float = 1.0,
    ):
        if mask_fill_strategy not in {"random", "magnitude", "first"}:
            raise ValueError("mask_fill_strategy must be random, magnitude, or first")
        min_keep_ratio = float(min_keep_ratio)
        max_keep_ratio = float(max_keep_ratio)
        if not 0.0 <= min_keep_ratio <= 1.0:
            raise ValueError("min_keep_ratio must be in [0, 1]")
        if not 0.0 <= max_keep_ratio <= 1.0:
            raise ValueError("max_keep_ratio must be in [0, 1]")
        if min_keep_ratio > max_keep_ratio:
            raise ValueError("min_keep_ratio must be <= max_keep_ratio")
        self.mask_fill_strategy = mask_fill_strategy
        self.seed = seed
        self.include_all_parameters = include_all_parameters
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio
        self.last_summary: dict = {}

    def build(self, model: nn.Module, scores: list[ComponentScore]) -> dict[str, torch.Tensor]:
        named_parameters = dict(model.named_parameters())
        if self.include_all_parameters:
            output_mask = {
                name: torch.zeros(parameter.shape, dtype=torch.bool, device="cpu")
                for name, parameter in named_parameters.items()
            }
        else:
            output_mask = {
                score.parameter_name: torch.zeros(score.shape, dtype=torch.bool, device="cpu")
                for score in scores
            }
        sorted_scores = sorted(scores, key=lambda item: item.rank_score, reverse=True)
        component_count = len(sorted_scores)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        per_component = []
        for order, score in enumerate(sorted_scores):
            if score.parameter_name not in output_mask:
                output_mask[score.parameter_name] = torch.zeros(score.shape, dtype=torch.bool, device="cpu")
            keep_ratio = self._keep_ratio(order=order, component_count=component_count)
            true_count = int(score.numel * keep_ratio)
            parameter = named_parameters.get(score.parameter_name)
            parameter_values = parameter.detach().cpu() if parameter is not None else None
            component_mask = _build_component_mask(
                score=score,
                parameter_values=parameter_values,
                true_count=true_count,
                fill_strategy=self.mask_fill_strategy,
                generator=generator,
            )
            _write_component_mask(output_mask[score.parameter_name], score, component_mask)
            per_component.append(
                {
                    "component_name": score.component_name,
                    "parameter_name": score.parameter_name,
                    "rank_score": float(score.rank_score),
                    "true_ratio": float(keep_ratio),
                    "keep_ratio": float(keep_ratio),
                    "freeze_ratio": float(1.0 - keep_ratio),
                    "true_count": int(component_mask.sum().item()),
                    "numel": int(score.numel),
                }
            )
        self.last_summary = {
            "mask_fill_strategy": self.mask_fill_strategy,
            "mask_seed": self.seed,
            "include_all_parameters": self.include_all_parameters,
            "min_keep_ratio": self.min_keep_ratio,
            "max_keep_ratio": self.max_keep_ratio,
            "component_count": component_count,
            "total_true": int(sum(mask.sum().item() for mask in output_mask.values())),
            "per_component": per_component,
        }
        return output_mask

    def _keep_ratio(self, order: int, component_count: int) -> float:
        if component_count <= 1:
            return self.max_keep_ratio
        rank_fraction = order / (component_count - 1)
        return self.max_keep_ratio - rank_fraction * (self.max_keep_ratio - self.min_keep_ratio)


def _build_component_mask(
    score: ComponentScore,
    parameter_values: torch.Tensor | None,
    true_count: int,
    fill_strategy: str,
    generator: torch.Generator,
) -> torch.Tensor:
    component_shape = _component_shape(score)
    component_numel = int(torch.tensor(component_shape).prod().item()) if component_shape else 0
    true_count = max(0, min(int(true_count), component_numel))
    flat = torch.zeros(component_numel, dtype=torch.bool)
    if true_count <= 0 or component_numel == 0:
        return flat.reshape(component_shape)
    if true_count >= component_numel:
        flat[:] = True
        return flat.reshape(component_shape)
    if fill_strategy == "random":
        indices = torch.randperm(component_numel, generator=generator)[:true_count]
    elif fill_strategy == "magnitude" and parameter_values is not None:
        values = _component_values(parameter_values, score).reshape(-1).abs().float()
        indices = torch.topk(values, k=true_count, largest=True).indices
    else:
        indices = torch.arange(true_count, dtype=torch.long)
    flat[indices] = True
    return flat.reshape(component_shape)


def _write_component_mask(target_mask: torch.Tensor, score: ComponentScore, component_mask: torch.Tensor) -> None:
    if score.row_slice is not None:
        start, end = score.row_slice
        target_mask[start:end, :] = component_mask
    elif score.col_slice is not None:
        start, end = score.col_slice
        target_mask[:, start:end] = component_mask
    else:
        target_mask.copy_(component_mask)


def _component_shape(score: ComponentScore) -> tuple[int, int]:
    out_features, in_features = score.shape
    if score.row_slice is not None:
        start, end = score.row_slice
        return (end - start, in_features)
    if score.col_slice is not None:
        start, end = score.col_slice
        return (out_features, end - start)
    return (out_features, in_features)


def _component_values(parameter_values: torch.Tensor, score: ComponentScore) -> torch.Tensor:
    if score.row_slice is not None:
        start, end = score.row_slice
        return parameter_values[start:end, :]
    if score.col_slice is not None:
        start, end = score.col_slice
        return parameter_values[:, start:end]
    return parameter_values
