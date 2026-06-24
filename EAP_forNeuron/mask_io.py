from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from EAP_forNeuron.schemas import NeuronTarget, ScoreShard


class MaskSpec:
    def __init__(
        self,
        mask: dict[str, torch.Tensor],
        targets: list[NeuronTarget],
        skipped: dict[str, str],
        total_neuron_count: int,
    ):
        self.mask = mask
        self.targets = targets
        self.skipped = skipped
        self.total_neuron_count = total_neuron_count

    @property
    def candidate_count(self) -> int:
        return sum(target.candidate_count for target in self.targets)

    @classmethod
    def load(
        cls,
        mask_path: str | Path,
        model: nn.Module,
        target_modules: tuple[str, ...] | list[str],
        include_lm_head: bool = False,
        include_embed_tokens: bool = False,
        unsupported_policy: str = "drop",
    ) -> "MaskSpec":
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        loaded = torch.load(mask_path, map_location="cpu")
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected mask dict, got {type(loaded)!r}")
        mask = {str(name): tensor.bool().cpu() for name, tensor in loaded.items()}
        parameter_shapes = {name: parameter.shape for name, parameter in model.named_parameters()}
        modules = dict(model.named_modules())
        targets: list[NeuronTarget] = []
        skipped: dict[str, str] = {}
        total_neuron_count = 0
        for parameter_name, parameter_mask in mask.items():
            expected_shape = parameter_shapes.get(parameter_name)
            if expected_shape is None:
                skipped[parameter_name] = "parameter_not_found"
                continue
            if tuple(parameter_mask.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Mask shape mismatch for {parameter_name}: "
                    f"mask={tuple(parameter_mask.shape)} parameter={tuple(expected_shape)}"
                )
            module_name = parameter_name.rsplit(".", 1)[0]
            parameter_leaf = parameter_name.rsplit(".", 1)[-1]
            module = modules.get(module_name)
            if not _is_supported_parameter(
                parameter_name=parameter_name,
                parameter_leaf=parameter_leaf,
                module=module,
                target_modules=target_modules,
                include_lm_head=include_lm_head,
                include_embed_tokens=include_embed_tokens,
            ):
                skipped[parameter_name] = "unsupported_parameter"
                continue
            assert isinstance(module, nn.Linear)
            flat_indices = parameter_mask.flatten().nonzero(as_tuple=False).flatten().long().cpu()
            total_neuron_count += int(parameter_mask.numel())
            if flat_indices.numel() == 0:
                skipped[parameter_name] = "no_true_candidates"
                continue
            weight_values = module.weight.detach().flatten()[
                flat_indices.to(module.weight.device)
            ].float().cpu()
            targets.append(
                NeuronTarget(
                    parameter_name=parameter_name,
                    module_name=module_name,
                    module=module,
                    weight=module.weight,
                    shape=module.weight.shape,
                    flat_indices=flat_indices,
                    weight_values=weight_values,
                )
            )
        if unsupported_policy == "error":
            unsupported_true = [
                name for name in skipped if name in mask and int(mask[name].sum().item()) > 0
            ]
            if unsupported_true:
                raise ValueError(
                    "Unsupported mask entries contain True values: "
                    + ", ".join(unsupported_true[:10])
                )
            if not targets:
                raise ValueError("No supported mask entries found for EAP_forNeuron.")
        return cls(mask=mask, targets=targets, skipped=skipped, total_neuron_count=total_neuron_count)

    def empty_output_mask(self) -> dict[str, torch.Tensor]:
        return {name: torch.zeros_like(tensor, dtype=torch.bool) for name, tensor in self.mask.items()}


def _is_supported_parameter(
    parameter_name: str,
    parameter_leaf: str,
    module,
    target_modules: tuple[str, ...] | list[str],
    include_lm_head: bool,
    include_embed_tokens: bool,
) -> bool:
    if parameter_leaf != "weight":
        return False
    if include_lm_head and parameter_name == "lm_head.weight" and isinstance(module, nn.Linear):
        return True
    if include_embed_tokens and parameter_name.endswith("embed_tokens.weight"):
        return False
    if not isinstance(module, nn.Linear):
        return False
    return any(parameter_name.endswith(f"{suffix}.weight") for suffix in target_modules)


def dense_scores_from_shards(
    shards: dict[str, ScoreShard],
    base_mask: dict[str, torch.Tensor],
    fill_value: float = float("nan"),
) -> dict[str, torch.Tensor]:
    dense: dict[str, torch.Tensor] = {}
    for name, mask in base_mask.items():
        tensor = torch.full(mask.shape, fill_value, dtype=torch.float32)
        shard = shards.get(name)
        if shard is not None and shard.flat_indices.numel() > 0:
            tensor.flatten()[shard.flat_indices] = shard.scores.float()
        dense[name] = tensor
    return dense


def save_outputs(
    output_dir: str | Path,
    output_ratio: float,
    new_mask: dict[str, torch.Tensor],
    summary: dict,
    skipped: dict[str, str],
    scores: dict[str, ScoreShard] | None = None,
    base_mask: dict[str, torch.Tensor] | None = None,
    save_scores: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"with_{output_ratio}.pt"
    torch.save(new_mask, mask_path)
    summary_path = output_dir / "summary.json"
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False)
    summary_path.write_text(summary_text, encoding="utf-8")
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(summary_text, encoding="utf-8")
    skipped_path = output_dir / "skipped_parameters.json"
    skipped_path.write_text(json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8")
    paths = {
        "mask": mask_path,
        "summary": summary_path,
        "metadata": metadata_path,
        "skipped": skipped_path,
    }
    if scores is not None:
        selected_scores_path = output_dir / "selected_scores.pt"
        torch.save(_selected_scores(new_mask, scores), selected_scores_path)
        paths["selected_scores"] = selected_scores_path
    if save_scores and scores is not None and base_mask is not None:
        scores_path = output_dir / "scores.pt"
        torch.save(dense_scores_from_shards(scores, base_mask), scores_path)
        paths["scores"] = scores_path
    return paths


def _selected_scores(
    selected_mask: dict[str, torch.Tensor], scores: dict[str, ScoreShard]
) -> dict[str, dict[str, torch.Tensor]]:
    selected: dict[str, dict[str, torch.Tensor]] = {}
    for name, shard in scores.items():
        if name not in selected_mask:
            continue
        selected_flags = selected_mask[name].flatten()[shard.flat_indices]
        if selected_flags.any():
            selected[name] = {
                "flat_indices": shard.flat_indices[selected_flags].cpu(),
                "scores": shard.scores[selected_flags].cpu(),
                "shape": torch.tensor(list(shard.shape), dtype=torch.long),
            }
    return selected
