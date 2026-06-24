from __future__ import annotations

import re
from collections.abc import Iterable

from torch import nn

from EAP_forComponent.schemas import ATTENTION_MODULES, ComponentTarget, MLP_MODULES


class ComponentRegistry:
    def __init__(self, targets: list[ComponentTarget], metadata: dict):
        self._targets = targets
        self.metadata = metadata

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        target_modules: Iterable[str],
        attention_granularity: str = "projection_matrix",
    ) -> "ComponentRegistry":
        if attention_granularity not in {"projection_matrix", "head"}:
            raise ValueError("attention_granularity must be 'projection_matrix' or 'head'")
        target_module_set = {item.strip() for item in target_modules if item.strip()}
        config = getattr(model, "config", None)
        num_attention_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        num_key_value_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
        if num_key_value_heads <= 0:
            num_key_value_heads = num_attention_heads
        targets: list[ComponentTarget] = []
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            component_type = module_name.rsplit(".", 1)[-1]
            if component_type not in target_module_set:
                continue
            if component_type not in ATTENTION_MODULES and component_type not in MLP_MODULES:
                continue
            parameter_name = f"{module_name}.weight"
            if _is_lora_or_adapter_parameter(parameter_name):
                continue
            layer_idx = _parse_layer_idx(module_name)
            if attention_granularity == "head" and component_type in ATTENTION_MODULES:
                targets.extend(
                    _head_targets(
                        parameter_name=parameter_name,
                        module_name=module_name,
                        module=module,
                        layer_idx=layer_idx,
                        component_type=component_type,
                        num_attention_heads=num_attention_heads,
                        num_key_value_heads=num_key_value_heads,
                    )
                )
                continue
            targets.append(
                ComponentTarget(
                    parameter_name=parameter_name,
                    module_name=module_name,
                    layer_idx=layer_idx,
                    component_type=component_type,
                    granularity="projection_matrix",
                    head_idx=None,
                    head_kind=None,
                    row_slice=None,
                    col_slice=None,
                    module=module,
                    shape=module.weight.shape,
                    numel=int(module.weight.numel()),
                )
            )
        if not targets:
            raise ValueError("No supported q/k/v/o/gate/up/down Linear components found.")
        metadata = {
            "attention_granularity": attention_granularity,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "query_heads_per_kv_head": (
                num_attention_heads // num_key_value_heads
                if num_attention_heads and num_key_value_heads
                else None
            ),
            "target_count": len(targets),
            "target_modules": sorted(target_module_set),
        }
        return cls(targets=targets, metadata=metadata)

    def targets(self) -> list[ComponentTarget]:
        return list(self._targets)

    def parameter_names(self) -> list[str]:
        return sorted({target.parameter_name for target in self._targets})


def _head_targets(
    parameter_name: str,
    module_name: str,
    module: nn.Linear,
    layer_idx: int,
    component_type: str,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> list[ComponentTarget]:
    out_features, in_features = module.weight.shape
    if component_type in {"q_proj", "o_proj"}:
        head_count = num_attention_heads or _infer_head_count(out_features, in_features)
        head_kind = "query" if component_type == "q_proj" else "output_input_slice"
    else:
        head_count = num_key_value_heads or _infer_head_count(out_features, in_features)
        head_kind = "key_value"
    if head_count <= 0:
        raise ValueError(f"Cannot infer head count for {parameter_name}")
    targets: list[ComponentTarget] = []
    if component_type == "o_proj":
        if in_features % head_count != 0:
            raise ValueError(f"{parameter_name} in_features={in_features} not divisible by {head_count}")
        head_dim = in_features // head_count
        for head_idx in range(head_count):
            start = head_idx * head_dim
            end = start + head_dim
            targets.append(
                ComponentTarget(
                    parameter_name=parameter_name,
                    module_name=module_name,
                    layer_idx=layer_idx,
                    component_type=component_type,
                    granularity="head",
                    head_idx=head_idx,
                    head_kind=head_kind,
                    row_slice=None,
                    col_slice=(start, end),
                    module=module,
                    shape=module.weight.shape,
                    numel=int(out_features * head_dim),
                )
            )
        return targets
    if out_features % head_count != 0:
        raise ValueError(f"{parameter_name} out_features={out_features} not divisible by {head_count}")
    head_dim = out_features // head_count
    for head_idx in range(head_count):
        start = head_idx * head_dim
        end = start + head_dim
        targets.append(
            ComponentTarget(
                parameter_name=parameter_name,
                module_name=module_name,
                layer_idx=layer_idx,
                component_type=component_type,
                granularity="head",
                head_idx=head_idx,
                head_kind=head_kind,
                row_slice=(start, end),
                col_slice=None,
                module=module,
                shape=module.weight.shape,
                numel=int(head_dim * in_features),
            )
        )
    return targets


def _parse_layer_idx(module_name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    return -1


def _is_lora_or_adapter_parameter(parameter_name: str) -> bool:
    lowered = parameter_name.lower()
    return "lora_" in lowered or ".adapter" in lowered or "modules_to_save" in lowered


def _infer_head_count(out_features: int, in_features: int) -> int:
    if out_features == in_features:
        return 1
    return 0
