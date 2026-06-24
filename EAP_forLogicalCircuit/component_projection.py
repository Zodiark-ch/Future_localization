from __future__ import annotations

import math
from collections import defaultdict

from EAP_forComponent.schemas import ComponentScore

from EAP_forLogicalCircuit.circuit_builder import CircuitEdge


def project_to_component_scores(
    circuit_edges: list[CircuitEdge],
    circuit_or_edges: list[CircuitEdge],
    module_metadata: dict[str, dict],
    score_token_mode: str,
    score_normalization: str,
    localization_mode: str,
) -> list[ComponentScore]:
    circuit_sum = _sum_by_module(circuit_edges)
    circuit_or_sum = _sum_by_module(circuit_or_edges)
    all_modules = sorted(set(circuit_sum) | set(circuit_or_sum))
    scores: list[ComponentScore] = []
    for module_name in all_modules:
        metadata = module_metadata.get(module_name)
        if metadata is None:
            continue
        score_from_circuit = float(circuit_sum.get(module_name, 0.0))
        score_from_circuit_or = float(circuit_or_sum.get(module_name, 0.0))
        raw_score = (score_from_circuit + score_from_circuit_or) / 2.0
        shape = tuple(int(value) for value in metadata["shape"])
        numel = int(metadata["numel"])
        element_count = max(1, numel)
        mean_score = raw_score / float(element_count)
        sqrt_numel_score = raw_score / max(1.0, math.sqrt(float(numel)))
        if score_normalization == "sum":
            rank_score = abs(raw_score)
        elif score_normalization == "mean":
            rank_score = abs(mean_score)
        elif score_normalization == "sqrt_numel":
            rank_score = abs(sqrt_numel_score)
        else:
            raise ValueError(f"Unsupported score_normalization: {score_normalization}")
        scores.append(
            ComponentScore(
                parameter_name=metadata["parameter_name"],
                module_name=metadata.get("module_name", module_name),
                layer_idx=int(metadata["layer_idx"]),
                component_type=str(metadata["component_type"]),
                granularity=str(metadata.get("granularity", "projection_matrix")),
                score_token_mode=score_token_mode,
                head_idx=metadata.get("head_idx"),
                head_kind=metadata.get("head_kind"),
                row_slice=metadata.get("row_slice"),
                col_slice=metadata.get("col_slice"),
                raw_score=float(raw_score),
                abs_score=float(abs(raw_score)),
                mean_score=float(mean_score),
                sqrt_numel_score=float(sqrt_numel_score),
                rank_score=float(rank_score),
                shape=shape,
                numel=numel,
                token_count=int(metadata.get("edge_count", 1)),
                element_count=element_count,
                localization_mode=localization_mode,
                current_score=float(raw_score),
            )
        )
    return scores


def build_module_metadata(edge_targets: list) -> dict[str, dict]:
    grouped = {}
    module_counts: dict[str, int] = defaultdict(int)
    for target in edge_targets:
        if str(target.component_type) == "logits":
            continue
        module_counts[target.destination_module] += 1
        parameter_name = target.destination_parameter_name
        module_name = parameter_name.removesuffix(".weight") if parameter_name.endswith(".weight") else parameter_name
        row_slice = getattr(target, "destination_head_slice", None) if target.destination_kind == "attention" else None
        head_idx = getattr(target, "destination_head_idx", None) if target.destination_kind == "attention" else None
        if row_slice is not None:
            start, end = row_slice
            numel = int((end - start) * target.module.in_features)
            granularity = "head"
        else:
            numel = int(target.module.weight.numel())
            granularity = "projection_matrix"
        grouped[target.destination_module] = {
            "parameter_name": parameter_name,
            "module_name": module_name,
            "layer_idx": target.layer_idx,
            "component_type": target.component_type,
            "granularity": granularity,
            "head_idx": head_idx,
            "head_kind": _head_kind(target),
            "row_slice": row_slice,
            "col_slice": None,
            "shape": tuple(int(value) for value in target.module.weight.shape),
            "numel": numel,
        }
    for module_name, count in module_counts.items():
        grouped[module_name]["edge_count"] = int(count)
    return grouped


def _sum_by_module(edges: list[CircuitEdge]) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    for edge in edges:
        values[edge.destination_module] += float(edge.score)
    return dict(values)


def _head_kind(target) -> str | None:
    if getattr(target, "destination_kind", None) != "attention":
        return None
    qkv = getattr(target, "destination_qkv", None)
    if qkv == "q":
        return "query"
    if qkv in {"k", "v"}:
        return "key_value"
    return None
