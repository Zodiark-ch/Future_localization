from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from torch import nn

from EAP_forComponent.outputs import component_score_to_dict
from EAP_forComponent.schemas import ComponentScore
from EAP_forLogicalCircuit.mask_builder import ComponentMaskBuilder
from EAP_forLogicalCircuit.rank_allocator import LoraRankAllocator


@dataclass
class TaskArtifact:
    task_name: str
    component_scores: list[ComponentScore]
    module_assignment: dict[str, tuple[int | None, str]]
    module_rank: dict[str, int]


def run_conflict_analysis(
    task_dirs: dict[str, Path],
    output_dir: Path,
    min_rank: int = 1,
    max_rank: int = 32,
    rank_budget: int | None = None,
    rank_multiple: int = 1,
    head_to_matrix_aggregation: str = "mean",
    rank_score_source: str = "normalized_abs",
    mask_fill_strategy: str = "magnitude",
    mask_seed: int = 0,
    mask_min_keep_ratio: float = 0.1,
    mask_max_keep_ratio: float = 0.9,
    write_dense_masks: bool = False,
) -> dict[str, Path]:
    task_artifacts = [load_task_artifact(task_name=name, task_dir=path) for name, path in task_dirs.items()]
    analysis = analyze_conflicts(task_artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: dict[str, Path] = {}
    for task_name, task_scores in analysis["task_all_components"].items():
        reasons = analysis["task_assignment_reasons"][task_name]
        task_out = output_dir / "task_all_components" / task_name
        paths = save_component_artifacts(
            output_dir=task_out,
            component_scores=task_scores,
            assignment_reasons=reasons,
            source_tasks=analysis["task_source_tasks"][task_name],
            summary={
                "scope": "task_all_components",
                "task_name": task_name,
                "component_count": len(task_scores),
            },
            min_rank=min_rank,
            max_rank=max_rank,
            rank_budget=rank_budget,
            rank_multiple=rank_multiple,
            head_to_matrix_aggregation=head_to_matrix_aggregation,
            rank_score_source=rank_score_source,
            mask_fill_strategy=mask_fill_strategy,
            mask_seed=mask_seed,
            mask_min_keep_ratio=mask_min_keep_ratio,
            mask_max_keep_ratio=mask_max_keep_ratio,
            write_dense_masks=write_dense_masks,
        )
        written_paths[f"task_all_components_{task_name}"] = task_out
        written_paths.update({f"task_all_components_{task_name}_{k}": v for k, v in paths.items()})

    conflict_out = output_dir / "conflict_components"
    conflict_paths = save_component_artifacts(
        output_dir=conflict_out,
        component_scores=analysis["conflict_components"],
        assignment_reasons=analysis["conflict_assignment_reasons"],
        source_tasks=analysis["conflict_source_tasks"],
        summary={
            "scope": "conflict_components",
            "component_count": len(analysis["conflict_components"]),
        },
        min_rank=min_rank,
        max_rank=max_rank,
        rank_budget=rank_budget,
        rank_multiple=rank_multiple,
        head_to_matrix_aggregation=head_to_matrix_aggregation,
        rank_score_source=rank_score_source,
        mask_fill_strategy=mask_fill_strategy,
        mask_seed=mask_seed,
        mask_min_keep_ratio=mask_min_keep_ratio,
        mask_max_keep_ratio=mask_max_keep_ratio,
        write_dense_masks=write_dense_masks,
    )
    written_paths["conflict_components"] = conflict_out
    written_paths.update({f"conflict_components_{k}": v for k, v in conflict_paths.items()})

    all_task_out = output_dir / "all_task_components"
    all_task_paths = save_component_artifacts(
        output_dir=all_task_out,
        component_scores=analysis["all_task_components"],
        assignment_reasons=analysis["all_task_assignment_reasons"],
        source_tasks=analysis["all_task_source_tasks"],
        summary={
            "scope": "all_task_components",
            "component_count": len(analysis["all_task_components"]),
        },
        min_rank=min_rank,
        max_rank=max_rank,
        rank_budget=rank_budget,
        rank_multiple=rank_multiple,
        head_to_matrix_aggregation=head_to_matrix_aggregation,
        rank_score_source=rank_score_source,
        mask_fill_strategy=mask_fill_strategy,
        mask_seed=mask_seed,
        mask_min_keep_ratio=mask_min_keep_ratio,
        mask_max_keep_ratio=mask_max_keep_ratio,
        write_dense_masks=write_dense_masks,
    )
    written_paths["all_task_components"] = all_task_out
    written_paths.update({f"all_task_components_{k}": v for k, v in all_task_paths.items()})

    conflict_summary_path = output_dir / "conflict_summary.json"
    conflict_summary = {
        "tasks": [artifact.task_name for artifact in task_artifacts],
        "task_component_counts": {
            artifact.task_name: len(analysis["task_all_components"][artifact.task_name]) for artifact in task_artifacts
        },
        "conflict_component_count": len(analysis["conflict_components"]),
        "all_task_component_count": len(analysis["all_task_components"]),
        "conflict_assignment_reasons": analysis["conflict_assignment_reasons"],
        "all_task_assignment_reasons": analysis["all_task_assignment_reasons"],
        "conflict_source_tasks": analysis["conflict_source_tasks"],
        "all_task_source_tasks": analysis["all_task_source_tasks"],
    }
    conflict_summary_path.write_text(json.dumps(conflict_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    written_paths["conflict_summary"] = conflict_summary_path
    return written_paths


def analyze_conflicts(task_artifacts: list[TaskArtifact]) -> dict:
    if not task_artifacts:
        raise ValueError("No task artifacts provided")

    module_templates: dict[str, ComponentScore] = {}
    module_task_scores: dict[str, dict[str, ComponentScore]] = defaultdict(dict)
    module_task_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    task_names = [artifact.task_name for artifact in task_artifacts]
    conflict_source_tasks: dict[str, list[str]] = {}
    all_task_source_tasks: dict[str, list[str]] = {}
    task_source_tasks: dict[str, dict[str, list[str]]] = {}

    for artifact in task_artifacts:
        for score in artifact.component_scores:
            component_key = score.component_name
            module_templates.setdefault(component_key, score)
            module_task_scores[component_key][artifact.task_name] = score
            if component_key in artifact.module_rank:
                module_task_ranks[component_key][artifact.task_name] = artifact.module_rank[component_key]
            elif score.module_name in artifact.module_rank:
                module_task_ranks[component_key][artifact.task_name] = artifact.module_rank[score.module_name]

    all_modules = sorted(module_templates)

    task_all_components: dict[str, list[ComponentScore]] = {}
    task_assignment_reasons: dict[str, dict[str, str]] = {}
    for artifact in task_artifacts:
        scores_for_task: list[ComponentScore] = []
        reasons_for_task: dict[str, str] = {}
        sources_for_task: dict[str, list[str]] = {}
        for module_name in all_modules:
            template = module_templates[module_name]
            score = module_task_scores.get(module_name, {}).get(artifact.task_name)
            if score is None:
                score = _zero_like(template)
            scores_for_task.append(score)
            _assignment_value, reason = _lookup_assignment(
                assignments=artifact.module_assignment,
                component_key=module_name,
                module_name=template.module_name,
            )
            reasons_for_task[module_name] = reason
            sources_for_task[module_name] = [
                task_name for task_name in task_names if task_name in module_task_scores.get(module_name, {})
            ]
        task_all_components[artifact.task_name] = scores_for_task
        task_assignment_reasons[artifact.task_name] = reasons_for_task
        task_source_tasks[artifact.task_name] = sources_for_task

    conflict_components: list[ComponentScore] = []
    all_task_components: list[ComponentScore] = []
    conflict_assignment_reasons: dict[str, str] = {}
    all_task_assignment_reasons: dict[str, str] = {}

    for module_name in all_modules:
        template = module_templates[module_name]
        assignment_values = []
        for artifact in task_artifacts:
            assignment, _reason = _lookup_assignment(
                assignments=artifact.module_assignment,
                component_key=module_name,
                module_name=template.module_name,
            )
            if assignment is not None:
                assignment_values.append(assignment)
        has_positive = 1 in assignment_values
        has_or_excluded = 0 in assignment_values

        per_task_scores = [
            module_task_scores[module_name][task_name]
            for task_name in task_names
            if task_name in module_task_scores[module_name]
        ]
        if not per_task_scores:
            continue
        mean_rank = _mean_rank_value(module_name, module_task_ranks, task_names)
        aggregated = _aggregate_scores(per_task_scores, template, mean_rank)

        if has_positive and has_or_excluded:
            conflict_components.append(aggregated)
            conflict_assignment_reasons[module_name] = "conflict_unresolved"
            conflict_source_tasks[module_name] = [
                task_name for task_name in task_names if task_name in module_task_scores[module_name]
            ]
            all_task_components.append(aggregated)
            all_task_assignment_reasons[module_name] = "conflict_unresolved"
            all_task_source_tasks[module_name] = [
                task_name for task_name in task_names if task_name in module_task_scores[module_name]
            ]
            continue
        if has_positive:
            all_task_components.append(aggregated)
            all_task_assignment_reasons[module_name] = "positive_path"
            all_task_source_tasks[module_name] = [
                task_name for task_name in task_names if task_name in module_task_scores[module_name]
            ]

    _apply_mean_rank_score(conflict_components)
    _apply_mean_rank_score(all_task_components)

    return {
        "task_all_components": task_all_components,
        "task_assignment_reasons": task_assignment_reasons,
        "task_source_tasks": task_source_tasks,
        "conflict_components": conflict_components,
        "conflict_assignment_reasons": conflict_assignment_reasons,
        "conflict_source_tasks": conflict_source_tasks,
        "all_task_components": all_task_components,
        "all_task_assignment_reasons": all_task_assignment_reasons,
        "all_task_source_tasks": all_task_source_tasks,
    }


def _lookup_assignment(
    assignments: dict[str, tuple[int | None, str]],
    component_key: str,
    module_name: str,
) -> tuple[int | None, str]:
    if component_key in assignments:
        return assignments[component_key]
    if module_name in assignments:
        return assignments[module_name]
    return (None, "absent")


def save_component_artifacts(
    output_dir: Path,
    component_scores: list[ComponentScore],
    assignment_reasons: dict[str, str],
    source_tasks: dict[str, list[str]],
    summary: dict,
    min_rank: int,
    max_rank: int,
    rank_budget: int | None,
    rank_multiple: int,
    head_to_matrix_aggregation: str,
    rank_score_source: str,
    mask_fill_strategy: str,
    mask_seed: int,
    mask_min_keep_ratio: float,
    mask_max_keep_ratio: float,
    write_dense_masks: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    allocator = LoraRankAllocator(
        min_rank=min_rank,
        max_rank=max_rank,
        rank_multiple=rank_multiple,
        rank_budget=rank_budget,
        head_to_matrix_aggregation=head_to_matrix_aggregation,
        rank_score_source=rank_score_source,
    )
    ranks = allocator.allocate(component_scores) if component_scores else {}
    rank_pattern = allocator.to_rank_pattern(ranks)

    mask_summary = {
        "write_dense_masks": bool(write_dense_masks),
        "mask_fill_strategy": mask_fill_strategy,
        "mask_seed": mask_seed,
        "min_keep_ratio": float(mask_min_keep_ratio),
        "max_keep_ratio": float(mask_max_keep_ratio),
        "include_all_parameters": False,
        "component_count": len(component_scores),
    }
    component_mask_path: Path | None = None
    if write_dense_masks:
        mask_builder = ComponentMaskBuilder(
            mask_fill_strategy=mask_fill_strategy,
            seed=mask_seed,
            include_all_parameters=False,
            min_keep_ratio=mask_min_keep_ratio,
            max_keep_ratio=mask_max_keep_ratio,
        )
        component_mask = mask_builder.build(nn.Module(), component_scores) if component_scores else {}
        component_mask_path = output_dir / "component_mask.pt"
        torch.save(component_mask, component_mask_path)
        mask_summary = dict(mask_builder.last_summary)
        mask_summary["write_dense_masks"] = True
    else:
        mask_summary["skipped_reason"] = "dense_masks_disabled_for_conflict_analysis"

    score_dicts = []
    for score in sorted(component_scores, key=lambda item: _sort_key(item)):
        component_key = score.component_name
        data = component_score_to_dict(score)
        data["assignment_reason"] = assignment_reasons.get(component_key, "absent")
        data["source_tasks"] = source_tasks.get(component_key, [])
        score_dicts.append(data)

    scores_json_path = output_dir / "component_scores.json"
    scores_json_path.write_text(json.dumps(score_dicts, indent=2, ensure_ascii=False), encoding="utf-8")
    scores_pt_path = output_dir / "component_scores.pt"
    torch.save(score_dicts, scores_pt_path)

    rank_pattern_path = output_dir / "rank_pattern.json"
    rank_pattern_path.write_text(json.dumps(rank_pattern, indent=2, ensure_ascii=False), encoding="utf-8")
    lora_allocation_path = output_dir / "lora_allocation.json"
    lora_allocation_path.write_text(
        json.dumps(_json_safe(allocator.last_allocation), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path = output_dir / "summary.json"
    summary_data = dict(summary)
    summary_data.update(
        {
            "rank_score_source": rank_score_source,
            "lora_allocation": allocator.last_allocation,
            "mask_builder": mask_summary,
            "assignment_reasons": assignment_reasons,
            "source_tasks": source_tasks,
        }
    )
    summary_path.write_text(json.dumps(_json_safe(summary_data), indent=2, ensure_ascii=False), encoding="utf-8")

    paths = {
        "component_scores_json": scores_json_path,
        "component_scores_pt": scores_pt_path,
        "rank_pattern": rank_pattern_path,
        "lora_allocation": lora_allocation_path,
        "summary": summary_path,
    }
    if component_mask_path is not None:
        paths["component_mask"] = component_mask_path
    return paths


def load_task_artifact(task_name: str, task_dir: Path) -> TaskArtifact:
    score_path = task_dir / "logical_component_scores.json"
    if not score_path.exists():
        score_path = task_dir / "component_scores.json"
    logical_edges_path = task_dir / "logical_edges.json"
    summary_path = task_dir / "summary.json"
    if not score_path.exists():
        raise FileNotFoundError(f"Missing component score file under {task_dir}")
    if not logical_edges_path.exists():
        raise FileNotFoundError(f"Missing logical_edges.json under {task_dir}")

    score_data = json.loads(score_path.read_text(encoding="utf-8"))
    logical_edges = json.loads(logical_edges_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    graph_metadata = summary.get("graph_registry", {}) if isinstance(summary, dict) else {}

    scores = [_component_score_from_dict(item) for item in score_data]
    module_assignment = build_module_assignment(
        logical_edges=logical_edges,
        component_scores=scores,
        graph_metadata=graph_metadata,
    )
    rank_map = _raw_rank_map(scores)
    return TaskArtifact(
        task_name=task_name,
        component_scores=scores,
        module_assignment=module_assignment,
        module_rank=rank_map,
    )


def build_module_assignment(
    logical_edges: list[dict],
    component_scores: list[ComponentScore] | None = None,
    graph_metadata: dict | None = None,
) -> dict[str, tuple[int | None, str]]:
    endpoint_assignments = _build_endpoint_assignment(logical_edges)
    if component_scores is not None:
        graph_metadata = graph_metadata or {}
        assignments: dict[str, tuple[int | None, str]] = {}
        for score in component_scores:
            component_values = [
                endpoint_assignments[key]
                for key in _component_assignment_endpoint_keys(score, graph_metadata)
                if key in endpoint_assignments
            ]
            assignments[score.component_name] = _merge_assignments(component_values)
        return assignments
    return _legacy_destination_assignment(logical_edges)


def _legacy_destination_assignment(logical_edges: list[dict]) -> dict[str, tuple[int | None, str]]:
    assignments: dict[str, tuple[int | None, str]] = {}
    grouped_gates: dict[str, list[str]] = defaultdict(list)
    grouped_assignments: dict[str, list[str]] = defaultdict(list)
    for edge in logical_edges:
        module_name = edge.get("destination_module")
        if not module_name:
            continue
        logical_gate = edge.get("logical_gate")
        logical_assignment = edge.get("logical_assignment")
        if logical_gate:
            grouped_gates[module_name].append(str(logical_gate).upper())
        if logical_assignment:
            grouped_assignments[module_name].append(str(logical_assignment))

    all_modules = set(grouped_gates) | set(grouped_assignments)
    for module_name in all_modules:
        gates = grouped_gates.get(module_name, [])
        if gates:
            # OR is an exclusion signal; AND/ADDER are positive-path signals.
            has_positive_gate = any(gate in {"AND", "ADDER"} for gate in gates)
            has_or_gate = any(gate == "OR" for gate in gates)
            if has_positive_gate:
                assignments[module_name] = (1, "and_or_adder_path")
            elif has_or_gate:
                assignments[module_name] = (0, "or_gate_excluded")
            else:
                assignments[module_name] = (None, "absent")
            continue

        # Backward compatibility for historical outputs without logical_gate.
        edge_assignments = grouped_assignments.get(module_name, [])
        has_positive = any(item in {"positive_path", "shared_path"} for item in edge_assignments)
        has_or = any(item == "or_path" for item in edge_assignments)
        if has_positive:
            assignments[module_name] = (1, "positive_path_legacy")
        elif has_or:
            assignments[module_name] = (0, "or_gate_excluded_legacy")
        else:
            assignments[module_name] = (None, "absent")
    return assignments


def _build_endpoint_assignment(logical_edges: list[dict]) -> dict[str, tuple[int | None, str]]:
    grouped_gates: dict[str, list[str]] = defaultdict(list)
    grouped_assignments: dict[str, list[str]] = defaultdict(list)
    for edge in logical_edges:
        logical_gate = edge.get("logical_gate")
        logical_assignment = edge.get("logical_assignment")
        for key in _logical_edge_endpoint_keys(edge):
            if logical_gate:
                grouped_gates[key].append(str(logical_gate).upper())
            if logical_assignment:
                grouped_assignments[key].append(str(logical_assignment))
    return {
        key: _assignment_from_gate_and_path_values(
            gates=grouped_gates.get(key, []),
            edge_assignments=grouped_assignments.get(key, []),
        )
        for key in sorted(set(grouped_gates) | set(grouped_assignments))
    }


def _logical_edge_endpoint_keys(edge: dict) -> list[str]:
    keys = []
    source_node = edge.get("source_node")
    destination_module = edge.get("destination_module")
    if source_node:
        keys.append(f"source:{source_node}")
    if destination_module:
        keys.append(f"dest:{destination_module}")
    return keys


def _assignment_from_gate_and_path_values(
    gates: list[str],
    edge_assignments: list[str],
) -> tuple[int | None, str]:
    if gates:
        has_positive_gate = any(gate in {"AND", "ADDER"} for gate in gates)
        has_or_gate = any(gate == "OR" for gate in gates)
        if has_positive_gate:
            return (1, "and_or_adder_path")
        if has_or_gate:
            return (0, "or_gate_excluded")
        return (None, "absent")
    has_positive = any(item in {"positive_path", "shared_path"} for item in edge_assignments)
    has_or = any(item == "or_path" for item in edge_assignments)
    if has_positive:
        return (1, "positive_path_legacy")
    if has_or:
        return (0, "or_gate_excluded_legacy")
    return (None, "absent")


def _merge_assignments(values: list[tuple[int | None, str]]) -> tuple[int | None, str]:
    present_values = [value for value, _reason in values if value is not None]
    if 1 in present_values:
        return (1, "and_or_adder_path")
    if 0 in present_values:
        return (0, "or_gate_excluded")
    return (None, "absent")


def _component_assignment_endpoint_keys(score: ComponentScore, graph_metadata: dict) -> list[str]:
    layer_idx = int(score.layer_idx)
    component_type = str(score.component_type)
    n_heads = int(graph_metadata.get("n_heads", 0) or 0)
    num_key_value_heads = int(graph_metadata.get("num_key_value_heads", 0) or n_heads or 0)
    query_heads_per_kv_head = int(graph_metadata.get("query_heads_per_kv_head", 1) or 1)
    if component_type == "q_proj":
        graph_heads = _query_graph_heads(score.head_idx, n_heads)
        return [f"dest:a{layer_idx}.h{head_idx}<q>" for head_idx in graph_heads]
    if component_type in {"k_proj", "v_proj"}:
        graph_heads = _kv_graph_heads(
            head_idx=score.head_idx,
            n_heads=n_heads,
            num_key_value_heads=num_key_value_heads,
            query_heads_per_kv_head=query_heads_per_kv_head,
        )
        qkv = "k" if component_type == "k_proj" else "v"
        return [f"dest:a{layer_idx}.h{head_idx}<{qkv}>" for head_idx in graph_heads]
    if component_type == "o_proj":
        graph_heads = _query_graph_heads(score.head_idx, n_heads)
        return [f"source:a{layer_idx}.h{head_idx}" for head_idx in graph_heads]
    if component_type in {"gate_proj", "up_proj"}:
        return [f"dest:m{layer_idx}"]
    if component_type == "down_proj":
        return [f"source:m{layer_idx}"]
    return [score.module_name]


def _query_graph_heads(head_idx: int | None, n_heads: int) -> list[int]:
    if head_idx is not None:
        return [int(head_idx)]
    return list(range(max(1, n_heads)))


def _kv_graph_heads(
    head_idx: int | None,
    n_heads: int,
    num_key_value_heads: int,
    query_heads_per_kv_head: int,
) -> list[int]:
    if head_idx is None:
        return list(range(max(1, n_heads)))
    kv_head_idx = int(head_idx)
    if num_key_value_heads and num_key_value_heads < n_heads:
        start = kv_head_idx * int(query_heads_per_kv_head)
        end = min(n_heads, start + int(query_heads_per_kv_head))
        return list(range(start, end))
    return [kv_head_idx]


def _component_score_from_dict(data: dict) -> ComponentScore:
    return ComponentScore(
        parameter_name=data["parameter_name"],
        module_name=data["module_name"],
        layer_idx=int(data.get("layer_idx", -1)),
        component_type=str(data.get("component_type", "unknown")),
        granularity=str(data.get("granularity", "projection_matrix")),
        score_token_mode=str(data.get("score_token_mode", "all_active")),
        head_idx=data.get("head_idx"),
        head_kind=data.get("head_kind"),
        row_slice=tuple(data["row_slice"]) if data.get("row_slice") is not None else None,
        col_slice=tuple(data["col_slice"]) if data.get("col_slice") is not None else None,
        raw_score=float(data["raw_score"]),
        abs_score=float(data.get("abs_score", abs(float(data["raw_score"])))),
        mean_score=float(data.get("mean_score", 0.0)),
        sqrt_numel_score=float(data.get("sqrt_numel_score", 0.0)),
        rank_score=float(data.get("rank_score", abs(float(data["raw_score"])))),
        shape=tuple(int(v) for v in data.get("shape", (1, 1))),
        numel=int(data.get("numel", 1)),
        token_count=int(data.get("token_count", 0)),
        element_count=int(data.get("element_count", 1)),
        localization_mode=str(data.get("localization_mode", "current")),
        current_score=(float(data["current_score"]) if data.get("current_score") is not None else None),
        future_directional_score_theta=(
            float(data["future_directional_score_theta"]) if data.get("future_directional_score_theta") is not None else None
        ),
        future_directional_score_theta_hat=(
            float(data["future_directional_score_theta_hat"])
            if data.get("future_directional_score_theta_hat") is not None
            else None
        ),
        future_correction=(float(data["future_correction"]) if data.get("future_correction") is not None else None),
        future_step_k=(float(data["future_step_k"]) if data.get("future_step_k") is not None else None),
        mean_raw_rank=(float(data["mean_raw_rank"]) if data.get("mean_raw_rank") is not None else None),
    )


def _raw_rank_map(scores: list[ComponentScore]) -> dict[str, int]:
    ranked = sorted(scores, key=lambda score: (-abs(float(score.raw_score)), score.component_name))
    return {score.component_name: rank for rank, score in enumerate(ranked, start=1)}


def _mean_rank_value(module_name: str, module_task_ranks: dict[str, dict[str, int]], task_names: list[str]) -> float | None:
    ranks = [module_task_ranks[module_name][task] for task in task_names if task in module_task_ranks[module_name]]
    if not ranks:
        return None
    return float(mean(ranks))


def _aggregate_scores(
    per_task_scores: list[ComponentScore],
    template: ComponentScore,
    mean_raw_rank: float | None,
) -> ComponentScore:
    raw_score = float(mean(score.raw_score for score in per_task_scores))
    abs_score = float(abs(raw_score))
    mean_score = float(mean(score.mean_score for score in per_task_scores))
    sqrt_numel_score = float(mean(score.sqrt_numel_score for score in per_task_scores))
    rank_score = float(abs(raw_score))
    return ComponentScore(
        parameter_name=template.parameter_name,
        module_name=template.module_name,
        layer_idx=template.layer_idx,
        component_type=template.component_type,
        granularity=template.granularity,
        score_token_mode=template.score_token_mode,
        head_idx=template.head_idx,
        head_kind=template.head_kind,
        row_slice=template.row_slice,
        col_slice=template.col_slice,
        raw_score=raw_score,
        abs_score=abs_score,
        mean_score=mean_score,
        sqrt_numel_score=sqrt_numel_score,
        rank_score=rank_score,
        shape=template.shape,
        numel=template.numel,
        token_count=int(sum(score.token_count for score in per_task_scores)),
        element_count=int(sum(max(1, score.element_count) for score in per_task_scores)),
        localization_mode=template.localization_mode,
        current_score=float(mean(score.current_score or 0.0 for score in per_task_scores)),
        mean_raw_rank=mean_raw_rank,
    )


def _apply_mean_rank_score(scores: list[ComponentScore]) -> None:
    component_count = len(scores)
    if component_count <= 0:
        return
    for score in scores:
        if score.mean_raw_rank is not None:
            score.rank_score = float(component_count + 1 - float(score.mean_raw_rank))


def _zero_like(template: ComponentScore) -> ComponentScore:
    return ComponentScore(
        parameter_name=template.parameter_name,
        module_name=template.module_name,
        layer_idx=template.layer_idx,
        component_type=template.component_type,
        granularity=template.granularity,
        score_token_mode=template.score_token_mode,
        head_idx=template.head_idx,
        head_kind=template.head_kind,
        row_slice=template.row_slice,
        col_slice=template.col_slice,
        raw_score=0.0,
        abs_score=0.0,
        mean_score=0.0,
        sqrt_numel_score=0.0,
        rank_score=0.0,
        shape=template.shape,
        numel=template.numel,
        token_count=0,
        element_count=max(1, template.numel),
        localization_mode=template.localization_mode,
        current_score=0.0,
    )


def _sort_key(score: ComponentScore):
    if score.mean_raw_rank is not None:
        return (float(score.mean_raw_rank), score.component_name)
    return (-float(score.rank_score), score.component_name)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def parse_task_dirs(raw: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            raise ValueError("Each task artifact must be in 'task_name=path' format")
        name, path = stripped.split("=", 1)
        mapping[name.strip()] = Path(path.strip())
    if not mapping:
        raise ValueError("No task artifact directories provided")
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze multi-task logical circuit conflicts and export component artifacts.")
    parser.add_argument("--task_artifact_dirs", required=True, help="Comma-separated task_name=artifact_dir pairs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_rank", type=int, default=0)
    parser.add_argument("--max_rank", type=int, default=32)
    parser.add_argument("--rank_budget", type=int, default=None)
    parser.add_argument("--rank_multiple", type=int, default=1)
    parser.add_argument("--head_to_matrix_aggregation", choices=["mean", "max", "sum"], default="mean")
    parser.add_argument(
        "--rank_score_source",
        choices=["normalized_abs", "rank_score", "raw_abs", "sum_abs", "mean_abs", "sqrt_numel_abs"],
        default="normalized_abs",
    )
    parser.add_argument("--mask_fill_strategy", choices=["random", "magnitude", "first"], default="random")
    parser.add_argument("--mask_seed", type=int, default=0)
    parser.add_argument("--mask_min_keep_ratio", type=float, default=0.0, help="Lowest-ranked component keep ratio in [0, 1].")
    parser.add_argument("--mask_max_keep_ratio", type=float, default=1.0, help="Highest-ranked component keep ratio in [0, 1].")
    parser.add_argument(
        "--write_dense_masks",
        action="store_true",
        help="Write dense component_mask.pt files. Disabled by default because derived conflict artifacts can be multi-GB.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    task_dirs = parse_task_dirs(args.task_artifact_dirs)
    paths = run_conflict_analysis(
        task_dirs=task_dirs,
        output_dir=Path(args.output_dir),
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        rank_budget=args.rank_budget,
        rank_multiple=args.rank_multiple,
        head_to_matrix_aggregation=args.head_to_matrix_aggregation,
        rank_score_source=args.rank_score_source,
        mask_fill_strategy=args.mask_fill_strategy,
        mask_seed=args.mask_seed,
        mask_min_keep_ratio=args.mask_min_keep_ratio,
        mask_max_keep_ratio=args.mask_max_keep_ratio,
        write_dense_masks=args.write_dense_masks,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
