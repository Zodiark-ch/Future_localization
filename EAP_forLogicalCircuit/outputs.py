from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from EAP_forLogicalCircuit.circuit_builder import CircuitEdge
from EAP_forLogicalCircuit.logical_fusion import LogicalEdge
from EAP_forLogicalCircuit.schemas import EdgeScore
from EAP_forComponent.outputs import component_score_to_dict
from EAP_forComponent.schemas import ComponentScore


def save_outputs(
    output_dir: str | Path,
    edge_scores: list[EdgeScore] | None,
    circuit_edges: list[CircuitEdge],
    circuit_or_edges: list[CircuitEdge],
    logical_edges: list[LogicalEdge],
    logical_component_scores: list[ComponentScore],
    rank_pattern: dict[str, int],
    lora_allocation: dict,
    component_mask: dict[str, torch.Tensor],
    summary: dict,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    edge_scores_json = output_dir / "edge_scores.json"
    edge_scores_pt = output_dir / "edge_scores.pt"
    if edge_scores is None:
        edge_scores_json.unlink(missing_ok=True)
        edge_scores_pt.unlink(missing_ok=True)
    else:
        score_dicts = [edge_score_to_dict(score) for score in sorted(edge_scores, key=_score_sort_key)]
        edge_scores_json.write_text(
            json.dumps(score_dicts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        torch.save(score_dicts, edge_scores_pt)
        paths["edge_scores_json"] = edge_scores_json
        paths["edge_scores_pt"] = edge_scores_pt
    circuit_edges_path = output_dir / "circuit_edges.json"
    circuit_edges_path.write_text(
        json.dumps([circuit_edge_to_dict(edge) for edge in circuit_edges], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    circuit_or_edges_path = output_dir / "circuit_or_edges.json"
    circuit_or_edges_path.write_text(
        json.dumps([circuit_edge_to_dict(edge) for edge in circuit_or_edges], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logical_edges_path = output_dir / "logical_edges.json"
    logical_edges_path.write_text(
        json.dumps([logical_edge_to_dict(edge) for edge in logical_edges], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logical_component_score_dicts = [
        component_score_to_dict(score) for score in sorted(logical_component_scores, key=lambda x: -x.rank_score)
    ]
    logical_component_scores_json = output_dir / "logical_component_scores.json"
    logical_component_scores_json.write_text(
        json.dumps(logical_component_score_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logical_component_scores_pt = output_dir / "logical_component_scores.pt"
    torch.save(logical_component_score_dicts, logical_component_scores_pt)
    # Compatibility alias for downstream readers that expect component_scores.* names.
    component_scores_json = output_dir / "component_scores.json"
    component_scores_json.write_text(
        json.dumps(logical_component_score_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    component_scores_pt = output_dir / "component_scores.pt"
    torch.save(logical_component_score_dicts, component_scores_pt)
    rank_pattern_path = output_dir / "rank_pattern.json"
    rank_pattern_path.write_text(
        json.dumps(rank_pattern, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lora_allocation_path = output_dir / "lora_allocation.json"
    lora_allocation_path.write_text(
        json.dumps(_json_safe(lora_allocation), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    component_mask_path = output_dir / "component_mask.pt"
    torch.save(component_mask, component_mask_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths.update(
        {
            "circuit_edges": circuit_edges_path,
            "circuit_or_edges": circuit_or_edges_path,
            "logical_edges": logical_edges_path,
            "logical_component_scores_json": logical_component_scores_json,
            "logical_component_scores_pt": logical_component_scores_pt,
            "component_scores_json": component_scores_json,
            "component_scores_pt": component_scores_pt,
            "rank_pattern": rank_pattern_path,
            "lora_allocation": lora_allocation_path,
            "component_mask": component_mask_path,
            "summary": summary_path,
        }
    )
    return paths


def edge_score_to_dict(score: EdgeScore) -> dict:
    return asdict(score)


def circuit_edge_to_dict(edge: CircuitEdge) -> dict:
    data = asdict(edge)
    if edge.score is None:
        data.pop("score", None)
        data.pop("abs_score", None)
        data.pop("selected_score", None)
    return data


def logical_edge_to_dict(edge: LogicalEdge) -> dict:
    data = asdict(edge)
    if edge.circuit_score is None:
        data.pop("circuit_score", None)
    if edge.circuit_or_score is None:
        data.pop("circuit_or_score", None)
    return data


def _score_sort_key(score: EdgeScore):
    if score.mean_raw_rank is not None:
        return (float(score.mean_raw_rank), score.edge_id)
    return (-float(score.rank_score), score.edge_id)


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
