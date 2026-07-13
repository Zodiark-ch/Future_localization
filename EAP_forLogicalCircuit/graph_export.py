from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from EAP_forComponent.schemas import ComponentScore
from EAP_forLogicalCircuit.circuit_builder import CircuitEdge
from EAP_forLogicalCircuit.node_circuit_builder import NodeInducedCircuit, build_node_induced_circuit
from EAP_forLogicalCircuit.schemas import EdgeScore


@dataclass
class CircuitGraphExport:
    edges: list[list[str]]
    edge_details: list[dict]
    summary: dict


@dataclass
class _GraphEdgeCandidate:
    order: int
    edge: CircuitEdge
    score: EdgeScore | None
    attribution_value: float


def build_circuit_graph_export(
    component_scores: list[ComponentScore],
    edge_scores: list[EdgeScore],
    graph_metadata: dict,
    node_topn: int = 25,
    edge_threshold_ratio: float = 0.1,
    edge_budget_multiplier: float = 3.0,
    input_edge_limit_ratio: float = 0.3,
) -> CircuitGraphExport:
    if node_topn <= 0:
        raise ValueError("node_topn must be > 0")
    if edge_threshold_ratio < 0:
        raise ValueError("edge_threshold_ratio must be >= 0")
    if edge_budget_multiplier < 0:
        raise ValueError("edge_budget_multiplier must be >= 0")
    if input_edge_limit_ratio < 0:
        raise ValueError("input_edge_limit_ratio must be >= 0")
    if not edge_scores:
        raise ValueError("edge_scores are required to export a circuit graph")

    node_circuit = build_node_induced_circuit(
        component_scores=component_scores,
        graph_metadata=graph_metadata,
        node_topn=node_topn,
        circuit_name="circuit_graph",
    )
    score_by_edge_id = {score.edge_id: score for score in edge_scores}
    candidates: list[_GraphEdgeCandidate] = []
    missing_edge_score_count = 0
    for order, edge in enumerate(node_circuit.edges):
        score = score_by_edge_id.get(edge.edge_id)
        if score is None:
            missing_edge_score_count += 1
        candidates.append(
            _GraphEdgeCandidate(
                order=order,
                edge=edge,
                score=score,
                attribution_value=_edge_attribution_value(score),
            )
        )

    selected_by_edge_id: dict[str, _GraphEdgeCandidate] = {}
    selection_reasons_by_edge_id: dict[str, set[str]] = {}
    connectivity_summaries = []
    missing_required_connectivity = []

    def select_candidate(candidate: _GraphEdgeCandidate, reason: str) -> None:
        selected_by_edge_id[candidate.edge.edge_id] = candidate
        selection_reasons_by_edge_id.setdefault(candidate.edge.edge_id, set()).add(reason)

    for node in node_circuit.selected_nodes:
        incoming_candidates = [
            candidate
            for candidate in candidates
            if _destination_base_node(candidate.edge.destination_module) == node.node_id
        ]
        outgoing_candidates = [candidate for candidate in candidates if candidate.edge.source_node == node.node_id]
        incoming_required = max(incoming_candidates, key=_candidate_rank_key) if incoming_candidates else None
        outgoing_required = max(outgoing_candidates, key=_candidate_rank_key) if outgoing_candidates else None
        if incoming_required is None:
            missing_required_connectivity.append({"node_id": node.node_id, "direction": "incoming"})
        else:
            select_candidate(incoming_required, "required_incoming")
        if outgoing_required is None:
            missing_required_connectivity.append({"node_id": node.node_id, "direction": "outgoing"})
        else:
            select_candidate(outgoing_required, "required_outgoing")
        connectivity_summaries.append(
            {
                "node_id": node.node_id,
                "candidate_incoming_edge_count": len(incoming_candidates),
                "candidate_outgoing_edge_count": len(outgoing_candidates),
                "required_incoming_edge_id": incoming_required.edge.edge_id if incoming_required else None,
                "required_outgoing_edge_id": outgoing_required.edge.edge_id if outgoing_required else None,
            }
        )

    required_edge_count = len(selected_by_edge_id)
    graph_node_count = len(node_circuit.selected_nodes) + 2
    edge_budget = int(math.ceil(float(edge_budget_multiplier) * float(graph_node_count)))
    budget_selected_edge_count = 0
    if len(selected_by_edge_id) < edge_budget:
        for candidate in sorted(candidates, key=_candidate_sort_key):
            if candidate.edge.edge_id in selected_by_edge_id:
                continue
            select_candidate(candidate, "attribution_budget")
            budget_selected_edge_count += 1
            if len(selected_by_edge_id) >= edge_budget:
                break

    selected_candidates = _deduplicate_selected_candidates(
        candidates=selected_by_edge_id.values(),
        selection_reasons_by_edge_id=selection_reasons_by_edge_id,
    )
    pre_input_limit_selected_edge_count = len(selected_candidates)
    selected_candidates, input_edge_limit_summary = _limit_input_outgoing_candidates(
        candidates=selected_candidates,
        graph_node_count=graph_node_count,
        input_edge_limit_ratio=input_edge_limit_ratio,
        selected_node_ids={node.node_id for node in node_circuit.selected_nodes},
    )
    graph_edges = [
        [_transform_src_node(candidate.edge.source_node), _transform_dst_node(candidate.edge.destination_module)]
        for candidate in selected_candidates
    ]
    edge_details = [
        _edge_detail(candidate, sorted(selection_reasons_by_edge_id.get(candidate.edge.edge_id, [])))
        for candidate in selected_candidates
    ]
    summary = _build_summary(
        node_circuit=node_circuit,
        node_topn=node_topn,
        edge_budget_multiplier=edge_budget_multiplier,
        graph_node_count=graph_node_count,
        edge_budget=edge_budget,
        edge_threshold_ratio=edge_threshold_ratio,
        candidate_edge_count=len(node_circuit.edges),
        selected_edge_count=len(selected_candidates),
        pre_dedup_selected_edge_count=len(selected_by_edge_id),
        pre_input_limit_selected_edge_count=pre_input_limit_selected_edge_count,
        input_edge_limit_summary=input_edge_limit_summary,
        required_edge_count=required_edge_count,
        budget_selected_edge_count=budget_selected_edge_count,
        missing_edge_score_count=missing_edge_score_count,
        missing_required_connectivity=missing_required_connectivity,
        connectivity_summaries=connectivity_summaries,
    )
    return CircuitGraphExport(edges=graph_edges, edge_details=edge_details, summary=summary)


def save_circuit_graph_export(
    output_dir: str | Path,
    component_scores: list[ComponentScore],
    edge_scores: list[EdgeScore],
    graph_metadata: dict,
    node_topn: int = 25,
    edge_threshold_ratio: float = 0.1,
    edge_budget_multiplier: float = 3.0,
    input_edge_limit_ratio: float = 0.3,
) -> tuple[dict[str, Path], dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export = build_circuit_graph_export(
        component_scores=component_scores,
        edge_scores=edge_scores,
        graph_metadata=graph_metadata,
        node_topn=node_topn,
        edge_threshold_ratio=edge_threshold_ratio,
        edge_budget_multiplier=edge_budget_multiplier,
        input_edge_limit_ratio=input_edge_limit_ratio,
    )
    graph_path = output_dir / "graph.json"
    graph_edges_path = output_dir / "graph_edges.json"
    graph_summary_path = output_dir / "graph_summary.json"
    graph_dot_path = output_dir / "graph.dot"
    graph_image_path = output_dir / "graph.jpg"
    graph_dot_path.write_text(_build_graph_dot(export.edge_details), encoding="utf-8")
    image_summary = _render_graph_image(graph_dot_path, graph_image_path)
    summary = dict(export.summary)
    summary["graph_dot_path"] = str(graph_dot_path)
    summary["graph_image"] = image_summary
    graph_path.write_text(json.dumps(export.edges, indent=4, ensure_ascii=False), encoding="utf-8")
    graph_edges_path.write_text(json.dumps(export.edge_details, indent=2, ensure_ascii=False), encoding="utf-8")
    graph_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths = {
        "graph": graph_path,
        "graph_edges": graph_edges_path,
        "graph_summary": graph_summary_path,
        "graph_dot": graph_dot_path,
    }
    if image_summary["status"] == "written":
        paths["graph_image"] = graph_image_path
    return paths, summary


def _build_graph_dot(edge_details: list[dict]) -> str:
    nodes = _ordered_visual_nodes(
        {
            _visual_source_node(detail)
            for detail in edge_details
        }
        | {
            _visual_destination_node(detail)
            for detail in edge_details
        }
    )
    lines = [
        "digraph circuit_graph {",
        '  graph [rankdir="TB", bgcolor="white", overlap="false", splines="true"];',
        '  node [shape="box", style="filled, rounded", fontname="Helvetica", color="black"];',
        '  edge [fontname="Helvetica", arrowsize="0.7", penwidth="1.0"];',
    ]
    for index, node in enumerate(nodes):
        lines.append(
            f"  {_dot_quote(node)} [label={_dot_quote(node)}, fillcolor={_dot_quote(_node_fillcolor(index))}];"
        )
    for detail in edge_details:
        lines.append(
            "  "
            f"{_dot_quote(_visual_source_node(detail))} -> {_dot_quote(_visual_destination_node(detail))} "
            f"[color={_dot_quote(_edge_color(detail))}];"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _deduplicate_selected_candidates(
    candidates,
    selection_reasons_by_edge_id: dict[str, set[str]],
) -> list[_GraphEdgeCandidate]:
    best_by_pair: dict[tuple[str, str], _GraphEdgeCandidate] = {}
    reasons_by_pair: dict[tuple[str, str], set[str]] = {}
    for candidate in candidates:
        pair_key = _visual_edge_key(candidate)
        reasons_by_pair.setdefault(pair_key, set()).update(
            selection_reasons_by_edge_id.get(candidate.edge.edge_id, set())
        )
        existing = best_by_pair.get(pair_key)
        if existing is None or _candidate_rank_key(candidate) > _candidate_rank_key(existing):
            best_by_pair[pair_key] = candidate
    for pair_key, candidate in best_by_pair.items():
        selection_reasons_by_edge_id[candidate.edge.edge_id] = reasons_by_pair[pair_key]
    return sorted(best_by_pair.values(), key=lambda candidate: candidate.order)


def _visual_edge_key(candidate: _GraphEdgeCandidate) -> tuple[str, str]:
    return (
        _transform_src_node(candidate.edge.source_node),
        _visual_destination_name(candidate.edge.destination_module),
    )


def _limit_input_outgoing_candidates(
    candidates: list[_GraphEdgeCandidate],
    graph_node_count: int,
    input_edge_limit_ratio: float,
    selected_node_ids: set[str],
) -> tuple[list[_GraphEdgeCandidate], dict]:
    input_candidates = [candidate for candidate in candidates if _transform_src_node(candidate.edge.source_node) == "input"]
    input_edge_limit = int(math.floor(float(input_edge_limit_ratio) * float(graph_node_count)))
    non_input_incoming_nodes = {
        _destination_base_node(candidate.edge.destination_module)
        for candidate in candidates
        if _transform_src_node(candidate.edge.source_node) != "input"
    }
    required_input_candidates = [
        candidate
        for candidate in input_candidates
        if _destination_base_node(candidate.edge.destination_module) in selected_node_ids
        and _destination_base_node(candidate.edge.destination_module) not in non_input_incoming_nodes
    ]
    required_input_edge_ids = {candidate.edge.edge_id for candidate in required_input_candidates}
    if len(input_candidates) <= input_edge_limit:
        return (
            sorted(candidates, key=lambda candidate: candidate.order),
            {
                "input_edge_limit_ratio": float(input_edge_limit_ratio),
                "input_edge_limit": int(input_edge_limit),
                "input_edge_count_before_limit": len(input_candidates),
                "input_edge_count": len(input_candidates),
                "required_input_edge_count": len(required_input_candidates),
                "input_edge_limit_exceeded_by_required_edges": max(
                    0,
                    len(required_input_candidates) - int(input_edge_limit),
                ),
                "removed_input_edge_count": 0,
                "removed_input_edge_ids_preview": [],
            },
        )
    remaining_input_slots = max(0, int(input_edge_limit) - len(required_input_edge_ids))
    optional_input_candidates = [
        candidate for candidate in input_candidates if candidate.edge.edge_id not in required_input_edge_ids
    ]
    kept_input_edge_ids = {
        candidate.edge.edge_id
        for candidate in sorted(optional_input_candidates, key=_candidate_sort_key)[:remaining_input_slots]
    }
    kept_input_edge_ids.update(required_input_edge_ids)
    removed_input_candidates = [
        candidate for candidate in input_candidates if candidate.edge.edge_id not in kept_input_edge_ids
    ]
    filtered_candidates = [
        candidate
        for candidate in candidates
        if _transform_src_node(candidate.edge.source_node) != "input" or candidate.edge.edge_id in kept_input_edge_ids
    ]
    return (
        sorted(filtered_candidates, key=lambda candidate: candidate.order),
        {
            "input_edge_limit_ratio": float(input_edge_limit_ratio),
            "input_edge_limit": int(input_edge_limit),
            "input_edge_count_before_limit": len(input_candidates),
            "input_edge_count": len(kept_input_edge_ids),
            "required_input_edge_count": len(required_input_edge_ids),
            "input_edge_limit_exceeded_by_required_edges": max(
                0,
                len(required_input_edge_ids) - int(input_edge_limit),
            ),
            "removed_input_edge_count": len(removed_input_candidates),
            "removed_input_edge_ids_preview": [candidate.edge.edge_id for candidate in removed_input_candidates[:20]],
        },
    )


def _render_graph_image(graph_dot_path: Path, graph_image_path: Path) -> dict:
    dot_executable = shutil.which("dot")
    if dot_executable is None:
        graph_image_path.unlink(missing_ok=True)
        return {
            "status": "skipped",
            "path": str(graph_image_path),
            "renderer": "graphviz_dot",
            "reason": "Graphviz 'dot' executable was not found; graph.dot was saved for later rendering.",
        }
    result = subprocess.run(
        [dot_executable, "-Tjpg", str(graph_dot_path), "-o", str(graph_image_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        graph_image_path.unlink(missing_ok=True)
        return {
            "status": "failed",
            "path": str(graph_image_path),
            "renderer": "graphviz_dot",
            "returncode": result.returncode,
            "reason": (result.stderr or result.stdout or "Graphviz dot failed without output.").strip(),
        }
    return {
        "status": "written",
        "path": str(graph_image_path),
        "renderer": "graphviz_dot",
    }


def _visual_source_node(detail: dict) -> str:
    return str(detail["export_source_node"])


def _visual_destination_node(detail: dict) -> str:
    return _visual_destination_name(str(detail["destination_node"]))


def _visual_destination_name(destination_node: str) -> str:
    return _destination_base_node(destination_node)


def _ordered_visual_nodes(nodes: set[str]) -> list[str]:
    return sorted(nodes, key=_visual_node_sort_key)


def _visual_node_sort_key(node: str) -> tuple[int, int, int, str]:
    if node == "input":
        return (-1, -1, -1, node)
    if node == "logits":
        return (10**9, 10**9, 10**9, node)
    attention_match = re.fullmatch(r"a(\d+)\.h(\d+)", node)
    if attention_match:
        return (int(attention_match.group(1)), 0, int(attention_match.group(2)), node)
    mlp_match = re.fullmatch(r"m(\d+)", node)
    if mlp_match:
        return (int(mlp_match.group(1)), 1, 0, node)
    return (10**8, 10**8, 10**8, node)


def _node_fillcolor(index: int) -> str:
    pastel2 = (
        "#b3e2cd",
        "#fdcdac",
        "#cbd5e8",
        "#f4cae4",
        "#e6f5c9",
        "#fff2ae",
        "#f1e2cc",
        "#cccccc",
    )
    return pastel2[index % len(pastel2)]


def _edge_color(detail: dict) -> str:
    destination_node = str(detail["destination_node"])
    qkv_match = re.search(r"<(q|k|v)>$", destination_node)
    if qkv_match:
        return {"q": "#FF00FF", "k": "#00FF00", "v": "#0000FF"}[qkv_match.group(1)]
    raw_score = detail.get("raw_score")
    if raw_score is not None and float(raw_score) < 0:
        return "#FF0000"
    return "#000000"


def _dot_quote(value: str) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _edge_attribution_value(score: EdgeScore | None) -> float:
    if score is None:
        return 0.0
    return float(score.rank_score)


def _candidate_rank_key(candidate: _GraphEdgeCandidate) -> tuple[float, float, str]:
    raw_abs = abs(float(candidate.score.raw_score)) if candidate.score is not None else 0.0
    return (candidate.attribution_value, raw_abs, candidate.edge.edge_id)


def _candidate_sort_key(candidate: _GraphEdgeCandidate) -> tuple[float, float, str]:
    raw_abs = abs(float(candidate.score.raw_score)) if candidate.score is not None else 0.0
    return (-float(candidate.attribution_value), -raw_abs, candidate.edge.edge_id)


def _destination_base_node(destination_node: str) -> str:
    if destination_node.startswith("a") and destination_node.endswith(">") and "<" in destination_node:
        return destination_node.split("<", 1)[0]
    return destination_node


def _transform_src_node(name: str) -> str:
    return name


def _transform_dst_node(name: str) -> str:
    return re.sub(r"<(q|k|v)>", r".\1", name)


def _edge_detail(candidate: _GraphEdgeCandidate, selection_reasons: list[str]) -> dict:
    score = candidate.score
    detail = {
        "edge_id": candidate.edge.edge_id,
        "source_node": candidate.edge.source_node,
        "destination_node": candidate.edge.destination_module,
        "export_source_node": _transform_src_node(candidate.edge.source_node),
        "export_destination_node": _transform_dst_node(candidate.edge.destination_module),
        "destination_base_node": _destination_base_node(candidate.edge.destination_module),
        "attribution_value": candidate.attribution_value,
        "selection_reasons": selection_reasons,
    }
    if score is not None:
        detail.update(
            {
                "raw_score": float(score.raw_score),
                "abs_score": float(score.abs_score),
                "rank_score": float(score.rank_score),
                "mean_score": float(score.mean_score),
                "sqrt_numel_score": float(score.sqrt_numel_score),
            }
        )
    return detail


def _build_summary(
    node_circuit: NodeInducedCircuit,
    node_topn: int,
    edge_budget_multiplier: float,
    graph_node_count: int,
    edge_budget: int,
    edge_threshold_ratio: float,
    candidate_edge_count: int,
    selected_edge_count: int,
    pre_dedup_selected_edge_count: int,
    pre_input_limit_selected_edge_count: int,
    input_edge_limit_summary: dict,
    required_edge_count: int,
    budget_selected_edge_count: int,
    missing_edge_score_count: int,
    missing_required_connectivity: list[dict],
    connectivity_summaries: list[dict],
) -> dict:
    return {
        "graph_kind": "circuit",
        "format": "dense_graph_json_edges",
        "node_topn": int(node_topn),
        "graph_node_count_including_input_logits": int(graph_node_count),
        "edge_budget_multiplier": float(edge_budget_multiplier),
        "edge_budget": int(edge_budget),
        "legacy_edge_threshold_ratio": float(edge_threshold_ratio),
        "edge_selection_rule": "for each selected nonterminal node, keep the highest-attribution incoming and outgoing candidate edge; then fill remaining edge budget by attribution rank_score",
        "selected_node_count": len(node_circuit.selected_nodes),
        "selected_nodes": [
            {
                "node_id": node.node_id,
                "raw_score": float(node.raw_score),
                "rank_score": float(node.rank_score),
                "component_count": int(node.component_count),
            }
            for node in node_circuit.selected_nodes
        ],
        "candidate_edge_count": int(candidate_edge_count),
        "pre_dedup_selected_edge_count": int(pre_dedup_selected_edge_count),
        "pre_input_limit_selected_edge_count": int(pre_input_limit_selected_edge_count),
        "selected_edge_count": int(selected_edge_count),
        "deduplicated_edge_count": max(
            0,
            int(pre_dedup_selected_edge_count) - int(pre_input_limit_selected_edge_count),
        ),
        "input_edge_limit": input_edge_limit_summary,
        "required_edge_count": int(required_edge_count),
        "budget_selected_edge_count": int(budget_selected_edge_count),
        "edge_budget_exceeded_by_required_edges": max(0, int(required_edge_count) - int(edge_budget)),
        "missing_edge_score_count": int(missing_edge_score_count),
        "required_connectivity_rule": "input is exempt from incoming-edge requirement; logits is exempt from outgoing-edge requirement; every selected circuit node needs at least one selected incoming and outgoing edge when candidates exist",
        "missing_required_connectivity_count": len(missing_required_connectivity),
        "missing_required_connectivity": missing_required_connectivity,
        "node_induced_summary": node_circuit.summary,
        "node_connectivity_summaries": connectivity_summaries,
    }