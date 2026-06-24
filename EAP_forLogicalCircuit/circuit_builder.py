from __future__ import annotations

from dataclasses import dataclass

from EAP_forLogicalCircuit.schemas import EdgeScore


@dataclass
class CircuitEdge:
    edge_id: str
    source_node: str
    destination_module: str
    layer_idx: int
    component_type: str
    score: float | None
    abs_score: float | None
    selected_score: float | None
    selection_reason: str


def select_circuit_edges(
    edge_scores: list[EdgeScore],
    edge_topn: int | None,
    edge_threshold: float | None,
    edge_score_abs: bool,
    circuit_name: str,
) -> list[CircuitEdge]:
    if edge_topn is not None and edge_topn <= 0:
        raise ValueError("edge_topn must be > 0 when provided")
    if edge_topn is None and edge_threshold is None:
        edge_topn = min(50, len(edge_scores))
    selected_scores = {
        score.edge_id: abs(score.raw_score) if edge_score_abs else score.raw_score
        for score in edge_scores
    }
    ranked = sorted(
        edge_scores,
        key=lambda score: (-float(selected_scores[score.edge_id]), score.edge_id),
    )
    selected: list[EdgeScore]
    if edge_threshold is not None:
        selected = [score for score in ranked if float(selected_scores[score.edge_id]) >= float(edge_threshold)]
        selection_reason = f"threshold>={edge_threshold}"
    else:
        assert edge_topn is not None
        selected = ranked[: int(edge_topn)]
        selection_reason = f"topn={int(edge_topn)}"
    return [
        CircuitEdge(
            edge_id=score.edge_id,
            source_node=score.source_node,
            destination_module=score.destination_module,
            layer_idx=score.layer_idx,
            component_type=score.component_type,
            score=float(score.raw_score),
            abs_score=float(abs(score.raw_score)),
            selected_score=float(selected_scores[score.edge_id]),
            selection_reason=f"{circuit_name}:{selection_reason}",
        )
        for score in selected
    ]
