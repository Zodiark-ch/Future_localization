from __future__ import annotations

from EAP_forLogicalCircuit.circuit_builder import CircuitEdge, select_circuit_edges
from EAP_forLogicalCircuit.logical_fusion import fuse_logical_edges
from EAP_forLogicalCircuit.schemas import EdgeScore


def _edge(edge_id: str, raw_score: float) -> EdgeScore:
    return EdgeScore(
        edge_id=edge_id,
        source_node="residual_stream",
        destination_module=edge_id,
        layer_idx=0,
        component_type="up_proj",
        score_token_mode="all_active",
        localization_mode="current",
        raw_score=raw_score,
        abs_score=abs(raw_score),
        mean_score=raw_score,
        sqrt_numel_score=raw_score,
        rank_score=abs(raw_score),
        token_count=1,
        element_count=1,
        numel=1,
    )


def test_select_circuit_edges_topn():
    edges = [_edge("e1", 3.0), _edge("e2", 1.0), _edge("e3", 2.0)]
    selected = select_circuit_edges(
        edge_scores=edges,
        edge_topn=2,
        edge_threshold=None,
        edge_score_abs=True,
        circuit_name="circuit",
    )
    assert [item.edge_id for item in selected] == ["e1", "e3"]


def test_logical_fusion_assignments():
    edges = [_edge("shared", 4.0), _edge("positive", 3.0), _edge("or_only", -5.0)]
    circuit = select_circuit_edges(
        edges,
        edge_topn=None,
        edge_threshold=3.0,
        edge_score_abs=False,
        circuit_name="circuit",
    )
    circuit_or = select_circuit_edges(
        edges,
        edge_topn=None,
        edge_threshold=4.0,
        edge_score_abs=True,
        circuit_name="circuit_or",
    )
    fused = fuse_logical_edges(circuit_edges=circuit, circuit_or_edges=circuit_or)
    by_id = {item.edge_id: item.logical_assignment for item in fused}
    assert by_id["shared"] == "shared_path"
    assert by_id["positive"] == "positive_path"
    assert by_id["or_only"] == "or_path"


def test_logical_fusion_gate_labels():
    # Destination fan-in based labeling follows EAP-IG get_logical_edge style rules.
    circuit = [
        CircuitEdge("and1", "src_a", "dst_and", 0, "up_proj", 1.0, 1.0, 1.0, "circuit"),
        CircuitEdge("and2", "src_b", "dst_and", 0, "up_proj", 1.0, 1.0, 1.0, "circuit"),
        CircuitEdge("shared", "src_c", "dst_shared", 0, "up_proj", 1.0, 1.0, 1.0, "circuit"),
    ]
    circuit_or = [
        CircuitEdge("or1", "src_x", "dst_or", 0, "up_proj", 1.0, 1.0, 1.0, "circuit_or"),
        CircuitEdge("or2", "src_y", "dst_or", 0, "up_proj", 1.0, 1.0, 1.0, "circuit_or"),
        CircuitEdge("shared", "src_c", "dst_shared", 0, "up_proj", 1.0, 1.0, 1.0, "circuit_or"),
    ]

    fused = fuse_logical_edges(circuit_edges=circuit, circuit_or_edges=circuit_or)
    by_id = {item.edge_id: item.logical_gate for item in fused}

    assert by_id["and1"] == "AND"
    assert by_id["and2"] == "AND"
    assert by_id["or1"] == "OR"
    assert by_id["or2"] == "OR"
    assert by_id["shared"] == "ADDER"


def test_logical_fusion_gate_labels_single_parent_fallback():
    # Minimal graph topology (one incoming edge per destination module) should still
    # expose AND/OR from set-difference edges instead of collapsing all to ADDER.
    circuit = [
        CircuitEdge("a1", "residual_stream", "m1", 0, "up_proj", 1.0, 1.0, 1.0, "circuit"),
        CircuitEdge("shared", "residual_stream", "m2", 0, "up_proj", 1.0, 1.0, 1.0, "circuit"),
    ]
    circuit_or = [
        CircuitEdge("o1", "residual_stream", "m3", 0, "up_proj", 1.0, 1.0, 1.0, "circuit_or"),
        CircuitEdge("shared", "residual_stream", "m2", 0, "up_proj", 1.0, 1.0, 1.0, "circuit_or"),
    ]

    fused = fuse_logical_edges(circuit_edges=circuit, circuit_or_edges=circuit_or)
    by_id = {item.edge_id: item.logical_gate for item in fused}
    assert by_id["a1"] == "AND"
    assert by_id["o1"] == "OR"
    assert by_id["shared"] == "ADDER"
