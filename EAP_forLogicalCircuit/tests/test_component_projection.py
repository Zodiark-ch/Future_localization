from __future__ import annotations

from EAP_forLogicalCircuit.circuit_builder import CircuitEdge
from EAP_forLogicalCircuit.component_projection import project_to_component_scores


def test_component_projection_average_of_circuit_and_or():
    module_metadata = {
        "layers.0.mlp.up_proj": {
            "parameter_name": "model.layers.0.mlp.up_proj.weight",
            "layer_idx": 0,
            "component_type": "up_proj",
            "shape": (2, 3),
            "numel": 6,
            "edge_count": 2,
        }
    }
    circuit_edges = [
        CircuitEdge(
            edge_id="e1",
            source_node="residual_stream",
            destination_module="layers.0.mlp.up_proj",
            layer_idx=0,
            component_type="up_proj",
            score=10.0,
            abs_score=10.0,
            selected_score=10.0,
            selection_reason="circuit:topn=1",
        )
    ]
    circuit_or_edges = [
        CircuitEdge(
            edge_id="e2",
            source_node="residual_stream",
            destination_module="layers.0.mlp.up_proj",
            layer_idx=0,
            component_type="up_proj",
            score=2.0,
            abs_score=2.0,
            selected_score=2.0,
            selection_reason="circuit_or:topn=1",
        )
    ]
    scores = project_to_component_scores(
        circuit_edges=circuit_edges,
        circuit_or_edges=circuit_or_edges,
        module_metadata=module_metadata,
        score_token_mode="all_active",
        score_normalization="sum",
        localization_mode="current",
    )
    assert len(scores) == 1
    score = scores[0]
    assert abs(score.raw_score - 6.0) < 1e-9
    assert abs(score.rank_score - 6.0) < 1e-9
    assert score.parameter_name == "model.layers.0.mlp.up_proj.weight"
