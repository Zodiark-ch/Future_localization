from __future__ import annotations

import torch
from torch import nn

from EAP_forLogicalCircuit.current_localization import CurrentEdgeAttributionScorer
from EAP_forLogicalCircuit.graph_registry import EdgeTarget


def test_current_edge_score_nonzero_tiny():
    edge_target = EdgeTarget(
        edge_id="residual_stream->layers.0.mlp.up_proj.input",
        source_node="residual_stream",
        destination_module="layers.0.mlp.up_proj",
        destination_parameter_name="layers.0.mlp.up_proj.weight",
        layer_idx=0,
        component_type="up_proj",
        num_features=2,
        module=nn.Linear(2, 2, bias=False),
    )
    scorer = CurrentEdgeAttributionScorer(
        edge_targets=[edge_target],
        score_token_mode="all_active",
        score_normalization="sum",
    )
    clean_inputs = {
        edge_target.destination_parameter_name: torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
    }
    corrupted_inputs = {
        edge_target.destination_parameter_name: torch.zeros(1, 2, 2),
    }
    input_grads = {
        edge_target.destination_parameter_name: torch.ones(1, 2, 2),
    }
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)
    label_positions = torch.tensor([1], dtype=torch.long)

    scorer.score_batch(
        clean_inputs=clean_inputs,
        corrupted_inputs=corrupted_inputs,
        input_grads=input_grads,
        attention_mask=attention_mask,
        label_positions=label_positions,
    )
    scores = scorer.finalize()
    assert len(scores) == 1
    assert abs(scores[0].raw_score - 10.0) < 1e-6
    assert scores[0].rank_score > 0.0
