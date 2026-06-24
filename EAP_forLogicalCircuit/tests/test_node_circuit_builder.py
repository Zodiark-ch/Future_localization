from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from EAP_forComponent.schemas import ComponentScore
from EAP_forLogicalCircuit.graph_registry import build_graph_metadata_from_model
from EAP_forLogicalCircuit.node_circuit_builder import build_node_induced_circuit


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _Mlp()


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=2)
        self.model = _Inner()
        self.lm_head = nn.Linear(4, 10, bias=False)


def _score(layer_idx: int, component_type: str, head_idx: int | None, rank_score: float) -> ComponentScore:
    module_name = f"model.layers.{layer_idx}.self_attn.{component_type}" if component_type.endswith("_proj") else f"model.layers.{layer_idx}.mlp.{component_type}"
    return ComponentScore(
        parameter_name=f"{module_name}.weight",
        module_name=module_name,
        layer_idx=layer_idx,
        component_type=component_type,
        granularity="head" if head_idx is not None else "projection_matrix",
        score_token_mode="all_active",
        head_idx=head_idx,
        head_kind=None,
        row_slice=None,
        col_slice=None,
        raw_score=rank_score,
        abs_score=abs(rank_score),
        mean_score=rank_score,
        sqrt_numel_score=rank_score,
        rank_score=abs(rank_score),
        shape=(4, 4),
        numel=16,
        token_count=1,
        element_count=1,
    )


def test_node_induced_edges_follow_eap_ig_order():
    scores = [
        _score(0, "q_proj", 0, 10.0),
        _score(1, "q_proj", 0, 9.0),
        _score(1, "gate_proj", None, 8.0),
    ]
    circuit = build_node_induced_circuit(
        component_scores=scores,
        graph_metadata=build_graph_metadata_from_model(_Model(), ["q_proj", "k_proj", "v_proj", "o_proj"]),
        node_topn=3,
        circuit_name="circuit",
    )
    edges = circuit.edges
    summary = circuit.summary
    edge_ids = {edge.edge_id for edge in edges}

    assert summary["selected_nodes"] == ["a0.h0", "a1.h0", "m1"]
    assert "a0.h0->a1.h0<q>" in edge_ids
    assert "input->m1" in edge_ids
    assert "a1.h0->m1" in edge_ids
    assert "m1->logits" in edge_ids
    assert all(not edge_id.startswith("a1.h0->a0.") for edge_id in edge_ids)
    assert all(edge.score is None for edge in edges)


def test_node_topn_limits_selected_nodes():
    scores = [
        _score(0, "q_proj", 0, 10.0),
        _score(1, "q_proj", 0, 9.0),
    ]
    circuit = build_node_induced_circuit(
        component_scores=scores,
        graph_metadata=build_graph_metadata_from_model(_Model(), ["q_proj", "k_proj", "v_proj"]),
        node_topn=1,
        circuit_name="circuit",
    )
    summary = circuit.summary
    assert summary["selected_nodes"] == ["a0.h0"]
    assert summary["selected_node_count"] == 1
