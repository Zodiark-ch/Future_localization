from __future__ import annotations

import torch.nn as nn
from types import SimpleNamespace

from EAP_forLogicalCircuit.graph_registry import GraphRegistry, build_graph_metadata_from_model


class _DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class _DummyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _DummyAttention()
        self.mlp = _DummyMlp()


class _DummyInnerModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_DummyLayer() for _ in range(num_layers)])


class DummyModel(nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=2)
        self.model = _DummyInnerModel(num_layers=num_layers)


def test_graph_registry_dense_component_edges():
    model = DummyModel(num_layers=2)
    registry = GraphRegistry.from_model(
        model=model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    edges = registry.edge_targets()

    # EAP-IG non-parallel formula for L=2,H=2:
    # layer0 attention=3*H*1, layer0 mlp=1+H,
    # layer1 attention=3*H*(1+H+1), layer1 mlp=1+H+1+H, logits=1+L*(H+1).
    assert len(edges) == 46

    edge_ids = {edge.edge_id for edge in edges}
    assert "input->a0.h0<q>" in edge_ids
    assert "a0.h1->a1.h0<q>" in edge_ids
    assert "m0->a1.h1<v>" in edge_ids
    assert "a1.h1->m1" in edge_ids
    assert "m1->logits" in edge_ids

    component_types = {edge.component_type for edge in edges}
    assert component_types == {"q_proj", "k_proj", "v_proj", "mlp", "logits"}

    assert registry.metadata["graph_version"] == "eap_ig_head_graph_v1"
    assert registry.metadata["edge_count"] == 46
    assert registry.metadata["n_forward"] == 7
    assert registry.metadata["n_backward"] == 15

    lightweight_metadata = build_graph_metadata_from_model(
        model=model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    assert lightweight_metadata["edge_count"] == len(edges)
    assert lightweight_metadata["source_nodes"] == registry.metadata["source_nodes"]
    assert lightweight_metadata["destination_nodes"] == registry.metadata["destination_nodes"]
