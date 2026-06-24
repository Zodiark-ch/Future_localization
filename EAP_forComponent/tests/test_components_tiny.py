from types import SimpleNamespace

from torch import nn

from EAP_forComponent.components import ComponentRegistry


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=1)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])
        self.lm_head = nn.Linear(4, 10, bias=False)


def test_component_registry_projection_matrix_counts_target_modules_only():
    registry = ComponentRegistry.from_model(
        TinyModel(),
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        attention_granularity="projection_matrix",
    )
    targets = registry.targets()
    assert len(targets) == 7
    assert all(not target.parameter_name.endswith("lm_head.weight") for target in targets)


def test_component_registry_head_mode_uses_query_and_kv_head_counts():
    registry = ComponentRegistry.from_model(
        TinyModel(),
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        attention_granularity="head",
    )
    targets = registry.targets()
    attention_targets = [target for target in targets if target.component_type in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    mlp_targets = [target for target in targets if target.component_type in {"gate_proj", "up_proj", "down_proj"}]
    assert len(attention_targets) == 6
    assert len(mlp_targets) == 3
    assert sum(target.component_type == "q_proj" for target in attention_targets) == 2
    assert sum(target.component_type == "k_proj" for target in attention_targets) == 1
    assert sum(target.component_type == "v_proj" for target in attention_targets) == 1
    assert sum(target.component_type == "o_proj" for target in attention_targets) == 2
