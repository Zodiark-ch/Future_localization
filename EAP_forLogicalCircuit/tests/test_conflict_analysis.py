from __future__ import annotations

from EAP_forComponent.schemas import ComponentScore
from EAP_forLogicalCircuit.conflict_analysis import TaskArtifact, analyze_conflicts, build_module_assignment


def _score(module_name: str, raw: float) -> ComponentScore:
    return ComponentScore(
        parameter_name=f"{module_name}.weight",
        module_name=module_name,
        layer_idx=0,
        component_type="up_proj",
        granularity="projection_matrix",
        score_token_mode="all_active",
        head_idx=None,
        head_kind=None,
        row_slice=None,
        col_slice=None,
        raw_score=raw,
        abs_score=abs(raw),
        mean_score=raw,
        sqrt_numel_score=raw,
        rank_score=abs(raw),
        shape=(2, 2),
        numel=4,
        token_count=1,
        element_count=4,
        localization_mode="current",
        current_score=raw,
    )


def _component_score(
    module_name: str,
    raw: float,
    layer_idx: int,
    component_type: str,
    head_idx: int | None = None,
) -> ComponentScore:
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
        raw_score=raw,
        abs_score=abs(raw),
        mean_score=raw,
        sqrt_numel_score=raw,
        rank_score=abs(raw),
        shape=(2, 2),
        numel=4,
        token_count=1,
        element_count=4,
        localization_mode="current",
        current_score=raw,
    )


def test_conflict_analysis_no_conflict_case():
    task_a = TaskArtifact(
        task_name="task_a",
        component_scores=[_score("m1", 5.0)],
        module_assignment={"m1": (1, "positive_path")},
        module_rank={"m1": 1},
    )
    task_b = TaskArtifact(
        task_name="task_b",
        component_scores=[_score("m1", 3.0)],
        module_assignment={"m1": (1, "positive_path")},
        module_rank={"m1": 1},
    )
    result = analyze_conflicts([task_a, task_b])
    assert len(result["conflict_components"]) == 0
    assert len(result["all_task_components"]) == 1
    assert result["all_task_assignment_reasons"]["m1.weight"] == "positive_path"


def test_conflict_analysis_full_conflict_and_or_excluded():
    task_a = TaskArtifact(
        task_name="task_a",
        component_scores=[_score("m1", 5.0)],
        module_assignment={"m1": (1, "positive_path")},
        module_rank={"m1": 1},
    )
    task_b = TaskArtifact(
        task_name="task_b",
        component_scores=[_score("m1", -2.0)],
        module_assignment={"m1": (0, "or_gate_excluded")},
        module_rank={"m1": 1},
    )
    result = analyze_conflicts([task_a, task_b])
    assert len(result["conflict_components"]) == 1
    assert result["conflict_assignment_reasons"]["m1.weight"] == "conflict_unresolved"
    assert result["task_assignment_reasons"]["task_b"]["m1.weight"] == "or_gate_excluded"


def test_build_module_assignment_maps_eap_ig_endpoints_to_components():
    scores = [
        _component_score("model.layers.1.self_attn.q_proj", 1.0, 1, "q_proj", head_idx=5),
        _component_score("model.layers.0.self_attn.o_proj", 1.0, 0, "o_proj", head_idx=3),
        _component_score("model.layers.3.self_attn.k_proj", 1.0, 3, "k_proj", head_idx=1),
        _component_score("model.layers.2.mlp.gate_proj", 1.0, 2, "gate_proj"),
        _component_score("model.layers.2.mlp.up_proj", 1.0, 2, "up_proj"),
        _component_score("model.layers.0.mlp.down_proj", 1.0, 0, "down_proj"),
    ]
    logical_edges = [
        {"source_node": "a0.h3", "destination_module": "a1.h5<q>", "logical_gate": "AND"},
        {"source_node": "input", "destination_module": "a3.h6<k>", "logical_gate": "OR"},
        {"source_node": "input", "destination_module": "m2", "logical_gate": "ADDER"},
        {"source_node": "m0", "destination_module": "logits", "logical_gate": "OR"},
    ]
    assignments = build_module_assignment(
        logical_edges=logical_edges,
        component_scores=scores,
        graph_metadata={"n_heads": 8, "num_key_value_heads": 2, "query_heads_per_kv_head": 4},
    )

    by_name = {score.component_name: assignments[score.component_name] for score in scores}
    assert by_name["model.layers.1.self_attn.q_proj.weight.head_5"] == (1, "and_or_adder_path")
    assert by_name["model.layers.0.self_attn.o_proj.weight.head_3"] == (1, "and_or_adder_path")
    assert by_name["model.layers.3.self_attn.k_proj.weight.head_1"] == (0, "or_gate_excluded")
    assert by_name["model.layers.2.mlp.gate_proj.weight"] == (1, "and_or_adder_path")
    assert by_name["model.layers.2.mlp.up_proj.weight"] == (1, "and_or_adder_path")
    assert by_name["model.layers.0.mlp.down_proj.weight"] == (0, "or_gate_excluded")
