import json

import torch
from torch import nn

from EAP_forComponent.mask_builder import ComponentMaskBuilder
from EAP_forComponent.outputs import save_outputs
from EAP_forComponent.rank_allocator import LoraRankAllocator
from EAP_forComponent.runner import _aggregate_future_k_scores, _future_step_k_values
from EAP_forComponent.schemas import ComponentScore, EAPComponentConfig
from EAP_forComponent.scorer import ComponentAttributionScorer


def make_score(parameter_name, module_name, rank_score, head_idx=None, row_slice=None, col_slice=None, numel=4):
    return ComponentScore(
        parameter_name=parameter_name,
        module_name=module_name,
        layer_idx=0,
        component_type="q_proj",
        granularity="head" if head_idx is not None else "projection_matrix",
        score_token_mode="all_active",
        head_idx=head_idx,
        head_kind="query" if head_idx is not None else None,
        row_slice=row_slice,
        col_slice=col_slice,
        raw_score=rank_score,
        abs_score=abs(rank_score),
        mean_score=rank_score,
        sqrt_numel_score=rank_score,
        rank_score=rank_score,
        shape=(2, 2),
        numel=numel,
        token_count=1,
        element_count=2,
    )


def test_rank_allocator_aggregates_head_scores_to_module_rank_pattern():
    scores = [
        make_score("layer.q_proj.weight", "layer.q_proj", 1.0, head_idx=0, row_slice=(0, 1), numel=2),
        make_score("layer.q_proj.weight", "layer.q_proj", 3.0, head_idx=1, row_slice=(1, 2), numel=2),
        make_score("layer.k_proj.weight", "layer.k_proj", 0.5),
    ]
    allocator = LoraRankAllocator(min_rank=0, max_rank=8, head_to_matrix_aggregation="max")
    ranks = allocator.allocate(scores)
    rank_pattern = allocator.to_rank_pattern(ranks)
    assert ranks["layer.q_proj"] >= ranks["layer.k_proj"]
    assert "layer.q_proj" in rank_pattern


def test_component_mask_builder_writes_head_row_slices_only():
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.q_proj = nn.Linear(2, 2, bias=False)
    scores = [
        make_score("layer.q_proj.weight", "layer.q_proj", 10.0, head_idx=0, row_slice=(0, 1), numel=2),
        make_score("layer.q_proj.weight", "layer.q_proj", 0.0, head_idx=1, row_slice=(1, 2), numel=2),
    ]
    builder = ComponentMaskBuilder(mask_fill_strategy="first", include_all_parameters=False)
    mask = builder.build(model, scores)["layer.q_proj.weight"]
    assert mask[0].all()
    assert not mask[1].any()


def test_component_mask_builder_respects_keep_ratio_bounds():
    model = nn.Module()
    model.a = nn.Linear(2, 2, bias=False)
    model.b = nn.Linear(2, 2, bias=False)
    model.c = nn.Linear(2, 2, bias=False)
    scores = [
        make_score("a.weight", "a", 3.0),
        make_score("b.weight", "b", 2.0),
        make_score("c.weight", "c", 1.0),
    ]
    builder = ComponentMaskBuilder(
        mask_fill_strategy="first",
        include_all_parameters=False,
        min_keep_ratio=0.25,
        max_keep_ratio=0.75,
    )
    mask = builder.build(model, scores)

    assert int(mask["a.weight"].sum().item()) == 3
    assert int(mask["b.weight"].sum().item()) == 2
    assert int(mask["c.weight"].sum().item()) == 1
    assert [item["keep_ratio"] for item in builder.last_summary["per_component"]] == [0.75, 0.5, 0.25]
    assert builder.last_summary["min_keep_ratio"] == 0.25
    assert builder.last_summary["max_keep_ratio"] == 0.75


def test_component_mask_builder_rejects_invalid_keep_ratio_bounds():
    try:
        ComponentMaskBuilder(min_keep_ratio=0.8, max_keep_ratio=0.2)
    except ValueError as error:
        assert "min_keep_ratio" in str(error)
    else:
        raise AssertionError("Expected invalid keep ratio bounds to raise")


def test_score_normalization_defaults_to_raw_sum():
    assert EAPComponentConfig().score_normalization == "sum"
    assert ComponentAttributionScorer(targets=[]).score_normalization == "sum"


def test_future_step_k_sampling_uses_unique_four_decimal_values():
    config = EAPComponentConfig(
        future_step_k_min=0.0,
        future_step_k_max=1.0,
        future_step_k_samples=10,
        future_step_k_seed=13,
    )
    values = _future_step_k_values(config)
    assert len(values) == 10
    assert len(set(values)) == 10
    assert values == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert all(round(value, 4) == value for value in values)


def test_future_k_aggregation_uses_mean_raw_rank_not_mean_score(tmp_path):
    score_runs = [
        [make_score("a.weight", "a", 100.0), make_score("b.weight", "b", 90.0), make_score("c.weight", "c", 80.0)],
        [make_score("a.weight", "a", 1.0), make_score("b.weight", "b", 2.0), make_score("c.weight", "c", 3.0)],
        [make_score("a.weight", "a", 1.0), make_score("b.weight", "b", 2.0), make_score("c.weight", "c", 3.0)],
    ]
    scores = _aggregate_future_k_scores(score_runs, k_values=[0.1, 0.2, 0.3])
    by_name = {score.parameter_name: score for score in scores}

    assert by_name["a.weight"].raw_score > by_name["b.weight"].raw_score > by_name["c.weight"].raw_score
    assert by_name["c.weight"].mean_raw_rank < by_name["b.weight"].mean_raw_rank < by_name["a.weight"].mean_raw_rank
    assert by_name["c.weight"].rank_score > by_name["b.weight"].rank_score > by_name["a.weight"].rank_score

    paths = save_outputs(
        output_dir=tmp_path,
        scores=scores,
        rank_pattern={},
        lora_allocation={},
        component_mask={},
        summary={},
    )
    saved_scores = json.loads(paths["component_scores_json"].read_text(encoding="utf-8"))
    assert [score["parameter_name"] for score in saved_scores] == ["c.weight", "b.weight", "a.weight"]