from __future__ import annotations

import torch
from torch import nn

from EAP_forComponent.schemas import ComponentScore
from EAP_forLogicalCircuit.mask_builder import ComponentMaskBuilder
from EAP_forLogicalCircuit.rank_allocator import LoraRankAllocator


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(2, 2, bias=False)
        self.w2 = nn.Linear(2, 2, bias=False)


def _score(parameter_name: str, rank_score: float) -> ComponentScore:
    return ComponentScore(
        parameter_name=parameter_name,
        module_name=parameter_name.removesuffix(".weight"),
        layer_idx=0,
        component_type="up_proj",
        granularity="projection_matrix",
        score_token_mode="all_active",
        head_idx=None,
        head_kind=None,
        row_slice=None,
        col_slice=None,
        raw_score=rank_score,
        abs_score=abs(rank_score),
        mean_score=rank_score / 4.0,
        sqrt_numel_score=rank_score / 2.0,
        rank_score=abs(rank_score),
        shape=(2, 2),
        numel=4,
        token_count=1,
        element_count=4,
    )


def test_rank_allocator_and_mask_builder_compatibility():
    scores = [_score("w1.weight", 4.0), _score("w2.weight", 1.0)]
    allocator = LoraRankAllocator(
        min_rank=0,
        max_rank=8,
        rank_multiple=1,
        rank_budget=None,
        head_to_matrix_aggregation="mean",
        rank_score_source="normalized_abs",
    )
    ranks = allocator.allocate(scores)
    assert set(ranks) == {"w1", "w2"}
    assert ranks["w1"] >= ranks["w2"]

    model = TinyModel()
    mask_builder = ComponentMaskBuilder(mask_fill_strategy="first", seed=0, include_all_parameters=True)
    mask = mask_builder.build(model, scores)
    assert "w1.weight" in mask
    assert mask["w1.weight"].dtype == torch.bool
    assert mask["w1.weight"].shape == torch.Size([2, 2])
