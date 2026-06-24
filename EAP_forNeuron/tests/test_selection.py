import torch

from EAP_forNeuron.schemas import ScoreShard
from EAP_forNeuron.selection import NeuronMaskSelector


def test_selector_keeps_subset_and_candidate_ratio():
    old_mask = {"linear.weight": torch.tensor([[True, False], [True, True]])}
    scores = {
        "linear.weight": ScoreShard(
            parameter_name="linear.weight",
            shape=torch.Size([2, 2]),
            flat_indices=torch.tensor([0, 2, 3]),
            scores=torch.tensor([0.1, 3.0, 2.0]),
        )
    }
    selector = NeuronMaskSelector(output_ratio=0.5, ratio_base="candidate", score_abs=True)
    new_mask, summary = selector.select(old_mask, scores, total_neuron_count=4)
    assert summary["actual_keep_count"] == 1
    assert new_mask["linear.weight"].sum().item() == 1
    assert new_mask["linear.weight"][1, 0]
    assert not (new_mask["linear.weight"] & ~old_mask["linear.weight"]).any()


def test_selector_all_ratio_is_clipped_by_candidates():
    old_mask = {"linear.weight": torch.tensor([[False, True], [False, False]])}
    scores = {
        "linear.weight": ScoreShard(
            parameter_name="linear.weight",
            shape=torch.Size([2, 2]),
            flat_indices=torch.tensor([1]),
            scores=torch.tensor([4.0]),
        )
    }
    selector = NeuronMaskSelector(output_ratio=1.0, ratio_base="all")
    new_mask, summary = selector.select(old_mask, scores, total_neuron_count=4)
    assert summary["target_keep_count"] == 4
    assert summary["actual_keep_count"] == 1
    assert torch.equal(new_mask["linear.weight"], old_mask["linear.weight"])
