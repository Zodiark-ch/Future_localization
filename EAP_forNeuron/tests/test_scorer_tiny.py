import torch

from EAP_forNeuron.scorer import score_linear_weight_candidates


def test_score_linear_weight_candidates_matches_manual_loop():
    weight = torch.tensor([[2.0, -1.0, 0.5], [3.0, 4.0, -2.0]])
    flat_indices = torch.tensor([0, 2, 4])
    clean = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]]])
    corrupted = torch.tensor([[[2.0, 1.0, 4.0], [1.0, 3.0, 5.0]]])
    grad = torch.tensor([[[0.5, -1.0], [2.0, 0.25]]])
    attention_mask = torch.tensor([[1, 1]])
    label_positions = torch.tensor([1])

    actual = score_linear_weight_candidates(
        weight=weight,
        flat_indices=flat_indices,
        clean_input=clean,
        corrupted_input=corrupted,
        output_grad=grad,
        attention_mask=attention_mask,
        label_positions=label_positions,
        score_token_mode="all_active",
    )

    expected = []
    for flat_index in flat_indices.tolist():
        row = flat_index // weight.size(1)
        col = flat_index % weight.size(1)
        value = ((corrupted[0, :, col] - clean[0, :, col]) * grad[0, :, row]).sum() * weight[row, col]
        expected.append(value)
    expected = torch.stack(expected)
    assert torch.allclose(actual.cpu(), expected.float())
