import torch

from EAP_forComponent.scorer import score_component_output, score_o_proj_head_input_slice


def test_score_component_output_all_active_matches_manual_sum():
    clean = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    corrupted = torch.tensor([[[2.0, 1.0], [5.0, 1.0]]])
    grad = torch.tensor([[[0.5, 2.0], [1.5, -1.0]]])
    attention_mask = torch.tensor([[1, 1]])
    label_positions = torch.tensor([1])
    actual, token_count, element_count = score_component_output(
        clean_output=clean,
        corrupted_output=corrupted,
        output_grad=grad,
        attention_mask=attention_mask,
        label_positions=label_positions,
        token_mode="all_active",
    )
    expected = ((clean - corrupted) * grad).sum().item()
    assert actual == expected
    assert token_count == 2
    assert element_count == 4


def test_score_component_output_label_position_uses_last_label_row():
    clean = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    corrupted = torch.tensor([[[100.0, 100.0], [5.0, 1.0]]])
    grad = torch.tensor([[[100.0, 100.0], [1.5, -1.0]]])
    attention_mask = torch.tensor([[1, 1]])
    label_positions = torch.tensor([1])
    actual, token_count, element_count = score_component_output(
        clean_output=clean,
        corrupted_output=corrupted,
        output_grad=grad,
        attention_mask=attention_mask,
        label_positions=label_positions,
        token_mode="label_position",
    )
    expected = ((clean[:, 1] - corrupted[:, 1]) * grad[:, 1]).sum().item()
    assert actual == expected
    assert token_count == 1
    assert element_count == 2


def test_o_proj_head_input_slice_matches_manual_contribution():
    clean_input = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    corrupted_input = torch.tensor([[[2.0, 0.0, 5.0, 1.0]]])
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 1.5]])
    grad = torch.tensor([[[0.25, -2.0]]])
    attention_mask = torch.tensor([[1]])
    label_positions = torch.tensor([0])
    actual, token_count, element_count = score_o_proj_head_input_slice(
        clean_input=clean_input,
        corrupted_input=corrupted_input,
        weight=weight,
        output_grad=grad,
        col_slice=(0, 2),
        attention_mask=attention_mask,
        label_positions=label_positions,
        token_mode="all_active",
    )
    clean_contrib = clean_input[:, :, :2].matmul(weight[:, :2].t())
    corrupted_contrib = corrupted_input[:, :, :2].matmul(weight[:, :2].t())
    expected = ((clean_contrib - corrupted_contrib) * grad).sum().item()
    assert actual == expected
    assert token_count == 1
    assert element_count == 2
