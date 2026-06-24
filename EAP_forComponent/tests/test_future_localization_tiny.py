from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from EAP_forComponent.future_localization import FutureLocalizationScorer
from EAP_forComponent.schemas import ComponentTarget, EAPComponentConfig, PairBatch


class TinyFutureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=5)
        self.embed = nn.Embedding(5, 3)
        self.block = nn.Module()
        self.block.q_proj = nn.Linear(3, 3, bias=False)
        self.lm_head = nn.Linear(3, 5, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.1, 0.2],
                        [0.2, -0.1, 0.0],
                        [0.1, 0.3, -0.2],
                        [-0.3, 0.2, 0.1],
                        [0.4, -0.2, 0.2],
                    ]
                )
            )
            self.block.q_proj.weight.copy_(
                torch.tensor(
                    [
                        [0.2, -0.1, 0.3],
                        [0.0, 0.4, -0.2],
                        [-0.3, 0.1, 0.2],
                    ]
                )
            )
            self.lm_head.weight.copy_(
                torch.tensor(
                    [
                        [0.1, 0.0, -0.2],
                        [0.3, -0.1, 0.2],
                        [-0.2, 0.4, 0.1],
                        [0.0, 0.2, -0.1],
                        [0.2, 0.1, 0.3],
                    ]
                )
            )

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        del attention_mask, position_ids, use_cache
        hidden = self.block.q_proj(self.embed(input_ids))
        logits = self.lm_head(torch.tanh(hidden))
        return SimpleNamespace(logits=logits)


class TinySDPAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=7)
        self.embed = nn.Embedding(7, 8)
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 7, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        del attention_mask, position_ids, use_cache
        hidden = self.embed(input_ids)
        query = self.q_proj(hidden).view(hidden.size(0), hidden.size(1), 2, 4).transpose(1, 2)
        key = self.k_proj(hidden).view(hidden.size(0), hidden.size(1), 2, 4).transpose(1, 2)
        value = self.v_proj(hidden).view(hidden.size(0), hidden.size(1), 2, 4).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(query, key, value)
        attn_output = attn_output.transpose(1, 2).reshape(hidden.size(0), hidden.size(1), 8)
        logits = self.lm_head(self.o_proj(attn_output))
        return SimpleNamespace(logits=logits)


def test_future_hvp_direction_matches_finite_difference_on_tiny_model():
    base_model = TinyFutureModel()
    future_state = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}
    future_state["block.q_proj.weight"] = future_state["block.q_proj.weight"] + torch.tensor(
        [
            [0.01, -0.02, 0.03],
            [0.02, 0.01, -0.01],
            [-0.02, 0.03, 0.01],
        ]
    )
    target = ComponentTarget(
        parameter_name="block.q_proj.weight",
        module_name="block.q_proj",
        layer_idx=0,
        component_type="q_proj",
        granularity="projection_matrix",
        head_idx=None,
        head_kind=None,
        row_slice=None,
        col_slice=None,
        module=base_model.block.q_proj,
        shape=base_model.block.q_proj.weight.shape,
        numel=int(base_model.block.q_proj.weight.numel()),
    )
    batch = PairBatch(
        clean_input_ids=torch.tensor([[1, 2]]),
        clean_attention_mask=torch.tensor([[1, 1]]),
        corrupted_input_ids=torch.tensor([[3, 2]]),
        corrupted_attention_mask=torch.tensor([[1, 1]]),
        labels=torch.full((1, 2), -100),
        correct_idx=torch.tensor([1]),
        incorrect_idx=torch.tensor([2]),
        label_positions=torch.tensor([1]),
    )
    hvp_config = EAPComponentConfig(
        localization_mode="future",
        metric="logit_diff",
        score_token_mode="all_active",
        score_normalization="sum",
        future_step_k=0.7,
        future_hvp_strategy="hvp",
    )
    hvp_score = FutureLocalizationScorer(
        model=base_model,
        targets=[target],
        config=hvp_config,
        device="cpu",
        future_state_dict=future_state,
    ).score([batch])[0]

    fd_model = TinyFutureModel()
    fd_target = ComponentTarget(
        parameter_name="block.q_proj.weight",
        module_name="block.q_proj",
        layer_idx=0,
        component_type="q_proj",
        granularity="projection_matrix",
        head_idx=None,
        head_kind=None,
        row_slice=None,
        col_slice=None,
        module=fd_model.block.q_proj,
        shape=fd_model.block.q_proj.weight.shape,
        numel=int(fd_model.block.q_proj.weight.numel()),
    )
    fd_config = EAPComponentConfig(
        localization_mode="future",
        metric="logit_diff",
        score_token_mode="all_active",
        score_normalization="sum",
        future_step_k=0.7,
        future_hvp_strategy="finite_difference",
        future_finite_difference_epsilon=1e-3,
    )
    fd_score = FutureLocalizationScorer(
        model=fd_model,
        targets=[fd_target],
        config=fd_config,
        device="cpu",
        future_state_dict=future_state,
    ).score([batch])[0]

    assert hvp_score.localization_mode == "future"
    assert hvp_score.future_step_k == 0.7
    assert hvp_score.future_directional_score_theta is not None
    assert fd_score.future_directional_score_theta is not None
    torch.testing.assert_close(
        torch.tensor(hvp_score.future_directional_score_theta),
        torch.tensor(fd_score.future_directional_score_theta),
        rtol=2e-2,
        atol=2e-4,
    )


def test_future_hvp_uses_math_sdp_for_second_order_attention():
    if not torch.cuda.is_available():
        return
    model = TinySDPAModel().cuda()
    future_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    future_state["q_proj.weight"] = future_state["q_proj.weight"] + 0.01 * torch.randn_like(
        future_state["q_proj.weight"]
    )
    target = ComponentTarget(
        parameter_name="q_proj.weight",
        module_name="q_proj",
        layer_idx=0,
        component_type="q_proj",
        granularity="projection_matrix",
        head_idx=None,
        head_kind=None,
        row_slice=None,
        col_slice=None,
        module=model.q_proj,
        shape=model.q_proj.weight.shape,
        numel=int(model.q_proj.weight.numel()),
    )
    batch = PairBatch(
        clean_input_ids=torch.tensor([[1, 2, 3]]),
        clean_attention_mask=torch.tensor([[1, 1, 1]]),
        corrupted_input_ids=torch.tensor([[4, 5, 6]]),
        corrupted_attention_mask=torch.tensor([[1, 1, 1]]),
        labels=torch.full((1, 3), -100),
        correct_idx=torch.tensor([1]),
        incorrect_idx=torch.tensor([2]),
        label_positions=torch.tensor([2]),
    )
    config = EAPComponentConfig(
        localization_mode="future",
        metric="logit_diff",
        score_token_mode="all_active",
        score_normalization="sum",
        future_step_k=1.0,
        future_hvp_strategy="hvp",
    )
    score = FutureLocalizationScorer(
        model=model,
        targets=[target],
        config=config,
        device=torch.device("cuda:0"),
        future_state_dict=future_state,
    ).score([batch])[0]
    assert score.localization_mode == "future"
    assert score.future_directional_score_theta is not None