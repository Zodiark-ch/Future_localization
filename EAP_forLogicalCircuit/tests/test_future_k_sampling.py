from __future__ import annotations

from EAP_forLogicalCircuit.runner import _aggregate_future_k_scores, _future_step_k_values
from EAP_forLogicalCircuit.schemas import EAPLogicalCircuitConfig, EdgeScore


def _score(edge_id: str, raw_score: float) -> EdgeScore:
    return EdgeScore(
        edge_id=edge_id,
        source_node="residual_stream",
        destination_module=edge_id,
        layer_idx=0,
        component_type="up_proj",
        score_token_mode="all_active",
        localization_mode="future",
        raw_score=raw_score,
        abs_score=abs(raw_score),
        mean_score=raw_score,
        sqrt_numel_score=raw_score,
        rank_score=abs(raw_score),
        token_count=1,
        element_count=1,
        numel=1,
        current_score=raw_score,
        future_directional_score_theta=0.0,
        future_directional_score_theta_hat=0.0,
        future_correction=0.0,
        future_step_k=1.0,
    )


def test_future_k_values_unique_and_4decimal():
    config = EAPLogicalCircuitConfig(
        future_step_k_min=0.0,
        future_step_k_max=1.0,
        future_step_k_samples=10,
        future_step_k_seed=7,
    )
    k_values = _future_step_k_values(config)
    assert len(k_values) == 10
    assert len(set(k_values)) == 10
    assert all(abs(value - round(value, 4)) < 1e-12 for value in k_values)


def test_aggregate_uses_mean_raw_rank_not_resort_mean_score():
    run1 = [_score("e1", 10.0), _score("e2", 9.0)]
    run2 = [_score("e1", 1.0), _score("e2", 100.0)]
    # Mean raw scores: e1=5.5, e2=54.5 (if re-ranked by mean score, e2 would be first)
    # Per-run abs(raw) ranks: run1(e1=1,e2=2), run2(e2=1,e1=2) => both mean rank=1.5 (tie).
    aggregated = _aggregate_future_k_scores([run1, run2], [0.1, 0.2])
    by_id = {score.edge_id: score for score in aggregated}
    assert abs(by_id["e1"].raw_score - 5.5) < 1e-9
    assert abs(by_id["e2"].raw_score - 54.5) < 1e-9
    assert abs(by_id["e1"].mean_raw_rank - 1.5) < 1e-9
    assert abs(by_id["e2"].mean_raw_rank - 1.5) < 1e-9
