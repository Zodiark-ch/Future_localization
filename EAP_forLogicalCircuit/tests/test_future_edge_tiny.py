from __future__ import annotations

from EAP_forLogicalCircuit.future_localization import future_trapezoid_raw_score


def test_future_trapezoid_no_external_k_multiplier():
    current_score = 1.2
    direction_theta = 0.8
    direction_theta_hat = -0.2
    # Direction terms are already K-scaled internally; final formula should not multiply K again.
    raw = future_trapezoid_raw_score(current_score, direction_theta, direction_theta_hat)
    assert abs(raw - 1.5) < 1e-9
