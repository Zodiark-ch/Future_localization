from __future__ import annotations

from EAP_forLogicalCircuit.conflict_analysis import build_module_assignment


def test_build_module_assignment_prefers_logical_gate():
    logical_edges = [
        {"destination_module": "m1", "logical_gate": "OR", "logical_assignment": "positive_path"},
        {"destination_module": "m2", "logical_gate": "AND", "logical_assignment": "or_path"},
        {"destination_module": "m3", "logical_gate": "ADDER", "logical_assignment": "or_path"},
    ]
    assignments = build_module_assignment(logical_edges)
    assert assignments["m1"] == (0, "or_gate_excluded")
    assert assignments["m2"] == (1, "and_or_adder_path")
    assert assignments["m3"] == (1, "and_or_adder_path")


def test_build_module_assignment_legacy_fallback():
    logical_edges = [
        {"destination_module": "m1", "logical_assignment": "positive_path"},
        {"destination_module": "m2", "logical_assignment": "or_path"},
    ]
    assignments = build_module_assignment(logical_edges)
    assert assignments["m1"] == (1, "positive_path_legacy")
    assert assignments["m2"] == (0, "or_gate_excluded_legacy")
