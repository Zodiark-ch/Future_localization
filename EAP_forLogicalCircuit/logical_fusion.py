from __future__ import annotations

from dataclasses import dataclass

from EAP_forLogicalCircuit.circuit_builder import CircuitEdge


@dataclass
class LogicalEdge:
    edge_id: str
    source_node: str
    destination_module: str
    layer_idx: int
    component_type: str
    in_circuit: bool
    in_circuit_or: bool
    logical_assignment: str
    logical_gate: str
    circuit_score: float | None
    circuit_or_score: float | None


def fuse_logical_edges(
    circuit_edges: list[CircuitEdge],
    circuit_or_edges: list[CircuitEdge],
) -> list[LogicalEdge]:
    circuit_map = {edge.edge_id: edge for edge in circuit_edges}
    or_map = {edge.edge_id: edge for edge in circuit_or_edges}
    and_edge_ids, or_edge_ids = _compute_gate_edge_ids(circuit_edges, circuit_or_edges)
    all_ids = sorted(set(circuit_map) | set(or_map))
    fused: list[LogicalEdge] = []
    for edge_id in all_ids:
        c_edge = circuit_map.get(edge_id)
        o_edge = or_map.get(edge_id)
        in_circuit = c_edge is not None
        in_circuit_or = o_edge is not None
        if in_circuit and in_circuit_or:
            assignment = "shared_path"
            template = c_edge
        elif in_circuit:
            assignment = "positive_path"
            template = c_edge
        elif in_circuit_or:
            assignment = "or_path"
            template = o_edge
        else:
            assignment = "unresolved"
            continue
        assert template is not None
        fused.append(
            LogicalEdge(
                edge_id=edge_id,
                source_node=template.source_node,
                destination_module=template.destination_module,
                layer_idx=template.layer_idx,
                component_type=template.component_type,
                in_circuit=in_circuit,
                in_circuit_or=in_circuit_or,
                logical_assignment=assignment,
                logical_gate=_logical_gate_for_edge(edge_id, and_edge_ids, or_edge_ids),
                circuit_score=float(c_edge.score) if c_edge is not None and c_edge.score is not None else None,
                circuit_or_score=float(o_edge.score) if o_edge is not None and o_edge.score is not None else None,
            )
        )
    return fused


def _logical_gate_for_edge(edge_id: str, and_edge_ids: set[str], or_edge_ids: set[str]) -> str:
    if edge_id in or_edge_ids:
        return "OR"
    if edge_id in and_edge_ids:
        return "AND"
    return "ADDER"


def _compute_gate_edge_ids(
    circuit_edges: list[CircuitEdge],
    circuit_or_edges: list[CircuitEdge],
) -> tuple[set[str], set[str]]:
    circuit_ids = {edge.edge_id for edge in circuit_edges}
    or_ids = {edge.edge_id for edge in circuit_or_edges}
    circuit_by_id = {edge.edge_id: edge for edge in circuit_edges}
    or_by_id = {edge.edge_id: edge for edge in circuit_or_edges}

    # Approximate EAP-IG get_logical_edge.py behavior using destination-module fan-in counts.
    and_candidates = circuit_ids - or_ids
    or_candidates = or_ids - circuit_ids
    and_edge_ids = _filter_gate_edges(
        candidates=and_candidates,
        primary_map=circuit_by_id,
        other_map=or_by_id,
    )
    or_edge_ids = _filter_gate_edges(
        candidates=or_candidates,
        primary_map=or_by_id,
        other_map=circuit_by_id,
    )
    return and_edge_ids, or_edge_ids


def _filter_gate_edges(
    candidates: set[str],
    primary_map: dict[str, CircuitEdge],
    other_map: dict[str, CircuitEdge],
) -> set[str]:
    filtered = []
    for edge_id in candidates:
        edge = primary_map.get(edge_id)
        if edge is None:
            continue
        if str(edge.destination_module).endswith(".o"):
            continue
        filtered.append(edge)

    to_count: dict[str, int] = {}
    for edge in filtered:
        to_count[edge.destination_module] = to_count.get(edge.destination_module, 0) + 1

    other_to_count: dict[str, int] = {}
    for edge in other_map.values():
        other_to_count[edge.destination_module] = other_to_count.get(edge.destination_module, 0) + 1
    for destination_module, count in other_to_count.items():
        if count == 1:
            to_count[destination_module] = to_count.get(destination_module, 0) + 1

    strict_result = {
        edge.edge_id
        for edge in filtered
        if to_count.get(edge.destination_module, 0) >= 2
    }

    # Fallback for phase1-minimal graph: each destination often has a single incoming edge
    # (e.g. residual_stream -> module.input), so strict fan-in rule from EAP-IG would classify
    # almost everything as ADDER. In that degenerate topology we label set-difference edges
    # directly as gate edges to preserve AND/OR signal for downstream conflict analysis.
    if strict_result:
        return strict_result
    if not filtered:
        return set()
    max_primary_fanin = 0
    primary_to_count: dict[str, int] = {}
    for edge in primary_map.values():
        primary_to_count[edge.destination_module] = primary_to_count.get(edge.destination_module, 0) + 1
    if primary_to_count:
        max_primary_fanin = max(primary_to_count.values())
    if max_primary_fanin <= 1:
        return {edge.edge_id for edge in filtered}
    return set()
