from __future__ import annotations

from dataclasses import dataclass, field, replace

from EAP_forComponent.schemas import ComponentScore
from EAP_forLogicalCircuit.circuit_builder import CircuitEdge


@dataclass
class RankedGraphNode:
    node_id: str
    raw_score: float = 0.0
    rank_score: float = 0.0
    component_count: int = 0
    component_names: list[str] = field(default_factory=list)


@dataclass
class NodeInducedCircuit:
    edges: list[CircuitEdge]
    selected_nodes: list[RankedGraphNode]
    component_scores: list[ComponentScore]
    summary: dict


def build_node_induced_circuit(
    component_scores: list[ComponentScore],
    graph_metadata: dict,
    node_topn: int,
    circuit_name: str,
) -> NodeInducedCircuit:
    if node_topn <= 0:
        raise ValueError("node_topn must be > 0")
    ranked_nodes = _rank_component_scores_as_graph_nodes(
        component_scores=component_scores,
        graph_metadata=graph_metadata,
    )
    selected_nodes = ranked_nodes[: min(int(node_topn), len(ranked_nodes))]
    selected_node_ids = {node.node_id for node in selected_nodes}
    selected_source_nodes = {"input", *selected_node_ids}
    selected_destination_nodes = {*selected_node_ids, "logits"}
    edges = _induce_edges_from_selected_nodes(
        selected_source_nodes=selected_source_nodes,
        selected_destination_nodes=selected_destination_nodes,
        graph_metadata=graph_metadata,
        circuit_name=circuit_name,
    )
    selected_component_scores = _select_component_scores_for_nodes(
        component_scores=component_scores,
        selected_node_ids=selected_node_ids,
        graph_metadata=graph_metadata,
    )
    summary = {
        "construction": "node_attribution_topn_induced_eap_ig_edges",
        "node_topn": int(node_topn),
        "available_node_count": len(ranked_nodes),
        "selected_node_count": len(selected_nodes),
        "selected_nodes": [node.node_id for node in selected_nodes],
        "selected_source_node_count": len(selected_source_nodes),
        "selected_destination_node_count": len(selected_destination_nodes),
        "selected_component_count": len(selected_component_scores),
        "induced_edge_count": len(edges),
        "mandatory_source_nodes": ["input"],
        "mandatory_destination_nodes": ["logits"],
        "source_node_spec": ["input", "a{layer}.h{head}", "m{layer}"],
        "destination_node_spec": ["a{layer}.h{head}<q|k|v>", "m{layer}", "logits"],
        "edge_score_computed": False,
    }
    return NodeInducedCircuit(
        edges=edges,
        selected_nodes=selected_nodes,
        component_scores=selected_component_scores,
        summary=summary,
    )


def combine_selected_component_scores(
    circuit_scores: list[ComponentScore],
    circuit_or_scores: list[ComponentScore],
    score_normalization: str,
    localization_mode: str,
) -> list[ComponentScore]:
    score_by_name: dict[str, list[ComponentScore]] = {}
    for score in circuit_scores:
        score_by_name.setdefault(score.component_name, [score, _zero_like_component_score(score)])
        score_by_name[score.component_name][0] = score
    for score in circuit_or_scores:
        score_by_name.setdefault(score.component_name, [_zero_like_component_score(score), score])
        score_by_name[score.component_name][1] = score

    combined: list[ComponentScore] = []
    for component_name in sorted(score_by_name):
        circuit_score, circuit_or_score = score_by_name[component_name]
        template = circuit_score if circuit_score.numel > 0 else circuit_or_score
        raw_score = (float(circuit_score.raw_score) + float(circuit_or_score.raw_score)) / 2.0
        abs_score = abs(raw_score)
        mean_score = (float(circuit_score.mean_score) + float(circuit_or_score.mean_score)) / 2.0
        sqrt_numel_score = (float(circuit_score.sqrt_numel_score) + float(circuit_or_score.sqrt_numel_score)) / 2.0
        rank_score = _rank_score_for_component(
            raw_score=raw_score,
            mean_score=mean_score,
            sqrt_numel_score=sqrt_numel_score,
            score_normalization=score_normalization,
        )
        combined.append(
            replace(
                template,
                raw_score=float(raw_score),
                abs_score=float(abs_score),
                mean_score=float(mean_score),
                sqrt_numel_score=float(sqrt_numel_score),
                rank_score=float(rank_score),
                token_count=int(circuit_score.token_count) + int(circuit_or_score.token_count),
                element_count=max(1, int(circuit_score.element_count) + int(circuit_or_score.element_count)),
                localization_mode=str(localization_mode),
                current_score=_mean_optional_pair(circuit_score.current_score, circuit_or_score.current_score),
                future_directional_score_theta=_mean_optional_pair(
                    circuit_score.future_directional_score_theta,
                    circuit_or_score.future_directional_score_theta,
                ),
                future_directional_score_theta_hat=_mean_optional_pair(
                    circuit_score.future_directional_score_theta_hat,
                    circuit_or_score.future_directional_score_theta_hat,
                ),
                future_correction=_mean_optional_pair(circuit_score.future_correction, circuit_or_score.future_correction),
                future_step_k=None,
                mean_raw_rank=None,
            )
        )
    return combined


def _induce_edges_from_selected_nodes(
    selected_source_nodes: set[str],
    selected_destination_nodes: set[str],
    graph_metadata: dict,
    circuit_name: str,
) -> list[CircuitEdge]:
    n_layers = int(graph_metadata.get("n_layers", 0) or 0)
    n_heads = int(graph_metadata.get("n_heads", 0) or 0)
    parallel_attn_mlp = bool(graph_metadata.get("parallel_attn_mlp", False))
    has_attention_destinations = list(graph_metadata.get("has_attention_destinations_by_layer") or [True] * n_layers)
    has_attention_sources = list(graph_metadata.get("has_attention_sources_by_layer") or [True] * n_layers)
    has_mlp_destinations = list(graph_metadata.get("has_mlp_destinations_by_layer") or [True] * n_layers)
    has_mlp_sources = list(graph_metadata.get("has_mlp_sources_by_layer") or [True] * n_layers)

    edges: list[CircuitEdge] = []
    residual_sources = ["input"] if "input" in selected_source_nodes else []
    for layer_idx in range(n_layers):
        attention_sources = []
        if layer_idx < len(has_attention_sources) and has_attention_sources[layer_idx]:
            attention_sources = [f"a{layer_idx}.h{head_idx}" for head_idx in range(n_heads)]

        if layer_idx < len(has_attention_destinations) and has_attention_destinations[layer_idx]:
            for head_idx in range(n_heads):
                for qkv in "qkv":
                    destination_node = f"a{layer_idx}.h{head_idx}<{qkv}>"
                    if _destination_base_node(destination_node) not in selected_destination_nodes:
                        continue
                    for source_node in residual_sources:
                        if source_node in selected_source_nodes:
                            edges.append(
                                _make_node_induced_edge(
                                    source_node=source_node,
                                    destination_node=destination_node,
                                    layer_idx=layer_idx,
                                    component_type=f"{qkv}_proj",
                                    circuit_name=circuit_name,
                                )
                            )

        mlp_residual_sources = residual_sources if parallel_attn_mlp else residual_sources + attention_sources
        if layer_idx < len(has_mlp_destinations) and has_mlp_destinations[layer_idx]:
            destination_node = f"m{layer_idx}"
            if destination_node in selected_destination_nodes:
                for source_node in mlp_residual_sources:
                    if source_node in selected_source_nodes:
                        edges.append(
                            _make_node_induced_edge(
                                source_node=source_node,
                                destination_node=destination_node,
                                layer_idx=layer_idx,
                                component_type="mlp",
                                circuit_name=circuit_name,
                            )
                        )

        if parallel_attn_mlp:
            residual_sources = residual_sources + [node for node in attention_sources if node in selected_source_nodes]
        else:
            residual_sources = residual_sources + [node for node in attention_sources if node in selected_source_nodes]
        mlp_source = f"m{layer_idx}"
        if layer_idx < len(has_mlp_sources) and has_mlp_sources[layer_idx] and mlp_source in selected_source_nodes:
            residual_sources.append(mlp_source)

    if "logits" in selected_destination_nodes:
        for source_node in residual_sources:
            if source_node in selected_source_nodes:
                edges.append(
                    _make_node_induced_edge(
                        source_node=source_node,
                        destination_node="logits",
                        layer_idx=max(0, n_layers - 1),
                        component_type="logits",
                        circuit_name=circuit_name,
                    )
                )
    return edges


def _make_node_induced_edge(
    source_node: str,
    destination_node: str,
    layer_idx: int,
    component_type: str,
    circuit_name: str,
) -> CircuitEdge:
    return CircuitEdge(
        edge_id=f"{source_node}->{destination_node}",
        source_node=source_node,
        destination_module=destination_node,
        layer_idx=int(layer_idx),
        component_type=component_type,
        score=None,
        abs_score=None,
        selected_score=None,
        selection_reason=f"{circuit_name}:node_induced:eap_ig_topology",
    )


def _rank_component_scores_as_graph_nodes(
    component_scores: list[ComponentScore],
    graph_metadata: dict,
) -> list[RankedGraphNode]:
    nodes: dict[str, RankedGraphNode] = {}
    source_node_set = set(graph_metadata.get("source_nodes") or [])
    destination_node_bases = {
        _destination_base_node(str(destination_node)) for destination_node in graph_metadata.get("destination_nodes", [])
    }
    allowed_node_ids = (source_node_set | destination_node_bases) - {"input", "logits"}
    for score in component_scores:
        for node_id in _component_score_graph_nodes(score=score, graph_metadata=graph_metadata):
            if allowed_node_ids and node_id not in allowed_node_ids:
                continue
            node = nodes.setdefault(node_id, RankedGraphNode(node_id=node_id))
            node.raw_score += float(score.raw_score)
            node.rank_score += float(score.rank_score)
            node.component_count += 1
            node.component_names.append(score.component_name)
    return sorted(nodes.values(), key=lambda node: (-float(node.rank_score), node.node_id))


def _select_component_scores_for_nodes(
    component_scores: list[ComponentScore],
    selected_node_ids: set[str],
    graph_metadata: dict,
) -> list[ComponentScore]:
    selected: dict[str, ComponentScore] = {}
    for score in component_scores:
        score_nodes = set(_component_score_graph_nodes(score=score, graph_metadata=graph_metadata))
        if score_nodes & selected_node_ids:
            selected[score.component_name] = score
    return sorted(selected.values(), key=lambda score: (-float(score.rank_score), score.component_name))


def _component_score_graph_nodes(score: ComponentScore, graph_metadata: dict) -> list[str]:
    n_heads = int(graph_metadata.get("n_heads", 0) or 0)
    num_key_value_heads = int(graph_metadata.get("num_key_value_heads", 0) or n_heads or 0)
    query_heads_per_kv_head = int(graph_metadata.get("query_heads_per_kv_head", 1) or 1)
    layer_idx = int(score.layer_idx)
    component_type = str(score.component_type)
    if component_type == "o_proj":
        if score.head_idx is None:
            return [f"a{layer_idx}.h{head_idx}" for head_idx in range(max(1, n_heads))]
        return [f"a{layer_idx}.h{int(score.head_idx)}"]
    if component_type == "q_proj":
        if score.head_idx is None:
            return [f"a{layer_idx}.h{head_idx}" for head_idx in range(max(1, n_heads))]
        return [f"a{layer_idx}.h{int(score.head_idx)}"]
    if component_type in {"k_proj", "v_proj"}:
        if score.head_idx is None:
            return [f"a{layer_idx}.h{head_idx}" for head_idx in range(max(1, n_heads))]
        kv_head_idx = int(score.head_idx)
        if num_key_value_heads and num_key_value_heads < n_heads:
            start = kv_head_idx * int(query_heads_per_kv_head)
            end = min(n_heads, start + int(query_heads_per_kv_head))
            return [f"a{layer_idx}.h{head_idx}" for head_idx in range(start, end)]
        return [f"a{layer_idx}.h{kv_head_idx}"]
    if component_type in {"gate_proj", "up_proj", "down_proj"}:
        return [f"m{layer_idx}"]
    return []


def _destination_base_node(destination_node: str) -> str:
    if destination_node.startswith("a") and destination_node.endswith(">") and "<" in destination_node:
        return destination_node.split("<", 1)[0]
    return destination_node


def _rank_score_for_component(
    raw_score: float,
    mean_score: float,
    sqrt_numel_score: float,
    score_normalization: str,
) -> float:
    if score_normalization == "sum":
        return abs(float(raw_score))
    if score_normalization == "mean":
        return abs(float(mean_score))
    if score_normalization == "sqrt_numel":
        return abs(float(sqrt_numel_score))
    raise ValueError(f"Unsupported score_normalization: {score_normalization}")


def _mean_optional_pair(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def _zero_like_component_score(score: ComponentScore) -> ComponentScore:
    return replace(
        score,
        raw_score=0.0,
        abs_score=0.0,
        mean_score=0.0,
        sqrt_numel_score=0.0,
        rank_score=0.0,
        token_count=0,
        element_count=max(1, int(score.numel)),
        current_score=0.0 if score.current_score is not None else None,
        future_directional_score_theta=0.0 if score.future_directional_score_theta is not None else None,
        future_directional_score_theta_hat=0.0 if score.future_directional_score_theta_hat is not None else None,
        future_correction=0.0 if score.future_correction is not None else None,
        future_step_k=None,
        mean_raw_rank=None,
    )
