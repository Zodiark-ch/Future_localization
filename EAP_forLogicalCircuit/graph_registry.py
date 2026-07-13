from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from torch import nn


DestinationKind = Literal["attention", "mlp", "logits"]
SourceKind = Literal["input", "attention", "mlp"]


@dataclass
class EdgeTarget:
    edge_id: str
    source_node: str
    destination_module: str
    destination_parameter_name: str
    layer_idx: int
    component_type: str
    num_features: int
    module: nn.Module
    source_module_name: str | None = None
    source_layer_idx: int = -1
    source_component_type: str = "input"
    source_module: nn.Linear | None = None
    source_head_idx: int | None = None
    destination_node: str | None = None
    destination_kind: DestinationKind = "mlp"
    destination_qkv: str | None = None
    destination_head_idx: int | None = None
    destination_head_slice: tuple[int, int] | None = None
    source_kind: SourceKind = "input"
    source_head_slice: tuple[int, int] | None = None
    destination_input_keys: tuple[str, ...] = ()
    destination_output_key: str | None = None
    destination_hook_modules: tuple[tuple[str, nn.Module], ...] = ()
    delta_parameter_names: tuple[str, ...] = ()


class GraphRegistry:
    def __init__(self, edge_targets: list[EdgeTarget], metadata: dict):
        self._edge_targets = edge_targets
        self.metadata = metadata

    @classmethod
    def from_model(cls, model: nn.Module, target_modules: Iterable[str]) -> "GraphRegistry":
        target_module_set = {item.strip() for item in target_modules if item.strip()}
        modules = _collect_layer_modules(model)
        if not modules:
            raise ValueError("No transformer layer Linear modules found for dense graph construction.")

        config = getattr(model, "config", None)
        num_layers = max(modules) + 1
        num_attention_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        if num_attention_heads <= 0:
            num_attention_heads = _infer_num_attention_heads(modules)
        if num_attention_heads <= 0:
            raise ValueError("Cannot infer num_attention_heads for dense graph construction.")
        num_key_value_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
        if num_key_value_heads <= 0:
            num_key_value_heads = num_attention_heads
        query_heads_per_kv_head = max(1, num_attention_heads // num_key_value_heads)
        parallel_attn_mlp = bool(getattr(config, "parallel_attn_mlp", False))

        source_nodes = _build_source_nodes(
            modules=modules,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
        )
        destination_nodes = _build_destination_nodes(
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
        )

        edge_targets: list[EdgeTarget] = []
        residual_stream: list[dict[str, object]] = [source_nodes["input"]]
        for layer_idx in range(num_layers):
            layer_modules = modules.get(layer_idx, {})
            q_proj = layer_modules.get("q_proj")
            k_proj = layer_modules.get("k_proj")
            v_proj = layer_modules.get("v_proj")
            o_proj = layer_modules.get("o_proj")
            gate_proj = layer_modules.get("gate_proj")
            up_proj = layer_modules.get("up_proj")

            attention_sources = []
            if o_proj is not None:
                attention_sources = [source_nodes[f"a{layer_idx}.h{head_idx}"] for head_idx in range(num_attention_heads)]

            if q_proj is not None and k_proj is not None and v_proj is not None:
                for head_idx in range(num_attention_heads):
                    for qkv, module_name, module in (
                        ("q", q_proj[0], q_proj[1]),
                        ("k", k_proj[0], k_proj[1]),
                        ("v", v_proj[0], v_proj[1]),
                    ):
                        destination_node = f"a{layer_idx}.h{head_idx}<{qkv}>"
                        destination_slice = _destination_head_slice(
                            module=module,
                            qkv=qkv,
                            graph_head_idx=head_idx,
                            num_attention_heads=num_attention_heads,
                            num_key_value_heads=num_key_value_heads,
                            query_heads_per_kv_head=query_heads_per_kv_head,
                        )
                        for source in residual_stream:
                            edge_targets.append(
                                _make_edge_target(
                                    source=source,
                                    destination_node=destination_node,
                                    destination_kind="attention",
                                    destination_qkv=qkv,
                                    destination_head_idx=head_idx,
                                    destination_head_slice=destination_slice,
                                    destination_module=destination_node,
                                    destination_module_obj=module,
                                    destination_input_keys=(f"{module_name}.weight",),
                                    destination_output_key=f"{module_name}.weight",
                                    destination_hook_modules=((f"{module_name}.weight", module),),
                                    delta_parameter_names=(f"{module_name}.weight",),
                                    layer_idx=layer_idx,
                                    component_type=f"{qkv}_proj",
                                )
                            )

            if parallel_attn_mlp:
                mlp_residual_stream = residual_stream
            else:
                residual_stream = residual_stream + attention_sources
                mlp_residual_stream = residual_stream

            if gate_proj is not None and up_proj is not None:
                module_name, module = gate_proj
                up_module_name, up_module = up_proj
                destination_node = f"m{layer_idx}"
                for source in mlp_residual_stream:
                    edge_targets.append(
                        _make_edge_target(
                            source=source,
                            destination_node=destination_node,
                            destination_kind="mlp",
                            destination_qkv=None,
                            destination_head_idx=None,
                            destination_head_slice=None,
                            destination_module=destination_node,
                            destination_module_obj=module,
                            destination_input_keys=(f"{module_name}.weight", f"{up_module_name}.weight"),
                            destination_output_key=None,
                            destination_hook_modules=(
                                (f"{module_name}.weight", module),
                                (f"{up_module_name}.weight", up_module),
                            ),
                            delta_parameter_names=(f"{module_name}.weight", f"{up_module_name}.weight"),
                            layer_idx=layer_idx,
                            component_type="mlp",
                        )
                    )

            if parallel_attn_mlp:
                residual_stream = residual_stream + attention_sources
            mlp_source = source_nodes.get(f"m{layer_idx}") if "down_proj" in layer_modules else None
            if mlp_source is not None:
                residual_stream = residual_stream + [mlp_source]

        logits_module_name, logits_module = _logits_destination_module(model, modules)
        logits_key = f"{logits_module_name}.input"
        for source in residual_stream:
            edge_targets.append(
                _make_edge_target(
                    source=source,
                    destination_node="logits",
                    destination_kind="logits",
                    destination_qkv=None,
                    destination_head_idx=None,
                    destination_head_slice=None,
                    destination_module="logits",
                    destination_module_obj=logits_module,
                    destination_input_keys=(logits_key,),
                    destination_output_key=None,
                    destination_hook_modules=((logits_key, logits_module),),
                    delta_parameter_names=tuple(),
                    layer_idx=num_layers - 1,
                    component_type="logits",
                )
            )

        if not edge_targets:
            raise ValueError("No dense graph edges were created. Check model module names.")

        metadata = {
            "graph_version": "dense_head_graph_v1",
            "edge_spec": "Dense residual-stream head-level source/destination construction without neuron-level nodes",
            "n_layers": num_layers,
            "n_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "query_heads_per_kv_head": query_heads_per_kv_head,
            "parallel_attn_mlp": parallel_attn_mlp,
            "n_forward": 1 + num_layers * (num_attention_heads + 1),
            "n_backward": num_layers * (3 * num_attention_heads + 1) + 1,
            "source_nodes": [source["node"] for source in source_nodes.values()],
            "source_node_count": len(source_nodes),
            "destination_nodes": destination_nodes,
            "destination_count": len(destination_nodes),
            "edge_count": len(edge_targets),
            "target_modules": sorted(target_module_set),
            "target_modules_note": "Dense graph construction is node-based; q/k/v, mlp, logits destinations are included regardless of projection target list.",
            "has_attention_destinations_by_layer": [
                _has_attention_destination(modules.get(layer_idx, {})) for layer_idx in range(num_layers)
            ],
            "has_attention_sources_by_layer": [
                "o_proj" in modules.get(layer_idx, {}) for layer_idx in range(num_layers)
            ],
            "has_mlp_destinations_by_layer": [
                _has_mlp_destination(modules.get(layer_idx, {})) for layer_idx in range(num_layers)
            ],
            "has_mlp_sources_by_layer": [
                "down_proj" in modules.get(layer_idx, {}) for layer_idx in range(num_layers)
            ],
        }
        return cls(edge_targets=edge_targets, metadata=metadata)

    def edge_targets(self) -> list[EdgeTarget]:
        return list(self._edge_targets)


def build_graph_metadata_from_model(model: nn.Module, target_modules: Iterable[str]) -> dict:
    target_module_set = {item.strip() for item in target_modules if item.strip()}
    modules = _collect_layer_modules(model)
    if not modules:
        raise ValueError("No transformer layer Linear modules found for dense graph construction.")

    config = getattr(model, "config", None)
    num_layers = max(modules) + 1
    num_attention_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    if num_attention_heads <= 0:
        num_attention_heads = _infer_num_attention_heads(modules)
    if num_attention_heads <= 0:
        raise ValueError("Cannot infer num_attention_heads for dense graph construction.")
    num_key_value_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    if num_key_value_heads <= 0:
        num_key_value_heads = num_attention_heads
    query_heads_per_kv_head = max(1, num_attention_heads // num_key_value_heads)
    parallel_attn_mlp = bool(getattr(config, "parallel_attn_mlp", False))

    source_nodes = _build_source_node_names(
        modules=modules,
        num_attention_heads=num_attention_heads,
        num_layers=num_layers,
    )
    destination_nodes = _build_destination_node_names(
        modules=modules,
        num_attention_heads=num_attention_heads,
        num_layers=num_layers,
    )
    return {
        "graph_version": "dense_head_graph_v1",
        "edge_spec": "Dense residual-stream head-level source/destination construction without neuron-level nodes",
        "n_layers": num_layers,
        "n_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "query_heads_per_kv_head": query_heads_per_kv_head,
        "parallel_attn_mlp": parallel_attn_mlp,
        "n_forward": 1 + num_layers * (num_attention_heads + 1),
        "n_backward": num_layers * (3 * num_attention_heads + 1) + 1,
        "source_nodes": source_nodes,
        "source_node_count": len(source_nodes),
        "destination_nodes": destination_nodes,
        "destination_count": len(destination_nodes),
        "edge_count": _count_graph_edges(
            modules=modules,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            parallel_attn_mlp=parallel_attn_mlp,
        ),
        "target_modules": sorted(target_module_set),
        "target_modules_note": "Dense graph construction is node-based; q/k/v, mlp, logits destinations are included regardless of projection target list.",
        "has_attention_destinations_by_layer": [
            _has_attention_destination(modules.get(layer_idx, {})) for layer_idx in range(num_layers)
        ],
        "has_attention_sources_by_layer": [
            "o_proj" in modules.get(layer_idx, {}) for layer_idx in range(num_layers)
        ],
        "has_mlp_destinations_by_layer": [
            _has_mlp_destination(modules.get(layer_idx, {})) for layer_idx in range(num_layers)
        ],
        "has_mlp_sources_by_layer": [
            "down_proj" in modules.get(layer_idx, {}) for layer_idx in range(num_layers)
        ],
    }


def _make_edge_target(
    source: dict[str, object],
    destination_node: str,
    destination_kind: DestinationKind,
    destination_qkv: str | None,
    destination_head_idx: int | None,
    destination_head_slice: tuple[int, int] | None,
    destination_module: str,
    destination_module_obj: nn.Module,
    destination_input_keys: tuple[str, ...],
    destination_output_key: str | None,
    destination_hook_modules: tuple[tuple[str, nn.Module], ...],
    delta_parameter_names: tuple[str, ...],
    layer_idx: int,
    component_type: str,
) -> EdgeTarget:
    edge_id = f"{source['node']}->{destination_node}"
    return EdgeTarget(
        edge_id=edge_id,
        source_node=str(source["node"]),
        source_module_name=str(source["module_name"]) if source["module_name"] is not None else None,
        source_layer_idx=int(source["layer_idx"]),
        source_component_type=str(source["component_type"]),
        source_module=source["module"],
        source_kind=source["kind"],
        source_head_idx=source["head_idx"],
        source_head_slice=source["head_slice"],
        destination_node=destination_node,
        destination_kind=destination_kind,
        destination_qkv=destination_qkv,
        destination_head_idx=destination_head_idx,
        destination_head_slice=destination_head_slice,
        destination_module=destination_module,
        destination_parameter_name=destination_input_keys[0],
        layer_idx=layer_idx,
        component_type=component_type,
        num_features=_num_features(destination_module_obj, destination_head_slice),
        module=destination_module_obj,
        destination_input_keys=destination_input_keys,
        destination_output_key=destination_output_key,
        destination_hook_modules=destination_hook_modules,
        delta_parameter_names=delta_parameter_names,
    )


def _build_source_nodes(
    modules: dict[int, dict[str, tuple[str, nn.Linear]]],
    num_attention_heads: int,
    num_layers: int,
) -> dict[str, dict[str, object]]:
    source_nodes: dict[str, dict[str, object]] = {
        "input": {
            "node": "input",
            "module_name": None,
            "module": None,
            "layer_idx": -1,
            "component_type": "input",
            "kind": "input",
            "head_idx": None,
            "head_slice": None,
        }
    }
    for layer_idx in range(num_layers):
        layer_modules = modules.get(layer_idx, {})
        o_proj = layer_modules.get("o_proj")
        if o_proj is not None:
            module_name, module = o_proj
            for head_idx in range(num_attention_heads):
                source_nodes[f"a{layer_idx}.h{head_idx}"] = {
                    "node": f"a{layer_idx}.h{head_idx}",
                    "module_name": module_name,
                    "module": module,
                    "layer_idx": layer_idx,
                    "component_type": "o_proj",
                    "kind": "attention",
                    "head_idx": head_idx,
                    "head_slice": _head_slice(module, num_attention_heads, axis="in", head_idx=head_idx),
                }
        down_proj = layer_modules.get("down_proj")
        if down_proj is not None:
            module_name, module = down_proj
            source_nodes[f"m{layer_idx}"] = {
                "node": f"m{layer_idx}",
                "module_name": module_name,
                "module": module,
                "layer_idx": layer_idx,
                "component_type": "down_proj",
                "kind": "mlp",
                "head_idx": None,
                "head_slice": None,
            }
    return source_nodes


def _build_destination_nodes(num_attention_heads: int, num_layers: int) -> list[str]:
    destination_nodes: list[str] = []
    for layer_idx in range(num_layers):
        for qkv in "qkv":
            for head_idx in range(num_attention_heads):
                destination_nodes.append(f"a{layer_idx}.h{head_idx}<{qkv}>")
        destination_nodes.append(f"m{layer_idx}")
    destination_nodes.append("logits")
    return destination_nodes


def _build_source_node_names(
    modules: dict[int, dict[str, tuple[str, nn.Linear]]],
    num_attention_heads: int,
    num_layers: int,
) -> list[str]:
    node_names = ["input"]
    for layer_idx in range(num_layers):
        layer_modules = modules.get(layer_idx, {})
        if "o_proj" in layer_modules:
            node_names.extend(f"a{layer_idx}.h{head_idx}" for head_idx in range(num_attention_heads))
        if "down_proj" in layer_modules:
            node_names.append(f"m{layer_idx}")
    return node_names


def _build_destination_node_names(
    modules: dict[int, dict[str, tuple[str, nn.Linear]]],
    num_attention_heads: int,
    num_layers: int,
) -> list[str]:
    destination_nodes: list[str] = []
    for layer_idx in range(num_layers):
        layer_modules = modules.get(layer_idx, {})
        if _has_attention_destination(layer_modules):
            for qkv in "qkv":
                for head_idx in range(num_attention_heads):
                    destination_nodes.append(f"a{layer_idx}.h{head_idx}<{qkv}>")
        if _has_mlp_destination(layer_modules):
            destination_nodes.append(f"m{layer_idx}")
    destination_nodes.append("logits")
    return destination_nodes


def _count_graph_edges(
    modules: dict[int, dict[str, tuple[str, nn.Linear]]],
    num_layers: int,
    num_attention_heads: int,
    parallel_attn_mlp: bool,
) -> int:
    count = 0
    residual_source_count = 1
    for layer_idx in range(num_layers):
        layer_modules = modules.get(layer_idx, {})
        attention_source_count = num_attention_heads if "o_proj" in layer_modules else 0
        if _has_attention_destination(layer_modules):
            count += residual_source_count * 3 * num_attention_heads
        mlp_source_count = residual_source_count if parallel_attn_mlp else residual_source_count + attention_source_count
        if _has_mlp_destination(layer_modules):
            count += mlp_source_count
        residual_source_count += attention_source_count
        if "down_proj" in layer_modules:
            residual_source_count += 1
    count += residual_source_count
    return count


def _has_attention_destination(layer_modules: dict[str, tuple[str, nn.Linear]]) -> bool:
    return all(name in layer_modules for name in ("q_proj", "k_proj", "v_proj"))


def _has_mlp_destination(layer_modules: dict[str, tuple[str, nn.Linear]]) -> bool:
    return "gate_proj" in layer_modules and "up_proj" in layer_modules


def _collect_layer_modules(model: nn.Module) -> dict[int, dict[str, tuple[str, nn.Linear]]]:
    modules: dict[int, dict[str, tuple[str, nn.Linear]]] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer_idx = _parse_layer_idx(module_name)
        if layer_idx < 0:
            continue
        component_type = module_name.rsplit(".", 1)[-1]
        if component_type not in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
            continue
        modules.setdefault(layer_idx, {})[component_type] = (module_name, module)
    return modules


def _parse_layer_idx(module_name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", module_name)
    if match:
        return int(match.group(1))
    return -1


def _infer_num_attention_heads(modules: dict[int, dict[str, tuple[str, nn.Linear]]]) -> int:
    for layer_modules in modules.values():
        q_proj = layer_modules.get("q_proj")
        o_proj = layer_modules.get("o_proj")
        if q_proj is None or o_proj is None:
            continue
        out_features = int(q_proj[1].out_features)
        in_features = int(o_proj[1].in_features)
        if out_features == in_features:
            return 1
    return 0


def _head_slice(module: nn.Linear, num_heads: int, axis: Literal["in", "out"], head_idx: int) -> tuple[int, int]:
    size = int(module.in_features if axis == "in" else module.out_features)
    if num_heads <= 0 or size % num_heads != 0:
        raise ValueError(f"Cannot split {axis}_features={size} into {num_heads} heads for {module}.")
    head_dim = size // num_heads
    start = int(head_idx) * head_dim
    return start, start + head_dim


def _destination_head_slice(
    module: nn.Linear,
    qkv: str,
    graph_head_idx: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    query_heads_per_kv_head: int,
) -> tuple[int, int]:
    if qkv == "q":
        return _head_slice(module, num_attention_heads, axis="out", head_idx=graph_head_idx)
    kv_head_idx = int(graph_head_idx) // int(query_heads_per_kv_head)
    return _head_slice(module, num_key_value_heads, axis="out", head_idx=kv_head_idx)


def _num_features(module: nn.Module, destination_head_slice: tuple[int, int] | None) -> int:
    if hasattr(module, "in_features"):
        return int(module.in_features)
    if hasattr(module, "weight") and module.weight.ndim == 2:
        return int(module.weight.shape[1])
    return 1


def _logits_destination_module(
    model: nn.Module,
    modules: dict[int, dict[str, tuple[str, nn.Linear]]],
) -> tuple[str, nn.Linear]:
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear):
        return "lm_head", lm_head
    last_layer_idx = max(modules)
    layer_modules = modules[last_layer_idx]
    if "down_proj" in layer_modules:
        return layer_modules["down_proj"]
    if "o_proj" in layer_modules:
        return layer_modules["o_proj"]
    raise ValueError("Cannot find lm_head/down_proj/o_proj module for logits destination hook fallback.")
