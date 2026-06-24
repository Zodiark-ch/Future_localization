from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import torch
import torch.nn.functional as F

from EAP_forLogicalCircuit.graph_registry import EdgeTarget


class DestinationInputCache:
    def __init__(
        self,
        edge_targets: list[EdgeTarget],
        capture_device: torch.device | str = "cpu",
        detach_tensors: bool = True,
        source_capture_device: torch.device | str | None = None,
        detach_source_tensors: bool | None = None,
        output_grad_capture_device: torch.device | str | None = None,
        detach_output_grads: bool | None = None,
    ):
        self.edge_targets = edge_targets
        self.capture_device = torch.device(capture_device)
        self.detach_tensors = bool(detach_tensors)
        self.source_capture_device = torch.device(source_capture_device) if source_capture_device is not None else self.capture_device
        self.detach_source_tensors = self.detach_tensors if detach_source_tensors is None else bool(detach_source_tensors)
        self.output_grad_capture_device = (
            torch.device(output_grad_capture_device) if output_grad_capture_device is not None else self.capture_device
        )
        self.detach_output_grads = self.detach_tensors if detach_output_grads is None else bool(detach_output_grads)
        self.mode: Literal["idle", "corrupted", "clean"] = "idle"
        self.clean_inputs: dict[str, torch.Tensor] = {}
        self.corrupted_inputs: dict[str, torch.Tensor] = {}
        self.clean_source_outputs: dict[str, torch.Tensor] = {}
        self.corrupted_source_outputs: dict[str, torch.Tensor] = {}
        self.input_grads: dict[str, torch.Tensor] = {}
        self.output_grads: dict[str, torch.Tensor] = {}
        self._handles = []
        self._destination_meta = self._build_destination_meta(edge_targets)

    def register(self) -> None:
        if self._handles:
            return
        seen_destination_hooks = set()
        for edge_target in self.edge_targets:
            hook_modules = edge_target.destination_hook_modules or ((edge_target.destination_parameter_name, edge_target.module),)
            for key, module in hook_modules:
                hook_id = (key, id(module))
                if hook_id in seen_destination_hooks:
                    continue
                seen_destination_hooks.add(hook_id)
                self._handles.append(module.register_forward_hook(self._make_forward_hook(key)))
        seen_source_nodes = set()
        for edge_target in self.edge_targets:
            source_module = edge_target.source_module
            if source_module is None:
                continue
            if edge_target.source_node in seen_source_nodes:
                continue
            seen_source_nodes.add(edge_target.source_node)
            self._handles.append(
                source_module.register_forward_hook(
                    self._make_source_forward_hook(edge_target)
                )
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def clear_batch(self) -> None:
        self.clean_inputs.clear()
        self.corrupted_inputs.clear()
        self.clean_source_outputs.clear()
        self.corrupted_source_outputs.clear()
        self.input_grads.clear()
        self.output_grads.clear()

    @contextmanager
    def capture(self, mode: Literal["clean", "corrupted"]):
        previous = self.mode
        self.mode = mode
        try:
            yield self
        finally:
            self.mode = previous

    def _make_forward_hook(self, key: str):
        destination_meta = self._destination_meta.get(key, {})

        def hook(_module, inputs, output):
            if self.mode == "idle":
                return None
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            input_activation = inputs[0]
            if self.mode == "clean":
                stored = self._store_tensor(input_activation)
                self.clean_inputs[key] = stored
                self._maybe_store_input_source(destination_meta, input_activation, clean=True)
                return None
            stored = self._store_tensor(input_activation)
            self.corrupted_inputs[key] = stored
            self._maybe_store_input_source(destination_meta, input_activation, clean=False)
            if input_activation.requires_grad:
                input_activation.register_hook(self._make_grad_hook(key))
            if torch.is_tensor(output) and output.requires_grad:
                output.register_hook(self._make_output_grad_hook(key))
            return None

        return hook

    def _make_source_forward_hook(self, edge_target: EdgeTarget):
        source_node = edge_target.source_node

        def hook(module, inputs, output):
            if self.mode == "idle":
                return None
            source_tensor = _source_tensor_from_hook(module, edge_target, inputs, output)
            if source_tensor is None:
                return None
            if self.mode == "clean":
                self.clean_source_outputs[source_node] = self._store_source_tensor(source_tensor)
            else:
                self.corrupted_source_outputs[source_node] = self._store_source_tensor(source_tensor)
            return None

        return hook

    def _make_grad_hook(self, key: str):
        def grad_hook(grad: torch.Tensor):
            self.input_grads[key] = self._store_tensor(grad)
            return grad

        return grad_hook

    def _make_output_grad_hook(self, key: str):
        def grad_hook(grad: torch.Tensor):
            self.output_grads[key] = self._store_output_grad(grad)
            return grad

        return grad_hook

    def _store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.detach_tensors:
            return tensor.detach().to(self.capture_device)
        return tensor

    def _store_source_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.detach_source_tensors:
            return tensor.detach().to(self.source_capture_device)
        return tensor

    def _store_output_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.detach_output_grads:
            return tensor.detach().to(self.output_grad_capture_device)
        return tensor

    def _maybe_store_input_source(self, destination_meta: dict[str, object], tensor: torch.Tensor, clean: bool) -> None:
        layer_idx = int(destination_meta.get("layer_idx", -1))
        component_type = str(destination_meta.get("component_type", ""))
        if layer_idx != 0 or component_type not in {"q_proj", "k_proj", "v_proj"}:
            return
        if clean:
            self.clean_source_outputs.setdefault("input", self._store_source_tensor(tensor))
        else:
            self.corrupted_source_outputs.setdefault("input", self._store_source_tensor(tensor))

    @staticmethod
    def _build_destination_meta(edge_targets: list[EdgeTarget]) -> dict[str, dict[str, object]]:
        meta: dict[str, dict[str, object]] = {}
        for target in edge_targets:
            for key in target.destination_input_keys:
                if key in meta:
                    continue
                meta[key] = {
                    "layer_idx": target.layer_idx,
                    "component_type": target.component_type,
                }
        return meta


def _source_tensor_from_hook(module, edge_target: EdgeTarget, inputs, output) -> torch.Tensor | None:
    if edge_target.source_kind != "attention":
        return output if torch.is_tensor(output) else None
    if not inputs or not torch.is_tensor(inputs[0]) or edge_target.source_head_slice is None:
        return None
    start, end = edge_target.source_head_slice
    head_input = inputs[0][..., start:end]
    if not hasattr(module, "weight"):
        return None
    weight = module.weight[:, start:end]
    bias = getattr(module, "bias", None)
    return F.linear(head_input, weight, bias=None if bias is None else bias / max(1, _head_count_from_slice(inputs[0], edge_target)))


def _head_count_from_slice(input_tensor: torch.Tensor, edge_target: EdgeTarget) -> int:
    if edge_target.source_head_slice is None:
        return 1
    start, end = edge_target.source_head_slice
    width = max(1, end - start)
    return max(1, int(input_tensor.size(-1)) // width)
