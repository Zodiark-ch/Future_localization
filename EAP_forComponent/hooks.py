from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import torch

from EAP_forComponent.schemas import ComponentTarget


class ComponentActivationCache:
    def __init__(self, targets: list[ComponentTarget], capture_device: torch.device | str = "cpu"):
        self.targets = targets
        self.capture_device = torch.device(capture_device)
        self.mode: Literal["idle", "corrupted", "clean"] = "idle"
        self.corrupted_outputs: dict[str, torch.Tensor] = {}
        self.clean_outputs: dict[str, torch.Tensor] = {}
        self.output_grads: dict[str, torch.Tensor] = {}
        self.corrupted_inputs: dict[str, torch.Tensor] = {}
        self.clean_inputs: dict[str, torch.Tensor] = {}
        self._handles = []

    def register(self) -> None:
        if self._handles:
            return
        seen_modules = set()
        for target in self.targets:
            module_id = id(target.module)
            if module_id in seen_modules:
                continue
            seen_modules.add(module_id)
            self._handles.append(target.module.register_forward_hook(self._make_forward_hook(target)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def clear_batch(self) -> None:
        self.corrupted_outputs.clear()
        self.clean_outputs.clear()
        self.output_grads.clear()
        self.corrupted_inputs.clear()
        self.clean_inputs.clear()

    @contextmanager
    def capture(self, mode: Literal["corrupted", "clean"]):
        previous_mode = self.mode
        self.mode = mode
        try:
            yield self
        finally:
            self.mode = previous_mode

    def _make_forward_hook(self, target: ComponentTarget):
        parameter_name = target.parameter_name

        def hook(_module, inputs, output):
            if self.mode == "idle" or not torch.is_tensor(output):
                return None
            input_activation = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            if self.mode == "corrupted":
                self.corrupted_outputs[parameter_name] = output.detach().to(self.capture_device)
                if input_activation is not None:
                    self.corrupted_inputs[parameter_name] = input_activation.detach().to(self.capture_device)
                if output.requires_grad:
                    output.register_hook(self._make_grad_hook(parameter_name))
                return None
            self.clean_outputs[parameter_name] = output.detach().to(self.capture_device)
            if input_activation is not None:
                self.clean_inputs[parameter_name] = input_activation.detach().to(self.capture_device)
            if output.requires_grad:
                output.register_hook(self._make_grad_hook(parameter_name))
            return None

        return hook

    def _make_grad_hook(self, parameter_name: str):
        def grad_hook(grad):
            self.output_grads[parameter_name] = grad.detach().to(self.capture_device)
            return grad

        return grad_hook
