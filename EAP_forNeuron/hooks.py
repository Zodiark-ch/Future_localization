from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import torch

from EAP_forNeuron.schemas import NeuronTarget


class LinearActivationCache:
    def __init__(self, targets: list[NeuronTarget], capture_device: torch.device | str = "cpu"):
        self.targets = targets
        self.capture_device = torch.device(capture_device)
        self.mode: Literal["idle", "corrupted", "clean"] = "idle"
        self.corrupted_inputs: dict[str, torch.Tensor] = {}
        self.clean_inputs: dict[str, torch.Tensor] = {}
        self.output_grads: dict[str, torch.Tensor] = {}
        self._handles = []

    def register(self) -> None:
        if self._handles:
            return
        for target in self.targets:
            self._handles.append(target.module.register_forward_hook(self._make_forward_hook(target)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def clear_batch(self) -> None:
        self.corrupted_inputs.clear()
        self.clean_inputs.clear()
        self.output_grads.clear()

    @contextmanager
    def capture(self, mode: Literal["corrupted", "clean"]):
        previous_mode = self.mode
        self.mode = mode
        try:
            yield self
        finally:
            self.mode = previous_mode

    def _make_forward_hook(self, target: NeuronTarget):
        def hook(_module, inputs, output):
            if self.mode == "idle":
                return None
            if not inputs:
                return None
            activation = inputs[0]
            if not torch.is_tensor(activation):
                return None
            if self.mode == "corrupted":
                self.corrupted_inputs[target.parameter_name] = activation.detach().to(self.capture_device)
                return None
            self.clean_inputs[target.parameter_name] = activation.detach().to(self.capture_device)
            if torch.is_tensor(output) and output.requires_grad:
                output.register_hook(self._make_grad_hook(target.parameter_name))
            return None
        return hook

    def _make_grad_hook(self, parameter_name: str):
        def grad_hook(grad):
            self.output_grads[parameter_name] = grad.detach().to(self.capture_device)
            return grad
        return grad_hook
