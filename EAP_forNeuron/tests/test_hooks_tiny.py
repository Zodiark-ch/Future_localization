import torch
from torch import nn

from EAP_forNeuron.hooks import LinearActivationCache
from EAP_forNeuron.schemas import NeuronTarget


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.proj(x)


def test_linear_activation_cache_captures_inputs_and_output_grads():
    model = TinyModel()
    flat_indices = torch.tensor([0, 2, 4])
    target = NeuronTarget(
        parameter_name="proj.weight",
        module_name="proj",
        module=model.proj,
        weight=model.proj.weight,
        shape=model.proj.weight.shape,
        flat_indices=flat_indices,
        weight_values=model.proj.weight.detach().flatten()[flat_indices].float().cpu(),
    )
    cache = LinearActivationCache([target], capture_device="cpu")
    cache.register()
    try:
        with torch.no_grad(), cache.capture("corrupted"):
            model(torch.ones(1, 2, 3))
        with cache.capture("clean"):
            output = model(torch.zeros(1, 2, 3))
            loss = output.sum()
        loss.backward()
    finally:
        cache.remove()

    assert cache.corrupted_inputs["proj.weight"].shape == (1, 2, 3)
    assert cache.clean_inputs["proj.weight"].shape == (1, 2, 3)
    assert cache.output_grads["proj.weight"].shape == (1, 2, 2)
