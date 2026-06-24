from copy import deepcopy

import torch
from transformers import Trainer

from modeling_patches import sequential_position_ids
from pruner.utils import find_layers
from training_losses import (
    kl_divergence as _kl_divergence,
    model_vocab_size as _model_vocab_size,
    sanitize_labels as _sanitize_labels,
    task_loss as _task_loss,
    trim_to_active_length as _trim_to_active_length,
)


class BaseFinetuningTrainer(Trainer):
    def __init__(
        self,
        eval_collector=None,
        alpha=0.0,
        target_weight=1.0,
        pervasiveness_weight=1.0,
        kl_weight=1.0,
        use_reference_model=False,
        mask=None,
        if_wanda=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_collector = eval_collector
        self.alpha = alpha or 0.0
        self.target_weight = target_weight if target_weight is not None else 1.0
        self.pervasiveness_weight = (
            pervasiveness_weight if pervasiveness_weight is not None else 1.0
        )
        self.kl_weight = kl_weight if kl_weight is not None else 1.0
        self.mask = mask
        self.if_wanda = if_wanda
        self.reference_model = None
        if use_reference_model:
            self.reference_model = deepcopy(self.model)
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad_(False)

    def _pervasiveness_items(self, inputs):
        for key, value in inputs.items():
            if key.startswith("pervasiveness") and value is not None:
                yield key, value

    def target_loss(self, model, inputs):
        loss, outputs = _task_loss(model, inputs.get("target"))
        if loss is None:
            loss = torch.tensor(0.0, device=next(model.parameters()).device)
        return loss, outputs

    def pervasiveness_ce_loss(self, model, inputs):
        losses = []
        last_outputs = None
        for _key, data in self._pervasiveness_items(inputs):
            loss, outputs = _task_loss(model, data)
            losses.append(loss)
            last_outputs = outputs
        if not losses:
            return torch.tensor(0.0, device=next(model.parameters()).device), last_outputs
        return sum(losses) / len(losses), last_outputs

    def pervasiveness_kl_loss(self, model, inputs):
        if self.reference_model is None:
            raise ValueError("TargetFT+PervasivenessKL requires a reference_model")
        losses = []
        last_outputs = None
        for _key, data in self._pervasiveness_items(inputs):
            input_ids, attention_mask, raw_labels = _trim_to_active_length(data[0], data[1], data[2])
            current_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": sequential_position_ids(input_ids),
                "use_cache": False,
            }
            labels = _sanitize_labels(raw_labels, attention_mask, _model_vocab_size(model))
            current_outputs = model(**current_inputs)
            with torch.no_grad():
                reference_outputs = self.reference_model(**current_inputs)
            losses.append(
                _kl_divergence(
                    current_outputs.logits,
                    reference_outputs.logits,
                    labels,
                )
            )
            last_outputs = current_outputs
        if not losses:
            return torch.tensor(0.0, device=next(model.parameters()).device), last_outputs
        return sum(losses) / len(losses), last_outputs

    def l1_loss(self, model):
        losses = [torch.norm(parameter, 1) for parameter in model.parameters() if parameter.requires_grad]
        if not losses:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        return sum(losses)

    def mask_gradient(self, model, if_wanda=False):
        if self.mask is None:
            return
        if not if_wanda:
            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if tensor.grad is not None and key in self.mask:
                        tensor.grad *= self.mask[key].to(tensor.grad.device)
            return

        layers = (
            model.model.layers
            if hasattr(model.model, "layers")
            else model.model.decoder.layers
        )
        cnt = 0
        for layer in layers:
            subset = find_layers(layer)
            for name in subset:
                key = cnt if cnt in self.mask else str(cnt)
                if key in self.mask and subset[name].weight.grad is not None:
                    subset[name].weight.grad *= self.mask[key].to(subset[name].weight.grad.device)
                cnt += 1
