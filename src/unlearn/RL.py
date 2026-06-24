import torch
from transformers import Trainer

from .base import BaseTrainer


class RL(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]
        # 获取第一个可用的retain数据集
        retain_data = None
        for key, data in inputs.items():
            if key.startswith("retain") and data is not None:
                retain_data = data
                break
        
        if retain_data is None:
            # 如果没有retain数据集，返回零损失
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        labels = forget_data[2]
        retain_labels = retain_data[2]
        new_labels = []

        if retain_labels.shape[1] < labels.shape[1]:
            for label in retain_labels.cpu().tolist():
                label = label + [-100] * (labels.shape[1] - retain_labels.shape[1])
                new_labels.append(label)
        else:
            for label in retain_labels.cpu().tolist():
                new_labels.append(label[: labels.shape[1]])

        new_labels = torch.tensor(new_labels).to(labels.device)
        input_ids = forget_data[0].clone()
        positions = forget_data[4]
        pad_id = input_ids[0][-1].item()
        for idx, position in enumerate(positions):
            input_ids[idx, position:] = new_labels[idx][position:].clone()
            mask = input_ids[idx] == -100
            input_ids[idx, mask] = pad_id
        forget_inputs = {
            "input_ids": input_ids,
            "attention_mask": forget_data[1],
            "labels": new_labels,
        }

        outputs = model(**forget_inputs)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
