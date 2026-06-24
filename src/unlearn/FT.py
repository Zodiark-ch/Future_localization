import torch
from transformers import Trainer

from .base import BaseTrainer


class FT(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # 处理多个retain数据集的情况
        retain_loss = 0.0
        retain_count = 0
        
        for key, retain_data in inputs.items():
            if key.startswith("retain") and retain_data is not None:
                retain_inputs = {
                    "input_ids": retain_data[0],
                    "attention_mask": retain_data[1],
                    "labels": retain_data[2],
                }

                outputs = model(**retain_inputs)
                retain_loss += outputs.loss
                retain_count += 1
        
        # 如果有retain数据集，计算平均损失
        if retain_count > 0:
            loss = retain_loss / retain_count
        else:
            # 如果没有retain数据集，返回零损失
            loss = torch.tensor(0.0, device=next(model.parameters()).device)

        return (loss, outputs) if return_outputs else loss


class FT_l1(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        retain_data = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        outputs = model(**retain_inputs)

        l1_norms = [torch.norm(p, 1) for p in model.parameters() if p.requires_grad]

        loss = outputs.loss + self.alpha * sum(l1_norms)

        return (loss, outputs) if return_outputs else loss
