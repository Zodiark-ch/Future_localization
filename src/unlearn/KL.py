import torch
from transformers import Trainer

from .base import BaseTrainer


def kl_loss(prob_p, prob_q):
    return -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()


class KL(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        forget_outputs = model(**forget_inputs)

        with torch.no_grad():
            infer_forget_outputs = self.infer_model(**forget_inputs)

        prob_forget_p = torch.softmax(forget_outputs.logits, dim=-1)
        prob_forget_q = torch.softmax(infer_forget_outputs.logits, dim=-1)

        forget_loss = kl_loss(prob_forget_p, prob_forget_q)

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

                retain_outputs = model(**retain_inputs)

                with torch.no_grad():
                    infer_retain_outputs = self.infer_model(**retain_inputs)

                prob_retain_p = torch.softmax(retain_outputs.logits, dim=-1)
                prob_retain_q = torch.softmax(infer_retain_outputs.logits, dim=-1)

                retain_loss += kl_loss(prob_retain_p, prob_retain_q)
                retain_count += 1
        
        # 如果有retain数据集，计算平均损失
        if retain_count > 0:
            retain_loss = retain_loss / retain_count
            loss = -self.gamma * forget_loss + retain_loss
        else:
            loss = -self.gamma * forget_loss

        return (loss, forget_outputs) if return_outputs else loss


class KL_GA(KL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        outputs = model(**forget_inputs)

        forget_loss = -outputs.loss

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

                retain_outputs = model(**retain_inputs)

                with torch.no_grad():
                    infer_retain_outputs = self.infer_model(**retain_inputs)

                prob_retain_p = torch.softmax(retain_outputs.logits, dim=-1)
                prob_retain_q = torch.softmax(infer_retain_outputs.logits, dim=-1)

                retain_loss += kl_loss(prob_retain_p, prob_retain_q)
                retain_count += 1
        
        # 如果有retain数据集，计算平均损失
        if retain_count > 0:
            retain_loss = retain_loss / retain_count
            loss = self.gamma * forget_loss + retain_loss
        else:
            loss = self.gamma * forget_loss

        return (loss, outputs) if return_outputs else loss


class KL_CL(KL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[3],
        }

        outputs = model(**forget_inputs)

        forget_loss = outputs.loss

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

                retain_outputs = model(**retain_inputs)

                with torch.no_grad():
                    infer_retain_outputs = self.infer_model(**retain_inputs)

                prob_retain_p = torch.softmax(retain_outputs.logits, dim=-1)
                prob_retain_q = torch.softmax(infer_retain_outputs.logits, dim=-1)

                retain_loss += kl_loss(prob_retain_p, prob_retain_q)
                retain_count += 1
        
        # 如果有retain数据集，计算平均损失
        if retain_count > 0:
            retain_loss = retain_loss / retain_count
            loss = self.gamma * forget_loss + retain_loss
        else:
            loss = self.gamma * forget_loss

        return (loss, outputs) if return_outputs else loss
