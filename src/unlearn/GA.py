import torch
from transformers import Trainer

from .base import BaseTrainer
from .KL import kl_loss


class GA(BaseTrainer):
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

        loss = -outputs.loss

        return (loss, outputs) if return_outputs else loss


class GA_FT(GA):
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
                retain_loss += retain_outputs.loss
                retain_count += 1
        
        # 如果有retain数据集，计算平均损失
        if retain_count > 0:
            retain_loss = retain_loss / retain_count
            loss = forget_loss + self.gamma * retain_loss
        else:
            loss = forget_loss
            
        return (loss, outputs) if return_outputs else loss


class GA_KL(GA):
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
            loss = forget_loss + (1 - self.gamma) * retain_loss
        else:
            loss = forget_loss

        return (loss, outputs) if return_outputs else loss
class NPO(BaseTrainer):
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
        current_forget_loss = outputs.loss

        with torch.no_grad():
            ref_outputs = self.infer_model(**forget_inputs)
            ref_forget_loss = ref_outputs.loss
        
        neg_log_ratios = current_forget_loss - ref_forget_loss

        
        forget_loss = -torch.nn.functional.logsigmoid(0.1*neg_log_ratios).mean()*2/0.1

        loss = forget_loss

        return (loss, outputs) if return_outputs else loss
    
class NPO_FT(BaseTrainer):
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
        current_forget_loss = outputs.loss

        with torch.no_grad():
            ref_outputs = self.infer_model(**forget_inputs)
            ref_forget_loss = ref_outputs.loss
        
        neg_log_ratios = current_forget_loss - ref_forget_loss

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
                retain_loss += retain_outputs.loss
                retain_count += 1
        
        # 如果有retain数据集，计算平均损失
        if retain_count > 0:
            retain_loss = retain_loss / retain_count
        else:
            retain_loss = 0.0
        
        forget_loss = -torch.nn.functional.logsigmoid(0.1*neg_log_ratios).mean()*2/0.1

        loss = forget_loss + self.gamma * retain_loss

        return (loss, outputs) if return_outputs else loss

           