#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.


import logging
import os
import psutil  # 添加系统监控
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLE_GPU_STATS"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"
# 设置内存优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# 设置CUDA内存分配策略
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 启用同步CUDA操作，便于调试（已注释，避免性能下降）
import torch
import gc
import pickle
import random
import sys
import json
import warnings
import math
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch.nn as nn
from torch.optim import AdamW

import sys
sys.path.append(
    os.path.join(
        os.getcwd(),
        "Edge-Pruning/src/modeling/"
    )
)   # Very hacky but the imports are annoying otherwise
from modeling_fmistral import FMistralForCausalLM

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

class FMistralInfoTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.target_edge_sparsity = kwargs.pop('target_edge_sparsity', 0.0)
        self.start_edge_sparsity = kwargs.pop('start_edge_sparsity', 0.0)
        self.target_layer_sparsity = kwargs.pop('target_layer_sparsity', 0.0)
        self.start_layer_sparsity = kwargs.pop('start_layer_sparsity', 0.0)
        if "num_edge_sparsity_warmup_steps" in kwargs:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_edge_sparsity_warmup_steps')
        else:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', 0)
        if "num_layer_sparsity_warmup_steps" in kwargs:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_layer_sparsity_warmup_steps')
        else:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', self.num_edge_sparsity_warmup_steps)
        _ = kwargs.pop('num_sparsity_warmup_steps', None)
        self.warmup_type = kwargs.pop('warmup_type', 'linear')
        self.mistral_model = kwargs.pop('mistral_model', None)
        self.skip_layer_loss_if_higher_sparsity = kwargs.pop('skip_layer_loss_if_higher_sparsity', False)
        self.device_count = torch.cuda.device_count()
        super().__init__(*args, **kwargs)



    def get_current_edge_target_sparsity(self, global_step):
        if global_step < self.num_edge_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_edge_sparsity + (self.target_edge_sparsity - self.start_edge_sparsity) * 
                    global_step / self.num_edge_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_edge_sparsity) + (math.log(1 - self.target_edge_sparsity) - 
                    math.log(1 - self.start_edge_sparsity)) * global_step / self.num_edge_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_edge_sparsity
        
    def get_current_layer_target_sparsity(self, global_step):
        if global_step < self.num_layer_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_layer_sparsity + (self.target_layer_sparsity - self.start_layer_sparsity) * 
                    global_step / self.num_layer_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_layer_sparsity) + (math.log(1 - self.target_layer_sparsity) - 
                    math.log(1 - self.start_layer_sparsity)) * global_step / self.num_layer_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_layer_sparsity

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        start_idxes = inputs.pop("start_idxes")
        end_idxes = inputs.pop("end_idxes")
        _ = inputs.pop("labels")
        corr_input_ids = inputs.pop("corr_input_ids")
        input_ids = inputs.pop("input_ids")
        
        bsz = input_ids.shape[0]
        
        with torch.no_grad():
            # First get the logits from the Mistral model
            mistral_output = self.mistral_model(input_ids=input_ids, **inputs)
            logits_mistral = mistral_output.logits
            
            # Now run the corrupted inputs through it, and retain the activations
            corr_x = self.mistral_model(
                input_ids=corr_input_ids, 
                **inputs,
                output_writer_states=True
            ).writer_states
            

            # Reshape corr_x in case we have distributed training
            # tgt_shape = (-1, bsz, *corr_x.shape[2:])
            # corr_x = corr_x.reshape(tgt_shape)
        
        current_edge_sparsity = self.get_current_edge_target_sparsity(self.state.global_step)
        current_node_sparsity = self.get_current_layer_target_sparsity(self.state.global_step)
        
        # 添加错误处理和内存监控
        try:
            outputs = model(
                input_ids=input_ids,
                **inputs, 
                target_edge_sparsity=current_edge_sparsity,
                target_node_sparsity=current_node_sparsity,
                corr_x=corr_x
            )
        except RuntimeError as e:
            print(f"CUDA错误: {e}")
            print("尝试清理GPU内存并重试...")
            # 暂时禁用自动缓存清理，避免CUDA错误
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            raise e
        
        # 处理 edge_loss 和 node_loss
        reg_edge_loss = outputs["edge_loss"]
        
        if self.skip_layer_loss_if_higher_sparsity and outputs["model_node_sparsity"] > outputs["target_node_sparsity"]:
            reg_layer_loss = torch.tensor(0.0, device=outputs["logits"].device)
        else:
            reg_layer_loss = outputs["node_loss"]
                
        reg_loss = reg_edge_loss + reg_layer_loss
        logits = outputs["logits"]
        
        kl_loss = 0
        for i in range(logits.shape[0]):
            start_idx = start_idxes[i]
            end_idx = end_idxes[i]
            
            # 添加调试信息
            # if self.state.global_step % 100 == 0:
                # print(f"  Sample {i}: start_idx={start_idx}, end_idx={end_idx}")
                # print(f"    logits shape: {logits[i].shape}")
                # print(f"    logits_mistral shape: {logits_mistral[i].shape}")
                # print(f"    logits slice shape: {logits[i, start_idx:end_idx].shape}")
                # print(f"    logits_mistral slice shape: {logits_mistral[i, start_idx:end_idx].shape}")
            
            logits_i = nn.functional.log_softmax(logits[i, start_idx:end_idx], dim=-1)
            logits_mistral_i = nn.functional.log_softmax(logits_mistral[i, start_idx:end_idx], dim=-1)
            
            kl_loss_component = nn.functional.kl_div(logits_i, logits_mistral_i, reduction='batchmean', log_target=True)
            kl_loss += kl_loss_component
            
            # 添加调试信息
            if self.state.global_step % 100 == 0:
                print(f"    kl_loss_component: {kl_loss_component.item():.6f}")
            
                
        
        kl_loss /= logits.shape[0]
        
        loss = kl_loss + reg_loss
        outputs["loss"] = loss
        outputs["kl_loss"] = kl_loss

        # 清理GPU内存
        del logits_mistral, corr_x
        # 暂时禁用自动缓存清理，避免CUDA错误
        # torch.cuda.empty_cache()
        gc.collect()

        return (loss, outputs) if return_outputs else loss

@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(
        default="./data/dataset/ioi/",
        metadata={"help": "The path to the directory with the JSON files of the task."},
    )
    train_split: Optional[str] = field(
        default="train",
        metadata={"help": "The split to use for training."},
    )
    max_train_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=64,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    start_edge_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial edge sparsity of the model."}
    )
    target_edge_sparsity: Optional[float] = field(
        default=0.97,
        metadata={"help": "The target edge sparsity of the model."}
    )
    start_layer_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial layer sparsity of the model."}
    )
    target_layer_sparsity: Optional[float] = field(
        default=0.72,
        metadata={"help": "The target layer sparsity of the model."}
    )
    stop_optimizing_layer_if_higher_sparsity: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to stop optimizing the layer sparsity if it is higher than the target."}
    )
    num_sparsity_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "The number of steps to reach the target sparsity."}
    )
    edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    layer_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_layer_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    warmup_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The type of warmup to use for the regularization term."}
    )
    with_embedding_nodes: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to include the embedding nodes"}
    )
    disable_linear_reg_term: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the linear regularization term."}
    )
    disable_node_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable node loss."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    initialize_from: str = field(
        default="HuggingFaceH4/zephyr-7b-beta",
        metadata={"help": "The model to initialize from."},
    )

def format_instance(instance, split):
    if isinstance(instance, dict) and "min_steps" in instance:
        return {
            "tokens": instance["tokens"],
            "split": split,
            "min_steps": instance["min_steps"],
        }
    else:
        return {
            "tokens": instance,
            "split": split,
        }

def load_datasets(dataset_path, max_train_samples, max_eval_samples, train_split="train"):
    # 检查是否是包含train和validation子目录的结构
    train_path = os.path.join(dataset_path, "train")
    validation_path = os.path.join(dataset_path, "validation")
    
    if os.path.exists(train_path) and os.path.exists(validation_path):
        # 加载train和validation子目录
        train_dataset = load_from_disk(train_path)
        validation_dataset = load_from_disk(validation_path)
        
        dataset = DatasetDict({
            train_split: train_dataset,
            "validation": validation_dataset,
        })
    elif os.path.exists(dataset_path):
        # 直接加载数据集目录
        dataset = load_from_disk(dataset_path)
    else:
        # 从HuggingFace Hub加载
        dataset = load_dataset(dataset_path)
    
    # 处理样本数量限制
    if "validation" not in dataset:
        assert max_eval_samples is not None, "Validation set is missing! (val)"
        assert max_train_samples is not None, "Validation set is missing! (train)"
        dataset = DatasetDict({
            train_split: dataset[train_split].select(range(max_train_samples)),
            "validation": dataset[train_split].select(range(max_train_samples, max_train_samples+max_eval_samples)),
        })
    else:
        if max_train_samples is not None and max_train_samples < len(dataset[train_split]):
            dataset[train_split] = dataset[train_split].select(range(max_train_samples))
        if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]):
            dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    
    return dataset

baba_templates = [
    "Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {B} and {A} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument, and afterwards {B} said to {A}",
    "After {B} and {A} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {B} and {A} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {B} and {A} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {B} and {A} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {B} and {A} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]

abba_templates = [
    "Then, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {A} and {B} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {A} and {B} had a long argument, and afterwards {B} said to {A}",
    "After {A} and {B} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {A} and {B} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {A} and {B} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {A} and {B} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {A} and {B} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {A} and {B} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]

def try_fit_template(string, template):
    pieces_s, pieces_t = string.strip(), template.strip()
    
    mapping = {}
    
    for s, t in zip(pieces_s.split(), pieces_t.split()):
        if s == t:
            continue
        if s[-1] == t[-1] and s[-1] in [',', '.']:
            s, t = s[:-1], t[:-1]
        if t not in ['{A}', '{B}', '{PLACE}', '{OBJECT}']:
            return None
        elif t[1:-1].lower() in mapping:
            if mapping[t[1:-1].lower()] != s:
                return None
        else:
            mapping[t[1:-1].lower()] = s
    
    if 'place' not in mapping:
        mapping['place'] = None
    if 'object' not in mapping:
        mapping['object'] = None
    
    return mapping

def find_template(string):
    for template in baba_templates:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({
                'template': template,
                'order': 'baba'
            })
            return mapping
    
    for template in abba_templates:
        mapping = try_fit_template(string, template)
        if mapping is not None:
            mapping.update({
                'template': template,
                'order': 'abba'
            })
            return mapping
    return None

class DataCollatorIOI:
    def __init__(
        self, 
        tokenizer,
        max_length
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        input_ids_all = []
        corr_input_ids_all = []
        labels_all = []
        start_idxes = []
        end_idxes = []
        
        key = "text" if "text" in examples[0] else "sentence"
        
        for example in examples:
            text = example[key]
            corr_text = example["corr_"+key]
            
            last_space = text.rfind(' ')
            non_loss_portion = text[:last_space]
            
            len_non_loss = self.tokenizer(non_loss_portion, return_tensors="pt").input_ids.shape[1]
            input_ids = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True).input_ids[0]
            corr_input_ids = self.tokenizer(corr_text, return_tensors="pt", max_length=self.max_length, padding='max_length', truncation=True).input_ids[0]
            labels = input_ids.clone()
            labels[:len_non_loss] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            input_ids_all.append(input_ids)
            corr_input_ids_all.append(corr_input_ids)
            labels_all.append(labels)
            
            # 安全地找到第一个padding位置
            pad_positions = (input_ids == self.tokenizer.pad_token_id).nonzero()
            if pad_positions.numel() > 0:
                first_pad = pad_positions[0]
                end_idx = first_pad - 1
            else:
                # 如果没有padding，使用序列长度
                first_pad = len(input_ids)
                end_idx = len(input_ids) - 1
            
            start_idxes.append(len_non_loss-1)
            end_idxes.append(end_idx)
        
        batch = {
            "input_ids": torch.stack(input_ids_all),
            "corr_input_ids": torch.stack(corr_input_ids_all),
            "labels": torch.stack(labels_all),
            "start_idxes": torch.LongTensor(start_idxes),
            "end_idxes": torch.LongTensor(end_idxes),
        }

        return batch     

def eval_fn(eval_pred): 
    logits, target_edge_sparsity, target_layer_sparsity, model_edge_sparsity, model_layer_sparsity, reg_edge_loss, reg_layer_loss, kl_loss = eval_pred.predictions
    if len(model_edge_sparsity.shape) > 0:
        model_edge_sparsity = model_edge_sparsity[0].item()
        model_layer_sparsity = model_layer_sparsity[0].item()
        target_edge_sparsity = target_edge_sparsity[0].item()
        target_layer_sparsity = target_layer_sparsity[0].item()
    else:
        model_edge_sparsity = model_edge_sparsity.item()
        model_layer_sparsity = model_layer_sparsity.item()
        target_edge_sparsity = target_edge_sparsity.item()
        target_layer_sparsity = target_layer_sparsity.item()
    
    predictions = np.argmax(logits, axis=-1)[:, :-1]
    labels = eval_pred.label_ids[:, 1:]

    eval_mask = (labels != -100).astype(int)
    predictions = predictions * eval_mask
    labels = labels * eval_mask
    
    correct = (predictions == labels).all(axis=1)
    accuracy = correct.sum().item() / correct.shape[0]
    
    kl_loss = kl_loss.mean().item()
    reg_edge_loss = reg_edge_loss.mean().item()
    reg_layer_loss = reg_layer_loss.mean().item()
    
    return {
        "eval_accuracy": accuracy,
        "model_edge_sparsity": model_edge_sparsity,
        "model_layer_sparsity": model_layer_sparsity,
        "target_edge_sparsity": target_edge_sparsity,
        "target_layer_sparsity": target_layer_sparsity,
        "eval_kl_loss": kl_loss,
        "eval_reg_edge_loss": reg_edge_loss,
        "eval_reg_layer_loss": reg_layer_loss,
    }
    
def print_memory_usage():
    """打印当前内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"CPU Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            memory_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
            print(f"GPU {i} Memory: {memory_allocated:.1f} MB allocated, {memory_reserved:.1f} MB reserved")

def freeze_all_except_pruning_params(model):
    for n, p in model.named_parameters():
        if 'log_alpha' in n or 'sparsity_lambda' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def get_optimizers(model, edges_lr, layers_lr, reg_edges_lr, reg_layers_lr, num_training_steps, warmup_steps=0, disable_node_loss=False):
    optimizer_1_group = []
    optimizer_2_group = []
    optimizer_3_group = []
    optimizer_4_group = []

    for n, p in model.named_parameters():
        if 'write_log_alpha' in n:
            optimizer_3_group.append(p)
        elif 'read_log_alpha' in n:
            optimizer_1_group.append(p)
        elif 'sparsity_lambda_edge' in n:
            optimizer_2_group.append(p)
        elif ('sparsity_lambda_node' in n) and (not disable_node_loss):
            optimizer_4_group.append(p)
    
    optimizer = AdamW(
        [
            {
                'params': optimizer_1_group,
                'lr': edges_lr,
            },
            {
                'params': optimizer_2_group,
                'maximize': True,
                'lr': reg_edges_lr,
            },
            {
                'params': optimizer_3_group,
                'lr': layers_lr,
            },
            {
                'params': optimizer_4_group,
                'maximize': True,
                'lr': reg_layers_lr,
            } 
        ],
        lr=edges_lr
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    return optimizer, scheduler

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    
    # 获取脚本所在目录的上级目录（Edge-Pruning目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    edge_pruning_dir = os.path.dirname(script_dir)  # 从 src/prune 到 src
    edge_pruning_dir = os.path.dirname(edge_pruning_dir)  # 从 src 到 Edge-Pruning
    config_path = os.path.join(edge_pruning_dir, 'config.json')
    
    sys.argv = [os.path.abspath(__file__), config_path]
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_datasets(data_args.dataset_path, data_args.max_train_samples, data_args.max_eval_samples, train_split=data_args.train_split)
    n_train = len(raw_datasets["train"])
    
    # 自动分配模型到多个GPU
    device_map = "auto"  # 或者使用 "balanced" 来平衡分配
    
    # 添加内存优化参数
    model = FMistralForCausalLM.from_pretrained(
        model_args.initialize_from,
        with_embedding_nodes=True,
        disable_linear_regularization_term=data_args.disable_linear_reg_term,
        device_map=device_map,
        torch_dtype=torch.float32,  # 使用float32避免半精度问题
        offload_folder="./offload",  # 设置模型卸载目录
    )
    mistral_model = FMistralForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta",
        with_embedding_nodes=True,
        cache_dir='/data/zodiark/CSAT/.cache',
        device_map=device_map,
        torch_dtype=torch.float32,  # 使用float32避免半精度问题
        offload_folder="./offload",  # 设置模型卸载目录
    )
    
    # 冻结mistral_model的所有参数
    for param in mistral_model.parameters():
        param.requires_grad = False
    
    # 验证mistral_model参数冻结状态
    mistral_trainable_params = sum(p.numel() for p in mistral_model.parameters() if p.requires_grad)
    print(f"mistral_model的所有参数已冻结，可训练参数数量: {mistral_trainable_params}")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    tokenizer.pad_token = tokenizer.eos_token
    
    freeze_all_except_pruning_params(model)
    
    # 打印可训练参数信息
    print("=== 可训练参数信息 ===")
    trainable_params = []
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.numel()))
            total_params += param.numel()
    
    print(f"总可训练参数数量: {total_params:,}")
    print("\n可训练参数详情:")
    for name, numel in trainable_params:
        print(f"  {name}: {numel:,} 参数")
    
    print("\n=== 参数分组信息 ===")
    optimizer_1_group = []
    optimizer_2_group = []
    optimizer_3_group = []
    optimizer_4_group = []
    
    for n, p in model.named_parameters():
        if 'write_log_alpha' in n:
            optimizer_3_group.append(n)
        elif 'read_log_alpha' in n:
            optimizer_1_group.append(n)
        elif 'sparsity_lambda_edges' in n:
            optimizer_2_group.append(n)
        elif ('sparsity_lambda_nodes' in n) and (not data_args.disable_node_loss):
            optimizer_4_group.append(n)
    
    print(f"Optimizer Group 1 (read_log_alpha): {len(optimizer_1_group)} 参数")
    for name in optimizer_1_group:
        print(f"  {name}")
    
    print(f"\nOptimizer Group 2 (sparsity_lambda_edges): {len(optimizer_2_group)} 参数")
    for name in optimizer_2_group:
        print(f"  {name}")
    
    print(f"\nOptimizer Group 3 (write_log_alpha): {len(optimizer_3_group)} 参数")
    for name in optimizer_3_group:
        print(f"  {name}")
    
    print(f"\nOptimizer Group 4 (sparsity_lambda_nodes): {len(optimizer_4_group)} 参数")
    for name in optimizer_4_group:
        print(f"  {name}")
    
    print("=" * 50)
    
    # 打印参数统计信息
    print("\n=== 参数统计信息 ===")
    param_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_type = "unknown"
            if 'log_alpha' in name:
                if 'read_log_alpha' in name:
                    param_type = "read_log_alpha"
                elif 'write_log_alpha' in name:
                    param_type = "write_log_alpha"
            elif 'sparsity_lambda' in name:
                if 'edges' in name:
                    param_type = "sparsity_lambda_edges"
                elif 'nodes' in name:
                    param_type = "sparsity_lambda_nodes"
            
            if param_type not in param_stats:
                param_stats[param_type] = {"count": 0, "total_params": 0}
            param_stats[param_type]["count"] += 1
            param_stats[param_type]["total_params"] += param.numel()
    
    for param_type, stats in param_stats.items():
        print(f"{param_type}: {stats['count']} 个参数组, 总计 {stats['total_params']:,} 参数")
    
    # 对比两个模型的参数状态
    print("\n=== 模型参数状态对比 ===")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mistral_total_params = sum(p.numel() for p in mistral_model.parameters())
    mistral_trainable_params = sum(p.numel() for p in mistral_model.parameters() if p.requires_grad)
    
    print(f"model: 总参数 {model_total_params:,}, 可训练参数 {model_trainable_params:,}")
    print(f"mistral_model: 总参数 {mistral_total_params:,}, 可训练参数 {mistral_trainable_params:,}")
    print(f"可训练参数比例: model={model_trainable_params/model_total_params*100:.2f}%, mistral_model={mistral_trainable_params/mistral_total_params*100:.2f}%")
    
    
    
    # 打印初始内存使用情况
    #print("=== 初始内存使用情况 ===")
    #print_memory_usage()

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

    # Data collator
    collator = DataCollatorIOI(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length
    )
    
    # 优化数据加载设置
    training_args.dataloader_num_workers = 2  # 减少数据加载进程数
    training_args.dataloader_pin_memory = False  # 禁用内存固定
    
    optimizers = get_optimizers(
        model, 
        edges_lr=data_args.edge_learning_rate,
        layers_lr=data_args.layer_learning_rate,
        reg_edges_lr=data_args.reg_edge_learning_rate,
        reg_layers_lr=data_args.reg_layer_learning_rate,
        num_training_steps=training_args.max_steps,
        warmup_steps=training_args.warmup_steps,
        disable_node_loss=data_args.disable_node_loss
    )

    # 禁用wandb报告
    training_args.report_to = []
    
    # Initialize our Trainer
    trainer = FMistralInfoTrainer(
        model=model,
        mistral_model=mistral_model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=eval_fn,
        optimizers=optimizers,
        start_edge_sparsity=data_args.start_edge_sparsity,
        target_edge_sparsity=data_args.target_edge_sparsity,
        start_layer_sparsity=data_args.start_layer_sparsity,
        target_layer_sparsity=data_args.target_layer_sparsity,
        skip_layer_loss_if_higher_sparsity=data_args.stop_optimizing_layer_if_higher_sparsity,
        num_sparsity_warmup_steps=data_args.num_sparsity_warmup_steps,
        warmup_type=data_args.warmup_type,
    )

    # Training
    if training_args.do_train:
        try:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(
                resume_from_checkpoint=checkpoint,
            )
        except Exception as e:
            print(f"Training error: {e}")
            # 暂时禁用自动缓存清理，避免CUDA错误
            # torch.cuda.empty_cache()
            gc.collect()
            raise
        finally:
            # 暂时禁用自动缓存清理，避免CUDA错误
            # torch.cuda.empty_cache()
            gc.collect()
            #print("=== 训练后内存使用情况 ===")
            #print_memory_usage()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    kwargs = {"finetuned_from": "HuggingFaceH4/zephyr-7b-beta"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()