from functools import partial

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute 
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        type=str,
        default="/data/zodiark/CSAT/files/logs/2026-03-27-20-30-44-500133/checkpoints/checkpoint-20260327-204547-epoch-6",
    )
    p.add_argument("--save_name", type=str, default="gender_graph_epoch6_or.pt")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
local_model_dir = args.model_dir
def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

def collate_EAP_OR(xs):
    clean, corrupted, labels = zip(*xs)
    # 与正向相反：输入对调 clean / corrupted
    corrupted_or = list(clean)
    clean_or = list(corrupted)
    # 正向标签为 [correct_idx, incorrect_idx]；反向寻路时每条样本两列互换
    labels = torch.tensor(labels, dtype=torch.long)
    labels = labels[:, [1, 0]].contiguous()
    return clean_or, corrupted_or, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
    
    def to_dataloader(self, batch_size: int, reverse: bool = False):
        collate = collate_EAP_OR if reverse else collate_EAP
        return DataLoader(self, batch_size=batch_size, collate_fn=collate)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results


#model_name = 'meta-llama/Meta-Llama-3-8B'
# device_map = "auto"  # 或者使用 "balanced" 来平衡分配
model_name = 'mistralai/Mistral-7B-v0.1'

hf_model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

model = HookedTransformer.from_pretrained(
    model_name,
    hf_model=hf_model,
    tokenizer=tokenizer,
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda',
    dtype=torch.float16,
)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True
    
ds = EAPDataset('induction.csv')
dataloader = ds.to_dataloader(2, reverse=True)

# Instantiate a graph with a model
g = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method='EAP-IG-inputs', ig_steps=5)

g.apply_topn(20000, True)
g.to_pt(args.save_name)

baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
results = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results}")

# 可视化 circuit：将 g 导出为 PNG（使用 eap/visualization 的配色）
# viz_path = 'ioi_graph.png'
# g.to_image(viz_path, colorscheme='Pastel2', minimum_penwidth=0.6, maximum_penwidth=5.0, layout='dot', seed=42)
# print(f"Circuit visualization saved to {viz_path}")