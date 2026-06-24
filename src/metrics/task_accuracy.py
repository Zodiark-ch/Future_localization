import json
import os

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from modeling_patches import patch_mistral_rotary_embedding, sequential_position_ids


def _label_field(item):
    if "labels" in item:
        return "labels"
    return "label"


def _collate_fn(batch):
    label_key = _label_field(batch[0])
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item[label_key] for item in batch]),
    }


def _decode_ids(tokenizer, token_ids):
    if not token_ids:
        return "<EMPTY>"
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _print_dataloader_sample(dataloader, tokenizer, tag):
    if tokenizer is None:
        return
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print(f"[Dataset Debug] tag={tag}: empty dataloader")
        return

    input_ids = batch["input_ids"][0].detach().cpu()
    attention_mask = batch["attention_mask"][0].detach().cpu()
    labels = batch["labels"][0].detach().cpu()
    labels = _sanitize_labels(labels.unsqueeze(0), attention_mask.unsqueeze(0), len(tokenizer))[0]

    active_input_ids = input_ids[attention_mask.bool()].tolist()
    raw_label_positions = labels.ne(-100).nonzero(as_tuple=True)[0].tolist()
    valid_label_positions = []
    valid_label_ids = []
    if len(raw_label_positions) == 1:
        position = raw_label_positions[0]
        label_id = int(labels[position].item())
        if 0 <= label_id < len(tokenizer):
            valid_label_positions.append(position)
            valid_label_ids.append(label_id)
    else:
        ignored_token_ids = set(tokenizer.all_special_ids)
        if tokenizer.pad_token_id is not None:
            ignored_token_ids.add(tokenizer.pad_token_id)
        for position, label_id in enumerate(labels.tolist()):
            if label_id == -100 or not 0 <= label_id < len(tokenizer):
                continue
            if label_id in ignored_token_ids:
                continue
            if tokenizer.decode([label_id], skip_special_tokens=False).strip() == "":
                continue
            valid_label_positions.append(position)
            valid_label_ids.append(label_id)

    print(f"[Dataset Debug] tag={tag}")
    print(f"[Dataset Debug] input_text: {_decode_ids(tokenizer, active_input_ids)}")
    print(f"[Dataset Debug] attention_mask_count: {int(attention_mask.sum().item())}")
    print(f"[Dataset Debug] label_positions: {valid_label_positions}")
    print(f"[Dataset Debug] label_ids: {valid_label_ids}")
    print(f"[Dataset Debug] labels_text: {_decode_ids(tokenizer, valid_label_ids)}")


def _model_vocab_size(model):
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        return output_embeddings.weight.shape[0]
    return getattr(model.config, "vocab_size", None)


def _sanitize_labels(labels, attention_mask, vocab_size):
    labels = labels.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    labels = labels.masked_fill((labels < 0) & (labels != -100), -100)
    if vocab_size is not None:
        labels = labels.masked_fill((labels != -100) & (labels >= vocab_size), -100)
    return labels


def _trim_to_active_length(input_ids, attention_mask, labels=None):
    max_active_len = int(attention_mask.long().sum(dim=1).max().detach().cpu().item())
    max_active_len = max(1, max_active_len)
    if max_active_len >= input_ids.size(1):
        return input_ids, attention_mask, labels
    input_ids = input_ids[:, :max_active_len].contiguous()
    attention_mask = attention_mask[:, :max_active_len].contiguous()
    if labels is not None:
        labels = labels[:, :max_active_len].contiguous()
    return input_ids, attention_mask, labels


def _causal_lm_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if not shift_labels.ne(-100).any():
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _score_batch(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    input_ids, attention_mask, labels = _trim_to_active_length(input_ids, attention_mask, labels)
    labels = _sanitize_labels(labels, attention_mask, _model_vocab_size(model))
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=sequential_position_ids(input_ids),
        use_cache=False,
    )
    selected_logits, selected_labels = _select_prediction_logits_and_labels(
        outputs.logits, labels, attention_mask, input_ids
    )
    if selected_logits is None:
        return 0, 0, 0.0
    loss = F.cross_entropy(selected_logits, selected_labels)
    predictions = selected_logits.argmax(dim=-1)
    correct = (predictions == selected_labels).sum().detach().item()
    total = selected_labels.numel()
    return correct, total, loss.detach().float().item()


def _select_prediction_logits_and_labels(logits, labels, attention_mask, input_ids):
    valid_mask = (labels != -100) & (attention_mask == 1)
    if not valid_mask.any():
        return None, None

    valid_counts = valid_mask.sum(dim=1)
    active_lengths = attention_mask.long().sum(dim=1)
    positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    last_label_positions = (valid_mask.long() * positions).max(dim=1).values
    next_token_rows = (
        (valid_counts == 1)
        & (active_lengths > 0)
        & (last_label_positions == active_lengths - 1)
    )
    selected_logits = []
    selected_labels = []
    if next_token_rows.any():
        rows = next_token_rows.nonzero(as_tuple=True)[0]
        selected_logits.append(logits[rows, last_label_positions[rows], :].float())
        selected_labels.append(labels[rows, last_label_positions[rows]].long())

    shifted_rows = ~next_token_rows
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().long()
    shift_valid_mask = shift_labels.ne(-100) & shifted_rows.unsqueeze(1)
    if shift_valid_mask.any():
        selected_logits.append(shift_logits[shift_valid_mask].float())
        selected_labels.append(shift_labels[shift_valid_mask])

    if not selected_logits:
        return None, None
    return torch.cat(selected_logits, dim=0), torch.cat(selected_labels, dim=0)


def eval_task_accuracy(model_name, task_dataset, output_dir=".", batch_size=8, device=None):
    patch_mistral_rotary_embedding()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        cache_dir="./.cache",
        low_cpu_mem_usage=True,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(
        task_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
    )
    _print_dataloader_sample(dataloader, tokenizer, "eval_task_accuracy")

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    losses = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating task accuracy"):
            correct, total, loss = _score_batch(model, batch, device)
            correct_predictions += correct
            total_predictions += total
            losses.append(loss)

    accuracy = (
        correct_predictions / total_predictions * 100 if total_predictions > 0 else 0.0
    )
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    result = {
        "accuracy": accuracy,
        "loss": mean_loss,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "model_name": model_name,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "accuracy.json"), "w") as file:
        json.dump(result, file, indent=4)
    return result


def eval_task_accuracy_in_memory(model, task_dataset, batch_size=8, tokenizer=None, debug_sample=True):
    patch_mistral_rotary_embedding()
    dataloader = DataLoader(
        task_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
    )
    if debug_sample:
        _print_dataloader_sample(dataloader, tokenizer, "eval_task_accuracy_in_memory")
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct_predictions = 0
    total_predictions = 0
    losses = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating task accuracy"):
            correct, total, loss = _score_batch(model, batch, device)
            correct_predictions += correct
            total_predictions += total
            losses.append(loss)

    if was_training:
        model.train()

    return {
        "accuracy": correct_predictions / total_predictions * 100 if total_predictions > 0 else 0.0,
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
    }
