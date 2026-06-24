import torch
import torch.nn.functional as F

from modeling_patches import sequential_position_ids


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def task_loss(model, data):
    if data is None:
        return None, None
    forward_model = unwrap_model(model)
    input_ids, attention_mask, raw_labels = trim_to_active_length(data[0], data[1], data[2])
    labels = sanitize_labels(raw_labels, attention_mask, model_vocab_size(forward_model))
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": sequential_position_ids(input_ids),
        "use_cache": False,
    }
    if not labels.ne(-100).any():
        outputs = forward_model(**inputs)
        return outputs.logits.sum() * 0.0, outputs
    outputs = forward_model(**inputs)
    return causal_lm_loss(outputs.logits, labels, attention_mask, input_ids), outputs


def trim_to_active_length(input_ids, attention_mask, labels=None):
    max_active_len = int(attention_mask.long().sum(dim=1).max().detach().cpu().item())
    max_active_len = max(1, max_active_len)
    if max_active_len >= input_ids.size(1):
        return input_ids, attention_mask, labels
    input_ids = input_ids[:, :max_active_len].contiguous()
    attention_mask = attention_mask[:, :max_active_len].contiguous()
    if labels is not None:
        labels = labels[:, :max_active_len].contiguous()
    return input_ids, attention_mask, labels


def model_vocab_size(model):
    unwrapped_model = unwrap_model(model)
    output_embeddings = unwrapped_model.get_output_embeddings()
    if output_embeddings is not None:
        return output_embeddings.weight.shape[0]
    return getattr(unwrapped_model.config, "vocab_size", None)


def sanitize_labels(labels, attention_mask, vocab_size):
    labels = labels.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    labels = labels.masked_fill((labels < 0) & (labels != -100), -100)
    if vocab_size is not None:
        labels = labels.masked_fill((labels != -100) & (labels >= vocab_size), -100)
    return labels


def causal_lm_loss(logits, labels, attention_mask=None, input_ids=None):
    labels = labels.long()
    vocab_size = logits.size(-1)
    labels = labels.masked_fill((labels < 0) & (labels != -100), -100)
    labels = labels.masked_fill((labels != -100) & (labels >= vocab_size), -100)

    valid_mask = labels.ne(-100)
    if attention_mask is not None:
        valid_mask = valid_mask & attention_mask.eq(1)
        active_lengths = attention_mask.long().sum(dim=1)
    else:
        active_lengths = torch.full(
            (labels.size(0),), labels.size(1), device=labels.device, dtype=torch.long
        )

    positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    label_positions = (valid_mask.long() * positions).max(dim=1).values
    valid_counts = valid_mask.sum(dim=1)
    next_token_rows = (
        (valid_counts == 1)
        & (active_lengths > 0)
        & (label_positions == active_lengths - 1)
    )
    loss_sum = logits.sum() * 0.0
    token_count = 0
    if next_token_rows.any():
        rows = next_token_rows.nonzero(as_tuple=True)[0]
        selected_logits = logits[rows, label_positions[rows], :].float()
        selected_labels = labels[rows, label_positions[rows]]
        loss_sum = loss_sum + F.cross_entropy(
            selected_logits, selected_labels, reduction="sum"
        )
        token_count += int(selected_labels.numel())

    shifted_rows = ~next_token_rows
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_valid_mask = shift_labels.ne(-100) & shifted_rows.unsqueeze(1)
    if shift_valid_mask.any():
        selected_logits = shift_logits[shift_valid_mask].float()
        selected_labels = shift_labels[shift_valid_mask]
        loss_sum = loss_sum + F.cross_entropy(
            selected_logits, selected_labels, reduction="sum"
        )
        token_count += int(selected_labels.numel())

    if token_count == 0:
        return logits.sum() * 0.0
    return loss_sum / token_count


def kl_divergence(current_logits, reference_logits, labels):
    valid_mask = labels.ne(-100)
    current_log_probs = torch.log_softmax(current_logits, dim=-1)
    reference_probs = torch.softmax(reference_logits, dim=-1)
    token_kl = torch.nn.functional.kl_div(
        current_log_probs,
        reference_probs,
        reduction="none",
    ).sum(dim=-1)
    token_kl = token_kl.masked_fill(~valid_mask, 0.0)
    denom = valid_mask.sum().clamp_min(1)
    return token_kl.sum() / denom
