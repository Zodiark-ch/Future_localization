import torch


def sequential_position_ids(input_ids):
    return torch.arange(
        input_ids.size(1), device=input_ids.device, dtype=torch.long
    ).unsqueeze(0)


def patch_mistral_rotary_embedding():
    try:
        from transformers.models.mistral import modeling_mistral
    except Exception:
        return False

    if getattr(modeling_mistral, "_csat_safe_rope_patch", False):
        return False

    rotate_half = modeling_mistral.rotate_half

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        ids = position_ids.to(device=cos.device, dtype=torch.long)
        seq_len = q.size(2)
        if ids.dim() == 2 and ids.size(1) == seq_len and cos.size(0) == seq_len:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        elif _is_local_sequential(ids, seq_len):
            cos = cos[: q.size(2)].unsqueeze(0)
            sin = sin[: q.size(2)].unsqueeze(0)
        else:
            flat_ids = ids.reshape(-1)
            _validate_rope_ids(flat_ids, cos.size(0))
            cos = cos.index_select(0, flat_ids).reshape(*ids.shape, cos.shape[-1])
            sin = sin.index_select(0, flat_ids).reshape(*ids.shape, sin.shape[-1])
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    modeling_mistral.apply_rotary_pos_emb = apply_rotary_pos_emb
    modeling_mistral._csat_safe_rope_patch = True
    return True


def _is_local_sequential(ids, seq_len):
    if ids.dim() != 2 or ids.size(0) != 1 or ids.size(1) != seq_len or seq_len == 0:
        return False
    return (
        int(ids[0, 0].detach().cpu()) == 0
        and int(ids[0, -1].detach().cpu()) == seq_len - 1
    )


def _validate_rope_ids(ids, cache_len):
    min_id = int(ids.min().detach().cpu())
    max_id = int(ids.max().detach().cpu())
    if min_id < 0 or max_id >= cache_len:
        raise RuntimeError(
            f"RoPE position ids out of range: min={min_id}, max={max_id}, cache_len={cache_len}."
        )