from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from EAP_forComponent.schemas import ComponentScore


def save_outputs(
    output_dir: str | Path,
    scores: list[ComponentScore],
    rank_pattern: dict[str, int],
    lora_allocation: dict,
    component_mask: dict[str, torch.Tensor],
    summary: dict,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dicts = [component_score_to_dict(score) for score in sorted(scores, key=_score_sort_key)]
    component_scores_json = output_dir / "component_scores.json"
    component_scores_json.write_text(
        json.dumps(score_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    component_scores_pt = output_dir / "component_scores.pt"
    torch.save(score_dicts, component_scores_pt)
    rank_pattern_path = output_dir / "rank_pattern.json"
    rank_pattern_path.write_text(
        json.dumps(rank_pattern, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lora_allocation_path = output_dir / "lora_allocation.json"
    lora_allocation_path.write_text(
        json.dumps(_json_safe(lora_allocation), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    component_mask_path = output_dir / "component_mask.pt"
    torch.save(component_mask, component_mask_path)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "component_scores_json": component_scores_json,
        "component_scores_pt": component_scores_pt,
        "rank_pattern": rank_pattern_path,
        "lora_allocation": lora_allocation_path,
        "component_mask": component_mask_path,
        "summary": summary_path,
    }


def component_score_to_dict(score: ComponentScore) -> dict:
    data = asdict(score)
    data["component_name"] = score.component_name
    data["rank_pattern_key"] = score.rank_pattern_key
    return data


def _score_sort_key(score: ComponentScore):
    if score.mean_raw_rank is not None:
        return (float(score.mean_raw_rank), score.component_name)
    return (-float(score.rank_score), score.component_name)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
