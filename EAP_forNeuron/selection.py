from __future__ import annotations

import math

import torch

from EAP_forNeuron.schemas import ScoreShard


class NeuronMaskSelector:
    def __init__(
        self,
        output_ratio: float,
        ratio_base: str = "all",
        score_abs: bool = True,
        max_concat_candidates: int = 50_000_000,
    ):
        if output_ratio < 0 or output_ratio > 1:
            raise ValueError("output_ratio must be in [0, 1]")
        if ratio_base not in {"all", "candidate"}:
            raise ValueError("ratio_base must be 'all' or 'candidate'")
        self.output_ratio = output_ratio
        self.ratio_base = ratio_base
        self.score_abs = score_abs
        self.max_concat_candidates = max_concat_candidates

    def select(
        self,
        old_mask: dict[str, torch.Tensor],
        scores: dict[str, ScoreShard],
        total_neuron_count: int,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        output_mask = {name: torch.zeros_like(mask, dtype=torch.bool) for name, mask in old_mask.items()}
        candidate_count = sum(shard.candidate_count for shard in scores.values())
        if self.ratio_base == "all":
            target_keep_count = int(total_neuron_count * self.output_ratio)
        else:
            target_keep_count = int(candidate_count * self.output_ratio)
        actual_target = min(candidate_count, target_keep_count)
        summary = {
            "total_neuron_count": int(total_neuron_count),
            "candidate_count": int(candidate_count),
            "target_keep_count": int(target_keep_count),
            "actual_keep_count": 0,
            "output_ratio": float(self.output_ratio),
            "ratio_base": self.ratio_base,
            "score_abs": bool(self.score_abs),
            "threshold": None,
            "per_parameter": {},
        }
        for name, mask in old_mask.items():
            shard = scores.get(name)
            summary["per_parameter"][name] = {
                "candidates": int(shard.candidate_count) if shard is not None else 0,
                "kept": 0,
                "skipped": int(mask.sum().item()) if shard is None else 0,
            }
        if actual_target <= 0 or candidate_count == 0:
            return output_mask, summary
        if actual_target >= candidate_count:
            for name, shard in scores.items():
                flat = output_mask[name].flatten()
                flat[shard.flat_indices] = True
                summary["per_parameter"][name]["kept"] = int(shard.candidate_count)
            summary["actual_keep_count"] = int(candidate_count)
            summary["threshold"] = None
            return output_mask, summary
        threshold = self._threshold(scores, actual_target)
        summary["threshold"] = float(threshold)
        ties: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        kept_count = 0
        for name, shard in scores.items():
            rank_scores = shard.scores.float().abs() if self.score_abs else shard.scores.float()
            finite = torch.isfinite(rank_scores)
            above = finite & (rank_scores > threshold)
            equal = finite & (rank_scores == threshold)
            flat = output_mask[name].flatten()
            if above.any():
                flat[shard.flat_indices[above]] = True
                kept = int(above.sum().item())
                kept_count += kept
                summary["per_parameter"][name]["kept"] += kept
            if equal.any():
                ties.append((name, shard.flat_indices[equal], rank_scores[equal]))
        remaining = actual_target - kept_count
        if remaining > 0 and ties:
            tie_indices = _stable_tie_indices(ties, remaining)
            for name, flat_indices in tie_indices:
                flat = output_mask[name].flatten()
                flat[flat_indices] = True
                kept = int(flat_indices.numel())
                kept_count += kept
                summary["per_parameter"][name]["kept"] += kept
        summary["actual_keep_count"] = int(kept_count)
        return output_mask, summary

    def _threshold(self, scores: dict[str, ScoreShard], keep_count: int) -> float:
        finite_count = sum(
            int(_rank_scores(shard, self.score_abs).isfinite().sum().item())
            for shard in scores.values()
        )
        if finite_count == 0:
            return math.inf
        if finite_count <= self.max_concat_candidates:
            all_scores = torch.cat(
                [_rank_scores(shard, self.score_abs).cpu() for shard in scores.values()]
            )
            all_scores = all_scores[torch.isfinite(all_scores)]
            if keep_count >= all_scores.numel():
                return float(all_scores.min().item())
            kth_smallest = all_scores.numel() - keep_count + 1
            return float(torch.kthvalue(all_scores, kth_smallest).values.item())
        if keep_count <= self.max_concat_candidates:
            top_values = None
            for shard in scores.values():
                shard_scores = _rank_scores(shard, self.score_abs).cpu()
                shard_scores = shard_scores[torch.isfinite(shard_scores)]
                if shard_scores.numel() == 0:
                    continue
                local_k = min(keep_count, int(shard_scores.numel()))
                shard_top = torch.topk(shard_scores, k=local_k, largest=True).values
                top_values = shard_top if top_values is None else torch.cat((top_values, shard_top))
                if top_values.numel() > keep_count:
                    top_values = torch.topk(top_values, k=keep_count, largest=True).values
            if top_values is None or top_values.numel() == 0:
                return math.inf
            return float(top_values.min().item())
        return self._binary_search_threshold(scores, keep_count)

    def _binary_search_threshold(self, scores: dict[str, ScoreShard], keep_count: int) -> float:
        minima = []
        maxima = []
        for shard in scores.values():
            shard_scores = _rank_scores(shard, self.score_abs)
            shard_scores = shard_scores[torch.isfinite(shard_scores)]
            if shard_scores.numel() == 0:
                continue
            minima.append(float(shard_scores.min().item()))
            maxima.append(float(shard_scores.max().item()))
        if not minima:
            return math.inf
        low = min(minima)
        high = max(maxima)
        for _ in range(40):
            mid = (low + high) / 2.0
            count = 0
            for shard in scores.values():
                shard_scores = _rank_scores(shard, self.score_abs)
                count += int((torch.isfinite(shard_scores) & (shard_scores > mid)).sum().item())
            if count >= keep_count:
                low = mid
            else:
                high = mid
        return high


def _stable_tie_indices(
    ties: list[tuple[str, torch.Tensor, torch.Tensor]], remaining: int
) -> list[tuple[str, torch.Tensor]]:
    selected: list[tuple[str, torch.Tensor]] = []
    left = remaining
    for name, flat_indices, _scores in ties:
        if left <= 0:
            break
        take = min(left, int(flat_indices.numel()))
        if take > 0:
            selected.append((name, flat_indices[:take]))
            left -= take
    return selected


def _rank_scores(shard: ScoreShard, score_abs: bool) -> torch.Tensor:
    rank_scores = shard.scores.float().abs() if score_abs else shard.scores.float()
    return rank_scores
