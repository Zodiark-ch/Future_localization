from __future__ import annotations

import math
from collections import defaultdict

from EAP_forComponent.schemas import ComponentScore


class LoraRankAllocator:
    def __init__(
        self,
        min_rank: int,
        max_rank: int,
        rank_multiple: int = 1,
        rank_budget: int | None = None,
        head_to_matrix_aggregation: str = "mean",
        rank_score_source: str = "normalized_abs",
    ):
        if min_rank < 0 or max_rank < min_rank:
            raise ValueError("Expected 0 <= min_rank <= max_rank")
        if rank_multiple < 1:
            raise ValueError("rank_multiple must be >= 1")
        if head_to_matrix_aggregation not in {"mean", "max", "sum"}:
            raise ValueError("head_to_matrix_aggregation must be mean, max, or sum")
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.rank_multiple = rank_multiple
        self.rank_budget = rank_budget
        self.head_to_matrix_aggregation = head_to_matrix_aggregation
        self.rank_score_source = rank_score_source
        self.last_allocation: dict = {}

    def allocate(self, scores: list[ComponentScore]) -> dict[str, int]:
        grouped_scores = self._group_scores(scores)
        ranks = self._allocate_without_budget(grouped_scores)
        if self.rank_budget is not None:
            ranks = self._apply_budget(ranks, grouped_scores)
        rank_pattern = self.to_rank_pattern(ranks)
        self.last_allocation = {
            "min_rank": self.min_rank,
            "max_rank": self.max_rank,
            "rank_multiple": self.rank_multiple,
            "rank_budget": self.rank_budget,
            "rank_score_source": self.rank_score_source,
            "head_to_matrix_aggregation": self.head_to_matrix_aggregation,
            "grouped_scores": grouped_scores,
            "ranks": ranks,
            "rank_pattern": rank_pattern,
        }
        return ranks

    def to_rank_pattern(self, ranks: dict[str, int]) -> dict[str, int]:
        return {module_name: int(rank) for module_name, rank in ranks.items() if int(rank) > 0}

    def _group_scores(self, scores: list[ComponentScore]) -> dict[str, float]:
        values_by_module: dict[str, list[float]] = defaultdict(list)
        for score in scores:
            values_by_module[score.rank_pattern_key].append(_score_value(score, self.rank_score_source))
        grouped: dict[str, float] = {}
        for module_name, values in values_by_module.items():
            if self.head_to_matrix_aggregation == "max":
                grouped[module_name] = max(values)
            elif self.head_to_matrix_aggregation == "sum":
                grouped[module_name] = sum(values)
            else:
                grouped[module_name] = sum(values) / max(1, len(values))
        return grouped

    def _allocate_without_budget(self, grouped_scores: dict[str, float]) -> dict[str, int]:
        if not grouped_scores:
            return {}
        values = list(grouped_scores.values())
        min_score = min(values)
        max_score = max(values)
        ranks: dict[str, int] = {}
        for module_name, value in grouped_scores.items():
            if math.isclose(max_score, min_score):
                rank = self.max_rank if value > 0 else self.min_rank
            else:
                normalized = (value - min_score) / (max_score - min_score)
                rank = self.min_rank + normalized * (self.max_rank - self.min_rank)
            ranks[module_name] = self._round_and_clip(rank)
        return ranks

    def _apply_budget(self, ranks: dict[str, int], grouped_scores: dict[str, float]) -> dict[str, int]:
        assert self.rank_budget is not None
        min_total = self.min_rank * len(ranks)
        if self.rank_budget < min_total:
            raise ValueError(
                f"rank_budget={self.rank_budget} is smaller than min total rank {min_total}"
            )
        adjusted = dict(ranks)
        low_to_high = sorted(grouped_scores, key=lambda module_name: grouped_scores[module_name])
        high_to_low = list(reversed(low_to_high))
        while sum(adjusted.values()) > self.rank_budget:
            changed = False
            for module_name in low_to_high:
                if adjusted[module_name] - self.rank_multiple >= self.min_rank:
                    adjusted[module_name] -= self.rank_multiple
                    changed = True
                    break
            if not changed:
                break
        while sum(adjusted.values()) + self.rank_multiple <= self.rank_budget:
            changed = False
            for module_name in high_to_low:
                if adjusted[module_name] + self.rank_multiple <= self.max_rank:
                    adjusted[module_name] += self.rank_multiple
                    changed = True
                    break
            if not changed:
                break
        return adjusted

    def _round_and_clip(self, rank: float) -> int:
        if self.rank_multiple == 1:
            rounded = int(round(rank))
        else:
            rounded = int(round(rank / self.rank_multiple) * self.rank_multiple)
        return max(self.min_rank, min(self.max_rank, rounded))


def _score_value(score: ComponentScore, source: str) -> float:
    if source in {"normalized_abs", "rank_score"}:
        return float(score.rank_score)
    if source in {"raw_abs", "sum_abs"}:
        return float(score.abs_score)
    if source == "mean_abs":
        return abs(float(score.mean_score))
    if source == "sqrt_numel_abs":
        return abs(float(score.sqrt_numel_score))
    raise ValueError(f"Unsupported rank_score_source: {source}")
