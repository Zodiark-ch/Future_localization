from __future__ import annotations

import random
import json
import time
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from statistics import mean

import re

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from EAP_forComponent.components import ComponentRegistry
from EAP_forComponent.future_localization import FutureLocalizationScorer, _load_future_state_dict
from EAP_forComponent.hooks import ComponentActivationCache
from EAP_forComponent.schemas import ComponentScore, EAPComponentConfig
from EAP_forComponent.scorer import ComponentAttributionScorer, normalize_localization_mode
from EAP_forComponent.schemas import PairBatch
from EAP_forLogicalCircuit.circuit_builder import select_circuit_edges
from EAP_forLogicalCircuit.component_projection import build_module_metadata, project_to_component_scores
from EAP_forLogicalCircuit.current_localization import CurrentEdgeAttributionScorer
from EAP_forLogicalCircuit.data import load_pair_dataset
from EAP_forLogicalCircuit.edge_hooks import DestinationInputCache
from EAP_forLogicalCircuit.future_localization import FutureEdgeLocalizationScorer
from EAP_forLogicalCircuit.graph_export import save_circuit_graph_export
from EAP_forLogicalCircuit.graph_registry import EdgeTarget, GraphRegistry, build_graph_metadata_from_model
from EAP_forLogicalCircuit.logical_fusion import fuse_logical_edges
from EAP_forLogicalCircuit.mask_builder import ComponentMaskBuilder
from EAP_forLogicalCircuit.model_loader import ensure_src_on_path, load_model_and_tokenizer
from EAP_forLogicalCircuit.node_circuit_builder import build_node_induced_circuit, combine_selected_component_scores
from EAP_forLogicalCircuit.outputs import save_outputs
from EAP_forLogicalCircuit.rank_allocator import LoraRankAllocator
from EAP_forLogicalCircuit.schemas import EAPLogicalCircuitConfig, EdgeScore


class EAPForLogicalCircuitRunner:
    def __init__(self, config: EAPLogicalCircuitConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.registry: GraphRegistry | None = None
        self.graph_metadata: dict = {}
        self.component_registry: ComponentRegistry | None = None
        self._future_localization_summary: dict = {}
        self._future_component_state_dict: dict[str, torch.Tensor] | None = None

    def run(self) -> dict[str, Path]:
        if not self.config.model_name_or_path:
            raise ValueError("model_name_or_path is required")
        if not self.config.output_dir:
            raise ValueError("output_dir is required")
        started = time.time()
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(
            model_name_or_path=self.config.model_name_or_path,
            tokenizer_name_or_path=self.config.tokenizer_name_or_path,
            cache_dir=self.config.cache_dir,
            device=self.config.device,
            use_bfloat16=self.config.use_bfloat16,
            use_cpu=self.config.use_cpu,
        )
        dataset, collator = load_pair_dataset(
            dataset_name=self.config.dataset_name,
            tokenizer=self.tokenizer,
            data_path=self.config.data_path,
            corruption_column=self.config.corruption_column,
            max_samples=self.config.max_samples,
            max_length=self.config.max_length,
            input_format=self.config.input_format,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        circuit_or_dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda examples: _reverse_pair_batch(collator(examples)),
        )
        circuit_construction = str(self.config.circuit_construction).lower()
        self.graph_metadata = build_graph_metadata_from_model(
            model=self.model,
            target_modules=self.config.target_modules,
        )
        component_granularity = "head" if circuit_construction == "node_induced" else self.config.component_granularity
        self.component_registry = ComponentRegistry.from_model(
            model=self.model,
            target_modules=self.config.target_modules,
            attention_granularity=component_granularity,
        )
        edge_scores: list[EdgeScore] | None = None
        graph_edge_scores: list[EdgeScore] | None = None
        node_scores: list[ComponentScore] = []
        circuit_or_node_scores: list[ComponentScore] = []
        circuit_construction_summary: dict = {"construction": circuit_construction}
        if circuit_construction == "node_induced":
            node_scores = self.run_node_attribution(dataloader)
            circuit_or_node_scores = self.run_node_attribution(circuit_or_dataloader)
            circuit = build_node_induced_circuit(
                component_scores=node_scores,
                graph_metadata=self.graph_metadata,
                node_topn=self.config.node_topn,
                circuit_name="circuit",
            )
            circuit_or = build_node_induced_circuit(
                component_scores=circuit_or_node_scores,
                graph_metadata=self.graph_metadata,
                node_topn=self.config.node_topn,
                circuit_name="circuit_or",
            )
            circuit_edges = circuit.edges
            circuit_or_edges = circuit_or.edges
            logical_component_scores = combine_selected_component_scores(
                circuit_scores=circuit.component_scores,
                circuit_or_scores=circuit_or.component_scores,
                score_normalization=self.config.score_normalization,
                localization_mode=str(self.config.localization_mode),
            )
            circuit_construction_summary = {
                "construction": "node_attribution_topn_induced_dense_edges",
                "node_topn": int(self.config.node_topn),
                "circuit": circuit.summary,
                "circuit_or": circuit_or.summary,
            }
        elif circuit_construction == "edge_attribution":
            self.registry = GraphRegistry.from_model(
                model=self.model,
                target_modules=self.config.target_modules,
            )
            self.graph_metadata = self.registry.metadata
            edge_scores = self.run_attribution(dataloader)
            graph_edge_scores = edge_scores
            circuit_or_edge_scores = self.run_attribution(circuit_or_dataloader)
            circuit_edges = select_circuit_edges(
                edge_scores=edge_scores,
                edge_topn=self.config.edge_topn,
                edge_threshold=self.config.edge_threshold,
                edge_score_abs=self.config.edge_score_abs,
                circuit_name="circuit",
            )
            circuit_or_edges = select_circuit_edges(
                edge_scores=circuit_or_edge_scores,
                edge_topn=self.config.edge_topn,
                edge_threshold=self.config.edge_threshold,
                edge_score_abs=self.config.edge_score_abs,
                circuit_name="circuit_or",
            )
            edge_targets = self.registry.edge_targets()
            module_metadata = build_module_metadata(edge_targets)
            logical_component_scores = project_to_component_scores(
                circuit_edges=circuit_edges,
                circuit_or_edges=circuit_or_edges,
                module_metadata=module_metadata,
                score_token_mode=self.config.score_token_mode,
                score_normalization=self.config.score_normalization,
                localization_mode=str(self.config.localization_mode),
            )
        else:
            raise ValueError(f"Unsupported circuit_construction: {self.config.circuit_construction}")
        logical_edges = fuse_logical_edges(circuit_edges=circuit_edges, circuit_or_edges=circuit_or_edges)
        logical_gate_counts: dict[str, int] = {}
        for edge in logical_edges:
            gate = str(getattr(edge, "logical_gate", "UNKNOWN"))
            logical_gate_counts[gate] = logical_gate_counts.get(gate, 0) + 1
        allocator = LoraRankAllocator(
            min_rank=self.config.min_rank,
            max_rank=self.config.max_rank,
            rank_multiple=self.config.rank_multiple,
            rank_budget=self.config.rank_budget,
            head_to_matrix_aggregation=self.config.head_to_matrix_aggregation,
            rank_score_source=self.config.rank_score_source,
        )
        ranks = allocator.allocate(logical_component_scores)
        rank_pattern = allocator.to_rank_pattern(ranks)
        mask_builder = ComponentMaskBuilder(
            mask_fill_strategy=self.config.mask_fill_strategy,
            seed=self.config.mask_seed,
            include_all_parameters=self.config.mask_all_parameters,
            min_keep_ratio=self.config.mask_min_keep_ratio,
            max_keep_ratio=self.config.mask_max_keep_ratio,
        )
        component_mask = mask_builder.build(self.model, logical_component_scores)
        elapsed = time.time() - started
        summary = {
            "model_name_or_path": self.config.model_name_or_path,
            "dataset_name": self.config.dataset_name,
            "data_path": str(self.config.data_path) if self.config.data_path else None,
            "metric": self.config.metric,
            "localization_mode": self.config.localization_mode,
            "future_model_name_or_path": self.config.future_model_name_or_path,
            "future_step_k": self.config.future_step_k,
            "future_step_k_min": self.config.future_step_k_min,
            "future_step_k_max": self.config.future_step_k_max,
            "future_step_k_samples": self.config.future_step_k_samples,
            "future_step_k_seed": self.config.future_step_k_seed,
            "future_hvp_strategy": self.config.future_hvp_strategy,
            "future_delta_parameter_filter": self.config.future_delta_parameter_filter,
            "score_token_mode": self.config.score_token_mode,
            "score_normalization": self.config.score_normalization,
            "circuit_construction": circuit_construction,
            "node_topn": self.config.node_topn,
            "circuit_or_construction": "reversed_clean_corrupted_and_swapped_labels",
            "batch_size": self.config.batch_size,
            "max_samples": self.config.max_samples,
            "edge_attribution_computed": edge_scores is not None,
            "graph_export_enabled": bool(self.config.graph),
            "graph_candidate_edge_count": int(self.graph_metadata.get("edge_count", 0)),
            "node_score_count": len(node_scores),
            "circuit_or_node_score_count": len(circuit_or_node_scores),
            "circuit_edge_count": len(circuit_edges),
            "circuit_or_edge_count": len(circuit_or_edges),
            "logical_edge_count": len(logical_edges),
            "logical_gate_counts": logical_gate_counts,
            "logical_component_count": len(logical_component_scores),
            "rank_score_source": self.config.rank_score_source,
            "lora_allocation": allocator.last_allocation,
            "mask_builder": mask_builder.last_summary,
            "graph_registry": self.graph_metadata,
            "component_registry": self.component_registry.metadata if self.component_registry else {},
            "circuit_construction_summary": circuit_construction_summary,
            "elapsed_seconds": elapsed,
        }
        if edge_scores is not None:
            summary.update(
                {
                    "edge_selection_mode": "threshold" if self.config.edge_threshold is not None else "topn",
                    "edge_topn": self.config.edge_topn,
                    "edge_threshold": self.config.edge_threshold,
                    "edge_score_abs": self.config.edge_score_abs,
                    "circuit_or_edge_score_abs": self.config.edge_score_abs,
                    "edge_attribution_count": len(edge_scores),
                }
            )
        if self._future_localization_summary:
            summary["future_localization"] = self._future_localization_summary
        summary.update(self.config.metadata)
        if self.config.graph:
            if not node_scores:
                node_scores = self.run_node_attribution(dataloader)
                summary["node_score_count"] = len(node_scores)
            if graph_edge_scores is None:
                self.registry = GraphRegistry.from_model(
                    model=self.model,
                    target_modules=self.config.target_modules,
                )
                self.graph_metadata = self.registry.metadata
                graph_edge_targets, graph_edge_target_summary = _graph_candidate_edge_targets(
                    registry=self.registry,
                    component_scores=node_scores,
                    graph_metadata=self.graph_metadata,
                    node_topn=self.config.graph_node_topn,
                )
                graph_edge_scores = self.run_attribution(
                    dataloader,
                    edge_targets=graph_edge_targets,
                    progress_label="EAP_forLogicalCircuit graph edge attribution",
                )
                summary["graph_registry"] = self.graph_metadata
                summary["graph_edge_target_filter"] = graph_edge_target_summary
            else:
                summary["graph_edge_target_filter"] = {
                    "scope": "reused_existing_edge_scores",
                    "edge_score_count": len(graph_edge_scores),
                }
            graph_paths, graph_summary = save_circuit_graph_export(
                output_dir=self.config.output_dir,
                component_scores=node_scores,
                edge_scores=graph_edge_scores,
                graph_metadata=self.graph_metadata,
                node_topn=self.config.graph_node_topn,
                edge_threshold_ratio=self.config.graph_edge_threshold_ratio,
                edge_budget_multiplier=self.config.graph_edge_budget_multiplier,
                input_edge_limit_ratio=self.config.graph_input_edge_limit_ratio,
            )
            summary["graph_export"] = graph_summary
            summary["graph_edge_attribution_count"] = len(graph_edge_scores)
            summary["elapsed_seconds"] = time.time() - started
        paths = save_outputs(
            output_dir=self.config.output_dir,
            edge_scores=edge_scores,
            circuit_edges=circuit_edges,
            circuit_or_edges=circuit_or_edges,
            logical_edges=logical_edges,
            logical_component_scores=logical_component_scores,
            rank_pattern=rank_pattern,
            lora_allocation=allocator.last_allocation,
            component_mask=component_mask,
            summary=summary,
        )
        if self.config.graph:
            paths.update(graph_paths)
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return paths

    def run_attribution(
        self,
        dataloader: DataLoader,
        edge_targets: list[EdgeTarget] | None = None,
        progress_label: str | None = None,
    ):
        localization_mode = str(self.config.localization_mode).lower()
        if localization_mode == "future":
            return self._run_future_attribution(dataloader, edge_targets=edge_targets)
        return self._run_current_attribution(dataloader, edge_targets=edge_targets, progress_label=progress_label)

    def run_node_attribution(self, dataloader: DataLoader) -> list[ComponentScore]:
        assert self.model is not None
        assert self.component_registry is not None
        assert self.device is not None
        localization_mode = normalize_localization_mode(self.config.localization_mode)
        if localization_mode == "future":
            return self._run_future_node_attribution(dataloader)
        return self._run_current_node_attribution(dataloader)

    def _run_current_node_attribution(self, dataloader: DataLoader) -> list[ComponentScore]:
        assert self.model is not None
        assert self.component_registry is not None
        assert self.device is not None
        targets = self.component_registry.targets()
        scorer = ComponentAttributionScorer(
            targets=targets,
            score_token_mode=self.config.score_token_mode,
            score_normalization=self.config.score_normalization,
            localization_mode=normalize_localization_mode(self.config.localization_mode),
        )
        cache = ComponentActivationCache(targets, capture_device=self.config.capture_device)
        cache.register()
        try:
            for batch in tqdm(dataloader, desc="EAP_forLogicalCircuit node attribution"):
                batch = batch.to(self.device)
                cache.clear_batch()
                with torch.no_grad(), cache.capture("clean"):
                    ensure_src_on_path()
                    from modeling_patches import sequential_position_ids

                    self.model(
                        input_ids=batch.clean_input_ids,
                        attention_mask=batch.clean_attention_mask,
                        position_ids=sequential_position_ids(batch.clean_input_ids),
                        use_cache=False,
                    )
                self.model.zero_grad(set_to_none=True)
                with cache.capture("corrupted"):
                    loss = self._compute_loss(batch, input_kind="corrupted")
                loss.backward()
                scorer.score_batch(
                    clean_outputs=cache.clean_outputs,
                    corrupted_outputs=cache.corrupted_outputs,
                    output_grads=cache.output_grads,
                    attention_mask=batch.corrupted_attention_mask.detach().cpu(),
                    label_positions=self._label_positions(batch, "corrupted").detach().cpu(),
                    clean_inputs=cache.clean_inputs,
                    corrupted_inputs=cache.corrupted_inputs,
                )
                self.model.zero_grad(set_to_none=True)
                cache.clear_batch()
        finally:
            cache.remove()
        return scorer.finalize()

    def _run_future_node_attribution(self, dataloader: DataLoader) -> list[ComponentScore]:
        assert self.model is not None
        assert self.component_registry is not None
        assert self.device is not None
        targets = self.component_registry.targets()
        k_values = _future_step_k_values(self.config)
        if self._future_component_state_dict is None:
            self._future_component_state_dict = _load_sampled_future_component_state_dict(
                model=self.model,
                targets=targets,
                config=self.config,
            )
        future_state_dict = self._future_component_state_dict
        score_runs: list[list[ComponentScore]] = []
        per_k_summaries = []
        for sample_idx, future_step_k in enumerate(k_values, start=1):
            print(
                f"[Future K sampling] Running node sample {sample_idx}/{len(k_values)} "
                f"with future_step_k={future_step_k:.4f}"
            )
            component_config = _component_config_from_logical_config(
                replace(self.config, future_step_k=future_step_k)
            )
            scorer = FutureLocalizationScorer(
                model=self.model,
                targets=targets,
                config=component_config,
                device=self.device,
                future_state_dict=future_state_dict,
            )
            score_runs.append(scorer.score(dataloader))
            per_k_summaries.append(scorer.last_summary)
        if len(score_runs) == 1:
            self._future_localization_summary = per_k_summaries[0]
            return score_runs[0]
        aggregated_scores = _aggregate_future_component_k_scores(score_runs=score_runs, k_values=k_values)
        self._future_localization_summary = {
            "aggregation": "mean_scores_and_mean_abs_raw_ranks",
            "future_step_k_values": k_values,
            "future_step_k_min": self.config.future_step_k_min,
            "future_step_k_max": self.config.future_step_k_max,
            "future_step_k_samples": len(k_values),
            "future_step_k_seed": self.config.future_step_k_seed,
            "score_definition": "mean of per-K node/component attribution scores",
            "rank_definition": "mean rank from per-K abs(raw_score) rankings; lower mean_raw_rank is better",
            "rank_score_definition": "node_count + 1 - mean_raw_rank",
            "per_k": per_k_summaries,
        }
        return aggregated_scores

    def _run_current_attribution(
        self,
        dataloader: DataLoader,
        edge_targets: list[EdgeTarget] | None = None,
        progress_label: str | None = None,
    ):
        if str(self.config.localization_mode).lower() != "current":
            raise ValueError("_run_current_attribution requires localization_mode='current'")
        assert self.model is not None
        assert self.device is not None
        if edge_targets is None:
            assert self.registry is not None
            edge_targets = self.registry.edge_targets()
        scorer = CurrentEdgeAttributionScorer(
            edge_targets=edge_targets,
            score_token_mode=self.config.score_token_mode,
            score_normalization=self.config.score_normalization,
        )
        cache = DestinationInputCache(
            edge_targets=edge_targets,
            capture_device=self.config.capture_device,
            detach_tensors=True,
        )
        cache.register()
        try:
            for batch in tqdm(dataloader, desc=progress_label or "EAP_forLogicalCircuit current attribution"):
                batch = batch.to(self.device)
                cache.clear_batch()
                with torch.no_grad(), cache.capture("clean"):
                    ensure_src_on_path()
                    from modeling_patches import sequential_position_ids

                    self.model(
                        input_ids=batch.clean_input_ids,
                        attention_mask=batch.clean_attention_mask,
                        position_ids=sequential_position_ids(batch.clean_input_ids),
                        use_cache=False,
                    )
                self.model.zero_grad(set_to_none=True)
                with cache.capture("corrupted"):
                    loss = self._compute_loss(batch, input_kind="corrupted")
                loss.backward()
                scorer.score_batch(
                    clean_inputs=cache.clean_inputs,
                    corrupted_inputs=cache.corrupted_inputs,
                    input_grads=cache.input_grads,
                    attention_mask=batch.corrupted_attention_mask.detach().cpu(),
                    label_positions=self._label_positions(batch, "corrupted").detach().cpu(),
                    source_clean_outputs=cache.clean_source_outputs,
                    source_corrupted_outputs=cache.corrupted_source_outputs,
                    output_grads=cache.output_grads,
                )
                self.model.zero_grad(set_to_none=True)
                cache.clear_batch()
        finally:
            cache.remove()
        return scorer.finalize()

    def _run_future_attribution(self, dataloader: DataLoader, edge_targets: list[EdgeTarget] | None = None):
        assert self.model is not None
        assert self.device is not None
        if edge_targets is None:
            assert self.registry is not None
            edge_targets = self.registry.edge_targets()
        k_values = _future_step_k_values(self.config)
        future_state_dict = (
            _load_sampled_future_state_dict(model=self.model, edge_targets=edge_targets, config=self.config)
            if len(k_values) > 1
            else None
        )
        score_runs: list[list[EdgeScore]] = []
        per_k_summaries = []
        for sample_idx, future_step_k in enumerate(k_values, start=1):
            print(
                f"[Future K sampling] Running sample {sample_idx}/{len(k_values)} "
                f"with future_step_k={future_step_k:.4f}"
            )
            sample_config = replace(self.config, future_step_k=future_step_k)
            scorer = FutureEdgeLocalizationScorer(
                model=self.model,
                edge_targets=edge_targets,
                config=sample_config,
                device=self.device,
                future_state_dict=future_state_dict,
            )
            score_runs.append(scorer.score(dataloader))
            per_k_summaries.append(scorer.last_summary)
        if len(score_runs) == 1:
            self._future_localization_summary = per_k_summaries[0]
            return score_runs[0]
        aggregated_scores = _aggregate_future_k_scores(score_runs=score_runs, k_values=k_values)
        self._future_localization_summary = {
            "aggregation": "mean_scores_and_mean_abs_raw_ranks",
            "future_step_k_values": k_values,
            "future_step_k_min": self.config.future_step_k_min,
            "future_step_k_max": self.config.future_step_k_max,
            "future_step_k_samples": len(k_values),
            "future_step_k_seed": self.config.future_step_k_seed,
            "score_definition": "mean of per-K attribution scores",
            "rank_definition": "mean rank from per-K abs(raw_score) rankings; lower mean_raw_rank is better",
            "rank_score_definition": "edge_count + 1 - mean_raw_rank",
            "per_k": per_k_summaries,
        }
        return aggregated_scores

    def _compute_loss(self, batch, input_kind: str = "clean") -> torch.Tensor:
        if self.config.metric == "task_loss":
            ensure_src_on_path()
            from training_losses import task_loss

            input_ids, attention_mask, labels = self._loss_inputs(batch, input_kind)
            loss, _outputs = task_loss(
                self.model,
                (input_ids, attention_mask, labels),
            )
            return loss
        if self.config.metric == "logit_diff":
            return self._logit_diff_loss(batch, input_kind=input_kind)
        raise ValueError(f"Unsupported metric: {self.config.metric}")

    def _loss_inputs(self, batch, input_kind: str):
        if input_kind == "clean":
            return batch.clean_input_ids, batch.clean_attention_mask, batch.labels
        if input_kind != "corrupted":
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        labels = torch.full_like(batch.corrupted_input_ids, -100)
        positions = self._label_positions(batch, "corrupted")
        rows = torch.arange(batch.corrupted_input_ids.size(0), device=batch.corrupted_input_ids.device)
        labels[rows, positions] = batch.correct_idx.to(labels.device)
        return batch.corrupted_input_ids, batch.corrupted_attention_mask, labels

    def _label_positions(self, batch, input_kind: str) -> torch.Tensor:
        if input_kind == "clean":
            return batch.label_positions
        if input_kind != "corrupted":
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        return batch.corrupted_attention_mask.long().sum(dim=1).sub(1).clamp_min(0)

    def _logit_diff_loss(self, batch, input_kind: str = "clean") -> torch.Tensor:
        ensure_src_on_path()
        from modeling_patches import sequential_position_ids

        if input_kind == "clean":
            input_ids = batch.clean_input_ids
            attention_mask = batch.clean_attention_mask
            positions = batch.label_positions.to(input_ids.device)
        elif input_kind == "corrupted":
            input_ids = batch.corrupted_input_ids
            attention_mask = batch.corrupted_attention_mask
            positions = self._label_positions(batch, "corrupted").to(input_ids.device)
        else:
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=sequential_position_ids(input_ids),
            use_cache=False,
        )
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        logits = outputs.logits[rows, positions, :].float()
        vocab_size = logits.size(-1)
        valid = (
            (batch.correct_idx >= 0)
            & (batch.correct_idx < vocab_size)
            & (batch.incorrect_idx >= 0)
            & (batch.incorrect_idx < vocab_size)
        )
        if not valid.any():
            return logits.sum() * 0.0
        valid_logits = logits[valid]
        correct = batch.correct_idx[valid].to(logits.device)
        incorrect = batch.incorrect_idx[valid].to(logits.device)
        logit_diff = valid_logits.gather(1, correct[:, None]).squeeze(1) - valid_logits.gather(
            1, incorrect[:, None]
        ).squeeze(1)
        return -logit_diff.mean()


def _future_step_k_values(config: EAPLogicalCircuitConfig) -> list[float]:
    sample_count = int(config.future_step_k_samples)
    if sample_count < 1:
        raise ValueError("future_step_k_samples must be >= 1")
    if sample_count == 1:
        return [round(float(config.future_step_k), 4)]
    min_units = _decimal_units(config.future_step_k_min, ROUND_CEILING)
    max_units = _decimal_units(config.future_step_k_max, ROUND_FLOOR)
    if min_units > max_units:
        raise ValueError("future_step_k_min must be <= future_step_k_max after 4-decimal rounding")
    capacity = max_units - min_units + 1
    if sample_count > capacity:
        raise ValueError(
            f"Cannot sample {sample_count} unique K values from the 4-decimal grid in "
            f"[{config.future_step_k_min}, {config.future_step_k_max}] with capacity {capacity}."
        )
    rng = random.Random(int(config.future_step_k_seed))
    sampled_units = sorted(rng.sample(range(min_units, max_units + 1), sample_count))
    return [unit / 10000.0 for unit in sampled_units]


def _reverse_pair_batch(batch: PairBatch) -> PairBatch:
    # induction_or semantics: swap clean/corrupted inputs and swap correct/incorrect labels.
    return PairBatch(
        clean_input_ids=batch.corrupted_input_ids,
        clean_attention_mask=batch.corrupted_attention_mask,
        corrupted_input_ids=batch.clean_input_ids,
        corrupted_attention_mask=batch.clean_attention_mask,
        labels=batch.labels,
        correct_idx=batch.incorrect_idx,
        incorrect_idx=batch.correct_idx,
        label_positions=batch.label_positions,
    )


def _graph_candidate_edge_targets(
    registry: GraphRegistry,
    component_scores: list[ComponentScore],
    graph_metadata: dict,
    node_topn: int,
) -> tuple[list[EdgeTarget], dict]:
    candidate_circuit = build_node_induced_circuit(
        component_scores=component_scores,
        graph_metadata=graph_metadata,
        node_topn=node_topn,
        circuit_name="graph_edge_attribution_filter",
    )
    candidate_edge_ids = {edge.edge_id for edge in candidate_circuit.edges}
    all_edge_targets = registry.edge_targets()
    filtered_targets = [target for target in all_edge_targets if target.edge_id in candidate_edge_ids]
    found_edge_ids = {target.edge_id for target in filtered_targets}
    missing_edge_ids = sorted(candidate_edge_ids - found_edge_ids)
    summary = {
        "scope": "graph_node_topn_candidate_edges",
        "graph_node_topn": int(node_topn),
        "full_edge_target_count": len(all_edge_targets),
        "candidate_edge_count": len(candidate_edge_ids),
        "filtered_edge_target_count": len(filtered_targets),
        "missing_candidate_edge_count": len(missing_edge_ids),
        "missing_candidate_edge_ids_preview": missing_edge_ids[:20],
    }
    if candidate_edge_ids and not filtered_targets:
        raise ValueError(
            "Graph export candidate edge filter produced no edge targets. "
            f"candidate_edge_count={len(candidate_edge_ids)}, missing_preview={missing_edge_ids[:10]}"
        )
    return filtered_targets, summary


def _load_sampled_future_state_dict(
    model: torch.nn.Module,
    edge_targets,
    config: EAPLogicalCircuitConfig,
) -> dict[str, torch.Tensor]:
    from EAP_forComponent.future_localization import _load_future_state_dict

    if not config.future_model_name_or_path:
        raise ValueError("future_model_name_or_path is required when localization_mode='future'")
    parameter_names = _future_delta_parameter_names(model=model, edge_targets=edge_targets, config=config)
    return _load_future_state_dict(
        model_name_or_path=str(config.future_model_name_or_path),
        cache_dir=config.future_model_cache_dir or config.cache_dir,
        parameter_names=parameter_names,
    )


def _future_delta_parameter_names(
    model: torch.nn.Module,
    edge_targets,
    config: EAPLogicalCircuitConfig,
) -> list[str]:
    base_parameters = dict(model.named_parameters())
    if config.future_delta_parameter_filter:
        try:
            pattern = re.compile(config.future_delta_parameter_filter)
        except re.error as error:
            raise ValueError(f"Invalid future_delta_parameter_filter regex: {error}") from error
        return [name for name in base_parameters if pattern.search(name)]
    edge_parameter_names = sorted(
        {
            parameter_name
            for target in edge_targets
            for parameter_name in getattr(target, "delta_parameter_names", (target.destination_parameter_name,))
        }
    )
    return [name for name in edge_parameter_names if name in base_parameters]


def _load_sampled_future_component_state_dict(
    model: torch.nn.Module,
    targets,
    config: EAPLogicalCircuitConfig,
) -> dict[str, torch.Tensor]:
    if not config.future_model_name_or_path:
        raise ValueError("future_model_name_or_path is required when localization_mode='future'")
    parameter_names = _future_component_delta_parameter_names(model=model, targets=targets, config=config)
    return _load_future_state_dict(
        model_name_or_path=str(config.future_model_name_or_path),
        cache_dir=config.future_model_cache_dir or config.cache_dir,
        parameter_names=parameter_names,
    )


def _future_component_delta_parameter_names(
    model: torch.nn.Module,
    targets,
    config: EAPLogicalCircuitConfig,
) -> list[str]:
    base_parameters = dict(model.named_parameters())
    if config.future_delta_parameter_filter:
        try:
            pattern = re.compile(config.future_delta_parameter_filter)
        except re.error as error:
            raise ValueError(f"Invalid future_delta_parameter_filter regex: {error}") from error
        return [name for name in base_parameters if pattern.search(name)]
    target_names = sorted({target.parameter_name for target in targets})
    return [name for name in target_names if name in base_parameters]


def _component_config_from_logical_config(config: EAPLogicalCircuitConfig) -> EAPComponentConfig:
    return EAPComponentConfig(
        model_name_or_path=config.model_name_or_path,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        output_dir=config.output_dir,
        dataset_name=config.dataset_name,
        data_path=config.data_path,
        corruption_column=config.corruption_column,
        input_format=config.input_format,
        metric=config.metric,
        max_samples=config.max_samples,
        batch_size=config.batch_size,
        max_length=config.max_length,
        target_modules=config.target_modules,
        attention_granularity=config.component_granularity,
        localization_mode=config.localization_mode,
        future_model_name_or_path=config.future_model_name_or_path,
        future_model_cache_dir=config.future_model_cache_dir,
        future_step_k=config.future_step_k,
        future_step_k_min=config.future_step_k_min,
        future_step_k_max=config.future_step_k_max,
        future_step_k_samples=config.future_step_k_samples,
        future_step_k_seed=config.future_step_k_seed,
        future_delta_parameter_filter=config.future_delta_parameter_filter,
        future_hvp_strategy=config.future_hvp_strategy,
        future_finite_difference_epsilon=config.future_finite_difference_epsilon,
        score_token_mode=config.score_token_mode,
        score_normalization=config.score_normalization,
        rank_score_source=config.rank_score_source,
        min_rank=config.min_rank,
        max_rank=config.max_rank,
        rank_budget=config.rank_budget,
        rank_multiple=config.rank_multiple,
        head_to_matrix_aggregation=config.head_to_matrix_aggregation,
        mask_fill_strategy=config.mask_fill_strategy,
        mask_seed=config.mask_seed,
        mask_min_keep_ratio=config.mask_min_keep_ratio,
        mask_max_keep_ratio=config.mask_max_keep_ratio,
        mask_all_parameters=config.mask_all_parameters,
        device=config.device,
        use_bfloat16=config.use_bfloat16,
        use_cpu=config.use_cpu,
        cache_dir=config.cache_dir,
        capture_device=config.capture_device,
        metadata=config.metadata,
    )


def _decimal_units(value: float, rounding) -> int:
    return int((Decimal(str(value)) * Decimal("10000")).to_integral_value(rounding=rounding))


def _aggregate_future_k_scores(
    score_runs: list[list[EdgeScore]],
    k_values: list[float],
) -> list[EdgeScore]:
    if not score_runs:
        return []
    if len(score_runs) != len(k_values):
        raise ValueError(f"Expected {len(k_values)} score runs, got {len(score_runs)}")
    first_ids = [score.edge_id for score in score_runs[0]]
    first_id_set = set(first_ids)
    score_maps = []
    for run_idx, scores in enumerate(score_runs, start=1):
        score_map = {score.edge_id: score for score in scores}
        if set(score_map) != first_id_set:
            missing = sorted(first_id_set - set(score_map))[:10]
            extra = sorted(set(score_map) - first_id_set)[:10]
            raise ValueError(f"Future K sample {run_idx} edge set mismatch: missing={missing}, extra={extra}")
        score_maps.append(score_map)
    rank_maps = [_abs_raw_rank_map(scores) for scores in score_runs]
    edge_count = len(first_ids)
    aggregated_scores = []
    for base_score in score_runs[0]:
        edge_id = base_score.edge_id
        edge_scores = [score_map[edge_id] for score_map in score_maps]
        mean_raw_rank = mean(rank_map[edge_id] for rank_map in rank_maps)
        aggregated_scores.append(
            replace(
                base_score,
                raw_score=_mean_score_field(edge_scores, "raw_score"),
                abs_score=_mean_score_field(edge_scores, "abs_score"),
                mean_score=_mean_score_field(edge_scores, "mean_score"),
                sqrt_numel_score=_mean_score_field(edge_scores, "sqrt_numel_score"),
                rank_score=float(edge_count + 1 - mean_raw_rank),
                current_score=_mean_optional_score_field(edge_scores, "current_score"),
                future_directional_score_theta=_mean_optional_score_field(
                    edge_scores, "future_directional_score_theta"
                ),
                future_directional_score_theta_hat=_mean_optional_score_field(
                    edge_scores, "future_directional_score_theta_hat"
                ),
                future_correction=_mean_optional_score_field(edge_scores, "future_correction"),
                future_step_k=None,
                mean_raw_rank=float(mean_raw_rank),
            )
        )
    return aggregated_scores


def _abs_raw_rank_map(scores: list[EdgeScore]) -> dict[str, int]:
    ranked = sorted(scores, key=lambda score: (-abs(float(score.raw_score)), score.edge_id))
    return {score.edge_id: rank for rank, score in enumerate(ranked, start=1)}


def _mean_score_field(scores: list[EdgeScore], field_name: str) -> float:
    return float(mean(float(getattr(score, field_name)) for score in scores))


def _mean_optional_score_field(scores: list[EdgeScore], field_name: str) -> float | None:
    values = [getattr(score, field_name) for score in scores]
    if any(value is None for value in values):
        return None
    return float(mean(float(value) for value in values))


def _aggregate_future_component_k_scores(
    score_runs: list[list[ComponentScore]],
    k_values: list[float],
) -> list[ComponentScore]:
    if not score_runs:
        return []
    if len(score_runs) != len(k_values):
        raise ValueError(f"Expected {len(k_values)} score runs, got {len(score_runs)}")
    first_names = [score.component_name for score in score_runs[0]]
    first_name_set = set(first_names)
    score_maps = []
    for run_idx, scores in enumerate(score_runs, start=1):
        score_map = {score.component_name: score for score in scores}
        if set(score_map) != first_name_set:
            missing = sorted(first_name_set - set(score_map))[:10]
            extra = sorted(set(score_map) - first_name_set)[:10]
            raise ValueError(
                f"Future K sample {run_idx} component set mismatch: missing={missing}, extra={extra}"
            )
        score_maps.append(score_map)
    rank_maps = [_abs_raw_component_rank_map(scores) for scores in score_runs]
    component_count = len(first_names)
    aggregated_scores = []
    for base_score in score_runs[0]:
        component_name = base_score.component_name
        component_scores = [score_map[component_name] for score_map in score_maps]
        mean_raw_rank = mean(rank_map[component_name] for rank_map in rank_maps)
        aggregated_scores.append(
            replace(
                base_score,
                raw_score=_mean_score_field(component_scores, "raw_score"),
                abs_score=_mean_score_field(component_scores, "abs_score"),
                mean_score=_mean_score_field(component_scores, "mean_score"),
                sqrt_numel_score=_mean_score_field(component_scores, "sqrt_numel_score"),
                rank_score=float(component_count + 1 - mean_raw_rank),
                current_score=_mean_optional_score_field(component_scores, "current_score"),
                future_directional_score_theta=_mean_optional_score_field(
                    component_scores, "future_directional_score_theta"
                ),
                future_directional_score_theta_hat=_mean_optional_score_field(
                    component_scores, "future_directional_score_theta_hat"
                ),
                future_correction=_mean_optional_score_field(component_scores, "future_correction"),
                future_step_k=None,
                mean_raw_rank=float(mean_raw_rank),
            )
        )
    return aggregated_scores


def _abs_raw_component_rank_map(scores: list[ComponentScore]) -> dict[str, int]:
    ranked = sorted(scores, key=lambda score: (-abs(float(score.raw_score)), score.component_name))
    return {score.component_name: rank for rank, score in enumerate(ranked, start=1)}
