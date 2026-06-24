from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from EAP_forNeuron.data import load_pair_dataset
from EAP_forNeuron.hooks import LinearActivationCache
from EAP_forNeuron.mask_io import MaskSpec, save_outputs
from EAP_forNeuron.model_loader import ensure_src_on_path, load_model_and_tokenizer
from EAP_forNeuron.schemas import EAPNeuronConfig, PairBatch
from EAP_forNeuron.scorer import ParameterNeuronScorer
from EAP_forNeuron.selection import NeuronMaskSelector


class EAPForNeuronRunner:
    def __init__(self, config: EAPNeuronConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.mask_spec: MaskSpec | None = None

    def run(self) -> dict[str, Path]:
        if self.config.mask_path is None:
            raise ValueError("mask_path is required")
        if self.config.output_dir is None:
            raise ValueError("output_dir is required")
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
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        self.mask_spec = MaskSpec.load(
            mask_path=self.config.mask_path,
            model=self.model,
            target_modules=self.config.target_modules,
            include_lm_head=self.config.include_lm_head,
            include_embed_tokens=self.config.include_embed_tokens,
            unsupported_policy=self.config.unsupported_policy,
        )
        scores = self.compute_scores(dataloader)
        selector = NeuronMaskSelector(
            output_ratio=self.config.output_ratio,
            ratio_base=self.config.ratio_base,
            score_abs=self.config.score_abs,
            max_concat_candidates=self.config.max_concat_candidates,
        )
        new_mask, summary = selector.select(
            old_mask=self.mask_spec.mask,
            scores=scores,
            total_neuron_count=self.mask_spec.total_neuron_count,
        )
        if self.config.unsupported_policy == "keep":
            for name in self.mask_spec.skipped:
                if name in self.mask_spec.mask:
                    new_mask[name] = self.mask_spec.mask[name].clone()
                    kept = int(new_mask[name].sum().item())
                    if name in summary["per_parameter"]:
                        summary["per_parameter"][name]["kept"] = kept
                        summary["per_parameter"][name]["skipped"] = 0
            summary["actual_keep_count"] = int(sum(mask.sum().item() for mask in new_mask.values()))
        summary.update(
            {
                "input_mask_path": str(self.config.mask_path),
                "dataset_name": self.config.dataset_name,
                "data_path": str(self.config.data_path) if self.config.data_path else None,
                "model_name_or_path": self.config.model_name_or_path,
                "metric": self.config.metric,
                "score_token_mode": self.config.score_token_mode,
                "target_modules": list(self.config.target_modules),
                "max_samples": self.config.max_samples,
                "batch_size": self.config.batch_size,
            }
        )
        summary.update(self.config.metadata)
        paths = save_outputs(
            output_dir=self.config.output_dir,
            output_ratio=self.config.output_ratio,
            new_mask=new_mask,
            summary=summary,
            skipped=self.mask_spec.skipped,
            scores=scores,
            base_mask=self.mask_spec.mask,
            save_scores=self.config.save_scores,
        )
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return paths

    def compute_scores(self, dataloader: DataLoader) -> dict:
        assert self.model is not None
        assert self.mask_spec is not None
        assert self.device is not None
        scorer = ParameterNeuronScorer(
            targets=self.mask_spec.targets,
            score_dtype=self.config.score_dtype,
            row_chunk_size=self.config.row_chunk_size,
            score_token_mode=self.config.score_token_mode,
        )
        cache = LinearActivationCache(self.mask_spec.targets, capture_device="cpu")
        cache.register()
        try:
            for batch in tqdm(dataloader, desc="EAP_forNeuron attribution"):
                batch = batch.to(self.device)
                cache.clear_batch()
                with torch.no_grad(), cache.capture("corrupted"):
                    ensure_src_on_path()
                    from modeling_patches import sequential_position_ids

                    self.model(
                        input_ids=batch.corrupted_input_ids,
                        attention_mask=batch.corrupted_attention_mask,
                        position_ids=sequential_position_ids(batch.corrupted_input_ids),
                        use_cache=False,
                    )
                self.model.zero_grad(set_to_none=True)
                with cache.capture("clean"):
                    loss = self._compute_loss(batch)
                loss.backward()
                scorer.score_batch(
                    clean_inputs=cache.clean_inputs,
                    corrupted_inputs=cache.corrupted_inputs,
                    output_grads=cache.output_grads,
                    attention_mask=batch.clean_attention_mask.detach().cpu(),
                    label_positions=batch.label_positions.detach().cpu(),
                )
                self.model.zero_grad(set_to_none=True)
                cache.clear_batch()
        finally:
            cache.remove()
        return scorer.finalize()

    def _compute_loss(self, batch: PairBatch) -> torch.Tensor:
        if self.config.metric == "task_loss":
            ensure_src_on_path()
            from training_losses import task_loss

            loss, _outputs = task_loss(
                self.model,
                (batch.clean_input_ids, batch.clean_attention_mask, batch.labels),
            )
            return loss
        if self.config.metric == "logit_diff":
            return self._logit_diff_loss(batch)
        raise ValueError(f"Unsupported metric: {self.config.metric}")

    def _logit_diff_loss(self, batch: PairBatch) -> torch.Tensor:
        ensure_src_on_path()
        from modeling_patches import sequential_position_ids

        outputs = self.model(
            input_ids=batch.clean_input_ids,
            attention_mask=batch.clean_attention_mask,
            position_ids=sequential_position_ids(batch.clean_input_ids),
            use_cache=False,
        )
        rows = torch.arange(batch.clean_input_ids.size(0), device=batch.clean_input_ids.device)
        positions = batch.label_positions.to(batch.clean_input_ids.device)
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
