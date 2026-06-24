import contextlib
import json
import os
from dataclasses import dataclass
from datetime import datetime

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import get_finetuning_dataset
from finetuning import GenerateMask, get_finetuning_method
from metrics import (
    DEFAULT_FEW_SHOT_TASKS,
    eval_few_shots,
    eval_task_accuracy,
    eval_task_accuracy_in_memory,
)
from modeling_patches import patch_mistral_rotary_embedding
from model.lora_utils import apply_lora_to_model
from optim import create_sophia_optimizer
from pruner.utils import find_layers


@dataclass
class TaskTrainingSpec:
    task_key: str
    display_name: str
    mask_path: str | None
    lora_info_dir: str | None


@dataclass
class StageSpec:
    name: str
    num_epochs: int
    task_specs: list[TaskTrainingSpec]
    mask_path: str | None
    lora_info_dir: str | None
    aggregate_losses: bool


class Finetuning:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.probing = bool(kwargs.get("probing", False))
        self.finetuning_method = kwargs["finetuning_method"]
        self.legacy_mask_path = kwargs.get("mask_path", None)
        self.target_mask_path = kwargs.get("target_mask_path", None)
        self.pervasiveness_mask_paths = self._parse_optional_paths(
            kwargs.get("pervasiveness_mask_paths", "")
        )
        self.conflict_mask_path = kwargs.get("conflict_mask_path", None)
        self.all_component_mask_path = kwargs.get("all_component_mask_path", None)
        self.mask_score_type = kwargs.get("mask_score_type", None)
        self.mask_ratio = kwargs.get("mask_ratio", None)
        self.mask_calibration_samples = kwargs.get("mask_calibration_samples", 128)

        self.batch_size = kwargs["batch_size"]
        self.dataset_names = kwargs["dataset_names"]
        self.dataset_seed = kwargs["dataset_seed"]
        self.target_ratio = kwargs.get("target_ratio", 1.0)
        self.target_holdout_as_pervasiveness = kwargs.get(
            "target_holdout_as_pervasiveness", False
        )

        self.num_epochs = kwargs["num_epochs"]
        self.num_devices = int(os.environ.get("WORLD_SIZE", 1))
        self.lr = kwargs["lr"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.weight_decay = kwargs["weight_decay"]
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.max_steps = kwargs.get("max_steps", -1)
        self.resume_path = kwargs.get("resume_path", None)
        self.use_lora = kwargs.get("use_lora", False)
        self.lora_mode = kwargs.get("lora_mode", "standard")
        self.legacy_lora_info_dir = kwargs.get("lora_info_dir", None)
        self.target_lora_info_dir = kwargs.get("target_lora_info_dir", None)
        self.pervasiveness_lora_info_dirs = self._parse_optional_paths(
            kwargs.get("pervasiveness_lora_info_dirs", "")
        )
        self.conflict_lora_info_dir = kwargs.get("conflict_lora_info_dir", None)
        self.all_component_lora_info_dir = kwargs.get("all_component_lora_info_dir", None)
        self.multi_task_schedule = kwargs.get("multi_task_schedule", "two_stage_alternating")
        self.stage1_num_epochs = int(kwargs.get("stage1_num_epochs", 1))
        self.stage2_num_epochs = int(kwargs.get("stage2_num_epochs", 1))
        self.lora_target_modules = kwargs.get("lora_target_modules", "auto")
        self.lora_default_rank = kwargs.get("lora_default_rank", 8)
        self.lora_alpha = kwargs.get("lora_alpha", 32)
        self.lora_dropout = kwargs.get("lora_dropout", 0.05)
        self.lora_alpha_strategy = kwargs.get("lora_alpha_strategy", "constant")
        self.lora_rank_pattern_path = kwargs.get("lora_rank_pattern_path", None)
        self.lora_alpha_pattern_path = kwargs.get("lora_alpha_pattern_path", None)
        self.lora_component_scores_path = kwargs.get("lora_component_scores_path", None)
        self.lora_head_min_rank = kwargs.get("lora_head_min_rank", 0)
        self.lora_head_max_rank = kwargs.get("lora_head_max_rank", 32)
        self.lora_head_rank_multiple = kwargs.get("lora_head_rank_multiple", 1)
        self.lora_head_rank_score_source = kwargs.get(
            "lora_head_rank_score_source", "rank_score"
        )
        self.lora_report = None
        self.use_cpu = bool(kwargs.get("use_cpu", False)) or os.environ.get(
            "CSAT_FORCE_CPU", ""
        ).lower() in ("1", "true", "yes")

        self.alpha = kwargs.get("alpha", 0.0)
        self.target_weight = kwargs.get("target_weight", 1.0)
        self.pervasiveness_weight = kwargs.get("pervasiveness_weight", 1.0)
        self.kl_weight = kwargs.get("kl_weight", 1.0)

        self.sophia = kwargs.get("sophia", False)
        self.betas_low = kwargs.get("betas_low", 0.9)
        self.betas_high = kwargs.get("betas_high", 0.95)
        self.betas = (self.betas_low, self.betas_high)
        self.rho = kwargs.get("rho", 0.03)
        self.p = kwargs.get("p", 0.0)
        self.q = kwargs.get("q", 0.0)
        self.mu = kwargs.get("mu", 1e-3)

        self.target_eval = bool(kwargs.get("target_eval", True))
        self.pervasiveness_eval = bool(kwargs.get("pervasiveness_eval", True))
        self.pervasiveness_lm_eval = bool(kwargs.get("pervasiveness_lm_eval", False))
        self.eval_batch_size = kwargs.get("eval_batch_size", 8)
        self.pervasiveness_lm_eval_tasks = self._parse_tasks(
            kwargs.get("pervasiveness_lm_eval_tasks", "")
        ) or list(DEFAULT_FEW_SHOT_TASKS)

        self.if_llama = "llama" in self.model_name.lower()
        self.if_wanda = False
        self.optimizer = None
        self.mask = None
        self.mask_name = None
        self._mask_cache = {}
        self._lora_module_cache = {}
        self._base_lora_trainable = {}
        self.active_lora_info_dir = None
        self.latest_checkpoint_path = None
        self.mask_path = self._normalize_optional_path(
            self.target_mask_path or self.legacy_mask_path
        )
        self.lora_info_dir = self._normalize_optional_path(
            self.target_lora_info_dir or self.legacy_lora_info_dir
        )
        if self._multitask_lora_enabled():
            self.lora_info_dir = self._normalize_optional_path(self.all_component_lora_info_dir)
        self._validate_multitask_lists()
        if self.probing:
            self.target_ratio = 0.01
            self.num_epochs = 1
            self.batch_size = 1
            self.lr = 1e-7
            self.max_steps = -1
            self.target_eval = False
            self.pervasiveness_eval = False
            self.pervasiveness_lm_eval = False
            print(
                "[Probing] Enabled: target_ratio=0.01, num_epochs=1, batch_size=1, lr=1e-7, "
                "all evaluation disabled."
            )

    def _parse_tasks(self, tasks):
        if tasks is None:
            return []
        if isinstance(tasks, list):
            return [task for task in tasks if task]
        return [task.strip() for task in str(tasks).split(",") if task.strip()]

    def _parse_optional_paths(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return [
                item.strip()
                for item in [str(path) for path in value]
                if item and item.strip().lower() != "none"
            ]
        text = str(value).strip()
        if not text or text.lower() == "none":
            return []
        return [
            item.strip()
            for item in text.split(",")
            if item.strip() and item.strip().lower() != "none"
        ]

    def _normalize_optional_path(self, value):
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "none":
            return None
        return text

    def _validate_multitask_lists(self):
        pervasiveness_count = len(self.dataset_names.get("pervasiveness", []))
        if self.pervasiveness_mask_paths and len(self.pervasiveness_mask_paths) != pervasiveness_count:
            raise ValueError(
                "pervasiveness_mask_paths count must match pervasiveness dataset count: "
                f"got {len(self.pervasiveness_mask_paths)} vs {pervasiveness_count}."
            )
        if self.pervasiveness_lora_info_dirs and len(self.pervasiveness_lora_info_dirs) != pervasiveness_count:
            raise ValueError(
                "pervasiveness_lora_info_dirs count must match pervasiveness dataset count: "
                f"got {len(self.pervasiveness_lora_info_dirs)} vs {pervasiveness_count}."
            )
        if self.stage1_num_epochs < 1 or self.stage2_num_epochs < 1:
            raise ValueError("stage1_num_epochs and stage2_num_epochs must be >= 1")

    def _pervasiveness_keys(self):
        names = self.dataset_names.get("pervasiveness", [])
        if not names:
            return []
        if len(names) == 1:
            return ["pervasiveness"]
        return [f"pervasiveness{idx + 1}" for idx in range(len(names))]

    def _multitask_mode_enabled(self):
        has_pervasiveness = len(self.dataset_names.get("pervasiveness", [])) > 0
        has_stage_paths = bool(self.pervasiveness_mask_paths) and bool(self.all_component_mask_path)
        return bool(has_pervasiveness and has_stage_paths and self.multi_task_schedule == "two_stage_alternating")

    def _multitask_lora_enabled(self):
        has_pervasiveness = len(self.dataset_names.get("pervasiveness", [])) > 0
        has_stage_paths = bool(self.pervasiveness_lora_info_dirs) and bool(self.all_component_lora_info_dir)
        return bool(
            self.use_lora
            and has_pervasiveness
            and has_stage_paths
            and self.multi_task_schedule == "two_stage_alternating"
        )

    def _resolved_multitask_config(self):
        return {
            "legacy_mask_path": self.legacy_mask_path,
            "target_mask_path": self.target_mask_path,
            "resolved_target_mask_path": self.mask_path,
            "pervasiveness_mask_paths": self.pervasiveness_mask_paths,
            "conflict_mask_path": self.conflict_mask_path,
            "all_component_mask_path": self.all_component_mask_path,
            "legacy_lora_info_dir": self.legacy_lora_info_dir,
            "target_lora_info_dir": self.target_lora_info_dir,
            "resolved_target_lora_info_dir": self.lora_info_dir,
            "pervasiveness_lora_info_dirs": self.pervasiveness_lora_info_dirs,
            "conflict_lora_info_dir": self.conflict_lora_info_dir,
            "all_component_lora_info_dir": self.all_component_lora_info_dir,
            "multi_task_schedule": self.multi_task_schedule,
            "stage1_num_epochs": self.stage1_num_epochs,
            "stage2_num_epochs": self.stage2_num_epochs,
            "multitask_mode_enabled": self._multitask_mode_enabled(),
            "multitask_lora_enabled": self._multitask_lora_enabled(),
            "dataset_names": self.dataset_names,
        }

    def _dump_multitask_config(self, logger):
        root = logger.get_root()
        output_path = os.path.join(root, "multitask_config_resolved.json")
        with open(output_path, "w") as file:
            json.dump(self._resolved_multitask_config(), file, indent=4)
        print(f"[Multitask Config] Saved resolved config to: {output_path}")

    def _training_args(self, logger_root, output_dir, **overrides):
        common = dict(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=not self.use_cpu,
            bf16_full_eval=False,
            logging_steps=max(1, self.max_steps // 20),
            logging_dir=f"{logger_root}/logs",
            output_dir=output_dir,
            optim="adamw_torch",
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            report_to=[],
        )
        common.update(overrides)
        return transformers.TrainingArguments(**common)

    def init_model(self):
        patch_mistral_rotary_embedding()
        load_kwargs = dict(
            pretrained_model_name_or_path=self.model_name,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
        )
        if self.use_cpu:
            load_kwargs["torch_dtype"] = torch.float32
            load_kwargs["device_map"] = "cpu"
            print("[use_cpu] Loading model on CPU for debugging.")
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = {"": 0}

        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        if not self.use_cpu:
            print("[device_map] Loading model on cuda:0 to avoid split lm_head zero-logit path.")
        model.config.use_cache = False
        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if tokenizer.pad_token_id is None:
            if self.if_llama:
                tokenizer.add_special_tokens({"pad_token": "[pad]"})
            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

        model.resize_token_embeddings(len(tokenizer))
        self.model = model
        self.tokenizer = tokenizer
        if self.use_lora:
            self.model, self.lora_report = apply_lora_to_model(
                model=self.model,
                mode=self.lora_mode,
                info_dir=self.lora_info_dir,
                target_modules=self.lora_target_modules,
                default_rank=self.lora_default_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                alpha_strategy=self.lora_alpha_strategy,
                rank_pattern_path=self.lora_rank_pattern_path,
                alpha_pattern_path=self.lora_alpha_pattern_path,
                component_scores_path=self.lora_component_scores_path,
                head_min_rank=self.lora_head_min_rank,
                head_max_rank=self.lora_head_max_rank,
                head_rank_multiple=self.lora_head_rank_multiple,
                head_rank_score_source=self.lora_head_rank_score_source,
            )
            print(f"[LoRA] {json.dumps(self.lora_report, indent=2)}")
            if hasattr(self.model, "print_trainable_parameters"):
                print(self.model.print_trainable_parameters())

            self._capture_base_lora_trainable()

    def _capture_base_lora_trainable(self):
        self._base_lora_trainable = {}
        for name, parameter in self.model.named_parameters():
            if "lora_" in name:
                self._base_lora_trainable[name] = bool(parameter.requires_grad)

    def _resolve_lora_info_dir(self, info_dir):
        if not info_dir:
            return None
        resolved = str(info_dir)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"LoRA info dir not found: {resolved}")
        return resolved

    def _lora_target_modules_for_standard(self):
        modules = [item.strip() for item in str(self.lora_target_modules).split(",") if item.strip()]
        if not modules or modules == ["auto"]:
            return ["q_proj", "v_proj"]
        return modules

    def _load_lora_module_set(self, info_dir):
        resolved = self._resolve_lora_info_dir(info_dir)
        if not resolved:
            return None
        if resolved in self._lora_module_cache:
            return self._lora_module_cache[resolved]

        module_names = set()
        if self.lora_mode == "projection_matrix":
            rank_pattern_path = os.path.join(resolved, "rank_pattern.json")
            if os.path.isfile(rank_pattern_path):
                with open(rank_pattern_path, "r") as file:
                    rank_pattern = json.load(file)
                module_names = {
                    str(module_name)
                    for module_name, rank in rank_pattern.items()
                    if int(rank) > 0
                }
        elif self.lora_mode == "head":
            component_scores_path = os.path.join(resolved, "component_scores.json")
            if os.path.isfile(component_scores_path):
                with open(component_scores_path, "r") as file:
                    scores = json.load(file)
                module_names = {
                    str(entry.get("module_name"))
                    for entry in scores
                    if entry.get("module_name")
                }
        else:
            module_names = set(self._lora_target_modules_for_standard())

        if not module_names:
            raise ValueError(
                f"No LoRA modules found in info dir: {resolved} (lora_mode={self.lora_mode})"
            )
        self._lora_module_cache[resolved] = module_names
        return module_names

    def _lora_param_matches_module(self, param_name, module_name):
        return f".{module_name}." in param_name or param_name.endswith(f".{module_name}")

    def _set_active_lora_modules(self, module_names, lora_info_dir=None, context=""):
        if not self.use_lora:
            return
        if not self._base_lora_trainable:
            self._capture_base_lora_trainable()
        active_params = 0
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if "lora_" not in name:
                continue
            total_params += int(parameter.numel())
            base_trainable = self._base_lora_trainable.get(name, False)
            if not base_trainable:
                parameter.requires_grad_(False)
                continue
            if module_names is None:
                enabled = True
            else:
                enabled = any(self._lora_param_matches_module(name, module_name) for module_name in module_names)
            parameter.requires_grad_(enabled)
            if enabled:
                active_params += int(parameter.numel())
        self.active_lora_info_dir = lora_info_dir
        print(
            f"[Multitask-LoRA] {context} active_params={active_params}/{total_params}, "
            f"info_dir={lora_info_dir}"
        )

    def init_dataset(self):
        (
            self.finetuning_dataset,
            self.target_test_datasets,
            self.pervasiveness_test_datasets,
            self.finetuning_collator,
            self.test_collator,
        ) = get_finetuning_dataset(
            self.dataset_names,
            self.tokenizer,
            self.dataset_seed,
            self.target_ratio,
            self.target_holdout_as_pervasiveness,
            self.if_llama,
        )
        denominator = self.batch_size * self.gradient_accumulation_steps * self.num_devices
        if self.max_steps == -1:
            self.max_steps = max(
                1, int(self.num_epochs * len(self.finetuning_dataset)) // denominator
            )
            self.steps_per_epoch = max(1, len(self.finetuning_dataset) // denominator)
        else:
            self.steps_per_epoch = max(1, self.max_steps // max(1, self.num_epochs))

    def _is_wanda_mask(self, mask):
        if not isinstance(mask, dict) or not mask:
            return False
        return all(isinstance(key, int) or str(key).isdigit() for key in mask.keys())

    def _move_mask_to_device(self, mask):
        if mask is None:
            return
        self.if_wanda = self._is_wanda_mask(mask)
        if self.if_wanda:
            try:
                layers = self.model.model.layers
            except AttributeError:
                layers = self.model.model.decoder.layers
            cnt = 0
            with torch.no_grad():
                for layer in layers:
                    subset = find_layers(layer)
                    for name in subset:
                        key = cnt if cnt in mask else str(cnt)
                        if key in mask:
                            mask[key] = mask[key].to(subset[name].weight.device)
                        cnt += 1
            return

        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in mask:
                    mask[name] = mask[name].to(parameter.device)

    def _load_mask_from_path(self, mask_path):
        if not mask_path:
            return None
        resolved = self._resolve_mask_path(str(mask_path))
        if resolved in self._mask_cache:
            return self._mask_cache[resolved]
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Mask file not found: {resolved}")
        mask = torch.load(resolved, map_location=torch.device("cpu"))
        self._move_mask_to_device(mask)
        self._mask_cache[resolved] = mask
        return mask

    def _set_active_mask(self, mask, mask_name=None):
        self.mask = mask
        self.mask_name = mask_name
        self.if_wanda = self._is_wanda_mask(mask) if mask is not None else False
        if hasattr(self, "finetuning_trainer") and self.finetuning_trainer is not None:
            self.finetuning_trainer.mask = mask
            self.finetuning_trainer.if_wanda = self.if_wanda

    def init_mask(self, logger):
        if not self.mask_path or str(self.mask_path).lower() == "none":
            self._set_active_mask(None, mask_name="none")
            return
        mask_path = self._resolve_mask_path(str(self.mask_path))
        if os.path.exists(mask_path):
            print(f"Loading finetuning mask: {mask_path}")
            loaded_mask = torch.load(mask_path, map_location=torch.device("cpu"))
            self._move_mask_to_device(loaded_mask)
            self._set_active_mask(loaded_mask, mask_name=mask_path)
            return
        generated_mask = self._generate_mask(mask_path, logger)
        if generated_mask is not None:
            self._move_mask_to_device(generated_mask)
            self._set_active_mask(generated_mask, mask_name=mask_path)
        else:
            self._set_active_mask(None, mask_name="none")

    def _resolve_mask_path(self, mask_path):
        if not os.path.isdir(mask_path):
            return mask_path
        component_mask_path = os.path.join(mask_path, "component_mask.pt")
        if os.path.isfile(component_mask_path):
            return component_mask_path
        raise IsADirectoryError(
            f"mask_path points to a directory: {mask_path}. Pass a single mask file such as "
            f"{component_mask_path}."
        )

    def _parse_mask_path(self, mask_path):
        parts = mask_path.split("/")
        score_type = self.mask_score_type or (parts[-2] if len(parts) > 1 else "gradient")
        if self.mask_ratio is not None:
            ratio = float(self.mask_ratio)
        else:
            ratio = float(os.path.basename(mask_path).split("_")[-1].split(".p")[0])
        mask_dir = mask_path.replace(f"with_{ratio}.pt", "")
        if mask_dir == mask_path:
            mask_dir = os.path.dirname(mask_path)
        return score_type, ratio, mask_dir

    def _generate_mask(self, mask_path, logger):
        score_type, ratio, mask_dir = self._parse_mask_path(mask_path)
        os.makedirs(mask_dir, exist_ok=True)
        mask_args = self._training_args(
            logger_root=logger.get_root(),
            output_dir=mask_dir,
            save_steps=self.steps_per_epoch,
            save_total_limit=3,
        )
        train_dataset = self.finetuning_dataset
        if score_type == "wanda" or self.mask_calibration_samples:
            train_dataset, _, _, _, _ = get_finetuning_dataset(
                self.dataset_names,
                self.tokenizer,
                self.dataset_seed,
                self.mask_calibration_samples,
                False,
                self.if_llama,
            )
        generator = GenerateMask(
            score_type=score_type,
            ratios=[ratio],
            mask_dir=mask_dir,
            model=self.model,
            data_collator=self.finetuning_collator,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=None,
            compute_metrics=None,
            args=mask_args,
            p=self.p,
            q=self.q,
            mu=self.mu,
        )
        mask = generator.get_mask()
        if mask is None and os.path.exists(mask_path):
            mask = torch.load(mask_path, map_location=torch.device("cpu"))
        elif mask is not None:
            torch.save(mask, mask_path)
        print(f"Generated finetuning mask saved to {mask_path}")
        return mask

    def init_optimizer(self):
        if self.sophia:
            self.optimizer = create_sophia_optimizer(
                self.model,
                lr=self.lr,
                betas=self.betas,
                rho=self.rho,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

    def init_finetuning_trainer(self, logger):
        root = logger.get_root()
        training_args = self._training_args(
            logger_root=root,
            output_dir=f"{root}/finetuning_checkpoint",
            save_steps=self.max_steps,
            save_total_limit=1,
        )
        trainer_kwargs = dict(
            name=self.finetuning_method,
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.finetuning_dataset,
            eval_dataset=None,
            compute_metrics=None,
            args=training_args,
            data_collator=self.finetuning_collator,
            eval_collector=self.test_collator,
            alpha=self.alpha,
            target_weight=self.target_weight,
            pervasiveness_weight=self.pervasiveness_weight,
            kl_weight=self.kl_weight,
            mask=self.mask,
            if_wanda=self.if_wanda,
        )
        if self.optimizer is not None:
            trainer_kwargs["optimizers"] = (self.optimizer, None)
        self.finetuning_trainer = get_finetuning_method(**trainer_kwargs)
        self._set_active_mask(self.mask, self.mask_name)

    def _build_stage_specs(self):
        stage1_task_specs = [
            TaskTrainingSpec(
                task_key="target",
                display_name="target",
                mask_path=self.mask_path,
                lora_info_dir=self.target_lora_info_dir or self.lora_info_dir,
            )
        ]
        pervasiveness_keys = self._pervasiveness_keys()
        pervasiveness_names = self.dataset_names.get("pervasiveness", [])
        for idx, task_key in enumerate(pervasiveness_keys):
            display_name = (
                pervasiveness_names[idx]
                if idx < len(pervasiveness_names)
                else task_key
            )
            mask_path = self.pervasiveness_mask_paths[idx] if idx < len(self.pervasiveness_mask_paths) else None
            lora_info_dir = (
                self.pervasiveness_lora_info_dirs[idx]
                if idx < len(self.pervasiveness_lora_info_dirs)
                else self.lora_info_dir
            )
            stage1_task_specs.append(
                TaskTrainingSpec(
                    task_key=task_key,
                    display_name=display_name,
                    mask_path=mask_path,
                    lora_info_dir=lora_info_dir,
                )
            )
        stage1 = StageSpec(
            name="stage1",
            num_epochs=self.stage1_num_epochs,
            task_specs=stage1_task_specs,
            mask_path=None,
            lora_info_dir=None,
            aggregate_losses=False,
        )
        stage2 = StageSpec(
            name="stage2",
            num_epochs=self.stage2_num_epochs,
            task_specs=[],
            mask_path=self.all_component_mask_path,
            lora_info_dir=self.all_component_lora_info_dir or self.lora_info_dir,
            aggregate_losses=True,
        )
        return [stage1, stage2]

    def _validate_multitask_paths(self):
        if self.conflict_mask_path:
            _ = self._resolve_mask_path(self.conflict_mask_path)
            if not os.path.exists(_):
                raise FileNotFoundError(f"conflict_mask_path not found: {_}")
        if self.all_component_mask_path:
            resolved = self._resolve_mask_path(self.all_component_mask_path)
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"all_component_mask_path not found: {resolved}")
        for path in self.pervasiveness_mask_paths:
            resolved = self._resolve_mask_path(path)
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"pervasiveness mask path not found: {resolved}")
        if self.conflict_lora_info_dir:
            self._resolve_lora_info_dir(self.conflict_lora_info_dir)
        if self.all_component_lora_info_dir:
            self._resolve_lora_info_dir(self.all_component_lora_info_dir)
        for path in self.pervasiveness_lora_info_dirs:
            self._resolve_lora_info_dir(path)

    def _activate_stage_lora(self, stage_spec: StageSpec):
        if not self._multitask_lora_enabled():
            return
        if stage_spec.aggregate_losses:
            module_names = self._load_lora_module_set(stage_spec.lora_info_dir)
            self._set_active_lora_modules(
                module_names,
                lora_info_dir=stage_spec.lora_info_dir,
                context=f"{stage_spec.name} aggregate",
            )

    def _activate_task_lora(self, stage_spec: StageSpec, task_spec: TaskTrainingSpec):
        if not self._multitask_lora_enabled():
            return
        module_names = self._load_lora_module_set(task_spec.lora_info_dir)
        self._set_active_lora_modules(
            module_names,
            lora_info_dir=task_spec.lora_info_dir,
            context=f"{stage_spec.name} task={task_spec.display_name}",
        )

    def _task_scoped_batch(self, batch, task_key):
        scoped = {"target": None}
        for key in self._pervasiveness_keys():
            scoped[key] = None
        if task_key == "target":
            scoped["target"] = batch.get("target")
            return scoped
        scoped[task_key] = batch.get(task_key)
        return scoped

    def _train_one_stage(self, logger, stage_spec: StageSpec):
        self.model.train()
        if self.optimizer is None:
            self.init_optimizer()
        steps_per_epoch = max(
            1,
            len(self.finetuning_dataset)
            // (self.batch_size * self.gradient_accumulation_steps * self.num_devices),
        )
        dataloader = torch.utils.data.DataLoader(
            self.finetuning_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.finetuning_collator,
        )
        device = self._batch_device()
        self.optimizer.zero_grad()
        print(
            f"[Multitask] Start {stage_spec.name}: epochs={stage_spec.num_epochs}, "
            f"steps_per_epoch={steps_per_epoch}, aggregate_losses={stage_spec.aggregate_losses}"
        )

        stage_mask = None
        if stage_spec.mask_path:
            stage_mask = self._load_mask_from_path(stage_spec.mask_path)
            self._set_active_mask(stage_mask, mask_name=stage_spec.mask_path)
            print(f"[Multitask] {stage_spec.name} using shared mask: {stage_spec.mask_path}")
        self._activate_stage_lora(stage_spec)

        for epoch in range(stage_spec.num_epochs):
            print(f"[Multitask] {stage_spec.name} epoch {epoch + 1}/{stage_spec.num_epochs}")
            epoch_loss = 0.0
            num_batches = 0
            for batch_idx, batch in enumerate(dataloader):
                batch = self._move_to_device(batch, device)

                if stage_spec.aggregate_losses:
                    active_batch = batch
                    task_name = "target+pervasiveness"
                else:
                    task_spec = stage_spec.task_specs[batch_idx % len(stage_spec.task_specs)]
                    if task_spec.mask_path:
                        task_mask = self._load_mask_from_path(task_spec.mask_path)
                        self._set_active_mask(task_mask, mask_name=task_spec.mask_path)
                    else:
                        self._set_active_mask(None, mask_name="none")
                    self._activate_task_lora(stage_spec, task_spec)
                    active_batch = self._task_scoped_batch(batch, task_spec.task_key)
                    task_name = task_spec.display_name
                    if all(value is None for value in active_batch.values()):
                        continue

                amp_context = (
                    torch.cuda.amp.autocast(dtype=torch.bfloat16)
                    if not self.use_cpu and torch.cuda.is_available()
                    else contextlib.nullcontext()
                )
                with amp_context:
                    loss = self.finetuning_trainer.compute_loss(self.model, active_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"[Multitask] {stage_spec.name} skip batch {batch_idx} "
                        f"task={task_name}: invalid loss {loss.item()}"
                    )
                    continue
                (loss / self.gradient_accumulation_steps).backward()
                should_step = (
                    (batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or batch_idx + 1 == len(dataloader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )
                    has_bad_grad = any(
                        parameter.grad is not None
                        and (torch.isnan(parameter.grad).any() or torch.isinf(parameter.grad).any())
                        for parameter in self.model.parameters()
                    )
                    if has_bad_grad:
                        print(
                            f"[Multitask] {stage_spec.name} skip optimizer step at "
                            f"batch {batch_idx}: invalid gradient"
                        )
                        self.optimizer.zero_grad()
                    else:
                        self.finetuning_trainer.mask_gradient(self.model, self.if_wanda)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                epoch_loss += loss.item()
                num_batches += 1
                if batch_idx % 10 == 0:
                    print(
                        f"[Multitask] {stage_spec.name} batch {batch_idx}/{len(dataloader)} "
                        f"task={task_name}, loss={loss.item():.4f}, lora_info_dir={self.active_lora_info_dir}"
                    )
            avg_loss = epoch_loss / num_batches if num_batches else 0.0
            print(f"[Multitask] {stage_spec.name} epoch {epoch + 1} avg_loss={avg_loss:.4f}")
            if not self.probing:
                self.save(logger, tag=f"{stage_spec.name}-epoch-{epoch + 1}")

        # restore stage shared mask (or none) after per-task switching loop
        if stage_spec.aggregate_losses:
            self._set_active_mask(stage_mask, mask_name=stage_spec.mask_path)

        if self._multitask_lora_enabled() and stage_spec.aggregate_losses:
            self._set_active_lora_modules(
                self._load_lora_module_set(stage_spec.lora_info_dir),
                lora_info_dir=stage_spec.lora_info_dir,
                context=f"{stage_spec.name} restore",
            )

    def _run_two_stage_multitask_training(self, logger):
        self._validate_multitask_paths()
        stage_specs = self._build_stage_specs()
        print(
            "[Multitask] Running two-stage schedule with conflict mask recorded only: "
            f"{self.conflict_mask_path}"
        )
        print(
            "[Multitask-LoRA] enabled="
            f"{self._multitask_lora_enabled()}, conflict_lora_info_dir={self.conflict_lora_info_dir}, "
            f"all_component_lora_info_dir={self.all_component_lora_info_dir}"
        )
        for stage_spec in stage_specs:
            self._train_one_stage(logger, stage_spec)
            if not self.probing:
                self._print_epoch_eval(tag=stage_spec.name)

    def _move_to_device(self, value, device):
        if hasattr(value, "to"):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item, device) for item in value]
        if isinstance(value, dict):
            return {key: self._move_to_device(item, device) for key, item in value.items()}
        return value

    def _batch_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _print_mask_freeze_report(self):
        total = sum(parameter.numel() for parameter in self.model.parameters())
        frozen = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if not parameter.requires_grad
        )
        print(f"[Freeze] requires_grad=False: {frozen}/{total}")
        if self.mask is None:
            print("[Freeze] No mask configured; using full-parameter finetuning.")
            return
        print(f"[Freeze] Using {'Wanda' if self.if_wanda else 'named parameter'} mask.")

    def _build_checkpoint_dir(self, root, tag=None):
        checkpoint_root = os.path.join(root, "checkpoints")
        os.makedirs(checkpoint_root, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"-{tag}" if tag else ""
        checkpoint_dir = os.path.join(checkpoint_root, f"checkpoint-{timestamp}{suffix}")
        duplicate_idx = 1
        while os.path.exists(checkpoint_dir):
            checkpoint_dir = os.path.join(
                checkpoint_root, f"checkpoint-{timestamp}{suffix}-{duplicate_idx}"
            )
            duplicate_idx += 1
        return checkpoint_dir

    def _resolve_latest_checkpoint(self, root):
        checkpoint_root = os.path.join(root, "checkpoints")
        if not os.path.isdir(checkpoint_root):
            return checkpoint_root
        checkpoint_dirs = [
            os.path.join(checkpoint_root, name)
            for name in os.listdir(checkpoint_root)
            if name.startswith("checkpoint-")
            and os.path.isdir(os.path.join(checkpoint_root, name))
        ]
        return max(checkpoint_dirs, key=os.path.getmtime) if checkpoint_dirs else checkpoint_root

    def save(self, logger, tag=None):
        checkpoint_dir = self._build_checkpoint_dir(logger.get_root(), tag=tag)
        model_to_save = self.model
        if self.use_lora and tag == "final" and hasattr(self.model, "merge_and_unload"):
            model_to_save = self.model.merge_and_unload()
            self.model = model_to_save
        model_to_save.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        if self.use_lora:
            lora_report_path = os.path.join(checkpoint_dir, "component_lora_report.json")
            with open(lora_report_path, "w") as file:
                json.dump(self.lora_report or {}, file, indent=4)
        self.latest_checkpoint_path = checkpoint_dir
        print(f"Checkpoint saved to: {checkpoint_dir}")

    def save_probing(self, logger):
        output_dir = os.path.join(logger.get_root(), "probingmodel.pt")
        if os.path.isfile(output_dir):
            os.remove(output_dir)
        elif os.path.isdir(output_dir):
            import shutil

            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model
        if self.use_lora and hasattr(self.model, "merge_and_unload"):
            model_to_save = self.model.merge_and_unload()
            self.model = model_to_save
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if self.use_lora:
            lora_report_path = os.path.join(output_dir, "component_lora_report.json")
            with open(lora_report_path, "w") as file:
                json.dump(self.lora_report or {}, file, indent=4)
        probing_summary_path = os.path.join(output_dir, "probing_summary.json")
        with open(probing_summary_path, "w") as file:
            json.dump(
                {
                    "model_name": self.model_name,
                    "probing": True,
                    "target_ratio": self.target_ratio,
                    "num_train_samples": len(self.finetuning_dataset),
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                },
                file,
                indent=4,
            )
        self.latest_checkpoint_path = output_dir
        print(f"[Probing] Checkpoint saved to: {output_dir}")

    def _print_epoch_eval(self, tag):
        target_results = {}
        pervasiveness_results = {}
        if self.target_eval:
            for name, dataset in self.target_test_datasets.items():
                if dataset is not None:
                    target_results[name] = eval_task_accuracy_in_memory(
                        self.model, dataset, batch_size=self.eval_batch_size, tokenizer=self.tokenizer
                    )
        if self.pervasiveness_eval:
            for name, dataset in self.pervasiveness_test_datasets.items():
                if dataset is not None:
                    pervasiveness_results[name] = eval_task_accuracy_in_memory(
                        self.model, dataset, batch_size=self.eval_batch_size, tokenizer=self.tokenizer
                    )
        print(f"[Epoch Eval] tag={tag} target={target_results}")
        print(f"[Epoch Eval] tag={tag} pervasiveness={pervasiveness_results}")

    def _debug_print_target_batch(self, batch, epoch, batch_idx):
        if epoch != 0 or batch_idx >= 3:
            return
        target = batch.get("target")
        if target is None:
            return

        input_ids, attention_mask, labels = target
        sample_idx = 0
        input_ids = input_ids[sample_idx].detach().cpu().tolist()
        attention_mask = attention_mask[sample_idx].detach().cpu().tolist()
        labels = labels[sample_idx].detach().cpu().tolist()

        active_input_ids = [token_id for token_id, mask in zip(input_ids, attention_mask) if mask]
        valid_label_ids = [token_id for token_id in labels if token_id != -100]

        print(f"[Target Debug] epoch={epoch + 1}, batch={batch_idx}, sample={sample_idx}")
        print(f"[Target Debug] input_ids text: {self.tokenizer.decode(input_ids, skip_special_tokens=False)}")
        print(f"[Target Debug] active input text: {self.tokenizer.decode(active_input_ids, skip_special_tokens=False)}")
        print(f"[Target Debug] attention_mask: {attention_mask}")
        print(f"[Target Debug] labels text ignore_-100: {self.tokenizer.decode(valid_label_ids, skip_special_tokens=False)}")
        print("[Target Debug] token table: idx | input_id | input_token | mask | label_id | label_token")
        for idx, (input_id, mask, label_id) in enumerate(zip(input_ids, attention_mask, labels)):
            input_token = self.tokenizer.convert_ids_to_tokens(input_id)
            label_token = "<IGNORE>" if label_id == -100 else self.tokenizer.convert_ids_to_tokens(label_id)
            print(
                f"[Target Debug] {idx:02d} | {input_id:6d} | {input_token!r} | "
                f"{mask} | {label_id:6d} | {label_token!r}"
            )

    def _run_finetuning_training(self, logger):
        if self._multitask_mode_enabled():
            self._run_two_stage_multitask_training(logger)
            print("Finetuning complete")
            return
        self.model.train()
        if self.optimizer is None:
            self.init_optimizer()
        self._print_mask_freeze_report()
        steps_per_epoch = max(
            1,
            len(self.finetuning_dataset)
            // (self.batch_size * self.gradient_accumulation_steps * self.num_devices),
        )
        total_steps = self.num_epochs * steps_per_epoch
        dataloader = torch.utils.data.DataLoader(
            self.finetuning_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.finetuning_collator,
        )
        device = self._batch_device()
        current_step = 0
        self.optimizer.zero_grad()
        print(f"Starting finetuning: epochs={self.num_epochs}, steps_per_epoch={steps_per_epoch}")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            epoch_loss = 0.0
            num_batches = 0
            for batch_idx, batch in enumerate(dataloader):
                if current_step >= total_steps:
                    break
                batch = self._move_to_device(batch, device)
                #self._debug_print_target_batch(batch, epoch, batch_idx)
                amp_context = (
                    torch.cuda.amp.autocast(dtype=torch.bfloat16)
                    if not self.use_cpu and torch.cuda.is_available()
                    else contextlib.nullcontext()
                )
                with amp_context:
                    loss = self.finetuning_trainer.compute_loss(self.model, batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Skipping batch {batch_idx}: invalid loss {loss.item()}")
                    continue
                (loss / self.gradient_accumulation_steps).backward()
                should_step = (
                    (batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or batch_idx + 1 == len(dataloader)
                )
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )
                    has_bad_grad = any(
                        parameter.grad is not None
                        and (torch.isnan(parameter.grad).any() or torch.isinf(parameter.grad).any())
                        for parameter in self.model.parameters()
                    )
                    if has_bad_grad:
                        print(f"Skipping optimizer step at batch {batch_idx}: invalid gradient")
                        self.optimizer.zero_grad()
                    else:
                        self.finetuning_trainer.mask_gradient(self.model, self.if_wanda)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    current_step += 1
                epoch_loss += loss.item()
                num_batches += 1
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(dataloader)}, loss={loss.item():.4f}")
            avg_loss = epoch_loss / num_batches if num_batches else 0.0
            print(f"Epoch {epoch + 1} complete, avg_loss={avg_loss:.4f}")
            if not self.probing:
                self.save(logger, tag=f"epoch-{epoch + 1}")
                self._print_epoch_eval(tag=f"epoch-{epoch + 1}")
        print("Finetuning complete")

    def eval(self, logger):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        root = logger.get_root()
        eval_root = os.path.join(root, "eval")
        os.makedirs(eval_root, exist_ok=True)
        model_name = self.resume_path or self.latest_checkpoint_path or self._resolve_latest_checkpoint(root)
        summary = {"model_name": model_name, "target": {}, "pervasiveness": {}}
        if self.target_eval:
            for name, dataset in self.target_test_datasets.items():
                if dataset is None:
                    continue
                summary["target"][name] = eval_task_accuracy(
                    model_name=model_name,
                    task_dataset=dataset,
                    output_dir=os.path.join(eval_root, "target", name),
                    batch_size=self.eval_batch_size,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if self.pervasiveness_eval:
            for name, dataset in self.pervasiveness_test_datasets.items():
                if dataset is None:
                    continue
                summary["pervasiveness"][name] = eval_task_accuracy(
                    model_name=model_name,
                    task_dataset=dataset,
                    output_dir=os.path.join(eval_root, "pervasiveness", "dataset_accuracy", name),
                    batch_size=self.eval_batch_size,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        summary["pervasiveness"]["lm_eval_enabled"] = self.pervasiveness_lm_eval
        if self.pervasiveness_lm_eval and self.pervasiveness_lm_eval_tasks:
            output_path = os.path.join(
                eval_root, "pervasiveness", "lm_eval", "few_shots.json"
            )
            eval_few_shots(
                model_name=model_name,
                task_list=self.pervasiveness_lm_eval_tasks,
                output_path=output_path,
                batch_size=self.eval_batch_size,
                cache_dir=self.cache_dir,
                device="cuda:0" if torch.cuda.is_available() and not self.use_cpu else "cpu",
            )
            summary["pervasiveness"]["lm_eval_tasks"] = self.pervasiveness_lm_eval_tasks
            summary["pervasiveness"]["lm_eval_output"] = output_path
        with open(os.path.join(eval_root, "finetuning_eval_summary.json"), "w") as file:
            json.dump(summary, file, indent=4)

    def run(self, logger):
        if self.resume_path is None:
            self.init_model()
            self.init_optimizer()
            self.init_dataset()
            self._dump_multitask_config(logger)
            self.init_mask(logger)
            self.init_finetuning_trainer(logger)
            if self.probing:
                self._run_finetuning_training(logger)
                self.save_probing(logger)
                checkpoint_dir = os.path.join(logger.get_root(), "finetuning_checkpoint")
                if os.path.isdir(checkpoint_dir):
                    import shutil

                    shutil.rmtree(checkpoint_dir)
                return
            self.save(logger, tag="init")
            self._print_epoch_eval(tag="init")
            self._run_finetuning_training(logger)
            self.save(logger, tag="final")
            checkpoint_dir = os.path.join(logger.get_root(), "finetuning_checkpoint")
            if os.path.isdir(checkpoint_dir):
                import shutil

                shutil.rmtree(checkpoint_dir)
            self.eval(logger)
        else:
            self.init_model()
            self.init_dataset()
            self._dump_multitask_config(logger)
            self.eval(logger)


def get(**kwargs):
    return Finetuning(**kwargs)