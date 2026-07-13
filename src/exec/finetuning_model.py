import argparse
import os
import random
import sys
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from fastargs.validation import BoolAsInt, Folder, OneOf


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _parse_cuda_version(version):
    if not version:
        return None
    parts = version.split(".")
    try:
        return tuple(int(part) for part in parts[:2])
    except ValueError:
        return None


def _check_runtime_environment(use_cpu=False):
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}; CUDA build: {torch.version.cuda}")
    if use_cpu or not torch.cuda.is_available():
        return

    capability = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA device: {device_name}; capability: sm_{capability[0]}{capability[1]}")

    cuda_version = _parse_cuda_version(torch.version.cuda)
    if capability >= (12, 0) and (cuda_version is None or cuda_version < (12, 8)):
        raise RuntimeError(
            "This run is using a PyTorch CUDA build that is incompatible with Blackwell "
            f"GPUs: torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
            f"device={device_name}, python={sys.executable}. Activate the Blackwell "
            "environment and run with its interpreter, for example: "
            "/home/chenhang/.conda/envs/LLMSFT_BW/bin/python src/exec/finetuning_model.py"
        )


Section("overall", "Overall configs").params(
    model_name=Param(
        str,
        required=True,
        default="mistralai/Mistral-7B-v0.1",
        desc="Model name",
    ),
    logger=Param(OneOf(["json", "none"]), default="json", desc="Logger to use"),
    cache_dir=Param(Folder(True), default="/home/chenhang/CSAT/.cache", desc="Cache directory"),
    seed=Param(int, default=0, desc="Random seed"),
    use_cpu=Param(BoolAsInt(), default=0, desc="Force CPU execution for debugging"),
)


Section("finetuning", "Finetuning configs").params(
    probing=Param(
        BoolAsInt(),
        default=False,
        desc="Run a short probing finetune: 1% train samples, batch size 1, lr=1e-7, one epoch, no evaluation, save HF checkpoint directory probingmodel.pt",
    ),
    finetuning_method=Param(
        OneOf(
            [
                "TargetFT",
                "TargetFT+PervasivenessFT",
                "TargetFT+PervasivenessKL",
                "TargetFT_L1",
            ]
        ),
        default="TargetFT",
        desc="Finetuning objective",
    ),
    mask_path=Param(str, default=None, desc="Path to a single finetuning mask"),#none is without mask
    target_mask_path=Param(
        str,
        #default="/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/IOI_future/component_mask.pt",
        default=None,
        desc="Target-task mask path. If empty, falls back to mask_path for backward compatibility.",

    ),
    pervasiveness_mask_paths=Param(
        str,
        default=None,
        #default="/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/sst2_future/component_mask.pt, /ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/bool_future/component_mask.pt,/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/5_digit_arithmetic_future/component_mask.pt",
        desc="Comma-separated pervasiveness task mask paths. Must match pervasiveness dataset count when provided.",
    ),
    conflict_mask_path=Param(
        str,
        default=None,
        #default="/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/conflict_analysis_future/conflict_components/component_mask.pt",
        desc="Conflict component mask path (recorded in Phase 6; not used by training loop yet).",
    ),
    all_component_mask_path=Param(
        str,
        default=None,
        #default="/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/conflict_analysis_future/all_task_components/component_mask.pt",
        desc="All-task component mask path for stage-2 multitask training (recorded in Phase 6).",
    ),
    mask_score_type=Param(str, default="gradient", desc="Mask score type"),
    mask_ratio=Param(float, default=0.2, desc="Mask ratio to load or generate"),
    mask_calibration_samples=Param(
        float,
        default=128,
        desc="Target samples used for mask generation; set 0 to use all training data",
    ),
    num_epochs=Param(int, default=3, desc="Number of epochs to train"),
    lr=Param(float, default=1e-5, desc="Learning rate"),
    weight_decay=Param(float, default=0.1, desc="Weight decay"),
    gradient_accumulation_steps=Param(int, default=1, desc="Gradient accumulation steps"),
    max_grad_norm=Param(float, default=1.0, desc="Maximum gradient norm for clipping"),
    sophia=Param(BoolAsInt(), default=False, desc="Whether to use SOPHIA"),
    p=Param(float, default=0.01, desc="p for snip_joint mask generation"),
    q=Param(float, default=0.01, desc="q for snip_joint mask generation"),
    resume_path=Param(Folder(False), default=None, desc="Path to trained model for evaluation"),
    max_steps=Param(int, default=-1, desc="Max steps for training"),
    use_lora=Param(BoolAsInt(), default=False, desc="Whether to use LoRA"),
    lora_mode=Param(
        OneOf(["standard", "projection_matrix", "head"]),
        default="head",
        desc="LoRA mode: fixed PEFT LoRA, EAP projection rank_pattern, or custom head-wise LoRA",
    ),
    lora_info_dir=Param(
        str,
        #default="/home/chenhang/CSAT/files/masks/Future/2_digit_arithmetic",
        default=None,
        desc="Directory produced by EAP_forComponent containing rank_pattern.json/component_scores.json/summary.json",
    ),
    target_lora_info_dir=Param(
        str,
        default=None,
        #default="/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/IOI_future",
        desc="Target-task LoRA info dir. If empty, falls back to lora_info_dir for backward compatibility.",
    ),
    pervasiveness_lora_info_dirs=Param(
        str,
        default=None,
        #default="/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/sst2_future, /ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/bool_future,/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/5_digit_arithmetic_future",
        desc="Comma-separated pervasiveness task LoRA info dirs. Must match pervasiveness dataset count when provided.",
    ),
    conflict_lora_info_dir=Param(
        str,
        default=None,
        #default="/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/conflict_analysis_future/conflict_components",
        desc="Conflict component LoRA info dir (recorded in Phase 6).",
    ),
    all_component_lora_info_dir=Param(
        str,
        default=None,
        #default="/ssd_users/chenhang/CSAT/files/logical_circuit/bool+IOI+sst2+arithmetic/conflict_analysis_future/all_task_components",
        desc="All-task component LoRA info dir for stage-2 training (recorded in Phase 6).",
    ),
    multi_task_schedule=Param(
        OneOf(["two_stage_alternating"]),
        default="two_stage_alternating",
        desc="Multitask schedule name for future two-stage training.",
    ),
    stage1_num_epochs=Param(
        int,
        default=1,
        desc="Stage-1 epoch count for multitask schedule.",
    ),
    stage2_num_epochs=Param(
        int,
        default=1,
        desc="Stage-2 epoch count for multitask schedule.",
    ),
    lora_target_modules=Param(
        str,
        default="auto",
        desc="Comma-separated LoRA target modules; auto uses q/v for standard, EAP outputs for projection/head modes",
    ),
    lora_default_rank=Param(int, default=8, desc="Default LoRA rank when no per-module rank pattern is used"),
    lora_alpha=Param(int, default=32, desc="LoRA alpha value"),
    lora_dropout=Param(float, default=0.05, desc="LoRA dropout"),
    lora_alpha_strategy=Param(
        OneOf(["constant", "twice_rank"]),
        default="twice_rank",
        desc="Use a constant alpha or set per-component alpha to twice its rank",
    ),
    lora_rank_pattern_path=Param(
        str,
        default=None,
        desc="Optional explicit PEFT rank_pattern.json path; defaults to lora_info_dir/rank_pattern.json",
    ),
    lora_alpha_pattern_path=Param(
        str,
        default=None,
        desc="Optional explicit PEFT alpha_pattern.json path; defaults to lora_info_dir/alpha_pattern.json if present",
    ),
    lora_component_scores_path=Param(
        str,
        default=None,
        desc="Optional explicit component_scores.json path for head-wise LoRA; defaults to lora_info_dir/component_scores.json",
    ),
    lora_head_min_rank=Param(int, default=1, desc="Minimum rank for custom head-wise/component-wise LoRA"),
    lora_head_max_rank=Param(int, default=32, desc="Maximum rank for custom head-wise/component-wise LoRA"),
    lora_head_rank_multiple=Param(int, default=1, desc="Round custom head-wise LoRA ranks to this multiple"),
    lora_head_rank_score_source=Param(
        OneOf(["rank_score", "normalized_abs", "raw_abs", "sum_abs", "mean_abs", "sqrt_numel_abs"]),
        default="rank_score",
        desc="Component score field used to allocate custom head-wise LoRA ranks",
    ),
    mu=Param(float, default=1e-6, desc="Hessian approximation parameter"),
    target_weight=Param(float, default=1.0, desc="Weight for target supervised loss"),
    pervasiveness_weight=Param(float, default=1.0, desc="Weight for pervasiveness supervised loss"),
    kl_weight=Param(float, default=1.0, desc="Weight for pervasiveness KL loss"),
    alpha=Param(float, default=0.0, desc="L1 regularization strength"),
)


Section("finetuning.sophia_params", "SOPHIA configs").enable_if(
    lambda cfg: cfg["finetuning.sophia"]
).params(
    betas_low=Param(float, default=0.9, desc="Betas lower for SOPHIA"),
    betas_high=Param(float, default=0.95, desc="Betas higher for SOPHIA"),
    rho=Param(float, default=0.03, desc="Rho for SOPHIA"),
)


Section("dataset", "Dataset configs").params(
    target_dataset_name=Param(str, default="winogrande", desc="Target finetuning dataset name"),#IOI, gender, bool，sst2, winogrande, docstring，induction, 1_digit_arithmetic-5_digit_arithmetic#
    pervasiveness_dataset_name=Param(
        str,
        default="IOI",
        desc="Comma-separated pervasiveness dataset names",
    ),
    dataset_seed=Param(int, default=1000, desc="Dataset seed"),
    target_ratio=Param(float, default=1.0, desc="Target dataset ratio or sample count"),
    target_holdout_as_pervasiveness=Param(
        BoolAsInt(),
        default=False,
        desc="Use held-out target samples as an extra pervasiveness dataset",
    ),
    batch_size=Param(int, default=32, desc="Batch size"),
)


Section("evaluation", "Evaluation configs").params(
    target_eval=Param(BoolAsInt(), default=True, desc="Evaluate target task accuracy"),
    pervasiveness_eval=Param(
        BoolAsInt(), default=True, desc="Evaluate pervasiveness task accuracy"
    ),
    pervasiveness_lm_eval=Param(
        BoolAsInt(), default=False, desc="Run lm-evaluation-harness final pervasiveness evaluation"
    ),
    pervasiveness_lm_eval_tasks=Param(
        str,
        default="boolq,rte,hellaswag,winogrande,arc_challenge,arc_easy,openbookqa,piqa,truthfulqa",
        desc="Comma-separated lm-evaluation-harness task names for final catastrophic-forgetting evaluation",
    ),
    eval_batch_size=Param(int, default=8, desc="Evaluation batch size"),
)


Section("logger", "General logger configs").params(
    name=Param(
        str,
        default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
        desc="Name of this run",
    ),
)


Section("logger.json", "JSON logger").enable_if(
    lambda cfg: cfg["overall.logger"] == "json"
).params(
    root=Param(Folder(True), default="/home/chenhang/CSAT/files/logs", desc="Path to log folder"),
)


class Main:
    def __init__(self) -> None:
        self.make_config()
        overall = self.config.get_section("overall")
        if overall.get("use_cpu"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _check_runtime_environment(use_cpu=overall.get("use_cpu"))
        self.setup_seed()
        self.init_model()
        self.init_logger()
        self.run()

    def make_config(self, quiet=False):
        self.config = get_current_config()
        parser = argparse.ArgumentParser("LLM finetuning")
        self.config.augment_argparse(parser)
        self.config.collect_argparse_args(parser)
        self.config.validate()
        if not quiet:
            self.config.summary()

    @param("overall.seed")
    def setup_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @param("overall.model_name")
    def init_model(self, model_name):
        kwargs = self.config.get_section("overall")
        kwargs.update(self.config.get_section("finetuning"))
        kwargs.update(self.config.get_section("dataset"))
        kwargs.update(self.config.get_section("evaluation"))
        if kwargs["sophia"]:
            kwargs.update(self.config.get_section("finetuning.sophia_params"))

        pervasiveness_name = kwargs["pervasiveness_dataset_name"]
        pervasiveness_datasets = [
            name.strip() for name in pervasiveness_name.split(",") if name.strip()
        ]
        kwargs["dataset_names"] = {
            "target": kwargs["target_dataset_name"],
            "pervasiveness": pervasiveness_datasets,
        }

        self.model = import_module("model.finetuning").get(**kwargs)

    @param("overall.logger")
    def init_logger(self, logger):
        kwargs = self.config.get_section("logger")
        kwargs.update(self.config.get_section(f"logger.{logger}"))
        kwargs["config"] = self.config.get_all_config()
        self.logger = import_module(f"loggers.{logger}_").get(**kwargs)

    def run(self):
        self.model.run(self.logger)


if __name__ == "__main__":
    Main()