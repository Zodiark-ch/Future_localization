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
from fastargs.validation import BoolAsInt, File, Folder, OneOf
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
Section("overall", "Overall configs").params(
    model_name=Param(str, required=True, default="mistralai/Mistral-7B-v0.1", desc="Model name"),
    logger=Param(OneOf(["json", "none"]), default="json", desc="Logger to use"),
    cache_dir=Param(Folder(True), default="/home/chenhang/CSAT/.cache", desc="Cache directory"),
    seed=Param(int, default=0, desc="Random seed"),
    use_cpu=Param(
        BoolAsInt(),
        default=0,
        desc="强制在 CPU 上加载模型并训练/评估（极慢，仅调试用）；为 1 时会清空 CUDA_VISIBLE_DEVICES",
    ),
)


Section("unlearn", "Unlearning configs").params(

    safe_unlearn_method=Param(
        OneOf(
            [
                "FT",
                "l1_sparse",
                "GA",
                "GA+FT",
                "origin",
                "CL",
                "RL",
                "KL",
                "CL+FT",
                "GA+KL",
                "CL+KL",
                "NPO",
                "NPO+FT"
            ]
        ),
        default="FT",
        desc="Safe unlearning method",
    ),
    conflict_unlearn_method=Param(
        OneOf(
            [
                "FT",
                "l1_sparse",
                "GA",
                "GA+FT",
                "origin",
                "CL",
                "RL",
                "KL",
                "CL+FT",
                "GA+KL",
                "CL+KL",
                "NPO+FT"
            ]
        ),
        default="FT",
        desc="Conflict unlearning method",
    ),
    safe_mask_path=Param(str, default="/home/chenhang/CSAT/wanda/zephyr/with_0.0.pt", desc="Path to safe mask file"),
    #/home/chenhang/CSAT/conflict/safe_mask.pt     /home/chenhang/CSAT/conflict/conflict_mask.pt
    conflict_mask_path=Param(str, default="/home/chenhang/CSAT/wanda/zephyr/with_0.0.pt", desc="Path to conflict mask file"),
    #/home/chenhang/CSAT/wanda/zephyr/with_0.9.pt
    alternate_frequency=Param(int, default=1, desc="Alternate frequency (epochs per switch)"),

    num_epochs=Param(int, default=6, desc="Number of epochs to train"),
    lr=Param(float, default=1e-05, desc="Learning rate"),
    weight_decay=Param(float, default=0.1, desc="Weight decay"),
    gradient_accumulation_steps=Param(
        int, default=4, desc="Gradient accumulation steps"
    ),
    max_grad_norm=Param(float, default=1.0, desc="Maximum gradient norm for clipping"),
    task_name=Param(
        OneOf(["toxic", "copyright", "tofu", "wmdp","downstream"]),
        default="downstream",
        desc="Task name",
    ),
    sophia=Param(BoolAsInt(), default=False, desc="Whether to use SOPHIA"),
    p=Param(float, default=0.01, desc="p for snip_joint"),
    q=Param(float, default=0.01, desc="q for snip_joint"),
    resume_path=Param(
        Folder(False), default=None, desc="Path to resume model for evaluation"
    ),
    max_steps=Param(int, default=-1, desc="Max steps for training"),
    use_lora=Param(BoolAsInt(), default=False, desc="Whether to use LoRA"),
    mu=Param(float, default=1e-06, desc="hessian approximantion parameter"),
)

Section("unlearn.sophia_params", "SOPHIA configs").enable_if(
    lambda cfg: cfg["unlearn.sophia"]
).params(
    betas_low=Param(float, default=0.9, desc="Betas lower for SOPHIA"),
    betas_high=Param(float, default=0.95, desc="Betas higher for SOPHIA"),
    rho=Param(float, default=0.03, desc="Rho for SOPHIA"),
)
Section("unlearn.NPO+FT", "NPO+FT unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.safe_unlearn_method"] == "NPO+FT" or cfg["unlearn.conflict_unlearn_method"] == "NPO+FT"
).params(
    gamma=Param(float, default=1.0, desc="hyperparameters before NPO loss"),
)

Section("unlearn.l1_sparse", "L1 sparse unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.safe_unlearn_method"] == "l1_sparse" or cfg["unlearn.conflict_unlearn_method"] == "l1_sparse"
).params(
    alpha=Param(float, default=0.0, desc="L1 regularization parameter"),
)

Section("unlearn.CL+KL", "CL+KL unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.safe_unlearn_method"] == "CL+KL" or cfg["unlearn.conflict_unlearn_method"] == "CL+KL"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before CL loss"),
)

Section("unlearn.GA+KL", "GA+KL unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.safe_unlearn_method"] == "GA+KL" or cfg["unlearn.conflict_unlearn_method"] == "GA+KL"
).params(
    gamma=Param(float, default=1.0, desc="hyperparameters before GA loss"),
)

Section("unlearn.CL+FT", "CL+FT unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.safe_unlearn_method"] == "CL+FT" or cfg["unlearn.conflict_unlearn_method"] == "CL+FT"
).params(
    gamma=Param(float, default=1.0, desc="hyperparameters before CL loss"),
)
Section("unlearn.GA+FT", "GA+FT unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.safe_unlearn_method"] == "GA+FT" or cfg["unlearn.conflict_unlearn_method"] == "GA+FT"
).params(
    gamma=Param(float, default=1.0, desc="hyperparameters before GA loss"),
)

Section("unlearn.KL", "KL unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.safe_unlearn_method"] == "KL" or cfg["unlearn.conflict_unlearn_method"] == "KL"
).params(
    gamma=Param(
        float, default=0.0, desc="hyperparameters before KL loss on forget dataset"
    ),
)

Section("dataset", "Dataset configs").params(
    forget_dataset_name=Param(str, default="WMDPCyber", desc="forget dataset name,SafePku,WMDPCyber,WMDPBio"),
    retain_dataset_name=Param(str, default="IOI,gender", desc="retain dataset name"),#bool,sst2,winogrande,IOI,gender,induction
    dataset_seed=Param(int, default=1000, desc="Dataset seed"),
    forget_ratio=Param(float, default=400, desc="Forget ratio"),
    self_retain=Param(BoolAsInt(), default=False, desc="Whether to retain self"),
    batch_size=Param(int, default=1, desc="Batch size"),
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
        _overall = self.config.get_section("overall")
        if _overall.get("use_cpu"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.setup_seed()
        self.init_model()
        self.init_logger()
        self.run()

    def make_config(self, quiet=False):
        self.config = get_current_config()
        parser = argparse.ArgumentParser("LLM unlearning")
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
        kwargs = self.config.get_section(f"overall")
        kwargs.update(self.config.get_section(f"unlearn"))
        kwargs.update(self.config.get_section(f"dataset"))


        kwargs.update(self.config.get_section(f"unlearn.{kwargs['safe_unlearn_method']}"))
        kwargs.update(self.config.get_section(f"unlearn.{kwargs['conflict_unlearn_method']}"))

        if kwargs["sophia"]:
            kwargs.update(self.config.get_section(f"unlearn.sophia_params"))


        retain_dataset_name = kwargs["retain_dataset_name"]
        if "," in retain_dataset_name:

            retain_datasets = [ds.strip() for ds in retain_dataset_name.split(",")]
            kwargs["dataset_names"] = {
                "forget": kwargs["forget_dataset_name"],
                "retain": retain_datasets,
            }
        else:

            kwargs["dataset_names"] = {
                "forget": kwargs["forget_dataset_name"],
                "retain": kwargs["retain_dataset_name"],
            }


        self.model = import_module(f"model.unlearn_conflict").get(**kwargs)

    @param("overall.logger")
    def init_logger(self, logger):
        kwargs = self.config.get_section(f"logger")
        kwargs.update(self.config.get_section(f"logger.{logger}"))
        kwargs["config"] = self.config.get_all_config()
        self.logger = import_module(f"loggers.{logger}_").get(**kwargs)

    def run(self):
        self.model.run(self.logger)


if __name__ == "__main__":
    Main()
