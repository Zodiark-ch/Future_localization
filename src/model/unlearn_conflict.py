import contextlib
import os
import sys
from datetime import datetime

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from peft import  get_peft_model, LoraConfig
from pruner.utils import WrappedGPT, find_layers
from dataset import get_dataset
from metrics import (
    eval_copyright,
    eval_few_shots,
    eval_PII,
    eval_ppl,
    eval_tofu,
    eval_toxic,
    eval_wmdp,
)
from metrics.simple_accuracy import eval_acc
from optim import create_sophia_optimizer
from unlearn import GenerateMask, get_unlearn_method

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir


        self.safe_unlearn_method = kwargs["safe_unlearn_method"]
        self.conflict_unlearn_method = kwargs["conflict_unlearn_method"]
        self.safe_mask_path = kwargs["safe_mask_path"]
        self.conflict_mask_path = kwargs["conflict_mask_path"]
        self.alternate_frequency = kwargs.get("alternate_frequency", 1)

        self.batch_size = kwargs["batch_size"]
        self.dataset_names = kwargs["dataset_names"]
        self.dataset_seed = kwargs["dataset_seed"]
        self.forget_ratio = kwargs["forget_ratio"]
        self.self_retain = kwargs["self_retain"]
        self.num_epochs = kwargs["num_epochs"]
        self.num_devices = int(os.environ.get("WORLD_SIZE", 1))
        self.lr = kwargs["lr"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.weight_decay = kwargs["weight_decay"]
        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
        self.alpha = kwargs.get("alpha", None)
        self.gamma = kwargs.get("gamma", None)
        self.task_name = kwargs.get("task_name", None)
        self.k = kwargs.get("k", 100)
        self.sophia = kwargs.get("sophia", False)
        self.betas_low = kwargs.get("betas_low", 0.9)
        self.betas_high = kwargs.get("betas_high", 0.95)
        self.betas = (self.betas_low, self.betas_high)
        self.rho = kwargs.get("rho", 0.03)
        self.p = kwargs.get("p", 0.0)
        self.q = kwargs.get("q", 0.0)
        self.if_llama = "llama" in self.model_name
        self.resume_path = kwargs.get("resume_path", None)
        self.max_steps = kwargs.get("max_steps", -1)
        self.use_lora = kwargs.get("use_lora", False)
        self.if_wanda = False
        self.mu = kwargs.get("mu", 1e-3)
        self.latest_checkpoint_path = None
        self.use_cpu = bool(kwargs.get("use_cpu", False)) or os.environ.get(
            "CSAT_FORCE_CPU", ""
        ).lower() in ("1", "true", "yes")

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

    def _move_mask_to_device(self, mask, if_wanda, mask_name):

        if mask is None:
            return


        print(f"处理{mask_name} mask，使用数字索引作为键")
        try:
            layers = self.model.model.layers
        except:
            layers = self.model.model.decoder.layers
        cnt = 0
        with torch.no_grad():
            for layer in layers:
                subset = find_layers(layer)
                for name in subset:
                    if cnt in mask:
                        mask[cnt] = mask[cnt].to(subset[name].weight.device)
                    cnt += 1

    def _count_frozen_by_requires_grad(self, model):
        total = 0
        frozen = 0
        for p in model.parameters():
            n = p.numel()
            total += n
            if not p.requires_grad:
                frozen += n
        return frozen, total

    def _count_frozen_weight_scalars_wanda_mask(self, model, mask):

        try:
            layers = model.model.layers
        except AttributeError:
            layers = model.model.decoder.layers
        cnt = 0
        frozen = 0
        total = 0
        for layer in layers:
            subset = find_layers(layer)
            for _name in subset:
                w = subset[_name].weight
                n = w.numel()
                total += n
                key = cnt
                if key not in mask and str(key) in mask:
                    key = str(key)
                if mask is not None and key in mask:
                    m = mask[key]
                    if m.shape != w.shape:
                        cnt += 1
                        continue
                    if m.dtype == torch.bool:
                        frozen += int((~m).sum().item())
                    else:
                        frozen += int((m == 0).sum().item())
                cnt += 1
        return frozen, total

    def _count_frozen_scalars_named_param_mask(self, model, mask):

        frozen = 0
        total = 0
        for name, p in model.named_parameters():
            if name not in mask:
                continue
            m = mask[name]
            if not isinstance(m, torch.Tensor) or m.shape != p.shape:
                continue
            n = p.numel()
            total += n
            if m.dtype == torch.bool:
                frozen += int((~m).sum().item())
            else:
                frozen += int((m == 0).sum().item())
        return frozen, total

    def _print_parameter_freeze_report(self, unlearner):

        model = self.model
        fr, tot = self._count_frozen_by_requires_grad(model)
        pct = 100.0 * fr / tot if tot else 0.0
        print(
            f"[参数冻结] requires_grad=False: {fr}/{tot} 标量 "
            f"({pct:.4f}%)，可训练标量: {tot - fr}/{tot} ({100.0 - pct:.4f}%)。"
        )

        mask = getattr(unlearner, "mask", None)
        if_wanda = getattr(unlearner, "if_wanda", False)
        if mask is None:
            print("[参数冻结] 当前 unlearner 未设置 mask，无 mask 粒度统计。")
            return

        if if_wanda:
            mf, mt = self._count_frozen_weight_scalars_wanda_mask(model, mask)
            mpct = 100.0 * mf / mt if mt else 0.0
            print(
                f"[参数冻结] Wanda mask（与 mask_gradient wanda 分支一致）: "
                f"参与 mask 的线性层权重中 mask==0 的标量 {mf}/{mt} ({mpct:.4f}%)，"
                f"mask==1（可更新梯度） {mt - mf}/{mt} ({100.0 - mpct:.4f}%)。"
            )
        else:
            mf, mt = self._count_frozen_scalars_named_param_mask(model, mask)
            mpct = 100.0 * mf / mt if mt else 0.0
            print(
                f"[参数冻结] 按参数名 mask（与 mask_gradient 非-wanda 分支一致）: "
                f"mask==0 的标量 {mf}/{mt} ({mpct:.4f}%)，"
                f"mask==1 {mt - mf}/{mt} ({100.0 - mpct:.4f}%)。"
            )
        print(
            "[参数冻结] 训练时会在每次 optimizer.step 前对应当前 unlearner 调用 "
            "BaseTrainer.mask_gradient，使 mask==0 处梯度为 0。"
        )

    def init_model(self):
        load_kw = dict(
            pretrained_model_name_or_path=self.model_name,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
        )
        if self.use_cpu:
            load_kw["torch_dtype"] = torch.float32
            load_kw["device_map"] = "cpu"
            print("[use_cpu] 在 CPU 上加载模型（float32，极慢，仅调试用）")
        else:
            load_kw["torch_dtype"] = torch.bfloat16
            load_kw["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(**load_kw)
        if self.use_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj","v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            print(model.print_trainable_parameters())

        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            if self.if_llama:
                tokenizer.add_special_tokens({"pad_token": "[pad]"})

            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs for DataParallel!")
        #     self.model = torch.nn.DataParallel(self.model)
        # try:
        #     self.device = torch.device("cuda:0")
        #     #self.device = model.hf_device_map["lm_head"]
        # except:
        #     self.device = torch.device("cuda:0")
        # self.model = self.model.to(self.device)

    def init_dataset(self):
        unlearn_dataset, test_datasets, unlearn_collator, test_collator, downstream_datasets = get_dataset(
            self.dataset_names,
            self.tokenizer,
            self.dataset_seed,
            self.forget_ratio,
            self.self_retain,
            self.if_llama,
        )
        self.unlearn_dataset = unlearn_dataset
        self.test_datasets = test_datasets
        self.downstream_datasets = downstream_datasets
        self.unlearn_collator = unlearn_collator
        self.test_collator = test_collator
        if self.max_steps == -1:
            self.max_steps = int(self.num_epochs * len(unlearn_dataset)) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
            self.steps_per_epoch = len(unlearn_dataset) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
        else:
            self.steps_per_epoch = self.max_steps // self.num_epochs

    def init_unlearner(self, logger):
        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"


        self._init_conflict_unlearners(logger)

    def _init_conflict_unlearners(self, logger):

        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"

        training_args = self._training_args(
            logger_root=root,
            output_dir=unlearn_checkpoint,
            save_steps=self.max_steps,
            save_total_limit=1,
        )


        if self.optimizer is not None:
            self.safe_unlearner = get_unlearn_method(
                name=self.safe_unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.safe_mask,
                optimizers=(self.optimizer, None),
                if_wanda=True,
            )
        else:
            self.safe_unlearner = get_unlearn_method(
                name=self.safe_unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.safe_mask,
                if_wanda=True,
            )


        if self.optimizer is not None:
            self.conflict_unlearner = get_unlearn_method(
                name=self.conflict_unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.conflict_mask,
                optimizers=(self.optimizer, None),
                if_wanda=True,
            )
        else:
            self.conflict_unlearner = get_unlearn_method(
                name=self.conflict_unlearn_method,
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=training_args,
                data_collator=self.unlearn_collator,
                eval_collector=self.test_collator,
                alpha=self.alpha,
                gamma=self.gamma,
                mask=self.conflict_mask,
                if_wanda=True,
            )

    def init_mask(self, logger):

        self.safe_mask = None
        self.conflict_mask = None


        if self.safe_mask_path is not None and os.path.exists(self.safe_mask_path):
            print(f"加载safe mask: {self.safe_mask_path}")
            self.safe_mask = torch.load(
                self.safe_mask_path, map_location=torch.device("cpu")
            )
            self._move_mask_to_device(self.safe_mask, None, "safe")


        if self.conflict_mask_path is not None and os.path.exists(self.conflict_mask_path):
            print(f"加载conflict mask: {self.conflict_mask_path}")
            self.conflict_mask = torch.load(
                self.conflict_mask_path, map_location=torch.device("cpu")
            )
            self._move_mask_to_device(self.conflict_mask, None, "conflict")


        if self.safe_mask is not None:
            self.mask = self.safe_mask
        else:
            self.mask = None


        if self.safe_mask_path is not None and not os.path.exists(self.safe_mask_path):
            self._generate_mask(self.safe_mask_path, logger, "safe")


        if self.conflict_mask_path is not None and not os.path.exists(self.conflict_mask_path):
            self._generate_mask(self.conflict_mask_path, logger, "conflict")

    def _generate_mask(self, mask_path, logger, mask_name):

        parts = mask_path.split("/")
        score_type = parts[-2]
        if score_type == "wanda":
            if_wanda = True
        else:
            if_wanda = False

        ratio = float(parts[-1].split("_")[-1].split(".p")[0])
        root = logger.get_root()
        mask_dir = mask_path.replace(f"with_{ratio}.pt", "")
        if mask_dir == mask_path:
            mask_dir = mask_path.replace(f"with_{self.p}_{self.q}.pt", "")
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        mask_args = self._training_args(
            logger_root=root,
            output_dir=mask_dir,
            save_steps=self.steps_per_epoch,
            save_total_limit=3,
        )
        if score_type == "wanda":
            unlearn_dataset,_,_,_,_ = get_dataset(
                self.dataset_names,
                self.tokenizer,
                self.dataset_seed,
                128,
                self.self_retain,
                self.if_llama,
            )
            mask = GenerateMask(
                score_type=score_type,
                ratios=[ratio],
                mask_dir=mask_dir,
                model=self.model,
                data_collator=self.unlearn_collator,
                tokenizer=self.tokenizer,
                train_dataset=unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=mask_args,
                p=self.p,
                q=self.q,
                mu=self.mu,
            ).get_mask()
        else:
            mask = GenerateMask(
                score_type=score_type,
                ratios=[ratio],
                mask_dir=mask_dir,
                model=self.model,
                data_collator=self.unlearn_collator,
                tokenizer=self.tokenizer,
                train_dataset=self.unlearn_dataset,
                eval_dataset=None,
                compute_metrics=None,
                args=mask_args,
                p=self.p,
                q=self.q,
                mu=self.mu,
            ).get_mask()
        if score_type == "snip_forget_reinit":
            mask = None
            os.system(f"rm -rf {mask_path}")
            return


        torch.save(mask, mask_path)
        print(f"Generated {mask_name} mask saved to {mask_path}")
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

            from torch.optim import AdamW
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

    def eval(self, logger):
        self.model = None
        torch.cuda.empty_cache()
        root = logger.get_root()


        import os
        os.makedirs(root, exist_ok=True)

        if self.resume_path is not None:
            model_name = self.resume_path
        else:

            if self.latest_checkpoint_path is not None:
                model_name = self.latest_checkpoint_path
            else:
                model_name = self._resolve_latest_checkpoint(root)
        if self.task_name == "downstream":
            if "WMDP" in self.dataset_names["forget"]:
                print("开始评估wmdp数据集...")
                eval_few_shots(model_name=model_name, task_list=["wmdp"], output_path=f"{root}/wmdp.json")
                torch.cuda.empty_cache()
                eval_few_shots(model_name=model_name,  task_list=["mmlu"],output_path=f"{root}/mmlu.json")
            torch.cuda.empty_cache()
            if self.dataset_names["forget"] == "SafePku":
                print("开始评估DETOX数据集...")
                eval_toxic(
                    model_name=model_name, output_dir=root, dataset=self.unlearn_dataset
                )
            torch.cuda.empty_cache()

            if isinstance(self.test_datasets, dict) and len(self.test_datasets) > 1:

                data_num=0
                for test_key, test_dataset in self.test_datasets.items():
                    if test_dataset is not None:
                        print(f"评估数据集: {self.dataset_names['retain'][data_num]}")
                        data_num+=1

                        test_output_dir = f"{root}/{test_key}"
                        os.makedirs(test_output_dir, exist_ok=True)
                        eval_acc(model_name=model_name, retain_dataset=test_dataset, output_dir=test_output_dir, batch_size=8)
                        torch.cuda.empty_cache()
            else:

                if isinstance(self.test_datasets, dict):

                    test_dataset = self.test_datasets.get("test")
                else:

                    test_dataset = self.test_datasets

                if test_dataset is not None:
                    eval_acc(model_name=model_name, retain_dataset=test_dataset, output_dir=root, batch_size=8)
                    torch.cuda.empty_cache()


            if self.downstream_datasets and len(self.downstream_datasets) > 0:
                print("开始评估downstream数据集...")
                for downstream_key, downstream_dataset in self.downstream_datasets.items():
                    if downstream_dataset is not None:

                        dataset_name = downstream_key.replace("downstream_", "")
                        print(f"评估downstream数据集: {dataset_name}")


                        downstream_output_dir = f"{root}/downstream_{dataset_name}"
                        os.makedirs(downstream_output_dir, exist_ok=True)

                        eval_acc(model_name=model_name, retain_dataset=downstream_dataset, output_dir=downstream_output_dir, batch_size=8)
                        torch.cuda.empty_cache()

            #eval_ppl(model_name=model_name, output_path=f"{root}/ppl.json")

            eval_few_shots(model_name=model_name, output_path=f"{root}/few_shots.json")
            torch.cuda.empty_cache()

    def eval_accuracy(self, model_name, output_dir=".", batch_size=8):

        accuracies = {}


        if isinstance(self.test_datasets, dict):

            for test_key, test_dataset in self.test_datasets.items():
                if test_dataset is not None:
                    print(f"评估数据集: {test_key}")
                    accuracy = eval_acc(
                        model_name=model_name,
                        retain_dataset=test_dataset,
                        output_dir=f"{output_dir}/{test_key}",
                        batch_size=batch_size
                    )
                    accuracies[test_key] = accuracy
        else:

            accuracy = eval_acc(
                model_name=model_name,
                retain_dataset=self.test_datasets,
                output_dir=output_dir,
                batch_size=batch_size
            )
            accuracies["test"] = accuracy

        return accuracies

    def _eval_acc_in_memory(self, retain_dataset, batch_size=8):

        from torch.utils.data import DataLoader
        import tqdm


        if (
            hasattr(retain_dataset, "retain_dataset")
            and retain_dataset.retain_dataset is not None
        ):
            print("检测到UnlearnDataset，使用其retain_dataset进行评估")
            actual_dataset = retain_dataset.retain_dataset
        else:
            print("使用普通数据集进行评估")
            actual_dataset = retain_dataset

        def collate_fn(batch):
            return {
                "input_ids": torch.stack([item["input_ids"] for item in batch]),
                "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
                "label": torch.stack([item["label"] for item in batch]),
            }

        dataloader = DataLoader(
            actual_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = self.model
        was_training = model.training
        model.eval()


        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        correct_predictions = 0
        total_predictions = 0

        print("开始评估准确率（in-memory）...")
        _dbg_decode_once = True
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="评估进度"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                print("model type:", type(model))
                print("config:", model.config.model_type, getattr(model.config, "architectures", None))
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                logits = outputs.logits
                predicted_tokens = torch.argmax(logits, dim=-1)

                valid_mask = (labels != -100) & (attention_mask == 1)

                batch_correct = 0
                batch_total = 0
                for i in range(len(labels)):
                    valid_positions = valid_mask[i].nonzero(as_tuple=True)[0]
                    if len(valid_positions) > 0:
                        last_valid_pos = valid_positions[-1]
                        if predicted_tokens[i, last_valid_pos] == labels[i, last_valid_pos]:
                            batch_correct += 1
                        batch_total += 1

                correct_predictions += batch_correct
                total_predictions += batch_total

        accuracy = (
            (correct_predictions / total_predictions * 100)
            if total_predictions > 0
            else 0
        )

        if was_training:
            model.train()

        return accuracy

    def eval_accuracy_in_memory(self, output_dir=".", batch_size=8):

        accuracies = {}

        if isinstance(self.test_datasets, dict):
            for test_key, test_dataset in self.test_datasets.items():
                if test_dataset is None:
                    continue
                print(f"评估数据集: {test_key}（in-memory）")
                accuracies[test_key] = self._eval_acc_in_memory(
                    retain_dataset=test_dataset, batch_size=batch_size
                )
        else:
            accuracies["test"] = self._eval_acc_in_memory(
                retain_dataset=self.test_datasets, batch_size=batch_size
            )

        return accuracies

    def _print_retain_accuracy(self, logger, tag, batch_size=8):
        root = logger.get_root()
        output_dir = os.path.join(root, f"retain_eval_{tag}")
        os.makedirs(output_dir, exist_ok=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"\n[Retain Eval] tag={tag} (in-memory)")
        accuracies = self.eval_accuracy_in_memory(
            output_dir=output_dir, batch_size=batch_size
        )
        print(f"[Retain Eval] tag={tag} accuracies={accuracies}\n")

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _build_checkpoint_dir(self, root, tag=None):
        checkpoint_root = os.path.join(root, "checkpoints")
        os.makedirs(checkpoint_root, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"-{tag}" if tag else ""
        checkpoint_dir = os.path.join(
            checkpoint_root, f"checkpoint-{timestamp}{suffix}"
        )
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

        checkpoint_dirs = []
        for name in os.listdir(checkpoint_root):
            path = os.path.join(checkpoint_root, name)
            if os.path.isdir(path) and name.startswith("checkpoint-"):
                checkpoint_dirs.append(path)

        if not checkpoint_dirs:
            return checkpoint_root

        return max(checkpoint_dirs, key=os.path.getmtime)

    def save(self, logger, tag=None):
        root = logger.get_root()
        checkpoint_dir = self._build_checkpoint_dir(root, tag=tag)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        self.latest_checkpoint_path = checkpoint_dir
        print(f"Checkpoint saved to: {checkpoint_dir}")

    def run(self, logger):
        if self.resume_path is None:
            self.init_model()
            self.init_optimizer()
            self.init_dataset()
            self.init_mask(logger)
            self.init_unlearner(logger)


            self.save(logger, tag="init")
            self._print_retain_accuracy(logger, tag="init")


            self._run_conflict_training(logger)

            self.save(logger, tag="final")
            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")
            self.eval(logger)
        else:
            self.init_model()
            self.init_dataset()
            self.eval(logger)

    def _run_conflict_training(self, logger):

        print(f"开始冲突学习训练，交替频率：{self.alternate_frequency} epochs")
        print(f"Safe unlearn method: {self.safe_unlearn_method}")
        print(f"Conflict unlearn method: {self.conflict_unlearn_method}")


        steps_per_epoch = len(self.unlearn_dataset) // (
            self.batch_size * self.gradient_accumulation_steps * self.num_devices
        )


        self._custom_conflict_training_loop(steps_per_epoch, logger)

    def _custom_conflict_training_loop(self, steps_per_epoch, logger):

        self.model.train()


        if self.optimizer is None:
            self.init_optimizer()


        training_unlearner = self.conflict_unlearner
        self._print_parameter_freeze_report(training_unlearner)


        total_steps = self.num_epochs * steps_per_epoch


        train_dataloader = torch.utils.data.DataLoader(
            self.unlearn_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.unlearn_collator,
        )

        current_step = 0

        print(f"开始训练，总epoch数：{self.num_epochs}，每epoch步数：{steps_per_epoch}")

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")


            # if epoch==0:
            #     current_unlearner = self.safe_unlearner
            #     current_method = self.safe_unlearn_method

            # else:
            #     current_unlearner = self.conflict_unlearner
            #     current_method = self.conflict_unlearn_method

            current_unlearner = self.conflict_unlearner
            current_method = self.conflict_unlearn_method
            print(f"使用 Conflict unlearner: {current_method}")
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_dataloader):
                if current_step >= total_steps:
                    break


                batch = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in batch.items()}


                amp_ctx = (
                    torch.cuda.amp.autocast(dtype=torch.bfloat16)
                    if not self.use_cpu
                    else contextlib.nullcontext()
                )
                with amp_ctx:
                    loss = current_unlearner.compute_loss(current_unlearner.model, batch)


                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告：Batch {batch_idx} 损失为 {loss.item()}，跳过此批次")
                    continue


                loss.backward()


                if self.optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)


                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.optimizer is not None:

                        has_nan_grad = False
                        for param in self.model.parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                has_nan_grad = True
                                break

                        if has_nan_grad:
                            print(f"警告：Batch {batch_idx} 梯度包含NaN，跳过此步骤")
                            self.optimizer.zero_grad()
                        else:
                            if current_unlearner.mask is not None:
                                current_unlearner.mask_gradient(
                                    current_unlearner.model,
                                    current_unlearner.if_wanda,
                                )
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    current_step += 1

                epoch_loss += loss.item()
                num_batches += 1


                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch + 1} 完成，平均损失: {avg_epoch_loss:.4f}")


            if (epoch + 1) % 1 == 0:
                self.save(logger, tag=f"epoch-{epoch + 1}")

                self._print_retain_accuracy(logger, tag=f"epoch-{epoch + 1}")

        print("冲突学习训练完成")


def get(**kwargs):
    return Unlearn(**kwargs)
