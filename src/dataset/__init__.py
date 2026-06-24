from collections import defaultdict

import torch
from transformers import default_data_collator

from .Base import FinetuningDataset, UnlearnDataset, finetuning_collator, unlearncollector
from .C4 import C4
from .HorryPotter import HP
from .SafePku import SafePkuDataset
from .Tofu import ToFU
from .wmdp import WMDPBio, WMDPCyber, WMDPALL
from .wikitext2 import wikitext
from .ioi_dataset import IOIDataset, NAMES
from .docstring import docstring_induction_prompt_generator
from .gender import load_datasets
from .winogrande import Winogrande
from .sst2 import SST2
from .arithmetic import ARITHMETIC_DATASET_NAMES, build_arithmetic_datasets
import huggingface_hub
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
def _single_token_ioi_names(tokenizer):
    valid_names = [
        name for name in NAMES
        if len(tokenizer.encode(name, add_special_tokens=False)) == 1
    ]
    if len(valid_names) < 3:
        raise ValueError("IOI requires at least three single-token names for prompt generation.")
    return valid_names


# IOI数据集包装器，用于将IOI数据集转换为与其他数据集兼容的格式
class IOIDatasetWrapper:
    def __init__(self, ioi_dataset, target_tokenizer):
        self.ioi_dataset = ioi_dataset
        self.target_tokenizer = target_tokenizer
        self.max_len = 50
        
        self.converted_data = []
        skipped = 0
        for i in range(len(ioi_dataset)):
            converted_item = self._convert_prompt(self.ioi_dataset.ioi_prompts[i])
            if converted_item is None:
                skipped += 1
                continue
            self.converted_data.append(converted_item)
        self.length = len(self.converted_data)
        if skipped:
            print(f"IOIDatasetWrapper skipped {skipped} prompts with invalid IOI name labels.")

    def _convert_prompt(self, prompt):
        original_text = prompt["text"].rstrip()
        io_name = prompt["IO"]
        io_token_ids = self.target_tokenizer.encode(io_name, add_special_tokens=False)
        if len(io_token_ids) != 1:
            return None
        if not original_text.endswith(io_name):
            return None
        prompt_text = original_text[:-len(io_name)].rstrip()
        if not prompt_text:
            return None

        tokenized = self.target_tokenizer(
            prompt_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = tokenized.input_ids[0].long()
        attention_mask = tokenized.attention_mask[0].long()
        active_len = int(attention_mask.sum().item())
        if active_len == 0:
            return None

        io_token_id = io_token_ids[0]
        label_position = active_len - 1
        labels = torch.full_like(input_ids, -100)
        labels[label_position] = io_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = IOIDatasetWrapper.__new__(IOIDatasetWrapper)
        new_wrapper.ioi_dataset = self.ioi_dataset
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.max_len = self.max_len
        new_wrapper.length = len(indices)
        # 直接复制原始数据，让__getitem__方法处理padding
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper

# Induction数据集包装器，用于将induction数据集转换为与其他数据集兼容的格式
class InductionDatasetWrapper:
    def __init__(self, induction_dataset, target_tokenizer):
        self.induction_dataset = induction_dataset
        self.target_tokenizer = target_tokenizer
        self.max_len = 600
        
        # 导入GPT-2 tokenizer用于解码
        from transformers import GPT2TokenizerFast
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('ArthurConmy/redwood_tokenizer')
        
        self.converted_data = []
        skipped = 0
        for i in range(len(induction_dataset)):
            converted_item = self._convert_item(self.induction_dataset[i])
            if converted_item is None:
                skipped += 1
                continue
            self.converted_data.append(converted_item)
        self.length = len(self.converted_data)
        if skipped:
            print(f"InductionDatasetWrapper skipped {skipped} samples without a final target token.")

    def _convert_item(self, gpt2_tokens):
        original_text = self.gpt2_tokenizer.decode(gpt2_tokens, skip_special_tokens=True)
        tokenized = self.target_tokenizer(
            original_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,
            return_tensors="pt",
        )

        full_input_ids = tokenized.input_ids[0].long()
        full_attention_mask = tokenized.attention_mask[0].long()
        full_active_len = int(full_attention_mask.sum().item())
        if full_active_len < 2:
            return None

        final_token_id = full_input_ids[full_active_len - 1].item()
        input_ids = full_input_ids[: full_active_len - 1]
        attention_mask = full_attention_mask[: full_active_len - 1]
        prompt_active_len = int(attention_mask.sum().item())
        if prompt_active_len == 0:
            return None

        pad_token_id = self.target_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        input_ids_padded = input_ids.tolist() + [pad_token_id] * (
            self.max_len - len(input_ids)
        )
        attention_mask_padded = attention_mask.tolist() + [0] * (
            self.max_len - len(attention_mask)
        )
        labels = torch.full((self.max_len,), -100, dtype=torch.long)
        labels[prompt_active_len - 1] = final_token_id

        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_padded, dtype=torch.long),
            "label": labels,
        }
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = InductionDatasetWrapper.__new__(InductionDatasetWrapper)
        new_wrapper.induction_dataset = self.induction_dataset
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.max_len = self.max_len
        new_wrapper.gpt2_tokenizer = self.gpt2_tokenizer
        new_wrapper.length = len(indices)
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper

# Docstring数据集包装器，用于将docstring数据集转换为与其他数据集兼容的格式
class DocstingDatasetWrapper:
    def __init__(self, docstring_data, target_tokenizer):
        self.docstring_data = docstring_data
        self.target_tokenizer = target_tokenizer
        self.max_len = 50
        
        self.converted_data = []
        skipped = 0
        for i in range(len(docstring_data)):
            converted_item = self._convert_item(self.docstring_data[i])
            if converted_item is None:
                skipped += 1
                continue
            self.converted_data.append(converted_item)
        self.length = len(self.converted_data)
        if skipped:
            print(f"DocstingDatasetWrapper skipped {skipped} samples without a final target token.")

    def _convert_item(self, prompt_item):
        full_text = prompt_item.clean_prompt + prompt_item.correct_answers[0]
        return _single_final_token_item(self.target_tokenizer, full_text, self.max_len)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = DocstingDatasetWrapper.__new__(DocstingDatasetWrapper)
        new_wrapper.docstring_data = self.docstring_data
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.max_len = self.max_len
        new_wrapper.length = len(indices)
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper


def _single_final_token_item(tokenizer, full_text, max_len):
    tokenized = tokenizer(
        full_text,
        add_special_tokens=True,
        return_tensors="pt",
    )
    full_input_ids = tokenized.input_ids[0].long()
    if full_input_ids.numel() < 2:
        return None

    final_token_id = int(full_input_ids[-1].item())
    input_ids = full_input_ids[:-1]
    if input_ids.numel() > max_len:
        input_ids = input_ids[-max_len:]

    active_len = int(input_ids.numel())
    if active_len == 0:
        return None

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    input_ids_padded = input_ids.tolist() + [pad_token_id] * (max_len - active_len)
    attention_mask = [1] * active_len + [0] * (max_len - active_len)
    labels = torch.full((max_len,), -100, dtype=torch.long)
    labels[active_len - 1] = final_token_id
    return {
        "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "label": labels,
    }

# Gender数据集包装器，用于将gender数据集转换为与其他数据集兼容的格式
class GenderDatasetWrapper:
    def __init__(self, gender_data, target_tokenizer):
        self.gender_data = gender_data
        self.target_tokenizer = target_tokenizer
        self.max_len = 50
        
        self.converted_data = []
        skipped = 0
        for i in range(len(gender_data)):
            converted_item = self._convert_item(self.gender_data[i])
            if converted_item is None:
                skipped += 1
                continue
            self.converted_data.append(converted_item)
        self.length = len(self.converted_data)
        if skipped:
            print(f"GenderDatasetWrapper skipped {skipped} samples with invalid pronoun labels.")

    def _convert_item(self, data_item):
        pronoun_token_ids = self.target_tokenizer.encode(
            data_item["pronoun"], add_special_tokens=False
        )
        if len(pronoun_token_ids) != 1:
            return None

        tokenized = self.target_tokenizer(
            data_item["prefix"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids[0].long()
        attention_mask = tokenized.attention_mask[0].long()
        active_len = int(attention_mask.sum().item())
        if active_len == 0:
            return None

        labels = torch.full_like(input_ids, -100)
        labels[active_len - 1] = pronoun_token_ids[0]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = GenderDatasetWrapper.__new__(GenderDatasetWrapper)
        new_wrapper.gender_data = self.gender_data
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.max_len = self.max_len
        new_wrapper.length = len(indices)
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper

# Bool数据集包装器，用于将boolean expression数据集转换为与其他数据集兼容的格式
class BoolDatasetWrapper:
    def __init__(self, bool_data, target_tokenizer):
        self.bool_data = bool_data
        self.target_tokenizer = target_tokenizer
        self.max_len = 100
        
        self.converted_data = []
        skipped = 0
        for i in range(len(bool_data)):
            converted_item = self._convert_item(self.bool_data[i])
            if converted_item is None:
                skipped += 1
                continue
            self.converted_data.append(converted_item)
        self.length = len(self.converted_data)
        if skipped:
            print(f"BoolDatasetWrapper skipped {skipped} samples with invalid answer labels.")

    def _convert_item(self, data_item):
        answer_token_ids = self.target_tokenizer.encode(
            data_item["target"], add_special_tokens=False
        )
        if len(answer_token_ids) != 1:
            return None

        expression = data_item["input"].strip()
        if expression.endswith(" is"):
            expression = expression[: -len(" is")].rstrip()
        prompt_text = (
            "[INST] <<SYS>>\n"
            "Evaluate the following boolean expression as either 'True' or 'False'.\n"
            "<</SYS>>\n\n"
            f"{expression} [/INST]"
        )

        tokenized = self.target_tokenizer(
            prompt_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids[0].long()
        attention_mask = tokenized.attention_mask[0].long()
        active_len = int(attention_mask.sum().item())
        if active_len == 0:
            return None

        labels = torch.full_like(input_ids, -100)
        labels[active_len - 1] = answer_token_ids[0]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 直接返回已经padding好的数据
        return self.converted_data[idx]
    
    def select(self, indices):
        """
        添加select方法以支持UnlearnDataset的build_unlearn_dataset方法
        """
        new_wrapper = BoolDatasetWrapper.__new__(BoolDatasetWrapper)
        new_wrapper.bool_data = self.bool_data
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.max_len = self.max_len
        new_wrapper.length = len(indices)
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper

def get_validation_data(num_examples=None, seq_len=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(
        validation_fname, map_location=torch.device("cpu")
    ).long()

    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]

def get_mask_repeat_candidates(num_examples=None, seq_len=None):
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = torch.load(
        mask_repeat_candidates_fname, map_location=torch.device("cpu")
    )
    mask_repeat_candidates.requires_grad = False

    if num_examples is None:
        return mask_repeat_candidates
    else:
        return mask_repeat_candidates[:num_examples, :seq_len]


def _to_labels_dataset(dataset):
    if dataset is None:
        return None
    if len(dataset) == 0:
        return dataset
    first = dataset[0]
    if "labels" in first:
        return dataset
    if "label" not in first:
        return dataset
    if hasattr(dataset, "rename_column"):
        try:
            return dataset.rename_column("label", "labels")
        except Exception:
            pass
    return _LabelAliasDataset(dataset)


class _LabelAliasDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        if "labels" not in item and "label" in item:
            item["labels"] = item["label"]
        return item

    def select(self, indices):
        if hasattr(self.dataset, "select"):
            return _LabelAliasDataset(self.dataset.select(indices))
        return _LabelAliasDataset([self.dataset[i] for i in indices])


def _load_task_dataset(dataset_name, tokenizer, dataset_seed, if_llama=False, role="target"):
    if dataset_name is None or dataset_name == "none":
        return None, None
    if dataset_name == "SafePku":
        dataset = SafePkuDataset("SafePku", if_llama=if_llama).build_dataset(tokenizer)
    elif dataset_name == "C4":
        dataset = C4("C4").build_dataset(tokenizer)
    elif dataset_name == "wikitext":
        dataset = wikitext("wikitext").build_dataset(tokenizer)
    elif dataset_name == "HP":
        dataset = HP("HP").build_dataset(tokenizer)
    elif "Tofu" in dataset_name:
        subset = dataset_name.split("_", 1)[1] if "_" in dataset_name else "full"
        dataset = ToFU("TOFU", subset=subset, if_llama=if_llama).build_dataset(tokenizer)
    elif dataset_name == "WMDPCyber":
        subset = "target" if role == "target" else "retain"
        subset = "forget" if subset == "target" else subset
        dataset = WMDPCyber("WMDPCyber", subset=subset).build_dataset(tokenizer)
    elif dataset_name == "WMDPBio":
        subset = "target" if role == "target" else "retain"
        subset = "forget" if subset == "target" else subset
        dataset = WMDPBio("WMDPBio", subset=subset).build_dataset(tokenizer)
    elif dataset_name == "WMDPALL":
        subset = "target" if role == "target" else "retain"
        subset = "forget" if subset == "target" else subset
        dataset = WMDPALL("WMDPALL", subset=subset).build_dataset(tokenizer)
    elif dataset_name == "winogrande":
        dataset = Winogrande("winogrande").build_dataset(tokenizer)
    elif dataset_name == "sst2":
        dataset = SST2("sst2").build_dataset(tokenizer)
    elif dataset_name == "IOI":
        n_train = 200 if role == "target" else 2400
        ioi_names = _single_token_ioi_names(tokenizer)
        ioi_dataset = IOIDataset(
            prompt_type="ABBA",
            N=n_train,
            nb_templates=1,
            seed=dataset_seed,
            tokenizer=None,
            names=ioi_names,
        )
        ioi_test = IOIDataset(
            prompt_type="ABBA",
            N=600,
            nb_templates=1,
            seed=dataset_seed,
            tokenizer=None,
            names=ioi_names,
        )
        return (
            _to_labels_dataset(IOIDatasetWrapper(ioi_dataset, tokenizer)),
            _to_labels_dataset(IOIDatasetWrapper(ioi_test, tokenizer)),
        )
    elif dataset_name == "induction":
        induction_dataset = get_validation_data(num_examples=3000)
        return (
            _to_labels_dataset(InductionDatasetWrapper(induction_dataset[:2400, :].contiguous(), tokenizer)),
            _to_labels_dataset(InductionDatasetWrapper(induction_dataset[2400:3000, :].contiguous(), tokenizer)),
        )
    elif dataset_name == "docstring":
        docstring_ind_prompt_kwargs = dict(
            n_matching_args=3,
            n_def_prefix_args=2,
            n_def_suffix_args=1,
            n_doc_prefix_args=0,
            met_desc_len=3,
            arg_desc_len=2,
        )
        raw_prompts = [
            docstring_induction_prompt_generator(
                "rest", **docstring_ind_prompt_kwargs, seed=j
            )
            for j in range(3000)
        ]
        return (
            _to_labels_dataset(DocstingDatasetWrapper(raw_prompts[:2400], tokenizer)),
            _to_labels_dataset(DocstingDatasetWrapper(raw_prompts[2400:3000], tokenizer)),
        )
    elif dataset_name == "gender":
        dataset = load_datasets("/ssd_users/chenhang/CSAT/files/data/gp", 3000, 600)
        return (
            _to_labels_dataset(GenderDatasetWrapper(dataset["train_3k"], tokenizer)),
            _to_labels_dataset(GenderDatasetWrapper(dataset["test"], tokenizer)),
        )
    elif dataset_name == "bool":
        dataset = load_datasets(
            "/ssd_users/chenhang/CSAT/files/data/boolean_expressions", 3000, 600
        )
        return (
            _to_labels_dataset(BoolDatasetWrapper(dataset["train"], tokenizer)),
            _to_labels_dataset(BoolDatasetWrapper(dataset["test"], tokenizer)),
        )
    elif dataset_name in ARITHMETIC_DATASET_NAMES:
        train_dataset, test_dataset = build_arithmetic_datasets(
            dataset_name, tokenizer, dataset_seed
        )
        return _to_labels_dataset(train_dataset), _to_labels_dataset(test_dataset)
    else:
        raise ValueError(f"No dataset: {dataset_name}")

    return _to_labels_dataset(dataset["train"]), _to_labels_dataset(dataset["test"])


def get_finetuning_dataset(
    dataset_names,
    tokenizer,
    dataset_seed,
    target_ratio,
    target_holdout_as_pervasiveness=False,
    if_llama=False,
):
    target_train, target_test = _load_task_dataset(
        dataset_names.get("target"), tokenizer, dataset_seed, if_llama=if_llama, role="target"
    )

    pervasiveness_names = dataset_names.get("pervasiveness")
    if isinstance(pervasiveness_names, list):
        names = pervasiveness_names
    elif pervasiveness_names is None:
        names = []
    else:
        names = [pervasiveness_names]

    pervasiveness_datasets = {}
    pervasiveness_test_datasets = {}
    for idx, name in enumerate(names):
        train_dataset, test_dataset = _load_task_dataset(
            name, tokenizer, dataset_seed, if_llama=if_llama, role="pervasiveness"
        )
        key = "pervasiveness" if len(names) == 1 else f"pervasiveness{idx + 1}"
        test_key = name if name else key
        pervasiveness_datasets[key] = train_dataset
        pervasiveness_test_datasets[test_key] = test_dataset

    if not pervasiveness_datasets:
        pervasiveness_datasets = {"pervasiveness": None}
        pervasiveness_test_datasets = {}

    finetuning_dataset = FinetuningDataset(
        {"target": target_train, **pervasiveness_datasets},
        target_ratio,
        dataset_seed,
        target_holdout_as_pervasiveness,
    )

    target_test_datasets = {}
    if target_test is not None:
        target_test_datasets[dataset_names.get("target", "target")] = target_test

    return (
        finetuning_dataset,
        target_test_datasets,
        pervasiveness_test_datasets,
        finetuning_collator,
        default_data_collator,
    )

def get_dataset(
    dataset_names,
    tokenizer,
    dataset_seed,
    forget_ratio,
    self_retain=False,
    if_llama=False,
):
    ### forget dataset & test dataset
    if dataset_names["forget"] == "SafePku":
        dataset = SafePkuDataset("SafePku", if_llama=if_llama)
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "wikitext":
        dataset = wikitext("wikitext")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "HP":
        dataset = HP("HP")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif "Tofu" in dataset_names["forget"]:
        subset = dataset_names["forget"].split("_")[1]
        dataset = ToFU("TOFU", subset=subset, if_llama=if_llama)
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "WMDPCyber":
        dataset = WMDPCyber("WMDPCyber", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "WMDPBio":
        dataset = WMDPBio("WMDPBio", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "WMDPALL":
        dataset = WMDPALL("WMDPALL", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "IOI":
        # 创建IOI数据集作为forget数据集，使用GPT-2 tokenizer生成数据
        ioi_names = _single_token_ioi_names(tokenizer)
        ioi_dataset = IOIDataset(
            prompt_type="ABBA",
            N=200,
            nb_templates=1,
            seed=dataset_seed,
            tokenizer=None,  # 使用默认的GPT-2 tokenizer
            names=ioi_names,
        )
        
        forget_dataset = IOIDatasetWrapper(ioi_dataset, tokenizer)
        test_dataset = IOIDatasetWrapper(ioi_dataset, tokenizer)  # 使用相同的数据集作为测试集
    elif "forget" not in dataset_names:
        forget_dataset = None
        test_dataset = None
    else:
        raise ValueError("No dataset")

    #### retain dataset
    retain_datasets = {}
    test_datasets = {}
    
    # 检查retain是否是列表（多个数据集）
    if isinstance(dataset_names["retain"], list):
        retain_dataset_names = dataset_names["retain"]
    else:
        retain_dataset_names = [dataset_names["retain"]]
    
    for i, retain_name in enumerate(retain_dataset_names):
        if retain_name == "SafePku":
            dataset = SafePkuDataset("SafePku")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "C4":
            dataset = C4("C4")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "winogrande":
            dataset = Winogrande("winogrande")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "sst2":
            dataset = SST2("sst2")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "wikitext":
            dataset = wikitext("wikitext")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif "Tofu" in retain_name:
            subset = retain_name.split("_")[1]
            dataset = ToFU("TOFU", subset=subset, if_llama=if_llama)
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "WMDPCyber":
            dataset = WMDPCyber("WMDPCyber", subset="retain")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "WMDPBio":
            dataset = WMDPBio("WMDPBio", subset="retain")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "WMDPALL":
            dataset = WMDPALL("WMDPALL", subset="retain")
            dataset = dataset.build_dataset(tokenizer)
            retain_datasets[f"retain{i+1}"] = dataset["train"]
            test_datasets[f"test{i+1}"] = dataset["test"]
        elif retain_name == "IOI":
            # 创建IOI数据集，使用GPT-2 tokenizer生成数据
            ioi_names = _single_token_ioi_names(tokenizer)
            ioi_dataset = IOIDataset(
                prompt_type="ABBA",
                N=2400,
                nb_templates=1,
                seed=dataset_seed,
                tokenizer=None,  # 使用默认的GPT-2 tokenizer
                names=ioi_names,
            )
            ioi_dataset_test=IOIDataset(
                prompt_type="ABBA",
                N=600,
                nb_templates=1,
                seed=dataset_seed,
                tokenizer=None,  # 使用默认的GPT-2 tokenizer
                names=ioi_names,
            )
            
            retain_datasets[f"retain{i+1}"] = IOIDatasetWrapper(ioi_dataset, tokenizer)
            test_datasets[f"test{i+1}"] = IOIDatasetWrapper(ioi_dataset_test, tokenizer)
        elif retain_name == "induction":
            induction_dataset=get_validation_data(num_examples=3000)
            validation_slice = slice(0, 2400)
            test_slice = slice(2400, 3000)
            validation_data = induction_dataset[validation_slice, :].contiguous()
            test_data = induction_dataset[test_slice, :].contiguous()
            retain_datasets[f"retain{i+1}"] = InductionDatasetWrapper(validation_data, tokenizer)
            test_datasets[f"test{i+1}"] = InductionDatasetWrapper(test_data, tokenizer)
        elif retain_name == "docstring":
            docstring_ind_prompt_kwargs = dict(
            n_matching_args=3, n_def_prefix_args=2, n_def_suffix_args=1, n_doc_prefix_args=0, met_desc_len=3, arg_desc_len=2
        )
            raw_prompts = [
            docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=j)
            for j in range(3000)
        ]
            # 划分数据：前800个作为validation_data，后200个作为test_data
            validation_data = raw_prompts[:2400]
            test_data = raw_prompts[2400:3000]
            retain_datasets[f"retain{i+1}"] = DocstingDatasetWrapper(validation_data, tokenizer)
            test_datasets[f"test{i+1}"] = DocstingDatasetWrapper(test_data, tokenizer)
        elif retain_name == "gender":
            dataset=load_datasets("/ssd_users/chenhang/CSAT/files/data/gp", 3000, 600)
            validation_data=dataset["train_3k"]
            test_data=dataset["test"]
            retain_datasets[f"retain{i+1}"] = GenderDatasetWrapper(validation_data, tokenizer)
            test_datasets[f"test{i+1}"] = GenderDatasetWrapper(test_data, tokenizer)
        elif retain_name == "bool":
            dataset=load_datasets("/ssd_users/chenhang/CSAT/files/data/boolean_expressions", 3000, 600)
            validation_data=dataset["train"]
            test_data=dataset["test"]
            retain_datasets[f"retain{i+1}"] = BoolDatasetWrapper(validation_data, tokenizer)
            test_datasets[f"test{i+1}"] = BoolDatasetWrapper(test_data, tokenizer)
        elif "retain" not in dataset_names:
            retain_datasets[f"retain{i+1}"] = None
            test_datasets[f"test{i+1}"] = None
        else:
            raise ValueError(f"No dataset: {retain_name}")

    # 如果没有retain数据集，设置为None
    if not retain_datasets:
        retain_datasets = {"retain": None}
        test_datasets = {"test": None}
    elif len(retain_datasets) == 1:
        # 如果是单个retain数据集，使用"retain"键名以保持向后兼容性
        single_retain_key = list(retain_datasets.keys())[0]
        single_test_key = list(test_datasets.keys())[0]
        retain_datasets = {"retain": retain_datasets[single_retain_key]}
        test_datasets = {"test": test_datasets[single_test_key]}

    #### downstream datasets
    downstream_datasets = {}
    downstream_dataset_names = ["induction", "IOI", "bool", "gender", "docstring","winogrande","sst2"]
    
    # 获取所有retain数据集名称（包括多个数据集的情况）
    all_retain_names = []
    if isinstance(dataset_names["retain"], list):
        all_retain_names = dataset_names["retain"]
    else:
        all_retain_names = [dataset_names["retain"]]
    
    # 为每个不在retain中的downstream数据集创建test_data
    for downstream_name in downstream_dataset_names:
        if downstream_name not in all_retain_names:
            if downstream_name == "induction":
                induction_dataset = get_validation_data(num_examples=3000)
                test_slice = slice(2400, 3000)
                test_data = induction_dataset[test_slice, :].contiguous()
                downstream_datasets[f"downstream_{downstream_name}"] = InductionDatasetWrapper(test_data, tokenizer)
            elif downstream_name == "IOI":
                ioi_names = _single_token_ioi_names(tokenizer)
                ioi_dataset_test = IOIDataset(
                    prompt_type="ABBA",
                    N=600,
                    nb_templates=1,
                    seed=dataset_seed,
                    tokenizer=None,
                    names=ioi_names,
                )
                downstream_datasets[f"downstream_{downstream_name}"] = IOIDatasetWrapper(ioi_dataset_test, tokenizer)
            elif downstream_name == "bool":
                dataset = load_datasets("/ssd_users/chenhang/CSAT/files/data/boolean_expressions", 3000, 600)
                test_data = dataset["test"]
                downstream_datasets[f"downstream_{downstream_name}"] = BoolDatasetWrapper(test_data, tokenizer)
            elif downstream_name == "gender":
                dataset = load_datasets("/ssd_users/chenhang/CSAT/files/data/gp", 3000, 600)
                test_data = dataset["test"]
                downstream_datasets[f"downstream_{downstream_name}"] = GenderDatasetWrapper(test_data, tokenizer)
            elif downstream_name == "docstring":
                docstring_ind_prompt_kwargs = dict(
                    n_matching_args=3, n_def_prefix_args=2, n_def_suffix_args=1, n_doc_prefix_args=0, met_desc_len=3, arg_desc_len=2
                )
                raw_prompts = [
                    docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=j)
                    for j in range(3000)
                ]
                test_data = raw_prompts[2400:3000]
                downstream_datasets[f"downstream_{downstream_name}"] = DocstingDatasetWrapper(test_data, tokenizer)

    unlearn_dataset = UnlearnDataset(
        {"forget": forget_dataset, **retain_datasets},
        forget_ratio,
        dataset_seed,
        self_retain,
    )
    unlearn_collator = unlearncollector

    test_collator = default_data_collator

    return unlearn_dataset, test_datasets, unlearn_collator, test_collator, downstream_datasets


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset_names = {"forget": "SafePku", "retain": "BookCorpus"}
    dataset_seed = 8888
    forget_ratio = 0.1
    self_retain = False
    unlearn_dataset, test_datasets, unlearn_collator, test_collator, downstream_datasets = get_dataset(
        dataset_names, tokenizer, dataset_seed, forget_ratio, self_retain
    )
    print(len(unlearn_dataset))

    print(f"测试数据集数量: {len(test_datasets)}")
    for key, dataset in test_datasets.items():
        if dataset is not None:
            print(f"{key}: {len(dataset)}")
        else:
            print(f"{key}: None")
    import torch

    dataloader = torch.utils.data.DataLoader(
        unlearn_dataset, batch_size=2, collate_fn=unlearn_collator
    )
    for batch in dataloader:
        print(batch)
        break
