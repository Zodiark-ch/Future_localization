import random

import torch
from torch.utils.data import Dataset


class BaseDataset:
    def __init__(self, dataset_name, with_retain=False, if_llama=False):
        self.dataset_name = dataset_name
        self.with_normal = with_retain
        self.if_llama = if_llama
        self.question_start_token = "### Question: "
        self.question_end_token = "\n"
        self.answer_start_token = "### Answer: "
    def get_dataset(self):
        pass

    def __preprocess__(self, tokenizer, dataset_ratio=None, dataset_seed=None):
        pass

    def build_dataset(self, tokenizer, dataset_ratio=None, dataset_seed=None):
        pass


class UnlearnDataset(Dataset):
    def __init__(self, datasets, forget_ratio, dataset_seed, self_retain=False):
        self.forget_ratio = forget_ratio
        self.dataset_seed = dataset_seed
        self.self_retain = self_retain

        if "forget" in datasets.keys():
            self.forget_dataset = datasets["forget"]
        else:
            self.forget_dataset = None

        # 处理多个retain数据集的情况
        self.retain_datasets = {}
        for key, value in datasets.items():
            if key.startswith("retain"):
                self.retain_datasets[key] = value
        
        # 如果没有retain数据集，设置为None
        if not self.retain_datasets:
            self.retain_datasets = {"retain": None}

        self.build_unlearn_dataset()

    def __len__(self):
        if self.forget_dataset:
            return len(self.forget_dataset)
        elif self.retain_datasets:
            # 返回第一个非None的retain数据集的长度
            for dataset in self.retain_datasets.values():
                if dataset is not None:
                    return len(dataset)
        else:
            raise ValueError("No dataset")

    def build_unlearn_dataset(self):
        if self.forget_dataset:
            if self.forget_ratio > 1:
                length = int(self.forget_ratio)

            elif self.forget_ratio <= 1 and self.forget_ratio > 0:
                length = int(len(self.forget_dataset) * self.forget_ratio)

            random.seed(self.dataset_seed)
            forget_index_list = random.sample(range(len(self.forget_dataset)), length)
            if self.self_retain:
                retain_index_list = list(
                    set(range(len(self.forget_dataset))) - set(forget_index_list)
                )
                # 对于多个retain数据集的情况，将self_retain的数据分配给第一个retain数据集
                first_retain_key = list(self.retain_datasets.keys())[0]
                self.retain_datasets[first_retain_key] = self.forget_dataset.select(retain_index_list)
            self.forget_dataset = self.forget_dataset.select(forget_index_list)

    def __getitem__(self, idx):
        if self.forget_dataset:
            forget_data = self.forget_dataset[idx]
            if self.retain_datasets:
                # 随机选择一个retain数据集
                available_retain_keys = [key for key, dataset in self.retain_datasets.items() if dataset is not None]
                if available_retain_keys:
                    selected_retain_key = random.choice(available_retain_keys)
                    selected_retain_dataset = self.retain_datasets[selected_retain_key]
                    retain_idx = random.randint(0, len(selected_retain_dataset) - 1)
                    retain_data = selected_retain_dataset[retain_idx]
                    
                    # 如果是单个retain数据集且键名是"retain"，保持向后兼容性
                    if len(self.retain_datasets) == 1 and "retain" in self.retain_datasets:
                        return {"forget": forget_data, "retain": retain_data}
                    else:
                        return {"forget": forget_data, selected_retain_key: retain_data}
                else:
                    return {"forget": forget_data, "retain": None}
            else:
                return {"forget": forget_data, "retain": None}
        else:
            # 如果没有forget数据集，从第一个可用的retain数据集中获取数据
            for key, dataset in self.retain_datasets.items():
                if dataset is not None:
                    retain_data = dataset[idx]
                    # 如果是单个retain数据集且键名是"retain"，保持向后兼容性
                    if len(self.retain_datasets) == 1 and "retain" in self.retain_datasets:
                        return {"forget": None, "retain": retain_data}
                    else:
                        return {"forget": None, key: retain_data}
            raise ValueError("No available dataset")


def unlearncollector(samples):
    res = {}
    if samples[0]["forget"]:
        forget_samples = [sample["forget"] for sample in samples]
        res["forget"] = (
            torch.stack([sample["input_ids"] for sample in forget_samples]),
            torch.stack([sample["attention_mask"] for sample in forget_samples]),
            torch.stack([sample["label"] for sample in forget_samples]),
            torch.stack([sample["refused_label"] for sample in forget_samples]),
            torch.stack([sample["question_length"] for sample in forget_samples]),
        )
    else:
        res["forget"] = None
    
    # 处理多个retain数据集的情况
    retain_keys = []
    for key in samples[0].keys():
        if key.startswith("retain"):
            retain_keys.append(key)
    
    # 如果没有找到retain键，尝试向后兼容的"retain"键
    if not retain_keys and "retain" in samples[0]:
        retain_keys = ["retain"]
    
    for retain_key in retain_keys:
        if samples[0][retain_key]:
            retain_samples = [sample[retain_key] for sample in samples]
            res[retain_key] = (
                torch.stack([sample["input_ids"] for sample in retain_samples]),
                torch.stack([sample["attention_mask"] for sample in retain_samples]),
                torch.stack([sample["label"] for sample in retain_samples]),
            )
        else:
            res[retain_key] = None
    
    return res


def _sample_dataset(dataset, ratio, seed):
    if dataset is None:
        return None
    if ratio is None or ratio <= 0:
        return dataset
    if ratio > 1:
        length = min(int(ratio), len(dataset))
    else:
        length = max(1, int(len(dataset) * ratio))
    random.seed(seed)
    index_list = random.sample(range(len(dataset)), length)
    if hasattr(dataset, "select"):
        return dataset.select(index_list)
    return [dataset[i] for i in index_list]


class FinetuningDataset(Dataset):
    def __init__(
        self,
        datasets,
        target_ratio,
        dataset_seed,
        target_holdout_as_pervasiveness=False,
    ):
        self.target_ratio = target_ratio
        self.dataset_seed = dataset_seed
        self.target_holdout_as_pervasiveness = target_holdout_as_pervasiveness
        self.target_dataset = datasets.get("target")
        self.pervasiveness_datasets = {
            key: value
            for key, value in datasets.items()
            if key.startswith("pervasiveness")
        }
        if not self.pervasiveness_datasets:
            self.pervasiveness_datasets = {"pervasiveness": None}
        self.build_finetuning_dataset()

    def __len__(self):
        if self.target_dataset is not None:
            return len(self.target_dataset)
        for dataset in self.pervasiveness_datasets.values():
            if dataset is not None:
                return len(dataset)
        raise ValueError("No dataset")

    def build_finetuning_dataset(self):
        if self.target_dataset is None:
            return
        if self.target_ratio is None or self.target_ratio <= 0:
            return

        if self.target_ratio > 1:
            length = min(int(self.target_ratio), len(self.target_dataset))
        else:
            length = max(1, int(len(self.target_dataset) * self.target_ratio))

        random.seed(self.dataset_seed)
        target_index_list = random.sample(range(len(self.target_dataset)), length)
        if self.target_holdout_as_pervasiveness:
            holdout_index_list = list(
                set(range(len(self.target_dataset))) - set(target_index_list)
            )
            first_key = list(self.pervasiveness_datasets.keys())[0]
            if hasattr(self.target_dataset, "select"):
                self.pervasiveness_datasets[first_key] = self.target_dataset.select(
                    holdout_index_list
                )
            else:
                self.pervasiveness_datasets[first_key] = [
                    self.target_dataset[i] for i in holdout_index_list
                ]

        if hasattr(self.target_dataset, "select"):
            self.target_dataset = self.target_dataset.select(target_index_list)
        else:
            self.target_dataset = [self.target_dataset[i] for i in target_index_list]

    def __getitem__(self, idx):
        target_data = None
        if self.target_dataset is not None:
            target_data = self.target_dataset[idx]

        available_keys = [
            key for key, dataset in self.pervasiveness_datasets.items() if dataset is not None
        ]
        if not available_keys:
            return {"target": target_data, "pervasiveness": None}

        selected_key = random.choice(available_keys)
        selected_dataset = self.pervasiveness_datasets[selected_key]
        pervasiveness_idx = random.randint(0, len(selected_dataset) - 1)
        pervasiveness_data = selected_dataset[pervasiveness_idx]
        if len(self.pervasiveness_datasets) == 1 and "pervasiveness" in self.pervasiveness_datasets:
            return {"target": target_data, "pervasiveness": pervasiveness_data}
        return {"target": target_data, selected_key: pervasiveness_data}


def _label_field(sample):
    if "labels" in sample:
        return "labels"
    return "label"


def _clean_labels(labels, attention_mask):
    labels = labels.clone()
    return labels.masked_fill(attention_mask == 0, -100)


def finetuning_collator(samples):
    res = {}
    if samples[0].get("target") is not None:
        target_samples = [sample["target"] for sample in samples]
        label_key = _label_field(target_samples[0])
        input_ids = torch.stack([sample["input_ids"] for sample in target_samples])
        attention_mask = torch.stack([sample["attention_mask"] for sample in target_samples])
        labels = torch.stack([sample[label_key] for sample in target_samples])
        res["target"] = (
            input_ids,
            attention_mask,
            _clean_labels(labels, attention_mask),
        )
    else:
        res["target"] = None

    pervasiveness_keys = [
        key for key in samples[0].keys() if key.startswith("pervasiveness")
    ]
    if not pervasiveness_keys and "pervasiveness" in samples[0]:
        pervasiveness_keys = ["pervasiveness"]

    for pervasiveness_key in pervasiveness_keys:
        if samples[0].get(pervasiveness_key) is not None:
            pervasiveness_samples = [sample[pervasiveness_key] for sample in samples]
            label_key = _label_field(pervasiveness_samples[0])
            input_ids = torch.stack([sample["input_ids"] for sample in pervasiveness_samples])
            attention_mask = torch.stack([sample["attention_mask"] for sample in pervasiveness_samples])
            labels = torch.stack([sample[label_key] for sample in pervasiveness_samples])
            res[pervasiveness_key] = (
                input_ids,
                attention_mask,
                _clean_labels(labels, attention_mask),
            )
        else:
            res[pervasiveness_key] = None

    return res
