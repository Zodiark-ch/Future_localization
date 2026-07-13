import random
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .Base import BaseDataset, UnlearnDataset


class Winogrande(BaseDataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        dataset = defaultdict()
        train_dataset = load_dataset(
            "allenai/winogrande",
            "winogrande_debiased",
            split="train",
            cache_dir="./.cache/data",
        )

        dataset["train"] = train_dataset
        print(f"Train dataset: {len(dataset['train'])}")
        dataset["test"] = load_dataset(
            "allenai/winogrande",
            "winogrande_debiased",
            split="validation",
            cache_dir="./.cache/data",
        )

        return dataset

    def __preprocess__(self, tokenizer):
        def preprocess(examples):
            results = {"input_ids": [], "attention_mask": [], "label": []}


            texts = []
            for i in range(len(examples['sentence'])):
                sample = {
                    'sentence': examples['sentence'][i],
                    'option1': examples['option1'][i],
                    'option2': examples['option2'][i],
                    'answer': examples['answer'][i]
                }

                label = str(sample['answer'])
                question = sample['sentence']
                answer = sample['option1'] if label == "1" else sample['option2']

                full_text = f"{question}\nShould the '_' be {answer}?\nAnswer: Yes"
                converted_item = _single_final_token_item(tokenizer, full_text, max_len=200)
                if converted_item is None:
                    continue
                results["input_ids"].append(converted_item["input_ids"])
                results["attention_mask"].append(converted_item["attention_mask"])
                results["label"].append(converted_item["label"])

            return results

        train_dataset = self.dataset["train"].map(
            preprocess, batched=True, remove_columns=["sentence", "option1", "option2", "answer"]
        )
        test_dataset = self.dataset["test"].map(
            preprocess, batched=True, remove_columns=["sentence", "option1", "option2", "answer"]
        )

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        self.dataset["train"] = train_dataset
        self.dataset["test"] = test_dataset
        return self.dataset

    def build_dataset(self, tokenizer):
        self.__preprocess__(tokenizer)
        return self.dataset


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
