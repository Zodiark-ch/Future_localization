import random
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .Base import BaseDataset, UnlearnDataset


class wikitext(BaseDataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        dataset = defaultdict()
        train_dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="train"
        )
        dataset["train"] = train_dataset
        print(f"Train dataset: {len(dataset['train'])}")
        dataset["test"] = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test"
        )

        return dataset

    def __preprocess__(self, tokenizer):
        ignored_label_token_ids = _ignored_label_token_ids(tokenizer)

        def preprocess(examples):
            results = {"input_ids": [], "attention_mask": [], "label": []}

            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
                max_length=512,
            )
            
            results["input_ids"] = tokenized.input_ids
            results["attention_mask"] = tokenized.attention_mask
            results["label"] = [
                _semantic_labels(input_ids, attention_mask, ignored_label_token_ids)
                for input_ids, attention_mask in zip(
                    tokenized.input_ids, tokenized.attention_mask
                )
            ]
            return results

        train_raw = self.dataset["train"].filter(lambda example: example["text"].strip() != "")
        test_raw = self.dataset["test"].filter(lambda example: example["text"].strip() != "")

        train_dataset = train_raw.map(
            preprocess, batched=True, remove_columns=["text"]
        )
        test_dataset = test_raw.map(
            preprocess, batched=True, remove_columns=["text"]
        )

        train_dataset = train_dataset.filter(
            lambda example: sum(label != -100 for label in example["label"]) >= 2
        )
        test_dataset = test_dataset.filter(
            lambda example: sum(label != -100 for label in example["label"]) >= 2
        )

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        return train_dataset, test_dataset

    def build_dataset(self, tokenizer):
        train_dataset, test_dataset = self.__preprocess__(tokenizer)
        self.dataset["train"] = train_dataset
        self.dataset["test"] = test_dataset
        return self.dataset


def _ignored_label_token_ids(tokenizer):
    ignored = set(tokenizer.all_special_ids)
    if tokenizer.pad_token_id is not None:
        ignored.add(tokenizer.pad_token_id)
    for token_id in range(len(tokenizer)):
        if tokenizer.decode([token_id], skip_special_tokens=False).strip() == "":
            ignored.add(token_id)
    return ignored


def _semantic_labels(input_ids, attention_mask, ignored_label_token_ids):
    labels = []
    for token_id, mask in zip(input_ids, attention_mask):
        if mask == 0 or token_id in ignored_label_token_ids:
            labels.append(-100)
        else:
            labels.append(token_id)
    return labels
