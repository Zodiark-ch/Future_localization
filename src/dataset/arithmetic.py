from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch


ARITHMETIC_DATASET_NAMES = tuple(f"{digit}_digit_arithmetic" for digit in range(1, 6))
ARITHMETIC_OPTION_LABELS = tuple("ABCDEFG")
ARITHMETIC_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "datasets" / "arithmetic"
ARITHMETIC_MAX_TRAIN_SAMPLES = 10_000
ARITHMETIC_MAX_EVAL_SAMPLES = 600


def arithmetic_json_path(dataset_name: str, data_root: str | Path | None = None) -> Path:
    if dataset_name not in ARITHMETIC_DATASET_NAMES:
        raise ValueError(f"Unknown arithmetic dataset: {dataset_name}")
    root = Path(data_root) if data_root is not None else ARITHMETIC_DATA_ROOT
    return root / f"{dataset_name}.json"


def arithmetic_label(example: dict, shuffle_seed: int = 0) -> str:
    option_items = _shuffled_option_items(example, shuffle_seed=shuffle_seed)
    correct_labels = [
        label
        for label, (_, score) in zip(ARITHMETIC_OPTION_LABELS, option_items)
        if float(score) == 1.0
    ]
    if len(correct_labels) != 1:
        raise ValueError(f"Expected exactly one correct arithmetic option, got {len(correct_labels)}.")
    return correct_labels[0]


def build_arithmetic_prompt(example: dict, shuffle_seed: int = 0) -> str:
    option_items = _shuffled_option_items(example, shuffle_seed=shuffle_seed)
    options = [
        f"{label}. {option}"
        for label, (option, _) in zip(ARITHMETIC_OPTION_LABELS, option_items)
    ]
    option_text = ", ".join(options)
    return (
        f"Please choose the correct option from the following: \"{example['input']}\" "
        f"Options: {option_text}. The answer is "
    )


def _shuffled_option_items(example: dict, shuffle_seed: int = 0) -> list[tuple[str, float]]:
    target_scores = example.get("target_scores") or {}
    if len(target_scores) > len(ARITHMETIC_OPTION_LABELS):
        raise ValueError("Arithmetic examples support at most seven options.")
    if not target_scores:
        raise ValueError("Arithmetic example has no target_scores options.")

    option_items = [(option, float(score)) for option, score in target_scores.items()]
    rng = random.Random(_stable_shuffle_seed(example, shuffle_seed))
    rng.shuffle(option_items)
    return option_items


def _stable_shuffle_seed(example: dict, shuffle_seed: int) -> int:
    payload = json.dumps(
        {
            "input": example.get("input"),
            "target": example.get("target"),
            "target_scores": list((example.get("target_scores") or {}).items()),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) ^ int(shuffle_seed)


def load_arithmetic_split(
    dataset_name: str,
    dataset_seed: int,
    max_train_samples: int = ARITHMETIC_MAX_TRAIN_SAMPLES,
    max_eval_samples: int = ARITHMETIC_MAX_EVAL_SAMPLES,
) -> tuple[list[dict], list[dict]]:
    path = arithmetic_json_path(dataset_name)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    examples = list(data["examples"])
    rng = random.Random(dataset_seed)
    rng.shuffle(examples)
    train_count = len(examples) if max_train_samples is None else min(len(examples), max_train_samples)
    eval_count = len(examples) if max_eval_samples is None else min(len(examples), max_eval_samples)
    return examples[:train_count], examples[:eval_count]


class ArithmeticDatasetWrapper:
    def __init__(self, arithmetic_data: list[dict], target_tokenizer, max_len: int = 160, shuffle_seed: int = 0):
        self.arithmetic_data = arithmetic_data
        self.target_tokenizer = target_tokenizer
        self.max_len = max_len
        self.shuffle_seed = shuffle_seed

        self.converted_data = []
        skipped = 0
        for data_item in arithmetic_data:
            converted_item = self._convert_item(data_item)
            if converted_item is None:
                skipped += 1
                continue
            self.converted_data.append(converted_item)
        self.length = len(self.converted_data)
        if skipped:
            print(f"ArithmeticDatasetWrapper skipped {skipped} samples with invalid option labels or length.")

    def _convert_item(self, data_item: dict):
        label = arithmetic_label(data_item, shuffle_seed=self.shuffle_seed)
        label_token_ids = self.target_tokenizer.encode(label, add_special_tokens=False)
        if len(label_token_ids) != 1:
            return None

        prompt_text = build_arithmetic_prompt(data_item, shuffle_seed=self.shuffle_seed)
        tokenized = self.target_tokenizer(
            prompt_text,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids[0].long()
        if input_ids.numel() == 0 or input_ids.numel() > self.max_len:
            return None

        active_len = int(input_ids.numel())
        pad_token_id = self.target_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        input_ids_padded = input_ids.tolist() + [pad_token_id] * (self.max_len - active_len)
        attention_mask = [1] * active_len + [0] * (self.max_len - active_len)
        labels = torch.full((self.max_len,), -100, dtype=torch.long)
        labels[active_len - 1] = label_token_ids[0]
        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": labels,
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.converted_data[idx]

    def select(self, indices):
        new_wrapper = ArithmeticDatasetWrapper.__new__(ArithmeticDatasetWrapper)
        new_wrapper.arithmetic_data = self.arithmetic_data
        new_wrapper.target_tokenizer = self.target_tokenizer
        new_wrapper.max_len = self.max_len
        new_wrapper.shuffle_seed = self.shuffle_seed
        new_wrapper.length = len(indices)
        new_wrapper.converted_data = [self.converted_data[i] for i in indices]
        return new_wrapper


def build_arithmetic_datasets(dataset_name: str, tokenizer, dataset_seed: int):
    train_examples, test_examples = load_arithmetic_split(dataset_name, dataset_seed)
    return (
        ArithmeticDatasetWrapper(train_examples, tokenizer, shuffle_seed=dataset_seed),
        ArithmeticDatasetWrapper(test_examples, tokenizer, shuffle_seed=dataset_seed),
    )