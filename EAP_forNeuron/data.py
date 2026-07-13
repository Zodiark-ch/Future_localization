from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from EAP_forNeuron.schemas import PairBatch, PairExample


BOOL_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n"
    "Evaluate the following boolean expression as either 'True' or 'False'.\n"
    "<</SYS>>\n\n"
    "{expression} [/INST] '"
)


DEFAULT_MAX_LENGTHS = {
    "bool": 128,
    "gender": 64,
    "ioi_mistral": 96,
    "1_digit_arithmetic": 160,
    "2_digit_arithmetic": 160,
    "3_digit_arithmetic": 160,
    "4_digit_arithmetic": 160,
    "5_digit_arithmetic": 160,
}


def default_data_path(dataset_name: str) -> Path:
    root = Path(__file__).resolve().parent
    candidate = root / "data" / f"{dataset_name}.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No default CSV found for dataset {dataset_name!r}.")


def format_bool_prompt(text: str, input_format: str = "auto") -> str:
    text = text.strip()
    if input_format == "prompt" or text.startswith("[INST]"):
        return text
    expression = text
    marker = " is"
    marker_index = expression.rfind(marker)
    if marker_index != -1:
        expression = expression[:marker_index]
    return BOOL_SYSTEM_PROMPT.format(expression=expression.strip())


def format_prompt(dataset_name: str, text: str, input_format: str = "auto") -> str:
    if dataset_name == "bool":
        return format_bool_prompt(text, input_format=input_format)
    return text.strip()


class EAPNeuronPairDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        dataset_name: str,
        corruption_column: str = "corrupted",
        max_samples: int | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.dataset_name = dataset_name
        self.corruption_column = corruption_column
        self.examples = self._load_examples(max_samples=max_samples)

    def _load_examples(self, max_samples: int | None) -> list[PairExample]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {self.csv_path}")
        examples: list[PairExample] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"clean", "correct_idx", "incorrect_idx"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.csv_path} missing required columns: {sorted(missing)}")
            for row in reader:
                corrupted = row.get(self.corruption_column) or row.get("corrupted")
                if corrupted is None or corrupted == "":
                    corrupted = row.get("corrupted", "")
                clean = row.get("clean", "")
                if not clean or not corrupted:
                    continue
                examples.append(
                    PairExample(
                        clean=clean,
                        corrupted=corrupted,
                        correct_idx=int(row["correct_idx"]),
                        incorrect_idx=int(row["incorrect_idx"]),
                    )
                )
                if max_samples is not None and len(examples) >= max_samples:
                    break
        if not examples:
            raise ValueError(f"No usable examples loaded from {self.csv_path}")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PairExample:
        return self.examples[index]


class EAPNeuronCollator:
    def __init__(
        self,
        tokenizer,
        dataset_name: str,
        max_length: int | None = None,
        input_format: str = "auto",
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length = max_length or DEFAULT_MAX_LENGTHS.get(dataset_name, 128)
        self.input_format = input_format

    def __call__(self, examples: list[PairExample]) -> PairBatch:
        clean_texts = [
            format_prompt(self.dataset_name, example.clean, self.input_format)
            for example in examples
        ]
        corrupted_texts = [
            format_prompt(self.dataset_name, example.corrupted, self.input_format)
            for example in examples
        ]
        clean = self.tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        corrupted = self.tokenizer(
            corrupted_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        clean_input_ids = clean["input_ids"]
        clean_attention_mask = clean["attention_mask"]
        labels = torch.full_like(clean_input_ids, -100)
        correct_idx = torch.tensor([example.correct_idx for example in examples], dtype=torch.long)
        incorrect_idx = torch.tensor([example.incorrect_idx for example in examples], dtype=torch.long)
        label_positions = clean_attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
        vocab_size = _tokenizer_vocab_size(self.tokenizer)
        for row, position in enumerate(label_positions.tolist()):
            label = int(correct_idx[row].item())
            if vocab_size is None or 0 <= label < vocab_size:
                labels[row, position] = label
        return PairBatch(
            clean_input_ids=clean_input_ids,
            clean_attention_mask=clean_attention_mask,
            corrupted_input_ids=corrupted["input_ids"],
            corrupted_attention_mask=corrupted["attention_mask"],
            labels=labels,
            correct_idx=correct_idx,
            incorrect_idx=incorrect_idx,
            label_positions=label_positions,
        )


def _tokenizer_vocab_size(tokenizer) -> int | None:
    try:
        return len(tokenizer)
    except TypeError:
        return getattr(tokenizer, "vocab_size", None)


def load_pair_dataset(
    dataset_name: str,
    tokenizer,
    data_path: str | Path | None = None,
    corruption_column: str = "corrupted",
    max_samples: int | None = None,
    max_length: int | None = None,
    input_format: str = "auto",
) -> tuple[EAPNeuronPairDataset, EAPNeuronCollator]:
    csv_path = Path(data_path) if data_path is not None else default_data_path(dataset_name)
    dataset = EAPNeuronPairDataset(
        csv_path=csv_path,
        dataset_name=dataset_name,
        corruption_column=corruption_column,
        max_samples=max_samples,
    )
    collator = EAPNeuronCollator(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        max_length=max_length,
        input_format=input_format,
    )
    return dataset, collator
