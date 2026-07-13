from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset.arithmetic import (  # noqa: E402
    ARITHMETIC_DATASET_NAMES,
    ARITHMETIC_OPTION_LABELS,
    arithmetic_label,
    build_arithmetic_prompt,
)


OPERATIONS = ("addition", "subtraction", "multiplication", "division")
EAP_DATA_DIRS = (
    REPO_ROOT / "EAP_forNeuron" / "data",
    REPO_ROOT / "EAP_forComponent" / "data",
    REPO_ROOT / "EAP_forLogicalCircuit" / "data",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged arithmetic JSON and EAP pair CSV files.")
    parser.add_argument("--arithmetic_dir", required=True)
    parser.add_argument("--tokenizer_name_or_path")
    parser.add_argument("--cache_dir")
    parser.add_argument("--shuffle_seed", type=int, default=1000)
    parser.add_argument("--skip_csv", action="store_true")
    args = parser.parse_args()
    if not args.skip_csv and not args.tokenizer_name_or_path:
        parser.error("--tokenizer_name_or_path is required unless --skip_csv is set")
    return args


def main() -> None:
    args = parse_args()
    arithmetic_dir = Path(args.arithmetic_dir)
    merged_paths = [merge_digit_json(arithmetic_dir, digit) for digit in range(1, 6)]
    if not args.skip_csv:
        tokenizer = load_tokenizer(args.tokenizer_name_or_path, args.cache_dir)
        label_token_ids = build_label_token_ids(tokenizer)
        for path in merged_paths:
            write_eap_csvs(path, label_token_ids, tokenizer, shuffle_seed=args.shuffle_seed)


def merge_digit_json(arithmetic_dir: Path, digit: int) -> Path:
    merged_examples = []
    source_names = []
    canary = None
    for operation in OPERATIONS:
        source_path = arithmetic_dir / f"{digit}_digit_{operation}.json"
        with source_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        source_names.append(data.get("name", source_path.stem))
        if canary is None:
            canary = data.get("canary")
        merged_examples.extend(data["examples"])

    dataset_name = f"{digit}_digit_arithmetic"
    output = {
        "canary": canary,
        "name": dataset_name,
        "description": f"{digit}-digit arithmetic across addition, subtraction, multiplication, and division.",
        "keywords": [
            "mathematics",
            "arithmetic",
            "addition",
            "subtraction",
            "multiplication",
            "division",
            "multiple choice",
        ],
        "preferred_score": "exact_str_match",
        "metrics": ["exact_str_match", "multiple_choice_grade"],
        "output_regex": "[-+]?\\d+",
        "source_datasets": source_names,
        "examples": merged_examples,
    }
    output_path = arithmetic_dir / f"{dataset_name}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {output_path} ({len(merged_examples)} examples)")
    return output_path


def load_tokenizer(tokenizer_name_or_path: str, cache_dir: str | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=cache_dir,
        use_fast=False,
    )


def build_label_token_ids(tokenizer) -> dict[str, int]:
    label_token_ids = {}
    for label in ARITHMETIC_OPTION_LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Option label {label!r} is not a single token: {token_ids}")
        label_token_ids[label] = int(token_ids[0])
    return label_token_ids


def write_eap_csvs(
    merged_json_path: Path,
    label_token_ids: dict[str, int],
    tokenizer,
    shuffle_seed: int,
) -> None:
    dataset_name = merged_json_path.stem
    if dataset_name not in ARITHMETIC_DATASET_NAMES:
        raise ValueError(f"Unexpected merged arithmetic dataset: {dataset_name}")
    with merged_json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    rows = []
    for row_id, example in enumerate(data["examples"], start=1):
        clean = build_arithmetic_prompt(example, shuffle_seed=shuffle_seed)
        corrupted = clean.replace("the correct option", "the first option", 1)
        if clean == corrupted:
            raise ValueError(f"Could not corrupt prompt for row {row_id} in {dataset_name}")
        if len(tokenizer.encode(clean, add_special_tokens=True)) != len(
            tokenizer.encode(corrupted, add_special_tokens=True)
        ):
            raise ValueError(f"Clean/corrupted token length mismatch for row {row_id} in {dataset_name}")
        label = arithmetic_label(example, shuffle_seed=shuffle_seed)
        rows.append(
            {
                "": row_id,
                "clean": clean,
                "corrupted": corrupted,
                "corrupted_hard": "",
                "correct_idx": label_token_ids[label],
                "incorrect_idx": label_token_ids["A"],
            }
        )

    for data_dir in EAP_DATA_DIRS:
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / f"{dataset_name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["", "clean", "corrupted", "corrupted_hard", "correct_idx", "incorrect_idx"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()