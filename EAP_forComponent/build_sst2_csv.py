from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_OUTPUTS = (
    "EAP_forComponent/data/sst2.csv",
    "EAP_forLogicalCircuit/data/sst2.csv",
)
CLEAN_PREFIX = "<s> Is the sentiment of following sentence positive or negative?"
CORRUPTED_PREFIX = "<s> Is the sentiment of following sentence exist or none?"
ANSWER_SUFFIX = "\nAnswer: It is"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build balanced SST2 CSV files for EAP pair attribution.")
    parser.add_argument("--model_name_or_path", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--cache_dir", default="/home/chenhang/CSAT/.cache")
    parser.add_argument("--dataset_cache_dir", default="./.cache/data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_paths", default=",".join(DEFAULT_OUTPUTS))
    parser.add_argument("--examples_per_label", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wrong_margin",
        type=float,
        default=0.0,
        help="Require opposite sentiment logit minus correct sentiment logit to be at least this value.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_bfloat16", type=_str_to_bool, default=True)
    parser.add_argument(
        "--disable_model_filter",
        action="store_true",
        help="Do not require Mistral to choose the wrong positive/negative label. Intended for debugging only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer_name = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.cache_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    positive_id = _single_token_id(tokenizer, "positive")
    negative_id = _single_token_id(tokenizer, "negative")
    exist_id = _single_token_id(tokenizer, "exist")
    label_token_ids = {1: positive_id, 0: negative_id}
    opposite_token_ids = {1: negative_id, 0: positive_id}

    model = None
    device = torch.device(args.device if torch.cuda.is_available() and not args.device.startswith("cpu") else "cpu")
    if not args.disable_model_filter:
        dtype = torch.bfloat16 if args.use_bfloat16 and device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            torch_dtype=dtype,
            device_map={"": device.index or 0} if device.type == "cuda" else "cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.config.use_cache = False

    dataset = load_dataset(
        "stanfordnlp/sst2",
        split=args.split,
        cache_dir=args.dataset_cache_dir,
    ).shuffle(seed=args.seed)

    selected: dict[int, list[dict[str, object]]] = {0: [], 1: []}
    candidates: list[dict[str, object]] = []
    stats = {
        "seen": 0,
        "token_length_mismatch": 0,
        "model_correct": 0,
        "selected_positive": 0,
        "selected_negative": 0,
    }

    for example in dataset:
        label = int(example["label"])
        if label not in selected or len(selected[label]) >= args.examples_per_label:
            continue
        stats["seen"] += 1
        row = _candidate_row(
            tokenizer=tokenizer,
            sentence=str(example["sentence"]),
            label=label,
            correct_idx=label_token_ids[label],
            incorrect_idx=exist_id,
        )
        if row is None:
            stats["token_length_mismatch"] += 1
            continue
        candidates.append(row)
        if len(candidates) >= args.batch_size:
            _consume_candidates(
                candidates=candidates,
                selected=selected,
                stats=stats,
                model=model,
                tokenizer=tokenizer,
                device=device,
                opposite_token_ids=opposite_token_ids,
                examples_per_label=args.examples_per_label,
                wrong_margin=args.wrong_margin,
            )
            candidates.clear()
        if all(len(items) >= args.examples_per_label for items in selected.values()):
            break

    if candidates:
        _consume_candidates(
            candidates=candidates,
            selected=selected,
            stats=stats,
            model=model,
            tokenizer=tokenizer,
            device=device,
            opposite_token_ids=opposite_token_ids,
            examples_per_label=args.examples_per_label,
            wrong_margin=args.wrong_margin,
        )

    missing = {label: args.examples_per_label - len(rows) for label, rows in selected.items() if len(rows) < args.examples_per_label}
    if missing:
        raise RuntimeError(f"Not enough Mistral-wrong SST2 examples selected: {missing}; stats={stats}")

    rows = []
    for row_idx in range(args.examples_per_label):
        rows.append(selected[1][row_idx])
        rows.append(selected[0][row_idx])
    for idx, row in enumerate(rows, start=1):
        row[""] = idx

    output_paths = [Path(item.strip()) for item in args.output_paths.split(",") if item.strip()]
    for output_path in output_paths:
        _write_csv(output_path, rows)
        print(f"Wrote {len(rows)} rows to {output_path}")
    print(
        {
            **stats,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "exist_id": exist_id,
            "selected_positive": len(selected[1]),
            "selected_negative": len(selected[0]),
        }
    )


def _candidate_row(tokenizer, sentence: str, label: int, correct_idx: int, incorrect_idx: int) -> dict[str, object] | None:
    sentence = sentence.strip()
    clean = f"{CLEAN_PREFIX}{sentence}{ANSWER_SUFFIX}"
    corrupted = f"{CORRUPTED_PREFIX}{sentence}{ANSWER_SUFFIX}"
    clean_ids = tokenizer.encode(clean, add_special_tokens=False)
    corrupted_ids = tokenizer.encode(corrupted, add_special_tokens=False)
    if len(clean_ids) != len(corrupted_ids):
        return None
    return {
        "": 0,
        "clean": clean,
        "corrupted": corrupted,
        "corrupted_hard": "",
        "correct_idx": int(correct_idx),
        "incorrect_idx": int(incorrect_idx),
        "label": int(label),
    }


def _consume_candidates(
    candidates: list[dict[str, object]],
    selected: dict[int, list[dict[str, object]]],
    stats: dict[str, int],
    model,
    tokenizer,
    device: torch.device,
    opposite_token_ids: dict[int, int],
    examples_per_label: int,
    wrong_margin: float,
) -> None:
    keep_flags = [True] * len(candidates)
    if model is not None:
        prompts = [str(candidate["clean"]) for candidate in candidates]
        encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoded, use_cache=False)
        positions = encoded["attention_mask"].long().sum(dim=1).sub(1).clamp_min(0)
        rows = torch.arange(len(candidates), device=device)
        next_logits = outputs.logits[rows, positions]
        for idx, candidate in enumerate(candidates):
            label = int(candidate["label"])
            correct_logit = float(next_logits[idx, int(candidate["correct_idx"])].item())
            opposite_logit = float(next_logits[idx, opposite_token_ids[label]].item())
            keep_flags[idx] = opposite_logit - correct_logit >= wrong_margin
            if not keep_flags[idx]:
                stats["model_correct"] += 1

    for candidate, keep in zip(candidates, keep_flags):
        label = int(candidate.pop("label"))
        if not keep or len(selected[label]) >= examples_per_label:
            continue
        selected[label].append(candidate)
        if label == 1:
            stats["selected_positive"] += 1
        else:
            stats["selected_negative"] += 1


def _single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected {text!r} to be a single token, got {token_ids}")
    return int(token_ids[0])


def _write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["", "clean", "corrupted", "corrupted_hard", "correct_idx", "incorrect_idx"])
        writer.writeheader()
        writer.writerows(rows)


def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)