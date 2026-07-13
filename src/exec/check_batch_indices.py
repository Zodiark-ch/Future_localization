#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os
import sys

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer


_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_EXEC_DIR, ".."))


if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from dataset import get_dataset  # noqa: E402


def _prepare_cache_dir(cache_dir: str | None) -> str | None:

    if cache_dir is None:
        return None
    s = cache_dir.strip()
    if not s:
        return None
    if "你的" in s:
        print(
            "错误: --cache_dir 里出现了占位符「你的」，请省略该参数或提供本机可写目录。"
        )
        sys.exit(2)
    path = os.path.abspath(os.path.expanduser(s))
    try:
        os.makedirs(path, mode=0o755, exist_ok=True)
    except OSError as e:
        print(
            f"错误: 无法创建或使用 cache 目录:\n  {path}\n原因: {e}\n"
            "请检查路径是否拼错、是否有写权限；或直接省略 --cache_dir 使用默认缓存。"
        )
        sys.exit(2)
    if not os.access(path, os.W_OK):
        print(f"错误: cache 目录不可写: {path}")
        sys.exit(2)
    return path


def _build_dataset_names(forget: str, retain: str) -> dict:
    if "," in retain:
        retain_list = [x.strip() for x in retain.split(",")]
        return {"forget": forget, "retain": retain_list}
    return {"forget": forget, "retain": retain}


def _collate_like_eval(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


def _violations_1d(
    flat: torch.Tensor,
    upper: int,
    ignore: frozenset[int],
):

    if flat.numel() == 0:
        return 0, []
    mask = torch.ones(flat.shape[0], dtype=torch.bool)
    for v in ignore:
        mask &= flat.ne(v)
    bad = mask & ((flat < 0) | (flat >= upper))
    cnt = int(bad.sum().item())
    if cnt == 0:
        return 0, []
    idx = bad.nonzero(as_tuple=True)[0][:20]
    samples = []
    for j in idx.tolist():
        samples.append((int(j), int(flat[j].item())))
    return cnt, samples


def check_batch(
    batch: dict,
    *,
    config_vocab: int,
    tokenizer_len: int,
    ignore_label: int = -100,
):

    input_ids = batch["input_ids"]
    labels = batch["label"]

    ignore_f = frozenset({ignore_label})


    flat_in = input_ids.reshape(-1)
    cnt_neg_in = int((flat_in < 0).sum().item())
    cnt_ge_tok = int((flat_in >= tokenizer_len).sum().item())
    cnt_ge_cfg = int((flat_in >= config_vocab).sum().item())

    # causal loss: shift_labels = labels[:, 1:]
    shift_labels = labels[:, 1:].contiguous().reshape(-1)
    cnt_lbl, ex_lbl_cfg = _violations_1d(shift_labels, config_vocab, ignore_f)
    cnt_lbl_tok, ex_lbl_tok = _violations_1d(shift_labels, tokenizer_len, ignore_f)

    return {
        "input_ids_neg": cnt_neg_in,
        "input_ids_ge_config_vocab": cnt_ge_cfg,
        "input_ids_ge_len_tokenizer": cnt_ge_tok,
        "shift_labels_bad_vs_config_vocab": cnt_lbl,
        "shift_labels_bad_vs_len_tokenizer": cnt_lbl_tok,
        "examples_shift_labels_config": ex_lbl_cfg,
        "examples_shift_labels_tokenizer": ex_lbl_tok,
    }


def main():
    parser = argparse.ArgumentParser(description="检查 batch 中 token/label 是否越界")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="可选缓存目录；省略时使用 Transformers 默认缓存。",
    )
    parser.add_argument("--forget_dataset_name", type=str, default="WMDPCyber")
    parser.add_argument("--retain_dataset_name", type=str, default="IOI,gender")
    parser.add_argument("--dataset_seed", type=int, default=1000)
    parser.add_argument(
        "--forget_ratio",
        type=float,
        default=400,
        help="与 get_dataset / fastargs 一致",
    )
    parser.add_argument("--self_retain", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="每个数据集最多检查多少个 batch，-1 表示全部",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="test",
        help="要扫的数据集：test | downstream | all",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="不访问 Hugging Face Hub，仅从本地 cache 读取（离线/内网机常用）",
    )
    args = parser.parse_args()

    if_llama = "llama" in args.model_name.lower()

    _raw_cache = args.cache_dir
    if isinstance(_raw_cache, str) and not _raw_cache.strip():
        cache_dir = None
    else:
        cache_dir = _prepare_cache_dir(_raw_cache)

    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    if cache_dir is not None:
        print("cache_dir:", cache_dir)
    else:
        print("cache_dir: Transformers default")
    print()

    load_kw = {}
    if cache_dir is not None:
        load_kw["cache_dir"] = cache_dir
    if args.local_files_only:
        load_kw["local_files_only"] = True
    config = AutoConfig.from_pretrained(args.model_name, **load_kw)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=False, **load_kw
    )
    if tokenizer.pad_token_id is None:
        if if_llama:
            tokenizer.add_special_tokens({"pad_token": "[pad]"})
        else:
            tokenizer.pad_token = tokenizer.eos_token

    config_vocab = int(config.vocab_size)
    tok_len = len(tokenizer)
    print(f"model_name: {args.model_name}")
    print(f"config.vocab_size: {config_vocab}")
    print(f"len(tokenizer):    {tok_len}")
    if tok_len != config_vocab:
        print(
            "[提示] len(tokenizer) != config.vocab_size。"
            "若训练里只对 model.resize_token_embeddings(len(tokenizer)) 而 loss 仍用旧 vocab_size，"
            "可能导致 cross_entropy 越界；以实际加载模型后的 model.config.vocab_size 为准再确认一次。"
        )
    print()

    dataset_names = _build_dataset_names(
        args.forget_dataset_name, args.retain_dataset_name
    )
    _unlearn_ds, test_datasets, _uc, _tc, downstream_datasets = get_dataset(
        dataset_names,
        tokenizer,
        args.dataset_seed,
        args.forget_ratio,
        bool(args.self_retain),
        if_llama,
    )

    to_scan: list[tuple[str, object]] = []
    if args.datasets in ("test", "all"):
        if isinstance(test_datasets, dict):
            for k, ds in test_datasets.items():
                if ds is not None:
                    to_scan.append((f"test_datasets[{k}]", ds))
        elif test_datasets is not None:
            to_scan.append(("test_datasets", test_datasets))
    if args.datasets in ("downstream", "all"):
        for k, ds in downstream_datasets.items():
            if ds is not None:
                to_scan.append((k, ds))

    if not to_scan:
        print("没有可扫描的数据集（均为 None 或未选）。")
        return 1

    any_fail = False
    for name, ds in to_scan:
        print("=" * 60)
        print(f"数据集: {name}  len={len(ds)}")
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate_like_eval,
        )
        bi = 0
        ds_fail = False
        for batch in loader:
            stats = check_batch(
                batch,
                config_vocab=config_vocab,
                tokenizer_len=tok_len,
            )
            bad = (
                stats["input_ids_neg"] > 0
                or stats["input_ids_ge_config_vocab"] > 0
                or stats["input_ids_ge_len_tokenizer"] > 0
                or stats["shift_labels_bad_vs_config_vocab"] > 0
                or stats["shift_labels_bad_vs_len_tokenizer"] > 0
            )
            if bad:
                ds_fail = True
                any_fail = True
                print(f"  --- batch {bi} ---")
                print(f"  input_ids shape: {batch['input_ids'].shape}")
                if stats["input_ids_neg"]:
                    print(f"  input_ids < 0 的数量: {stats['input_ids_neg']}")
                if stats["input_ids_ge_len_tokenizer"]:
                    print(
                        f"  input_ids >= len(tokenizer) 的数量: {stats['input_ids_ge_len_tokenizer']}"
                    )
                if stats["input_ids_ge_config_vocab"]:
                    print(
                        f"  input_ids >= config.vocab_size 的数量: {stats['input_ids_ge_config_vocab']}"
                    )
                if stats["shift_labels_bad_vs_config_vocab"]:
                    print(
                        f"  shift_labels 相对 config.vocab_size 越界数量: {stats['shift_labels_bad_vs_config_vocab']}"
                    )
                    print(
                        f"    示例 (flat_idx, value): {stats['examples_shift_labels_config']}"
                    )
                if stats["shift_labels_bad_vs_len_tokenizer"]:
                    print(
                        f"  shift_labels 相对 len(tokenizer) 越界数量: {stats['shift_labels_bad_vs_len_tokenizer']}"
                    )
                    print(
                        f"    示例 (flat_idx, value): {stats['examples_shift_labels_tokenizer']}"
                    )
            bi += 1
            if args.max_batches >= 0 and bi >= args.max_batches:
                break
        if not ds_fail:
            print(f"  [{name}] 已检查 {bi} 个 batch，未发现越界（相对上述两个上界）。")
        else:
            print(
                f"  [{name}] 至少一个 batch 有问题；已扫描到 batch 索引 0..{bi - 1}"
                f"（可能因 max_batches 提前结束）。"
            )

    print("=" * 60)
    if any_fail:
        print("结论: 发现越界或异常 id，与 CUDA cross_entropy assert 一致时请优先修数据或对齐 tokenizer/model。")
        return 1
    print("结论: 抽样范围内未发现明显越界。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
