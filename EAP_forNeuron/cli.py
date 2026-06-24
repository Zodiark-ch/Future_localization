from __future__ import annotations

import argparse

from EAP_forNeuron.schemas import DEFAULT_CACHE_DIR, DEFAULT_TARGET_MODULES, EAPNeuronConfig
from EAP_forNeuron.datasets import SUPPORTED_DATASET_NAMES
from EAP_forNeuron.runner import EAPForNeuronRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine a CSAT gradient mask with parameter-level EAP attribution.")
    parser.add_argument("--model_name_or_path", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--tokenizer_name_or_path", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--mask_path",  default="/home/chenhang/CSAT/files/masks/gradient/IOI/with_0.2.pt")
    parser.add_argument("--output_dir",  default="/home/chenhang/CSAT/files/masks/Future/IOI/with_0.2.pt")
    parser.add_argument("--dataset_name", choices=SUPPORTED_DATASET_NAMES, default="ioi_mistral")
    parser.add_argument("--data_path", default="/ssd_users/chenhang/CSAT/EAP_forNeuron/data/ioi_mistral.csv")
    parser.add_argument("--corruption_column", default="corrupted")
    parser.add_argument("--metric", choices=["task_loss", "logit_diff"], default="task_loss")
    parser.add_argument("--output_ratio", type=float, default=0.1)
    parser.add_argument("--ratio_base", choices=["all", "candidate"], default="all")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--score_abs", type=_str_to_bool, default=True)
    parser.add_argument("--score_token_mode", choices=["label_position", "all_active"], default="label_position")
    parser.add_argument("--target_modules", default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--include_lm_head", action="store_true")
    parser.add_argument("--include_embed_tokens", action="store_true")
    parser.add_argument("--unsupported_policy", choices=["drop", "keep", "error"], default="drop")
    parser.add_argument("--score_dtype", choices=["float32", "float16", "bfloat16", "fp32", "fp16", "bf16"], default="float32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_bfloat16", type=_str_to_bool, default=True)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--save_scores", action="store_true")
    parser.add_argument("--row_chunk_size", type=int, default=256)
    parser.add_argument("--max_concat_candidates", type=int, default=50_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EAPNeuronConfig(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        mask_path=args.mask_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        corruption_column=args.corruption_column,
        metric=args.metric,
        output_ratio=args.output_ratio,
        ratio_base=args.ratio_base,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        score_abs=args.score_abs,
        score_token_mode=args.score_token_mode,
        target_modules=tuple(item.strip() for item in args.target_modules.split(",") if item.strip()),
        include_lm_head=args.include_lm_head,
        include_embed_tokens=args.include_embed_tokens,
        unsupported_policy=args.unsupported_policy,
        score_dtype=args.score_dtype,
        device=args.device,
        use_bfloat16=args.use_bfloat16,
        use_cpu=args.use_cpu,
        cache_dir=args.cache_dir,
        save_scores=args.save_scores,
        row_chunk_size=args.row_chunk_size,
        max_concat_candidates=args.max_concat_candidates,
    )
    EAPForNeuronRunner(config).run()


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
    main()
