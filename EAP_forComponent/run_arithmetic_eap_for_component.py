from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "EAP_forComponent" / "run_eap_for_component.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EAP_forComponent sequentially for 1- to 5-digit arithmetic datasets."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tokenizer_name_or_path")
    parser.add_argument("--cache_dir")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Parent directory for per-dataset EAP outputs.",
    )
    parser.add_argument(
        "--task_model",
        action="append",
        required=True,
        type=_parse_task_model,
        metavar="DATASET=MODEL",
        help="Arithmetic dataset and its future model. Repeat for each task.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    for dataset_name, future_model_name_or_path in args.task_model:
        output_dir = output_root / dataset_name
        command = [
            sys.executable,
            str(RUNNER),
            "--model_name_or_path",
            args.model_name_or_path,
            "--dataset_name",
            dataset_name,
            "--localization_mode",
            "future",
            "--future_model_name_or_path",
            future_model_name_or_path,
            "--output_dir",
            str(output_dir),
        ]
        if args.tokenizer_name_or_path:
            command.extend(["--tokenizer_name_or_path", args.tokenizer_name_or_path])
        if args.cache_dir:
            command.extend(["--cache_dir", args.cache_dir])
        print("\n[Arithmetic EAP] " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


def _parse_task_model(value: str) -> tuple[str, str]:
    dataset_name, separator, model_name_or_path = value.partition("=")
    dataset_name = dataset_name.strip()
    model_name_or_path = model_name_or_path.strip()
    if not separator or not dataset_name or not model_name_or_path:
        raise argparse.ArgumentTypeError("Expected DATASET=MODEL")
    return dataset_name, model_name_or_path


if __name__ == "__main__":
    main()