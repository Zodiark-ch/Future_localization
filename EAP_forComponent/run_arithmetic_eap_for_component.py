from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "EAP_forComponent" / "run_eap_for_component.py"
DEFAULT_OUTPUT_ROOT = Path("/home/chenhang/CSAT/files/masks/Future")


TASKS = (
    (
        "1_digit_arithmetic",
        "/home/chenhang/CSAT/files/logs/2026-06-23-21-00-50-135055/probingmodel.pt",
    ),
    (
        "2_digit_arithmetic",
        "/home/chenhang/CSAT/files/logs/2026-06-23-21-00-07-288240/probingmodel.pt",
    ),
    (
        "3_digit_arithmetic",
        "/home/chenhang/CSAT/files/logs/2026-06-23-20-59-28-178801/probingmodel.pt",
    ),
    (
        "4_digit_arithmetic",
        "/home/chenhang/CSAT/files/logs/2026-06-23-20-58-39-372632/probingmodel.pt",
    ),
    (
        "5_digit_arithmetic",
        "/home/chenhang/CSAT/files/logs/2026-06-23-20-56-01-284411/probingmodel.pt",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EAP_forComponent sequentially for 1- to 5-digit arithmetic datasets."
    )
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Parent directory for per-dataset EAP outputs.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run EAP_forComponent/run_eap_for_component.py.",
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
    for dataset_name, future_model_name_or_path in TASKS:
        output_dir = output_root / dataset_name
        command = [
            args.python,
            str(RUNNER),
            "--dataset_name",
            dataset_name,
            "--future_model_name_or_path",
            future_model_name_or_path,
            "--output_dir",
            str(output_dir),
        ]
        print("\n[Arithmetic EAP] " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()