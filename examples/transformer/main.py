"""Unified CLI entry for transformer experiments with YAML config + schema validation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config_loader import (
    apply_env,
    default_config_path,
    default_schema_path,
    get_task_args,
    get_task_from_config,
    load_config,
    validate_config,
)

TASK_TO_SCRIPT = {
    "benchmark": "benchmark.py",
    "train": "train.py",
    "capture": "capture.py",
    "custom-train": "custom_train.py",
    "translate": "translate.py",
    "zh-en-train": "zh_en_train.py",
    "zh-en-translate": "zh_en_translate.py",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Unified transformer runner with YAML config and schema checks."
    )
    parser.add_argument(
        "task",
        nargs="?",
        choices=sorted(TASK_TO_SCRIPT.keys()),
        help="Optional task override. If absent, use run.task from config.",
    )
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--schema",
        default=str(default_schema_path()),
        help="Path to JSON Schema for config validation.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate config and exit.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print resolved command before execution.",
    )
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()

    config_path = Path(args.config).resolve()
    schema_path = Path(args.schema).resolve()

    config = load_config(config_path)
    validate_config(config, schema_path)
    apply_env(config)

    task = args.task or get_task_from_config(config)
    task_args = get_task_args(config, task)

    if args.validate_only:
        print(f"Config valid: {config_path}")
        print(f"Schema used:  {schema_path}")
        print(f"Task:         {task}")
        return 0

    script = TASK_TO_SCRIPT[task]
    script_path = Path(__file__).resolve().parent / script

    cmd = [sys.executable, str(script_path), *task_args, *passthrough]
    if args.print_command:
        print("Resolved command:")
        print(" ".join(cmd))
        print(f"RECOMPUTE={os.environ.get('RECOMPUTE', '{}')}")

    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
