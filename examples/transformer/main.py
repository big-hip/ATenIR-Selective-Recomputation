"""Single entrypoint for transformer workflow with strict YAML validation."""

import argparse

from config_strict import (
    apply_env,
    default_input_config_path,
    load_input_config,
    validate_input_for_task,
)


def _resolve_task(config: dict, cli_task: str | None) -> str:
    if cli_task:
        return cli_task
    run = config.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("config.run must be a mapping")
    task = run.get("task")
    if task not in {"benchmark", "capture"}:
        raise ValueError("config.run.task must be one of: benchmark, capture")
    return task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer single workflow entry")
    parser.add_argument(
        "task",
        nargs="?",
        choices=["benchmark", "capture"],
        help="Main workflow task to run (optional; defaults to YAML run.task)",
    )
    parser.add_argument(
        "--config",
        default=str(default_input_config_path()),
        help="Path to YAML input file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_input_config(args.config)
    task = _resolve_task(config, args.task)
    validate_input_for_task(config, task)
    apply_env(config)

    if task == "benchmark":
        from benchmark_flow.runner import main as benchmark_main

        benchmark_main(config_path=args.config)
        return 0

    from capture_flow.runner import main as capture_main

    capture_main(config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
