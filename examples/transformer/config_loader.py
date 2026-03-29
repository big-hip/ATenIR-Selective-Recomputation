"""YAML config loader and schema validator for transformer experiments."""

import json
import os
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List

import yaml

_jsonschema = importlib.util.find_spec("jsonschema")
if _jsonschema is not None:
    _jsonschema_mod = importlib.import_module("jsonschema")
    Draft202012Validator = _jsonschema_mod.Draft202012Validator
    _HAS_JSONSCHEMA = True
else:
    Draft202012Validator = None
    _HAS_JSONSCHEMA = False


def _root_dir() -> Path:
    return Path(__file__).resolve().parent


def default_config_path() -> Path:
    return _root_dir() / "configs" / "experiment.yaml"


def default_schema_path() -> Path:
    return _root_dir() / "configs" / "experiment.schema.json"


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping object: {config_path}")
    return data


def validate_config(config: Dict, schema_path: Path) -> None:
    with open(schema_path, "r", encoding="utf-8") as handle:
        schema = json.load(handle)

    if _HAS_JSONSCHEMA:
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(config), key=lambda item: list(item.path))
        if errors:
            lines = ["Config schema validation failed:"]
            for err in errors:
                path = ".".join(str(p) for p in err.path) or "<root>"
                lines.append(f"- {path}: {err.message}")
            raise ValueError("\n".join(lines))
        return

    _fallback_validate_config(config)


def _fallback_validate_config(config: Dict) -> None:
    """Fallback validator when jsonschema package is unavailable."""
    allowed_tasks = {
        "benchmark",
        "train",
        "capture",
        "custom-train",
        "translate",
        "zh-en-train",
        "zh-en-translate",
    }
    allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping object")

    for key in ("version", "env", "run", "tasks"):
        if key not in config:
            raise ValueError(f"Config missing required field: {key}")

    env = config["env"]
    if not isinstance(env, dict):
        raise ValueError("env must be an object")
    for key in ("model_name", "recompute", "recompute_log_level"):
        if key not in env:
            raise ValueError(f"env missing required field: {key}")

    if not isinstance(env["model_name"], str) or not env["model_name"].strip():
        raise ValueError("env.model_name must be a non-empty string")
    if not isinstance(env["recompute"], dict):
        raise ValueError("env.recompute must be an object")
    if str(env["recompute_log_level"]) not in allowed_levels:
        raise ValueError("env.recompute_log_level must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL")

    run = config["run"]
    if not isinstance(run, dict):
        raise ValueError("run must be an object")
    task = run.get("task")
    if task not in allowed_tasks:
        raise ValueError(f"run.task must be one of: {sorted(allowed_tasks)}")

    tasks = config["tasks"]
    if not isinstance(tasks, dict):
        raise ValueError("tasks must be an object")
    for task_name, task_cfg in tasks.items():
        if not isinstance(task_cfg, dict):
            raise ValueError(f"tasks.{task_name} must be an object")
        if "args" not in task_cfg:
            raise ValueError(f"tasks.{task_name} missing required field: args")
        if not isinstance(task_cfg["args"], list) or not all(isinstance(x, str) for x in task_cfg["args"]):
            raise ValueError(f"tasks.{task_name}.args must be a string array")


def apply_env(config: Dict) -> None:
    env = config["env"]
    os.environ["MODEL_NAME"] = str(env["model_name"])
    os.environ["RECOMPUTE_LOG_LEVEL"] = str(env["recompute_log_level"])
    os.environ["RECOMPUTE"] = json.dumps(env["recompute"], ensure_ascii=False)

    project_root = env.get("project_root")
    if project_root:
        root = (_root_dir() / project_root).resolve()
        os.environ["PROJECT_ROOT"] = str(root)


def get_task_from_config(config: Dict) -> str:
    return str(config["run"]["task"])


def get_task_args(config: Dict, task: str) -> List[str]:
    task_cfg = config["tasks"].get(task, {})
    args = task_cfg.get("args", [])
    return [str(item) for item in args]
