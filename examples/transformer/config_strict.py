"""Strict config loading/validation for transformer workflows.

This module intentionally avoids silent fallback defaults for required fields.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

_REQUIRED_ENV_KEYS = ("model_name", "recompute", "recompute_log_level")
_REQUIRED_MODEL_KEYS = (
    "src_vocab_size",
    "tgt_vocab_size",
    "d_model",
    "num_heads",
    "num_layers",
    "d_ff",
    "max_seq_length",
    "dropout",
    "padding_idx",
)
_REQUIRED_CAPTURE_KEYS = (
    "mode",
    "batch_size",
    "seq_len",
    "dynamic",
    "static_profile",
    "compare_runtime",
)
_REQUIRED_BENCHMARK_KEYS = ("batch_size", "n_steps")


def default_input_config_path() -> Path:
    return Path(__file__).resolve().parent / "input.yaml"


def default_model_config_path() -> Path:
    return Path(__file__).resolve().parent / "model_config.yaml"


def _load_yaml_required(path: str | Path, label: str) -> Dict[str, Any]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be a mapping object: {p}")
    return data


def _require_mapping(cfg: Dict[str, Any], key: str, where: str) -> Dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{where}.{key} must be a mapping")
    return value


def _require_keys(mapping: Dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"Missing required keys in {where}: {', '.join(missing)}")


def load_input_config(path: str | None = None) -> Dict[str, Any]:
    cfg = _load_yaml_required(path or default_input_config_path(), "input config")
    _require_mapping(cfg, "run", "config")
    _require_mapping(cfg, "env", "config")
    return cfg


def validate_input_for_task(cfg: Dict[str, Any], task: str) -> None:
    run = _require_mapping(cfg, "run", "config")
    env = _require_mapping(cfg, "env", "config")
    _require_keys(env, _REQUIRED_ENV_KEYS, "config.env")

    task_in_cfg = run.get("task")
    if task_in_cfg not in {"capture", "benchmark"}:
        raise ValueError("config.run.task must be one of: capture, benchmark")

    if task == "capture":
        capture = _require_mapping(cfg, "capture", "config")
        _require_keys(capture, _REQUIRED_CAPTURE_KEYS, "config.capture")
    elif task == "benchmark":
        benchmark = _require_mapping(cfg, "benchmark", "config")
        _require_keys(benchmark, _REQUIRED_BENCHMARK_KEYS, "config.benchmark")
    else:
        raise ValueError(f"Unsupported task: {task}")


def apply_env(cfg: Dict[str, Any]) -> None:
    env = _require_mapping(cfg, "env", "config")
    _require_keys(env, _REQUIRED_ENV_KEYS, "config.env")

    os.environ["MODEL_NAME"] = str(env["model_name"])
    os.environ["RECOMPUTE_LOG_LEVEL"] = str(env["recompute_log_level"])
    os.environ["RECOMPUTE"] = json.dumps(env["recompute"], ensure_ascii=False)

    if "project_root" in env:
        os.environ["PROJECT_ROOT"] = str(env["project_root"])
    if "run_id" in env:
        os.environ["RUN_ID"] = str(env["run_id"])
    if "save_joint_graph" in env:
        os.environ["SAVE_JOINT_GRAPH"] = "1" if bool(env["save_joint_graph"]) else "0"


def load_model_map(path: str | None = None) -> Dict[str, Any]:
    cfg = _load_yaml_required(path or default_model_config_path(), "model config")
    model = cfg.get("model", cfg)
    if not isinstance(model, dict):
        raise ValueError("model config must contain a mapping named 'model' or be a mapping itself")
    _require_keys(model, _REQUIRED_MODEL_KEYS, "model config")
    return model
