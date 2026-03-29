"""Backward-compatible facade over strict config loading helpers."""

from pathlib import Path
from typing import Dict

from config_strict import (
    apply_env,
    default_input_config_path,
    load_input_config,
    load_model_map,
)


def _default_config_path() -> Path:
    return default_input_config_path()


def load_and_apply_config(path: str | None = None) -> Dict:
    cfg = load_input_config(path or str(_default_config_path()))
    apply_env(cfg)
    return cfg


def load_model_config(path: str | None = None) -> Dict:
    return {"model": load_model_map(path)}
