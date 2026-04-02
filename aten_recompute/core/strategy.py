"""Shared strategy metadata helpers used by core and examples."""

import json
from typing import Any, Dict, Mapping


STRATEGY_NAMES: Dict[str, str] = {
    "0": "不重计算",
    "1": "全部重计算",
    "2": "按节点名称关键字",
    "3": "按层步长选择",
    "4": "按比例选择前 N% 层",
    "5": "按 ATen 算子类型",
    "6": "自动廉价重计算（链深度）",
    "7": "min-cut 最优重计算",
}


def validate_strategy_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize strategy config mapping.

    The strategy config must contain at most one strategy ID.
    """
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Strategy config must be a mapping, got: {type(cfg).__name__}")

    normalized = {str(k): v for k, v in dict(cfg).items()}
    if not normalized:
        return {}

    if len(normalized) > 1:
        raise ValueError(
            "Strategy config currently supports only one strategy at a time, "
            f"but got: {list(normalized.keys())}"
        )

    key, val = next(iter(normalized.items()))
    if key not in STRATEGY_NAMES:
        raise ValueError(f"Unknown strategy id '{key}', expected one of: {sorted(STRATEGY_NAMES)}")

    if key in {"0", "1"}:
        return normalized

    if key in {"2", "5"}:
        if not isinstance(val, list) or not all(isinstance(item, str) and item for item in val):
            raise ValueError(f"Strategy {key} expects a non-empty string list parameter")
        return normalized

    if key == "3":
        if (
            not isinstance(val, (list, tuple))
            or len(val) != 2
            or not all(isinstance(x, int) for x in val)
            or val[0] < 0
            or val[1] <= 0
        ):
            raise ValueError("Strategy 3 expects [start:int>=0, stride:int>0]")
        return normalized

    if key == "4":
        if not isinstance(val, (int, float)) or not (0.0 <= float(val) <= 1.0):
            raise ValueError("Strategy 4 expects ratio in [0, 1]")
        return normalized

    if key == "6":
        if not isinstance(val, int) or val < 0:
            raise ValueError("Strategy 6 expects non-negative integer max_depth")
        return normalized

    if key == "7":
        if not isinstance(val, (int, float)) or float(val) <= 0:
            raise ValueError("Strategy 7 expects positive memory_budget")
        return normalized

    return normalized


def parse_strategy_config(raw: str | None, default: str = '{"6": 0}') -> Dict[str, Any]:
    """Parse strategy JSON string and validate it.

    If raw is None/empty, default is used.
    """
    text = raw if raw and raw.strip() else default
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid strategy JSON: {text}") from exc
    return validate_strategy_config(parsed)


def describe_strategy(cfg: Dict[str, Any]) -> str:
    """Return a human readable strategy description string."""
    cfg = validate_strategy_config(cfg)
    if not cfg:
        return "策略 0: 不重计算"
    key, val = next(iter(cfg.items()))
    name = STRATEGY_NAMES.get(str(key), "未知策略")
    param = f", 参数: {val}" if val is not None else ""
    return f"策略 {key}: {name}{param}"
