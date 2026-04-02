from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkMethod:
    name: str
    family: str
    execution_mode: str
    impl: str
    strategy_config: dict | None = None
    provenance: str = "measured"
    enabled_by_default: bool = True
    diagnostic: bool = False
    comparison_baseline: str | None = None
    semantic_group: str = "main"
    note: str = ""


_MAIN_REGION_GROUP = "region"
_MAIN_GRAPH_GROUP = "graph"
_DIAGNOSTIC_GROUP = "diagnostic"


def _strategy_key(strategy_config: dict | None) -> str:
    if not strategy_config:
        return "0"
    return next(iter(strategy_config), "0")


def build_strategy_tag(strategy_config: dict) -> tuple[str, str]:
    strat_key = _strategy_key(strategy_config)
    return strat_key, f"ATenIR_strat{strat_key}"


def build_primary_atenir_method(strategy_config: dict | None) -> BenchmarkMethod:
    strategy = strategy_config or {"0": None}
    key = _strategy_key(strategy)
    return BenchmarkMethod(
        name=f"ATenIR_strat{key}",
        family="graph",
        execution_mode="compiled",
        impl="atenir",
        strategy_config=strategy,
        comparison_baseline="compiled_no_recompute" if key != "0" else None,
        semantic_group=_MAIN_GRAPH_GROUP,
    )


def _main_region_method(name: str, impl: str) -> BenchmarkMethod:
    return BenchmarkMethod(
        name=name,
        family="region",
        execution_mode="eager",
        impl=impl,
        comparison_baseline=None if name == "eager_baseline" else "eager_baseline",
        semantic_group=_MAIN_REGION_GROUP,
    )


def _main_graph_method(name: str, impl: str, *, strategy_config: dict | None = None) -> BenchmarkMethod:
    return BenchmarkMethod(
        name=name,
        family="graph",
        execution_mode="compiled",
        impl=impl,
        strategy_config=strategy_config,
        comparison_baseline=None if name == "compiled_no_recompute" else "compiled_no_recompute",
        semantic_group=_MAIN_GRAPH_GROUP,
    )


def _diagnostic_graph_method(name: str, impl: str, *, strategy_config: dict | None = None, note: str) -> BenchmarkMethod:
    return BenchmarkMethod(
        name=name,
        family="graph",
        execution_mode="compiled",
        impl=impl,
        strategy_config=strategy_config,
        enabled_by_default=False,
        diagnostic=True,
        comparison_baseline="compiled_no_recompute",
        semantic_group=_DIAGNOSTIC_GROUP,
        note=note,
    )


def _primary_graph_method(primary_strategy: dict | None) -> BenchmarkMethod:
    if _strategy_key(primary_strategy) == "7":
        return _main_graph_method(
            name="ATenIR_strat7_b1.0",
            impl="atenir",
            strategy_config=primary_strategy,
        )
    return build_primary_atenir_method(primary_strategy)


def _dedupe_methods(methods: list[BenchmarkMethod]) -> list[BenchmarkMethod]:
    deduped = []
    seen = set()
    for method in methods:
        key = (
            method.name,
            method.impl,
            json.dumps(method.strategy_config, sort_keys=True)
            if method.strategy_config is not None
            else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(method)
    return deduped


def build_plan_aligned_method_matrix(primary_strategy: dict | None) -> list[BenchmarkMethod]:
    return _dedupe_methods([
        _main_region_method("eager_baseline", "baseline"),
        _main_region_method("eager_checkpoint", "checkpoint"),
        _main_region_method("eager_sac", "sac"),
        _main_graph_method("compiled_no_recompute", "baseline", strategy_config={"0": None}),
        _primary_graph_method(primary_strategy),
    ])


def build_diagnostic_method_matrix(_primary_strategy: dict | None = None) -> list[BenchmarkMethod]:
    return _dedupe_methods([
        _diagnostic_graph_method(
            name="pytorch_graph_min_cut",
            impl="pytorch_min_cut",
            note="partitioner-only diagnostic; 不能当作 PyTorch 官方完整 graph built-in 方法",
        ),
        _diagnostic_graph_method(
            name="ATenIR_strat1",
            impl="atenir",
            strategy_config={"1": 0},
            note="用于 ATenIR 策略诊断，不进入当前主结论",
        ),
        _diagnostic_graph_method(
            name="ATenIR_strat6_d0",
            impl="atenir",
            strategy_config={"6": 0},
            note="用于 ATenIR 策略诊断，不进入当前主结论",
        ),
        _diagnostic_graph_method(
            name="ATenIR_strat6_d1",
            impl="atenir",
            strategy_config={"6": 1},
            note="用于 ATenIR 策略诊断，不进入当前主结论",
        ),
        _diagnostic_graph_method(
            name="ATenIR_strat6_d2",
            impl="atenir",
            strategy_config={"6": 2},
            note="用于 ATenIR 策略诊断，不进入当前主结论",
        ),
        _diagnostic_graph_method(
            name="ATenIR_strat7_b0.75",
            impl="atenir",
            strategy_config={"7": 0.75},
            note="用于 ATenIR 策略诊断，不进入当前主结论",
        ),
        _diagnostic_graph_method(
            name="ATenIR_strat7_b0.5",
            impl="atenir",
            strategy_config={"7": 0.5},
            note="用于 ATenIR 策略诊断，不进入当前主结论",
        ),
    ])


def build_method_matrix(primary_strategy: dict | None) -> list[BenchmarkMethod]:
    return [method for method in build_plan_aligned_method_matrix(primary_strategy) if method.enabled_by_default]


def build_full_method_matrix(primary_strategy: dict | None) -> list[BenchmarkMethod]:
    return _dedupe_methods(
        build_plan_aligned_method_matrix(primary_strategy)
        + build_diagnostic_method_matrix(primary_strategy)
    )


def build_method_index(primary_strategy: dict | None) -> dict[str, BenchmarkMethod]:
    return {method.name: method for method in build_method_matrix(primary_strategy)}


def build_full_method_index(primary_strategy: dict | None) -> dict[str, BenchmarkMethod]:
    return {method.name: method for method in build_full_method_matrix(primary_strategy)}


def default_main_method_names(primary_strategy: dict | None) -> list[str]:
    return [method.name for method in build_method_matrix(primary_strategy)]


def diagnostic_method_names(primary_strategy: dict | None) -> list[str]:
    return [method.name for method in build_diagnostic_method_matrix(primary_strategy)]


def comparison_baseline_name(method: BenchmarkMethod) -> str | None:
    return method.comparison_baseline


def is_diagnostic_method(method: BenchmarkMethod) -> bool:
    return method.diagnostic
