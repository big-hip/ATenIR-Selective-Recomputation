import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark_flow.config import load_benchmark_config
from benchmark_flow.runtime_phase import run_runtime_analysis
from benchmark_flow.static_phase import run_static_analysis
from benchmark_flow.strategy import build_strategy_tag
from benchmark_flow.train_phase import run_train_validation

_LINE = "─" * 68
_BOLD = "═" * 68


def main(config_path: str | None = None):
    _, bcfg, model_map, model_name, strategy_config, methods = load_benchmark_config(config_path)
    _, strat_tag = build_strategy_tag(strategy_config)

    train_ctx = run_train_validation(
        model_map=model_map,
        bcfg=bcfg,
        strategy_config=strategy_config,
        methods=methods,
        model_name=model_name,
        line=_LINE,
        bold=_BOLD,
    )

    static_ctx = run_static_analysis(
        train_ctx=train_ctx,
        methods=methods,
        strategy_config=strategy_config,
        strat_tag=strat_tag,
        model_name=model_name,
        line=_LINE,
    )

    run_runtime_analysis(
        train_ctx=train_ctx,
        static_ctx=static_ctx,
        methods=methods,
        strategy_config=strategy_config,
        strat_tag=strat_tag,
        model_name=model_name,
        line=_LINE,
        bold=_BOLD,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
