import os

from config_strict import apply_env, load_input_config, load_model_map, validate_input_for_task
from aten_recompute.core import parse_strategy_config
from benchmark_flow.strategy import build_method_matrix


def load_benchmark_config(config_path: str | None = None):
    cfg = load_input_config(config_path)
    validate_input_for_task(cfg, "benchmark")
    apply_env(cfg)

    bcfg = cfg["benchmark"]
    model_map = load_model_map()
    model_name = os.environ["MODEL_NAME"]
    strategy_config = parse_strategy_config(os.environ["RECOMPUTE"])
    methods = build_method_matrix(strategy_config)

    return cfg, bcfg, model_map, model_name, strategy_config, methods
