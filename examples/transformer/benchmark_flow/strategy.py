def build_strategy_tag(strategy_config: dict) -> tuple[str, str]:
    strat_key = next(iter(strategy_config), "0")
    strat_tag = f"ATenIR_strat{strat_key}"
    return strat_key, strat_tag
