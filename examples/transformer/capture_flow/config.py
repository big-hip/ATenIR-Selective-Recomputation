from config_strict import (
    apply_env,
    default_input_config_path,
    load_input_config,
    load_model_map,
    validate_input_for_task,
)


def resolve_capture_inputs(args):
    cfg = load_input_config(args.config)
    validate_input_for_task(cfg, "capture")
    apply_env(cfg)

    model_map = load_model_map()
    capture_cfg = cfg["capture"]

    # capture 运行参数仅允许来自 YAML，禁用 CLI 覆盖。
    args.mode = str(capture_cfg["mode"])
    args.batch_size = int(capture_cfg["batch_size"])
    args.seq_len = int(capture_cfg["seq_len"])
    args.static_profile = str(capture_cfg["static_profile"])
    args.dynamic = bool(capture_cfg["dynamic"])
    args.compare_runtime = bool(capture_cfg["compare_runtime"])

    return cfg, model_map


def default_config_path_str() -> str:
    return str(default_input_config_path())
