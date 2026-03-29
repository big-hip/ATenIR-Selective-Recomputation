"""
capture_flow/runner.py

内部模块：由 main.py 调用，执行 IR 捕获流程。
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from capture_flow.analysis_phase import run_runtime_signature_compare
from capture_flow.capture_phase import execute_capture
from capture_flow.config import default_config_path_str, resolve_capture_inputs
from model import device as default_device
from aten_recompute.core import describe_strategy, parse_strategy_config
from aten_recompute.utils.save_ir import _default_ir_dir

_LINE = "─" * 68
_BOLD = "═" * 68


def main(config_path: str | None = None):
    parser = argparse.ArgumentParser(description="ATenIR graph capture helper")
    parser.add_argument(
        "--config",
        default=config_path or default_config_path_str(),
        help="Path to input YAML file",
    )
    args = parser.parse_args()

    _, model_map = resolve_capture_inputs(args)

    model_name = os.environ["MODEL_NAME"]
    strategy_config = parse_strategy_config(os.environ["RECOMPUTE"])
    num_layers = int(model_map["num_layers"])
    d_model = int(model_map["d_model"])

    run_device = torch.device("cpu") if args.mode == "static" else default_device

    print(f"\n{_BOLD}")
    print("  ATenIR Graph Capture")
    print(_BOLD)
    print(f"  捕获模式:     {args.mode}")
    print(f"  模型:         {model_name} ({num_layers} layers, d_model={d_model})")
    print(f"  设备:         {run_device}")
    print(f"  重计算策略:   {describe_strategy(strategy_config)}")
    print(_LINE)

    print("\n[1/3] 构建+编译+触发捕获")
    print(_LINE)
    ctx = execute_capture(
        args=args,
        model_map=model_map,
        strategy_config=strategy_config,
        run_device=run_device,
    )

    if args.mode == "static":
        print("  static 捕获后端: CompilerBackend(mode='static', use_meta=True)")
        print(f"  static 仿真配置: {args.static_profile}")
        if ctx["fallback_used"]:
            print("  static 回退路径: high 失败后自动切到 fast")
        if ctx["static_exec_error"]:
            print("  static 执行状态: 图已捕获，跳过执行")
            print(f"  跳过原因: {ctx['static_exec_error']}")

    if ctx["loss_value"] is not None:
        print(f"  loss = {ctx['loss_value']:.4f}  (仅用于触发编译，数值无意义)")
    else:
        print("  loss = N/A  (static 模式仅用于捕获图)")
    print(f"  触发耗时: {ctx['elapsed']:.2f}s")

    ir_dir = _default_ir_dir(model_name)
    print(f"\n{_BOLD}")
    print("  完成。IR 文件已保存至:")
    print(f"    {ir_dir}")
    print(f"{_BOLD}\n")

    print("[2/3] （可选）static/runtime 图签名对照")
    print(_LINE)
    run_runtime_signature_compare(
        args=args,
        ctx=ctx,
        model_map=model_map,
        strategy_config=strategy_config,
    )

    print("\n[3/3] 结束")
    print(_LINE)
