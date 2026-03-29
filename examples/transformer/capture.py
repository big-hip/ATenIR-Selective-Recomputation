"""
capture.py

仅编译模型并捕获 ATen IR 计算图，不执行完整 benchmark。
用于快速查看指定重计算策略下的前向/后向 FX 图结构。

用法::

    cd examples/transformer
    RECOMPUTE='{"6": 0}' python capture.py
    python capture.py --mode static
    # 或通过 run.sh:
    ./run.sh capture
"""
import argparse
import os
import sys
import time

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from model import Transformer, device
from aten_recompute.core import (
    CompilerBackend,
    describe_strategy,
    parse_strategy_config,
)
from aten_recompute.utils.save_ir import _default_ir_dir
from meta_pipeline import (
    compare_graph_signatures,
    inject_transformer_layer_tags,
    run_train_step,
)

# ── 输出工具 ──────────────────────────────────────────────────────────────────

_LINE = "─" * 68
_BOLD = "═" * 68


def main():
    parser = argparse.ArgumentParser(description="ATenIR graph capture helper")
    parser.add_argument(
        "--mode",
        choices=["runtime", "static"],
        default="runtime",
        help="runtime: 真实设备编译+反向；static: Fake/meta 静态捕获（不跑真实 CUDA）",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="启用 torch.compile dynamic=True（默认关闭，便于图稳定对比）",
    )
    parser.add_argument(
        "--static-profile",
        choices=["high", "fast"],
        default="high",
        help="static 模式配置: high=训练态更接近真实图，fast=eval 更快更稳",
    )
    parser.add_argument(
        "--compare-runtime",
        action="store_true",
        help="static 模式下额外运行一次 runtime 捕获并做图签名对照（更慢）",
    )
    args = parser.parse_args()

    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "INFO")
    if args.mode == "static":
        os.environ.setdefault("SAVE_JOINT_GRAPH", "1")

    model_name = os.getenv("MODEL_NAME", "Transformer")
    strategy_config = parse_strategy_config(os.getenv("RECOMPUTE"))

    # ── 模型配置 ──────────────────────────────────────────────────────────────
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model        = 512
    num_heads      = 8
    num_layers     = 6
    d_ff           = 2048
    max_seq_length = args.seq_len
    dropout        = 0.1
    batch_size     = args.batch_size  # 仅用于触发编译，小 batch 即可
    run_device     = torch.device("cpu") if args.mode == "static" else device

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}")
    print("  ATenIR Graph Capture")
    print(_BOLD)
    print(f"  捕获模式:     {args.mode}")
    print(f"  模型:         {model_name} ({num_layers} layers, d_model={d_model})")
    print(f"  设备:         {run_device}")
    print(f"  重计算策略:   {describe_strategy(strategy_config)}")
    print(_LINE)

    # ── 构建模型 + 注入层级标签 ───────────────────────────────────────────────
    print("\n[1/3] 构建模型 & 注入层级标签")
    print(_LINE)

    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads,
        num_layers, d_ff, max_seq_length, dropout,
    )
    transformer.to(run_device)

    inject_transformer_layer_tags(transformer)

    print(f"  encoder layers: {len(transformer.encoder_layers)}")
    print(f"  decoder layers: {len(transformer.decoder_layers)}")

    # ── 编译（save_ir=True 自动保存前向/后向 IR）──────────────────────────────
    print(f"\n[2/3] 编译模型")
    print(_LINE)

    backend = CompilerBackend(
        strategy_config=strategy_config,
        save_ir=True,
        mode=args.mode,
        use_meta=(args.mode == "static"),
        use_decomp=True,
    )
    compiled = torch.compile(transformer, backend=backend, dynamic=args.dynamic)

    # ── 触发编译：一次前向 + 反向 ────────────────────────────────────────────
    print(f"\n[3/3] 触发编译（1 次前向 + 反向）")
    print(_LINE)

    torch.manual_seed(0)
    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length)).to(run_device)
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length)).to(run_device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # high: 训练态图更接近真实训练；fast: eval 更快更稳但与训练图有偏差。
    if args.mode == "static" and args.static_profile == "fast":
        transformer.eval()
    else:
        transformer.train()
    t0 = time.time()
    loss_value = None
    static_exec_error = None
    fallback_used = False
    try:
        loss_value = run_train_step(compiled, src_data, tgt_data, tgt_vocab_size, criterion)
    except Exception as exc:
        if args.mode != "static":
            raise

        static_exec_error = str(exc).splitlines()[0]
        # high 模式若因随机相关 custom op 限制失败，自动回退到 fast 模式。
        if (backend.fw_gm is None or backend.bw_gm is None) and args.static_profile == "high":
            from torch import _dynamo
            _dynamo.reset()
            transformer.eval()
            backend = CompilerBackend(
                strategy_config=strategy_config,
                save_ir=True,
                mode="static",
                use_meta=True,
                use_decomp=True,
            )
            compiled = torch.compile(transformer, backend=backend, dynamic=args.dynamic)
            loss_value = run_train_step(compiled, src_data, tgt_data, tgt_vocab_size, criterion)
            fallback_used = True
        elif backend.fw_gm is None or backend.bw_gm is None:
            raise
    elapsed = time.time() - t0

    if args.mode == "static":
        print("  static 捕获后端: CompilerBackend(mode='static', use_meta=True)")
        print(f"  static 仿真配置: {args.static_profile}")
        if fallback_used:
            print("  static 回退路径: high 失败后自动切到 fast")
        if static_exec_error:
            print("  static 执行状态: 图已捕获，跳过执行")
            print(f"  跳过原因: {static_exec_error}")

    if loss_value is not None:
        print(f"  loss = {loss_value:.4f}  (仅用于触发编译，数值无意义)")
    else:
        print("  loss = N/A  (static 模式仅用于捕获图)")
    print(f"  触发耗时: {elapsed:.2f}s")

    # ── 输出信息 ──────────────────────────────────────────────────────────────
    ir_dir = _default_ir_dir(model_name)

    print(f"\n{_BOLD}")
    print(f"  完成。IR 文件已保存至:")
    print(f"    {ir_dir}")
    print(f"{_BOLD}\n")

    if args.mode == "static" and args.compare_runtime and torch.cuda.is_available():
        print(f"{_BOLD}")
        print("  static 与 runtime 图签名对照")
        print(f"{_BOLD}")
        rt_model = Transformer(
            src_vocab_size, tgt_vocab_size, d_model, num_heads,
            num_layers, d_ff, max_seq_length, dropout,
        ).to(device)
        inject_transformer_layer_tags(rt_model)
        rt_backend = CompilerBackend(
            strategy_config=strategy_config,
            save_ir=False,
            mode="runtime",
            use_meta=False,
            use_decomp=True,
        )
        rt_compiled = torch.compile(rt_model, backend=rt_backend, dynamic=args.dynamic)

        rt_src = src_data.to(device)
        rt_tgt = tgt_data.to(device)
        rt_model.train()
        run_train_step(rt_compiled, rt_src, rt_tgt, tgt_vocab_size, criterion)

        fw_cmp = compare_graph_signatures(backend.fw_gm, rt_backend.fw_gm)
        bw_cmp = compare_graph_signatures(backend.bw_gm, rt_backend.bw_gm)
        print(
            f"  [FW] meta/runtime 节点数: {fw_cmp['meta_nodes']}/{fw_cmp['runtime_nodes']} "
            f"(ratio={fw_cmp['ratio']:.3f})"
        )
        print(f"  [FW] 算子集合重叠: {fw_cmp['overlap']:.1%}")
        if fw_cmp["top_diffs"]:
            print("  [FW] Top-5 算子计数差异 (meta - runtime):")
            for item in fw_cmp["top_diffs"]:
                print(f"    {item['op']}: {item['delta']:+d}")

        print(
            f"  [BW] meta/runtime 节点数: {bw_cmp['meta_nodes']}/{bw_cmp['runtime_nodes']} "
            f"(ratio={bw_cmp['ratio']:.3f})"
        )
        print(f"  [BW] 算子集合重叠: {bw_cmp['overlap']:.1%}")
        if bw_cmp["top_diffs"]:
            print("  [BW] Top-5 算子计数差异 (meta - runtime):")
            for item in bw_cmp["top_diffs"]:
                print(f"    {item['op']}: {item['delta']:+d}")


if __name__ == "__main__":
    main()
