"""
capture.py

仅编译模型并捕获 ATen IR 计算图，不执行完整 benchmark。
用于快速查看指定重计算策略下的前向/后向 FX 图结构。

用法::

    cd examples/transformer
    RECOMPUTE='{"6": 0}' python capture.py
    # 或通过 run.sh:
    ./run.sh capture
"""
import json
import os
import sys

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from model import Transformer, device
from aten_recompute.core import CompilerBackend, inject_layer_tags

# ── 输出工具 ──────────────────────────────────────────────────────────────────

_LINE = "─" * 68
_BOLD = "═" * 68

STRATEGY_NAMES = {
    "0": "不重计算",
    "1": "全部重计算",
    "2": "按节点名称关键字",
    "3": "按层步长选择",
    "4": "按比例选择前 N% 层",
    "5": "按 ATen 算子类型",
    "6": "自动廉价重计算（链深度）",
    "7": "min-cut 最优重计算",
}


def _strategy_desc(cfg: dict) -> str:
    if not cfg:
        return "策略 0: 不重计算"
    key, val = next(iter(cfg.items()))
    name = STRATEGY_NAMES.get(str(key), "未知策略")
    param = f", 参数: {val}" if val is not None else ""
    return f"策略 {key}: {name}{param}"


def main():
    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "INFO")

    model_name = os.getenv("MODEL_NAME", "Transformer")
    strategy_config = json.loads(os.getenv("RECOMPUTE", '{"6": 0}'))

    # ── 模型配置 ──────────────────────────────────────────────────────────────
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model        = 512
    num_heads      = 8
    num_layers     = 6
    d_ff           = 2048
    max_seq_length = 100
    dropout        = 0.1
    batch_size     = 4  # 仅用于触发编译，小 batch 即可

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}")
    print("  ATenIR Graph Capture")
    print(_BOLD)
    print(f"  模型:         {model_name} ({num_layers} layers, d_model={d_model})")
    print(f"  设备:         {device}")
    print(f"  重计算策略:   {_strategy_desc(strategy_config)}")
    print(_LINE)

    # ── 构建模型 + 注入层级标签 ───────────────────────────────────────────────
    print("\n[1/3] 构建模型 & 注入层级标签")
    print(_LINE)

    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads,
        num_layers, d_ff, max_seq_length, dropout,
    )
    transformer.to(device)

    enc_layers = [(layer, i) for i, layer in enumerate(transformer.encoder_layers)]
    dec_layers = [
        (layer, len(transformer.encoder_layers) + i)
        for i, layer in enumerate(transformer.decoder_layers)
    ]
    inject_layer_tags(enc_layers + dec_layers)

    print(f"  encoder layers: {len(transformer.encoder_layers)}")
    print(f"  decoder layers: {len(transformer.decoder_layers)}")

    # ── 编译（save_ir=True 自动保存前向/后向 IR）──────────────────────────────
    print(f"\n[2/3] 编译模型 (save_ir=True)")
    print(_LINE)

    backend = CompilerBackend(strategy_config=strategy_config, save_ir=True)
    compiled = torch.compile(transformer, backend=backend, dynamic=True)

    # ── 触发编译：一次前向 + 反向 ────────────────────────────────────────────
    print(f"\n[3/3] 触发编译（1 次前向 + 反向）")
    print(_LINE)

    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length)).to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    transformer.train()
    output = compiled(src_data, tgt_data[:, :-1])
    loss = criterion(
        output.contiguous().view(-1, tgt_vocab_size),
        tgt_data[:, 1:].contiguous().view(-1),
    )
    loss.backward()

    print(f"  loss = {loss.item():.4f}  (仅用于触发编译，数值无意义)")

    # ── 输出信息 ──────────────────────────────────────────────────────────────
    from aten_recompute.utils.save_ir import _default_ir_dir
    ir_dir = _default_ir_dir(model_name)

    print(f"\n{_BOLD}")
    print(f"  完成。IR 文件已保存至:")
    print(f"    {ir_dir}")
    print(f"{_BOLD}\n")


if __name__ == "__main__":
    main()
