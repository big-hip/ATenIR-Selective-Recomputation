import copy
import os
import sys

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # ──────────────────────────────────────────────────────────────────────────
    # 0. 全局配置
    # ──────────────────────────────────────────────────────────────────────────
    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "INFO")

    import json
    import torch
    import torch.nn as nn

    from Transformer import Transformer, device
    from aten_recompute.get_Aten_IR.Graph_compile_capture import CompilerBackend
    from aten_recompute.core.Tag import inject_layer_tags
    from aten_recompute.utils import MemoryAnalyzer, apply_activation_checkpoint

    model_name = os.getenv("MODEL_NAME", "Transformer")

    # 策略配置：优先从环境变量 RECOMPUTE 读取，否则使用默认策略 6（自动廉价）
    recompute_env = os.getenv("RECOMPUTE", '{"6": 0}')
    strategy_config = json.loads(recompute_env)
    print(f"[配置] 重计算策略: {strategy_config}")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. 初始化模型与数据
    # ──────────────────────────────────────────────────────────────────────────
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model        = 512
    num_heads      = 8
    num_layers     = 1
    d_ff           = 2048
    max_seq_length = 100
    dropout        = 0.1

    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads,
        num_layers, d_ff, max_seq_length, dropout,
    )
    transformer.to(device)

    # 注入层级标签（mark_layer 会在 partition_fn 内被分析并清理）
    _enc_layers = [(layer, i) for i, layer in enumerate(transformer.encoder_layers)]
    _dec_layers = [
        (layer, len(transformer.encoder_layers) + i)
        for i, layer in enumerate(transformer.decoder_layers)
    ]
    _tag_handles = inject_layer_tags(_enc_layers + _dec_layers)

    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length)).to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer.train()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. 编译（partition_fn 内自动完成重计算策略选择 + mark_layer 清理）
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/3] 编译模型（含选择性重计算 partition）...")
    backend = CompilerBackend(strategy_config=strategy_config, save_ir=True)
    compiled_transformer = torch.compile(
        transformer, backend=backend, dynamic=True
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. 训练验证
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/3] 训练验证（10 步）...")
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9
    )

    n_steps = 10
    for step in range(n_steps):
        optimizer.zero_grad()
        output = compiled_transformer(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()

        if step % 2 == 0 or step == n_steps - 1:
            print(f"  step {step:3d} | loss = {loss.item():.4f}")

    print("训练验证完成！")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. 运行时峰值显存 & 耗时对比（同进程内）
    # ──────────────────────────────────────────────────────────────────────────
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if _device != "cuda":
        print("\n[3/3] 跳过 CUDA 内存分析（非 GPU 环境）。")
        return

    print("\n[3/3] 开始运行时峰值显存 & 耗时对比...")
    analyzer = MemoryAnalyzer(device=_device)
    analyzer.estimate_parameter_memory(transformer)

    def _gpu_cleanup():
        """清理 GPU 缓存，确保各组 profile 基底一致。"""
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(_device)

    # ── (a) eager baseline：无编译、无 checkpoint ─────────────────────────────
    eager_model = copy.deepcopy(transformer)
    eager_model.to(device)
    eager_model.train()
    _eager_opt = torch.optim.Adam(eager_model.parameters(), lr=1e-4)

    def _eager_step():
        _eager_opt.zero_grad()
        out = eager_model(src_data, tgt_data[:, :-1])
        l = criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        l.backward()
        _eager_opt.step()

    analyzer.profile_step("eager_baseline", fn=_eager_step, warmup=2, steps=5)
    del eager_model, _eager_opt
    _gpu_cleanup()

    # ── (b) PyTorch checkpoint：eager + activation checkpoint ────────────────
    ckpt_model = copy.deepcopy(transformer)
    ckpt_model.to(device)
    ckpt_model.train()
    apply_activation_checkpoint(
        ckpt_model,
        module_lists=[ckpt_model.encoder_layers, ckpt_model.decoder_layers],
        use_reentrant=False,
    )
    _ckpt_opt = torch.optim.Adam(ckpt_model.parameters(), lr=1e-4)

    def _ckpt_step():
        _ckpt_opt.zero_grad()
        out = ckpt_model(src_data, tgt_data[:, :-1])
        l = criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        l.backward()
        _ckpt_opt.step()

    analyzer.profile_step("pytorch_checkpoint", fn=_ckpt_step, warmup=2, steps=5)
    del ckpt_model, _ckpt_opt
    _gpu_cleanup()

    # ── (c) 当前 ATenIR 策略（compiled_transformer）──────────────────────────
    _prof_opt = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    def _recomputed_step():
        _prof_opt.zero_grad()
        out = compiled_transformer(src_data, tgt_data[:, :-1])
        l = criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        l.backward()
        _prof_opt.step()

    analyzer.profile_step("ATenIR_recompute", fn=_recomputed_step, warmup=2, steps=5)

    # ── 报告与保存 ───────────────────────────────────────────────────────────
    analyzer.report()
    analyzer.save_report(model_name=model_name)
    print("内存分析报告已保存。")


if __name__ == "__main__":
    main()
