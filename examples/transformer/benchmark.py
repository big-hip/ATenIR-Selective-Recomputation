import copy
import os
import sys

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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
    """返回策略的可读描述，如 '策略 6: 自动廉价重计算（链深度）, 参数: 0'"""
    if not cfg:
        return "策略 0: 不重计算"
    key, val = next(iter(cfg.items()))
    name = STRATEGY_NAMES.get(str(key), "未知策略")
    param = f", 参数: {val}" if val is not None else ""
    return f"策略 {key}: {name}{param}"


def main():
    # ──────────────────────────────────────────────────────────────────────────
    # 0. 全局配置
    # ──────────────────────────────────────────────────────────────────────────
    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "INFO")

    import json
    import torch
    import torch.nn as nn

    from model import Transformer, device
    from aten_recompute.core import CompilerBackend, inject_layer_tags
    from aten_recompute.analysis import MemoryProfiler, StaticEstimator, print_method_comparison
    from aten_recompute.utils import apply_activation_checkpoint

    model_name = os.getenv("MODEL_NAME", "Transformer")

    # 策略配置：优先从环境变量 RECOMPUTE 读取，否则使用默认策略 6（自动廉价）
    recompute_env = os.getenv("RECOMPUTE", '{"6": 0}')
    strategy_config = json.loads(recompute_env)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. 初始化模型与数据
    # ──────────────────────────────────────────────────────────────────────────
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model        = 512
    num_heads      = 8
    num_layers     = 6
    d_ff           = 2048
    max_seq_length = 100
    dropout        = 0.1
    batch_size     = 64
    n_steps        = 10

    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads,
        num_layers, d_ff, max_seq_length, dropout,
    )
    transformer.to(device)

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}")
    print("  ATenIR Selective Recomputation")
    print(_BOLD)
    print(f"  模型:         {model_name} ({num_layers} layers, d_model={d_model})")
    print(f"  设备:         {device}")
    print(f"  重计算策略:   {_strategy_desc(strategy_config)}")
    print(f"  批次大小:     {batch_size}")
    print(f"  序列长度:     {max_seq_length}")
    print(f"  训练步数:     {n_steps}")
    print(_LINE)

    # 注入层级标签（mark_layer 会在 partition_fn 内被分析并清理）
    # 先保存一份无钩子的干净模型副本，用于后续 eager / checkpoint 基准对比。
    # mark_layer 在 eager 模式下会调用 x.clone()，污染基准测试的显存和耗时。
    _clean_model = copy.deepcopy(transformer)

    _enc_layers = [(layer, i) for i, layer in enumerate(transformer.encoder_layers)]
    _dec_layers = [
        (layer, len(transformer.encoder_layers) + i)
        for i, layer in enumerate(transformer.decoder_layers)
    ]
    _tag_handles = inject_layer_tags(_enc_layers + _dec_layers)

    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length)).to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer.train()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. 编译（partition_fn 内自动完成重计算策略选择 + mark_layer 清理）
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[阶段 1/4] 编译模型")
    print(_LINE)
    backend = CompilerBackend(strategy_config=strategy_config, save_ir=True)
    compiled_transformer = torch.compile(
        transformer, backend=backend, dynamic=True
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. 训练验证
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[阶段 2/4] 训练验证 ({n_steps} 步)")
    print(_LINE)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9
    )

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

    print(f"{_LINE}")
    print(f"  训练验证完成")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. 静态峰值显存估算（基于 FX 图 FakeTensor 元信息，无需 GPU 运行）
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[阶段 3/4] 静态峰值显存估算")
    print(_LINE)

    # 构造当前策略的标签名
    _strat_key = next(iter(strategy_config), "0")
    _strat_tag = f"ATenIR_strat{_strat_key}"

    static_estimator = StaticEstimator()
    static_estimator.compare_strategies(
        model=transformer,
        sample_inputs=(src_data, tgt_data[:, :-1]),
        strategies={
            "no_recompute":  {"0": None},
            _strat_tag:      strategy_config,
            "pytorch_ckpt":  "checkpoint",
        },
        module_lists=lambda m: [m.encoder_layers, m.decoder_layers],
        loss_fn=lambda out: criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        ),
        optimizer_type="adam",
    )
    static_estimator.report()
    static_estimator.save_report(model_name=model_name)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. 运行时峰值显存 & 耗时对比（同进程内）
    # ──────────────────────────────────────────────────────────────────────────
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if _device != "cuda":
        print(f"\n[阶段 4/4] 跳过显存分析（非 GPU 环境）")
        print(_BOLD)
        return

    print(f"\n[阶段 4/4] 运行时峰值显存 & 耗时对比")
    print(_LINE)
    analyzer = MemoryProfiler(device=_device)
    analyzer.estimate_parameter_memory(transformer)

    def _gpu_cleanup():
        """清理 GPU 缓存，确保各组 profile 基底一致。"""
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(_device)

    # ── (a) eager baseline：无编译、无 checkpoint ─────────────────────────────
    # 使用 _clean_model（无 mark_layer 钩子），确保基准不含 clone() 开销
    eager_model = copy.deepcopy(_clean_model)
    eager_model.to(device)
    eager_model.train()
    _eager_opt = torch.optim.Adam(eager_model.parameters(), lr=1e-4)

    def _eager_forward():
        out = eager_model(src_data, tgt_data[:, :-1])
        return criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

    analyzer.profile_step("eager_baseline",
                          forward_fn=_eager_forward, optimizer=_eager_opt)
    del eager_model, _eager_opt
    _gpu_cleanup()

    # ── (b) compiled baseline：编译但不重计算 ────────────────────────────────
    # 用 strategy 0（不重计算）走完整编译流水线，隔离 torch.compile 加速效果
    # 对比: compiled_no_recompute vs eager = 纯编译加速
    #        ATenIR_recompute vs compiled_no_recompute = 纯重计算效果
    compiled_nr_model = copy.deepcopy(_clean_model)
    compiled_nr_model.to(device)
    compiled_nr_model.train()
    _nr_backend = CompilerBackend(strategy_config={"0": None}, save_ir=False)
    compiled_nr = torch.compile(compiled_nr_model, backend=_nr_backend, dynamic=True)
    _nr_opt = torch.optim.Adam(compiled_nr_model.parameters(), lr=1e-4)

    def _compiled_nr_forward():
        out = compiled_nr(src_data, tgt_data[:, :-1])
        return criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

    analyzer.profile_step("compiled_no_recompute",
                          forward_fn=_compiled_nr_forward, optimizer=_nr_opt)
    del compiled_nr_model, compiled_nr, _nr_backend, _nr_opt
    _gpu_cleanup()

    # ── (c) PyTorch checkpoint：eager + activation checkpoint ────────────────
    # 同样使用 _clean_model，避免 mark_layer 钩子污染
    ckpt_model = copy.deepcopy(_clean_model)
    ckpt_model.to(device)
    ckpt_model.train()
    apply_activation_checkpoint(
        ckpt_model,
        module_lists=[ckpt_model.encoder_layers, ckpt_model.decoder_layers],
        use_reentrant=False,
    )
    _ckpt_opt = torch.optim.Adam(ckpt_model.parameters(), lr=1e-4)

    def _ckpt_forward():
        out = ckpt_model(src_data, tgt_data[:, :-1])
        return criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

    analyzer.profile_step("pytorch_checkpoint",
                          forward_fn=_ckpt_forward, optimizer=_ckpt_opt)
    del ckpt_model, _ckpt_opt
    _gpu_cleanup()

    # ── (d) PyTorch SAC：选择性激活检查点 ────────────────────────────────────
    # SAC 是 ATenIR Strategy 6 的最直接竞品：
    #   - 都做"保存昂贵算子输出、重计算廉价算子"
    #   - SAC 通过 policy_fn + torch.compile 实现
    #   - ATenIR 通过 partition_fn 在 AOT 图分割阶段实现
    try:
        import functools
        from torch.utils.checkpoint import (
            checkpoint as torch_checkpoint,
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        # 定义 SAC 策略：保存计算密集算子，重计算廉价算子
        _COMPUTE_OPS = {
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
        }
        # 尝试添加 flash/efficient attention（部分平台可能不支持）
        for _attn_op in (
            "torch.ops.aten._scaled_dot_product_flash_attention.default",
            "torch.ops.aten._scaled_dot_product_efficient_attention.default",
        ):
            try:
                _COMPUTE_OPS.add(eval(_attn_op))
            except (AttributeError, Exception):
                pass

        def _sac_policy(ctx, op, *args, **kwargs):
            if op in _COMPUTE_OPS:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        _sac_context_fn = functools.partial(
            create_selective_checkpoint_contexts, _sac_policy
        )

        sac_model = copy.deepcopy(_clean_model)
        sac_model.to(device)
        sac_model.train()

        # 包装每层的 forward 使用 SAC
        for _ml in [sac_model.encoder_layers, sac_model.decoder_layers]:
            for _layer in _ml:
                _orig_forward = _layer.forward

                def _make_sac_forward(orig_fwd, ctx_fn):
                    def _sac_forward(*args, **kwargs):
                        return torch_checkpoint(
                            orig_fwd, *args,
                            use_reentrant=False,
                            context_fn=ctx_fn,
                            **kwargs,
                        )
                    return _sac_forward

                _layer.forward = _make_sac_forward(_orig_forward, _sac_context_fn)

        # SAC 需要 torch.compile 才能生效
        compiled_sac = torch.compile(sac_model, dynamic=True)
        _sac_opt = torch.optim.Adam(sac_model.parameters(), lr=1e-4)

        def _sac_forward():
            out = compiled_sac(src_data, tgt_data[:, :-1])
            return criterion(
                out.contiguous().view(-1, tgt_vocab_size),
                tgt_data[:, 1:].contiguous().view(-1),
            )

        analyzer.profile_step("pytorch_SAC",
                              forward_fn=_sac_forward, optimizer=_sac_opt)
        del sac_model, compiled_sac, _sac_opt
        _gpu_cleanup()

    except (ImportError, Exception) as _sac_err:
        print(f"  [pytorch_SAC] 跳过：当前 PyTorch 版本不支持 SAC ({_sac_err})")

    # ── (e) 当前 ATenIR 策略（compiled_transformer）──────────────────────────
    _prof_opt = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    def _recomputed_forward():
        out = compiled_transformer(src_data, tgt_data[:, :-1])
        return criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

    analyzer.profile_step("ATenIR_recompute",
                          forward_fn=_recomputed_forward, optimizer=_prof_opt)

    # ── 报告与保存 ───────────────────────────────────────────────────────────
    analyzer.report()
    report_path = analyzer.save_report(model_name=model_name)

    # ── 方法理论对比 ──────────────────────────────────────────────────────
    print_method_comparison()

    print(_BOLD)
    print(f"  完成。报告已保存至: {report_path}")
    print(f"{_BOLD}\n")


if __name__ == "__main__":
    main()
