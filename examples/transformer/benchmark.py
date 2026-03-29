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


def main():
    # ──────────────────────────────────────────────────────────────────────────
    # 0. 全局配置
    # ──────────────────────────────────────────────────────────────────────────
    os.environ.setdefault("RECOMPUTE_LOG_LEVEL", "INFO")

    import torch
    import torch.nn as nn

    from model import Transformer, device
    from aten_recompute.core import (
        CompilerBackend,
        describe_strategy,
        parse_strategy_config,
    )
    from aten_recompute.analysis import (
        MemoryProfiler, StaticEstimator, FLOPsEstimator, print_method_comparison,
    )
    from aten_recompute.utils import apply_activation_checkpoint
    from meta_pipeline import capture_static_with_retry, inject_transformer_layer_tags

    model_name = os.getenv("MODEL_NAME", "Transformer")

    # 策略配置：优先从环境变量 RECOMPUTE 读取，否则使用默认策略 6（自动廉价）
    strategy_config = parse_strategy_config(os.getenv("RECOMPUTE"))

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
    print(f"  重计算策略:   {describe_strategy(strategy_config)}")
    print(f"  批次大小:     {batch_size}")
    print(f"  序列长度:     {max_seq_length}")
    print(f"  训练步数:     {n_steps}")
    print(_LINE)

    # 注入层级标签（mark_layer 会在 partition_fn 内被分析并清理）
    # 先保存一份无钩子的干净模型副本，用于后续 eager / checkpoint 基准对比。
    # mark_layer 在 eager 模式下会调用 x.clone()，污染基准测试的显存和耗时。
    _clean_model = copy.deepcopy(transformer)

    inject_transformer_layer_tags(transformer)

    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length)).to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer.train()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. 编译（partition_fn 内自动完成重计算策略选择 + mark_layer 清理）
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[阶段 1/5] 编译模型")
    print(_LINE)
    backend = CompilerBackend(strategy_config=strategy_config, save_ir=True)
    compiled_transformer = torch.compile(
        transformer, backend=backend, dynamic=True
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 3. 训练验证
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[阶段 2/5] 训练验证 ({n_steps} 步)")
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
    # 4. 统一图捕获 + 静态分析（显存 + FLOPs + 时间）
    # ──────────────────────────────────────────────────────────────────────────
    # strategy N 的图复用阶段 1 CompilerBackend 已捕获的 fw_gm/bw_gm；
    # strategy 0 仅需一次轻量编译（跳过 Inductor）。
    print(f"\n[阶段 3/5] 静态分析（显存 + FLOPs + 时间）")
    print(_LINE)

    _strat_key = next(iter(strategy_config), "0")
    _strat_tag = f"ATenIR_strat{_strat_key}"

    _module_lists_fn = lambda m: [m.encoder_layers, m.decoder_layers]
    _loss_fn = lambda out: criterion(
        out.contiguous().view(-1, tgt_vocab_size),
        tgt_data[:, 1:].contiguous().view(-1),
    )

    # ── 图捕获 ────────────────────────────────────────────────────────────
    _captured_graphs = {}  # {tag: (fw_gm, bw_gm)}

    # strategy N：复用阶段 1 已编译的 CompilerBackend 捕获的图
    if backend.fw_gm is not None and backend.bw_gm is not None:
        _captured_graphs[_strat_tag] = (backend.fw_gm, backend.bw_gm)
        print(f"  [{_strat_tag}] 复用阶段 1 编译图"
              f"（FW {len(list(backend.fw_gm.graph.nodes))} 节点"
              f" / BW {len(list(backend.bw_gm.graph.nodes))} 节点）")

    # strategy 0：轻量编译（跳过 Inductor，仅捕获图）
    import time as _time
    _t0 = _time.time()

    _model_copy = copy.deepcopy(transformer)
    _model_copy.train()
    inject_transformer_layer_tags(_model_copy)
    sample_inputs = (src_data, tgt_data[:, :-1])
    _nr_backend, _nr_use_decomp, _nr_exec_err = capture_static_with_retry(
        _model_copy,
        sample_inputs,
        _loss_fn,
        strategy_config={"0": None},
        dynamic=True,
        use_meta=True,
    )

    _elapsed = _time.time() - _t0
    if _nr_backend.fw_gm is not None and _nr_backend.bw_gm is not None:
        _captured_graphs["no_recompute"] = (_nr_backend.fw_gm, _nr_backend.bw_gm)
        print(f"  [no_recompute] 轻量编译完成 ({_elapsed:.1f}s)，"
              f"FW {len(list(_nr_backend.fw_gm.graph.nodes))} 节点"
              f" / BW {len(list(_nr_backend.bw_gm.graph.nodes))} 节点")
        print(f"  [no_recompute] decomp 路径: {'on' if _nr_use_decomp else 'off'}")
        if _nr_exec_err is not None:
            print(
                f"  [no_recompute] 执行告警: 捕获成功但执行失败 "
                f"({type(_nr_exec_err).__name__}: {_nr_exec_err})"
            )
    del _model_copy, _nr_backend

    # ── 静态峰值显存估算（复用捕获的图）────────────────────────────────────
    print(f"\n  ── 静态峰值显存估算 ──")
    static_estimator = StaticEstimator()

    for _tag, (_fw, _bw) in _captured_graphs.items():
        static_estimator.estimate_from_graphs(
            _fw, _bw, transformer, optimizer_type="adam", tag=_tag,
        )

    # checkpoint 公式推导（需要 no_recompute 作为 baseline，不需要编译）
    if "no_recompute" in static_estimator._results:
        static_estimator.compare_strategies(
            model=transformer,
            sample_inputs=sample_inputs,
            strategies={"pytorch_ckpt": "checkpoint"},
            module_lists=_module_lists_fn,
            loss_fn=_loss_fn,
            optimizer_type="adam",
        )

    static_estimator.report()
    static_estimator.save_report(model_name=model_name)

    # ── FLOPs & 执行时间估算（复用捕获的图）──────────────────────────────
    print(f"\n  ── FLOPs & 执行时间估算 ──")
    flops_estimator = FLOPsEstimator(device=device)

    for _tag, (_fw, _bw) in _captured_graphs.items():
        flops_estimator.estimate_from_graphs(_fw, _bw, tag=_tag)

    flops_estimator.report()
    flops_estimator.save_report(model_name=model_name)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. 运行时峰值显存 & 耗时对比（同进程内）
    # ──────────────────────────────────────────────────────────────────────────
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if _device != "cuda":
        print(f"\n[阶段 4/5] 跳过显存分析（非 GPU 环境）")
        print(_BOLD)
        return

    print(f"\n[阶段 4/5] 运行时峰值显存 & 耗时对比")
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

    # ── 静态激活对比（修复 static_memory.png）────────────────────────────────
    # 用阶段 3 统一捕获的 strategy 0 和 strategy N 的 FW/BW 图
    if "no_recompute" in _captured_graphs and _strat_tag in _captured_graphs:
        _nr_fw, _nr_bw = _captured_graphs["no_recompute"]
        _rc_fw, _rc_bw = _captured_graphs[_strat_tag]
        analyzer.estimate(
            fw_before=_nr_fw, bw_before=_nr_bw,
            fw_after=_rc_fw, bw_after=_rc_bw,
        )

    # ── 静态 vs 运行时精度对比 ────────────────────────────────────────────
    analyzer.compare_with_static(static_estimator)

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
