import copy
import time as _time


def run_static_analysis(train_ctx: dict, strategy_config: dict, strat_tag: str, model_name: str, line: str):
    from aten_recompute.analysis import FLOPsEstimator, StaticEstimator
    from meta_pipeline import capture_static_with_retry, inject_transformer_layer_tags

    transformer = train_ctx["transformer"]
    backend = train_ctx["backend"]
    criterion = train_ctx["criterion"]
    src_data = train_ctx["src_data"]
    tgt_data = train_ctx["tgt_data"]
    tgt_vocab_size = train_ctx["tgt_vocab_size"]
    torch = train_ctx["torch"]

    print(f"\n[阶段 3/5] 静态分析（显存 + FLOPs + 时间）")
    print(line)

    module_lists_fn = lambda m: [m.encoder_layers, m.decoder_layers]
    loss_fn = lambda out: criterion(
        out.contiguous().view(-1, tgt_vocab_size),
        tgt_data[:, 1:].contiguous().view(-1),
    )

    captured_graphs = {}

    if backend.fw_gm is not None and backend.bw_gm is not None:
        captured_graphs[strat_tag] = (backend.fw_gm, backend.bw_gm)
        print(
            f"  [{strat_tag}] 复用阶段 1 编译图"
            f"（FW {len(list(backend.fw_gm.graph.nodes))} 节点"
            f" / BW {len(list(backend.bw_gm.graph.nodes))} 节点）"
        )

    t0 = _time.time()
    model_copy = copy.deepcopy(transformer)
    model_copy.train()
    inject_transformer_layer_tags(model_copy)
    sample_inputs = (src_data, tgt_data[:, :-1])
    nr_backend, nr_use_decomp, nr_exec_err = capture_static_with_retry(
        model_copy,
        sample_inputs,
        loss_fn,
        strategy_config={"0": None},
        dynamic=True,
        use_meta=True,
    )

    # PyTorch 2.6 静态路径下偶发 inductor seed alias 问题：改为 eval() 后重试。
    if nr_exec_err is not None and "inductor_lookup_seed" in str(nr_exec_err):
        print("  [no_recompute] 检测到 inductor_lookup_seed 执行告警，切换 eval 模式重试...")
        model_copy.eval()
        nr_backend, nr_use_decomp, nr_exec_err = capture_static_with_retry(
            model_copy,
            sample_inputs,
            loss_fn,
            strategy_config={"0": None},
            dynamic=True,
            use_meta=True,
        )

    elapsed = _time.time() - t0
    if nr_backend.fw_gm is not None and nr_backend.bw_gm is not None:
        captured_graphs["no_recompute"] = (nr_backend.fw_gm, nr_backend.bw_gm)
        print(
            f"  [no_recompute] 轻量编译完成 ({elapsed:.1f}s)，"
            f"FW {len(list(nr_backend.fw_gm.graph.nodes))} 节点"
            f" / BW {len(list(nr_backend.bw_gm.graph.nodes))} 节点"
        )
        print(f"  [no_recompute] decomp 路径: {'on' if nr_use_decomp else 'off'}")
        if nr_exec_err is not None:
            print(
                f"  [no_recompute] 执行告警: 捕获成功但执行失败 "
                f"({type(nr_exec_err).__name__}: {nr_exec_err})"
            )

    del model_copy, nr_backend

    print(f"\n  ── 静态峰值显存估算 ──")
    static_estimator = StaticEstimator()
    for tag, (fw, bw) in captured_graphs.items():
        static_estimator.estimate_from_graphs(
            fw, bw, transformer, optimizer_type="adam", tag=tag
        )

    if "no_recompute" in static_estimator._results:
        static_estimator.compare_strategies(
            model=transformer,
            sample_inputs=sample_inputs,
            strategies={"pytorch_ckpt": "checkpoint"},
            module_lists=module_lists_fn,
            loss_fn=loss_fn,
            optimizer_type="adam",
        )

    static_estimator.report()
    static_estimator.save_report(model_name=model_name)

    print(f"\n  ── FLOPs & 执行时间估算 ──")
    flops_estimator = FLOPsEstimator(device=train_ctx["device"])
    for tag, (fw, bw) in captured_graphs.items():
        flops_estimator.estimate_from_graphs(fw, bw, tag=tag)
    flops_estimator.report()
    flops_estimator.save_report(model_name=model_name)

    return {
        "captured_graphs": captured_graphs,
        "static_estimator": static_estimator,
        "sample_inputs": sample_inputs,
        "module_lists_fn": module_lists_fn,
        "loss_fn": loss_fn,
    }
