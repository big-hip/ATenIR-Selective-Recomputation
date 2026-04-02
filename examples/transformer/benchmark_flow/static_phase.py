import copy
import time as _time


def _static_context(train_ctx):
    transformer = train_ctx["transformer"]
    criterion = train_ctx["criterion"]
    src_data = train_ctx["src_data"]
    tgt_data = train_ctx["tgt_data"]
    tgt_vocab_size = train_ctx["tgt_vocab_size"]

    return {
        "transformer": transformer,
        "backend": train_ctx["backend"],
        "sample_inputs": (src_data, tgt_data[:, :-1]),
        "loss_fn": lambda out: criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        ),
        "module_lists_fn": lambda m: [m.encoder_layers, m.decoder_layers],
    }


# ── Graph capture helpers ───────────────────────────────────────────────────

def _graph_methods(methods):
    return [
        method for method in methods
        if method.execution_mode == "compiled" and method.impl in {"baseline", "atenir"}
    ]


def _primary_graph_tag(graph_methods, strategy_config, strat_tag):
    primary_method = next(
        (
            method for method in graph_methods
            if method.impl == "atenir" and (method.strategy_config or strategy_config) == strategy_config
        ),
        None,
    )
    return primary_method.name if primary_method is not None else strat_tag


def _capture_graph_for_method(transformer, sample_inputs, loss_fn, method):
    from meta_pipeline import capture_static_with_retry, inject_transformer_layer_tags

    model_copy = copy.deepcopy(transformer)
    model_copy.train()
    inject_transformer_layer_tags(model_copy)
    backend, use_decomp, exec_err = capture_static_with_retry(
        model_copy,
        sample_inputs,
        loss_fn,
        strategy_config=method.strategy_config or {"0": None},
        dynamic=True,
        use_meta=True,
    )
    return model_copy, backend, use_decomp, exec_err


def _collect_captured_graphs(static_ctx, methods, strategy_config, strat_tag, line):
    backend = static_ctx["backend"]
    transformer = static_ctx["transformer"]
    sample_inputs = static_ctx["sample_inputs"]
    loss_fn = static_ctx["loss_fn"]

    captured_graphs = {}
    graph_methods = _graph_methods(methods)
    primary_tag = _primary_graph_tag(graph_methods, strategy_config, strat_tag)

    if backend.fw_gm is not None and backend.bw_gm is not None:
        captured_graphs[primary_tag] = (backend.fw_gm, backend.bw_gm)
        print(
            f"  [{primary_tag}] 复用阶段 1 编译图"
            f"（FW {len(list(backend.fw_gm.graph.nodes))} 节点"
            f" / BW {len(list(backend.bw_gm.graph.nodes))} 节点）"
        )
        if primary_tag != strat_tag:
            print(f"  [{strat_tag}] 已映射到主方法标签 [{primary_tag}]，避免静态报告重复展示。")

    seen_compile_cfg = set()
    if backend.fw_gm is not None and backend.bw_gm is not None:
        seen_compile_cfg.add(tuple(sorted((strategy_config or {"0": None}).items())))

    for method in graph_methods:
        cfg = method.strategy_config or {"0": None}
        cfg_key = tuple(sorted(cfg.items()))
        if method.name in captured_graphs or cfg_key in seen_compile_cfg:
            continue
        seen_compile_cfg.add(cfg_key)

        t0 = _time.time()
        model_copy, method_backend, method_use_decomp, method_exec_err = _capture_graph_for_method(
            transformer,
            sample_inputs,
            loss_fn,
            method,
        )
        elapsed = _time.time() - t0

        if method_backend.fw_gm is None or method_backend.bw_gm is None:
            print(f"  [{method.name}] 图捕获失败，跳过静态估算。")
            del model_copy, method_backend
            continue

        captured_graphs[method.name] = (method_backend.fw_gm, method_backend.bw_gm)
        print(
            f"  [{method.name}] 补充捕获完成 ({elapsed:.1f}s)，"
            f"FW {len(list(method_backend.fw_gm.graph.nodes))} 节点"
            f" / BW {len(list(method_backend.bw_gm.graph.nodes))} 节点"
        )
        print(f"  [{method.name}] decomp 路径: {'on' if method_use_decomp else 'off'}")
        if method_exec_err is not None:
            print(f"  [{method.name}] 执行告警: {type(method_exec_err).__name__}: {method_exec_err}")
        del model_copy, method_backend

    ordered_graphs = {}
    if "compiled_no_recompute" in captured_graphs:
        ordered_graphs["compiled_no_recompute"] = captured_graphs["compiled_no_recompute"]
    for method in graph_methods:
        if method.name == "compiled_no_recompute":
            continue
        if method.name in captured_graphs:
            ordered_graphs[method.name] = captured_graphs[method.name]
    for tag, graphs in captured_graphs.items():
        if tag not in ordered_graphs:
            ordered_graphs[tag] = graphs
    return ordered_graphs


# ── Result annotation / estimation helpers ──────────────────────────────────

def _annotate_estimate_result(result, method):
    if method is not None:
        result["family"] = method.family
        result["execution_mode"] = method.execution_mode
        result["impl"] = method.impl
        result["method_name"] = method.name
    else:
        result.setdefault("family", "graph")
        result.setdefault("execution_mode", "compiled")
        result.setdefault("impl", "atenir")
        result.setdefault("method_name", result.get("tag", "unknown"))


def _run_static_estimator(static_ctx, methods, captured_graphs, model_name):
    from aten_recompute.analysis import StaticEstimator

    transformer = static_ctx["transformer"]
    sample_inputs = static_ctx["sample_inputs"]
    module_lists_fn = static_ctx["module_lists_fn"]
    loss_fn = static_ctx["loss_fn"]

    method_index = {method.name: method for method in methods}
    static_estimator = StaticEstimator()

    print(f"\n  ── 静态峰值显存估算 ──")
    for tag, (fw, bw) in captured_graphs.items():
        result = static_estimator.estimate_from_graphs(
            fw, bw, transformer, optimizer_type="adam", tag=tag
        )
        _annotate_estimate_result(result, method_index.get(tag))

    region_methods = [method for method in methods if method.impl == "checkpoint"]
    checkpoint_specs = {method.name: "checkpoint" for method in region_methods}
    if checkpoint_specs and "compiled_no_recompute" in static_estimator._results:
        static_estimator.compare_strategies(
            model=transformer,
            sample_inputs=sample_inputs,
            strategies=checkpoint_specs,
            module_lists=module_lists_fn,
            loss_fn=loss_fn,
            optimizer_type="adam",
        )
        for method in region_methods:
            result = static_estimator._results.get(method.name)
            if result is None:
                continue
            _annotate_estimate_result(result, method)
            result.setdefault("provenance", "formula_derived")
    elif checkpoint_specs:
        print("  checkpoint 静态推导跳过：缺少 compiled_no_recompute 基线。")

    for tag, result in static_estimator._results.items():
        print(
            f"  [{tag}] provenance={result.get('provenance', 'unknown')}"
            f" | family={result.get('family', 'unknown')}"
            f" | mode={result.get('execution_mode', 'unknown')}"
        )

    static_estimator.report()
    static_estimator.save_report(model_name=model_name)
    return static_estimator


def _run_flops_estimator(train_ctx, captured_graphs, model_name):
    from aten_recompute.analysis import FLOPsEstimator

    print(f"\n  ── FLOPs & 执行时间估算 ──")
    flops_estimator = FLOPsEstimator(device=train_ctx["device"])
    for tag, (fw, bw) in captured_graphs.items():
        flops_estimator.estimate_from_graphs(fw, bw, tag=tag)
    flops_estimator.report()
    flops_estimator.save_report(model_name=model_name)
    return flops_estimator


# ── Public entrypoint ────────────────────────────────────────────────────────

def run_static_analysis(
    train_ctx: dict,
    methods: list,
    strategy_config: dict,
    strat_tag: str,
    model_name: str,
    line: str,
):
    print(f"\n[阶段 3/5] 静态分析（显存 + FLOPs + 时间）")
    print(line)

    static_ctx = _static_context(train_ctx)
    captured_graphs = _collect_captured_graphs(static_ctx, methods, strategy_config, strat_tag, line)
    static_estimator = _run_static_estimator(static_ctx, methods, captured_graphs, model_name)
    _run_flops_estimator(train_ctx, captured_graphs, model_name)

    return {
        "captured_graphs": captured_graphs,
        "static_estimator": static_estimator,
        "sample_inputs": static_ctx["sample_inputs"],
        "module_lists_fn": static_ctx["module_lists_fn"],
        "loss_fn": static_ctx["loss_fn"],
    }
