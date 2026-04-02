import copy
import functools


def _fmt_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1 << 30):.2f} GB"


def _classify(runtime_delta: float, time_delta: float) -> str:
    if runtime_delta > 0 and time_delta <= 0:
        return "runtime 改善"
    if runtime_delta > 0 and time_delta > 0:
        return "省显存但更慢"
    if runtime_delta <= 0 and time_delta > 0:
        return "更慢且未省显存"
    return "无明显收益"


def _build_sac_policy(torch_module):
    compute_ops = {
        torch_module.ops.aten.mm.default,
        torch_module.ops.aten.bmm.default,
        torch_module.ops.aten.addmm.default,
    }

    def _policy(ctx, op, *args, **kwargs):
        if op in compute_ops:
            return torch_module.utils.checkpoint.CheckpointPolicy.MUST_SAVE
        return torch_module.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

    return _policy


def _wrap_modulelist_checkpoint(module_lists, checkpoint_fn, *, use_reentrant=False, context_fn=None):
    for module_list in module_lists:
        for module in module_list:
            original_forward = module.forward

            def _make_forward(orig_forward):
                def _wrapped(*args, **kwargs):
                    call_kwargs = dict(kwargs)
                    if context_fn is not None:
                        call_kwargs["context_fn"] = context_fn
                    return checkpoint_fn(
                        orig_forward,
                        *args,
                        use_reentrant=use_reentrant,
                        **call_kwargs,
                    )

                return _wrapped

            module.forward = _make_forward(original_forward)


class _PyTorchMinCutBackend:
    def __init__(self):
        self.fw_gm = None
        self.bw_gm = None

    def __call__(self, gm, sample_inputs):
        import torch
        from torch._functorch.aot_autograd import aot_module_simplified
        from torch._functorch.partitioners import min_cut_rematerialization_partition
        from torch._guards import detect_fake_mode
        from torch._inductor.compile_fx import compile_fx_inner
        from torch._inductor.decomposition import select_decomp_table
        from torch._inductor.virtualized import V

        def fw_compiler(fw_gm, fw_inputs):
            self.fw_gm = fw_gm
            return compile_fx_inner(fw_gm, fw_inputs)

        def bw_compiler(bw_gm, bw_inputs):
            self.bw_gm = bw_gm
            return compile_fx_inner(bw_gm, bw_inputs, is_backward=True)

        fake_mode = detect_fake_mode(sample_inputs)
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)

        with V.set_fake_mode(fake_mode):
            return aot_module_simplified(
                gm,
                sample_inputs,
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=min_cut_rematerialization_partition,
                decompositions=select_decomp_table(),
            )


def _describe_method(method):
    if method.impl == "atenir":
        from aten_recompute.core import describe_strategy

        return describe_strategy(method.strategy_config)
    if method.impl == "checkpoint":
        return "PyTorch eager activation checkpoint"
    if method.impl == "sac":
        return "PyTorch eager SAC"
    if method.impl == "pytorch_min_cut":
        return "PyTorch graph min-cut rematerialization"
    if method.execution_mode == "compiled":
        return "compiled baseline"
    return "eager baseline"


def _set_dropout_rate(module, dropout_p: float) -> None:
    import torch.nn as nn

    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.p = dropout_p
        if isinstance(submodule, nn.MultiheadAttention):
            submodule.dropout = dropout_p


def _has_active_dropout(module) -> bool:
    import torch.nn as nn

    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout) and float(submodule.p) > 0:
            return True
        if isinstance(submodule, nn.MultiheadAttention) and float(submodule.dropout) > 0:
            return True
    return False


def _snapshot_named_params(model):
    return {
        name: param.detach().float().clone()
        for name, param in model.named_parameters()
    }


def _grad_absmax(model) -> float:
    max_val = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        val = float(param.grad.detach().abs().max().item())
        if val > max_val:
            max_val = val
    return max_val


def _param_update_absmax(before, model) -> float:
    max_val = 0.0
    for name, param in model.named_parameters():
        delta = (param.detach().float() - before[name]).abs().max().item()
        if delta > max_val:
            max_val = float(delta)
    return max_val


def _runtime_context(train_ctx):
    return {
        "clean_model": train_ctx["clean_model"],
        "criterion": train_ctx["criterion"],
        "src_data": train_ctx["src_data"],
        "tgt_data": train_ctx["tgt_data"],
        "tgt_vocab_size": train_ctx["tgt_vocab_size"],
        "device": train_ctx["device"],
    }


# ── Runtime method construction ─────────────────────────────────────────────

def _build_atenir_runtime_model(model, method):
    import torch

    from aten_recompute.core import CompilerBackend
    from meta_pipeline import inject_transformer_layer_tags

    inject_transformer_layer_tags(model)
    backend = CompilerBackend(strategy_config=method.strategy_config, save_ir=False)
    return torch.compile(model, backend=backend, dynamic=True)


def _build_pytorch_min_cut_runtime_model(model):
    import torch

    backend = _PyTorchMinCutBackend()
    return torch.compile(model, backend=backend, dynamic=True)


def _build_checkpoint_runtime_model(model):
    from aten_recompute.utils.checkpoint import apply_activation_checkpoint

    apply_activation_checkpoint(
        model,
        module_lists=[model.encoder_layers, model.decoder_layers],
        use_reentrant=False,
    )
    return model


def _build_sac_runtime_model(model):
    import torch
    from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

    sac_policy = _build_sac_policy(torch)
    sac_context_fn = functools.partial(create_selective_checkpoint_contexts, sac_policy)
    _wrap_modulelist_checkpoint(
        [model.encoder_layers, model.decoder_layers],
        checkpoint,
        use_reentrant=False,
        context_fn=sac_context_fn,
    )
    return model


def _build_callable_runtime_model(model, method):
    import torch

    if method.impl == "atenir":
        return _build_atenir_runtime_model(model, method)
    if method.impl == "pytorch_min_cut":
        return _build_pytorch_min_cut_runtime_model(model)
    if method.impl == "checkpoint":
        return _build_checkpoint_runtime_model(model)
    if method.impl == "sac":
        return _build_sac_runtime_model(model)
    if method.execution_mode == "compiled":
        return torch.compile(model, dynamic=True)
    return model


def _build_forward_fn(callable_model, ctx):
    criterion = ctx["criterion"]
    src_data = ctx["src_data"]
    tgt_data = ctx["tgt_data"]
    tgt_vocab_size = ctx["tgt_vocab_size"]

    def forward_fn():
        output = callable_model(src_data, tgt_data[:, :-1])
        return criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

    return forward_fn


def _instantiate_runtime_method(method, train_ctx, *, dropout_override=None):
    import torch

    ctx = _runtime_context(train_ctx)
    model = copy.deepcopy(ctx["clean_model"]).to(ctx["device"])
    if dropout_override is not None:
        _set_dropout_rate(model, dropout_override)
    model.train()

    callable_model = _build_callable_runtime_model(model, method)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9
    )
    return model, _build_forward_fn(callable_model, ctx), optimizer


# ── Correctness + profiling helpers ─────────────────────────────────────────



def _run_correctness_step(method, train_ctx, *, dropout_override=None, seed=20260330):
    import torch

    model, forward_fn, optimizer = _instantiate_runtime_method(
        method,
        train_ctx,
        dropout_override=dropout_override,
    )
    before = _snapshot_named_params(model)
    optimizer.zero_grad()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loss = forward_fn()
    loss_value = float(loss.item())
    loss.backward()
    grad_absmax = _grad_absmax(model)
    optimizer.step()
    update_absmax = _param_update_absmax(before, model)
    return {
        "loss": loss_value,
        "grad_absmax": grad_absmax,
        "update_absmax": update_absmax,
        "dropout_override": dropout_override,
        "baseline_key": "eager_baseline" if dropout_override is None else f"eager_baseline_dropout_{dropout_override}",
    }


def _build_correctness_baselines(train_ctx, methods):
    baselines = {}
    clean_model = train_ctx["clean_model"]

    eager_baseline_method = next((method for method in methods if method.name == "eager_baseline"), None)
    if eager_baseline_method is not None:
        baselines["eager_baseline"] = _run_correctness_step(eager_baseline_method, train_ctx)
        if _has_active_dropout(clean_model):
            baselines["eager_baseline_dropout_0.0"] = _run_correctness_step(
                eager_baseline_method,
                train_ctx,
                dropout_override=0.0,
            )

    compiled_baseline_method = next((method for method in methods if method.name == "compiled_no_recompute"), None)
    if compiled_baseline_method is not None:
        baselines["compiled_no_recompute"] = _run_correctness_step(compiled_baseline_method, train_ctx)
        if _has_active_dropout(clean_model):
            baselines["compiled_no_recompute_dropout_0.0"] = _run_correctness_step(
                compiled_baseline_method,
                train_ctx,
                dropout_override=0.0,
            )

    return baselines


def _correctness_baseline_key_for_method(method, dropout_override):
    if method.family == "graph":
        return "compiled_no_recompute" if dropout_override is None else f"compiled_no_recompute_dropout_{dropout_override}"
    return "eager_baseline" if dropout_override is None else f"eager_baseline_dropout_{dropout_override}"


def _correctness_baseline_tag_for_print(key: str) -> str:
    if key == "compiled_no_recompute":
        return "compiled_no_recompute"
    if key == "compiled_no_recompute_dropout_0.0":
        return "compiled_no_recompute(dropout=0.0)"
    return _canonical_correctness_baseline_name(key)


def _correctness_summary_header(methods):
    has_region = any(method.family == "region" for method in methods)
    has_graph = any(method.family == "graph" for method in methods)
    if has_region and has_graph:
        return "  固定 RNG 后，region 组相对 eager_baseline，graph 组相对 compiled_no_recompute"
    if has_graph:
        return "  固定 RNG 后，对比单步 loss / grad / 参数更新（相对 compiled_no_recompute）"
    return "  固定 RNG 后，对比单步 loss / grad / 参数更新（相对 eager_baseline）"


def _correctness_dropout_override(method, clean_model):
    if method.impl == "pytorch_min_cut" and _has_active_dropout(clean_model):
        return 0.0
    return None


def _run_method_correctness(method, train_ctx, clean_model):
    dropout_override = _correctness_dropout_override(method, clean_model)
    item = _run_correctness_step(method, train_ctx, dropout_override=dropout_override)
    item["baseline_key"] = _correctness_baseline_key_for_method(method, dropout_override)
    return item


def _correctness_baseline_for_item(item, baselines):
    return baselines.get(item.get("baseline_key"))


def _print_correctness_baselines(baselines):
    for baseline_key, baseline in baselines.items():
        _print_correctness_baseline(_correctness_baseline_tag_for_print(baseline_key), baseline)


def _correctness_should_skip_method(method, item):
    return method.name == item.get("baseline_key")


def _correctness_error_message(method, item):
    return f"  [{method.name}] correctness 失败: {item['error']}"


def _correctness_missing_baseline_message(method, item):
    return f"  [{method.name}] correctness 缺少 baseline: {item.get('baseline_key')}"


def _format_correctness_delta(value: float) -> str:
    return f"{value:+.6e}"


def _print_correctness_item(tag, item, baseline):
    loss_delta = item["loss"] - baseline["loss"]
    grad_delta = item["grad_absmax"] - baseline["grad_absmax"]
    update_delta = item["update_absmax"] - baseline["update_absmax"]
    extra = ""
    if item.get("dropout_override") is not None:
        extra = f" | diagnostic dropout={item['dropout_override']}"
    print(
        f"  [{tag}] "
        f"loss Δ {_format_correctness_delta(loss_delta)} | "
        f"grad_absmax Δ {_format_correctness_delta(grad_delta)} | "
        f"update_absmax Δ {_format_correctness_delta(update_delta)}"
        f"{extra}"
    )


def _print_correctness_baseline(tag, item):
    extra = ""
    if item.get("dropout_override") is not None:
        extra = f" | dropout={item['dropout_override']}"
    print(
        f"  [{tag}] baseline | "
        f"loss={item['loss']:.6f} | "
        f"grad_absmax={item['grad_absmax']:.6e} | "
        f"update_absmax={item['update_absmax']:.6e}"
        f"{extra}"
    )


def _canonical_correctness_baseline_name(key: str) -> str:
    if key == "eager_baseline":
        return "eager_baseline"
    if key == "eager_baseline_dropout_0.0":
        return "eager_baseline(dropout=0.0)"
    return key


def _print_correctness_summary(results, baselines, methods, line):
    if not baselines:
        return

    print(f"\n[阶段 4/6] Correctness 对照")
    print(line)
    print(_correctness_summary_header(methods))
    _print_correctness_baselines(baselines)

    for method in methods:
        item = results.get(method.name)
        if item is None or _correctness_should_skip_method(method, item):
            continue
        if "error" in item:
            print(_correctness_error_message(method, item))
            continue
        baseline = _correctness_baseline_for_item(item, baselines)
        if baseline is None:
            print(_correctness_missing_baseline_message(method, item))
            continue
        _print_correctness_item(method.name, item, baseline)


def _run_correctness_checks(train_ctx, methods, line):
    results = {}
    baselines = _build_correctness_baselines(train_ctx, methods)
    clean_model = train_ctx["clean_model"]
    for method in methods:
        try:
            results[method.name] = _run_method_correctness(method, train_ctx, clean_model)
        except Exception as exc:
            results[method.name] = {"error": f"{type(exc).__name__}: {exc}"}
    _print_correctness_summary(results, baselines, methods, line)
    return {
        "baselines": baselines,
        "methods": results,
    }


def _profile_runtime_method(runtime_profiler, method, train_ctx):
    runtime_note = None
    try:
        _model, forward_fn, optimizer = _instantiate_runtime_method(method, train_ctx)
        runtime_result = runtime_profiler.profile_step(
            method.name,
            forward_fn=forward_fn,
            optimizer=optimizer,
            warmup=3,
            steps=10,
        )
        return runtime_result, runtime_note, None
    except Exception as exc:
        if method.impl != "pytorch_min_cut" or not _has_active_dropout(train_ctx["clean_model"]):
            return None, runtime_note, exc
        print("    检测到 dropout/random lowering 问题，改用 dropout=0 诊断路径重试。")
        runtime_note = "diagnostic_only_dropout_forced_to_0.0"
        try:
            _model, forward_fn, optimizer = _instantiate_runtime_method(
                method,
                train_ctx,
                dropout_override=0.0,
            )
            runtime_result = runtime_profiler.profile_step(
                method.name,
                forward_fn=forward_fn,
                optimizer=optimizer,
                warmup=3,
                steps=10,
            )
            runtime_result["note"] = runtime_note
            return runtime_result, runtime_note, None
        except Exception as retry_exc:
            return None, runtime_note, retry_exc


def _summarize_family(results, family, baseline_name, line):
    print(f"\n{line}")
    print(f"  {family} 语义汇总（相对 {baseline_name}）")
    print(line)
    base = results.get(baseline_name)
    if not base or "runtime" not in base:
        print(f"  {baseline_name} 基线缺失，跳过。")
        return

    for tag, item in results.items():
        method = item.get("method")
        if method is None or method.family != family or tag == baseline_name or "runtime" not in item:
            continue
        base_peak = base["runtime"].get("avg_peak_abs_bytes", base["runtime"]["avg_peak_bytes"])
        item_peak = item["runtime"].get("avg_peak_abs_bytes", item["runtime"]["avg_peak_bytes"])
        runtime_delta = base_peak - item_peak
        time_delta = item["runtime"]["iqr_elapsed_ms"] - base["runtime"]["iqr_elapsed_ms"]
        verdict = _classify(runtime_delta, time_delta)
        note = item.get("runtime_note")
        note_suffix = f" | {note}" if note else ""
        print(
            f"  [{tag}] {verdict} | "
            f"runtime Δ {_fmt_gb(runtime_delta)} | "
            f"IQR步时 Δ {time_delta:+.1f} ms"
            f"{note_suffix}"
        )


def _fmt_mb(num_bytes: float) -> str:
    return f"{num_bytes / (1 << 20):.2f} MB"


def _get_static_entry(static_ctx, tag):
    estimator = static_ctx.get("static_estimator")
    if estimator is None:
        return None
    return estimator._results.get(tag)


def _get_runtime_entry(runtime_results, tag):
    entry = runtime_results.get(tag)
    if not entry:
        return None
    return entry.get("runtime")


def _compute_graph_gap_breakdown(static_ctx, runtime_results):
    base_tag = "compiled_no_recompute"
    compare_tag = "ATenIR_strat7_b1.0"

    static_base = _get_static_entry(static_ctx, base_tag)
    static_cmp = _get_static_entry(static_ctx, compare_tag)
    runtime_base = _get_runtime_entry(runtime_results, base_tag)
    runtime_cmp = _get_runtime_entry(runtime_results, compare_tag)
    if not all([static_base, static_cmp, runtime_base, runtime_cmp]):
        return None

    static_saved_delta = static_base["saved_act_bytes"] - static_cmp["saved_act_bytes"]
    static_fw_delta = static_base["fw_phase_peak"] - static_cmp["fw_phase_peak"]
    static_bw_delta = static_base["bw_phase_peak"] - static_cmp["bw_phase_peak"]
    static_overall_delta = static_base["estimated_peak"] - static_cmp["estimated_peak"]

    runtime_base_mem_delta = runtime_base.get("avg_base_memory_bytes", -1.0) - runtime_cmp.get("avg_base_memory_bytes", -1.0)
    runtime_fw_delta = runtime_base.get("avg_forward_peak_abs_bytes", -1.0) - runtime_cmp.get("avg_forward_peak_abs_bytes", -1.0)
    runtime_bw_delta = runtime_base.get("avg_backward_peak_abs_bytes", -1.0) - runtime_cmp.get("avg_backward_peak_abs_bytes", -1.0)
    runtime_overall_delta = runtime_base.get("avg_peak_abs_bytes", -1.0) - runtime_cmp.get("avg_peak_abs_bytes", -1.0)

    result = {
        "baseline": base_tag,
        "compared": compare_tag,
        "static_saved_act_delta_bytes": static_saved_delta,
        "static_fw_phase_delta_bytes": static_fw_delta,
        "static_bw_phase_delta_bytes": static_bw_delta,
        "static_overall_delta_bytes": static_overall_delta,
        "runtime_base_memory_delta_bytes": runtime_base_mem_delta,
        "runtime_fw_peak_abs_delta_bytes": runtime_fw_delta,
        "runtime_bw_peak_abs_delta_bytes": runtime_bw_delta,
        "runtime_overall_peak_abs_delta_bytes": runtime_overall_delta,
    }
    if static_overall_delta > 0 and runtime_overall_delta >= 0:
        result["overall_realization_ratio"] = runtime_overall_delta / static_overall_delta
    if static_fw_delta > 0 and runtime_fw_delta >= 0:
        result["fw_realization_ratio"] = runtime_fw_delta / static_fw_delta
    if static_bw_delta > 0 and runtime_bw_delta >= 0:
        result["bw_realization_ratio"] = runtime_bw_delta / static_bw_delta
    return result


def _print_graph_gap_breakdown(gap_breakdown, line):
    if gap_breakdown is None:
        return

    print(f"\n{line}")
    print("  graph 主轨道 static/runtime gap 分解（compiled_no_recompute → ATenIR_strat7_b1.0）")
    print(line)
    print(f"  static 保存激活减少: {_fmt_gb(gap_breakdown['static_saved_act_delta_bytes'])} ({_fmt_mb(gap_breakdown['static_saved_act_delta_bytes'])})")
    print(f"  static FW phase 下降: {_fmt_gb(gap_breakdown['static_fw_phase_delta_bytes'])}")
    print(f"  static BW phase 下降: {_fmt_gb(gap_breakdown['static_bw_phase_delta_bytes'])}")
    print(f"  static 总峰值下降: {_fmt_gb(gap_breakdown['static_overall_delta_bytes'])}")
    print(f"  runtime 基线占用变化: {_fmt_gb(gap_breakdown['runtime_base_memory_delta_bytes'])}")
    print(f"  runtime FW 绝对峰值下降: {_fmt_gb(gap_breakdown['runtime_fw_peak_abs_delta_bytes'])}")
    print(f"  runtime BW 绝对峰值下降: {_fmt_gb(gap_breakdown['runtime_bw_peak_abs_delta_bytes'])}")
    print(f"  runtime 总峰值下降: {_fmt_gb(gap_breakdown['runtime_overall_peak_abs_delta_bytes'])}")
    if 'overall_realization_ratio' in gap_breakdown:
        print(f"  overall 兑现率: {gap_breakdown['overall_realization_ratio']:.1%}")
    if 'fw_realization_ratio' in gap_breakdown:
        print(f"  FW 兑现率: {gap_breakdown['fw_realization_ratio']:.1%}")
    if 'bw_realization_ratio' in gap_breakdown:
        print(f"  BW 兑现率: {gap_breakdown['bw_realization_ratio']:.1%}")


def _analyze_bw_gap_components(static_ctx, runtime_results):
    base_tag = "compiled_no_recompute"
    compare_tag = "ATenIR_strat7_b1.0"
    static_base = _get_static_entry(static_ctx, base_tag)
    static_cmp = _get_static_entry(static_ctx, compare_tag)
    runtime_base = _get_runtime_entry(runtime_results, base_tag)
    runtime_cmp = _get_runtime_entry(runtime_results, compare_tag)
    if not all([static_base, static_cmp, runtime_base, runtime_cmp]):
        return None

    return {
        "baseline": base_tag,
        "compared": compare_tag,
        "static_saved_act_delta_bytes": static_base["saved_act_bytes"] - static_cmp["saved_act_bytes"],
        "static_bw_phase_delta_bytes": static_base["bw_phase_peak"] - static_cmp["bw_phase_peak"],
        "runtime_bw_peak_abs_delta_bytes": runtime_base.get("avg_backward_peak_abs_bytes", -1.0) - runtime_cmp.get("avg_backward_peak_abs_bytes", -1.0),
        "runtime_bw_peak_delta_bytes": runtime_base.get("avg_backward_peak_bytes", -1.0) - runtime_cmp.get("avg_backward_peak_bytes", -1.0),
        "runtime_base_memory_delta_bytes": runtime_base.get("avg_base_memory_bytes", -1.0) - runtime_cmp.get("avg_base_memory_bytes", -1.0),
        "runtime_fw_peak_abs_delta_bytes": runtime_base.get("avg_forward_peak_abs_bytes", -1.0) - runtime_cmp.get("avg_forward_peak_abs_bytes", -1.0),
    }


def _print_bw_gap_components(bw_gap, line):
    if bw_gap is None:
        return
    print(f"\n{line}")
    print("  BW gap 细分（为什么 static BW 节省未完全兑现）")
    print(line)
    print(f"  static 保存激活减少: {_fmt_gb(bw_gap['static_saved_act_delta_bytes'])}")
    print(f"  static BW phase 下降: {_fmt_gb(bw_gap['static_bw_phase_delta_bytes'])}")
    print(f"  runtime BW 绝对峰值下降: {_fmt_gb(bw_gap['runtime_bw_peak_abs_delta_bytes'])}")
    print(f"  runtime BW 增量峰值下降: {_fmt_gb(bw_gap['runtime_bw_peak_delta_bytes'])}")
    print(f"  runtime 基线占用变化: {_fmt_gb(bw_gap['runtime_base_memory_delta_bytes'])}")
    print(f"  runtime FW 绝对峰值下降: {_fmt_gb(bw_gap['runtime_fw_peak_abs_delta_bytes'])}")
    print("  解释: BW 兑现率不足不只是 saved activation，仍受 runtime 调度/临时量/绝对峰值落点影响。")



def run_runtime_analysis(
    train_ctx: dict,
    static_ctx: dict,
    methods: list,
    strategy_config: dict,
    strat_tag: str,
    model_name: str,
    line: str,
    bold: str,
):
    from aten_recompute.analysis import MemoryProfiler

    benchmark_cfg = train_ctx.get("benchmark_config", {})
    run_correctness_checks = bool(benchmark_cfg.get("run_correctness_checks", True))
    correctness_results = {}
    if run_correctness_checks:
        correctness_results = _run_correctness_checks(train_ctx, methods, line)

    print(f"\n[阶段 5/6] 多方法 runtime 对照")
    print(line)

    runtime_profiler = MemoryProfiler(device=str(train_ctx["device"]))
    results = {}

    for method in methods:
        print(f"\n  [{method.name}] {_describe_method(method)}")
        try:
            runtime_result, runtime_note, runtime_error = _profile_runtime_method(
                runtime_profiler,
                method,
                train_ctx,
            )
            if runtime_error is not None:
                raise runtime_error
            results[method.name] = {
                "method": method,
                "runtime": runtime_result,
            }
            if runtime_note is not None:
                results[method.name]["runtime_note"] = runtime_note
        except Exception as exc:
            print(f"    runtime 失败: {type(exc).__name__}: {exc}")
            results[method.name] = {
                "method": method,
                "runtime_error": f"{type(exc).__name__}: {exc}",
            }

    _summarize_family(results, "region", "eager_baseline", line)
    _summarize_family(results, "graph", "compiled_no_recompute", line)

    graph_gap_breakdown = _compute_graph_gap_breakdown(static_ctx, results)
    bw_gap_breakdown = _analyze_bw_gap_components(static_ctx, results)
    _print_graph_gap_breakdown(graph_gap_breakdown, line)
    _print_bw_gap_components(bw_gap_breakdown, line)

    print(f"\n[阶段 6/6] Runtime profiler 报告")
    print(line)
    runtime_profiler.report()
    runtime_profiler.save_report(
        model_name=model_name,
        extra_payload={
            "graph_gap_breakdown": graph_gap_breakdown,
            "bw_gap_breakdown": bw_gap_breakdown,
        },
    )
    return {
        "runtime_results": results,
        "correctness_results": correctness_results,
        "graph_gap_breakdown": graph_gap_breakdown,
        "bw_gap_breakdown": bw_gap_breakdown,
    }
