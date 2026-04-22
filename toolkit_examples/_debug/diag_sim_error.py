#!/usr/bin/env python
"""诊断仿真误差根因 — 对比有/无重计算时 BW 图的结构差异"""

import gc
import sys
from pathlib import Path

import torch
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.utils import setup_experiment_env, count_unique_params, format_bytes
setup_experiment_env()

from toolkit.capture import capture_graphs
from toolkit.models import ModelRegistry
from toolkit.profiler import measure_phased
from toolkit.simulation.graph_estimator import (
    estimate_graph_peak, estimate_training_peak, _forwarded_primal_bytes
)
from toolkit.strategy import get_partition_fn, wrap_with_checkpoint
from toolkit.utils import is_view_node, val_bytes

DEVICE = "cuda"
MODEL_NAME = "llama"
MODEL_OVERRIDES = dict(
    hidden_size=2048,
    num_hidden_layers=16,
    num_attention_heads=16,
    intermediate_size=5504,
    num_key_value_heads=8,
    max_position_embeddings=1024,
)
BATCH = 8
SEQ = 512
OPTIMIZER_CLS = torch.optim.Adam
OPTIMIZER_KWARGS = dict(lr=1e-3, fused=True)


def graph_diag(gm, label):
    """打印 FX 图的详细统计信息"""
    nodes = list(gm.graph.nodes)
    n_ph = sum(1 for n in nodes if n.op == "placeholder")
    n_output = sum(1 for n in nodes if n.op == "output")
    n_view = 0
    n_compute = 0
    total_ph_bytes = 0
    total_compute_bytes = 0
    total_view_bytes = 0

    for n in nodes:
        if n.op == "placeholder":
            val = n.meta.get("val")
            if val is not None:
                total_ph_bytes += val_bytes(val)
        elif n.op in ("call_function", "call_method"):
            if is_view_node(n):
                n_view += 1
                val = n.meta.get("val")
                if val is not None:
                    total_view_bytes += val_bytes(val)
            else:
                n_compute += 1
                val = n.meta.get("val")
                if val is not None:
                    total_compute_bytes += val_bytes(val)

    # Count output bytes
    output_node = next((n for n in nodes if n.op == "output"), None)
    n_outputs = 0
    output_bytes = 0
    if output_node:
        for arg in output_node.args[0]:
            if hasattr(arg, 'meta'):
                val = arg.meta.get("val")
                if val is not None:
                    output_bytes += val_bytes(val)
                    n_outputs += 1

    print(f"\n  [{label}] Graph Diagnostics:")
    print(f"    Total nodes:       {len(nodes)}")
    print(f"    Placeholders:      {n_ph:>5d}  ({format_bytes(total_ph_bytes):>10s})")
    print(f"    Compute nodes:     {n_compute:>5d}  ({format_bytes(total_compute_bytes):>10s})")
    print(f"    View nodes:        {n_view:>5d}  ({format_bytes(total_view_bytes):>10s})")
    print(f"    Output args:       {n_outputs:>5d}  ({format_bytes(output_bytes):>10s})")

    return {
        "n_ph": n_ph, "n_compute": n_compute, "n_view": n_view,
        "ph_bytes": total_ph_bytes, "compute_bytes": total_compute_bytes,
        "output_bytes": output_bytes, "n_outputs": n_outputs,
        "total_nodes": len(nodes),
    }


def simulation_diag(fw_gm, bw_gm, model, label):
    """打印仿真各阶段的详细数值"""
    param_bytes = count_unique_params(model)
    n_params = len(list(model.parameters())) + len(list(model.buffers()))

    fw_result = estimate_graph_peak(fw_gm, pin_output_inputs=True)
    bw_result = estimate_graph_peak(bw_gm, pin_output_inputs=True)
    fwd_primal = _forwarded_primal_bytes(fw_gm, n_params)

    fw_ph_overlap = min(param_bytes, fw_result["peak_ph_alive"])
    bw_ph_overlap = min(fwd_primal, bw_result["peak_ph_alive"])

    static_base = param_bytes + 2 * param_bytes  # Adam 2x
    grad_bytes = param_bytes

    fw_peak = static_base + max(0, fw_result["peak_bytes"] - fw_ph_overlap)
    bw_peak = static_base + max(0, bw_result["peak_bytes"] - bw_ph_overlap)
    opt_peak = static_base + grad_bytes  # fused

    print(f"\n  [{label}] Simulation Breakdown:")
    print(f"    param_bytes       = {format_bytes(param_bytes)}")
    print(f"    static_base       = {format_bytes(static_base)}  (param + optim)")
    print(f"    grad_bytes        = {format_bytes(grad_bytes)}")
    print(f"    fwd_primal_bytes  = {format_bytes(fwd_primal)}")
    print(f"    -------- FW --------")
    print(f"    fw_graph_peak     = {format_bytes(fw_result['peak_bytes'])}")
    print(f"    fw_ph_alive@peak  = {format_bytes(fw_result['peak_ph_alive'])}")
    print(f"    fw_ph_overlap     = {format_bytes(fw_ph_overlap)}")
    print(f"    fw_peak (abs)     = {format_bytes(fw_peak)}")
    print(f"    -------- BW --------")
    print(f"    bw_graph_peak     = {format_bytes(bw_result['peak_bytes'])}")
    print(f"    bw_ph_alive@peak  = {format_bytes(bw_result['peak_ph_alive'])}")
    print(f"    bw_ph_overlap     = {format_bytes(bw_ph_overlap)}")
    print(f"    bw_peak (abs)     = {format_bytes(bw_peak)}")
    print(f"    -------- OPT --------")
    print(f"    opt_peak (abs)    = {format_bytes(opt_peak)}")
    print(f"    -------- OVERALL --------")
    print(f"    true_peak         = {format_bytes(max(fw_peak, bw_peak, opt_peak))}")
    print(f"    peak_phase        = {'FW' if max(fw_peak, bw_peak, opt_peak) == fw_peak else 'BW' if max(fw_peak, bw_peak, opt_peak) == bw_peak else 'OPT'}")

    # BW timeline: find top-5 allocation events
    timeline = bw_result["timeline"]
    alloc_events = [e for e in timeline if e["event"] == "alloc"]
    alloc_events.sort(key=lambda e: e["current"], reverse=True)
    print(f"\n    BW top-5 memory moments:")
    for i, e in enumerate(alloc_events[:5]):
        print(f"      #{i+1} current={format_bytes(e['current']):>10s} "
              f"peak={format_bytes(e['peak']):>10s} "
              f"node={e['node'][:50]}")

    return {
        "fw_graph_peak": fw_result["peak_bytes"],
        "bw_graph_peak": bw_result["peak_bytes"],
        "fw_ph_overlap": fw_ph_overlap,
        "bw_ph_overlap": bw_ph_overlap,
        "fwd_primal": fwd_primal,
        "fw_peak": fw_peak, "bw_peak": bw_peak, "opt_peak": opt_peak,
        "bw_n_allocs": bw_result["n_allocs"],
        "bw_n_placeholders": bw_result["num_placeholders"],
    }


def runtime_diag(label, model, compiled, input_ids):
    """运行时分阶段测量"""
    opt = OPTIMIZER_CLS(model.parameters(), **OPTIMIZER_KWARGS)
    rt = measure_phased(
        label, lambda: compiled(input_ids=input_ids, labels=input_ids).loss,
        opt, repeats=3, warmup=2, device=DEVICE,
    )
    print(f"\n  [{label}] Runtime Measurement:")
    print(f"    base       = {format_bytes(rt.base_allocated)}")
    print(f"    fw_peak    = {format_bytes(rt.fw_peak)}")
    print(f"    bw_peak    = {format_bytes(rt.bw_peak)}")
    print(f"    opt_peak   = {format_bytes(rt.opt_peak)}")
    print(f"    true_peak  = {format_bytes(rt.overall_peak)}")
    print(f"    peak_phase = {rt.peak_phase}")
    del opt
    return rt


def main():
    if not torch.cuda.is_available():
        raise SystemExit("GPU required")

    registry = ModelRegistry()
    config = registry.get_config(MODEL_NAME, **MODEL_OVERRIDES)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)
    block_cls = registry.get_block_class_name(MODEL_NAME)

    print("=" * 80)
    print("  诊断: 仿真误差根因分析")
    print("  对比 S05 (无重计算) vs S10 (AC+重计算)")
    print("=" * 80)

    # ── S05: aot_eager + default (NO recomputation) ──
    print("\n" + "━" * 80)
    print("  策略 S05: aot_eager + default_partition (无重计算)")
    print("━" * 80)

    m = registry.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()
    fw_gm_s05, bw_gm_s05 = capture_graphs(
        m, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
        partition_fn=get_partition_fn("default"),
    )
    s05_fw_diag = graph_diag(fw_gm_s05, "S05-FW")
    s05_bw_diag = graph_diag(bw_gm_s05, "S05-BW")
    s05_sim = simulation_diag(fw_gm_s05, bw_gm_s05, m, "S05")

    del m, fw_gm_s05, bw_gm_s05
    gc.collect(); torch.cuda.empty_cache()
    torch._dynamo.reset()

    # Runtime for S05
    m = registry.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()
    def _aot_backend(gm, example_inputs):
        def fw_c(fw_gm, _): return make_boxed_func(fw_gm.forward)
        def bw_c(bw_gm, _): return make_boxed_func(bw_gm.forward)
        return aot_module_simplified(gm, example_inputs,
            fw_compiler=fw_c, bw_compiler=bw_c,
            partition_fn=get_partition_fn("default"))
    torch._dynamo.reset()
    compiled_s05 = torch.compile(m, backend=_aot_backend, dynamic=True)
    rt_s05 = runtime_diag("S05-RT", m, compiled_s05, input_ids)
    del m, compiled_s05
    gc.collect(); torch.cuda.empty_cache()
    torch._dynamo.reset()

    # ── S10: AC + aot_eager + default (WITH recomputation) ──
    print("\n" + "━" * 80)
    print("  策略 S10: AC + aot_eager + default_partition (有重计算)")
    print("━" * 80)

    m = registry.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()
    wrap_with_checkpoint(m, block_cls)
    fw_gm_s10, bw_gm_s10 = capture_graphs(
        m, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
        partition_fn=get_partition_fn("default"),
    )
    s10_fw_diag = graph_diag(fw_gm_s10, "S10-FW")
    s10_bw_diag = graph_diag(bw_gm_s10, "S10-BW")
    s10_sim = simulation_diag(fw_gm_s10, bw_gm_s10, m, "S10")

    del m, fw_gm_s10, bw_gm_s10
    gc.collect(); torch.cuda.empty_cache()
    torch._dynamo.reset()

    # Runtime for S10
    m = registry.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()
    wrap_with_checkpoint(m, block_cls)
    torch._dynamo.reset()
    compiled_s10 = torch.compile(m, backend=_aot_backend, dynamic=True)
    rt_s10 = runtime_diag("S10-RT", m, compiled_s10, input_ids)
    del m, compiled_s10
    gc.collect(); torch.cuda.empty_cache()
    torch._dynamo.reset()

    # ── 对比分析 ──
    print("\n" + "=" * 80)
    print("  根因对比分析")
    print("=" * 80)

    print(f"\n  {'指标':30s} {'S05(无AC)':>15s} {'S10(有AC)':>15s} {'差异':>15s}")
    print("  " + "-" * 78)

    metrics = [
        ("BW graph total nodes", s05_bw_diag["total_nodes"], s10_bw_diag["total_nodes"]),
        ("BW placeholders", s05_bw_diag["n_ph"], s10_bw_diag["n_ph"]),
        ("BW compute nodes", s05_bw_diag["n_compute"], s10_bw_diag["n_compute"]),
        ("BW view nodes", s05_bw_diag["n_view"], s10_bw_diag["n_view"]),
        ("BW placeholder bytes", s05_bw_diag["ph_bytes"], s10_bw_diag["ph_bytes"]),
        ("BW compute alloc bytes", s05_bw_diag["compute_bytes"], s10_bw_diag["compute_bytes"]),
        ("BW output args", s05_bw_diag["n_outputs"], s10_bw_diag["n_outputs"]),
        ("BW output bytes", s05_bw_diag["output_bytes"], s10_bw_diag["output_bytes"]),
    ]
    for name, v1, v2 in metrics:
        if v1 > 1e6:
            print(f"  {name:30s} {format_bytes(v1):>15s} {format_bytes(v2):>15s} {format_bytes(v2-v1):>15s}")
        else:
            print(f"  {name:30s} {v1:>15d} {v2:>15d} {v2-v1:>+15d}")

    byte_metrics = [
        ("sim bw_graph_peak", s05_sim["bw_graph_peak"], s10_sim["bw_graph_peak"]),
        ("sim bw_ph_overlap", s05_sim["bw_ph_overlap"], s10_sim["bw_ph_overlap"]),
        ("sim fwd_primal_bytes", s05_sim["fwd_primal"], s10_sim["fwd_primal"]),
        ("sim bw_peak (abs)", s05_sim["bw_peak"], s10_sim["bw_peak"]),
        ("RT bw_peak (abs)", rt_s05.bw_peak, rt_s10.bw_peak),
        ("BW gap (RT-sim)", rt_s05.bw_peak - s05_sim["bw_peak"], rt_s10.bw_peak - s10_sim["bw_peak"]),
    ]
    print()
    for name, v1, v2 in byte_metrics:
        print(f"  {name:30s} {format_bytes(v1):>15s} {format_bytes(v2):>15s} {format_bytes(v2-v1):>15s}")

    # Key diagnostic: check if BW graph has recomputation nodes
    print(f"\n  关键诊断:")
    print(f"    S05 BW compute nodes: {s05_bw_diag['n_compute']}  (仅梯度计算)")
    print(f"    S10 BW compute nodes: {s10_bw_diag['n_compute']}  (重计算 + 梯度计算)")
    ratio = s10_bw_diag['n_compute'] / max(1, s05_bw_diag['n_compute'])
    print(f"    节点比 S10/S05 = {ratio:.2f}x")

    if ratio < 1.3:
        print(f"\n  ★ 结论: BW 图中缺少重计算节点!")
        print(f"    AC 模型的 BW 图与无 AC 模型的 BW 图结构相似,")
        print(f"    说明 torch.compile + AC 的重计算发生在图外部 (autograd engine),")
        print(f"    FX graph capture 无法捕获这部分内存开销。")
    else:
        print(f"\n  ★ 结论: BW 图中包含重计算节点 ({s10_bw_diag['n_compute'] - s05_bw_diag['n_compute']} 额外节点)")
        print(f"    但 live-range 仿真仍然低估峰值,")
        print(f"    可能原因: 拓扑序遍历时释放内存的时机与运行时不一致。")

    bw_gap_s10 = rt_s10.bw_peak - s10_sim["bw_peak"]
    param_bytes = count_unique_params(registry.create_model(MODEL_NAME, **MODEL_OVERRIDES).cpu())
    print(f"\n    S10 BW 仿真缺口 = {format_bytes(bw_gap_s10)}  (~{bw_gap_s10/param_bytes:.1f}x param_bytes)")


if __name__ == "__main__":
    main()
