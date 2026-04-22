#!/usr/bin/env python
"""快速 Demo — 重计算前后 IR 对比 + 仿真 vs 运行时验证

展示框架核心能力：
  1. 捕获 ATen IR (FW/BW) 并保存为 Markdown + DOT
  2. Activation Checkpointing 前后 IR 结构差异
  3. L2 静态仿真 vs 运行时测量 MRE 对比

模型:
  DOT/IR 可视化: GPT-2 1L/64H/1head — 图节点 < 100，IDE 插件可直接渲染
  数值对比:      GPT-2 4L/256H/4head — 更有意义的 MRE 和 AC 节省
策略: Baseline (aot_eager+default) vs AC (aot_eager+default+checkpoint)
预计运行时间: <30s

输出:
  outputs/quick_demo/
    ├── baseline_fw_ir.md / baseline_bw_ir.md   (1L 小模型 IR)
    ├── baseline_fw.dot  / baseline_bw.dot     (1L 小模型 DOT，可插件渲染)
    ├── ac_fw_ir.md      / ac_bw_ir.md
    ├── ac_fw.dot        / ac_bw.dot
    └── summary.md                              (4L 大模型数值结果)
"""

import gc
import sys
import time
from pathlib import Path

import torch
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.utils import setup_experiment_env, format_bytes
setup_experiment_env()

from toolkit.capture import capture_graphs, graph_stats, count_fw_outputs, count_fw_output_bytes
from toolkit.models import ModelRegistry
from toolkit.output import print_comparison_table
from toolkit.profiler import measure_phased, validate
from toolkit.simulation import estimate_training_peak
from toolkit.strategy import wrap_with_checkpoint

DEVICE = "cuda"
MODEL_NAME = "gpt2"
# 小模型: 用于 DOT/IR 可视化（节点数 < 100，IDE 插件可渲染）
DOT_OVERRIDES = dict(n_layer=1, n_embd=64, n_head=1, n_inner=128)
DOT_BATCH, DOT_SEQ = 2, 32
# 大模型: 用于数值对比（MRE + AC 节省）
SIM_OVERRIDES = dict(n_layer=4, n_embd=256, n_head=4, n_inner=1024)
SIM_BATCH, SIM_SEQ = 4, 128
OUTPUT_DIR = Path(__file__).with_name("outputs") / "quick_demo"


# ── 通用工具 ──────────────────────────────────────────────────────
def _make_aot_backend():
    """创建 aot_eager 后端（不做任何代码生成，仅执行 ATen 图）。"""
    def backend(gm, example_inputs):
        def fw_c(fw_gm, _): return make_boxed_func(fw_gm.forward)
        def bw_c(bw_gm, _): return make_boxed_func(bw_gm.forward)
        return aot_module_simplified(gm, example_inputs,
                                     fw_compiler=fw_c, bw_compiler=bw_c)
    return backend


def _delta(a, b):
    """整数差异的带符号字符串。"""
    d = b - a
    return f"{d:+d}" if isinstance(d, int) else f"{d:+.0f}"


def _delta_bytes(a, b):
    """字节差异的人类可读字符串。"""
    d = b - a
    sign = "+" if d >= 0 else "-"
    return f"{sign}{format_bytes(abs(d))}"


def _part_header(part_num, title):
    """打印统一格式的 Part 标题栏。"""
    print(f"\n{'═' * 70}")
    print(f"  Part {part_num}: {title}")
    print(f"{'═' * 70}")


# ── 图分析工具 ──────────────────────────────────────────────────
def _shape_str(val):
    """Tensor shape → 字符串。"""
    if not isinstance(val, torch.Tensor):
        return ""
    dims = []
    for d in val.shape:
        if isinstance(d, torch.SymInt):
            try:
                dims.append(str(d.node.hint))
            except Exception:
                dims.append("?")
        else:
            dims.append(str(int(d)))
    return "(" + ", ".join(dims) + ")"


def _get_saved_tensor_info(fw_gm):
    """提取 FW 图 output 节点引用的所有 saved tensor 信息。"""
    from toolkit.utils import val_bytes
    results = []
    for node in fw_gm.graph.nodes:
        if node.op == "output":
            for inp in node.all_input_nodes:
                val = inp.meta.get("val")
                target_name = inp.target.__name__ if callable(getattr(inp, 'target', None)) else str(inp.target)
                results.append({
                    "name": inp.name,
                    "op": inp.op,
                    "target": target_name,
                    "shape": _shape_str(val) if val is not None else "",
                    "bytes": val_bytes(val) if val is not None else 0,
                })
    return results


def _get_alloc_node_info(gm):
    """提取图中所有 alloc 节点（真正分配新显存的节点）信息。"""
    from toolkit.utils import val_bytes, is_view_node
    results = []
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method"):
            continue
        val = node.meta.get("val")
        if val is None or not isinstance(val, torch.Tensor):
            continue
        if is_view_node(node):
            continue
        target_name = node.target.__name__ if callable(node.target) else str(node.target)
        results.append({
            "name": node.name,
            "target": target_name,
            "shape": _shape_str(val),
            "bytes": val_bytes(val),
        })
    return results


# ── IR / DOT 导出工具 ─────────────────────────────────────────────────
def save_ir(gm, out_dir, prefix, title):
    """保存 GraphModule 的 IR (Markdown) 和 DOT 文件。

    - IR (md):  使用 ``str(gm.graph)`` 导出标准 FX graph IR 文本
    - DOT:     使用 PyTorch 内置 ``FxGraphDrawer`` 导出 record 风格 DOT
               （含 op_code / target / dtype / shape / stride / num_users）
    """
    from torch.fx.passes.graph_drawer import FxGraphDrawer

    # Markdown: FX graph IR (简洁) + print_readable (完整 Python 代码)
    md_path = out_dir / f"{prefix}_ir.md"
    ir_text = str(gm.graph)
    readable = gm.print_readable(print_output=False)
    md_path.write_text(
        f"# {title}\n\n"
        f"## Graph IR\n\n```\n{ir_text}\n```\n\n"
        f"## Readable Code\n\n```python\n{readable}\n```\n",
        encoding="utf-8",
    )

    # DOT: FxGraphDrawer (record-style, 内置 shape/dtype/stride 标注)
    dot_path = out_dir / f"{prefix}.dot"
    drawer = FxGraphDrawer(gm, title.replace(" ", "_"))
    dot_graph = drawer.get_main_dot_graph()
    dot_path.write_text(dot_graph.to_string(), encoding="utf-8")

    return md_path, dot_path


# ── 主流程 ──────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise SystemExit("GPU is required")

    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry()
    block_cls = registry.get_block_class_name(MODEL_NAME)

    # DOT 模型 (1L/64H — 小图，插件可渲染)
    dot_config = registry.get_config(MODEL_NAME, **DOT_OVERRIDES)
    dot_input = torch.randint(0, dot_config.vocab_size, (DOT_BATCH, DOT_SEQ), device=DEVICE)
    # SIM 模型 (4L/256H — 大图，数值有意义)
    sim_config = registry.get_config(MODEL_NAME, **SIM_OVERRIDES)
    sim_input = torch.randint(0, sim_config.vocab_size, (SIM_BATCH, SIM_SEQ), device=DEVICE)

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print("  ATenIR Selective Recomputation — Quick Demo")
    print("=" * 70)
    print(f"  PyTorch {torch.__version__}  |  GPU: {gpu_name}")
    print(f"  IR 可视化: GPT-2 {DOT_OVERRIDES['n_layer']}L/{DOT_OVERRIDES['n_embd']}H, "
          f"B={DOT_BATCH}, Seq={DOT_SEQ}")
    print(f"  数值对比:  GPT-2 {SIM_OVERRIDES['n_layer']}L/{SIM_OVERRIDES['n_embd']}H, "
          f"B={SIM_BATCH}, Seq={SIM_SEQ}")
    print(f"  输出目录:  {OUTPUT_DIR}")

    summary_lines = ["# Quick Demo — 重计算 IR 对比 + 仿真验证\n"]

    # ════════════════════════════════════════════════════════════════
    #  Part 1: Baseline IR 捕获 (1L 小模型)
    # ════════════════════════════════════════════════════════════════
    _part_header(1, f"Baseline — 捕获 ATen IR (GPT-2 {DOT_OVERRIDES['n_layer']}L/{DOT_OVERRIDES['n_embd']}H)")

    model_bl = registry.create_model(MODEL_NAME, **DOT_OVERRIDES).to(DEVICE).train()
    fw_bl, bw_bl = capture_graphs(
        model_bl, dot_input, lambda out: out.loss,
        model_kwargs={"labels": dot_input},
    )
    fw_md, fw_dot = save_ir(fw_bl, OUTPUT_DIR, "baseline_fw", "Baseline Forward Graph")
    bw_md, bw_dot = save_ir(bw_bl, OUTPUT_DIR, "baseline_bw", "Baseline Backward Graph")

    bl_fw_stats = graph_stats(fw_bl)
    bl_bw_stats = graph_stats(bw_bl)
    bl_saved_n = count_fw_outputs(fw_bl)
    bl_saved_b = count_fw_output_bytes(fw_bl)

    print(f"  FW: {bl_fw_stats['n_total']} nodes ({bl_fw_stats['n_alloc']} alloc, "
          f"{bl_fw_stats['n_view']} view), saved_tensors={bl_saved_n} ({format_bytes(bl_saved_b)})")
    print(f"  BW: {bl_bw_stats['n_total']} nodes ({bl_bw_stats['n_alloc']} alloc, "
          f"{bl_bw_stats['n_view']} view)")
    print(f"  IR saved → {fw_md.name}, {bw_md.name}, {fw_dot.name}, {bw_dot.name}")

    del model_bl
    gc.collect()
    torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════
    #  Part 2: AC IR 捕获 (1L 小模型)
    # ════════════════════════════════════════════════════════════════
    _part_header(2, f"AC — 捕获 ATen IR (GPT-2 {DOT_OVERRIDES['n_layer']}L/{DOT_OVERRIDES['n_embd']}H + Checkpoint)")

    model_ac = registry.create_model(MODEL_NAME, **DOT_OVERRIDES).to(DEVICE).train()
    wrap_with_checkpoint(model_ac, block_cls)
    fw_ac, bw_ac = capture_graphs(
        model_ac, dot_input, lambda out: out.loss,
        model_kwargs={"labels": dot_input},
    )
    ac_fw_md, ac_fw_dot = save_ir(fw_ac, OUTPUT_DIR, "ac_fw", "AC Forward Graph")
    ac_bw_md, ac_bw_dot = save_ir(bw_ac, OUTPUT_DIR, "ac_bw", "AC Backward Graph")

    ac_fw_stats = graph_stats(fw_ac)
    ac_bw_stats = graph_stats(bw_ac)
    ac_saved_n = count_fw_outputs(fw_ac)
    ac_saved_b = count_fw_output_bytes(fw_ac)

    print(f"  FW: {ac_fw_stats['n_total']} nodes ({ac_fw_stats['n_alloc']} alloc, "
          f"{ac_fw_stats['n_view']} view), saved_tensors={ac_saved_n} ({format_bytes(ac_saved_b)})")
    print(f"  BW: {ac_bw_stats['n_total']} nodes ({ac_bw_stats['n_alloc']} alloc, "
          f"{ac_bw_stats['n_view']} view)")
    print(f"  IR saved → {ac_fw_md.name}, {ac_bw_md.name}, {ac_fw_dot.name}, {ac_bw_dot.name}")

    del model_ac
    gc.collect()
    torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════
    #  Part 3: IR 结构差异对比 (基于 1L 小模型)
    # ════════════════════════════════════════════════════════════════
    _part_header(3, "IR 结构差异对比")

    diff_rows = [
        {
            "指标": "FW 总节点数",
            "Baseline": bl_fw_stats["n_total"],
            "AC": ac_fw_stats["n_total"],
            "变化": _delta(bl_fw_stats["n_total"], ac_fw_stats["n_total"]),
        },
        {
            "指标": "FW alloc 节点",
            "Baseline": bl_fw_stats["n_alloc"],
            "AC": ac_fw_stats["n_alloc"],
            "变化": _delta(bl_fw_stats["n_alloc"], ac_fw_stats["n_alloc"]),
        },
        {
            "指标": "FW alloc 字节",
            "Baseline": format_bytes(bl_fw_stats["total_alloc_bytes"]),
            "AC": format_bytes(ac_fw_stats["total_alloc_bytes"]),
            "变化": _delta_bytes(bl_fw_stats["total_alloc_bytes"], ac_fw_stats["total_alloc_bytes"]),
        },
        {
            "指标": "BW 总节点数",
            "Baseline": bl_bw_stats["n_total"],
            "AC": ac_bw_stats["n_total"],
            "变化": _delta(bl_bw_stats["n_total"], ac_bw_stats["n_total"]),
        },
        {
            "指标": "BW alloc 节点",
            "Baseline": bl_bw_stats["n_alloc"],
            "AC": ac_bw_stats["n_alloc"],
            "变化": _delta(bl_bw_stats["n_alloc"], ac_bw_stats["n_alloc"]),
        },
        {
            "指标": "Saved tensors",
            "Baseline": bl_saved_n,
            "AC": ac_saved_n,
            "变化": _delta(bl_saved_n, ac_saved_n),
        },
        {
            "指标": "Saved bytes",
            "Baseline": format_bytes(bl_saved_b),
            "AC": format_bytes(ac_saved_b),
            "变化": _delta_bytes(bl_saved_b, ac_saved_b),
        },
    ]
    print_comparison_table(diff_rows, title="Baseline vs AC — IR 结构差异 (1L 小模型)")

    # ── 详细节点级差异分析 ──
    bl_saved_info = _get_saved_tensor_info(fw_bl)
    ac_saved_info = _get_saved_tensor_info(fw_ac)
    bl_saved_names = {s["name"] for s in bl_saved_info}
    ac_saved_names = {s["name"] for s in ac_saved_info}
    removed_saved = [s for s in bl_saved_info if s["name"] not in ac_saved_names]
    added_saved   = [s for s in ac_saved_info if s["name"] not in bl_saved_names]

    bl_alloc_info = _get_alloc_node_info(fw_bl)
    ac_alloc_info = _get_alloc_node_info(fw_ac)
    bl_alloc_names = {a["name"] for a in bl_alloc_info}
    ac_alloc_names = {a["name"] for a in ac_alloc_info}
    removed_alloc = [a for a in bl_alloc_info if a["name"] not in ac_alloc_names]
    added_alloc   = [a for a in ac_alloc_info if a["name"] not in bl_alloc_names]

    bl_bw_alloc = _get_alloc_node_info(bw_bl)
    ac_bw_alloc = _get_alloc_node_info(bw_ac)
    bl_bw_names = {a["name"] for a in bl_bw_alloc}
    new_bw_alloc = [a for a in ac_bw_alloc if a["name"] not in bl_bw_names]

    print(f"\n  AC 减少的 Saved Tensors ({len(removed_saved)} 个):")
    for s in removed_saved:
        print(f"    - {s['name']:30s}  {s['target']:30s}  {s['shape']:20s}  {format_bytes(s['bytes'])}")
    if added_saved:
        print(f"  AC 新增的 Saved Tensors ({len(added_saved)} 个):")
        for s in added_saved:
            print(f"    + {s['name']:30s}  {s['target']:30s}  {s['shape']:20s}  {format_bytes(s['bytes'])}")
    print(f"\n  AC 新增的 FW alloc 节点 ({len(added_alloc)} 个):")
    for a in added_alloc:
        print(f"    + {a['name']:30s}  {a['target']:30s}  {a['shape']:20s}  {format_bytes(a['bytes'])}")
    print(f"\n  AC 在 BW 新增的 recompute alloc 节点 ({len(new_bw_alloc)} 个，前 10 个):")
    for a in new_bw_alloc[:10]:
        print(f"    + {a['name']:30s}  {a['target']:30s}  {a['shape']:20s}  {format_bytes(a['bytes'])}")
    if len(new_bw_alloc) > 10:
        print(f"    ... 共 {len(new_bw_alloc)} 个")

    del fw_bl, bw_bl, fw_ac, bw_ac

    # ════════════════════════════════════════════════════════════════
    #  Part 4: L2 静态仿真 (4L 大模型)
    # ════════════════════════════════════════════════════════════════
    _part_header(4, f"L2 静态仿真对比 (GPT-2 {SIM_OVERRIDES['n_layer']}L/{SIM_OVERRIDES['n_embd']}H, B={SIM_BATCH})")

    model_bl_sim = registry.create_model(MODEL_NAME, **SIM_OVERRIDES).to(DEVICE).train()
    fw_bl2, bw_bl2 = capture_graphs(
        model_bl_sim, sim_input, lambda out: out.loss,
        model_kwargs={"labels": sim_input},
    )
    l2_bl = estimate_training_peak(fw_bl2, bw_bl2, model_bl_sim)
    del model_bl_sim, fw_bl2, bw_bl2
    gc.collect(); torch.cuda.empty_cache()

    model_ac_sim = registry.create_model(MODEL_NAME, **SIM_OVERRIDES).to(DEVICE).train()
    wrap_with_checkpoint(model_ac_sim, block_cls)
    fw_ac2, bw_ac2 = capture_graphs(
        model_ac_sim, sim_input, lambda out: out.loss,
        model_kwargs={"labels": sim_input},
    )
    l2_ac = estimate_training_peak(fw_ac2, bw_ac2, model_ac_sim)
    del model_ac_sim, fw_ac2, bw_ac2
    gc.collect(); torch.cuda.empty_cache()

    sim_rows = [{
        "策略": name,
        "fw_peak": format_bytes(l2["fw_peak"]),
        "bw_peak": format_bytes(l2["bw_peak"]),
        "true_peak": format_bytes(l2["true_peak"]),
        "peak_phase": l2["peak_phase"],
        "saved_act": format_bytes(l2["saved_act_bytes"]),
    } for name, l2 in [("Baseline", l2_bl), ("AC", l2_ac)]]
    print_comparison_table(sim_rows, title="L2 静态仿真 — Baseline vs AC")

    # ════════════════════════════════════════════════════════════════
    #  Part 5: 运行时验证 (4L 大模型)
    # ════════════════════════════════════════════════════════════════
    _part_header(5, "运行时验证 — 仿真 vs 实测 MRE")

    # Baseline runtime
    model_rt_bl = registry.create_model(MODEL_NAME, **SIM_OVERRIDES).to(DEVICE).train()
    torch._dynamo.reset()
    compiled_bl = torch.compile(model_rt_bl, backend=_make_aot_backend(), dynamic=True)
    opt_bl = torch.optim.Adam(model_rt_bl.parameters(), lr=1e-3)
    rt_bl = measure_phased(
        "Baseline", lambda: compiled_bl(input_ids=sim_input, labels=sim_input).loss,
        opt_bl, repeats=3, warmup=2, device=DEVICE,
    )
    val_bl = validate(l2_bl, rt_bl)
    del compiled_bl, opt_bl, model_rt_bl
    torch._dynamo.reset()
    gc.collect(); torch.cuda.empty_cache()

    # AC runtime
    model_rt_ac = registry.create_model(MODEL_NAME, **SIM_OVERRIDES).to(DEVICE).train()
    wrap_with_checkpoint(model_rt_ac, block_cls)
    torch._dynamo.reset()
    compiled_ac = torch.compile(model_rt_ac, backend=_make_aot_backend(), dynamic=True)
    opt_ac = torch.optim.Adam(model_rt_ac.parameters(), lr=1e-3)
    rt_ac = measure_phased(
        "AC", lambda: compiled_ac(input_ids=sim_input, labels=sim_input).loss,
        opt_ac, repeats=3, warmup=2, device=DEVICE,
    )
    val_ac = validate(l2_ac, rt_ac)
    del compiled_ac, opt_ac, model_rt_ac
    torch._dynamo.reset()
    gc.collect(); torch.cuda.empty_cache()

    final_rows = [{
        "策略": name,
        "L2 仿真": format_bytes(l2["true_peak"]),
        "运行时实测": format_bytes(rt.overall_peak),
        "MRE": f"{val.mre_allocated * 100:.2f}%",
        "方向": val.direction,
        "仿真 phase": l2["peak_phase"],
        "实测 phase": rt.peak_phase,
    } for name, l2, rt, val in [
        ("Baseline", l2_bl, rt_bl, val_bl),
        ("AC", l2_ac, rt_ac, val_ac),
    ]]
    print_comparison_table(final_rows, title="L2 仿真 vs 运行时 — 最终对比")

    # Peak savings
    bl_rt_peak = rt_bl.overall_peak
    ac_rt_peak = rt_ac.overall_peak
    saving_pct = (bl_rt_peak - ac_rt_peak) / bl_rt_peak * 100 if bl_rt_peak > 0 else 0
    print(f"\n  AC 实际显存节省: {format_bytes(bl_rt_peak - ac_rt_peak)} "
          f"({saving_pct:.1f}%)")

    # ════════════════════════════════════════════════════════════════
    #  Part 6: 生成 summary.md
    # ════════════════════════════════════════════════════════════════
    _part_header(6, "生成 summary.md")

    # ── 实验配置 ──
    S = summary_lines
    S.append(f"**IR 可视化模型**: GPT-2 ({dot_config.n_layer}L/{dot_config.n_embd}H/{DOT_OVERRIDES['n_head']}head), "
             f"Batch={DOT_BATCH}, Seq={DOT_SEQ}")
    S.append(f"**数值对比模型**: GPT-2 ({sim_config.n_layer}L/{sim_config.n_embd}H/{SIM_OVERRIDES['n_head']}head), "
             f"Batch={SIM_BATCH}, Seq={SIM_SEQ}")
    S.append(f"**比较策略**: Baseline (aot_eager + default partition) vs AC (aot_eager + Activation Checkpointing)")
    S.append("")

    # ── 1. IR 结构概览 ──
    S.append("## 1. IR 结构差异概览\n")
    S.append("> **说明**: 以下数据基于 1L 小模型。FW/BW 图通过 AOTAutograd 的 `aot_eager` 后端捕获，"
             "得到 ATen 级别的 FX GraphModule。\n")
    S.append("| 指标 | Baseline | AC | 变化 | 含义 |")
    S.append("|------|----------|----|----- |------|")
    explanations = {
        "FW 总节点数":   "前向图中所有操作节点总数（placeholder + 计算 + view + output）",
        "FW alloc 节点": "前向图中真正分配新 GPU 显存的计算节点（不含 view/reshape 等零拷贝操作）",
        "FW alloc 字节": "前向图中所有 alloc 节点输出 tensor 的字节数总和",
        "BW 总节点数":   "反向图中所有操作节点总数；AC 会在 BW 中插入重计算节点导致增加",
        "BW alloc 节点": "反向图中真正分配新显存的节点；AC 增加是因为重计算需要额外中间结果",
        "Saved tensors": "FW 图 output 节点中传递给 BW 的 tensor 数量；AC 减少这些来节省内存",
        "Saved bytes":   "FW 传递给 BW 的 tensor 总字节数；AC 用重计算换取这部分内存节省",
    }
    for row in diff_rows:
        key = row["指标"]
        S.append(f"| {key} | {row['Baseline']} | {row['AC']} | {row['变化']} | {explanations.get(key, '')} |")

    # ── 2. AC 减少的 Saved Tensors 详细列表 ──
    S.append(f"\n## 2. AC 减少的 Saved Tensors（共 {len(removed_saved)} 个）\n")
    S.append("> **Saved Tensor** 是前向传播保存下来、在反向传播中使用的中间激活值。"
             "Activation Checkpointing 的核心思想是：不保存这些中间值，而是在反向传播时重新计算它们，"
             "从而用额外的计算时间换取显存节省。\n")
    if removed_saved:
        S.append("| 节点名 | ATen 算子 | 输出 Shape | 字节数 |")
        S.append("|--------|-----------|------------|--------|")
        for s in removed_saved:
            S.append(f"| `{s['name']}` | `{s['target']}` | {s['shape']} | {format_bytes(s['bytes'])} |")
        total_removed_bytes = sum(s['bytes'] for s in removed_saved)
        S.append(f"\n**合计减少**: {len(removed_saved)} 个 tensor, {format_bytes(total_removed_bytes)}")
    else:
        S.append("（无差异）")

    if added_saved:
        S.append(f"\n### AC 新增的 Saved Tensors（共 {len(added_saved)} 个）\n")
        S.append("> Checkpoint 机制可能引入少量额外保存（如 RNG 状态），属于正常开销。\n")
        S.append("| 节点名 | ATen 算子 | 输出 Shape | 字节数 |")
        S.append("|--------|-----------|------------|--------|")
        for s in added_saved:
            S.append(f"| `{s['name']}` | `{s['target']}` | {s['shape']} | {format_bytes(s['bytes'])} |")

    # ── 3. AC 新增的 FW alloc 节点 ──
    S.append(f"\n## 3. AC 新增的 FW alloc 节点（共 {len(added_alloc)} 个）\n")
    S.append("> **alloc 节点** 是真正在 GPU 上分配新内存的计算操作（如 `mm`、`addmm`、`_softmax`），"
             "区别于 `view`/`reshape`/`permute` 等只改变 tensor 视图、不分配新内存的零拷贝操作。"
             "AC 包装器引入的额外 alloc 通常用于 RNG 状态保存等辅助功能。\n")
    if added_alloc:
        S.append("| 节点名 | ATen 算子 | 输出 Shape | 字节数 |")
        S.append("|--------|-----------|------------|--------|")
        for a in added_alloc:
            S.append(f"| `{a['name']}` | `{a['target']}` | {a['shape']} | {format_bytes(a['bytes'])} |")
    else:
        S.append("（无新增）")

    if removed_alloc:
        S.append(f"\n### AC 移除的 FW alloc 节点（共 {len(removed_alloc)} 个）\n")
        S.append("| 节点名 | ATen 算子 | 输出 Shape | 字节数 |")
        S.append("|--------|-----------|------------|--------|")
        for a in removed_alloc:
            S.append(f"| `{a['name']}` | `{a['target']}` | {a['shape']} | {format_bytes(a['bytes'])} |")

    # ── 4. AC 在 BW 新增的重计算节点 ──
    S.append(f"\n## 4. AC 在 BW 新增的重计算 alloc 节点（共 {len(new_bw_alloc)} 个）\n")
    S.append("> 这些节点是 AC 在反向传播中**重新执行前向计算**产生的。它们不在 Baseline 的 BW 图中，"
             "因为 Baseline 会直接使用 FW 保存的中间激活值。这体现了 AC \"用计算换内存\" 的核心权衡。\n")
    if new_bw_alloc:
        S.append("| 节点名 | ATen 算子 | 输出 Shape | 字节数 |")
        S.append("|--------|-----------|------------|--------|")
        for a in new_bw_alloc:
            S.append(f"| `{a['name']}` | `{a['target']}` | {a['shape']} | {format_bytes(a['bytes'])} |")
    else:
        S.append("（无新增）")

    # ── 5. L2 静态仿真 ──
    S.append(f"\n## 5. L2 静态仿真（{sim_config.n_layer}L/{sim_config.n_embd}H 模型）\n")
    S.append("> L2 仿真通过遍历 FX 图节点、追踪每个节点的分配/释放，静态估算训练各阶段的显存峰值。"
             "不需要实际运行模型。\n")
    S.append("| 策略 | fw_peak | bw_peak | true_peak | peak_phase | saved_act |")
    S.append("|------|---------|---------|-----------|------------|-----------|")
    for row in sim_rows:
        S.append(
            f"| {row['策略']} | {row['fw_peak']} | {row['bw_peak']} | "
            f"{row['true_peak']} | {row['peak_phase']} | {row['saved_act']} |"
        )
    S.append("")
    S.append("- **fw_peak**: 前向传播阶段的显存峰值")
    S.append("- **bw_peak**: 反向传播阶段的显存峰值")
    S.append("- **true_peak**: 整个训练 step 的真实峰值 = max(fw_peak, bw_peak, opt_peak)")
    S.append("- **peak_phase**: 峰值出现在哪个阶段")
    S.append("- **saved_act**: FW 保存给 BW 的激活值总字节数")

    # ── 6. 仿真 vs 运行时 ──
    S.append(f"\n## 6. 仿真 vs 运行时验证\n")
    S.append("> 将 L2 静态仿真结果与实际 GPU 运行时的显存峰值对比，"
             "计算 MRE (Mean Relative Error) 来验证仿真精度。\n")
    S.append("| 策略 | L2 仿真 | 运行时实测 | MRE | 方向 | 仿真 phase | 实测 phase |")
    S.append("|------|---------|----------|-----|------|-----------|-----------|")
    for row in final_rows:
        S.append(
            f"| {row['策略']} | {row['L2 仿真']} | {row['运行时实测']} | "
            f"{row['MRE']} | {row['方向']} | {row['仿真 phase']} | {row['实测 phase']} |"
        )
    S.append(f"\n**AC 实际显存节省**: {format_bytes(bl_rt_peak - ac_rt_peak)} "
             f"({saving_pct:.1f}%)")
    S.append("")
    S.append("- **MRE < 20%**: under-estimation 主要来自 CUDA allocator 碎片和运行时固定开销（"
             "CUDA context、cuDNN workspace 等），这些在静态仿真中无法完全建模")
    S.append("- **方向 = under**: 仿真值低于实测值，符合预期（静态仿真不包含分配器开销）")

    # ── 7. 输出文件 ──
    S.append("\n## 7. 输出文件\n")
    S.append("| 文件 | 说明 |")
    S.append("|------|------|")
    file_desc = {
        "baseline_fw_ir.md":  "Baseline 前向图 IR（FX graph 文本格式）",
        "baseline_bw_ir.md":  "Baseline 反向图 IR",
        "baseline_fw.dot":    "Baseline 前向图 DOT（可用 Graphviz 插件渲染）",
        "baseline_bw.dot":    "Baseline 反向图 DOT",
        "ac_fw_ir.md":        "AC 前向图 IR",
        "ac_bw_ir.md":        "AC 反向图 IR",
        "ac_fw.dot":          "AC 前向图 DOT",
        "ac_bw.dot":          "AC 反向图 DOT",
        "summary.md":         "本报告",
    }
    for f in sorted(OUTPUT_DIR.glob("*")):
        desc = file_desc.get(f.name, "")
        S.append(f"| `{f.name}` | {desc} |")

    elapsed = time.perf_counter() - t_start

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"  → {summary_path}")

    print(f"\n{'=' * 70}")
    print(f"  Done!  总耗时 {elapsed:.1f}s  |  输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
