#!/usr/bin/env python
"""BW graph detailed diagnostic for b=0.0 overestimation.

Breaks down the BW graph to understand why simulation overestimates by ~22%.
"""
import gc, sys, torch
sys.path.insert(0, ".")

from toolkit.models import ModelRegistry
from toolkit.capture import capture_inductor_graphs
from toolkit.simulation.graph_estimator import estimate_graph_peak
from toolkit.utils import val_bytes, is_view_node, format_bytes, count_unique_params
from toolkit.profiler import measure_phased
from toolkit.strategy import set_memory_budget, clear_memory_budget

DEVICE = "cuda"
BATCH, SEQ = 4, 512
MODEL_NAME = "llama"

def _cleanup():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def main():
    reg = ModelRegistry()
    cfg = reg.get_config(MODEL_NAME)
    input_ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ), device=DEVICE)

    # ── Runtime measurement for b=0.0 ──
    _cleanup()
    m = reg.create_model(MODEL_NAME).to(DEVICE).train()
    set_memory_budget(0.0)
    torch._dynamo.reset()
    compiled = torch.compile(m, backend="inductor", dynamic=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    fwd = lambda: compiled(input_ids=input_ids, labels=input_ids).loss
    _cleanup()
    rt = measure_phased("b=0.0", fwd, opt, repeats=3, warmup=2, device=DEVICE)
    print(f"RT BW peak = {format_bytes(rt.bw_peak)}")
    print(f"RT overall = {format_bytes(rt.overall_peak)}")

    del compiled, opt, m
    clear_memory_budget(); torch._dynamo.reset(); _cleanup()

    # ── Capture graphs for b=0.0 ──
    m2 = reg.create_model(MODEL_NAME).to(DEVICE).train()
    param_bytes = count_unique_params(m2)
    optim_bytes = param_bytes * 2  # Adam
    static_base = param_bytes + optim_bytes
    print(f"\nstatic_base = {format_bytes(static_base)}")
    print(f"param_bytes = {format_bytes(param_bytes)}")
    print(f"RT BW graph-only = {format_bytes(rt.bw_peak - static_base)}")

    cap = capture_inductor_graphs(
        m2, input_ids, loss_fn=lambda out: out.logits.sum(),
        model_kwargs={"labels": input_ids}, budget=0.0)

    bw_gm = cap["bw_gm"]
    sched_bw = cap.get("sched_bw_peak")
    print(f"sched_bw = {format_bytes(sched_bw)}")

    # ── BW graph analysis ──
    nodes = list(bw_gm.graph.nodes)
    print(f"\nBW graph: {len(nodes)} nodes total")

    placeholders, outputs, views, compute, other = [], [], [], [], []
    for n in nodes:
        if n.op == "placeholder": placeholders.append(n)
        elif n.op == "output": outputs.append(n)
        elif n.op in ("call_function", "call_method") and is_view_node(n):
            views.append(n)
        elif n.op in ("call_function", "call_method"):
            compute.append(n)
        else:
            other.append(n)

    print(f"  placeholders: {len(placeholders)}")
    print(f"  views: {len(views)}")
    print(f"  compute: {len(compute)}")
    print(f"  output: {len(outputs)}")
    print(f"  other: {len(other)}")

    # ── Size analysis ──
    ph_total = sum(val_bytes(n.meta.get("val")) or 0 for n in placeholders)
    compute_total = sum(val_bytes(n.meta.get("val")) or 0 for n in compute)
    print(f"\n  Placeholder total bytes: {format_bytes(ph_total)}")
    print(f"  Compute total bytes: {format_bytes(compute_total)}")

    # ── Output inputs (pinned) ──
    output_inputs = set()
    for n in outputs:
        output_inputs.update(n.all_input_nodes)

    # trace back to non-view bases for pinned nodes
    pinned_non_view = set()
    for oi in output_inputs:
        if oi.op in ("call_function", "call_method") and is_view_node(oi):
            # trace to base
            cur = oi
            while cur.op in ("call_function", "call_method") and is_view_node(cur):
                cur = cur.all_input_nodes[0]
            pinned_non_view.add(cur)
        else:
            pinned_non_view.add(oi)

    pinned_bytes = sum(val_bytes(n.meta.get("val")) or 0 for n in pinned_non_view)
    print(f"\n  Pinned output-input nodes: {len(output_inputs)} (non-view bases: {len(pinned_non_view)})")
    print(f"  Pinned bytes: {format_bytes(pinned_bytes)}")

    # ── Check for potential in-place candidates ──
    # A node could be in-place if: output shape == some input shape, and that input's
    # only remaining user is this node (last use).
    align = 512
    node_size = {}
    for n in nodes:
        if n.op == "output": continue
        if n.op in ("call_function", "call_method") and is_view_node(n): continue
        val = n.meta.get("val")
        nb = val_bytes(val) if val is not None else 0
        if nb > 0:
            node_size[n] = (nb + align - 1) & ~(align - 1)

    last_use = {}
    for idx, n in enumerate(nodes):
        for inp in n.all_input_nodes:
            if inp in node_size:
                last_use[inp] = idx

    inplace_candidates = 0
    inplace_bytes = 0
    for idx, n in enumerate(compute):
        if n not in node_size:
            continue
        nsize = node_size[n]
        for inp in n.all_input_nodes:
            if inp in node_size and node_size[inp] == nsize:
                if last_use.get(inp, -1) == nodes.index(n):
                    inplace_candidates += 1
                    inplace_bytes += nsize
                    break

    print(f"\n  Potential in-place candidates: {inplace_candidates}")
    print(f"  In-place candidate bytes: {format_bytes(inplace_bytes)}")

    # ── Run graph peak estimation and get timeline ──
    result = estimate_graph_peak(bw_gm, pin_output_inputs=True, fusion_aware=True)
    peak = result["peak_bytes"]
    print(f"\n  Estimated BW graph peak: {format_bytes(peak)}")

    # Find peak point in timeline
    timeline = result["timeline"]
    peak_event = max(timeline, key=lambda e: e["current"])
    print(f"  Peak at index {peak_event['index']}, node={peak_event['node']}, "
          f"event={peak_event['event']}, current={format_bytes(peak_event['current'])}")

    # Count live nodes at peak
    live_at_peak = set()
    live_sizes = {}
    running = 0
    for evt in timeline:
        if evt["event"] == "alloc":
            live_at_peak.add(evt["node"])
            live_sizes[evt["node"]] = evt["bytes"]
            running += evt["bytes"]
        elif evt["event"] == "free":
            live_at_peak.discard(evt["node"])
            running -= live_sizes.pop(evt["node"], 0)
        if running == peak:
            break

    # Categorize live nodes at peak
    node_by_name = {n.name: n for n in nodes}
    ph_live = sum(live_sizes[name] for name in live_at_peak
                  if name in node_by_name and node_by_name[name].op == "placeholder")
    comp_live = sum(live_sizes[name] for name in live_at_peak
                    if name in node_by_name and node_by_name[name].op != "placeholder")
    print(f"\n  At peak: {len(live_at_peak)} live nodes")
    print(f"    Placeholder (saved acts): {format_bytes(ph_live)}")
    print(f"    Compute (intermediates):  {format_bytes(comp_live)}")

    # Top 10 largest live nodes at peak
    sorted_live = sorted(live_sizes.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 live nodes at peak:")
    for name, size in sorted_live[:10]:
        n = node_by_name.get(name)
        op_name = str(n.target).split(".")[-1] if n else "?"
        ntype = "PH" if n and n.op == "placeholder" else "COMP"
        print(f"    {name:40s} {format_bytes(size):>10}  [{ntype}] {op_name}")

    del m2; _cleanup()

if __name__ == "__main__":
    main()
