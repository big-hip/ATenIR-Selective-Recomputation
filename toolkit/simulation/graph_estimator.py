import torch
import torch.fx as fx
import torch.nn as nn

from toolkit.capture import count_fw_output_bytes
from toolkit.utils import count_unique_params, is_view_node, val_bytes


def _find_view_base(node: fx.Node) -> fx.Node:
    """Trace a view-node chain back to the base allocation node.

    When a view node is saved as a graph output, its underlying storage
    comes from a non-view base tensor.  This helper follows the chain
    ``view -> input that shares storage -> ...`` until it reaches a
    non-view node, which is the real memory owner.
    """
    visited: set[int] = set()
    current = node
    while current.op in ("call_function", "call_method") and is_view_node(current):
        if id(current) in visited:
            break
        visited.add(id(current))
        val = current.meta.get("val")
        if not isinstance(val, torch.Tensor):
            break
        try:
            val_cdata = val.untyped_storage()._cdata
        except Exception:
            break
        found = False
        for inp in current.all_input_nodes:
            inp_val = inp.meta.get("val")
            if isinstance(inp_val, torch.Tensor):
                try:
                    if inp_val.untyped_storage()._cdata == val_cdata:
                        current = inp
                        found = True
                        break
                except Exception:
                    pass
        if not found:
            break
    return current


def _schedule_min_peak(
    nodes: list[fx.Node],
    node_size: dict[fx.Node, int],
    output_inputs: set[fx.Node],
) -> list[fx.Node]:
    """Reorder graph nodes to minimise peak live memory (greedy heuristic).

    Simulates what Inductor's Scheduler does for memory optimisation:
    at each step the ready node that **frees** the most memory (net of
    its own allocation) is scheduled next.

    This is especially effective for the BW graph where the default
    topological order keeps all saved-activation placeholders alive
    simultaneously, but a smarter order can free them layer-by-layer.

    Complexity: O(N × avg_ready_set_size) – fast for ≤ 2 k nodes.
    """
    placeholders = [n for n in nodes if n.op == "placeholder"]
    output_node = [n for n in nodes if n.op == "output"]
    compute = [n for n in nodes if n.op not in ("placeholder", "output")]

    if not compute:
        return nodes

    compute_set = set(compute)

    # deps[n] = set of compute-node predecessors that must run before n
    deps: dict[fx.Node, set[fx.Node]] = {}
    # rev_deps[n] = compute-node successors unblocked when n is done
    rev_deps: dict[fx.Node, set[fx.Node]] = {n: set() for n in compute}
    for n in compute:
        d = set()
        for inp in n.all_input_nodes:
            if inp in compute_set:
                d.add(inp)
                rev_deps[inp].add(n)
        deps[n] = d

    # remaining_users[n] = how many unscheduled compute-nodes still need n
    remaining_users: dict[fx.Node, int] = {}
    for n in compute:
        for inp in n.all_input_nodes:
            if inp in node_size:                     # only memory-bearing
                remaining_users[inp] = remaining_users.get(inp, 0) + 1

    scheduled: set[fx.Node] = set(placeholders)
    ready: set[fx.Node] = set()
    for n in compute:
        if not deps[n]:
            ready.add(n)

    ordered: list[fx.Node] = []

    while ready:
        best = None
        best_score = float("-inf")
        for n in ready:
            freed = 0
            for inp in n.all_input_nodes:
                if (inp in remaining_users
                        and inp not in output_inputs
                        and remaining_users[inp] == 1):
                    freed += node_size.get(inp, 0)
            alloc = node_size.get(n, 0)
            score = freed - alloc
            if score > best_score:
                best_score = score
                best = n
        if best is None:                             # safety guard
            break

        ordered.append(best)
        ready.discard(best)
        scheduled.add(best)

        # Update remaining user counts
        for inp in best.all_input_nodes:
            if inp in remaining_users:
                remaining_users[inp] = max(0, remaining_users[inp] - 1)

        # Unlock successors
        for succ in rev_deps.get(best, set()):
            if succ not in scheduled and succ not in ready:
                if all(d in scheduled for d in deps[succ]):
                    ready.add(succ)

    # Fallback: any unscheduled nodes (should not happen in valid graphs)
    remaining = [n for n in compute if n not in scheduled]

    return placeholders + ordered + remaining + output_node


def _forwarded_primal_bytes(fw_gm: fx.GraphModule, n_params: int) -> int:
    """Count bytes of FW outputs that are forwarded model-parameter primals.

    In AOTAutograd's partitioned FW graph, the first *n_params* placeholders
    are model parameters (primals).  Some of them are passed straight through
    (or via a view chain) to the FW output tuple so that the BW graph can use
    them for gradient computation.  At runtime these forwarded primals share
    storage with the model's weight tensors — they do NOT allocate new memory.

    Returns the total bytes of such forwarded primals.
    """
    placeholders = [n for n in fw_gm.graph.nodes if n.op == "placeholder"]
    param_phs = set(placeholders[:n_params])

    output_node = next((n for n in fw_gm.graph.nodes if n.op == "output"), None)
    if output_node is None:
        return 0

    output_args = output_node.args[0] if output_node.args else []

    forwarded = 0
    for arg in output_args:
        if not isinstance(arg, fx.Node):
            continue
        # Trace through view chains to the base allocation
        base = _find_view_base(arg)
        if base in param_phs:
            forwarded += val_bytes(base)
    return int(forwarded)


def estimate_graph_peak(
    gm: fx.GraphModule,
    pin_output_inputs: bool = False,
    align: int = 512,
    fusion_aware: bool = False,
    optimize_order: bool = False,
    simulate_inplace: bool = False,
) -> dict:
    """Estimate peak activation memory from an FX graph via live-range analysis.

    Args:
        gm: post-grad FX GraphModule (ATen IR).
        pin_output_inputs: keep graph-output tensors alive until the end.
        align: allocation alignment in bytes.
        fusion_aware: if True, identify Inductor-style fusion groups and
            zero-out allocations for intermediate tensors that would never
            be materialized in GPU global memory (L2.5 mode).
        simulate_inplace: if True, model Inductor-style buffer reuse:
            when a compute node's output has the same aligned size as one
            of its inputs whose last user is this node, the output reuses
            the input buffer (zero net allocation).  This approximates
            Inductor's codegen ``MemoryPlanning`` pass.

    Returns:
        dict with peak_bytes, allocation counts, timeline, and (when
        fusion_aware) fusion_groups / internal_nodes / internal_bytes.
    """
    nodes = list(gm.graph.nodes)

    output_inputs = set()
    if pin_output_inputs:
        for node in nodes:
            if node.op == "output":
                output_inputs.update(node.all_input_nodes)
        # Pin the *base* allocation for every pinned view node so that
        # freeing the base does not discard storage still referenced by
        # a view that is kept alive as a graph output.
        for n in list(output_inputs):
            if n.op in ("call_function", "call_method") and is_view_node(n):
                base = _find_view_base(n)
                output_inputs.add(base)

    node_size: dict[fx.Node, int] = {}
    view_count = 0
    for node in nodes:
        if node.op == "output":
            continue
        if node.op in ("call_function", "call_method") and is_view_node(node):
            view_count += 1
            continue
        val = node.meta.get("val")
        nb = val_bytes(val) if val is not None else 0
        if nb > 0:
            node_size[node] = (nb + align - 1) & ~(align - 1)

    # ── L2.5: fusion-aware allocation elimination ────────────────────
    fusion_stats = None
    if fusion_aware:
        from .fusion_groups import identify_fusion_groups, fusion_group_stats

        group_id, internal_nodes = identify_fusion_groups(
            nodes, node_size, output_inputs
        )
        fusion_stats = fusion_group_stats(group_id, internal_nodes, node_size)
        # Zero-out allocations for fusion-internal intermediates
        for n in internal_nodes:
            node_size[n] = 0

    # ── L2.5+: memory-optimal execution order ────────────────────────
    if optimize_order:
        nodes = _schedule_min_peak(nodes, node_size, output_inputs)

    last_use: dict[fx.Node, int] = {}
    for index, node in enumerate(nodes):
        for input_node in node.all_input_nodes:
            if input_node in node_size:
                last_use[input_node] = index

    current = 0
    peak = 0
    peak_ph_alive = 0          # placeholder bytes alive when peak is reached
    n_allocs = 0
    n_placeholders = 0
    n_frees = 0
    timeline = []
    live: dict[fx.Node, int] = {}
    ph_nodes: set[fx.Node] = set()   # track which live nodes are placeholders

    n_reuses = 0
    for index, node in enumerate(nodes):
        if node in node_size:
            size = node_size[node]

            # ── In-place buffer reuse: if a dying input has the same
            #    aligned size, recycle it instead of allocating fresh.
            reused_from = None
            if (simulate_inplace
                    and size > 0
                    and node.op in ("call_function", "call_method")):
                for inp in node.all_input_nodes:
                    if (inp in live
                            and live[inp] == size
                            and last_use.get(inp, -1) <= index
                            and inp not in output_inputs):
                        reused_from = inp
                        break

            if reused_from is not None:
                # Recycle: remove old entry, add new under current node
                live.pop(reused_from)
                live[node] = size
                n_reuses += 1
                timeline.append({
                    "index": index,
                    "node": node.name,
                    "event": "reuse",
                    "bytes": size,
                    "current": current,
                    "peak": peak,
                    "reused_from": reused_from.name,
                })
            else:
                live[node] = size
                current += size
                if node.op == "placeholder":
                    n_placeholders += 1
                    ph_nodes.add(node)
                else:
                    n_allocs += 1
                if current > peak:
                    peak = current
                    peak_ph_alive = sum(live[n] for n in ph_nodes if n in live)
                timeline.append({
                    "index": index,
                    "node": node.name,
                    "event": "alloc",
                    "bytes": size,
                    "current": current,
                    "peak": peak,
                })

        to_free = [
            live_node
            for live_node in tuple(live)
            if live_node is not node and last_use.get(live_node, -1) <= index and live_node not in output_inputs
        ]
        for live_node in to_free:
            size = live.pop(live_node)
            current -= size
            n_frees += 1
            ph_nodes.discard(live_node)
            timeline.append({
                "index": index,
                "node": live_node.name,
                "event": "free",
                "bytes": size,
                "current": current,
                "peak": peak,
            })

    result = {
        "peak_bytes": peak,
        "peak_ph_alive": peak_ph_alive,
        "num_alloc_nodes": n_allocs,
        "num_placeholders": n_placeholders,
        "num_view_nodes": view_count,
        "n_allocs": n_allocs,
        "n_frees": n_frees,
        "n_reuses": n_reuses,
        "timeline": timeline,
    }
    if fusion_stats is not None:
        result["fusion_groups"] = fusion_stats["num_groups"]
        result["internal_nodes"] = fusion_stats["num_internal"]
        result["internal_bytes"] = fusion_stats["internal_bytes"]
    return result


def estimate_training_peak(
    fw_gm: fx.GraphModule,
    bw_gm: fx.GraphModule,
    model: nn.Module,
    optimizer_cls=torch.optim.Adam,
    fused_optimizer: bool = False,
) -> dict:
    param_bytes = count_unique_params(model)
    grad_bytes = param_bytes

    if optimizer_cls in (torch.optim.Adam, torch.optim.AdamW):
        optim_mul = 2
    elif optimizer_cls is torch.optim.SGD:
        optim_mul = 0
    else:
        optim_mul = 2
    optim_bytes = param_bytes * optim_mul

    fw_result = estimate_graph_peak(fw_gm, pin_output_inputs=True)
    bw_result = estimate_graph_peak(bw_gm, pin_output_inputs=True)

    # Graph-level activation-only peaks (internal analysis)
    fw_graph_peak = fw_result["peak_bytes"]
    bw_graph_peak = bw_result["peak_bytes"]
    act_peak = max(fw_graph_peak, bw_graph_peak)

    # Fixed base: param + optimizer states (after zero_grad set_to_none=True)
    static_base = param_bytes + optim_bytes

    # Optimizer temporary memory
    #   foreach Adam (default): _foreach_sqrt over all params -> param_bytes
    #   fused Adam: single CUDA kernel -> 0
    #   SGD: no momentum states -> 0
    if not fused_optimizer and optim_mul > 0:
        opt_temp = param_bytes
    else:
        opt_temp = 0

    # Absolute peaks (consistent with runtime measure_phased semantics)
    # Note: grad_bytes are NOT added to bw_peak because gradient tensors
    # are already modelled as BW-graph output nodes (alive until graph end)
    # and their view-bases are pinned via pin_output_inputs=True.
    #
    # FIX: FW graph placeholders include model parameters already in
    # static_base.  Subtract the placeholder bytes alive at peak (mostly
    # params, pinned via output_inputs), capped at param_bytes.
    #
    # For BW: placeholders are saved activations + forwarded primals.
    # Only forwarded primals overlap with static_base.  Use
    # _forwarded_primal_bytes to determine how many param bytes were
    # saved from FW → BW; cap at peak_ph_alive.
    fw_ph_overlap = min(param_bytes, fw_result["peak_ph_alive"])
    n_params = len(list(model.parameters())) + len(list(model.buffers()))
    fwd_primal = _forwarded_primal_bytes(fw_gm, n_params)
    bw_ph_overlap = min(fwd_primal, bw_result["peak_ph_alive"])
    fw_peak = static_base + max(0, fw_graph_peak - fw_ph_overlap)
    bw_peak = static_base + max(0, bw_graph_peak - bw_ph_overlap)
    opt_peak = static_base + grad_bytes + opt_temp
    fwbw_peak = max(fw_peak, bw_peak)
    true_peak = max(fw_peak, bw_peak, opt_peak)

    if true_peak == fw_peak:
        peak_phase = "FW"
    elif true_peak == bw_peak:
        peak_phase = "BW"
    else:
        peak_phase = "OPT"

    # Timeline sample points (for phase_timeline_chart)
    after_fw = static_base + max(0, fw_graph_peak - fw_ph_overlap)  # approx: live set at forward end
    after_bw = static_base + grad_bytes     # activations freed
    after_opt = static_base + grad_bytes    # temp freed

    return {
        "tag": "graph_L2",
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "optim_bytes": optim_bytes,
        "optimizer_bytes": optim_bytes,
        # absolute peaks (same semantics as runtime)
        "fw_peak": fw_peak,
        "bw_peak": bw_peak,
        "opt_peak": opt_peak,
        "fwbw_peak": fwbw_peak,
        "true_peak": true_peak,
        "estimated_peak": true_peak,  # backward-compatible alias
        "peak_phase": peak_phase,
        "opt_temp": opt_temp,
        # graph-level activation-only peaks (analysis + backward compat)
        "fw_graph_peak": fw_graph_peak,
        "bw_graph_peak": bw_graph_peak,
        "fw_peak_bytes": fw_graph_peak,  # backward-compatible alias
        "bw_peak_bytes": bw_graph_peak,  # backward-compatible alias
        "act_peak": act_peak,            # backward-compatible alias
        "fw_views": fw_result["num_view_nodes"],
        "bw_views": bw_result["num_view_nodes"],
        "saved_act_bytes": count_fw_output_bytes(fw_gm),
        # timeline sample points
        "base": static_base,
        "after_fw": after_fw,
        "after_bw": after_bw,
        "after_opt": after_opt,
    }


def estimate_inductor_training_peak(
    capture_result: dict,
    model: nn.Module,
    optimizer_cls=torch.optim.Adam,
    fused_optimizer: bool = False,
) -> dict:
    """Triple-layer (L2 + L2.5 + L3) training peak estimation from inductor capture.

    Args:
        capture_result: dict from ``capture_inductor_graphs()`` containing
            fw_gm, bw_gm, sched_fw_peak, sched_bw_peak.
        model: the model (for param counting).
        optimizer_cls: optimizer class for state estimation.
        fused_optimizer: whether fused optimizer is used.

    Returns:
        dict with all fields from ``estimate_training_peak`` (L2) plus:
            L2.5 fields: l25_fw_peak, l25_bw_peak, l25_opt_peak,
                l25_fwbw_peak, l25_true_peak, l25_peak_phase,
                fusion_fw_groups, fusion_bw_groups,
                fusion_fw_eliminated_bytes, fusion_bw_eliminated_bytes.
            L3 fields: l3_fw_peak, l3_bw_peak, l3_opt_peak,
                l3_fwbw_peak, l3_true_peak, l3_peak_phase,
                sched_fw_peak, sched_bw_peak.
        If Scheduler peaks were not captured, L3 fields are None.
    """
    fw_gm = capture_result["fw_gm"]
    bw_gm = capture_result["bw_gm"]
    sched_fw = capture_result.get("sched_fw_peak")
    sched_bw = capture_result.get("sched_bw_peak")

    # L2: standard graph-level estimation
    l2 = estimate_training_peak(
        fw_gm, bw_gm, model,
        optimizer_cls=optimizer_cls,
        fused_optimizer=fused_optimizer,
    )
    l2["tag"] = "inductor_L2L25L3"

    static_base = l2["base"]  # param_bytes + optim_bytes
    grad_bytes = l2["grad_bytes"]
    opt_temp = l2["opt_temp"]

    # L2.5: fusion-aware + in-place buffer reuse estimation
    #
    # Two complementary optimisations modelled statically:
    #   1. Fusion elimination — Triton fused kernels never materialise
    #      intermediate tensors in global memory (zeroed in live-range).
    #   2. In-place buffer reuse — when a compute node's output has the
    #      same aligned size as a dying input, the codegen can reuse the
    #      buffer (modelled as zero-net allocation in live-range).
    #
    # When the Inductor Scheduler BW peak is also available, we take the
    # tighter (lower) of the two estimates for BW, giving the best of
    # graph-level analysis and Scheduler-level simulation.
    fw_fa = estimate_graph_peak(fw_gm, pin_output_inputs=True,
                                fusion_aware=True, simulate_inplace=True)
    bw_fa = estimate_graph_peak(bw_gm, pin_output_inputs=True,
                                fusion_aware=True, simulate_inplace=True)

    param_bytes = l2["param_bytes"]
    fw_fa_ph_overlap = min(param_bytes, fw_fa["peak_ph_alive"])
    l25_fw_peak = static_base + max(0, fw_fa["peak_bytes"] - fw_fa_ph_overlap)

    # BW: take the tighter (lower) of Scheduler peak vs fusion+inplace
    # graph peak.  For b=0.0 (max recomputation), fusion+inplace is
    # significantly tighter than the Scheduler; for b=1.0, the Scheduler
    # is marginally tighter.  min() gives the best of both.
    # NOTE: sched_bw does NOT include param placeholders (it comes from
    # Inductor Scheduler codegen), so only adjust the graph-level peak.
    n_params = len(list(model.parameters())) + len(list(model.buffers()))
    fwd_primal = _forwarded_primal_bytes(fw_gm, n_params)
    bw_fa_ph_overlap = min(fwd_primal, bw_fa["peak_ph_alive"])
    bw_fi_peak = max(0, bw_fa["peak_bytes"] - bw_fa_ph_overlap)
    if sched_bw is not None:
        l25_bw_peak = static_base + min(sched_bw, bw_fi_peak)
    else:
        l25_bw_peak = static_base + bw_fi_peak

    l25_opt_peak = static_base + grad_bytes + opt_temp
    l25_fwbw_peak = max(l25_fw_peak, l25_bw_peak)
    l25_true_peak = max(l25_fw_peak, l25_bw_peak, l25_opt_peak)

    if l25_true_peak == l25_fw_peak:
        l25_peak_phase = "FW"
    elif l25_true_peak == l25_bw_peak:
        l25_peak_phase = "BW"
    else:
        l25_peak_phase = "OPT"

    l2.update({
        "l25_fw_peak": l25_fw_peak,
        "l25_bw_peak": l25_bw_peak,
        "l25_opt_peak": l25_opt_peak,
        "l25_fwbw_peak": l25_fwbw_peak,
        "l25_true_peak": l25_true_peak,
        "l25_peak_phase": l25_peak_phase,
        "fusion_fw_groups": fw_fa.get("fusion_groups", 0),
        "fusion_bw_groups": bw_fa.get("fusion_groups", 0),
        "fusion_fw_eliminated_bytes": fw_fa.get("internal_bytes", 0),
        "fusion_bw_eliminated_bytes": bw_fa.get("internal_bytes", 0),
    })

    # L3: Scheduler-level estimation
    if sched_fw is not None and sched_bw is not None:
        l3_fw_peak = static_base + sched_fw
        l3_bw_peak = static_base + sched_bw
        l3_opt_peak = static_base + grad_bytes + opt_temp
        l3_fwbw_peak = max(l3_fw_peak, l3_bw_peak)
        l3_true_peak = max(l3_fw_peak, l3_bw_peak, l3_opt_peak)

        if l3_true_peak == l3_fw_peak:
            l3_peak_phase = "FW"
        elif l3_true_peak == l3_bw_peak:
            l3_peak_phase = "BW"
        else:
            l3_peak_phase = "OPT"

        l2.update({
            "l3_fw_peak": l3_fw_peak,
            "l3_bw_peak": l3_bw_peak,
            "l3_opt_peak": l3_opt_peak,
            "l3_fwbw_peak": l3_fwbw_peak,
            "l3_true_peak": l3_true_peak,
            "l3_peak_phase": l3_peak_phase,
        })
    else:
        l2.update({
            "l3_fw_peak": None,
            "l3_bw_peak": None,
            "l3_opt_peak": None,
            "l3_fwbw_peak": None,
            "l3_true_peak": None,
            "l3_peak_phase": None,
        })

    l2["sched_fw_peak"] = sched_fw
    l2["sched_bw_peak"] = sched_bw
    return l2


def make_level_stub(est: dict, prefix: str) -> dict | None:
    """Build a validation-ready stub dict from an L2.5/L3 estimation result.

    Args:
        est: result dict from ``estimate_inductor_training_peak``.
        prefix: ``"l25"`` or ``"l3"``.

    Returns:
        A dict compatible with ``validate()`` (has true_peak, fw_peak, …)
        or None if the level was not computed.
    """
    tp = est.get(f"{prefix}_true_peak")
    if tp is None:
        return None
    fw = est[f"{prefix}_fw_peak"]
    bw = est[f"{prefix}_bw_peak"]
    opt = est[f"{prefix}_opt_peak"]
    fwbw = max(fw, bw)
    if tp == fw:
        phase = "FW"
    elif tp == bw:
        phase = "BW"
    else:
        phase = "OPT"
    return {
        "true_peak": tp,
        "fw_peak": fw,
        "bw_peak": bw,
        "opt_peak": opt,
        "fwbw_peak": fwbw,
        "peak_phase": phase,
        "param_bytes": est["param_bytes"],
        "grad_bytes": est["grad_bytes"],
        "optimizer_bytes": est["optimizer_bytes"],
    }
