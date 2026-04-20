"""Fusion group identification for L2.5 memory estimation.

Identifies groups of consecutive fusable ATen IR ops that Inductor would
fuse into single Triton kernels.  Intermediate tensors within a fusion
group are never materialized in GPU global memory, so their allocations
should be excluded from the peak memory estimate.

Algorithm (greedy topological scan):
  1. Walk FX graph nodes in topological order.
  2. Each extern (mm/bmm/sdpa) node starts its own singleton group.
  3. Each fusable node tries to join the group of its fusable producer.
     - If exactly one fusable-producer group exists → join it.
     - Otherwise (zero or multiple producer groups) → start a new group.
  4. After grouping, mark a node as "internal" (zero-alloc) if ALL of its
     users are in the same group (or are view ops that themselves only
     feed same-group consumers).
  5. Graph outputs and pinned outputs are never marked internal.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

import torch.fx as fx

from toolkit.utils import is_view_node

from .fusion_ops import is_extern_op


def identify_fusion_groups(
    nodes: list[fx.Node],
    node_size: dict[fx.Node, int],
    output_inputs: set[fx.Node] | None = None,
) -> Tuple[Dict[fx.Node, int], Set[fx.Node]]:
    """Identify fusion groups and internal (zero-alloc) nodes.

    Args:
        nodes: FX graph nodes in topological order.
        node_size: mapping from compute nodes to their aligned byte size
            (as produced by the sizing pass in ``estimate_graph_peak``).
            View nodes and output nodes should NOT be in this dict.
        output_inputs: set of nodes pinned as graph outputs (their
            allocations must not be eliminated).

    Returns:
        (group_id, internal_nodes):
            group_id: dict mapping each compute node → int group id
            internal_nodes: set of nodes whose allocation can be
                eliminated (they are fusion-internal intermediates)
    """
    if output_inputs is None:
        output_inputs = set()

    group_id: Dict[fx.Node, int] = {}
    next_gid = 0

    # ── Pass 1: assign groups ────────────────────────────────────────
    for node in nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op in ("call_function", "call_method") and is_view_node(node):
            continue
        if node not in node_size:
            continue

        if is_extern_op(node):
            # Extern ops are fusion barriers — each gets its own group
            group_id[node] = next_gid
            next_gid += 1
        else:
            # Fusable op: try to join a producer's fusable group
            producer_groups: set[int] = set()
            for inp in node.all_input_nodes:
                if inp in group_id and not is_extern_op(inp):
                    producer_groups.add(group_id[inp])

            if len(producer_groups) == 1:
                group_id[node] = next(iter(producer_groups))
            else:
                # 0 or ≥2 producer groups → start a new group
                group_id[node] = next_gid
                next_gid += 1

    # ── Pass 2: identify internal nodes ──────────────────────────────
    # A node is "internal" if:
    #   - it is fusable (not extern)
    #   - it is NOT a graph output (not in output_inputs)
    #   - ALL of its users are in the SAME fusion group
    #     (view-node users are traced through to their real consumers)
    internal: Set[fx.Node] = set()

    for node, gid in group_id.items():
        if is_extern_op(node):
            continue  # extern outputs always materialize
        if node in output_inputs:
            continue  # pinned as graph output

        consumers = _real_consumers(node, group_id)
        if not consumers:
            continue  # no consumers → graph-terminal, keep alive

        all_in_group = all(
            group_id.get(c) == gid for c in consumers
        )
        if all_in_group:
            internal.add(node)

    return group_id, internal


def _real_consumers(
    node: fx.Node, group_id: dict[fx.Node, int]
) -> list[fx.Node]:
    """Get non-view, non-output consumer nodes, tracing through views."""
    result: list[fx.Node] = []
    worklist = list(node.users)
    visited: set[int] = set()

    while worklist:
        user = worklist.pop()
        uid = id(user)
        if uid in visited:
            continue
        visited.add(uid)

        if user.op == "output":
            # Node feeds graph output → it must stay alive
            # Return empty to signal "not all in group"
            return []

        if user.op in ("call_function", "call_method") and is_view_node(user):
            # View node: trace through to its consumers
            worklist.extend(user.users)
            continue

        if user in group_id:
            result.append(user)
        else:
            # Consumer not in group_id (e.g. a node with size=0 that was
            # excluded from node_size) — conservatively keep alive
            return []

    return result


def fusion_group_stats(
    group_id: dict[fx.Node, int],
    internal: set[fx.Node],
    node_size: dict[fx.Node, int],
) -> dict:
    """Compute summary statistics about fusion groups.

    Returns:
        dict with keys:
            num_groups: total number of fusion groups
            num_internal: nodes marked as zero-alloc
            internal_bytes: total bytes eliminated
            num_extern_groups: groups containing an extern op
            num_fusable_groups: groups containing only fusable ops
    """
    from collections import Counter

    groups_set = set(group_id.values())
    extern_groups = set()
    for node, gid in group_id.items():
        if is_extern_op(node):
            extern_groups.add(gid)

    group_sizes = Counter(group_id.values())
    multi_node_groups = sum(1 for g, cnt in group_sizes.items()
                           if cnt > 1 and g not in extern_groups)

    internal_bytes = sum(node_size.get(n, 0) for n in internal)

    return {
        "num_groups": len(groups_set),
        "num_internal": len(internal),
        "internal_bytes": internal_bytes,
        "num_extern_groups": len(extern_groups),
        "num_fusable_groups": len(groups_set) - len(extern_groups),
        "num_multi_node_groups": multi_node_groups,
    }
