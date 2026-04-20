"""Tests for L2.5 fusion-aware memory estimation.

Tests cover:
  - Op classification (extern vs fusable)
  - Fusion group identification (linear chain, extern barrier, multi-user)
  - Graph output pinning (internal nodes that feed output are not eliminated)
  - Integration: fusion_aware=True reduces peak vs fusion_aware=False
  - Regression: fusion_aware has no effect on aot_eager graphs (no extern ops)
"""

import pytest
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from toolkit.simulation.graph_estimator import estimate_graph_peak
from toolkit.simulation.fusion_ops import EXTERN_OPS, _populate_extern_ops, is_extern_op, is_fusable_op
from toolkit.simulation.fusion_groups import identify_fusion_groups, fusion_group_stats
from toolkit.utils import is_view_node, val_bytes


# ── Helpers ──────────────────────────────────────────────────────────


def _trace(fn, *inputs):
    """Trace fn with FakeTensors to get an FX graph with meta['val']."""
    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_inputs = [
            fake_mode.from_tensor(inp) if isinstance(inp, torch.Tensor) else inp
            for inp in inputs
        ]
        return make_fx(fn, tracing_mode="fake")(*fake_inputs)


def _build_node_size(gm, align=512):
    """Build node_size dict like estimate_graph_peak does."""
    node_size = {}
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        if node.op in ("call_function", "call_method") and is_view_node(node):
            continue
        val = node.meta.get("val")
        nb = val_bytes(val) if val is not None else 0
        if nb > 0:
            node_size[node] = (nb + align - 1) & ~(align - 1)
    return node_size


# ── Op Classification Tests ─────────────────────────────────────────


def test_extern_ops_populated():
    """EXTERN_OPS should contain mm, bmm, addmm after population."""
    _populate_extern_ops()
    assert len(EXTERN_OPS) > 0
    # Check known extern ops are present
    assert torch.ops.aten.mm.default in EXTERN_OPS
    assert torch.ops.aten.bmm.default in EXTERN_OPS


def test_pointwise_not_extern():
    """Common pointwise ops should not be in EXTERN_OPS."""
    _populate_extern_ops()
    pointwise_ops = [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.relu.default,
    ]
    for op in pointwise_ops:
        assert op not in EXTERN_OPS, f"{op} should not be extern"


def test_is_extern_op_on_nodes():
    """is_extern_op correctly classifies mm vs add in a traced graph."""
    def fn(x, y):
        return torch.mm(x, y) + 1.0

    gm = _trace(fn, torch.randn(4, 4), torch.randn(4, 4))
    nodes = list(gm.graph.nodes)

    mm_found = False
    add_found = False
    for n in nodes:
        if n.op == "call_function":
            if n.target == torch.ops.aten.mm.default:
                assert is_extern_op(n)
                assert not is_fusable_op(n)
                mm_found = True
            elif n.target == torch.ops.aten.add.Tensor:
                assert not is_extern_op(n)
                assert is_fusable_op(n)
                add_found = True
    assert mm_found, "mm node not found in graph"
    assert add_found, "add node not found in graph"


# ── Fusion Group Tests ───────────────────────────────────────────────


def test_linear_chain_fused():
    """a → b → c (all pointwise): b should be internal."""
    def fn(x):
        a = x + 1.0        # pointwise
        b = a * 2.0         # pointwise
        c = b - 0.5         # pointwise
        return c

    gm = _trace(fn, torch.randn(4, 4))
    nodes = list(gm.graph.nodes)
    ns = _build_node_size(gm)

    group_id, internal = identify_fusion_groups(nodes, ns)

    # At least the middle node(s) should be internal
    assert len(internal) > 0, "Linear chain should have internal nodes"
    # The final node (c) should NOT be internal (it has no users in group)
    compute_nodes = [n for n in nodes if n in group_id]
    last_compute = compute_nodes[-1]
    assert last_compute not in internal, "Last compute node should not be internal"


def test_extern_breaks_group():
    """pointwise → mm → pointwise should have separate groups."""
    def fn(x):
        a = x + 1.0                  # fusable group 1
        b = torch.mm(a, a)           # extern (barrier)
        c = b * 2.0                  # fusable group 2
        return c

    gm = _trace(fn, torch.randn(4, 4))
    nodes = list(gm.graph.nodes)
    ns = _build_node_size(gm)

    group_id, internal = identify_fusion_groups(nodes, ns)

    # Find the mm node and its neighbors
    mm_node = None
    add_node = None
    mul_node = None
    for n in nodes:
        if n.op == "call_function":
            if n.target == torch.ops.aten.mm.default:
                mm_node = n
            elif n.target == torch.ops.aten.add.Tensor:
                add_node = n
            elif n.target == torch.ops.aten.mul.Tensor:
                mul_node = n

    assert mm_node is not None
    # mm should be in a different group from both add and mul
    if add_node in group_id and mul_node in group_id and mm_node in group_id:
        assert group_id[add_node] != group_id[mm_node], "add and mm should be in different groups"
        assert group_id[mul_node] != group_id[mm_node], "mul and mm should be in different groups"
    # mm output should NOT be internal (it's extern)
    assert mm_node not in internal


def test_graph_output_not_internal():
    """Even if a node is in a fusion group, it should not be internal if
    it feeds the graph output."""
    def fn(x):
        a = x + 1.0
        b = a * 2.0     # b feeds output → not internal
        return b

    gm = _trace(fn, torch.randn(4, 4))
    nodes = list(gm.graph.nodes)
    ns = _build_node_size(gm)

    # Pin output inputs
    output_inputs = set()
    for n in nodes:
        if n.op == "output":
            output_inputs.update(n.all_input_nodes)

    group_id, internal = identify_fusion_groups(nodes, ns, output_inputs)

    # b should not be internal because it's a graph output
    for n in nodes:
        if n.op == "call_function" and n.target == torch.ops.aten.mul.Tensor:
            assert n not in internal, "Graph output node should not be internal"


def test_multi_user_not_internal():
    """If a node is used by both group-internal AND group-external consumers,
    it should not be internal."""
    def fn(x):
        a = x + 1.0
        b = torch.mm(a, a)    # extern: uses a externally
        c = a * 2.0           # fusable: uses a internally
        return b + c

    gm = _trace(fn, torch.randn(4, 4))
    nodes = list(gm.graph.nodes)
    ns = _build_node_size(gm)

    group_id, internal = identify_fusion_groups(nodes, ns)

    # 'a' has users in different groups (mm is extern, mul is fusable)
    # so 'a' should NOT be internal
    for n in nodes:
        if n.op == "call_function" and n.target == torch.ops.aten.add.Tensor:
            assert n not in internal, (
                "Node with users in different groups should not be internal"
            )


# ── Integration: estimate_graph_peak with fusion_aware ───────────────


def test_fusion_aware_reduces_peak():
    """fusion_aware=True should give peak <= fusion_aware=False."""
    def fn(x):
        a = x + 1.0
        b = a * 2.0
        c = b + 3.0
        d = torch.mm(c, c)
        e = d - 1.0
        return e

    gm = _trace(fn, torch.randn(4, 4))

    r_normal = estimate_graph_peak(gm, fusion_aware=False)
    r_fusion = estimate_graph_peak(gm, fusion_aware=True)

    assert r_fusion["peak_bytes"] <= r_normal["peak_bytes"], (
        f"Fusion-aware peak ({r_fusion['peak_bytes']}) should be "
        f"<= normal peak ({r_normal['peak_bytes']})"
    )
    # Fusion stats should be present
    assert "fusion_groups" in r_fusion
    assert "internal_nodes" in r_fusion
    assert "internal_bytes" in r_fusion
    assert r_fusion["internal_bytes"] >= 0


def test_fusion_aware_no_effect_without_fusion():
    """For a graph with only extern ops, fusion_aware should not change peak."""
    def fn(x, y):
        return torch.mm(x, y)

    gm = _trace(fn, torch.randn(4, 4), torch.randn(4, 4))

    r_normal = estimate_graph_peak(gm, fusion_aware=False)
    r_fusion = estimate_graph_peak(gm, fusion_aware=True)

    assert r_fusion["peak_bytes"] == r_normal["peak_bytes"]
    assert r_fusion["internal_bytes"] == 0


def test_fusion_stats_keys():
    """fusion_group_stats should return expected keys."""
    def fn(x):
        a = x + 1.0
        b = torch.mm(a, a)
        c = b * 2.0
        return c

    gm = _trace(fn, torch.randn(4, 4))
    nodes = list(gm.graph.nodes)
    ns = _build_node_size(gm)
    group_id, internal = identify_fusion_groups(nodes, ns)
    stats = fusion_group_stats(group_id, internal, ns)

    for key in ("num_groups", "num_internal", "internal_bytes",
                "num_extern_groups", "num_fusable_groups"):
        assert key in stats, f"Missing key: {key}"
    assert stats["num_extern_groups"] >= 1  # mm
    assert stats["num_groups"] >= 2  # at least fusable + extern
