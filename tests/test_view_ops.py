import operator

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from toolkit.simulation.graph_estimator import estimate_graph_peak
from toolkit.utils import is_view_node


def _trace_fake(fn, *inputs):
    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_inputs = [
            fake_mode.from_tensor(inp) if isinstance(inp, torch.Tensor) else inp
            for inp in inputs
        ]
        return make_fx(fn, tracing_mode="fake")(*fake_inputs)


def test_getitem_is_view():
    def fn(x):
        out = torch.ops.aten.native_layer_norm.default(x, [x.shape[-1]], None, None, 1e-5)
        return out[0], out[1], out[2]

    gm = _trace_fake(fn, torch.randn(2, 4))
    getitem_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == operator.getitem
    ]

    assert getitem_nodes
    assert all(is_view_node(node) for node in getitem_nodes)


def test_getitem_output_pins_tuple_producing_base():
    def fn(x):
        out = torch.ops.aten.native_layer_norm.default(x, [x.shape[-1]], None, None, 1e-5)
        return out[0]

    gm = _trace_fake(fn, torch.randn(2, 4))
    result = estimate_graph_peak(gm, pin_output_inputs=True)
    freed = {
        event["node"]
        for event in result["timeline"]
        if event["event"] == "free"
    }

    assert not any("native_layer_norm" in name for name in freed)


def test_actual_alloc_not_view():
    def fn(x):
        return x + 1

    gm = _trace_fake(fn, torch.randn(2, 4))
    add_node = next(
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and "aten.add.Tensor" in str(node.target)
    )

    assert not is_view_node(add_node)


def test_all_aten_views_detected():
    def fn(x):
        first_row = x[:1, :]
        return (
            x.view(2, 8),
            x.permute(1, 0),
            first_row.expand(4, 4),
            x[:, :2],
            x.unsqueeze(0),
            x.t(),
            x.transpose(0, 1),
        )

    gm = _trace_fake(fn, torch.randn(4, 4))
    expected = {
        "aten.view.default": False,
        "aten.permute.default": False,
        "aten.expand.default": False,
        "aten.slice.Tensor": False,
        "aten.unsqueeze.default": False,
        "aten.t.default": False,
        "aten.transpose.int": False,
    }

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        for opname in expected:
            if opname in target:
                assert is_view_node(node)
                expected[opname] = True

    assert all(expected.values())
