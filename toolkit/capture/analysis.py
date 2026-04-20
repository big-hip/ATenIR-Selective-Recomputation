import torch
import torch.fx as fx

from toolkit.utils import is_view_node, val_bytes


def _iter_output_args(output_arg):
    if isinstance(output_arg, (tuple, list)):
        for item in output_arg:
            yield item
        return
    yield output_arg


def graph_stats(gm: fx.GraphModule) -> dict:
    n_total = 0
    n_placeholder = 0
    n_view = 0
    n_alloc = 0
    total_alloc_bytes = 0
    total_view_bytes = 0
    symint_ok = 0
    symint_fail = 0

    for node in gm.graph.nodes:
        n_total += 1
        if node.op == "placeholder":
            n_placeholder += 1
        elif node.op in ("call_function", "call_method"):
            val = node.meta.get("val")
            if val is not None:
                nb = val_bytes(val)
                if is_view_node(node):
                    n_view += 1
                    total_view_bytes += nb
                elif isinstance(val, torch.Tensor):
                    n_alloc += 1
                    total_alloc_bytes += nb

        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            for dim in val.shape:
                if isinstance(dim, torch.SymInt):
                    try:
                        int(dim)
                        symint_ok += 1
                    except Exception:
                        symint_fail += 1

    return {
        "n_total": n_total,
        "n_placeholder": n_placeholder,
        "n_view": n_view,
        "n_alloc": n_alloc,
        "total_alloc_bytes": total_alloc_bytes,
        "total_view_bytes": total_view_bytes,
        "symint_ok": symint_ok,
        "symint_fail": symint_fail,
    }


def analyze_graph(gm: fx.GraphModule) -> dict:
    nodes = []
    for node in gm.graph.nodes:
        val = node.meta.get("val")
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "is_view": is_view_node(node) if node.op in ("call_function", "call_method") else False,
                "bytes": val_bytes(val) if val is not None else 0,
            }
        )
    return {"num_nodes": len(nodes), "nodes": nodes}


def count_fw_outputs(fw_gm: fx.GraphModule) -> int:
    for node in fw_gm.graph.nodes:
        if node.op == "output":
            output_arg = node.args[0] if node.args else ()
            return sum(1 for _ in _iter_output_args(output_arg))
    return 0


def count_fw_output_bytes(fw_gm: fx.GraphModule) -> int:
    total = 0
    for node in fw_gm.graph.nodes:
        if node.op == "output":
            for input_node in node.all_input_nodes:
                val = input_node.meta.get("val")
                if val is not None:
                    total += val_bytes(val)
    return total
