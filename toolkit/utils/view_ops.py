import torch
import torch.fx as fx


def _iter_tensor_like_values(value):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensor_like_values(item)


def _storage_cdata(tensor: torch.Tensor):
    try:
        storage = tensor.untyped_storage()
    except Exception:
        return None
    return getattr(storage, "_cdata", None)


def is_view_node(node: fx.Node) -> bool:
    """Return True when node output shares storage with any input tensor."""
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return False

    val_cdata = _storage_cdata(val)
    if val_cdata is None:
        return False

    for input_node in node.all_input_nodes:
        input_val = input_node.meta.get("val")
        for tensor in _iter_tensor_like_values(input_val):
            if _storage_cdata(tensor) == val_cdata:
                return True
    return False
