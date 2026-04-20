import copy
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified
from torch._functorch.partitioners import default_partition


def capture_graphs(
    model: nn.Module,
    sample_input_ids: torch.Tensor,
    loss_fn: Callable,
    partition_fn=default_partition,
    dynamic: bool = True,
    model_kwargs: dict | None = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """Capture training forward/backward graphs through torch.compile + AOTAutograd."""
    if not model.training:
        raise ValueError("capture_graphs requires model.train() before compile")
    if model_kwargs is None:
        model_kwargs = {}

    captured: dict[str, fx.GraphModule] = {}

    def _backend(gm, example_inputs):
        def fw_compiler(fw_gm, _inputs):
            try:
                captured["fw"] = copy.deepcopy(fw_gm)
            except Exception:
                captured["fw"] = fw_gm
            return make_boxed_func(fw_gm.forward)

        def bw_compiler(bw_gm, _inputs):
            try:
                captured["bw"] = copy.deepcopy(bw_gm)
            except Exception:
                captured["bw"] = bw_gm
            return make_boxed_func(bw_gm.forward)

        return aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )

    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, backend=_backend, dynamic=dynamic)
        output = compiled(sample_input_ids, **model_kwargs)
        loss = loss_fn(output)
        loss.backward()

        fw_gm = captured.get("fw")
        bw_gm = captured.get("bw")
        if fw_gm is None or bw_gm is None:
            raise RuntimeError("Failed to capture both fw and bw graph modules")
        return fw_gm, bw_gm
    finally:
        torch._dynamo.reset()
