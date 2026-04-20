import copy
import logging
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn

from toolkit.strategy import clear_memory_budget, set_memory_budget

logger = logging.getLogger(__name__)


def capture_inductor_graphs(
    model: nn.Module,
    sample_input_ids: torch.Tensor,
    loss_fn: Callable,
    *,
    model_kwargs: dict | None = None,
    dynamic: bool = True,
    budget: float | None = None,
) -> dict:
    """Capture inductor post-grad FX graphs + Scheduler L3 peak estimates.

    Uses ``compile_fx(inner_compile=hook)`` to intercept post-grad FW/BW
    ``GraphModule`` (ATen IR after decomposition + post_grad_passes), and
    monkey-patches ``Scheduler.__init__`` to call ``estimate_peak_memory``
    for L3 (Scheduler-level) peak estimation.

    Args:
        model: model in training mode
        sample_input_ids: input tensor (typically token ids)
        loss_fn: callable that takes model output and returns a scalar loss
        model_kwargs: extra kwargs passed to the model (e.g. labels=...)
        dynamic: whether to use dynamic shapes in torch.compile
        budget: activation_memory_budget for the AOT partitioner (None=default)

    Returns:
        dict with keys:
            "fw_gm": fx.GraphModule     - post-grad FW graph (ATen IR)
            "bw_gm": fx.GraphModule     - post-grad BW graph (ATen IR)
            "sched_fw_peak": int | None - Scheduler FW peak bytes
            "sched_bw_peak": int | None - Scheduler BW peak bytes
    """
    if not model.training:
        raise ValueError("capture_inductor_graphs requires model.train()")
    if model_kwargs is None:
        model_kwargs = {}

    # -- State containers --
    captured: dict[str, fx.GraphModule] = {}
    sched_peaks: dict[str, int] = {}
    current_phase: list[str] = ["fw"]  # mutable container for closure

    # -- Import inductor internals --
    from torch._inductor.compile_fx import compile_fx, compile_fx_inner
    from torch._inductor.scheduler import Scheduler
    import torch._inductor.config as inductor_config

    # -- Save originals for restoration --
    orig_scheduler_init = Scheduler.__init__
    orig_force_disable_caches = inductor_config.force_disable_caches

    # -- inner_compile hook: intercept post-grad FX graphs --
    def my_inner_compile(gm, example_inputs, **kwargs):
        phase = "bw" if kwargs.get("is_backward", False) else "fw"
        current_phase[0] = phase
        try:
            captured[phase] = copy.deepcopy(gm)
        except Exception:
            captured[phase] = gm
        return compile_fx_inner(gm, example_inputs, **kwargs)

    # -- Scheduler hook: call estimate_peak_memory after initialization --
    def patched_scheduler_init(self, nodes):
        orig_scheduler_init(self, nodes)
        phase = current_phase[0]
        if phase in sched_peaks:
            return  # avoid duplicate for same phase
        try:
            from torch._inductor.virtualized import V
            from torch._inductor.memory import (
                get_freeable_input_buf,
                assign_memory_planning_info_for_scheduler_buffers,
                assign_memory_planning_info_for_scheduler_nodes,
                estimate_peak_memory,
            )
            graph_outputs = set(V.graph.get_output_names())
            graph_inputs = set(V.graph.graph_inputs.keys())

            n2fib = get_freeable_input_buf(self.nodes, graph_inputs)
            assign_memory_planning_info_for_scheduler_buffers(
                self.nodes, self.name_to_buf
            )
            assign_memory_planning_info_for_scheduler_nodes(
                self.nodes, self.name_to_fused_node,
                self.name_to_buf, n2fib,
            )
            peak, _memories = estimate_peak_memory(
                self.nodes, n2fib, graph_outputs
            )
            sched_peaks[phase] = peak
            logger.debug(
                "Scheduler L3 peak [%s]: %d bytes (%.1f MB)",
                phase, peak, peak / (1024 ** 2),
            )
        except Exception as e:
            logger.warning("Scheduler hook failed for phase %s: %s", phase, e)

    # -- Apply patches and compile --
    Scheduler.__init__ = patched_scheduler_init
    inductor_config.force_disable_caches = True

    if budget is not None:
        set_memory_budget(budget)

    torch._dynamo.reset()
    try:
        def backend(gm, example_inputs):
            return compile_fx(gm, example_inputs, inner_compile=my_inner_compile)

        compiled = torch.compile(model, backend=backend, dynamic=dynamic)
        output = compiled(sample_input_ids, **model_kwargs)
        loss = loss_fn(output)
        loss.backward()

        fw_gm = captured.get("fw")
        bw_gm = captured.get("bw")
        if fw_gm is None or bw_gm is None:
            raise RuntimeError(
                f"Failed to capture inductor graphs. "
                f"Captured phases: {list(captured.keys())}"
            )

        return {
            "fw_gm": fw_gm,
            "bw_gm": bw_gm,
            "sched_fw_peak": sched_peaks.get("fw"),
            "sched_bw_peak": sched_peaks.get("bw"),
        }
    finally:
        Scheduler.__init__ = orig_scheduler_init
        inductor_config.force_disable_caches = orig_force_disable_caches
        if budget is not None:
            clear_memory_budget()
        torch._dynamo.reset()
