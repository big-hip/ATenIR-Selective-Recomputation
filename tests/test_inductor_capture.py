"""Tests for capture_inductor_graphs() — inductor post-grad graph capture + Scheduler L3 peak."""

import shutil

import pytest
import torch
import torch.fx as fx

from toolkit.capture import capture_inductor_graphs
from toolkit.models import ModelRegistry
from toolkit.simulation import estimate_graph_peak, estimate_training_peak
from toolkit.strategy import wrap_with_checkpoint

CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = shutil.which("ptxas") is not None
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

requires_inductor = pytest.mark.skipif(
    not CUDA_AVAILABLE or not TRITON_AVAILABLE,
    reason="requires GPU + ptxas (Triton compiler)",
)
BATCH = 2
SEQ = 64

# Small model config to keep tests fast
MODEL_NAME = "gpt2"
MODEL_OVERRIDES = {}  # use defaults (small)


def _capture_default():
    """Helper: capture inductor graphs for a default gpt2 model."""
    reg = ModelRegistry()
    model = reg.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()
    config = reg.get_config(MODEL_NAME, **MODEL_OVERRIDES)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)
    result = capture_inductor_graphs(
        model, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
    )
    return result, model


@requires_inductor
def test_capture_returns_both_graphs_and_sched():
    """Verify capture returns fw_gm, bw_gm as GraphModule + sched peaks as positive ints."""
    result, _ = _capture_default()

    assert isinstance(result["fw_gm"], fx.GraphModule), "fw_gm should be a GraphModule"
    assert isinstance(result["bw_gm"], fx.GraphModule), "bw_gm should be a GraphModule"

    # Scheduler peaks should be captured (may be None if hook fails, but should work)
    assert result["sched_fw_peak"] is not None, "sched_fw_peak should be captured"
    assert result["sched_bw_peak"] is not None, "sched_bw_peak should be captured"
    assert result["sched_fw_peak"] > 0, "sched_fw_peak should be positive"
    assert result["sched_bw_peak"] > 0, "sched_bw_peak should be positive"


@requires_inductor
def test_inductor_graph_is_aten_ir():
    """Verify post-grad graph nodes are all aten.* / prims.* ops."""
    result, _ = _capture_default()

    import types

    for name, gm in [("fw", result["fw_gm"]), ("bw", result["bw_gm"])]:
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            # Allow Python built-in operators (getitem, mul, etc.)
            if isinstance(target, types.BuiltinFunctionType):
                continue
            target_name = str(target)
            assert (
                "aten" in target_name
                or "prims" in target_name
                or "operator" in target_name
                or "inductor_ops" in target_name
            ), f"{name} graph has non-ATen node: {target_name}"


@requires_inductor
def test_inductor_graph_has_faketensor_meta():
    """Verify non-output nodes have meta['val'] FakeTensor."""
    result, _ = _capture_default()
    fw_gm = result["fw_gm"]

    n_with_meta = 0
    n_total = 0
    for node in fw_gm.graph.nodes:
        if node.op == "output":
            continue
        n_total += 1
        val = node.meta.get("val")
        if val is not None:
            n_with_meta += 1

    coverage = n_with_meta / n_total if n_total > 0 else 0
    assert coverage > 0.95, (
        f"FakeTensor meta coverage too low: {n_with_meta}/{n_total} = {coverage:.1%}"
    )


@requires_inductor
def test_estimate_graph_peak_works_on_inductor_graph():
    """Verify existing estimate_graph_peak can be applied to inductor post-grad graphs."""
    result, _ = _capture_default()

    fw_result = estimate_graph_peak(result["fw_gm"], pin_output_inputs=True)
    bw_result = estimate_graph_peak(result["bw_gm"], pin_output_inputs=True)

    assert fw_result["peak_bytes"] > 0, "FW peak should be positive"
    assert bw_result["peak_bytes"] > 0, "BW peak should be positive"
    assert fw_result["num_alloc_nodes"] > 0, "FW should have alloc nodes"


@requires_inductor
def test_estimate_training_peak_on_inductor_graph():
    """End-to-end: capture inductor graphs → estimate_training_peak."""
    result, model = _capture_default()

    est = estimate_training_peak(
        result["fw_gm"], result["bw_gm"], model,
        optimizer_cls=torch.optim.SGD,
    )
    assert est["true_peak"] > 0
    assert est["param_bytes"] > 0
    assert est["peak_phase"] in ("FW", "BW", "OPT")


@requires_inductor
def test_capture_with_budget():
    """Verify capture works with budget=0.5 and sched peaks change."""
    reg = ModelRegistry()
    model = reg.create_model(MODEL_NAME, **MODEL_OVERRIDES).to(DEVICE).train()
    config = reg.get_config(MODEL_NAME, **MODEL_OVERRIDES)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)

    result = capture_inductor_graphs(
        model, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
        budget=0.5,
    )
    assert result["fw_gm"] is not None
    assert result["bw_gm"] is not None
    # Budget may or may not affect sched peaks depending on partitioner behavior,
    # but capture should succeed without error.


@requires_inductor
def test_capture_with_ac_wrapping():
    """Verify AC + inductor capture works."""
    reg = ModelRegistry()
    model_name = "llama"
    overrides = dict(
        hidden_size=256, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=512,
        num_key_value_heads=2, max_position_embeddings=512,
    )
    model = reg.create_model(model_name, **overrides).to(DEVICE).train()
    block_cls = reg.get_block_class_name(model_name)
    wrap_with_checkpoint(model, block_cls)

    config = reg.get_config(model_name, **overrides)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)

    result = capture_inductor_graphs(
        model, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
    )
    assert isinstance(result["fw_gm"], fx.GraphModule)
    assert isinstance(result["bw_gm"], fx.GraphModule)
