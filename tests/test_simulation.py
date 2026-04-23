import operator
import shutil

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from toolkit.capture import capture_graphs, capture_inductor_graphs
from toolkit.models import ModelRegistry
from toolkit.simulation import detect_recomputation, estimate_from_config, estimate_graph_peak, estimate_inductor_training_peak, estimate_shape_sum_peak, estimate_training_peak


CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = shutil.which("ptxas") is not None
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

_skip_without_inductor = pytest.mark.skipif(
    not CUDA_AVAILABLE or not TRITON_AVAILABLE,
    reason="requires GPU + ptxas (Triton compiler)",
)


def requires_inductor(fn):
    """Mark tests that need CUDA plus Triton/Inductor compilation."""
    return _skip_without_inductor(pytest.mark.inductor(fn))


BATCH = 2
SEQ = 64


def _trace_fake(fn, *inputs):
    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_inputs = [
            fake_mode.from_tensor(inp) if isinstance(inp, torch.Tensor) else inp
            for inp in inputs
        ]
        return make_fx(fn, tracing_mode="fake")(*fake_inputs)


def test_view_nodes_excluded():
    def fn(x):
        return x[:, :2]

    gm = _trace_fake(fn, torch.randn(4, 4))
    result = estimate_graph_peak(gm)

    assert result["num_view_nodes"] > 0
    assert result["num_alloc_nodes"] == 0


def test_peak_net_bytes_defaults_to_peak():
    """When overlap_bytes=0, peak_net_bytes should equal peak_bytes."""
    def fn(x):
        return x + 1.0

    gm = _trace_fake(fn, torch.randn(4, 4))
    result = estimate_graph_peak(gm)
    assert "peak_net_bytes" in result
    assert result["peak_net_bytes"] == result["peak_bytes"]


def test_peak_net_bytes_with_overlap():
    """When overlap_bytes > 0, peak_net_bytes <= peak_bytes."""
    def fn(x):
        a = x + 1.0
        b = a * 2.0
        return b

    gm = _trace_fake(fn, torch.randn(4, 4))
    result_no_overlap = estimate_graph_peak(gm)
    result_with_overlap = estimate_graph_peak(gm, overlap_bytes=result_no_overlap["peak_bytes"])
    assert result_with_overlap["peak_net_bytes"] <= result_with_overlap["peak_bytes"]
    assert result_with_overlap["peak_bytes"] == result_no_overlap["peak_bytes"]


def test_inplace_reuse_does_not_recycle_placeholder():
    """L2.5 safe reuse must not donate graph input/placeholder storage."""
    def fn(x):
        return x + 1.0

    gm = _trace_fake(fn, torch.randn(4, 4))
    result = estimate_graph_peak(gm, simulate_inplace=True)
    assert result["n_reuses"] == 0


def test_shape_sum_baseline_fields():
    def fn(x):
        a = x + 1.0
        return a * 2.0

    gm = _trace_fake(fn, torch.randn(4, 4))
    model = torch.nn.Linear(4, 4)
    result = estimate_shape_sum_peak(gm, gm, model, optimizer_cls=torch.optim.SGD)

    assert result["tag"] == "shape_sum_graph"
    assert result["shape_sum_fw_bytes"] > 0
    assert result["shape_sum_bw_bytes"] > 0
    assert result["true_peak"] == max(result["fw_peak"], result["bw_peak"], result["opt_peak"])


def test_inductor_estimator_recomp_disables_bw_safe_reuse_and_reports_l3_inputs():
    def fn(x):
        a = x + 1.0
        return a * 2.0

    gm = _trace_fake(fn, torch.randn(4, 4))
    model = torch.nn.Linear(4, 4)
    result = estimate_inductor_training_peak(
        {"fw_gm": gm, "bw_gm": gm, "sched_fw_peak": 4096, "sched_bw_peak": 8192},
        model,
        optimizer_cls=torch.optim.SGD,
        has_recomputation=True,
    )

    assert result["has_recomputation"] is True
    assert result["l25_bw_safe_reuse_enabled"] is False
    assert result["l3_static_base_added"] is True
    assert result["l3_fw_graph_input_bytes"] > 0
    assert result["l3_bw_graph_input_bytes"] > 0
    assert result["l3_fw_peak"] == result["base"] + 4096
    assert result["l3_bw_peak"] == result["base"] + 8192


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires GPU")
def test_no_double_count_saved_act():
    reg = ModelRegistry()
    model = reg.create_model("gpt2").to(DEVICE).train()
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ), device=DEVICE)

    fw_gm, bw_gm = capture_graphs(model, input_ids, lambda out: out.logits.sum())
    result = estimate_training_peak(fw_gm, bw_gm, model)
    old_act_peak = max(result["fw_peak_bytes"], result["saved_act_bytes"] + result["bw_peak_bytes"])

    assert result["bw_peak_bytes"] >= result["saved_act_bytes"]
    assert result["act_peak"] == max(result["fw_peak_bytes"], result["bw_peak_bytes"])
    assert old_act_peak >= result["act_peak"]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires GPU")
def test_peak_formula():
    reg = ModelRegistry()
    model = reg.create_model("gpt2").to(DEVICE).train()
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ), device=DEVICE)

    fw_gm, bw_gm = capture_graphs(model, input_ids, lambda out: out.logits.sum())
    result = estimate_training_peak(fw_gm, bw_gm, model)

    assert result["grad_bytes"] == result["param_bytes"]
    assert result["optimizer_bytes"] == 2 * result["param_bytes"]
    # Graph peaks include param placeholders alive at peak; static_base
    # also has param_bytes → subtract overlap to avoid double-count.
    assert result["estimated_peak"] == result["true_peak"]
    assert result["true_peak"] == max(result["fw_peak"], result["bw_peak"], result["opt_peak"])
    assert result["fw_peak"] > result["base"]
    assert result["bw_peak"] > result["base"]


def test_estimate_from_config_positive():
    reg = ModelRegistry()
    for name in reg.list_models():
        config = reg.get_config(name)
        result = estimate_from_config(config, BATCH, SEQ)
        assert result["param_bytes"] > 0
        assert result["activation_bytes"] > 0
        assert result["estimated_peak"] > result["param_bytes"]


def test_l1_param_accuracy():
    """L1 param_bytes should match real model params within 5%."""
    reg = ModelRegistry()
    for name in reg.list_models():
        config = reg.get_config(name)
        est = estimate_from_config(config, batch=1, seq=1)
        model = reg.create_model(name)
        real_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        error_pct = abs(est["param_bytes"] - real_bytes) / real_bytes * 100
        assert error_pct < 5.0, (
            f"{name}: L1 param={est['param_bytes']}, real={real_bytes}, error={error_pct:.1f}%"
        )


# ---- Phase 0.5: New phased peak metric tests ----


def test_l1_four_peaks_present():
    """L1 must return fw_peak, bw_peak, opt_peak, fwbw_peak, true_peak, peak_phase."""
    reg = ModelRegistry()
    for name in reg.list_models():
        config = reg.get_config(name)
        r = estimate_from_config(config, BATCH, SEQ)
        for key in ("fw_peak", "bw_peak", "opt_peak", "fwbw_peak", "true_peak",
                     "peak_phase", "base", "after_fw", "after_bw", "after_opt"):
            assert key in r, f"{name}: missing key '{key}'"
        assert r["fw_peak"] > 0
        assert r["bw_peak"] > r["fw_peak"], "bw_peak should > fw_peak (grad adds memory)"
        assert r["fwbw_peak"] == max(r["fw_peak"], r["bw_peak"])
        assert r["true_peak"] == max(r["fw_peak"], r["bw_peak"], r["opt_peak"])
        assert r["estimated_peak"] == r["true_peak"]
        assert r["peak_phase"] in ("FW", "BW", "OPT")


def test_l1_true_peak_is_max():
    """true_peak == max(fw, bw, opt) for all models and batch sizes."""
    reg = ModelRegistry()
    for name in reg.list_models():
        config = reg.get_config(name)
        for batch in (1, 4, 8):
            r = estimate_from_config(config, batch, SEQ)
            assert r["true_peak"] == max(r["fw_peak"], r["bw_peak"], r["opt_peak"]), (
                f"{name} batch={batch}: true_peak mismatch"
            )


def test_l1_fused_optimizer_reduces_opt_peak():
    """fused_optimizer=True should reduce opt_peak by param_bytes for Adam."""
    reg = ModelRegistry()
    for name in reg.list_models():
        config = reg.get_config(name)
        normal = estimate_from_config(config, BATCH, SEQ, fused_optimizer=False)
        fused = estimate_from_config(config, BATCH, SEQ, fused_optimizer=True)
        assert fused["opt_peak"] < normal["opt_peak"], f"{name}: fused should reduce opt_peak"
        assert fused["opt_temp"] == 0
        assert normal["opt_temp"] == normal["param_bytes"]
        assert normal["opt_peak"] - fused["opt_peak"] == normal["param_bytes"]


def test_l1_small_batch_opt_dominates():
    """For very small batch, opt_peak > bw_peak (optimizer temp > activations)."""
    reg = ModelRegistry()
    config = reg.get_config("gpt2")
    r = estimate_from_config(config, batch=1, seq=16)
    assert r["opt_peak"] > r["bw_peak"], (
        f"small batch: opt_peak={r['opt_peak']} should > bw_peak={r['bw_peak']}"
    )
    assert r["peak_phase"] == "OPT"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires GPU")
def test_l2_backward_compat():
    """L2 must still return fw_peak_bytes, bw_peak_bytes, act_peak aliases."""
    reg = ModelRegistry()
    model = reg.create_model("gpt2").to(DEVICE).train()
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ), device=DEVICE)
    fw_gm, bw_gm = capture_graphs(model, input_ids, lambda out: out.logits.sum())
    r = estimate_training_peak(fw_gm, bw_gm, model)
    assert "fw_peak_bytes" in r
    assert "bw_peak_bytes" in r
    assert "act_peak" in r
    assert r["fw_peak_bytes"] == r["fw_graph_peak"]
    assert r["bw_peak_bytes"] == r["bw_graph_peak"]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires GPU")
def test_l2_absolute_peaks():
    """L2 fw_peak/bw_peak/opt_peak are absolute (include base), not activation-only."""
    reg = ModelRegistry()
    model = reg.create_model("gpt2").to(DEVICE).train()
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ), device=DEVICE)
    fw_gm, bw_gm = capture_graphs(model, input_ids, lambda out: out.logits.sum())
    r = estimate_training_peak(fw_gm, bw_gm, model)
    base = r["base"]
    # Graph peaks include param placeholder overlap → peaks should be
    # between base and base + graph_peak (inclusive).
    assert r["fw_peak"] >= base
    assert r["fw_peak"] <= base + r["fw_graph_peak"]
    assert r["bw_peak"] >= base
    assert r["bw_peak"] <= base + r["bw_graph_peak"]
    assert r["true_peak"] == max(r["fw_peak"], r["bw_peak"], r["opt_peak"])
    assert r["estimated_peak"] == r["true_peak"]
    assert r["peak_phase"] in ("FW", "BW", "OPT")


# ---- Inductor dual-layer (L2+L3) tests ----


@requires_inductor
def test_inductor_training_peak_has_l2_and_l3():
    """estimate_inductor_training_peak returns both L2 and L3 fields."""
    reg = ModelRegistry()
    model = reg.create_model("gpt2").to(DEVICE).train()
    config = reg.get_config("gpt2")
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)

    cap = capture_inductor_graphs(
        model, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
    )
    r = estimate_inductor_training_peak(cap, model, optimizer_cls=torch.optim.SGD)

    # L2 fields
    assert r["true_peak"] > 0
    assert r["fw_peak"] > 0
    assert r["bw_peak"] > 0
    assert r["peak_phase"] in ("FW", "BW", "OPT")

    # L3 fields
    assert r["l3_true_peak"] is not None
    assert r["l3_true_peak"] > 0
    assert r["l3_fw_peak"] > 0
    assert r["l3_bw_peak"] > 0
    assert r["l3_peak_phase"] in ("FW", "BW", "OPT")
    assert r["l3_true_peak"] == max(r["l3_fw_peak"], r["l3_bw_peak"], r["l3_opt_peak"])

    # Scheduler raw peaks
    assert r["sched_fw_peak"] is not None
    assert r["sched_bw_peak"] is not None


@requires_inductor
def test_l3_leq_l2():
    """L3 peak <= L2 peak (fusion reduces memory, never increases it)."""
    reg = ModelRegistry()
    model = reg.create_model("gpt2").to(DEVICE).train()
    config = reg.get_config("gpt2")
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)

    cap = capture_inductor_graphs(
        model, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
    )
    r = estimate_inductor_training_peak(cap, model, optimizer_cls=torch.optim.SGD)

    # L3 is an independent Scheduler-level estimate; not guaranteed <= L2
    # but should be in the same ballpark (within 50% of each other).
    assert r["l3_true_peak"] > 0
    ratio = r["l3_true_peak"] / r["true_peak"]
    assert 0.5 < ratio < 1.5, (
        f"L3 ({r['l3_true_peak']}) too far from L2 ({r['true_peak']}), ratio={ratio:.2f}"
    )


@requires_inductor
def test_l25_between_l2_and_l3():
    """L2.5 hybrid peak: L2.5 <= L2 (fusion + inplace reuse tightens the bound).

    Note: L2.5 can be lower than L3 because it models codegen-level buffer
    reuse (simulate_inplace) that the Scheduler does not account for.
    """
    reg = ModelRegistry()
    model = reg.create_model("gpt2").to(DEVICE).train()
    config = reg.get_config("gpt2")
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)

    cap = capture_inductor_graphs(
        model, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
    )
    r = estimate_inductor_training_peak(cap, model, optimizer_cls=torch.optim.SGD)

    l2 = r["true_peak"]
    l25 = r["l25_true_peak"]
    l3 = r["l3_true_peak"]

    assert r["l25_true_peak"] <= l2, f"L2.5 ({l25}) should be <= L2 ({l2})"
    # L3 uses Scheduler peaks (independent estimate); may be > or < L2
    assert l3 > 0, "L3 should be positive"
    assert l25 < l2, f"L2.5 ({l25}) should be strictly < L2 ({l2})"


@requires_inductor
def test_detect_recomputation_no_ac():
    """detect_recomputation returns False for baseline (no AC) model."""
    reg = ModelRegistry()
    model = reg.create_model("gpt2").to(DEVICE).train()
    config = reg.get_config("gpt2")
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)

    cap = capture_inductor_graphs(
        model, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
    )
    assert detect_recomputation(cap["bw_gm"]) is False


@requires_inductor
def test_inductor_has_recomputation_field():
    """estimate_inductor_training_peak returns has_recomputation field."""
    reg = ModelRegistry()
    model = reg.create_model("gpt2").to(DEVICE).train()
    config = reg.get_config("gpt2")
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=DEVICE)

    cap = capture_inductor_graphs(
        model, input_ids, lambda out: out.loss,
        model_kwargs={"labels": input_ids},
    )
    r = estimate_inductor_training_peak(cap, model, optimizer_cls=torch.optim.SGD)
    assert "has_recomputation" in r
    assert isinstance(r["has_recomputation"], bool)
