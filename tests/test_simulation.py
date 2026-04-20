import operator
import shutil

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from toolkit.capture import capture_graphs, capture_inductor_graphs
from toolkit.models import ModelRegistry
from toolkit.simulation import estimate_from_config, estimate_graph_peak, estimate_inductor_training_peak, estimate_training_peak


CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = shutil.which("ptxas") is not None
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

requires_inductor = pytest.mark.skipif(
    not CUDA_AVAILABLE or not TRITON_AVAILABLE,
    reason="requires GPU + ptxas (Triton compiler)",
)
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
    # FW: no grad (set_to_none=True); BW: grad modelled inside graph
    fw_total = result["param_bytes"] + result["optimizer_bytes"] + result["fw_peak_bytes"]
    bw_total = (result["param_bytes"]
                + result["optimizer_bytes"] + result["bw_peak_bytes"])
    assert result["estimated_peak"] == max(fw_total, bw_total)


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
    assert r["fw_peak"] == base + r["fw_graph_peak"]
    assert r["bw_peak"] == base + r["bw_graph_peak"]
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

    assert r["l3_true_peak"] <= r["true_peak"], (
        f"L3 ({r['l3_true_peak']}) should be <= L2 ({r['true_peak']})"
    )


@requires_inductor
def test_l25_between_l2_and_l3():
    """L2.5 hybrid peak: L3 <= L2.5 <= L2 (fusion-aware FW + Scheduler BW)."""
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

    assert l25 <= l2, f"L2.5 ({l25}) should be <= L2 ({l2})"
    assert l3 <= l25, f"L3 ({l3}) should be <= L2.5 ({l25})"
    assert l25 < l2, f"L2.5 ({l25}) should be strictly < L2 ({l2})"
