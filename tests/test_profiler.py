from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from toolkit.capture import capture_graphs
from toolkit.models import ModelRegistry
from toolkit.profiler import PhaseResult, StepResult, analyze_error_sources, measure_phased, measure_step, validate
import toolkit.profiler.step_profiler as step_profiler_module


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
BATCH = 2
SEQ = 32


class _DummyLoss:
    def backward(self):
        return None


class _DummyOptimizer:
    def zero_grad(self, set_to_none=True):
        return None

    def step(self):
        return None


class _FakeEvent:
    _counter = 0

    def __init__(self, enable_timing=True):
        self.index = _FakeEvent._counter
        _FakeEvent._counter += 1

    def record(self):
        return None

    def elapsed_time(self, other):
        return float(other.index - self.index)


class CEModelWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, labels: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.labels = labels

    def forward(self, input_ids):
        return self.base_model(input_ids, labels=self.labels)


def _install_fake_measure_step_env(monkeypatch, *, bases, peak_allocs, peak_reserved=None):
    if peak_reserved is None:
        peak_reserved = peak_allocs

    base_iter = iter(bases)
    peak_alloc_iter = iter(peak_allocs)
    peak_res_iter = iter(peak_reserved)
    _FakeEvent._counter = 0

    monkeypatch.setattr(step_profiler_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(step_profiler_module.torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(step_profiler_module.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(step_profiler_module.torch.cuda, "memory_allocated", lambda device=None: next(base_iter))
    monkeypatch.setattr(
        step_profiler_module.torch.cuda,
        "memory_stats",
        lambda device=None: {
            "allocated_bytes.all.peak": next(peak_alloc_iter),
            "reserved_bytes.all.peak": next(peak_res_iter),
        },
    )
    monkeypatch.setattr(step_profiler_module.torch.cuda, "Event", _FakeEvent)


def _install_fake_measure_phased_env(monkeypatch, *, bases, max_peaks, after_values):
    base_iter = iter(bases)
    max_peak_iter = iter(max_peaks)
    after_iter = iter(after_values)
    _FakeEvent._counter = 0

    monkeypatch.setattr(step_profiler_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(step_profiler_module.torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(step_profiler_module.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(step_profiler_module.torch.cuda, "memory_allocated", lambda device=None: next(base_iter if False else after_iter))
    base_values = list(bases)
    after_values = list(after_values)
    state = {"base_idx": 0, "after_idx": 0, "expect_base": True}

    def fake_memory_allocated(device=None):
        if state["expect_base"]:
            value = base_values[state["base_idx"]]
            state["base_idx"] += 1
            state["expect_base"] = False
            return value
        value = after_values[state["after_idx"]]
        state["after_idx"] += 1
        if state["after_idx"] % 3 == 0:
            state["expect_base"] = True
        return value

    monkeypatch.setattr(step_profiler_module.torch.cuda, "memory_allocated", fake_memory_allocated)
    monkeypatch.setattr(step_profiler_module.torch.cuda, "max_memory_allocated", lambda device=None: next(max_peak_iter))
    monkeypatch.setattr(step_profiler_module.torch.cuda, "Event", _FakeEvent)


def test_measure_step_base_collected_per_repeat(monkeypatch):
    _install_fake_measure_step_env(
        monkeypatch,
        bases=[100, 200, 300],
        peak_allocs=[110, 220, 330],
        peak_reserved=[120, 240, 360],
    )

    result = measure_step(
        "dummy",
        lambda: _DummyLoss(),
        _DummyOptimizer(),
        repeats=3,
        warmup=0,
        device="cuda",
    )

    assert result.base_allocated == 200


def test_measure_step_uses_iqr_mean(monkeypatch):
    _install_fake_measure_step_env(
        monkeypatch,
        bases=[10, 10, 10, 10, 10, 10],
        peak_allocs=[10, 12, 1000, 11, 9, 13],
        peak_reserved=[20, 22, 2000, 21, 19, 23],
    )

    result = measure_step(
        "iqr",
        lambda: _DummyLoss(),
        _DummyOptimizer(),
        repeats=6,
        warmup=0,
        device="cuda",
    )

    assert result.peak_allocated == int((10 + 11 + 12 + 13) / 4)
    assert result.peak_reserved == int((20 + 21 + 22 + 23) / 4)


def test_measure_phased_overall_peak_is_max_of_three_phases(monkeypatch):
    _install_fake_measure_phased_env(
        monkeypatch,
        bases=[50],
        max_peaks=[100, 200, 150],
        after_values=[60, 70, 65],
    )

    result = measure_phased(
        "phased",
        lambda: _DummyLoss(),
        _DummyOptimizer(),
        repeats=1,
        warmup=0,
        device="cuda",
    )

    assert result.overall_peak == 200
    assert result.activation_delta == 150


def test_validate_defaults_to_compiled():
    runtime_result = StepResult(
        name="runtime",
        peak_allocated=100,
        peak_reserved=120,
        base_allocated=60,
        activation_delta=40,
        elapsed_ms=1.0,
        fw_ms=0.3,
        bw_ms=0.4,
        opt_ms=0.3,
    )
    static_result = {
        "estimated_peak": 110,
        "param_bytes": 20,
        "grad_bytes": 20,
        "optimizer_bytes": 40,
        "act_peak": 30,
        "fw_peak_bytes": 25,
        "bw_peak_bytes": 30,
    }

    result = validate(static_result, runtime_result)

    assert result.run_mode == "compiled"
    assert result.direction == "over"
    assert result.breakdown["runtime_base"] == 60


def test_analyze_error_sources_uses_buffer_and_absolute_phase_peaks():
    runtime_result = PhaseResult(
        name="runtime",
        fw_peak=100,
        bw_peak=150,
        opt_peak=130,
        after_fw=90,
        after_bw=120,
        after_opt=120,
        base_allocated=60,
        overall_peak=150,
        activation_delta=90,
        fw_ms=0.3,
        bw_ms=0.4,
        opt_ms=0.3,
        step_ms=1.0,
    )
    static_result = {
        "true_peak": 170,
        "fw_peak": 110,
        "bw_peak": 170,
        "opt_peak": 125,
        "param_bytes": 20,
        "buffer_bytes": 5,
        "grad_bytes": 20,
        "optimizer_bytes": 40,
    }

    result = analyze_error_sources(static_result, runtime_result)

    assert result["sources"][0] == ("fixed (param+optim+buffer) vs base", 5)
    assert result["sources"][1] == ("activation phase peak error", 20)
    assert result["phase_errors"] == {"fw_peak": 10, "bw_peak": 20, "opt_peak": -5}


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires GPU")
def test_capture_and_profiler_share_same_loss_contract():
    reg = ModelRegistry()
    input_ids = torch.randint(0, reg.get_config("gpt2").vocab_size, (BATCH, SEQ), device=DEVICE)
    loss_fn = reg.default_loss_fn("gpt2")

    base_model = reg.create_model("gpt2").to(DEVICE).train()
    wrapped_model = CEModelWrapper(base_model, input_ids).train()
    fw_gm, bw_gm = capture_graphs(wrapped_model, input_ids, loss_fn)

    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=1e-4)
    step_result = measure_step(
        "gpt2_wrapped",
        lambda: loss_fn(wrapped_model(input_ids)),
        optimizer,
        repeats=1,
        warmup=0,
        device=DEVICE,
    )

    assert fw_gm is not None and bw_gm is not None
    assert loss_fn(wrapped_model(input_ids)).ndim == 0
    assert step_result.peak_allocated > 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires GPU")
def test_capture_and_profiler_share_training_mode():
    reg = ModelRegistry()
    input_ids = torch.randint(0, reg.get_config("gpt2").vocab_size, (BATCH, SEQ), device=DEVICE)
    loss_fn = reg.default_loss_fn("gpt2")

    base_model = reg.create_model("gpt2").to(DEVICE).train()
    wrapped_model = CEModelWrapper(base_model, input_ids).train()
    capture_graphs(wrapped_model, input_ids, loss_fn)

    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=1e-4)
    measure_step(
        "gpt2_train_mode",
        lambda: loss_fn(wrapped_model(input_ids)),
        optimizer,
        repeats=1,
        warmup=0,
        device=DEVICE,
    )

    assert wrapped_model.training is True
    assert base_model.training is True
