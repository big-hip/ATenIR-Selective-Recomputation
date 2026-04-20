from dataclasses import dataclass
from typing import Union

from .step_profiler import PhaseResult, StepResult


@dataclass
class ValidationResult:
    tag: str
    static_peak: int
    runtime_peak: int
    runtime_reserved: int
    mre_allocated: float
    mre_reserved: float
    run_mode: str
    direction: str
    breakdown: dict


def _pick(mapping: dict, *keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Missing any of keys {keys} in static result")


def validate(static_result: dict, runtime_result: Union[StepResult, PhaseResult], run_mode: str = "compiled") -> ValidationResult:
    static_peak = _pick(static_result, "true_peak", "estimated_peak", "total")

    is_phased = isinstance(runtime_result, PhaseResult)
    runtime_peak = runtime_result.overall_peak if is_phased else runtime_result.peak_allocated
    runtime_reserved = getattr(runtime_result, "peak_reserved", 0)
    mre_allocated = abs(static_peak - runtime_peak) / runtime_peak if runtime_peak > 0 else float("inf")
    mre_reserved = abs(static_peak - runtime_reserved) / runtime_reserved if runtime_reserved > 0 else float("inf")
    direction = "over" if static_peak > runtime_peak else "under"

    breakdown = {
        "static_param": _pick(static_result, "param_bytes", "param"),
        "static_grad": _pick(static_result, "grad_bytes", "grad"),
        "static_optim": _pick(static_result, "optimizer_bytes", "optim_bytes", "optim"),
        "static_act": static_result.get("act_peak", static_result.get("fwbw_peak", 0)),
        "static_fw_peak": _pick(static_result, "fw_peak_bytes", "fw_graph_peak", "fw_peak"),
        "static_bw_peak": _pick(static_result, "bw_peak_bytes", "bw_graph_peak", "bw_peak"),
        "runtime_peak": runtime_peak,
        "runtime_base": runtime_result.base_allocated,
        "runtime_act_delta": runtime_result.activation_delta,
        "allocator_overhead": runtime_reserved - runtime_peak if runtime_reserved else 0,
    }

    # Phased MRE breakdown (when both static and runtime have phased data)
    if is_phased and "fw_peak" in static_result:
        for phase in ("fw_peak", "bw_peak", "opt_peak"):
            s_val = static_result.get(phase, 0)
            r_val = getattr(runtime_result, phase, 0)
            if r_val > 0:
                breakdown[f"mre_{phase}"] = abs(s_val - r_val) / r_val

    return ValidationResult(
        tag=run_mode,
        static_peak=static_peak,
        runtime_peak=runtime_peak,
        runtime_reserved=runtime_reserved,
        mre_allocated=mre_allocated,
        mre_reserved=mre_reserved,
        run_mode=run_mode,
        direction=direction,
        breakdown=breakdown,
    )


def analyze_error_sources(static_result: dict, runtime_result: Union[StepResult, PhaseResult]) -> dict:
    static_peak = _pick(static_result, "true_peak", "estimated_peak", "total")
    static_fixed = (
        _pick(static_result, "param_bytes", "param")
        + _pick(static_result, "grad_bytes", "grad")
        + _pick(static_result, "optimizer_bytes", "optim_bytes", "optim")
    )
    static_act = static_result.get("act_peak", static_result.get("fwbw_peak", 0))

    is_phased = isinstance(runtime_result, PhaseResult)
    runtime_peak = runtime_result.overall_peak if is_phased else runtime_result.peak_allocated
    runtime_base = runtime_result.base_allocated
    runtime_act = runtime_result.activation_delta
    runtime_reserved = getattr(runtime_result, "peak_reserved", 0)
    allocator_overhead = runtime_reserved - runtime_peak if runtime_reserved else 0
    total_error = static_peak - runtime_peak

    return {
        "total_error": total_error,
        "total_error_pct": total_error / runtime_peak * 100 if runtime_peak > 0 else 0,
        "sources": [
            ("fixed (param+grad+optim) vs base", static_fixed - runtime_base),
            ("activation (static vs delta)", static_act - runtime_act),
            ("allocator overhead (reserved-alloc)", allocator_overhead),
        ],
    }
