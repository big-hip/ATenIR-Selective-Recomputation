from dataclasses import dataclass

import torch


def iqr_mean(values):
    if not values:
        raise ValueError("iqr_mean requires at least one value")
    if len(values) <= 2:
        return sum(values) / len(values)
    ordered = sorted(values)
    n = len(ordered)
    q1 = n // 4
    q3 = n - n // 4
    middle = ordered[q1:q3]
    return sum(middle) / len(middle) if middle else sum(ordered) / n


@dataclass
class StepResult:
    name: str
    peak_allocated: int
    peak_reserved: int
    base_allocated: int
    activation_delta: int
    elapsed_ms: float
    fw_ms: float
    bw_ms: float
    opt_ms: float


@dataclass
class PhaseResult:
    name: str
    fw_peak: int
    bw_peak: int
    opt_peak: int
    after_fw: int
    after_bw: int
    after_opt: int
    base_allocated: int
    overall_peak: int
    activation_delta: int
    fw_ms: float
    bw_ms: float
    opt_ms: float
    step_ms: float
    bw_fw_delta: int = 0
    fwbw_peak: int = 0
    peak_phase: str = ""


def measure_step(name, forward_fn, optimizer, *, repeats=6, warmup=2, device="cuda") -> StepResult:
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_fn()
        loss.backward()
        optimizer.step()

    peaks_alloc = []
    peaks_reserved = []
    bases = []
    deltas = []
    total_ms = []
    fw_ms = []
    bw_ms = []
    opt_ms = []

    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated(device)

        ev_s, ev_f, ev_b, ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

        ev_s.record()
        loss = forward_fn()
        ev_f.record()

        loss.backward()
        ev_b.record()

        optimizer.step()
        ev_e.record()
        torch.cuda.synchronize()

        stats = torch.cuda.memory_stats(device)
        peak_alloc = stats["allocated_bytes.all.peak"]
        peak_reserved = stats["reserved_bytes.all.peak"]

        peaks_alloc.append(peak_alloc)
        peaks_reserved.append(peak_reserved)
        bases.append(base)
        deltas.append(peak_alloc - base)

        total_ms.append(ev_s.elapsed_time(ev_e))
        fw_ms.append(ev_s.elapsed_time(ev_f))
        bw_ms.append(ev_f.elapsed_time(ev_b))
        opt_ms.append(ev_b.elapsed_time(ev_e))

    return StepResult(
        name=name,
        peak_allocated=int(iqr_mean(peaks_alloc)),
        peak_reserved=int(iqr_mean(peaks_reserved)),
        base_allocated=int(sum(bases) / len(bases)),
        activation_delta=int(iqr_mean(deltas)),
        elapsed_ms=iqr_mean(total_ms),
        fw_ms=iqr_mean(fw_ms),
        bw_ms=iqr_mean(bw_ms),
        opt_ms=iqr_mean(opt_ms),
    )


def measure_phased(name, forward_fn, optimizer, *, repeats=5, warmup=3, device="cuda") -> PhaseResult:
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = forward_fn()
        loss.backward()
        optimizer.step()

    results = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated(device)

        ev_s, ev_f, ev_b, ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

        torch.cuda.reset_peak_memory_stats(device)
        ev_s.record()
        loss = forward_fn()
        torch.cuda.synchronize()
        fw_peak = torch.cuda.max_memory_allocated(device)
        after_fw = torch.cuda.memory_allocated(device)

        torch.cuda.reset_peak_memory_stats(device)
        ev_f.record()
        loss.backward()
        torch.cuda.synchronize()
        bw_peak = torch.cuda.max_memory_allocated(device)
        after_bw = torch.cuda.memory_allocated(device)

        torch.cuda.reset_peak_memory_stats(device)
        ev_b.record()
        optimizer.step()
        ev_e.record()
        torch.cuda.synchronize()
        opt_peak = torch.cuda.max_memory_allocated(device)
        after_opt = torch.cuda.memory_allocated(device)

        overall_peak = max(fw_peak, bw_peak, opt_peak)
        results.append(
            PhaseResult(
                name=name,
                fw_peak=fw_peak,
                bw_peak=bw_peak,
                opt_peak=opt_peak,
                after_fw=after_fw,
                after_bw=after_bw,
                after_opt=after_opt,
                base_allocated=base,
                overall_peak=overall_peak,
                activation_delta=overall_peak - base,
                fw_ms=ev_s.elapsed_time(ev_f),
                bw_ms=ev_f.elapsed_time(ev_b),
                opt_ms=ev_b.elapsed_time(ev_e),
                step_ms=ev_s.elapsed_time(ev_e),
            )
        )

    agg_fw = int(iqr_mean([r.fw_peak for r in results]))
    agg_bw = int(iqr_mean([r.bw_peak for r in results]))
    agg_opt = int(iqr_mean([r.opt_peak for r in results]))
    agg_grad = int(iqr_mean([r.after_bw - r.after_fw for r in results]))
    fwbw = max(agg_fw, agg_bw)
    overall = max(agg_fw, agg_bw, agg_opt)
    if overall == agg_fw:
        _peak_phase = "FW"
    elif overall == agg_bw:
        _peak_phase = "BW"
    else:
        _peak_phase = "OPT"

    return PhaseResult(
        name=name,
        fw_peak=agg_fw,
        bw_peak=agg_bw,
        opt_peak=agg_opt,
        after_fw=int(iqr_mean([r.after_fw for r in results])),
        after_bw=int(iqr_mean([r.after_bw for r in results])),
        after_opt=int(iqr_mean([r.after_opt for r in results])),
        base_allocated=int(sum(r.base_allocated for r in results) / len(results)),
        overall_peak=overall,
        activation_delta=int(iqr_mean([r.activation_delta for r in results])),
        fw_ms=iqr_mean([r.fw_ms for r in results]),
        bw_ms=iqr_mean([r.bw_ms for r in results]),
        opt_ms=iqr_mean([r.opt_ms for r in results]),
        step_ms=iqr_mean([r.step_ms for r in results]),
        bw_fw_delta=agg_grad,
        fwbw_peak=fwbw,
        peak_phase=_peak_phase,
    )
