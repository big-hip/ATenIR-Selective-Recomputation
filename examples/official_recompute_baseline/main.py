import copy
import functools

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from model import OfficialRecomputeMLP, device


def _build_sac_policy():
    compute_ops = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
    }

    def _policy(ctx, op, *args, **kwargs):
        if op in compute_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return _policy


def _wrap_checkpoint(model):
    wrapped = copy.deepcopy(model)
    linear_indices = [i for i, block in enumerate(wrapped.blocks) if isinstance(block, torch.nn.Linear)]
    for idx in linear_indices:
        original_forward = wrapped.blocks[idx].forward

        def _make_forward(orig_forward):
            def _wrapped(*args, **kwargs):
                return checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)

            return _wrapped

        wrapped.blocks[idx].forward = _make_forward(original_forward)
    return wrapped


def _wrap_sac(model):
    wrapped = copy.deepcopy(model)
    sac_context_fn = functools.partial(create_selective_checkpoint_contexts, _build_sac_policy())
    linear_indices = [i for i, block in enumerate(wrapped.blocks) if isinstance(block, torch.nn.Linear)]
    for idx in linear_indices:
        original_forward = wrapped.blocks[idx].forward

        def _make_forward(orig_forward):
            def _wrapped(*args, **kwargs):
                return checkpoint(
                    orig_forward,
                    *args,
                    use_reentrant=False,
                    context_fn=sac_context_fn,
                    **kwargs,
                )

            return _wrapped

        wrapped.blocks[idx].forward = _make_forward(original_forward)
    return wrapped


def _instantiate(method_name: str):
    base = OfficialRecomputeMLP().to(device).train()
    if method_name == "eager_baseline":
        model = base
        runner = model
    elif method_name == "eager_checkpoint":
        model = _wrap_checkpoint(base)
        runner = model
    elif method_name == "eager_sac":
        model = _wrap_sac(base)
        runner = model
    elif method_name == "compiled_baseline":
        model = base
        runner = torch.compile(model, dynamic=True)
    elif method_name == "compiled_checkpoint":
        model = _wrap_checkpoint(base)
        runner = torch.compile(model, dynamic=True)
    elif method_name == "compiled_sac":
        model = _wrap_sac(base)
        runner = torch.compile(model, dynamic=True)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    return model, runner


def _measure_step(method_name: str, steps: int = 6, warmup: int = 2):
    model, runner = _instantiate(method_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(32, 1024, device=device)
    target = torch.randn(32, 1024, device=device)
    loss_fn = torch.nn.MSELoss()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        optimizer.zero_grad()
        loss = loss_fn(runner(x), target)
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    peaks = []
    times = []
    loss_value = None
    for _ in range(steps):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = None
            end = None
            t0 = torch.perf_counter()
        loss = loss_fn(runner(x), target)
        loss_value = float(loss.item())
        loss.backward()
        optimizer.step()
        if torch.cuda.is_available():
            end.record()
            torch.cuda.synchronize(device)
            peaks.append(torch.cuda.max_memory_allocated(device))
            times.append(start.elapsed_time(end))
        else:
            times.append((torch.perf_counter() - t0) * 1000)
            peaks.append(-1)

    return {
        "loss": loss_value,
        "avg_peak_bytes": sum(peaks) / len(peaks),
        "avg_step_ms": sum(times) / len(times),
    }


def _fmt_bytes(num_bytes: float) -> str:
    if num_bytes < 0:
        return "N/A"
    gib = num_bytes / (1 << 30)
    mib = num_bytes / (1 << 20)
    if gib >= 1:
        return f"{gib:.2f} GB"
    return f"{mib:.2f} MB"


def main():
    methods = [
        ("eager_baseline", "user-facing eager baseline"),
        ("eager_checkpoint", "official checkpoint(use_reentrant=False)"),
        ("eager_sac", "official selective checkpoint"),
        ("compiled_baseline", "torch.compile baseline"),
        ("compiled_checkpoint", "official checkpoint + torch.compile"),
        ("compiled_sac", "official selective checkpoint + torch.compile"),
    ]

    print("\n=== Official PyTorch recomputation baseline ===")
    print(f"device: {device}")
    print("methods:")
    for name, desc in methods:
        print(f"  - {name}: {desc}")
    print("\nnotes:")
    print("  - checkpoint(..., use_reentrant=False) is the primary official user-facing API")
    print("  - selective checkpointing uses create_selective_checkpoint_contexts + CheckpointPolicy")
    print("  - compile-mode behavior here is the official user API running under torch.compile")
    print("  - this example does not treat direct min_cut_rematerialization_partition calls as the public API")

    results = {}
    for name, desc in methods:
        print(f"\n[{name}] {desc}")
        result = _measure_step(name)
        results[name] = result
        print(f"  loss: {result['loss']:.6f}")
        print(f"  avg_peak: {_fmt_bytes(result['avg_peak_bytes'])}")
        print(f"  avg_step: {result['avg_step_ms']:.1f} ms")

    eager_base = results["eager_baseline"]
    compiled_base = results["compiled_baseline"]

    print("\n=== Effectiveness summary ===")
    for name in ["eager_checkpoint", "eager_sac"]:
        delta = eager_base["avg_peak_bytes"] - results[name]["avg_peak_bytes"]
        print(f"  [{name} vs eager_baseline] peak delta: {_fmt_bytes(delta)}")
    for name in ["compiled_checkpoint", "compiled_sac"]:
        delta = compiled_base["avg_peak_bytes"] - results[name]["avg_peak_bytes"]
        print(f"  [{name} vs compiled_baseline] peak delta: {_fmt_bytes(delta)}")


if __name__ == "__main__":
    raise SystemExit(main())
