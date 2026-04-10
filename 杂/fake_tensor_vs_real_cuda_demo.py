import gc
import torch


def mb(x: int) -> float:
    return x / (1024 ** 2)


def cuda_mem() -> int:
    return torch.cuda.memory_allocated()


def run_real_cuda_path(device: torch.device) -> None:
    print("\n=== A) Real CUDA path (real tensor + real kernels) ===")
    gc.collect()
    torch.cuda.empty_cache()

    x = torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize()
    m0 = cuda_mem()

    # Real matmul will allocate output tensor and launch CUDA kernel.
    y = x @ x.t()
    torch.cuda.synchronize()
    m1 = cuda_mem()

    print(f"x type={type(x).__name__}, device={x.device}, is_meta={x.is_meta}")
    print(f"y type={type(y).__name__}, device={y.device}, is_meta={y.is_meta}")
    print(f"memory before matmul: {mb(m0):.2f} MB")
    print(f"memory after  matmul: {mb(m1):.2f} MB")
    print(f"delta: {mb(m1 - m0):.2f} MB (real allocation)")

    # Real tensors hold real numeric data.
    print(f"y.mean().item() works: {float(y.mean().item()):.6f}")


def run_fake_mode_path(device: torch.device) -> None:
    print("\n=== B) FakeTensorMode over CUDA semantics ===")
    gc.collect()
    torch.cuda.empty_cache()

    x_real = torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize()
    m0 = cuda_mem()

    # Import path can vary slightly across PyTorch versions.
    from torch._subclasses import FakeTensorMode

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    with fake_mode:
        x_fake = fake_mode.from_tensor(x_real)
        y_fake = x_fake @ x_fake.t()

    torch.cuda.synchronize()
    m1 = cuda_mem()

    print(
        f"x_fake type={type(x_fake).__name__}, device={x_fake.device}, is_meta={x_fake.is_meta}"
    )
    print(
        f"y_fake type={type(y_fake).__name__}, device={y_fake.device}, is_meta={y_fake.is_meta}"
    )
    print(f"memory before fake matmul: {mb(m0):.2f} MB")
    print(f"memory after  fake matmul: {mb(m1):.2f} MB")
    print(f"delta: {mb(m1 - m0):.2f} MB (near zero expected)")

    print("y_fake has shape/dtype/device semantics but no real numeric storage.")
    try:
        _ = y_fake.mean().item()
    except Exception as exc:
        print(f"y_fake.mean().item() fails as expected: {type(exc).__name__}: {exc}")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    device = torch.device("cuda")
    print(f"torch={torch.__version__}, device={device}")

    run_real_cuda_path(device)
    run_fake_mode_path(device)

    print("\n=== Interpretation ===")
    print("1) Real CUDA path: real kernels + real output allocation + real values.")
    print("2) FakeTensorMode path: keeps CUDA semantics for tracing/analysis,")
    print("   but avoids real kernel work for fake tensors.")


if __name__ == "__main__":
    main()
