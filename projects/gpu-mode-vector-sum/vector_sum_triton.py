import torch
import triton
import triton.language as tl


@triton.jit
def _partial_sum_kernel(x_ptr, partial_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.store(partial_ptr + pid, partial)


def vector_sum(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    n = x.numel()
    num_blocks = triton.cdiv(n, block_size)
    partial = torch.empty((num_blocks,), device=x.device, dtype=torch.float32)

    _partial_sum_kernel[(num_blocks,)](
        x,
        partial,
        n,
        BLOCK_SIZE=block_size,
    )
    return torch.sum(partial)


def check_correctness() -> None:
    sizes = [10, 1024, 1_638_400, 52_428_800]
    for n in sizes:
        torch.manual_seed(0)
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        expected = torch.sum(x)
        actual = vector_sum(x)
        diff = (expected - actual).abs().item()
        print(f"N={n:>9} expected={expected.item():>12.4f} actual={actual.item():>12.4f} diff={diff:.6f}")


def benchmark() -> None:
    sizes = [
        1_638_400,
        3_276_800,
        6_553_600,
        13_107_200,
        26_214_400,
        52_428_800,
    ]

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        torch_ms = triton.testing.do_bench(lambda: torch.sum(x))
        triton_ms = triton.testing.do_bench(lambda: vector_sum(x))

        gb = x.numel() * x.element_size() / 1e9
        torch_bw = gb / (torch_ms / 1000)
        triton_bw = gb / (triton_ms / 1000)

        print(
            f"N={n:>9} "
            f"torch={torch_ms:.4f} ms {torch_bw:>7.2f} GB/s  "
            f"triton={triton_ms:.4f} ms {triton_bw:>7.2f} GB/s"
        )


if __name__ == "__main__":
    print("Correctness")
    check_correctness()
    print()
    print("Benchmark")
    benchmark()
