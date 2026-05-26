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


@triton.jit
def _final_sum_kernel(partial_ptr, out_ptr, num_blocks: tl.constexpr, FINAL_BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, FINAL_BLOCK_SIZE)
    mask = offsets < num_blocks
    values = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    final = tl.sum(values, axis=0)
    tl.store(out_ptr, final)


def vector_sum_two_stage(
    x: torch.Tensor,
    block_size: int = 8192,
    final_block_size: int = 8192,
) -> torch.Tensor:
    n = x.numel()
    num_blocks = triton.cdiv(n, block_size)
    if num_blocks > final_block_size:
        raise ValueError(
            f"num_blocks={num_blocks} exceeds final_block_size={final_block_size}; "
            "increase final_block_size or use a recursive reduction"
        )

    partial = torch.empty((num_blocks,), device=x.device, dtype=torch.float32)
    out = torch.empty((), device=x.device, dtype=torch.float32)

    _partial_sum_kernel[(num_blocks,)](
        x,
        partial,
        n,
        BLOCK_SIZE=block_size,
    )
    _final_sum_kernel[(1,)](
        partial,
        out,
        num_blocks,
        FINAL_BLOCK_SIZE=final_block_size,
    )
    return out


def check_correctness() -> None:
    sizes = [10, 1024, 1_638_400, 52_428_800]
    for n in sizes:
        torch.manual_seed(0)
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        expected = torch.sum(x)
        actual = vector_sum_two_stage(x)
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
        gb = x.numel() * x.element_size() / 1e9

        torch_ms = triton.testing.do_bench(lambda: torch.sum(x))
        two_stage_ms = triton.testing.do_bench(lambda: vector_sum_two_stage(x))

        torch_bw = gb / (torch_ms / 1000)
        two_stage_bw = gb / (two_stage_ms / 1000)

        print(
            f"N={n:>9} "
            f"torch={torch_ms:.4f} ms {torch_bw:>7.2f} GB/s  "
            f"two_stage={two_stage_ms:.4f} ms {two_stage_bw:>7.2f} GB/s"
        )


if __name__ == "__main__":
    print("Correctness")
    check_correctness()
    print()
    print("Benchmark")
    benchmark()
