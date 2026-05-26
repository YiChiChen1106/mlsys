import torch
import triton

from vector_sum_triton import vector_sum

SIZES = [
    1_638_400,
    3_276_800,
    6_553_600,
    13_107_200,
    26_214_400,
    52_428_800,
]

BLOCK_SIZES = [512, 1024, 2048, 4096, 8192]

for n in SIZES:
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    gb = x.numel() * x.element_size() / 1e9

    print(f"\nN={n}")
    torch_ms = triton.testing.do_bench(lambda: torch.sum(x))
    print(f"  torch.sum    {torch_ms:.4f} ms  {gb / (torch_ms / 1000):.2f} GB/s")

    for block_size in BLOCK_SIZES:
        y = vector_sum(x, block_size=block_size)
        expected = torch.sum(x)
        diff = (expected - y).abs().item()

        ms = triton.testing.do_bench(lambda: vector_sum(x, block_size=block_size))
        bandwidth = gb / (ms / 1000)
        print(
            f"  block={block_size:<5} "
            f"{ms:.4f} ms  {bandwidth:>7.2f} GB/s  diff={diff:.6f}"
        )
