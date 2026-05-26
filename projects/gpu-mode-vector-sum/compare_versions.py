import torch
import triton

from vector_sum_triton import vector_sum
from vector_sum_two_stage import vector_sum_two_stage

SIZES = [
    1_638_400,
    3_276_800,
    6_553_600,
    13_107_200,
    26_214_400,
    52_428_800,
]

for n in SIZES:
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    gb = x.numel() * x.element_size() / 1e9

    torch_ms = triton.testing.do_bench(lambda: torch.sum(x))
    mixed_ms = triton.testing.do_bench(lambda: vector_sum(x, block_size=8192))
    two_stage_ms = triton.testing.do_bench(lambda: vector_sum_two_stage(x))

    print(f"\nN={n}")
    print(f"  torch.sum      {torch_ms:.4f} ms  {gb / (torch_ms / 1000):.2f} GB/s")
    print(f"  mixed          {mixed_ms:.4f} ms  {gb / (mixed_ms / 1000):.2f} GB/s")
    print(f"  two_stage      {two_stage_ms:.4f} ms  {gb / (two_stage_ms / 1000):.2f} GB/s")
