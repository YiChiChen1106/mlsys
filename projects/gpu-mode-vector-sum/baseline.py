import torch
import triton

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

    for _ in range(10):
        y = torch.sum(x)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(lambda: torch.sum(x))
    gb = x.numel() * x.element_size() / 1e9
    bandwidth = gb / (ms / 1000)

    print(f"N={n:>9}  torch.sum={ms:.4f} ms  bandwidth={bandwidth:.2f} GB/s")
