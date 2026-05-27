from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


TEST_SIZES = [1023, 1024, 1025, 2048, 4096]
BENCHMARK_SIZES = [
    1_638_400,
    3_276_800,
    6_553_600,
    13_107_200,
    26_214_400,
    52_428_800,
]


def load_submission(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"missing submission file: {path}")

    spec = importlib.util.spec_from_file_location("submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "custom_kernel"):
        raise AttributeError("submission.py must define custom_kernel")
    return module


def generate_input(size: int, seed: int):
    import torch

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    data = torch.randn(size, device="cuda", dtype=torch.float32, generator=gen).contiguous()

    offset_gen = torch.Generator(device="cuda")
    offset_gen.manual_seed(seed + 1)
    scale_gen = torch.Generator(device="cuda")
    scale_gen.manual_seed(seed + 2)

    offset = (torch.rand(1, device="cuda", generator=offset_gen) * 200 - 100).item()
    scale = (torch.rand(1, device="cuda", generator=scale_gen) * 9.9 + 0.1).item()

    input_tensor = (data * scale + offset).contiguous()
    output_tensor = torch.empty(1, device="cuda", dtype=torch.float32)
    return input_tensor, output_tensor


def reference_sum(x):
    import torch

    return x.to(torch.float64).sum().to(torch.float32)


def run_correctness(custom_kernel) -> None:
    import torch

    for i, size in enumerate(TEST_SIZES):
        data = generate_input(size, 4242 + i)
        x, _ = data
        actual = custom_kernel(data)
        expected = reference_sum(x)
        torch.cuda.synchronize()

        if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-8):
            diff = (actual - expected).abs().item()
            raise AssertionError(
                f"size={size} mismatch actual={actual.item()} expected={expected.item()} diff={diff}"
            )
        print(f"correctness size={size} pass actual={actual.item():.6f}")


def run_benchmark(custom_kernel) -> None:
    import torch
    import triton

    for i, size in enumerate(BENCHMARK_SIZES):
        data = generate_input(size, 93246 + i)
        x, _ = data
        gb = x.numel() * x.element_size() / 1e9
        ms = triton.testing.do_bench(lambda: custom_kernel(data))
        torch.cuda.synchronize()
        print(f"benchmark size={size} {ms:.4f} ms {gb / (ms / 1000):.2f} GB/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission.py")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    module = load_submission(Path(args.submission))
    run_correctness(module.custom_kernel)
    if args.benchmark:
        run_benchmark(module.custom_kernel)


if __name__ == "__main__":
    main()
