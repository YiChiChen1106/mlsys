from __future__ import annotations

from pathlib import Path


def test_cuda_atomic_uses_official_load_inline_template() -> None:
    source = Path("submission_cuda_atomic.py").read_text()

    assert "#!POPCORN leaderboard vectorsum_v2" in source
    assert "#!POPCORN gpu A100" in source
    assert "load_inline(" in source
    assert "functions=[\"vector_sum_atomic\"]" in source
    assert "PYBIND11_MODULE" not in source
    assert "torch.cuda.current_stream" not in source
    assert "at::cuda::getCurrentCUDAStream" not in source


def test_cuda_atomic_zeroes_output_before_atomic_kernel() -> None:
    source = Path("submission_cuda_atomic.py").read_text()

    assert "zero_output_kernel<<<1, 1>>>" in source
    assert "atomic_sum_kernel<<<num_blocks, THREADS>>>" in source
    assert "atomicAdd(output, block_sum)" in source
    assert source.index("zero_output_kernel<<<1, 1>>>") < source.index("atomic_sum_kernel<<<num_blocks, THREADS>>>")


def test_cuda_atomic_custom_kernel_calls_single_wrapper() -> None:
    source = Path("submission_cuda_atomic.py").read_text()

    assert "def custom_kernel(data: input_t) -> output_t:" in source
    assert "x, output = data" in source
    assert "_CUDA_MODULE.vector_sum_atomic(x, output, n_elements, num_blocks)" in source
    assert "return output[0]" in source
