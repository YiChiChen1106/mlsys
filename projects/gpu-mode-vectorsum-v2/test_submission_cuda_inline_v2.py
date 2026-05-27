from __future__ import annotations

from pathlib import Path


def test_cuda_inline_v2_uses_official_load_inline_template_shape() -> None:
    source = Path("submission_cuda_inline_v2.py").read_text()

    assert "#!POPCORN leaderboard vectorsum_v2" in source
    assert "#!POPCORN gpu A100" in source
    assert "load_inline(" in source
    assert "functions=[\"vector_sum_stage1\", \"vector_sum_stage2\"]" in source
    assert "PYBIND11_MODULE" not in source
    assert "torch.cuda.current_stream" not in source
    assert "at::cuda::getCurrentCUDAStream" not in source


def test_cuda_inline_v2_exposes_vectorsum_custom_kernel() -> None:
    source = Path("submission_cuda_inline_v2.py").read_text()

    assert "def custom_kernel(data: input_t) -> output_t:" in source
    assert "x, output = data" in source
    assert "_CUDA_MODULE.vector_sum_stage1(x, partial, n_elements)" in source
    assert "_CUDA_MODULE.vector_sum_stage2(partial, output, num_blocks)" in source
    assert "return output[0]" in source


def test_cuda_inline_v2_uses_manual_warp_block_reduction() -> None:
    source = Path("submission_cuda_inline_v2.py").read_text()

    assert "__shfl_down_sync" in source
    assert "warp_reduce_sum" in source
    assert "block_reduce_sum" in source
    assert "THREADS = 256" in source
    assert "ITEMS_PER_THREAD = 32" in source
    assert "ELEMENTS_PER_BLOCK = THREADS * ITEMS_PER_THREAD" in source
