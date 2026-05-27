from __future__ import annotations

from pathlib import Path


def test_cuda_persistent_uses_official_load_inline_template() -> None:
    source = Path("submission_cuda_persistent.py").read_text()

    assert "#!POPCORN leaderboard vectorsum_v2" in source
    assert "#!POPCORN gpu A100" in source
    assert "load_inline(" in source
    assert "functions=[\"vector_sum_stage1\", \"vector_sum_stage2\"]" in source
    assert "PYBIND11_MODULE" not in source
    assert "torch.cuda.current_stream" not in source
    assert "at::cuda::getCurrentCUDAStream" not in source


def test_cuda_persistent_stage1_uses_grid_stride_chunks() -> None:
    source = Path("submission_cuda_persistent.py").read_text()

    assert "GRID_BLOCKS = 512" in source
    assert "ELEMENTS_PER_CHUNK = THREADS * ITEMS_PER_THREAD" in source
    assert "for (int64_t block_start = static_cast<int64_t>(blockIdx.x) * ELEMENTS_PER_CHUNK;" in source
    assert "block_start += static_cast<int64_t>(gridDim.x) * ELEMENTS_PER_CHUNK" in source
    assert "partial[blockIdx.x] = block_sum" in source


def test_cuda_persistent_custom_kernel_uses_fixed_partial_count() -> None:
    source = Path("submission_cuda_persistent.py").read_text()

    assert "num_partials = min(GRID_BLOCKS, triton.cdiv(n_elements, ELEMENTS_PER_CHUNK))" in source
    assert "partial = _partial_buffer(x, num_partials)" in source
    assert "_CUDA_MODULE.vector_sum_stage1(x, partial, n_elements, num_partials)" in source
    assert "_CUDA_MODULE.vector_sum_stage2(partial, output, num_partials)" in source
    assert "return output[0]" in source
