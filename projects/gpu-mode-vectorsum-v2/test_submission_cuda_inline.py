from __future__ import annotations

from pathlib import Path


def test_cuda_inline_submission_exposes_popcorn_custom_kernel() -> None:
    source = Path("submission_cuda_inline.py").read_text()

    assert "#!POPCORN leaderboard vectorsum_v2" in source
    assert "#!POPCORN gpu A100" in source
    assert "def custom_kernel(data: input_t) -> output_t:" in source
    assert "return output[0]" in source


def test_cuda_inline_submission_uses_two_stage_block_reduction() -> None:
    source = Path("submission_cuda_inline.py").read_text()

    assert "load_inline(" in source
    assert "partial_sum_kernel" in source
    assert "final_sum_kernel" in source
    assert "__shfl_down_sync" in source
    assert "block_reduce_sum" in source
    assert "stage1(x, partial, n_elements" in source
    assert "stage2(partial, output, num_blocks" in source


def test_cuda_inline_submission_passes_python_current_stream_to_cpp() -> None:
    source = Path("submission_cuda_inline.py").read_text()

    assert "torch.cuda.current_stream(x.device).cuda_stream" in source
    assert "reinterpret_cast<cudaStream_t>(stream)" in source
    assert "uint64_t stream" in source
    assert "stage1(x, partial, n_elements, stream)" in source
    assert "stage2(partial, output, num_blocks, stream)" in source


def test_cuda_inline_submission_matches_current_block_size() -> None:
    source = Path("submission_cuda_inline.py").read_text()

    assert "THREADS = 256" in source
    assert "ITEMS_PER_THREAD = 32" in source
    assert "ELEMENTS_PER_BLOCK = THREADS * ITEMS_PER_THREAD" in source
