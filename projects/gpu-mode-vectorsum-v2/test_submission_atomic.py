from __future__ import annotations

from pathlib import Path


def test_atomic_submission_zeroes_output_before_atomic_add() -> None:
    source = Path("submission_atomic.py").read_text()

    assert "#!POPCORN leaderboard vectorsum_v2" in source
    assert "#!POPCORN gpu A100" in source
    assert "def custom_kernel(data: input_t) -> output_t:" in source
    assert "def _zero_output_kernel(output_ptr):" in source
    assert "def _atomic_sum_kernel(" in source
    assert "tl.atomic_add(output_ptr, partial" in source

    zero_launch = source.index("_zero_output_kernel[(1,)]")
    atomic_launch = source.index("_atomic_sum_kernel[(num_blocks,)]")
    assert zero_launch < atomic_launch
    assert "return output[0]" in source
