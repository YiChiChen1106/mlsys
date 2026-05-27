from __future__ import annotations

from pathlib import Path

from sweep_a100_block_size import (
    candidate_for_block_size,
    render_submission,
)


def test_candidate_final_block_size_covers_largest_official_shape() -> None:
    expected = {
        4096: (12800, 16384),
        8192: (6400, 8192),
        16384: (3200, 4096),
        32768: (1600, 2048),
    }

    for block_size, (num_blocks, final_block_size) in expected.items():
        candidate = candidate_for_block_size(block_size)
        assert candidate.num_blocks == num_blocks
        assert candidate.final_block_size == final_block_size


def test_render_submission_updates_only_sweep_constants() -> None:
    source = Path("submission.py").read_text()

    rendered = render_submission(source, candidate_for_block_size(4096))

    assert "#!POPCORN leaderboard vectorsum_v2" in rendered
    assert "#!POPCORN gpu A100" in rendered
    assert "BLOCK_SIZE = 4096\n" in rendered
    assert "FINAL_BLOCK_SIZE = 16384\n" in rendered
    assert "def custom_kernel(data: input_t) -> output_t:" in rendered
