from __future__ import annotations

from pathlib import Path

from sweep_a100_num_warps import (
    Candidate,
    candidate_grid,
    parse_warps,
    render_submission,
)


def test_candidate_grid_uses_cross_product_names() -> None:
    candidates = candidate_grid((4, 8), (4, 8))

    assert candidates == [
        Candidate(num_warps=4, final_num_warps=4, filename="submission_wp4_wf4.py"),
        Candidate(num_warps=4, final_num_warps=8, filename="submission_wp4_wf8.py"),
        Candidate(num_warps=8, final_num_warps=4, filename="submission_wp8_wf4.py"),
        Candidate(num_warps=8, final_num_warps=8, filename="submission_wp8_wf8.py"),
    ]


def test_parse_warps_accepts_space_and_comma_groups() -> None:
    assert parse_warps([], default=(4, 8)) == (4, 8)
    assert parse_warps(["4,8", "16"], default=(4,)) == (4, 8, 16)


def test_render_submission_adds_num_warps_to_both_kernel_launches() -> None:
    source = Path("submission.py").read_text()

    rendered = render_submission(
        source,
        Candidate(num_warps=4, final_num_warps=8, filename="submission_wp4_wf8.py"),
    )

    assert "#!POPCORN leaderboard vectorsum_v2" in rendered
    assert "#!POPCORN gpu A100" in rendered
    assert "BLOCK_SIZE = 8192\n" in rendered
    assert "FINAL_BLOCK_SIZE = 8192\n" in rendered
    assert "        BLOCK_SIZE=BLOCK_SIZE,\n        num_warps=4,\n    )" in rendered
    assert "        FINAL_BLOCK_SIZE=FINAL_BLOCK_SIZE,\n        num_warps=8,\n    )" in rendered
