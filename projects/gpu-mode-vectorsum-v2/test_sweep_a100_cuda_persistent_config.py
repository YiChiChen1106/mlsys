from __future__ import annotations

from pathlib import Path

from sweep_a100_cuda_persistent_config import (
    Candidate,
    DEFAULT_CANDIDATES,
    parse_candidate_specs,
    render_submission,
)


def test_parse_candidate_specs_uses_default_and_named_specs() -> None:
    assert parse_candidate_specs([]) == DEFAULT_CANDIDATES
    assert parse_candidate_specs(["256", "1024"]) == (
        Candidate(grid_blocks=256, filename="submission_cuda_persistent_g256.py"),
        Candidate(grid_blocks=1024, filename="submission_cuda_persistent_g1024.py"),
    )


def test_parse_candidate_specs_rejects_invalid_grid_blocks() -> None:
    try:
        parse_candidate_specs(["0"])
    except ValueError as exc:
        assert "grid_blocks must be positive" in str(exc)
    else:
        raise AssertionError("invalid grid block count should fail")


def test_render_submission_updates_grid_blocks_and_module_name() -> None:
    source = Path("submission_cuda_persistent.py").read_text()

    rendered = render_submission(source, Candidate(grid_blocks=1024, filename="submission_cuda_persistent_g1024.py"))

    assert "#!POPCORN leaderboard vectorsum_v2" in rendered
    assert "GRID_BLOCKS = 1024\n" in rendered
    assert 'name="vectorsum_cuda_persistent_g1024"' in rendered
    assert "functions=[\"vector_sum_stage1\", \"vector_sum_stage2\"]" in rendered
    assert "PYBIND11_MODULE" not in rendered
