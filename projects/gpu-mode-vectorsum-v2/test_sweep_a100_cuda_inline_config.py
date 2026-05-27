from __future__ import annotations

from pathlib import Path

from sweep_a100_cuda_inline_config import (
    Candidate,
    DEFAULT_CANDIDATES,
    parse_candidate_specs,
    render_submission,
)


def test_parse_candidate_specs_uses_default_and_named_specs() -> None:
    assert parse_candidate_specs([]) == DEFAULT_CANDIDATES
    assert parse_candidate_specs(["128x32", "256x64"]) == (
        Candidate(threads=128, items_per_thread=32, filename="submission_cuda_t128_i32.py"),
        Candidate(threads=256, items_per_thread=64, filename="submission_cuda_t256_i64.py"),
    )


def test_parse_candidate_specs_rejects_invalid_thread_counts() -> None:
    try:
        parse_candidate_specs(["129x32"])
    except ValueError as exc:
        assert "threads must be a multiple of 32" in str(exc)
    else:
        raise AssertionError("invalid thread count should fail")


def test_render_submission_updates_cuda_config_and_module_name() -> None:
    source = Path("submission_cuda_inline_v2.py").read_text()

    rendered = render_submission(
        source,
        Candidate(threads=128, items_per_thread=64, filename="submission_cuda_t128_i64.py"),
    )

    assert "#!POPCORN leaderboard vectorsum_v2" in rendered
    assert "THREADS = 128\n" in rendered
    assert "ITEMS_PER_THREAD = 64\n" in rendered
    assert "ELEMENTS_PER_BLOCK = THREADS * ITEMS_PER_THREAD" in rendered
    assert 'name="vectorsum_cuda_inline_t128_i64"' in rendered
    assert "functions=[\"vector_sum_stage1\", \"vector_sum_stage2\"]" in rendered
    assert "PYBIND11_MODULE" not in rendered
