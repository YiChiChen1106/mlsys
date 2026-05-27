from __future__ import annotations

from pathlib import Path

from sweep_a100_cuda_atomic_config import (
    Candidate,
    DEFAULT_CANDIDATES,
    parse_candidate_specs,
    render_submission,
)


def test_parse_candidate_specs_uses_default_and_named_specs() -> None:
    assert parse_candidate_specs([]) == DEFAULT_CANDIDATES
    assert parse_candidate_specs(["256x32", "256x128"]) == (
        Candidate(threads=256, items_per_thread=32, filename="submission_cuda_atomic_t256_i32.py"),
        Candidate(threads=256, items_per_thread=128, filename="submission_cuda_atomic_t256_i128.py"),
    )


def test_parse_candidate_specs_rejects_invalid_thread_counts() -> None:
    try:
        parse_candidate_specs(["33x32"])
    except ValueError as exc:
        assert "threads must be a multiple of 32" in str(exc)
    else:
        raise AssertionError("invalid thread count should fail")


def test_render_submission_updates_cuda_atomic_config_and_module_name() -> None:
    source = Path("submission_cuda_atomic.py").read_text()

    rendered = render_submission(
        source,
        Candidate(threads=512, items_per_thread=32, filename="submission_cuda_atomic_t512_i32.py"),
    )

    assert "THREADS = 512\n" in rendered
    assert "ITEMS_PER_THREAD = 32\n" in rendered
    assert 'name="vectorsum_cuda_atomic_t512_i32"' in rendered
    assert "functions=[\"vector_sum_atomic\"]" in rendered
    assert "PYBIND11_MODULE" not in rendered
