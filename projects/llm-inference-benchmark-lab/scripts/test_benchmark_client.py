import pytest

from benchmark_client import RequestMetrics, aggregate_metrics, build_prompt
from benchmark_client import parse_sse_content


def test_build_prompt_uses_requested_token_bucket():
    short = build_prompt("short")
    medium = build_prompt("medium")
    long = build_prompt("long")

    assert len(short.split()) < len(medium.split()) < len(long.split())
    assert "ML systems" in short


def test_aggregate_metrics_computes_latency_and_throughput():
    rows = [
        RequestMetrics(
            request_id=0,
            ok=True,
            ttft_s=0.20,
            latency_s=1.20,
            output_tokens=20,
            error="",
        ),
        RequestMetrics(
            request_id=1,
            ok=True,
            ttft_s=0.40,
            latency_s=1.40,
            output_tokens=20,
            error="",
        ),
        RequestMetrics(
            request_id=2,
            ok=False,
            ttft_s=0.0,
            latency_s=0.50,
            output_tokens=0,
            error="timeout",
        ),
    ]

    summary = aggregate_metrics(rows)

    assert summary["requests"] == 3
    assert summary["successful_requests"] == 2
    assert summary["failure_rate"] == pytest.approx(1 / 3)
    assert summary["avg_ttft_s"] == pytest.approx(0.30)
    assert summary["avg_latency_s"] == pytest.approx(1.30)
    assert summary["avg_tpot_s"] == pytest.approx(0.05)
    assert summary["output_tokens_per_s"] == pytest.approx(40 / 2.60)


def test_parse_sse_content_extracts_delta_text():
    line = 'data: {"choices":[{"delta":{"content":"hello"}}]}'

    assert parse_sse_content(line) == "hello"


def test_parse_sse_content_ignores_done_marker():
    assert parse_sse_content("data: [DONE]") == ""
