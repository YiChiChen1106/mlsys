import time
from threading import Lock

import pytest

from benchmark_client import RequestMetrics, aggregate_metrics, build_prompt, run_benchmark
from benchmark_client import parse_sse_content, run_measured_benchmark


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


def test_run_benchmark_excludes_warmup_rows_from_returned_measurements():
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs["request_id"])
        return RequestMetrics(
            request_id=kwargs["request_id"],
            ok=True,
            ttft_s=0.10 + kwargs["request_id"],
            latency_s=0.20 + kwargs["request_id"],
            output_tokens=10,
            error="",
        )

    rows = run_benchmark(
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="dummy",
        prompt="hello",
        max_tokens=8,
        measured_requests=2,
        warmup_requests=3,
        timeout_s=5.0,
        runner=fake_runner,
    )

    assert calls == [0, 1, 2, 3, 4]
    assert [row.request_id for row in rows] == [3, 4]


def test_run_benchmark_runs_requests_with_requested_concurrency():
    lock = Lock()
    active = 0
    max_active = 0

    def fake_runner(**kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return RequestMetrics(
            request_id=kwargs["request_id"],
            ok=True,
            ttft_s=0.10,
            latency_s=0.20,
            output_tokens=10,
            error="",
        )

    rows = run_benchmark(
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="dummy",
        prompt="hello",
        max_tokens=8,
        measured_requests=4,
        warmup_requests=0,
        concurrency=2,
        timeout_s=5.0,
        runner=fake_runner,
    )

    assert [row.request_id for row in rows] == [0, 1, 2, 3]
    assert max_active == 2


def test_aggregate_metrics_can_use_wall_time_for_concurrent_throughput():
    rows = [
        RequestMetrics(
            request_id=0,
            ok=True,
            ttft_s=0.10,
            latency_s=1.00,
            output_tokens=10,
            error="",
        ),
        RequestMetrics(
            request_id=1,
            ok=True,
            ttft_s=0.10,
            latency_s=1.00,
            output_tokens=10,
            error="",
        ),
    ]

    summary = aggregate_metrics(rows, wall_time_s=1.0)

    assert summary["output_tokens_per_s"] == pytest.approx(20.0)


def test_run_measured_benchmark_wall_time_excludes_warmup():
    def fake_runner(**kwargs):
        if kwargs["request_id"] < 2:
            time.sleep(0.05)
        else:
            time.sleep(0.01)
        return RequestMetrics(
            request_id=kwargs["request_id"],
            ok=True,
            ttft_s=0.01,
            latency_s=0.02,
            output_tokens=5,
            error="",
        )

    _, wall_time_s = run_measured_benchmark(
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        model="dummy",
        prompt="hello",
        max_tokens=8,
        measured_requests=2,
        warmup_requests=2,
        concurrency=1,
        timeout_s=5.0,
        runner=fake_runner,
    )

    assert wall_time_s < 0.08
