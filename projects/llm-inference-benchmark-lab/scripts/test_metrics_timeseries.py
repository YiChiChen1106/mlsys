import pytest

from metrics_timeseries import summarize_rows


def test_summarize_rows_reports_peaks_and_counter_deltas():
    rows = [
        {
            "elapsed_s": 0.0,
            "vllm:kv_cache_usage_perc": 0.10,
            "vllm:num_requests_running": 2.0,
            "vllm:num_requests_waiting": 0.0,
            "vllm:num_requests_waiting_by_reason:capacity": 0.0,
            "vllm:num_preemptions_total": 0.0,
            "vllm:prefix_cache_queries_total": 100.0,
            "vllm:prefix_cache_hits_total": 10.0,
        },
        {
            "elapsed_s": 0.2,
            "vllm:kv_cache_usage_perc": 0.75,
            "vllm:num_requests_running": 8.0,
            "vllm:num_requests_waiting": 16.0,
            "vllm:num_requests_waiting_by_reason:capacity": 12.0,
            "vllm:num_preemptions_total": 1.0,
            "vllm:prefix_cache_queries_total": 180.0,
            "vllm:prefix_cache_hits_total": 30.0,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["samples"] == 2
    assert summary["max_kv_cache_usage_perc"] == pytest.approx(0.75)
    assert summary["max_num_requests_running"] == pytest.approx(8.0)
    assert summary["max_num_requests_waiting"] == pytest.approx(16.0)
    assert summary["max_num_requests_waiting_capacity"] == pytest.approx(12.0)
    assert summary["num_preemptions_delta"] == pytest.approx(1.0)
    assert summary["prefix_cache_queries_delta"] == pytest.approx(80.0)
    assert summary["prefix_cache_hits_delta"] == pytest.approx(20.0)
    assert summary["prefix_cache_hit_ratio"] == pytest.approx(0.25)
