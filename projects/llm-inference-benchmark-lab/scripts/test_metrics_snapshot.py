import pytest

from metrics_snapshot import diff_snapshots, parse_metrics


def test_parse_metrics_extracts_selected_vllm_values():
    text = """
# HELP vllm:prefix_cache_queries_total Prefix cache queries.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="dummy"} 100.0
vllm:prefix_cache_hits_total{engine="0",model_name="dummy"} 25.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="dummy",reason="capacity"} 3.0
python_info{version="3.12"} 1.0
"""

    values = parse_metrics(
        text,
        metric_names={
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_hits_total",
            "vllm:num_requests_waiting_by_reason",
        },
    )

    assert values["vllm:prefix_cache_queries_total"] == 100.0
    assert values["vllm:prefix_cache_hits_total"] == 25.0
    assert values["vllm:num_requests_waiting_by_reason:capacity"] == 3.0
    assert "python_info" not in values


def test_diff_snapshots_computes_prefix_cache_hit_ratio():
    before = {
        "vllm:prefix_cache_queries_total": 100.0,
        "vllm:prefix_cache_hits_total": 20.0,
    }
    after = {
        "vllm:prefix_cache_queries_total": 180.0,
        "vllm:prefix_cache_hits_total": 60.0,
    }

    delta = diff_snapshots(before, after)

    assert delta["vllm:prefix_cache_queries_total"] == 80.0
    assert delta["vllm:prefix_cache_hits_total"] == 40.0
    assert delta["prefix_cache_hit_ratio"] == pytest.approx(0.5)
