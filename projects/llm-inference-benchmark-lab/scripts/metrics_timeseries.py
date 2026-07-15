from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests

from metrics_snapshot import parse_metrics


WATCH_METRICS = {
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_waiting_by_reason",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
}

CSV_FIELDS = [
    "elapsed_s",
    "captured_at_unix_s",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_waiting_by_reason:capacity",
    "vllm:num_requests_waiting_by_reason:deferred",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
]


def fetch_watch_metrics(endpoint: str) -> dict[str, float]:
    response = requests.get(endpoint, timeout=10)
    response.raise_for_status()
    return parse_metrics(response.text, metric_names=WATCH_METRICS)


def row_from_metrics(
    *,
    elapsed_s: float,
    captured_at_unix_s: float,
    metrics: dict[str, float],
) -> dict[str, float]:
    row = {field: 0.0 for field in CSV_FIELDS}
    row["elapsed_s"] = elapsed_s
    row["captured_at_unix_s"] = captured_at_unix_s
    for key, value in metrics.items():
        if key in row:
            row[key] = value
    return row


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "samples": 0,
            "duration_s": 0.0,
            "max_kv_cache_usage_perc": 0.0,
            "max_num_requests_running": 0.0,
            "max_num_requests_waiting": 0.0,
            "max_num_requests_waiting_capacity": 0.0,
            "max_num_requests_waiting_deferred": 0.0,
            "num_preemptions_delta": 0.0,
            "prefix_cache_queries_delta": 0.0,
            "prefix_cache_hits_delta": 0.0,
            "prefix_cache_hit_ratio": 0.0,
        }

    first = rows[0]
    last = rows[-1]
    def value(row: dict[str, float], key: str) -> float:
        return row.get(key, 0.0)

    preemptions_delta = value(last, "vllm:num_preemptions_total") - value(first, "vllm:num_preemptions_total")
    prefix_queries_delta = value(last, "vllm:prefix_cache_queries_total") - value(
        first, "vllm:prefix_cache_queries_total"
    )
    prefix_hits_delta = value(last, "vllm:prefix_cache_hits_total") - value(first, "vllm:prefix_cache_hits_total")

    return {
        "samples": len(rows),
        "duration_s": last["elapsed_s"] - first["elapsed_s"],
        "max_kv_cache_usage_perc": max(value(row, "vllm:kv_cache_usage_perc") for row in rows),
        "max_num_requests_running": max(value(row, "vllm:num_requests_running") for row in rows),
        "max_num_requests_waiting": max(value(row, "vllm:num_requests_waiting") for row in rows),
        "max_num_requests_waiting_capacity": max(
            value(row, "vllm:num_requests_waiting_by_reason:capacity") for row in rows
        ),
        "max_num_requests_waiting_deferred": max(
            value(row, "vllm:num_requests_waiting_by_reason:deferred") for row in rows
        ),
        "num_preemptions_delta": preemptions_delta,
        "prefix_cache_queries_delta": prefix_queries_delta,
        "prefix_cache_hits_delta": prefix_hits_delta,
        "prefix_cache_hit_ratio": prefix_hits_delta / prefix_queries_delta if prefix_queries_delta else 0.0,
    }


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def record_timeseries(
    *,
    endpoint: str,
    interval_s: float,
    max_duration_s: float,
    stop_file: Path | None = None,
) -> list[dict[str, float]]:
    started = time.perf_counter()
    rows = []
    while True:
        elapsed_s = time.perf_counter() - started
        metrics = fetch_watch_metrics(endpoint)
        rows.append(
            row_from_metrics(
                elapsed_s=elapsed_s,
                captured_at_unix_s=time.time(),
                metrics=metrics,
            )
        )
        if elapsed_s >= max_duration_s:
            break
        if stop_file is not None and stop_file.exists():
            break
        time.sleep(interval_s)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--max-duration-s", type=float, default=120.0)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    rows = record_timeseries(
        endpoint=args.endpoint,
        interval_s=args.interval_s,
        max_duration_s=args.max_duration_s,
        stop_file=args.stop_file,
    )
    write_rows(args.out, rows)
    write_summary(args.summary_out, summarize_rows(rows))


if __name__ == "__main__":
    main()
