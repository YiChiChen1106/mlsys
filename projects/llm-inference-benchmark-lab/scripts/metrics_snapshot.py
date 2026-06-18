from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path


METRICS = {
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
}

METRIC_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$")


def parse_labels(raw_labels: str | None) -> dict[str, str]:
    if not raw_labels:
        return {}
    labels = {}
    for item in raw_labels.split(","):
        key, value = item.split("=", 1)
        labels[key] = value.strip('"')
    return labels


def parse_metrics(text: str, *, metric_names: set[str] = METRICS) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name not in metric_names:
            continue
        labels = parse_labels(match.group("labels"))
        reason = labels.get("reason")
        key = f"{name}:{reason}" if reason else name
        values[key] = float(match.group("value"))
    return values


def fetch_metrics(endpoint: str) -> dict[str, float]:
    import requests

    response = requests.get(endpoint, timeout=10)
    response.raise_for_status()
    return parse_metrics(response.text)


def write_snapshot(path: Path, endpoint: str, values: dict[str, float]) -> None:
    payload = {
        "captured_at_unix_s": time.time(),
        "endpoint": endpoint,
        "metrics": values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_snapshot(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["metrics"].items()}


def diff_snapshots(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    delta = {key: after.get(key, 0.0) - before.get(key, 0.0) for key in sorted(keys)}
    queries = delta.get("vllm:prefix_cache_queries_total", 0.0)
    hits = delta.get("vllm:prefix_cache_hits_total", 0.0)
    delta["prefix_cache_hit_ratio"] = hits / queries if queries else 0.0
    return delta


def write_csv(path: Path, label: str, values: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "metric", "value"])
        writer.writeheader()
        for metric, value in sorted(values.items()):
            writer.writerow({"label": label, "metric": metric, "value": value})


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--endpoint", default="http://127.0.0.1:8000/metrics")
    snapshot.add_argument("--out", type=Path, required=True)

    diff = subparsers.add_parser("diff")
    diff.add_argument("--before", type=Path, required=True)
    diff.add_argument("--after", type=Path, required=True)
    diff.add_argument("--label", required=True)
    diff.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "snapshot":
        write_snapshot(args.out, args.endpoint, fetch_metrics(args.endpoint))
    elif args.command == "diff":
        values = diff_snapshots(load_snapshot(args.before), load_snapshot(args.after))
        write_csv(args.out, args.label, values)


if __name__ == "__main__":
    main()
