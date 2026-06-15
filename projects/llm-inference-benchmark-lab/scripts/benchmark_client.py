from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROMPTS = {
    "short": "Explain ML systems inference in two concise sentences.",
    "medium": (
        "Explain how an ML systems engineer should think about LLM inference. "
        "Include latency, throughput, batching, KV cache, and GPU memory pressure."
    ),
    "long": (
        "You are mentoring a student learning AI infrastructure. "
        "Give a careful explanation of LLM inference serving, including prefill, decode, "
        "KV cache allocation, continuous batching, tensor parallelism, GPU memory limits, "
        "and how to design a benchmark that measures TTFT, TPOT, throughput, and failures."
    ),
}


@dataclass(frozen=True)
class RequestMetrics:
    request_id: int
    ok: bool
    ttft_s: float
    latency_s: float
    output_tokens: int
    error: str


def build_prompt(length: str) -> str:
    if length not in PROMPTS:
        raise ValueError(f"unknown prompt length: {length}")
    return PROMPTS[length]


def parse_sse_content(line: str) -> str:
    if not line.startswith("data: "):
        return ""
    payload = line.removeprefix("data: ").strip()
    if payload == "[DONE]":
        return ""
    data = json.loads(payload)
    return data["choices"][0].get("delta", {}).get("content", "")


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def run_one_request(
    *,
    request_id: int,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> RequestMetrics:
    import requests

    started = time.perf_counter()
    first_token_at = 0.0
    generated_text = []
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    try:
        with requests.post(endpoint, json=payload, stream=True, timeout=timeout_s) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                content = parse_sse_content(raw_line)
                if content and first_token_at == 0.0:
                    first_token_at = time.perf_counter()
                generated_text.append(content)
        ended = time.perf_counter()
        return RequestMetrics(
            request_id=request_id,
            ok=True,
            ttft_s=first_token_at - started if first_token_at else ended - started,
            latency_s=ended - started,
            output_tokens=estimate_tokens(" ".join(generated_text)),
            error="",
        )
    except Exception as exc:
        ended = time.perf_counter()
        return RequestMetrics(
            request_id=request_id,
            ok=False,
            ttft_s=0.0,
            latency_s=ended - started,
            output_tokens=0,
            error=str(exc),
        )


def run_benchmark(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    measured_requests: int,
    warmup_requests: int,
    timeout_s: float,
    runner=run_one_request,
) -> list[RequestMetrics]:
    rows = []
    total_requests = warmup_requests + measured_requests
    for request_id in range(total_requests):
        row = runner(
            request_id=request_id,
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        if request_id >= warmup_requests:
            rows.append(row)
    return rows


def aggregate_metrics(rows: list[RequestMetrics]) -> dict[str, float | int]:
    requests_count = len(rows)
    if requests_count == 0:
        raise ValueError("cannot aggregate zero requests")

    successes = [row for row in rows if row.ok]
    successful_requests = len(successes)
    failed_requests = requests_count - successful_requests
    if successful_requests == 0:
        return {
            "requests": requests_count,
            "successful_requests": 0,
            "failed_requests": failed_requests,
            "failure_rate": 1.0,
            "avg_ttft_s": 0.0,
            "avg_latency_s": 0.0,
            "avg_tpot_s": 0.0,
            "output_tokens_per_s": 0.0,
        }

    total_success_latency = sum(row.latency_s for row in successes)
    total_output_tokens = sum(row.output_tokens for row in successes)
    avg_tpot_values = [
        (row.latency_s - row.ttft_s) / row.output_tokens
        for row in successes
        if row.output_tokens > 0
    ]

    return {
        "requests": requests_count,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "failure_rate": failed_requests / requests_count,
        "avg_ttft_s": sum(row.ttft_s for row in successes) / successful_requests,
        "avg_latency_s": total_success_latency / successful_requests,
        "avg_tpot_s": sum(avg_tpot_values) / len(avg_tpot_values),
        "output_tokens_per_s": total_output_tokens / total_success_latency,
    }


def write_csv(path: Path, rows: Iterable[RequestMetrics]) -> None:
    row_list = list(rows)
    if not row_list:
        raise ValueError("cannot write zero benchmark rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row_list[0]).keys()))
        writer.writeheader()
        for row in row_list:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-length", choices=sorted(PROMPTS), default="short")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompt = build_prompt(args.prompt_length)
    rows = run_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        measured_requests=args.requests,
        warmup_requests=args.warmup_requests,
        timeout_s=args.timeout_s,
    )
    write_csv(args.out, rows)
    print(json.dumps(aggregate_metrics(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
