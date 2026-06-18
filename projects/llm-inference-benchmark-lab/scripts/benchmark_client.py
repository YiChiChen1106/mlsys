from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
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


@dataclass(frozen=True)
class StreamEvent:
    content: str
    completion_tokens: int | None


def build_prompt(length: str) -> str:
    if length not in PROMPTS:
        raise ValueError(f"unknown prompt length: {length}")
    return PROMPTS[length]


def parse_sse_event(line: str) -> StreamEvent:
    if not line.startswith("data: "):
        return StreamEvent(content="", completion_tokens=None)
    payload = line.removeprefix("data: ").strip()
    if payload == "[DONE]":
        return StreamEvent(content="", completion_tokens=None)
    data = json.loads(payload)
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    choices = data.get("choices") or []
    if not choices:
        return StreamEvent(content="", completion_tokens=completion_tokens)
    content = choices[0].get("delta", {}).get("content", "")
    return StreamEvent(content=content, completion_tokens=completion_tokens)


def parse_sse_content(line: str) -> str:
    return parse_sse_event(line).content


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def prompt_for_request(prompt: str, request_id: int, *, vary_prompts: bool) -> str:
    if not vary_prompts:
        return prompt
    return f"[request {request_id}] {prompt}"


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
    output_tokens = None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        with requests.post(endpoint, json=payload, stream=True, timeout=timeout_s) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                event = parse_sse_event(raw_line)
                content = event.content
                if content and first_token_at == 0.0:
                    first_token_at = time.perf_counter()
                generated_text.append(content)
                if event.completion_tokens is not None:
                    output_tokens = event.completion_tokens
        ended = time.perf_counter()
        return RequestMetrics(
            request_id=request_id,
            ok=True,
            ttft_s=first_token_at - started if first_token_at else ended - started,
            latency_s=ended - started,
            output_tokens=output_tokens if output_tokens is not None else estimate_tokens(" ".join(generated_text)),
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
    concurrency: int = 1,
    vary_prompts: bool = False,
    timeout_s: float,
    runner=run_one_request,
) -> list[RequestMetrics]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    rows = []
    for request_id in range(warmup_requests):
        runner(
            request_id=request_id,
            endpoint=endpoint,
            model=model,
            prompt=prompt_for_request(prompt, request_id, vary_prompts=vary_prompts),
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    measured_ids = range(warmup_requests, warmup_requests + measured_requests)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                runner,
                request_id=request_id,
                endpoint=endpoint,
                model=model,
                prompt=prompt_for_request(prompt, request_id, vary_prompts=vary_prompts),
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            for request_id in measured_ids
        ]
        for future in futures:
            rows.append(future.result())
    return rows


def run_measured_benchmark(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    measured_requests: int,
    warmup_requests: int,
    concurrency: int = 1,
    vary_prompts: bool = False,
    timeout_s: float,
    runner=run_one_request,
) -> tuple[list[RequestMetrics], float]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    for request_id in range(warmup_requests):
        runner(
            request_id=request_id,
            endpoint=endpoint,
            model=model,
            prompt=prompt_for_request(prompt, request_id, vary_prompts=vary_prompts),
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    measured_started = time.perf_counter()
    rows = []
    measured_ids = range(warmup_requests, warmup_requests + measured_requests)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                runner,
                request_id=request_id,
                endpoint=endpoint,
                model=model,
                prompt=prompt_for_request(prompt, request_id, vary_prompts=vary_prompts),
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            for request_id in measured_ids
        ]
        for future in futures:
            rows.append(future.result())
    return rows, time.perf_counter() - measured_started


def aggregate_metrics(
    rows: list[RequestMetrics],
    *,
    wall_time_s: float | None = None,
) -> dict[str, float | int]:
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
    throughput_denominator = wall_time_s if wall_time_s is not None else total_success_latency
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
        "output_tokens_per_s": total_output_tokens / throughput_denominator,
        "wall_time_s": wall_time_s or 0.0,
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
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--vary-prompts", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompt = build_prompt(args.prompt_length)
    rows, measured_wall_time_s = run_measured_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        measured_requests=args.requests,
        warmup_requests=args.warmup_requests,
        concurrency=args.concurrency,
        vary_prompts=args.vary_prompts,
        timeout_s=args.timeout_s,
    )
    write_csv(args.out, rows)
    print(json.dumps(aggregate_metrics(rows, wall_time_s=measured_wall_time_s), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
