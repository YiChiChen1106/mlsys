# vLLM Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable vLLM baseline workflow for `projects/llm-inference-benchmark-lab`.

**Architecture:** Keep the project small and testable: shell scripts inspect/start the remote environment, while Python code handles benchmark request generation, response timing, result aggregation, and CSV output. The first milestone targets vLLM only; SGLang and llama.cpp stay documented future work.

**Tech Stack:** Python standard library, `requests` for HTTP client behavior, `pytest` for local tests, vLLM OpenAI-compatible server on `pink`.

---

## File Structure

- Create `projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh`: records GPU, Docker, Python, and disk context on `pink`.
- Create `projects/llm-inference-benchmark-lab/scripts/run_vllm_server.sh`: starts a vLLM OpenAI-compatible server with explicit model and tensor-parallel parameters.
- Create `projects/llm-inference-benchmark-lab/scripts/benchmark_client.py`: sends benchmark requests, measures TTFT and end-to-end latency, computes TPOT and throughput, and writes CSV rows.
- Create `projects/llm-inference-benchmark-lab/scripts/test_benchmark_client.py`: unit tests for prompt generation, streaming parsing, and metric aggregation.
- Create `projects/llm-inference-benchmark-lab/configs/vllm_baseline.json`: first benchmark matrix.
- Modify `projects/llm-inference-benchmark-lab/README.md`: add exact setup and run commands after scripts exist.
- Modify `experiments/inference-vllm-baseline-pink.md`: replace placeholders with environment findings and first-run instructions.
- Modify `daily/2026-06-15.md`: record completed setup and next step.

## Task 1: Add Benchmark Client Metrics With Tests

**Files:**

- Create: `projects/llm-inference-benchmark-lab/scripts/test_benchmark_client.py`
- Create: `projects/llm-inference-benchmark-lab/scripts/benchmark_client.py`

- [ ] **Step 1: Write failing tests for metric aggregation**

Create `projects/llm-inference-benchmark-lab/scripts/test_benchmark_client.py`:

```python
from benchmark_client import RequestMetrics, aggregate_metrics, build_prompt


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
    assert summary["failure_rate"] == 1 / 3
    assert summary["avg_ttft_s"] == 0.30
    assert summary["avg_latency_s"] == 1.30
    assert summary["avg_tpot_s"] == 0.05
    assert summary["output_tokens_per_s"] == 40 / 2.60
```

- [ ] **Step 2: Run tests and verify they fail because the module is missing**

Run:

```bash
cd projects/llm-inference-benchmark-lab/scripts
python -m pytest test_benchmark_client.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'benchmark_client'
```

- [ ] **Step 3: Implement minimal metric and prompt code**

Create `projects/llm-inference-benchmark-lab/scripts/benchmark_client.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


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


def aggregate_metrics(rows: list[RequestMetrics]) -> dict[str, float | int]:
    requests = len(rows)
    successes = [row for row in rows if row.ok]
    successful_requests = len(successes)
    failed_requests = requests - successful_requests
    total_success_latency = sum(row.latency_s for row in successes)
    total_output_tokens = sum(row.output_tokens for row in successes)

    avg_ttft_s = sum(row.ttft_s for row in successes) / successful_requests
    avg_latency_s = total_success_latency / successful_requests
    avg_tpot_s = sum(
        (row.latency_s - row.ttft_s) / row.output_tokens
        for row in successes
        if row.output_tokens > 0
    ) / successful_requests

    return {
        "requests": requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "failure_rate": failed_requests / requests,
        "avg_ttft_s": avg_ttft_s,
        "avg_latency_s": avg_latency_s,
        "avg_tpot_s": avg_tpot_s,
        "output_tokens_per_s": total_output_tokens / total_success_latency,
    }
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
cd projects/llm-inference-benchmark-lab/scripts
python -m pytest test_benchmark_client.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add projects/llm-inference-benchmark-lab/scripts/benchmark_client.py projects/llm-inference-benchmark-lab/scripts/test_benchmark_client.py
git commit -m "bench: add inference benchmark metrics"
```

## Task 2: Add OpenAI-Compatible Streaming Client

**Files:**

- Modify: `projects/llm-inference-benchmark-lab/scripts/test_benchmark_client.py`
- Modify: `projects/llm-inference-benchmark-lab/scripts/benchmark_client.py`

- [ ] **Step 1: Write failing tests for streamed chunk parsing**

Append to `projects/llm-inference-benchmark-lab/scripts/test_benchmark_client.py`:

```python
from benchmark_client import parse_sse_content


def test_parse_sse_content_extracts_delta_text():
    line = 'data: {"choices":[{"delta":{"content":"hello"}}]}'

    assert parse_sse_content(line) == "hello"


def test_parse_sse_content_ignores_done_marker():
    assert parse_sse_content("data: [DONE]") == ""
```

- [ ] **Step 2: Run tests and verify they fail because parser is missing**

Run:

```bash
cd projects/llm-inference-benchmark-lab/scripts
python -m pytest test_benchmark_client.py -q
```

Expected:

```text
ImportError: cannot import name 'parse_sse_content'
```

- [ ] **Step 3: Implement parser and request runner**

Replace `projects/llm-inference-benchmark-lab/scripts/benchmark_client.py` with:

```python
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests


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


def aggregate_metrics(rows: list[RequestMetrics]) -> dict[str, float | int]:
    requests_count = len(rows)
    successes = [row for row in rows if row.ok]
    successful_requests = len(successes)
    failed_requests = requests_count - successful_requests
    if requests_count == 0:
        raise ValueError("cannot aggregate zero requests")
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
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompt = build_prompt(args.prompt_length)
    rows = [
        run_one_request(
            request_id=request_id,
            endpoint=args.endpoint,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
        )
        for request_id in range(args.requests)
    ]
    write_csv(args.out, rows)
    print(json.dumps(aggregate_metrics(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
cd projects/llm-inference-benchmark-lab/scripts
python -m pytest test_benchmark_client.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add projects/llm-inference-benchmark-lab/scripts/benchmark_client.py projects/llm-inference-benchmark-lab/scripts/test_benchmark_client.py
git commit -m "bench: add streaming inference client"
```

## Task 3: Add Remote Environment And vLLM Server Scripts

**Files:**

- Create: `projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh`
- Create: `projects/llm-inference-benchmark-lab/scripts/run_vllm_server.sh`
- Create: `projects/llm-inference-benchmark-lab/configs/vllm_baseline.json`

- [ ] **Step 1: Add environment inspection script**

Create `projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "## Host"
hostname

echo "## GPU"
nvidia-smi

echo "## Docker"
if command -v docker >/dev/null 2>&1; then
  docker --version
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
else
  echo "docker: not found"
fi

echo "## Python"
if command -v python >/dev/null 2>&1; then
  python --version
else
  echo "python: not found"
fi

echo "## Disk"
df -h .
```

- [ ] **Step 2: Add vLLM server script**

Create `projects/llm-inference-benchmark-lab/scripts/run_vllm_server.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --host 0.0.0.0 \
  --port "${PORT}"
```

- [ ] **Step 3: Add first benchmark config**

Create `projects/llm-inference-benchmark-lab/configs/vllm_baseline.json`:

```json
{
  "framework": "vllm",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "gpu_counts": [1, 2],
  "concurrency": [1, 2, 4, 8, 16],
  "prompt_lengths": ["short", "medium", "long"],
  "output_lengths": [64, 256],
  "metrics": [
    "ttft_s",
    "tpot_s",
    "latency_s",
    "output_tokens_per_s",
    "peak_gpu_memory",
    "failure_rate"
  ]
}
```

- [ ] **Step 4: Run shell syntax checks**

Run:

```bash
bash -n projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh
bash -n projects/llm-inference-benchmark-lab/scripts/run_vllm_server.sh
```

Expected: no output and exit code 0.

- [ ] **Step 5: Commit**

```bash
git add projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh projects/llm-inference-benchmark-lab/scripts/run_vllm_server.sh projects/llm-inference-benchmark-lab/configs/vllm_baseline.json
git commit -m "bench: add vllm baseline scripts"
```

## Task 4: Update Public Docs For First Run

**Files:**

- Modify: `projects/llm-inference-benchmark-lab/README.md`
- Modify: `experiments/inference-vllm-baseline-pink.md`
- Modify: `daily/2026-06-15.md`

- [ ] **Step 1: Update project README commands**

Add this section to `projects/llm-inference-benchmark-lab/README.md`:

```md
## First vLLM Run

Inspect `pink`:

```bash
ssh pink 'bash -s' < projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh
```

Start vLLM on `pink` after copying or checking out this repository there:

```bash
cd ~/mlsys/projects/llm-inference-benchmark-lab
MODEL=Qwen/Qwen2.5-7B-Instruct TENSOR_PARALLEL_SIZE=1 bash scripts/run_vllm_server.sh
```

From a second shell on `pink`, run a smoke benchmark:

```bash
python scripts/benchmark_client.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt-length short \
  --max-tokens 64 \
  --requests 1 \
  --out results/vllm_smoke.csv
```
```

- [ ] **Step 2: Update experiment note with the same smoke path**

Replace the placeholder setup commands in `experiments/inference-vllm-baseline-pink.md` with the inspect/server/client commands from Step 1.

- [ ] **Step 3: Update daily note next step**

Change the next step in `daily/2026-06-15.md` to:

```md
### Next Step

- Run `inspect_pink.sh` on `pink`.
- Install or activate vLLM.
- Start the vLLM server with `TENSOR_PARALLEL_SIZE=1`.
- Run the one-request smoke benchmark and paste the summary into `experiments/inference-vllm-baseline-pink.md`.
```

- [ ] **Step 4: Verify docs contain no stale placeholders**

Run:

```bash
rg -n "To be filled|Pending first run|to be selected" projects/llm-inference-benchmark-lab experiments/inference-vllm-baseline-pink.md daily/2026-06-15.md
```

Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add projects/llm-inference-benchmark-lab/README.md experiments/inference-vllm-baseline-pink.md daily/2026-06-15.md
git commit -m "docs: document first vllm baseline run"
```

## Task 5: Run Local Verification And Push Branch

**Files:**

- No code changes expected.

- [ ] **Step 1: Run Python tests**

Run:

```bash
cd projects/llm-inference-benchmark-lab/scripts
python -m pytest test_benchmark_client.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 2: Run shell syntax checks**

Run:

```bash
bash -n projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh
bash -n projects/llm-inference-benchmark-lab/scripts/run_vllm_server.sh
```

Expected: no output and exit code 0.

- [ ] **Step 3: Review changed files**

Run:

```bash
git status --short
git diff --stat main...HEAD
```

Expected: only inference-track docs, notes, scripts, configs, and project files changed.

- [ ] **Step 4: Push branch**

Run:

```bash
git push -u origin codex/inference-benchmark-lab
```

Expected: branch pushed to GitHub.

## Self-Review

- Spec coverage: the plan creates the project workflow, benchmark client, vLLM server script, config, docs, and handoff notes needed for the first vLLM baseline.
- Placeholder scan: no task uses open-ended placeholders. Runtime environment values are intentionally discovered by `inspect_pink.sh`.
- Type consistency: `RequestMetrics`, `aggregate_metrics`, `build_prompt`, and `parse_sse_content` are introduced before use and referenced consistently.
