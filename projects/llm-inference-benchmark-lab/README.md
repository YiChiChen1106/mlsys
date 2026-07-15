# LLM Inference Benchmark Lab

This project studies LLM inference serving from an AI infra perspective. The first goal is to build a clean vLLM baseline on `pink` with 2 x RTX 4090, then compare serving behavior across frameworks and GPU configurations.

## Why This Project Exists

The goal is to learn how inference frameworks trade off:

- time to first token,
- time per output token,
- throughput,
- GPU memory,
- request concurrency,
- operational simplicity.

This project should become a public GitHub portfolio artifact and a practical notebook for future interviews.

## Milestones

### M0: Project Setup

- Create public project structure.
- Document benchmark metrics.
- Define the first experiment matrix.

### M1: vLLM Baseline On `pink`

- Serve one 7B/8B class model with vLLM.
- Run a small benchmark matrix on 1 x RTX 4090.
- Repeat on 2 x RTX 4090 when the setup is stable.
- Write `experiments/inference-vllm-baseline-pink.md`.

### M2: SGLang Baseline

- Repeat the benchmark matrix with SGLang.
- Compare scheduler and prefix-cache oriented behavior.
- Start with a smoke run on `pink` using the local Qwen2.5-7B-Instruct model.
- Keep the first matrix small: short-prompt concurrency, synthetic salted prefill, output-length decode, and repeated-vs-salted prefix reuse.
- See `experiments/inference-sglang-baseline-pink.md`.

### M3: llama.cpp Baseline

- Add a quantized/local-serving comparison point.
- Compare operational simplicity and memory footprint.

### M4: Report

- Summarize framework differences.
- Explain TTFT, TPOT, throughput, and memory tradeoffs.
- Convert findings into resume/interview bullets.

## First Benchmark Matrix

Keep the first run small:

| Dimension | Values |
| --- | --- |
| Framework | vLLM |
| Model | 7B/8B instruct model |
| GPU count | 1, then 2 |
| Concurrency | 1, 2, 4, 8, 16 |
| Prompt length | short, medium, long |
| Output length | short, medium |

## Metrics

- TTFT: time to first token.
- TPOT: time per output token.
- End-to-end latency.
- Output tokens per second.
- Prompt tokens.
- P50/P95/P99 TTFT and latency.
- Requests per second.
- Peak GPU memory.
- Failure rate.

## Current vLLM Findings

The first vLLM baseline uses Qwen2.5-7B-Instruct on `pink` with 2 x RTX 4090.

- TP=2 improved decode TPOT from about 15.5 ms/token to about 8.8 ms/token.
- Salted synthetic prefill runs showed TP=1 TTFT growing from about 71 ms at 575 prompt tokens to about 201 ms at 1984 prompt tokens.
- TP=2 had lower end-to-end latency but higher long-prompt TTFT than TP=1 in the salted prefill sweep, which suggests tensor-parallel communication overhead during prefill.
- At concurrency 64, TP=1 p99 latency was about 880 ms and TP=2 p99 latency was about 822 ms.
- With `max_model_len=2048`, a 1968-token prompt plus 80 requested output tokens succeeded, while 81 requested output tokens was rejected.
- KV pressure experiments showed that long prompts at concurrency 32 can push p99 TTFT into the 3-6 s range even with 0% failures.
- Prefix-cache contrast showed repeated synthetic_768 prompts reducing TTFT from about 170-206 ms to about 36-42 ms.
- Metrics-backed prefix-cache runs measured about 96-99% hit ratio for repeated prompts and about 2% hit ratio for salted varied prompts.
- KV pressure time-series showed capacity waiting queues of 27-29 requests during a synthetic_896/output64/concurrency32 run; TP=1 reported about 97% peak KV usage, while TP=2 reported about 15%.

Full results are in `experiments/inference-vllm-baseline-pink.md`.

## Next SGLang Plan

The next framework comparison is SGLang on the same `pink` machine and the same local model path:

```text
/mnt/hdd/users/cychi/hf_models/Qwen2.5-7B-Instruct
```

The first SGLang experiment should answer:

- Can SGLang serve the same local model through an OpenAI-compatible API?
- Can the existing benchmark client target SGLang by changing base URL?
- How do TTFT, TPOT, throughput, and prefix-heavy cases compare with vLLM?
- What metrics does SGLang expose for scheduler/cache analysis?

Full plan: `experiments/inference-sglang-baseline-pink.md`.

## Working Notes

Recommended first commands on the local machine:

```bash
ssh pink
nvidia-smi
```

Recommended first questions on `pink`:

- Is Docker available?
- Which CUDA driver is installed?
- Is there an existing PyTorch/vLLM environment?
- Where should benchmark outputs live?

## First vLLM Run

Inspect `pink`:

```bash
Get-Content -Raw projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh | ssh pink bash -s
```

Start vLLM on `pink` after copying or checking out this repository there:

```bash
cd ~/mlsys/projects/llm-inference-benchmark-lab
MODEL=Qwen/Qwen2.5-7B-Instruct TENSOR_PARALLEL_SIZE=1 bash scripts/run_vllm_server.sh
```

For Docker smoke tests on a cached local model snapshot:

```bash
SNAP=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
MODEL=$SNAP SERVED_MODEL_NAME=qwen2.5-0.5b-smoke CONTAINER_NAME=vllm-smoke bash scripts/run_vllm_docker.sh
```

From a second shell on `pink`, run a smoke benchmark:

```bash
python scripts/benchmark_client.py \
  --model qwen2.5-0.5b-smoke \
  --prompt-length short \
  --max-tokens 64 \
  --warmup-requests 2 \
  --requests 1 \
  --out results/vllm_smoke.csv
```

## File Layout

```text
projects/llm-inference-benchmark-lab/
  README.md
  scripts/      benchmark and helper scripts
  configs/      framework/model configs
  results/      small CSV summaries only
  reports/      polished project reports
```

Large logs, model weights, datasets, caches, and generated outputs should not be committed.
