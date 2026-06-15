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
- Requests per second.
- Peak GPU memory.
- Failure rate.

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
