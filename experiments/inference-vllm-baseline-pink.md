# Experiment: vLLM Baseline On Pink

## Goal

Measure a first vLLM serving baseline on `pink` and understand how latency and throughput change with concurrency, prompt length, output length, and GPU count.

## Environment

- Machine: `pink`
- GPU: 2 x NVIDIA GeForce RTX 4090, 24 GB each
- Driver:
- Docker image:
- Python:
- Framework: vLLM
- Model: `Qwen/Qwen2.5-7B-Instruct`

## Setup

```bash
ssh pink
nvidia-smi
ssh pink 'bash -s' < projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh
```

## Server Command

```bash
cd ~/mlsys/projects/llm-inference-benchmark-lab
MODEL=Qwen/Qwen2.5-7B-Instruct TENSOR_PARALLEL_SIZE=1 bash scripts/run_vllm_server.sh
```

## Benchmark Command

```bash
python scripts/benchmark_client.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt-length short \
  --max-tokens 64 \
  --requests 1 \
  --out results/vllm_smoke.csv
```

## Benchmark Matrix

| GPU Count | Concurrency | Prompt Length | Output Length |
| ---: | ---: | ---: | ---: |
| 1 | 1, 2, 4, 8, 16 | short, medium, long | short, medium |
| 2 | 1, 2, 4, 8, 16 | short, medium, long | short, medium |

## Results

| GPU Count | Concurrency | TTFT | TPOT | Output tok/s | Peak GPU Memory | Failure Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  |  |  |  |  |  |  |

## Interpretation

- Ready for the first run. The initial smoke test should validate that the server starts, streams tokens, records TTFT and latency, and writes a CSV.

## Next Step

- Run `inspect_pink.sh` on `pink`.
- Install or activate vLLM.
- Start the vLLM server with `TENSOR_PARALLEL_SIZE=1`.
- Run the one-request smoke benchmark and paste the summary here.

## Resume / Interview Sentence

```text
I built a small LLM inference benchmark lab to measure vLLM serving behavior on dual RTX 4090 GPUs, tracking TTFT, TPOT, throughput, memory, and failure rate across concurrency and sequence-length sweeps.
```
