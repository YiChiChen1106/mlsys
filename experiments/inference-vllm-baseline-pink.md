# Experiment: vLLM Baseline On Pink

## Goal

Measure a first vLLM serving baseline on `pink` and understand how latency and throughput change with concurrency, prompt length, output length, and GPU count.

## Environment

- Machine: `pink`
- GPU: 2 x NVIDIA GeForce RTX 4090, 24 GB each
- Driver: NVIDIA 580.142, CUDA 13.0 reported by `nvidia-smi`
- Docker image: not selected yet; Docker 29.1.5 is installed
- Python: 3.12.2 system Python
- Framework: vLLM
- Model: `Qwen/Qwen2.5-7B-Instruct`

## Setup

```bash
ssh pink
nvidia-smi
Get-Content -Raw projects/llm-inference-benchmark-lab/scripts/inspect_pink.sh | ssh pink bash -s
```

## Server Command

```bash
SNAP=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
docker run -d \
  --name vllm-smoke \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 2048 \
  --served-model-name qwen2.5-0.5b-smoke
```

## Benchmark Command

```bash
python ~/mlsys-inference-smoke/scripts/benchmark_client.py \
  --model qwen2.5-0.5b-smoke \
  --prompt-length short \
  --max-tokens 64 \
  --warmup-requests 2 \
  --requests 1 \
  --out ~/mlsys-inference-smoke/results/vllm_smoke.csv
```

## Benchmark Matrix

| GPU Count | Concurrency | Prompt Length | Output Length |
| ---: | ---: | ---: | ---: |
| 1 | 1, 2, 4, 8, 16 | short, medium, long | short, medium |
| 2 | 1, 2, 4, 8, 16 | short, medium, long | short, medium |

## Results

### Cold Start

| GPU Count | Concurrency | TTFT | TPOT | Output tok/s | Peak GPU Memory | Failure Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.9094 s | 0.0021 s | 61.27 | 19,778 MiB | 0.0 |

### Warmed Steady State

| GPU Count | Concurrency | TTFT | TPOT | Output tok/s | Peak GPU Memory | Failure Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.0177 s | 0.0018 s | 488.53 | 19,778 MiB | 0.0 |

## Interpretation

- `pink` has two idle RTX 4090 GPUs, a recent NVIDIA driver, Docker, Python 3.12.2, and enough disk space.
- Docker GPU runtime works: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` sees both RTX 4090 GPUs.
- The vLLM Docker image was already present on `pink`.
- Starting from the Hugging Face model id failed inside the container because vLLM could not retrieve the Hugging Face file list from container networking.
- Starting from the cached local snapshot path with `HF_HUB_OFFLINE=1` worked.
- The first smoke request succeeded and wrote `~/mlsys-inference-smoke/results/vllm_smoke.csv`.
- The first request had high TTFT because the logs showed a Triton kernel JIT compilation during inference: `_compute_slot_mapping_kernel`.
- The first attempt to pipe the inspection script from Windows exposed a CRLF issue, so shell scripts are now forced to LF with `.gitattributes`.
- Warmup requests changed the benchmark a lot: after 2 warmup requests, measured TTFT dropped from about 0.91s to about 0.018s and output throughput rose from about 61 tokens/s to about 489 tokens/s.

## What I Learned

- vLLM is an inference serving engine, not a model. It wraps a model checkpoint with scheduling, KV cache management, optimized execution, and OpenAI-compatible APIs.
- Docker is a good fit for inference experiments because it fixes the vLLM/PyTorch/CUDA runtime and makes the benchmark easier to reproduce on other machines.
- A model id such as `Qwen/Qwen2.5-0.5B-Instruct` may trigger network calls from inside the container. A local Hugging Face snapshot path plus offline mode avoids network variance.
- First-request latency can include warmup work such as CUDA graph capture or Triton kernel JIT, so cold-start latency and steady-state latency should be measured separately.
- Warmup requests are not optional bookkeeping; they are part of benchmark methodology. Otherwise you can accidentally report cold-start behavior as steady-state performance.

## Interview Answer

```text
I started by running vLLM in Docker on a dual RTX 4090 server and used a cached Qwen2.5-0.5B snapshot for a smoke test. The container could see both GPUs, but loading by Hugging Face model id failed because the container could not retrieve the file list from Hugging Face. I fixed the setup by mounting the host Hugging Face cache into the container and starting vLLM from the local snapshot path with HF_HUB_OFFLINE=1. The first request succeeded, but the TTFT was high because the logs showed a Triton JIT compilation during inference. After adding two warmup requests, steady-state TTFT dropped from about 0.91s to about 0.018s, which is why benchmark methodology has to separate cold start from steady-state performance.
```

## Follow-up Questions

- How much does TTFT drop after one or more warmup requests?
- Does vLLM Docker still need offline mode for larger cached models?
- What changes when moving from 0.5B smoke testing to a 7B/8B baseline?
- How should I choose a warmup count for a real 7B/8B benchmark?

## Next Step

- Install or activate vLLM.
- Start the vLLM server with `TENSOR_PARALLEL_SIZE=1`.
- Run the one-request smoke benchmark and paste the summary here.

## Resume / Interview Sentence

```text
I built a small LLM inference benchmark lab to measure vLLM serving behavior on dual RTX 4090 GPUs, tracking TTFT, TPOT, throughput, memory, and failure rate across concurrency and sequence-length sweeps.
```
