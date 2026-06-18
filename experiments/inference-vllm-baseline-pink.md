# Experiment: vLLM Baseline On Pink

## Goal

Measure a first vLLM serving baseline on `pink` and understand how latency and throughput change with concurrency, prompt length, output length, and GPU count.

## Environment

- Machine: `pink`
- GPU: 2 x NVIDIA GeForce RTX 4090, 24 GB each
- Driver: NVIDIA 580.142, CUDA 13.0 reported by `nvidia-smi`
- Docker image: `vllm/vllm-openai:latest`
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

### 7B Baseline

```bash
docker run -d \
  --name vllm-qwen25-7b-tp1 \
  --gpus '"device=0"' \
  --ipc=host \
  -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /mnt/hdd/users/cychi/hf_models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 2048 \
  --served-model-name qwen2.5-7b-instruct
```

### 0.5B Smoke Test

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

### 7B Corrected Concurrency Matrix

```bash
python3 scripts/benchmark_client.py \
  --model qwen2.5-7b-instruct \
  --prompt-length short \
  --max-tokens 64 \
  --warmup-requests 2 \
  --requests 8 \
  --concurrency 4 \
  --timeout-s 180 \
  --out results/vllm_qwen25_7b_tp1_short_64_warm2_req8_c4_corrected.csv
```

The benchmark client measures throughput with measured wall time only. Warmup requests are excluded from the measured wall time.

### 0.5B Smoke Test

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

#### 7B, 1 x RTX 4090, Short Prompt, 64 Max Tokens

| GPU Count | Concurrency | Requests | Avg TTFT | Avg Latency | Avg TPOT | Output tok/s | Peak GPU Memory | Failure Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 8 | 0.0325 s | 0.5493 s | 0.0157 s | 60.04 | 19,714 MiB | 0.0 |
| 1 | 2 | 8 | 0.0453 s | 0.5720 s | 0.0160 s | 114.86 | 19,714 MiB | 0.0 |
| 1 | 4 | 8 | 0.0546 s | 0.5840 s | 0.0160 s | 223.54 | 19,714 MiB | 0.0 |

Raw CSV summaries:

- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm2_req8_c1_corrected.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm2_req8_c2_corrected.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm2_req8_c4_corrected.csv`

#### 0.5B Smoke Test

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
- The 7B model was copied to `pink` at `/mnt/hdd/users/cychi/hf_models/Qwen2.5-7B-Instruct` and served from that local directory with offline mode. This avoids container-side Hugging Face network failures.
- Qwen2.5-7B-Instruct fits on one RTX 4090 for this short-context baseline with `max_model_len=2048`; GPU 0 used about 19.7 GiB and GPU 1 stayed idle.
- Increasing concurrency from 1 to 4 improved measured output throughput from about 60 tok/s to about 224 tok/s, while average TTFT increased from about 33 ms to about 55 ms. This is the latency-throughput tradeoff from batching and scheduling.
- A first version of the concurrency benchmark accidentally included warmup time in measured wall-clock throughput. TTFT and per-request latency were still valid, but output tok/s was underestimated. The benchmark client now excludes warmup from measured wall time.

## What I Learned

- vLLM is an inference serving engine, not a model. It wraps a model checkpoint with scheduling, KV cache management, optimized execution, and OpenAI-compatible APIs.
- Docker is a good fit for inference experiments because it fixes the vLLM/PyTorch/CUDA runtime and makes the benchmark easier to reproduce on other machines.
- A model id such as `Qwen/Qwen2.5-0.5B-Instruct` may trigger network calls from inside the container. A local Hugging Face snapshot path plus offline mode avoids network variance.
- First-request latency can include warmup work such as CUDA graph capture or Triton kernel JIT, so cold-start latency and steady-state latency should be measured separately.
- Warmup requests are not optional bookkeeping; they are part of benchmark methodology. Otherwise you can accidentally report cold-start behavior as steady-state performance.
- For concurrent serving, throughput should be computed from total generated tokens divided by measured wall-clock time, not by summing each request's latency. Summed latency double-counts overlapping requests.
- Warmup should be outside the measured window. Otherwise benchmark numbers mix methodology overhead with steady-state serving behavior.
- vLLM's value starts to show under concurrent load: average latency rises slightly, but wall-clock throughput increases because the scheduler can batch decode work across requests.

## Interview Answer

```text
I started by running vLLM in Docker on a dual RTX 4090 server and used a cached Qwen2.5-0.5B snapshot for a smoke test. The container could see both GPUs, but loading by Hugging Face model id failed because the container could not retrieve the file list from Hugging Face. I fixed the setup by mounting the host Hugging Face cache into the container and starting vLLM from the local snapshot path with HF_HUB_OFFLINE=1. The first request succeeded, but the TTFT was high because the logs showed a Triton JIT compilation during inference. After adding two warmup requests, steady-state TTFT dropped from about 0.91s to about 0.018s, which is why benchmark methodology has to separate cold start from steady-state performance.

For the 7B baseline, I copied Qwen2.5-7B-Instruct to the server and served it from a local directory in vLLM Docker with offline mode. On one RTX 4090, a short-prompt 64-token benchmark used about 19.7 GiB of GPU memory. With two warmup requests and measured concurrency 1, 2, and 4, output throughput increased from about 60 to 115 to 224 tokens/s, while TTFT rose from about 33 ms to 45 ms to 55 ms. I also fixed the benchmark client so concurrent throughput uses measured wall-clock time excluding warmup, because summing per-request latency would not represent server throughput under overlapping requests.
```

## Follow-up Questions

- How much does TTFT drop after one or more warmup requests?
- Does vLLM Docker still need offline mode for larger cached models?
- How does the curve continue at concurrency 8 and 16?
- How much does prompt length affect TTFT through prefill cost?
- How much does output length affect TPOT and decode throughput?
- Does 2-GPU tensor parallelism help on dual RTX 4090, or does PCIe communication erase the benefit for this model size?

## Next Step

- Sweep concurrency 8 and 16 on the 7B model.
- Add medium and long prompt cases to expose prefill cost.
- Repeat with 2 x RTX 4090 tensor parallelism and compare throughput, TTFT, and memory.

## Resume / Interview Sentence

```text
I built a small LLM inference benchmark lab to measure vLLM serving behavior on dual RTX 4090 GPUs, tracking TTFT, TPOT, throughput, memory, and failure rate across concurrency and sequence-length sweeps.
```
