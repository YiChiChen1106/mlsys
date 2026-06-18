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

#### TP=1

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

#### TP=2

```bash
docker run -d \
  --name vllm-qwen25-7b-tp2 \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /mnt/hdd/users/cychi/hf_models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 2048 \
  --served-model-name qwen2.5-7b-instruct-tp2 \
  --disable-custom-all-reduce
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
  --requests 16 \
  --concurrency 16 \
  --timeout-s 180 \
  --out results/vllm_qwen25_7b_tp1_short_64_warm2_req16_c16_usage.csv
```

The benchmark client measures throughput with measured wall time only. Warmup requests are excluded from the measured wall time.
Streaming requests use `stream_options.include_usage=true`, so `output_tokens` comes from vLLM's `completion_tokens` usage field instead of whitespace-based text estimation.

For prefill-sensitive prompt-length checks, vary prompts so prefix caching does not hide most prompt processing work:

```bash
python3 scripts/benchmark_client.py \
  --model qwen2.5-7b-instruct \
  --prompt-length long \
  --max-tokens 64 \
  --warmup-requests 2 \
  --requests 8 \
  --concurrency 4 \
  --vary-prompts \
  --timeout-s 180 \
  --out results/vllm_qwen25_7b_tp1_long_64_warm2_req8_c4_usage_varied.csv
```

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

#### 7B, 1 x RTX 4090, Short Prompt, 64 Max Tokens, Repeated Prompt

This sweep uses repeated prompts. It is useful for observing decode batching and scheduler behavior, but prefix cache can make it less representative of unrelated user traffic.

| GPU Count | Concurrency | Requests | Avg TTFT | Avg Latency | Avg TPOT | Output tok/s | Peak GPU Memory | Failure Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 16 | 0.0325 s | 0.5493 s | 0.0152 s | 61.86 | 19,714 MiB | 0.0 |
| 1 | 2 | 16 | 0.0461 s | 0.5730 s | 0.0155 s | 118.39 | 19,714 MiB | 0.0 |
| 1 | 4 | 16 | 0.0582 s | 0.5872 s | 0.0156 s | 229.41 | 19,714 MiB | 0.0 |
| 1 | 8 | 16 | 0.0702 s | 0.5991 s | 0.0156 s | 443.82 | 19,714 MiB | 0.0 |
| 1 | 16 | 16 | 0.1184 s | 0.7297 s | 0.0180 s | 710.20 | 19,714 MiB | 0.0 |

#### 7B, 2 x RTX 4090, TP=2, Short Prompt, 64 Max Tokens, Repeated Prompt

Topology:

```text
GPU0 <-> GPU1: SYS
```

This means the two GPUs communicate across PCIe and the CPU interconnect, not NVLink.

| GPU Count | Tensor Parallel | Concurrency | Requests | Avg TTFT | Avg Latency | Avg TPOT | Output tok/s | Peak GPU Memory | Failure Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 1 | 16 | 0.0222 s | 0.3157 s | 0.0086 s | 107.61 | 20,662 MiB/GPU | 0.0 |
| 2 | 2 | 2 | 16 | 0.0371 s | 0.3456 s | 0.0091 s | 195.86 | 20,662 MiB/GPU | 0.0 |
| 2 | 2 | 4 | 16 | 0.0585 s | 0.3924 s | 0.0098 s | 341.35 | 20,662 MiB/GPU | 0.0 |
| 2 | 2 | 8 | 16 | 0.0535 s | 0.3914 s | 0.0099 s | 674.72 | 20,662 MiB/GPU | 0.0 |
| 2 | 2 | 16 | 16 | 0.0717 s | 0.4542 s | 0.0112 s | 1099.00 | 20,662 MiB/GPU | 0.0 |

#### TP=1 vs TP=2 Comparison

| Concurrency | TP=1 tok/s | TP=2 tok/s | TP=2 Throughput Speedup | TP=1 TTFT | TP=2 TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 61.86 | 107.61 | 1.74x | 0.0325 s | 0.0222 s |
| 2 | 118.39 | 195.86 | 1.65x | 0.0461 s | 0.0371 s |
| 4 | 229.41 | 341.35 | 1.49x | 0.0582 s | 0.0585 s |
| 8 | 443.82 | 674.72 | 1.52x | 0.0702 s | 0.0535 s |
| 16 | 710.20 | 1099.00 | 1.55x | 0.1184 s | 0.0717 s |

#### 7B, 1 x RTX 4090, Varied Prompt, 64 Max Tokens

This smaller sweep prepends a unique request id to each prompt to reduce prefix-cache reuse.

| Prompt | GPU Count | Concurrency | Requests | Avg TTFT | Avg Latency | Avg TPOT | Output tok/s | Failure Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | 1 | 1 | 8 | 0.0319 s | 1.0180 s | 0.0154 s | 62.85 | 0.0 |
| medium | 1 | 4 | 8 | 0.0548 s | 1.0634 s | 0.0158 s | 238.79 | 0.0 |
| long | 1 | 1 | 8 | 0.0330 s | 1.0198 s | 0.0154 s | 62.74 | 0.0 |
| long | 1 | 4 | 8 | 0.0559 s | 1.0642 s | 0.0158 s | 238.66 | 0.0 |

Raw CSV summaries:

- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm2_req16_c1_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm2_req16_c2_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm2_req16_c4_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm2_req16_c8_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm2_req16_c16_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_short_64_warm2_req16_c1_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_short_64_warm2_req16_c2_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_short_64_warm2_req16_c4_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_short_64_warm2_req16_c8_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_short_64_warm2_req16_c16_usage.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_medium_64_warm2_req8_c1_usage_varied.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_medium_64_warm2_req8_c4_usage_varied.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_long_64_warm2_req8_c1_usage_varied.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_long_64_warm2_req8_c4_usage_varied.csv`

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
- Increasing concurrency from 1 to 16 improved measured output throughput from about 62 tok/s to about 710 tok/s, while average TTFT increased from about 32 ms to about 118 ms. This is the latency-throughput tradeoff from batching and scheduling.
- A first version of the concurrency benchmark accidentally included warmup time in measured wall-clock throughput. TTFT and per-request latency were still valid, but output tok/s was underestimated. The benchmark client now excludes warmup from measured wall time.
- A second benchmark-client correction uses vLLM streaming usage to count completion tokens. This is more accurate than estimating tokens from generated text with whitespace splitting.
- The medium and long prompts used here are still short in token terms. Their TTFT stayed close to the short-prompt run; the larger end-to-end latency came mostly from generating 64 output tokens instead of about 34 output tokens in the short prompt.
- Repeated prompt benchmarks can show optimistic prefill behavior because prefix cache may reuse shared prompt prefixes. The varied prompt mode reduces that effect by making request prefixes different.
- `nvidia-smi topo -m` reports `SYS` between GPU0 and GPU1, so tensor parallel communication crosses PCIe and the CPU interconnect rather than NVLink.
- TP=2 still improved short-prompt decode-heavy throughput by about 1.5x to 1.7x across this small concurrency sweep. At concurrency 16, throughput rose from about 710 tok/s to about 1099 tok/s and TTFT dropped from about 118 ms to about 72 ms.
- TP=2 used about 20.7 GiB on each GPU. Tensor parallelism splits model compute, but serving memory also includes KV cache blocks, CUDA graphs, runtime buffers, and per-rank overhead; do not expect visible `nvidia-smi` memory to halve.
- TP=2 was launched with `--disable-custom-all-reduce` for stability on this PCIe/SYS topology.

## What I Learned

- vLLM is an inference serving engine, not a model. It wraps a model checkpoint with scheduling, KV cache management, optimized execution, and OpenAI-compatible APIs.
- Docker is a good fit for inference experiments because it fixes the vLLM/PyTorch/CUDA runtime and makes the benchmark easier to reproduce on other machines.
- A model id such as `Qwen/Qwen2.5-0.5B-Instruct` may trigger network calls from inside the container. A local Hugging Face snapshot path plus offline mode avoids network variance.
- First-request latency can include warmup work such as CUDA graph capture or Triton kernel JIT, so cold-start latency and steady-state latency should be measured separately.
- Warmup requests are not optional bookkeeping; they are part of benchmark methodology. Otherwise you can accidentally report cold-start behavior as steady-state performance.
- For concurrent serving, throughput should be computed from total generated tokens divided by measured wall-clock time, not by summing each request's latency. Summed latency double-counts overlapping requests.
- Warmup should be outside the measured window. Otherwise benchmark numbers mix methodology overhead with steady-state serving behavior.
- vLLM's value starts to show under concurrent load: average latency rises slightly, but wall-clock throughput increases because the scheduler can batch decode work across requests.
- `max_tokens` is a cap, not a guarantee. The model can stop early, so token throughput needs actual completion token counts from the server.
- Prompt-length experiments need either varied prompts or disabled prefix caching. Otherwise the benchmark can accidentally measure cache hits instead of prefill work.
- Tensor parallelism is not "free multi-GPU speedup." Each layer introduces cross-GPU communication, so the benefit depends on compute saved versus communication overhead. On this short-output 7B run, TP=2 helped, but the speedup was below 2x.
- GPU memory under vLLM includes allocated KV cache and execution buffers, so apparent memory usage can remain high on every GPU even when weights are sharded.

## Interview Answer

```text
I started by running vLLM in Docker on a dual RTX 4090 server and used a cached Qwen2.5-0.5B snapshot for a smoke test. The container could see both GPUs, but loading by Hugging Face model id failed because the container could not retrieve the file list from Hugging Face. I fixed the setup by mounting the host Hugging Face cache into the container and starting vLLM from the local snapshot path with HF_HUB_OFFLINE=1. The first request succeeded, but the TTFT was high because the logs showed a Triton JIT compilation during inference. After adding two warmup requests, steady-state TTFT dropped from about 0.91s to about 0.018s, which is why benchmark methodology has to separate cold start from steady-state performance.

For the 7B baseline, I copied Qwen2.5-7B-Instruct to the server and served it from a local directory in vLLM Docker with offline mode. On one RTX 4090, the service used about 19.7 GiB of GPU memory. With two warmup requests and measured concurrency 1, 2, 4, 8, and 16, output throughput increased from about 62 to 118 to 229 to 444 to 710 tokens/s, while TTFT rose from about 32 ms to 118 ms. I fixed the benchmark client so concurrent throughput uses measured wall-clock time excluding warmup, and token throughput uses vLLM's streaming completion token usage instead of whitespace estimation. I also added a varied-prompt mode because repeated prompts can hit prefix cache and hide prefill cost.

Then I repeated the same short-prompt benchmark with tensor-parallel size 2 across both RTX 4090s. The GPU topology was `SYS`, so the GPUs communicate over PCIe/CPU interconnect rather than NVLink. Even with that communication cost, TP=2 improved throughput from about 62 to 108 tok/s at concurrency 1 and from about 710 to 1099 tok/s at concurrency 16. The speedup was meaningful but below 2x, which is expected because tensor parallelism adds all-reduce communication and serving overhead. TP=2 also used about 20.7 GiB on each GPU, reminding me that serving memory is not just model weights; KV cache, CUDA graphs, and runtime buffers matter too.
```

## Follow-up Questions

- How much does TTFT drop after one or more warmup requests?
- Does vLLM Docker still need offline mode for larger cached models?
- At what concurrency does tail latency start to rise sharply?
- How much does truly long prompt length affect TTFT through prefill cost?
- How much does output length affect TPOT and decode throughput?
- Does 2-GPU tensor parallelism help on dual RTX 4090, or does PCIe communication erase the benefit for this model size?
- Should prefix caching be disabled for a clean prefill benchmark, or should the report show both cached and uncached traffic?
- Would TP=2 still win on truly long prompts and larger output lengths?
- Would concurrency 32 saturate TP=2 or start hurting tail latency?

## Next Step

- Add p50/p95 latency and TTFT reporting.
- Add truly longer synthetic prompts to expose prefill cost.
- Run TP=1 vs TP=2 with longer prompts and longer outputs.
- Add concurrency 32 for the short-prompt sweep.

## Resume / Interview Sentence

```text
I built a small LLM inference benchmark lab to measure vLLM serving behavior on dual RTX 4090 GPUs, tracking TTFT, TPOT, throughput, memory, and failure rate across concurrency and sequence-length sweeps.
```
