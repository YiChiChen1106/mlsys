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

#### TP=1 vs TP=2 Tail Latency At Concurrency 16

This run uses 64 measured requests and 4 warmup requests. Tail latency is more useful than average latency for scheduler work because queueing and batch admission usually show up first in p95/p99.

| TP Size | Requests | Concurrency | Avg Latency | P50 Latency | P95 Latency | P99 Latency | Avg TTFT | P95 TTFT | P99 TTFT | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 16 | 0.6619 s | 0.6398 s | 0.7585 s | 0.7655 s | 0.0963 s | 0.1161 s | 0.2135 s | 801.95 |
| 2 | 64 | 16 | 0.4689 s | 0.4699 s | 0.4967 s | 0.5059 s | 0.0823 s | 0.1007 s | 0.1140 s | 1125.94 |

#### 7B, 1 x RTX 4090, Varied Prompt, 64 Max Tokens

This smaller sweep prepends a unique request id to each prompt to reduce prefix-cache reuse.

| Prompt | GPU Count | Concurrency | Requests | Avg TTFT | Avg Latency | Avg TPOT | Output tok/s | Failure Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | 1 | 1 | 8 | 0.0319 s | 1.0180 s | 0.0154 s | 62.85 | 0.0 |
| medium | 1 | 4 | 8 | 0.0548 s | 1.0634 s | 0.0158 s | 238.79 | 0.0 |
| long | 1 | 1 | 8 | 0.0330 s | 1.0198 s | 0.0154 s | 62.74 | 0.0 |
| long | 1 | 4 | 8 | 0.0559 s | 1.0642 s | 0.0158 s | 238.66 | 0.0 |

#### Salted Varied Synthetic Prefill Sweep

This sweep uses synthetic prompt buckets, `--vary-prompts`, and a per-bucket `--prompt-salt`. The salt matters because request-id-only variation can still allow cross-experiment prefix-cache reuse when two prompt buckets share the same long synthetic prefix.

| TP Size | Prompt Bucket | Avg Prompt Tokens | Requests | Avg TTFT | P95 TTFT | Avg Latency | P95 Latency | Avg TPOT |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | synthetic_256 | 575.5 | 16 | 0.0711 s | 0.0757 s | 0.3047 s | 0.3056 s | 0.01460 s |
| 1 | synthetic_512 | 1087.5 | 16 | 0.1189 s | 0.1199 s | 0.3540 s | 0.3551 s | 0.01469 s |
| 1 | synthetic_768 | 1599.5 | 16 | 0.1676 s | 0.1697 s | 0.4029 s | 0.4088 s | 0.01471 s |
| 1 | synthetic_896 | 1855.5 | 16 | 0.1926 s | 0.1938 s | 0.4273 s | 0.4286 s | 0.01466 s |
| 1 | synthetic_960 | 1983.5 | 16 | 0.2013 s | 0.2031 s | 0.4357 s | 0.4370 s | 0.01465 s |
| 2 | synthetic_256 | 575.5 | 16 | 0.0837 s | 0.1087 s | 0.2169 s | 0.2416 s | 0.00832 s |
| 2 | synthetic_512 | 1087.5 | 16 | 0.1301 s | 0.1315 s | 0.2630 s | 0.2637 s | 0.00831 s |
| 2 | synthetic_768 | 1599.5 | 16 | 0.2049 s | 0.2056 s | 0.3378 s | 0.3385 s | 0.00831 s |
| 2 | synthetic_896 | 1855.5 | 16 | 0.2124 s | 0.2137 s | 0.3452 s | 0.3461 s | 0.00830 s |
| 2 | synthetic_960 | 1983.5 | 16 | 0.2242 s | 0.2256 s | 0.3570 s | 0.3585 s | 0.00830 s |

Takeaway: TP=2 improves decode-token time, so end-to-end latency is lower even for these short 16-token outputs. But TTFT is higher for TP=2 at the same prompt length, which is consistent with tensor-parallel communication overhead showing up during prefill.

#### Decode Output-Length Sweep

| TP Size | Max Output Tokens | Avg Prompt Tokens | Requests | Avg TTFT | Avg Latency | Avg TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 93.5 | 16 | 0.0323 s | 0.2671 s | 0.01467 s |
| 1 | 64 | 93.5 | 16 | 0.0311 s | 1.0181 s | 0.01542 s |
| 1 | 128 | 93.5 | 16 | 0.0321 s | 2.0220 s | 0.01555 s |
| 1 | 256 | 93.5 | 16 | 0.0333 s | 4.0302 s | 0.01561 s |
| 2 | 16 | 88.0 | 16 | 0.0242 s | 0.1580 s | 0.00836 s |
| 2 | 64 | 88.0 | 16 | 0.0237 s | 0.5846 s | 0.00876 s |
| 2 | 128 | 88.0 | 16 | 0.0253 s | 1.1560 s | 0.00883 s |
| 2 | 256 | 93.5 | 16 | 0.0314 s | 2.2999 s | 0.00886 s |

Takeaway: decode latency scales almost linearly with generated tokens. TP=2 cuts TPOT from about 15.5 ms/token to about 8.8 ms/token on this dual-4090 setup.

#### Scheduler Stress At Concurrency 32 And 64

| TP Size | Concurrency | Requests | Avg TTFT | P95 TTFT | P99 TTFT | Avg Latency | P95 Latency | P99 Latency | Avg TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 64 | 0.1085 s | 0.1400 s | 0.1477 s | 0.6995 s | 0.7333 s | 0.7383 s | 0.01738 s |
| 2 | 32 | 64 | 0.1303 s | 0.1795 s | 0.1917 s | 0.6445 s | 0.7270 s | 0.7372 s | 0.01512 s |
| 1 | 64 | 128 | 0.1560 s | 0.2387 s | 0.2521 s | 0.7902 s | 0.8558 s | 0.8805 s | 0.01865 s |
| 2 | 64 | 128 | 0.1546 s | 0.2092 s | 0.2121 s | 0.7125 s | 0.7926 s | 0.8220 s | 0.01641 s |

Takeaway: increasing concurrency from 32 to 64 raises tail TTFT and latency. TP=2 keeps lower decode cost, but scheduler evaluation should look at p95/p99, not only average throughput.

#### Context-Length Boundary

The server was launched with `--max-model-len 2048`. With the unsalted `synthetic_960` prompt, vLLM reported 1968 prompt tokens.

| TP Size | Prompt Tokens | Requested Output Tokens | Total Budget | Result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1968 | 80 | 2048 | success |
| 1 | 1968 | 81 | 2049 | rejected |
| 2 | 1968 | 80 | 2048 | success |
| 2 | 1968 | 81 | 2049 | rejected |

The vLLM log for the rejected request says the model maximum context length is 2048 tokens, but the request asked for 81 output tokens with at least 1968 input tokens, for a total of at least 2049 tokens.

#### KV Cache Pressure Sweep

This sweep uses salted varied prompts to reduce prefix-cache reuse. It increases prompt length and concurrency while keeping total requested sequence length within `max_model_len=2048`.

| TP Size | Prompt Bucket | Max Output Tokens | Concurrency | Avg Prompt Tokens | Avg TTFT | P95 TTFT | P99 TTFT | Avg Latency | P95 Latency | Avg TPOT | Failure Rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | synthetic_512 | 64 | 1 | 1084.8 | 0.1202 s | 0.1218 s | 0.1219 s | 1.1103 s | 1.1116 s | 0.01547 s | 0.0 |
| 1 | synthetic_512 | 64 | 8 | 1084.8 | 0.5565 s | 0.9510 s | 0.9686 s | 1.8385 s | 2.0142 s | 0.02003 s | 0.0 |
| 1 | synthetic_512 | 64 | 32 | 1085.8 | 1.6819 s | 3.0360 s | 3.1033 s | 4.2016 s | 4.3094 s | 0.03937 s | 0.0 |
| 1 | synthetic_768 | 64 | 1 | 1596.8 | 0.1690 s | 0.1701 s | 0.1704 s | 1.1602 s | 1.1612 s | 0.01549 s | 0.0 |
| 1 | synthetic_768 | 64 | 8 | 1596.8 | 0.5932 s | 1.0045 s | 1.1583 s | 2.1725 s | 2.4258 s | 0.02468 s | 0.0 |
| 1 | synthetic_768 | 64 | 32 | 1597.8 | 2.5353 s | 4.6225 s | 4.7233 s | 5.7977 s | 5.9669 s | 0.05097 s | 0.0 |
| 1 | synthetic_896 | 64 | 32 | 1853.8 | 2.9227 s | 5.2540 s | 5.4755 s | 6.5733 s | 6.7704 s | 0.05709 s | 0.0 |
| 1 | synthetic_960 | 32 | 32 | 1981.8 | 3.1177 s | 5.6503 s | 5.8997 s | 6.2643 s | 6.4831 s | 0.09833 s | 0.0 |
| 2 | synthetic_512 | 64 | 1 | 1084.8 | 0.1309 s | 0.1320 s | 0.1332 s | 0.6916 s | 0.6931 s | 0.00876 s | 0.0 |
| 2 | synthetic_512 | 64 | 8 | 1084.8 | 0.5470 s | 0.8330 s | 0.9019 s | 1.5078 s | 1.6584 s | 0.01501 s | 0.0 |
| 2 | synthetic_512 | 64 | 32 | 1085.8 | 1.9088 s | 3.4446 s | 3.4636 s | 4.2680 s | 4.3297 s | 0.03686 s | 0.0 |
| 2 | synthetic_768 | 64 | 1 | 1596.8 | 0.2076 s | 0.2126 s | 0.2229 s | 0.7697 s | 0.7741 s | 0.00878 s | 0.0 |
| 2 | synthetic_768 | 64 | 8 | 1596.8 | 0.7433 s | 1.1353 s | 1.2987 s | 1.9421 s | 2.1499 s | 0.01873 s | 0.0 |
| 2 | synthetic_768 | 64 | 32 | 1597.8 | 2.8128 s | 5.1039 s | 5.2077 s | 5.9816 s | 6.0889 s | 0.04951 s | 0.0 |
| 2 | synthetic_896 | 64 | 32 | 1853.8 | 3.2328 s | 5.7754 s | 6.0111 s | 6.8178 s | 6.9434 s | 0.05602 s | 0.0 |
| 2 | synthetic_960 | 32 | 32 | 1981.8 | 3.4463 s | 6.2118 s | 6.4436 s | 6.7354 s | 6.8730 s | 0.10278 s | 0.0 |

Takeaway: all requests succeeded, but high concurrency with long sequence budgets turned TTFT from sub-second into multi-second tail latency. This is the scheduler and KV-cache capacity pressure that a cache-system optimization should target.

#### Prefix Cache Contrast

This sweep compares the same synthetic_768 prompt under repeated prompts, request-id varied prompts, and salted varied prompts.

| TP Size | Mode | Avg Prompt Tokens | Avg TTFT | P95 TTFT | Avg Latency | Avg TPOT |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | repeated | 1584.0 | 0.0416 s | 0.0437 s | 0.2764 s | 0.01467 s |
| 1 | varied | 1589.5 | 0.1694 s | 0.1706 s | 0.4041 s | 0.01467 s |
| 1 | salted varied | 1596.5 | 0.1699 s | 0.1729 s | 0.4047 s | 0.01467 s |
| 2 | repeated | 1584.0 | 0.0356 s | 0.0364 s | 0.1683 s | 0.00829 s |
| 2 | varied | 1589.5 | 0.2062 s | 0.2077 s | 0.3393 s | 0.00832 s |
| 2 | salted varied | 1596.5 | 0.2064 s | 0.2080 s | 0.3393 s | 0.00831 s |

Takeaway: repeated prompts make TTFT look much better because prefix cache reuses prompt work. That is valuable for prefix-heavy products, but it should not be confused with uncached prefill performance.

#### Metrics-Backed Prefix Cache Hit Ratio

This sweep wraps each prefix-cache benchmark with `/metrics` snapshots and computes counter deltas from:

- `vllm:prefix_cache_queries_total`
- `vllm:prefix_cache_hits_total`
- `vllm:kv_cache_usage_perc`
- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:num_preemptions_total`

| TP Size | Mode | Prefix Cache Queries | Prefix Cache Hits | Hit Ratio | Avg TTFT | P95 TTFT | Avg Latency | Avg TPOT |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | repeated | 53,856 | 53,312 | 98.99% | 0.0416 s | 0.0427 s | 0.2762 s | 0.01466 s |
| 1 | varied | 54,050 | 28,768 | 53.22% | 0.1047 s | 0.1703 s | 0.3393 s | 0.01466 s |
| 1 | salted varied | 54,356 | 1,072 | 1.97% | 0.1678 s | 0.1696 s | 0.4024 s | 0.01466 s |
| 2 | repeated | 53,856 | 51,744 | 96.08% | 0.0367 s | 0.0387 s | 0.1692 s | 0.00828 s |
| 2 | varied | 54,050 | 544 | 1.01% | 0.2068 s | 0.2079 s | 0.3396 s | 0.00830 s |
| 2 | salted varied | 54,356 | 1,072 | 1.97% | 0.2064 s | 0.2081 s | 0.3392 s | 0.00830 s |

Takeaway: prefix-cache metrics confirm the latency story. Repeated prompts had very high hit ratios and very low TTFT. Salted varied prompts had near-zero hit ratios and exposed uncached prefill cost. TP=1 varied prompts still had a partial hit ratio, which shows why the server-side metrics are useful: prompt formatting alone does not prove whether a run is cached or uncached.

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
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm4_req64_c16_tail.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_short_64_warm4_req64_c16_tail.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_medium_64_warm2_req8_c1_usage_varied.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_medium_64_warm2_req8_c4_usage_varied.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_long_64_warm2_req8_c1_usage_varied.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_long_64_warm2_req8_c4_usage_varied.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_256_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_512_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_768_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_896_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_960_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_256_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_512_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_768_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_896_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_960_out16_warm2_req16_c1_salted_varied_prefill.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_long_out16_warm2_req16_c1_decode.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_long_out64_warm2_req16_c1_decode.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_long_out128_warm2_req16_c1_decode.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_long_out256_warm2_req16_c1_decode.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_long_out16_warm2_req16_c1_decode.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_long_out64_warm2_req16_c1_decode.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_long_out128_warm2_req16_c1_decode.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_long_out256_warm2_req16_c1_decode.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_short_64_warm4_req128_c64_tail.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_short_64_warm4_req128_c64_tail.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_960_out80_req4_c1_context_boundary.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_960_out81_req4_c1_context_boundary.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_960_out80_req4_c1_context_boundary.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_960_out81_req4_c1_context_boundary.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_768_out16_warm2_req32_c1_metrics_prefix_repeated.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_768_out16_warm2_req32_c1_metrics_prefix_varied.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp1_synthetic_768_out16_warm2_req32_c1_metrics_prefix_salted.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_768_out16_warm2_req32_c1_metrics_prefix_repeated.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_768_out16_warm2_req32_c1_metrics_prefix_varied.csv`
- `projects/llm-inference-benchmark-lab/results/vllm_qwen25_7b_tp2_synthetic_768_out16_warm2_req32_c1_metrics_prefix_salted.csv`
- `projects/llm-inference-benchmark-lab/results/metrics/vllm_qwen25_7b_tp1_prefix_metrics_repeated_delta.csv`
- `projects/llm-inference-benchmark-lab/results/metrics/vllm_qwen25_7b_tp1_prefix_metrics_varied_delta.csv`
- `projects/llm-inference-benchmark-lab/results/metrics/vllm_qwen25_7b_tp1_prefix_metrics_salted_delta.csv`
- `projects/llm-inference-benchmark-lab/results/metrics/vllm_qwen25_7b_tp2_prefix_metrics_repeated_delta.csv`
- `projects/llm-inference-benchmark-lab/results/metrics/vllm_qwen25_7b_tp2_prefix_metrics_varied_delta.csv`
- `projects/llm-inference-benchmark-lab/results/metrics/vllm_qwen25_7b_tp2_prefix_metrics_salted_delta.csv`

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
- The benchmark client now also records server-reported `prompt_tokens`, which makes it possible to separate prefill-sensitive input length from decode-sensitive output length.
- The medium and long prompts used here are still short in token terms. Their TTFT stayed close to the short-prompt run; the larger end-to-end latency came mostly from generating 64 output tokens instead of about 34 output tokens in the short prompt.
- Repeated prompt benchmarks can show optimistic prefill behavior because prefix cache may reuse shared prompt prefixes. The varied prompt mode reduces that effect by making request prefixes different.
- Request-id-only varied prompts were still not enough for a clean synthetic prefill sweep because different prompt buckets can share long prefixes across experiments. The client now supports `--prompt-salt` so each bucket can break cross-experiment prefix-cache reuse.
- `nvidia-smi topo -m` reports `SYS` between GPU0 and GPU1, so tensor parallel communication crosses PCIe and the CPU interconnect rather than NVLink.
- TP=2 still improved short-prompt decode-heavy throughput by about 1.5x to 1.7x across this small concurrency sweep. At concurrency 16, throughput rose from about 710 tok/s to about 1099 tok/s and TTFT dropped from about 118 ms to about 72 ms.
- TP=2 used about 20.7 GiB on each GPU. Tensor parallelism splits model compute, but serving memory also includes KV cache blocks, CUDA graphs, runtime buffers, and per-rank overhead; do not expect visible `nvidia-smi` memory to halve.
- TP=2 was launched with `--disable-custom-all-reduce` for stability on this PCIe/SYS topology.
- With 64 measured requests at concurrency 16, TP=2 also improved tail latency: p99 latency dropped from about 765 ms to about 506 ms, and p99 TTFT dropped from about 213 ms to about 114 ms.
- In the salted synthetic prefill sweep, TP=1 TTFT rose from about 71 ms at 575 prompt tokens to about 201 ms at 1984 prompt tokens. This shows prefill cost growing with input length.
- In the same salted prefill sweep, TP=2 had better TPOT but higher TTFT than TP=1 at matching prompt lengths. This is a useful example of tensor parallelism helping decode while adding communication overhead to prefill.
- In the output-length sweep, decode latency scaled almost linearly with generated tokens. TP=1 stayed near 15 ms/token, while TP=2 stayed near 8.8 ms/token.
- At concurrency 64, p99 TTFT reached about 252 ms for TP=1 and 212 ms for TP=2. Scheduler work should track these tail metrics because averages can hide queueing behavior.
- With `max_model_len=2048`, a 1968-token prompt plus 80 requested output tokens succeeded, while 81 requested output tokens was rejected. vLLM admission checks the requested sequence budget, not just the prompt length.
- In the KV pressure sweep, all requests still succeeded, but long prompts at concurrency 32 produced multi-second tail TTFT. This means the system had enough capacity to admit the requests, but scheduling and cache pressure made first-token latency much worse.
- For synthetic_768, repeated prompts reduced TTFT from about 169 ms to 42 ms on TP=1, and from about 206 ms to 36 ms on TP=2. Prefix cache can be a large win, but only for traffic with shared prefixes.
- Prefix-cache benchmarks and clean prefill benchmarks answer different questions. Prefix-cache tests measure reuse; salted varied prompt tests measure uncached prompt processing.
- vLLM `/metrics` exposes prefix-cache counters, so hit ratio can be measured directly instead of inferred from TTFT. In the metrics-backed run, repeated prompts had about 96-99% hit ratio, while salted varied prompts were about 2%.
- TP=1 varied prompts had a partial prefix-cache hit ratio of about 53%, while TP=2 varied prompts were about 1% in the later restarted service. This is a reminder to trust server-side counters over assumptions about prompt text.

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
- For synthetic prompt sweeps, varied request ids alone may not be enough. If prompt buckets share the same long prefix across runs, prefix cache can still leak across experiments; adding a per-run salt makes the benchmark cleaner.
- `prompt_tokens + requested max_tokens` must fit within the served `max_model_len`. The server can reject the request even before generation starts.
- Prefill and decode can move in opposite directions under tensor parallelism: TP=2 improved TPOT, but salted long-prompt TTFT was higher than TP=1 on this PCIe/SYS topology.
- Decode-heavy latency is mostly linear in generated token count when the prompt is fixed, so TPOT is a good compact way to compare decode performance.
- KV pressure can show up as high TTFT and tail latency before it shows up as outright failure. A request can be accepted but still wait a long time for scheduling or cache capacity.
- Prefix cache should be evaluated as its own feature. It can dramatically reduce TTFT for repeated/shared prompts, but it should not be used accidentally when trying to measure raw prefill cost.
- Tensor parallelism is not "free multi-GPU speedup." Each layer introduces cross-GPU communication, so the benefit depends on compute saved versus communication overhead. On this short-output 7B run, TP=2 helped, but the speedup was below 2x.
- GPU memory under vLLM includes allocated KV cache and execution buffers, so apparent memory usage can remain high on every GPU even when weights are sharded.
- Scheduler-related experiments should report tail percentiles, not only averages. p95/p99 reveal queueing and batch-admission effects that averages can hide.

## Interview Answer

```text
I started by running vLLM in Docker on a dual RTX 4090 server and used a cached Qwen2.5-0.5B snapshot for a smoke test. The container could see both GPUs, but loading by Hugging Face model id failed because the container could not retrieve the file list from Hugging Face. I fixed the setup by mounting the host Hugging Face cache into the container and starting vLLM from the local snapshot path with HF_HUB_OFFLINE=1. The first request succeeded, but the TTFT was high because the logs showed a Triton JIT compilation during inference. After adding two warmup requests, steady-state TTFT dropped from about 0.91s to about 0.018s, which is why benchmark methodology has to separate cold start from steady-state performance.

For the 7B baseline, I copied Qwen2.5-7B-Instruct to the server and served it from a local directory in vLLM Docker with offline mode. On one RTX 4090, the service used about 19.7 GiB of GPU memory. With two warmup requests and measured concurrency 1, 2, 4, 8, and 16, output throughput increased from about 62 to 118 to 229 to 444 to 710 tokens/s, while TTFT rose from about 32 ms to 118 ms. I fixed the benchmark client so concurrent throughput uses measured wall-clock time excluding warmup, and token throughput uses vLLM's streaming completion token usage instead of whitespace estimation. I also added a varied-prompt mode because repeated prompts can hit prefix cache and hide prefill cost.

Then I repeated the same short-prompt benchmark with tensor-parallel size 2 across both RTX 4090s. The GPU topology was `SYS`, so the GPUs communicate over PCIe/CPU interconnect rather than NVLink. Even with that communication cost, TP=2 improved throughput from about 62 to 108 tok/s at concurrency 1 and from about 710 to 1099 tok/s at concurrency 16. The speedup was meaningful but below 2x, which is expected because tensor parallelism adds all-reduce communication and serving overhead. TP=2 also used about 20.7 GiB on each GPU, reminding me that serving memory is not just model weights; KV cache, CUDA graphs, and runtime buffers matter too.

I then added p50/p95/p99 latency and TTFT to the benchmark client and reran a higher-sample concurrency-16 comparison with 64 measured requests. TP=2 improved not just average latency but also tail latency: p99 latency dropped from about 765 ms to 506 ms, and p99 TTFT dropped from about 213 ms to 114 ms. This matters for scheduler optimization because queueing and batch admission problems usually show up in tail metrics before they show up in averages.

Next I split prefill and decode more explicitly. I added prompt-token accounting, synthetic prompt buckets, and a prompt salt to avoid cross-experiment prefix-cache leakage. In the salted prefill sweep, TP=1 TTFT rose from about 71 ms at 575 prompt tokens to about 201 ms at 1984 prompt tokens. TP=2 had lower TPOT, around 8.3 ms/token instead of 14.6 ms/token, but its long-prompt TTFT was higher than TP=1, which is a good example of tensor parallelism helping decode while adding communication cost during prefill. I also swept output length from 16 to 256 tokens and saw decode latency scale almost linearly with generated tokens.

Finally I tested the context-length admission boundary. With `max_model_len=2048`, a 1968-token prompt plus 80 requested output tokens succeeded, while 81 requested output tokens was rejected because the requested sequence budget became 2049. That taught me to reason about `prompt_tokens + max_new_tokens`, not just prompt length or actual generated length.
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
- Which part of p99 TTFT comes from queueing versus prefill execution?

## Next Step

- Add truly longer synthetic prompts to expose prefill cost.
- Run TP=1 vs TP=2 with longer prompts and longer outputs.
- Add concurrency 32 for the short-prompt sweep.
- Add request phase instrumentation: waiting time, prefill time, decode time, and KV cache block usage.

## Resume / Interview Sentence

```text
I built a small LLM inference benchmark lab to measure vLLM serving behavior on dual RTX 4090 GPUs, tracking TTFT, TPOT, throughput, memory, and failure rate across concurrency and sequence-length sweeps.
```
