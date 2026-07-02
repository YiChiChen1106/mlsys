# Experiment Plan: SGLang Baseline On Pink

## Goal

Run a first SGLang baseline on `pink` and compare it against the existing vLLM baseline using the same model, similar serving settings, and the same benchmark client whenever possible.

The main question is not "which framework is always faster". The main question is:

```text
How does SGLang behave on the same hardware and workload dimensions:
TTFT, TPOT, throughput, prefix-heavy traffic, long prompts, and high concurrency?
```

## Why SGLang Next

The vLLM baseline already covers:

- OpenAI-compatible serving,
- TTFT / TPOT / latency / throughput,
- tensor parallel comparison,
- prefix cache behavior,
- KV pressure,
- scheduler parameter tradeoffs,
- server-side metrics.

SGLang is the natural next framework because it is also a high-performance serving runtime, but it is especially interesting for:

- structured generation,
- programmatic LLM workflows,
- prefix-heavy traffic,
- RadixAttention-style prefix reuse.

## Environment

- Machine: `pink`
- GPU: 2 x RTX 4090, 24 GB each
- Model path on `pink`: `/mnt/hdd/users/cychi/hf_models/Qwen2.5-7B-Instruct`
- Preferred access: `ssh pink`
- Fallback access: `ssh pink-outer`
- Framework: SGLang
- Intended Docker image: `lmsysorg/sglang:latest`
- SGLang server port: `30000`

Use local model files and offline Hugging Face behavior when possible, following the vLLM setup lesson:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

## Source Docs Checked

- SGLang docs: https://docs.sglang.ai/
- SGLang Docker docs: https://docs.sglang.ai/developer_guide/development_guide_using_docker.html
- SGLang metrics docs: https://docs.sglang.ai/advanced_features/observability.html

## Phase 0: Safety Check

Before starting SGLang, verify that no previous serving process is occupying the shared GPUs:

```bash
ssh pink
nvidia-smi
docker ps
```

If old vLLM/SGLang containers are running from this project, stop them before starting the next server.

## Phase 1: SGLang Smoke Test

Start with one GPU and a small request count.

Draft Docker command:

```bash
docker run -d \
  --name sglang-qwen25-7b-tp1 \
  --gpus '"device=0"' \
  --ipc=host \
  --network=host \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /mnt/hdd/users/cychi/hf_models:/models \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path /models/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-metrics
```

Smoke request:

```bash
curl http://127.0.0.1:30000/v1/models
```

Then run one tiny benchmark through the existing OpenAI-compatible benchmark client by pointing it to SGLang's base URL if the client supports overriding the base URL.

If not, update `benchmark_client.py` to accept a `--base-url` argument before running the SGLang matrix.

## Phase 2: Minimal Comparable Matrix

Keep the first SGLang run small.

| Case | GPU | Prompt | Output | Concurrency | Requests | Purpose |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| smoke | 1 | short | 64 | 1 | 1 | API and generation check |
| baseline | 1 | short | 64 | 1, 4, 16 | 16 | Compare basic batching behavior |
| prefill | 1 | synthetic_768 salted varied | 16 | 1 | 16 | Compare uncached prefill TTFT |
| decode | 1 | long | 16, 64, 128 | 1 | 16 | Compare decode TPOT |
| prefix | 1 | synthetic_768 repeated/salted | 16 | 1 | 32 | Compare prefix reuse behavior |

After TP=1 is stable, repeat a smaller TP=2 matrix:

| Case | GPU | Prompt | Output | Concurrency | Requests | Purpose |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| baseline | 2 | short | 64 | 1, 4, 16 | 16 | TP=2 throughput and latency |
| prefill | 2 | synthetic_768 salted varied | 16 | 1 | 16 | TP=2 prefill behavior |
| prefix | 2 | synthetic_768 repeated/salted | 16 | 1 | 32 | Prefix reuse behavior |

## Phase 3: Metrics

SGLang supports metrics through an enable flag. For this project, collect at least:

- request latency,
- time to first token if exposed,
- token throughput,
- cache-related metrics if exposed,
- waiting/running scheduler metrics if exposed.

Use the same principle as vLLM:

```text
counter metrics -> before/after delta
gauge metrics   -> time-series sampling during benchmark
```

If metric names differ from vLLM, document the mapping instead of forcing the same names.

## Phase 4: Compare Against vLLM

Do not compare only one tokens/s number. Compare by workload:

| Question | vLLM Baseline | SGLang Baseline |
| --- | --- | --- |
| Short prompt throughput | concurrency sweep | same concurrency sweep |
| Decode speed | output length sweep | same output length sweep |
| Uncached prefill | salted synthetic prompts | same salted prompts |
| Prefix reuse | repeated vs salted hit/latency | repeated vs salted hit/latency |
| High concurrency pressure | KV/time-series style run | comparable pressure run |

## Expected Interview Story

```text
After building a vLLM baseline, I designed a SGLang comparison on the same dual-4090 server and local Qwen2.5-7B model. I kept the workload dimensions the same: concurrency, prompt length, output length, repeated prefixes, salted varied prompts, and TP=1/TP=2. The goal was to compare framework behavior, not just a single throughput number. Since vLLM emphasizes PagedAttention-based KV cache management and SGLang emphasizes RadixAttention-style prefix reuse and structured generation workflows, I planned the matrix to include both uncached prefill and prefix-heavy cases.
```

## Next Step

Before running, verify:

- whether `lmsysorg/sglang:latest` is already present on `pink`,
- whether port `30000` is free,
- whether existing `benchmark_client.py` can set SGLang's base URL,
- whether SGLang model name returned by `/v1/models` needs to be passed into the benchmark client.
