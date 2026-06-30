# Serving Benchmarking

Serving benchmarks should explain the latency-throughput tradeoff of an inference system, not just produce one tokens-per-second number.

## Minimum Metrics

- TTFT: time to first token.
- TPOT: time per output token after the first token.
- End-to-end latency: full request duration.
- Output throughput: generated tokens per second.
- Request throughput: completed requests per second.
- GPU memory peak.
- Failure rate.

## Client Metrics vs Server Metrics

Client-side benchmark metrics show what the user experiences:

- TTFT,
- TPOT,
- end-to-end latency,
- p50/p95/p99 latency,
- output tokens per second,
- failure rate.

These are necessary, but they do not fully explain why the system is slow.

Server-side metrics show what the inference engine is doing internally:

- KV cache usage,
- number of running requests,
- number of waiting requests,
- waiting reason,
- preemptions,
- prefix cache hits and queries.

The beginner rule is:

```text
client metrics tell you what happened
server metrics help explain why it happened
```

For example:

```text
high TTFT
could be queueing
could be long prefill
could be waiting for KV cache capacity
could be a large batch step
```

Without server metrics, these cases can look similar from the client side.

## Counters vs Gauges

Server metrics usually include two kinds of values.

Counters only increase over time:

- prefix cache queries,
- prefix cache hits,
- total preemptions.

For counters, use before/after deltas:

```text
hit_ratio = delta(prefix_cache_hits_total) / delta(prefix_cache_queries_total)
```

Gauges are point-in-time values:

- current KV cache usage,
- current running requests,
- current waiting requests.

For gauges, before/after snapshots are not enough. The value may return to idle after the benchmark ends.

Use time-series sampling during the benchmark:

```text
sample /metrics every 0.2 seconds
record max KV usage, max running, max waiting
```

This is why the KV pressure experiment needed `metrics_timeseries.py`, not only `metrics_snapshot.py`.

## Profiling Levels

There are several levels of performance investigation:

1. Client benchmark: user-visible latency and throughput.
2. Server metrics: scheduler, KV cache, waiting queue, prefix cache, preemptions.
3. Framework logs: request admission, warnings, errors, initialization behavior.
4. PyTorch profiler / Nsight Systems: GPU kernels, CPU overhead, communication, synchronization.

For the current project, the first two levels are already useful. Nsight and `torch.profiler` are natural next steps after the benchmark stories are clear.

## Variables To Sweep

- Framework: vLLM, SGLang, llama.cpp.
- Model: start with a 7B/8B class model.
- GPU count: 1 x RTX 4090, then 2 x RTX 4090.
- Concurrency: 1, 2, 4, 8, 16.
- Prompt length: short, medium, long.
- Output length: short, medium.

## Benchmark Pitfalls From vLLM Baseline

- Warmup should be outside the measured window. Otherwise cold-start work such as Triton JIT or CUDA graph setup can pollute steady-state numbers.
- Concurrent throughput should use wall-clock time, not summed request latency. Summed latency double-counts overlapping work.
- Token throughput should use server-reported completion tokens when available. Text whitespace splitting is not tokenizer-accurate.
- Prefill experiments need server-reported prompt tokens, not just a human label such as short or long.
- Repeated prompts can make prefix cache hide prefill work.
- Request-id-only varied prompts can still leak prefix-cache state across prompt buckets if different buckets share the same long prefix. A per-run prompt salt helps break cross-experiment reuse.
- Context admission depends on `prompt_tokens + requested max_tokens`, not only actual generated tokens.
- KV pressure should be tested with long prompts and high concurrency. If all requests succeed but p95/p99 TTFT becomes seconds, the system is still under meaningful cache/scheduler pressure.
- Prefix-cache benchmarks should be labeled separately from clean prefill benchmarks. Repeated prompts answer a product/cache-reuse question; salted varied prompts answer an uncached prefill question.
- When available, collect server-side metrics before and after each run. For vLLM, `prefix_cache_hits_total`, `prefix_cache_queries_total`, `kv_cache_usage_perc`, `num_requests_waiting`, and `num_preemptions_total` turn client-side latency into a more explainable result.
- Use before/after deltas for counters such as prefix-cache hits. Use time-series sampling for gauges such as KV usage and waiting requests.

## Interview Sentence

```text
When I benchmark an LLM server, I separate TTFT, TPOT, end-to-end latency, throughput, memory, and failures. For prefill experiments I vary and salt prompts to avoid prefix-cache contamination, and for decode experiments I sweep output length because latency should scale roughly with generated tokens.
```

```text
For cache-system experiments, I do not only look for failures. I increase sequence budget and concurrency, then watch p95/p99 TTFT and latency. In my vLLM run, long prompts at concurrency 32 all succeeded, but p99 TTFT reached several seconds, which is exactly the kind of pressure signal a scheduler or KV-cache optimization should address.
```

```text
For prefix-cache experiments, I verify cache behavior with vLLM counters instead of assuming from prompt text. I take /metrics snapshots before and after the run, then compute delta(prefix_cache_hits_total) divided by delta(prefix_cache_queries_total). In my run, repeated prompts had about 96-99% hit ratio, while salted varied prompts were about 2%.
```

```text
For KV pressure experiments, I sample gauge metrics during the run rather than only before and after. In one vLLM run, p99 TTFT was around 5-6 seconds while max waiting requests reached 27-29. TP=1 also reported about 97% KV usage, while TP=2 showed waiting even with much lower reported KV usage, which points to scheduler capacity and batch-token limits as follow-up variables.
```

Chinese interview sentence:

```text
我做推理 benchmark 时会把客户端指标和服务端指标分开看。客户端指标，比如 TTFT、TPOT、p95/p99 latency 和吞吐，告诉我用户看到的性能现象；服务端 metrics，比如 KV cache usage、running/waiting requests、preemptions、prefix cache hit ratio，帮助解释为什么会慢。

对于 counter 类指标，比如 prefix_cache_hits_total 和 prefix_cache_queries_total，我会用前后差值计算 hit ratio。对于 gauge 类指标，比如 kv_cache_usage_perc 和 num_requests_waiting，前后 snapshot 不够，因为 benchmark 结束后它们会回到 idle，所以我会在压测过程中做 time-series 采样。我的 KV pressure 实验里就是通过 time-series 看到了 27-29 个 waiting requests 和 TP=1 约 97% 的 KV usage peak。
```

## Record Format

Each experiment note should include:

- Goal.
- Environment.
- Model and framework.
- Commands.
- Benchmark matrix.
- Results table.
- Interpretation.
- Next step.

Use `templates/inference-experiment.md`.
