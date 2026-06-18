# KV Cache

KV cache stores the attention keys and values from earlier tokens so the model does not recompute the full prompt at every decode step.

## Why It Matters

Without KV cache, each generated token would repeatedly process all previous tokens. With KV cache, decode can reuse earlier attention states and only process the newest token.

## Cost

KV cache can dominate inference memory, especially with:

- long prompts,
- long generated outputs,
- high concurrency,
- large hidden sizes,
- many layers,
- tensor parallel serving.

## Context-Length Admission

Serving systems reserve sequence budget before generation. In vLLM, a request must satisfy:

```text
prompt_tokens + requested_max_tokens <= max_model_len
```

In the `pink` vLLM baseline, the server used `--max-model-len 2048`. A 1968-token prompt with 80 requested output tokens succeeded because the total budget was 2048. The same prompt with 81 requested output tokens was rejected because the total budget became 2049.

This is related to KV cache because every accepted token position needs cache space. The server cannot assume the model will stop early; it has to admit based on the requested maximum.

## Pressure Symptoms

KV cache pressure does not have to appear first as an OOM or rejected request. In the vLLM baseline, long salted prompts at concurrency 32 all succeeded, but TTFT moved into multi-second territory:

- TP=1 synthetic_896, output 64, concurrency 32: p99 TTFT about 5.48 s.
- TP=2 synthetic_896, output 64, concurrency 32: p99 TTFT about 6.01 s.
- TP=2 synthetic_960, output 32, concurrency 32: p99 TTFT about 6.44 s.

This means the cache and scheduler still admitted the requests, but first-token latency became much worse. For cache-system optimization, this is a more useful signal than only checking whether requests fail.

## Prefix Cache

Prefix cache is different from normal KV cache reuse during decode. It reuses prompt-prefix computation across requests with shared prefixes.

In the vLLM prefix-cache contrast on synthetic_768:

- TP=1 repeated prompt TTFT: about 42 ms.
- TP=1 salted varied prompt TTFT: about 170 ms.
- TP=2 repeated prompt TTFT: about 36 ms.
- TP=2 salted varied prompt TTFT: about 206 ms.

This is a big win for traffic with shared prefixes, but it can hide raw prefill cost in a benchmark.

The hit ratio can be measured directly from vLLM metrics:

```text
delta(vllm:prefix_cache_hits_total) / delta(vllm:prefix_cache_queries_total)
```

Measured examples:

- TP=1 repeated: about 98.99% hit ratio.
- TP=1 salted varied: about 1.97% hit ratio.
- TP=2 repeated: about 96.08% hit ratio.
- TP=2 salted varied: about 1.97% hit ratio.

The TP=1 varied run had about 53.22% hit ratio, while TP=2 varied after service restart had about 1.01%. This is a reminder that server-side counters are more reliable than assumptions about prompt text.

## Inference Connection

For a serving system, KV cache is not just a tensor. It becomes a resource-management problem:

```text
requests arrive
-> scheduler accepts some requests
-> KV cache blocks are allocated
-> decode reuses blocks
-> finished requests release blocks
```

## Questions For Experiments

- How much memory is used before and after a request starts decoding?
- What happens to throughput when prompts become longer?
- What happens when concurrency increases until memory is near full?
- Does the framework expose cache hit rate or block usage metrics?
