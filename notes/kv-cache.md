# KV Cache

KV cache stores the attention keys and values from earlier tokens so the model does not recompute the full prompt at every decode step.

## Beginner Mental Model

An LLM generates text one token at a time. When it generates a new token, it needs to attend to the previous tokens.

Without KV cache, the model would repeatedly redo work for the old tokens:

```text
step 1: process prompt -> generate token 1
step 2: process prompt + token 1 again -> generate token 2
step 3: process prompt + token 1 + token 2 again -> generate token 3
```

This is wasteful because most of the history did not change.

KV cache avoids this repeated work:

```text
prefill: process prompt once and store K/V
decode step 1: read old K/V, compute only the new token's K/V
decode step 2: read old K/V, compute only the new token's K/V
decode step 3: read old K/V, compute only the new token's K/V
```

The simple idea is:

```text
KV cache = saved attention state for previous tokens
```

It makes decode much faster, but it consumes GPU memory.

## Why Attention Has K And V

In transformer attention, each token produces three vectors:

```text
Q = query
K = key
V = value
```

The query asks: "What previous information do I need?"

The keys help decide which previous tokens are relevant.

The values contain the information to mix into the current token representation.

During decode, the old tokens' K and V do not change. So the framework stores them and reuses them for later decode steps.

## Prefill And Decode View

KV cache is created during prefill and extended during decode:

```text
prefill(prompt tokens)
-> write K/V for all prompt tokens

decode(new token)
-> read old K/V
-> compute K/V for new token
-> append new K/V to cache
```

This is why prefill and decode are connected:

- Prefill builds the initial cache.
- Decode depends on the cache every step.
- Longer prompts create larger initial cache.
- Longer outputs keep appending to the cache.

## Why KV Cache Becomes A Systems Problem

For one request, KV cache is easy to understand. For many concurrent requests, it becomes a memory-management problem.

The framework has to answer:

- How many cache blocks can fit in GPU memory?
- Which requests can be admitted now?
- Which requests must wait?
- What happens if a request grows longer than expected?
- When a request finishes, how quickly can its cache blocks be reused?

This is why inference framework jobs care so much about cache systems and schedulers.

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

### Beginner Explanation

`max_model_len` is the maximum sequence length that the serving engine allows for one request.

For a generation request, the sequence is:

```text
input prompt tokens + generated output tokens
```

So the server has to check the total possible sequence length before starting:

```text
prompt_tokens + requested max_tokens
```

The important detail is `requested max_tokens`, not actual generated tokens.

Why? Because before generation starts, the server does not know whether the model will stop early. If the user requests up to 512 output tokens, the engine must assume the sequence may grow by 512 tokens and reserve/admit capacity accordingly.

Example from the experiment:

```text
max_model_len = 2048
prompt_tokens = 1968
```

Case 1:

```text
requested max_tokens = 80
total budget = 1968 + 80 = 2048
result = accepted
```

Case 2:

```text
requested max_tokens = 81
total budget = 1968 + 81 = 2049
result = rejected
```

Even if the model might have stopped after fewer than 81 tokens, the server cannot rely on that. Admission control uses the requested budget.

### Why It Matters For Systems

This rule protects the server from accepting requests that may exceed:

- model context length,
- KV cache capacity,
- scheduler sequence budget,
- memory planning assumptions.

It also means users can get rejected by asking for too much output even when their prompt alone fits.

Chinese interview sentence:

```text
max_model_len 是单个请求允许的最大序列长度，序列长度不是只看 prompt，而是 prompt_tokens 加上请求的 max_tokens。服务端在生成前不知道模型会不会提前停止，所以 admission control 必须按 requested max_tokens 做最坏情况预算。我的实验里 max_model_len=2048，1968-token prompt 加 80 个 requested output token 正好等于 2048，所以成功；加 81 个就变成 2049，超过限制被 vLLM 拒绝。这个规则和 KV cache 管理有关，因为每个可能的位置都需要预留上下文和 cache 容量。
```

## Pressure Symptoms

KV cache pressure does not have to appear first as an OOM or rejected request. In the vLLM baseline, long salted prompts at concurrency 32 all succeeded, but TTFT moved into multi-second territory:

- TP=1 synthetic_896, output 64, concurrency 32: p99 TTFT about 5.48 s.
- TP=2 synthetic_896, output 64, concurrency 32: p99 TTFT about 6.01 s.
- TP=2 synthetic_960, output 32, concurrency 32: p99 TTFT about 6.44 s.

This means the cache and scheduler still admitted the requests, but first-token latency became much worse. For cache-system optimization, this is a more useful signal than only checking whether requests fail.

## Time-Series Metrics

Gauge metrics such as `vllm:kv_cache_usage_perc`, `vllm:num_requests_running`, and `vllm:num_requests_waiting` should be sampled while the benchmark is running. A before/after snapshot can miss the peak because the engine returns to idle after requests finish.

In a synthetic_896/output64/concurrency32 run with 96 measured requests:

- TP=1 p99 TTFT was about 5.41 s and peak reported KV usage was about 97.06%.
- TP=2 p99 TTFT was about 5.94 s and peak reported KV usage was about 15.39%.
- Both runs reached 32 running requests and built capacity waiting queues of 27-29 requests.
- No preemptions were reported.

This suggests a useful follow-up: tune scheduler and batching limits, because waiting queues can appear even when reported KV usage is not close to 100%.

## Prefix Cache

Prefix cache is different from normal KV cache reuse during decode. It reuses prompt-prefix computation across requests with shared prefixes.

## Normal KV Cache vs Prefix Cache

These two concepts are related but not the same.

| Concept | Reuse Scope | What It Reuses | Main Benefit |
| --- | --- | --- | --- |
| Normal KV cache | Inside one request | Previous tokens in the same sequence | Faster decode |
| Prefix cache | Across requests | Shared prompt prefix computation | Lower prefill cost and TTFT |

Beginner mental model:

```text
Normal KV cache:
same request, later decode tokens reuse earlier tokens

Prefix cache:
different requests, same prompt prefix reuse already computed prefix
```

Example:

```text
Request A: "You are a helpful assistant. Summarize paper A..."
Request B: "You are a helpful assistant. Summarize paper B..."
```

Both requests share this prefix:

```text
"You are a helpful assistant. Summarize"
```

If prefix cache hits, the server can reuse the computation for the shared prefix instead of prefilling it from scratch again.

This is especially useful for:

- chat systems with the same system prompt,
- agent workflows with repeated tool instructions,
- retrieval or summarization templates,
- batch jobs with common prompt headers.

It is less useful when every request has a completely different prefix.

## Why Prefix Cache Can Mislead Benchmarks

Prefix cache is good for products that really have shared prefixes, but it can accidentally make a prefill benchmark look too good.

If a benchmark repeatedly sends the exact same prompt, TTFT may be measuring prefix-cache hits rather than raw prefill speed.

That is why clean prefill experiments should use varied or salted prompts, and prefix-cache experiments should be labeled separately.

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

Chinese interview sentence:

```text
普通 KV cache 和 prefix cache 的复用范围不一样。普通 KV cache 是同一个请求内部，在 decode 阶段复用历史 token 的 K/V，主要加速后续 token 生成。Prefix cache 是不同请求之间复用相同 prompt 前缀的计算结果，主要降低 prefill 成本和 TTFT。我的实验里 repeated prompt 的 prefix-cache hit ratio 达到 96-99%，TTFT 明显更低；而 salted varied prompt 的 hit ratio 只有约 2%，更接近 uncached prefill。因此做 benchmark 时要区分 prefix-cache 场景和纯 prefill 场景，不能把缓存命中误当成模型本身 prefill 很快。
```

## Inference Connection

For a serving system, KV cache is not just a tensor. It becomes a resource-management problem:

```text
requests arrive
-> scheduler accepts some requests
-> KV cache blocks are allocated
-> decode reuses blocks
-> finished requests release blocks
```

## PagedAttention And KV Blocks

PagedAttention is vLLM's solution to KV cache memory management for many variable-length sequences.

The important idea:

```text
sequence logical blocks
-> block table
-> physical KV cache blocks
```

The sequence can be logically continuous even if its KV cache blocks are physically scattered in GPU memory.

Example:

```text
logical block 0 -> physical block 12
logical block 1 -> physical block 5
logical block 2 -> physical block 31
```

During attention, vLLM uses the block table to fetch the correct K/V blocks.

Why it matters for cache systems:

- Variable-length requests do not need one large contiguous allocation.
- Decode can append new KV blocks as a sequence grows.
- Finished requests release their physical blocks back to the allocator.
- Memory waste is reduced because only the last block may be partially unused.
- Prefix sharing can reuse existing blocks across requests.

For scheduler work, block availability becomes an admission signal. If enough physical KV blocks are not available, new requests may wait even before outright OOM.

Chinese interview sentence:

```text
PagedAttention 把 KV cache 管理成 block/page。每个请求的序列在逻辑上是连续的，但实际 K/V 可以分散存放在不同物理 block 里，框架用 block table 做映射。这样请求不需要预留一大段连续 KV cache，而是随着 decode 增长按需分配 block；请求结束后 block 归还。它减少了变长请求带来的显存浪费和碎片，也让 scheduler 可以根据可用 KV block 做 admission 和 waiting 判断。
```

## Questions For Experiments

- How much memory is used before and after a request starts decoding?
- What happens to throughput when prompts become longer?
- What happens when concurrency increases until memory is near full?
- Does the framework expose cache hit rate or block usage metrics?
