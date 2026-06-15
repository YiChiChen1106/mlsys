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
