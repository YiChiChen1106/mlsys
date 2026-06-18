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
