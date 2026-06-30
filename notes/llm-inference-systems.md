# LLM Inference Systems

LLM inference systems turn a model checkpoint into a service that can answer many requests with predictable latency, high throughput, and controlled GPU memory use.

## Core Mental Model

```text
request
-> tokenizer
-> scheduler
-> prefill
-> KV cache allocation
-> decode loop
-> sampling
-> streaming response
```

The main performance tension is:

```text
low latency for one user
vs
high throughput across many users
vs
limited GPU memory
```

## Request Lifecycle, Beginner Version

An LLM inference framework is the system layer between a model checkpoint and real user traffic. A checkpoint only stores model weights. The inference framework turns those weights into an online service.

A single request usually goes through this path:

```text
HTTP request
-> tokenizer
-> request queue
-> scheduler
-> prefill
-> KV cache write
-> decode loop
-> sampling
-> detokenizer
-> streaming response
```

### 1. HTTP Request

The user or benchmark client sends a request to the model server. vLLM exposes an OpenAI-compatible API, so the request often looks like a normal `/v1/chat/completions` or `/v1/completions` call.

At this layer, the server sees fields such as:

- model name,
- prompt or messages,
- `max_tokens`,
- temperature and sampling options,
- whether streaming is enabled.

### 2. Tokenizer

The model does not directly read human text. The tokenizer converts text into token ids.

Example mental model:

```text
"KV cache is useful"
-> [token_1, token_2, token_3, ...]
```

For performance work, token count matters more than character count. A prompt that looks short to a human may still become many tokens, depending on language, formatting, and tokenizer behavior.

### 3. Request Queue

After tokenization, the request waits in a queue if the engine cannot run it immediately.

This is why TTFT is not only model compute time. TTFT can include:

- queueing delay,
- scheduler admission delay,
- prefill compute,
- first decode step,
- networking and Python overhead.

### 4. Scheduler

The scheduler decides which requests should run in the next engine step.

It has to balance several goals:

- keep the GPU busy,
- avoid running out of KV cache memory,
- avoid starving waiting requests,
- batch compatible requests together,
- control tail latency.

In an LLM server, scheduling is hard because each request has a different prompt length, output length, arrival time, and remaining decode length.

### 5. Prefill

Prefill is the phase that processes the input prompt.

If the prompt has 1,000 tokens, prefill computes the model forward pass over those prompt tokens and builds the initial KV cache. This phase is relatively parallel because the full prompt is already known.

Prefill mainly affects TTFT:

```text
longer prompt -> more prefill work -> higher TTFT
```

This is why prompt-length experiments should track server-reported `prompt_tokens`, not only labels like short, medium, or long.

### 6. KV Cache Write

During prefill, each transformer layer produces attention keys and values for the prompt tokens. The server stores these tensors in GPU memory as KV cache.

The point of KV cache is:

```text
reuse past token attention state during decode
instead of recomputing the whole history every step
```

KV cache is a memory-for-speed tradeoff. It makes decode much faster, but it consumes GPU memory proportional to active sequence count and sequence length.

### 7. Decode Loop

Decode generates new tokens one at a time.

The model is autoregressive:

```text
prompt -> token 1
prompt + token 1 -> token 2
prompt + token 1 + token 2 -> token 3
...
```

This means output length matters a lot. If one request generates 256 tokens, the server must run many decode iterations.

Decode mainly affects TPOT and total latency:

```text
longer output -> more decode steps -> higher end-to-end latency
```

### 8. Sampling

After each decode forward pass, the model produces logits. The sampling logic chooses the next token according to settings such as temperature, top-p, top-k, or greedy decoding.

For framework performance, sampling is usually not the largest compute cost compared with transformer execution, but it is part of the per-token serving loop.

### 9. Detokenizer And Streaming Response

The generated token ids are converted back into text. If streaming is enabled, the server sends partial text chunks back as tokens are generated.

Streaming improves user experience because the user sees the first token early, even if the full answer takes longer.

This is why TTFT matters so much for chat products.

## The Three Latency Questions

For every serving result, ask three separate questions:

```text
How long until the first token?        -> TTFT
How fast are later tokens generated?   -> TPOT
How long until the request is done?     -> latency
```

A useful approximate formula is:

```text
end_to_end_latency ~= TTFT + output_tokens * TPOT
```

This is not exact, but it is a good mental model for interviews and experiments.

## Key Concepts

- Prefill: processes the input prompt and builds the initial KV cache.
- Decode: generates one token at a time using the KV cache.
- KV cache: stored attention keys and values reused during decode.
- Batching: groups multiple requests to improve GPU utilization.
- Continuous batching: admits and removes requests while decoding is already running.
- Prefix caching: reuses computation for shared prompt prefixes.
- Tensor parallelism: splits model computation across GPUs.
- Quantization: reduces memory footprint and sometimes improves throughput.

## TP=1 vs TP=2 Mental Model

Tensor parallelism can improve decode throughput because each GPU handles part of the model computation. It is not a free 2x speedup because every layer may introduce cross-GPU communication.

On the dual RTX 4090 `pink` server, GPU0-to-GPU1 topology is `SYS`, not NVLink. In the vLLM 7B baseline:

- TP=2 reduced decode TPOT from about 15.5 ms/token to about 8.8 ms/token.
- TP=2 had higher long-prompt salted prefill TTFT than TP=1 at the same prompt length.
- The useful interview framing is: TP can help decode-heavy workloads, but prefill and scheduler behavior may pay communication overhead.

## Prefill vs Decode

- Prefill processes the input prompt and builds KV cache. It mostly shows up in TTFT.
- Decode generates one token at a time. It mostly shows up in TPOT and output-length scaling.
- A fixed prompt with output length 16, 64, 128, and 256 should have roughly linear latency growth during decode.
- A fixed output length with prompt length increasing from about 575 to 1984 prompt tokens should show TTFT growth.

## First Frameworks To Study

1. vLLM: first baseline for serving and benchmarking.
2. SGLang: second framework for prefix-heavy and structured-generation workloads.
3. llama.cpp: useful comparison point for local and quantized inference.
4. TensorRT-LLM: later, after the serving and scheduling concepts are clearer.

## Questions To Answer

- How does throughput change as concurrency increases?
- How does TTFT change with prompt length?
- How does TPOT change with output length and concurrency?
- When does GPU memory, not compute, become the bottleneck?
- How much does two-GPU tensor parallelism help on 2 x RTX 4090?
- Which framework behavior can be explained by scheduling or KV cache management?

## Current Project

See `projects/llm-inference-benchmark-lab/README.md`.

The first milestone is a vLLM baseline on `pink`.
