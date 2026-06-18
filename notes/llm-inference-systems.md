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
