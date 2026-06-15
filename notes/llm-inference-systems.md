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
