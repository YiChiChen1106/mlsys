# SGLang

SGLang is a high-performance serving framework for large language models and multimodal models. It is also historically a system for executing structured language model programs.

Beginner mental model:

```text
SGLang = serving runtime + structured/programmatic generation system
```

It can be used like a normal OpenAI-compatible inference server, but its design is especially interesting for workloads with repeated prefixes, multi-step generation, structured outputs, and agent-style workflows.

## Why It Exists

Many real LLM applications are not just one prompt and one answer.

Examples:

- agent workflows with tool instructions,
- retrieval-augmented generation,
- multi-turn chat,
- few-shot prompting,
- self-consistency or branch-and-merge reasoning,
- JSON or schema-constrained output,
- multimodal workflows.

These workloads often have shared prompt prefixes and multiple generation calls. If the system recomputes every shared prefix from scratch, it wastes both compute and KV cache memory.

SGLang tries to exploit this structure.

## Main Pieces

Modern SGLang has two useful mental layers:

```text
Frontend / API layer
Runtime / serving layer
```

### Frontend / API Layer

SGLang supports OpenAI-compatible APIs, so it can behave like a normal model server.

It also has a native/programmatic interface for expressing structured language model programs. The original paper describes primitives for generation and parallelism control, such as generation calls and fork/join-style workflows.

For this project, the first baseline should use the OpenAI-compatible API because that lets us reuse the existing benchmark client.

### Runtime / Serving Layer

The runtime is the performance-critical part.

It handles:

- request scheduling,
- continuous batching,
- prefill and decode execution,
- KV cache reuse,
- RadixAttention,
- structured output decoding,
- distributed serving,
- metrics and observability.

## Core Feature: RadixAttention

RadixAttention is the SGLang concept you should remember first.

Beginner mental model:

```text
many requests share prefixes
-> store reusable KV cache in a radix tree
-> later requests search matching prefixes
-> reuse matched KV cache instead of recomputing it
```

Example:

```text
Request A: system prompt + tool instructions + user question A
Request B: system prompt + tool instructions + user question B
```

The system prompt and tool instructions are shared. SGLang can reuse their KV cache if the prefix is already stored.

## Radix Tree Intuition

A radix tree is a compressed prefix tree.

For SGLang, the tree stores token prefixes and their corresponding KV cache.

The runtime can do:

- prefix search,
- prefix insertion,
- cache reuse,
- cache eviction.

The important interview point:

```text
RadixAttention is about automatic KV cache reuse across requests and generation calls with shared prefixes.
```

## Cache-Aware Scheduling

If two waiting requests share a cached prefix, running them in a cache-friendly order can improve hit rate.

So SGLang's cache system is connected to scheduling:

```text
better prefix reuse
-> less repeated prefill
-> lower TTFT
-> higher throughput
```

This is why SGLang is a good framework to study for scheduler and cache-system roles.

## Structured Outputs

SGLang is also strong in structured generation.

Structured output means the model must produce text that follows a format, such as:

- JSON schema,
- regular expression,
- EBNF grammar,
- tool-call format.

The original SGLang paper discusses compressed finite state machines for faster constrained decoding. The beginner idea is:

```text
constrained decoding:
only allow tokens that keep the output valid
```

This is useful for production systems because broken JSON or invalid tool arguments can break downstream pipelines.

## Runtime Features To Know

SGLang documentation and repository describe features such as:

- OpenAI-compatible API,
- RadixAttention,
- continuous batching,
- prefix caching,
- multi-GPU parallelism,
- tensor / pipeline / expert / data parallelism,
- prefill-decode disaggregation,
- speculative decoding,
- chunked prefill,
- structured outputs,
- quantization,
- multi-LoRA batching,
- observability and metrics.

The exact feature set can change by version, so always check the docs before running experiments.

## SGLang vs vLLM

Both are high-performance inference frameworks and overlap heavily.

A useful beginner distinction:

```text
vLLM:
general-purpose high-throughput serving engine

SGLang:
serving runtime with strong structured-generation and prefix-reuse focus
```

PagedAttention vs RadixAttention:

```text
PagedAttention:
How do I store many variable-length KV caches efficiently?

RadixAttention:
How do I reuse KV cache across requests with shared prefixes?
```

This distinction is not absolute. vLLM also has prefix caching, and SGLang also has paged attention and general serving features. The point is to understand their design emphasis.

## Why It Matters For This Job Track

The target role cares about:

- scheduler performance,
- cache system performance,
- decode latency,
- distributed parallelism,
- profiling and metrics.

SGLang connects to all of these:

- RadixAttention and prefix cache are cache-system topics.
- Continuous batching and cache-aware scheduling are scheduler topics.
- Structured outputs affect decode behavior.
- TP/PP/EP/DP and prefill-decode disaggregation connect to parallelism.
- Metrics and tracing connect to profiling.

## How To Benchmark It In This Project

Use the same baseline discipline as vLLM:

1. Same server: `pink`.
2. Same model path: `/mnt/hdd/users/cychi/hf_models/Qwen2.5-7B-Instruct`.
3. Same benchmark client.
4. Same prompt/output/concurrency matrix where possible.
5. Compare repeated-prefix workloads and salted varied workloads separately.
6. Collect server metrics if available.

First SGLang benchmark questions:

- Can it serve the same local model through OpenAI-compatible API?
- What is TTFT/TPOT on short prompt baseline?
- How does it behave on repeated-prefix prompts?
- How does it behave on salted varied prompts?
- What metrics expose cache hit rate, waiting, or scheduler pressure?

## Chinese Interview Sentence

```text
SGLang 是一个高性能 LLM serving framework，也可以理解成面向结构化/程序化 LLM workflow 的推理系统。它既能像普通 OpenAI-compatible server 一样服务模型，也强调多步生成、结构化输出和共享前缀复用。它最核心的机制之一是 RadixAttention：把可复用的 KV cache 前缀组织到 radix tree 里，让后续有相同前缀的请求或 generation call 可以复用已有 KV cache，减少重复 prefill。

和 vLLM 对比，我会说 vLLM 的 PagedAttention 更偏 KV cache 的存储管理，解决很多变长请求的 KV block 怎么高效放；SGLang 的 RadixAttention 更偏 prefix reuse，解决复杂 workflow 或 prefix-heavy traffic 里相同前缀怎么自动复用。当然两个框架现在功能有很多重叠，所以我不会只看宣传点，而会在同一台机器、同一模型、同一 workload 下比较 TTFT、TPOT、prefix cache、scheduler metrics 和 tail latency。
```

## Sources

- SGLang docs: https://docs.sglang.ai/
- SGLang introduction: https://sgl-project-sglang-93.mintlify.app/introduction
- SGLang GitHub: https://github.com/sgl-project/sglang
- SGLang paper: https://arxiv.org/html/2312.07104v2
