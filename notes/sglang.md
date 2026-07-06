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

## RadixAttention, Slightly Deeper

RadixAttention maintains a radix tree whose edges represent token sequences and whose nodes point to reusable KV cache.

When a new request arrives, the runtime can:

1. Search the radix tree for the longest cached prefix.
2. Reuse the KV cache for the matched prefix.
3. Prefill only the unmatched suffix.
4. Insert the newly computed suffix back into the tree.

Beginner flow:

```text
new request tokens
-> longest prefix match in radix tree
-> reuse matched prefix KV
-> compute missing suffix
-> insert suffix KV into tree
```

### Example

Suppose the cache already contains:

```text
"system prompt + tool instruction + summarize"
```

A new request arrives:

```text
"system prompt + tool instruction + summarize document B"
```

The runtime can match:

```text
"system prompt + tool instruction + summarize"
```

Then it only needs to prefill:

```text
"document B"
```

This reduces repeated prefill work and can lower TTFT.

### Node Split Intuition

If a new sequence partially overlaps an existing cached sequence, the radix tree may split an edge.

Example:

```text
existing: A B C D
new:      A B X Y
```

Shared prefix:

```text
A B
```

The tree can split into:

```text
A B
├── C D
└── X Y
```

This is how the tree represents shared prefixes without storing every full prompt as a separate independent cache entry.

### Eviction Intuition

GPU memory is limited, so prefix cache cannot grow forever.

SGLang's documentation describes cache eviction with an LRU-style policy on leaf nodes and reference counts for running batches.

Beginner interpretation:

- Prefer evicting cache entries that are old and not actively used.
- Do not evict KV cache that a running request still depends on.
- Free cache entries when memory is needed for new requests.

This connects prefix cache to cache-system engineering:

```text
hit rate
vs
GPU memory capacity
vs
eviction safety
vs
scheduler decisions
```

### Why Cache-Aware Scheduling Matters

If two waiting requests have different prefix-cache opportunities, the scheduler can improve efficiency by considering cache locality.

Example:

```text
Request A: shares 2,000 cached tokens
Request B: shares 0 cached tokens
```

Running A first may produce a much lower prefill cost than treating both requests as equal. This is why RadixAttention is not only a data structure; it can influence scheduling policy.

### When RadixAttention Helps Most

It helps most when traffic has shared prefixes:

- same system prompt,
- same few-shot examples,
- same tool instructions,
- same RAG template,
- same conversation prefix,
- branching workflows.

It helps less when every request starts with unrelated random text.

### Cost And Tradeoff

RadixAttention adds runtime bookkeeping:

- prefix tree search,
- insertion and node splitting,
- reference counting,
- eviction management,
- cache-aware scheduling logic.

The intended tradeoff:

```text
small management overhead
in exchange for
less repeated prefill and better KV reuse
```

If there are no shared prefixes, the framework should avoid adding too much overhead. The SGLang paper reports low overhead even on workloads with limited sharing, but actual results should still be validated by benchmark.

### Chinese Interview Sentence

```text
RadixAttention 可以理解成 SGLang 的自动前缀 KV cache 复用机制。它用 radix tree 存 token prefix 和对应的 KV cache。新请求进来时，runtime 先在树里找最长匹配前缀，命中的部分直接复用 KV cache，只对没命中的 suffix 做 prefill，然后把新算出来的 suffix 插回树里。如果新请求和已有缓存只有部分重合，radix tree 会做节点分裂，把共享前缀提出来。显存不够时，需要按类似 LRU 的策略淘汰不活跃的叶子节点，同时用引用计数避免删掉 running batch 还在用的 cache。

所以 RadixAttention 不只是一个 cache 数据结构，它会影响 scheduler：如果某个请求能命中很长前缀，优先调度它可能显著降低 prefill 成本和 TTFT。它最适合 system prompt、tool instruction、RAG template、多轮对话、agent workflow 这类 prefix-heavy workload；如果请求前缀完全随机，收益就会小很多。
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
- SGLang RadixAttention docs: https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention
- SGLang GitHub: https://github.com/sgl-project/sglang
- SGLang paper: https://arxiv.org/html/2312.07104v2
