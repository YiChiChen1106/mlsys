# Inference Interview Stories

This note turns the vLLM benchmark lab into interview-ready stories.

Interview answers should be Chinese-first. Add English only when needed for specific terms or bilingual practice.

## Story 1: Benchmark Methodology

### Problem

Naive LLM serving benchmarks can produce misleading numbers. Cold-start latency, warmup requests, bad token counting, and incorrect throughput formulas can all hide the actual steady-state behavior of an inference framework.

### What I Did

- Started with a vLLM OpenAI-compatible server on `pink`.
- Served Qwen2.5 models from a local Hugging Face snapshot in Docker offline mode.
- Added warmup requests so steady-state latency does not include first-request initialization.
- Measured concurrent throughput using wall-clock time after warmup.
- Counted generated tokens from vLLM streaming `usage.completion_tokens`.
- Added p50, p95, and p99 latency and TTFT metrics.
- Added prompt-token accounting to separate prefill-sensitive and decode-sensitive cases.

### Result

The benchmark became reliable enough to compare:

- concurrency changes,
- TP=1 vs TP=2,
- prompt-length effects,
- output-length effects,
- KV cache pressure,
- prefix-cache behavior,
- scheduler parameter changes.

### Interview Answer

```text
I learned that LLM serving benchmarks need careful methodology. My first vLLM request had high TTFT because it included warmup effects such as Triton JIT and runtime initialization, so I added warmup requests and excluded them from the measured window. For throughput, I used total completion tokens divided by measured wall-clock time, because concurrent requests overlap and summing request latencies would be wrong. I also used vLLM's streaming usage field for completion-token counts instead of whitespace-based estimation. This gave me reliable TTFT, TPOT, p95/p99 latency, and output-token throughput for later scheduler and KV-cache experiments.
```

### Follow-Up Questions

- Why is first-request latency different from steady-state latency?
- Why is wall-clock throughput the right metric under concurrency?
- Why is `max_tokens` not the same as actual generated tokens?
- Why do p95 and p99 matter for inference serving?
- What is the difference between latency, TTFT, and TPOT?

## Story 2: Tensor Parallel Scaling

### Problem

Adding a second GPU does not automatically double LLM inference throughput. Tensor parallelism reduces per-GPU compute, but it also adds cross-GPU communication.

### Beginner Explanation

Tensor parallelism means splitting one model layer across multiple GPUs.

For example, a large matrix multiplication can be split so GPU 0 computes one part and GPU 1 computes another part. This reduces the compute burden on each GPU, but the GPUs must exchange partial results. That communication usually happens every layer.

So TP has two sides:

```text
benefit: less compute per GPU
cost: cross-GPU communication and synchronization
```

If the GPUs are connected by NVLink, communication is faster. On the `pink` server, GPU0-GPU1 topology is `SYS`, so communication goes through PCIe/CPU interconnect. This makes communication overhead more visible.

### What I Did

- Ran Qwen2.5-7B-Instruct with TP=1 on one RTX 4090.
- Ran the same model with TP=2 across two RTX 4090s.
- Checked GPU topology and found GPU0-GPU1 communication is `SYS`, not NVLink.
- Used `--disable-custom-all-reduce` for stability on the PCIe/SYS setup.
- Compared throughput, TTFT, latency, TPOT, and tail percentiles.

### Result

TP=2 improved short-prompt throughput from about 710 tok/s to about 1099 tok/s at concurrency 16, a roughly 1.55x speedup. Tail latency also improved at concurrency 16: p99 latency dropped from about 765 ms to about 506 ms, and p99 TTFT dropped from about 213 ms to about 114 ms.

The speedup stayed below 2x because tensor parallelism adds communication overhead. On this machine the two GPUs communicate through a `SYS` topology rather than NVLink, so all-reduce and synchronization costs matter.

### Interview Answer

```text
I compared TP=1 and TP=2 on a dual RTX 4090 server using vLLM. TP=2 improved throughput and tail latency, but the speedup was around 1.5x to 1.7x instead of 2x. The reason is that tensor parallelism reduces per-GPU compute but adds cross-GPU communication every layer. On this server, the GPU topology was SYS rather than NVLink, so communication went through PCIe and the CPU interconnect. That made TP=2 helpful for decode-heavy traffic, but not a free 2x scaling path.
```

### 中文面试表达

```text
我在双 4090 服务器上对比了 vLLM 的 TP=1 和 TP=2。TP=2 确实提升了吞吐和尾延迟，比如 concurrency 16 时吞吐从大约 710 tok/s 提升到 1099 tok/s，p99 latency 也从大约 765 ms 降到 506 ms。但它不是 2 倍加速，而是大概 1.5 到 1.7 倍。

原因是 tensor parallel 一方面把模型计算切到多张 GPU 上，降低了每张卡的计算量；另一方面每一层都可能需要 all-reduce 或同步通信。pink 这台机器两张 4090 的拓扑是 SYS，不是 NVLink，所以跨卡通信要走 PCIe/CPU interconnect，通信开销会抵消一部分计算收益。因此 TP=2 对 decode-heavy workload 有帮助，但不是免费的线性扩展。
```

### 面试官可能追问

- 为什么 TP=2 不一定比 TP=1 好？
- 为什么 NVLink 和 PCIe/SYS 拓扑会影响 TP 扩展效率？
- TP 对 prefill 和 decode 的收益一样吗？
- 为什么 TP=2 后每张 GPU 的显存占用没有简单减半？
- `all-reduce` 在 tensor parallel 里为什么会出现？

## Story 3: Prefill vs Decode

### Problem

LLM inference has two different phases. Prefill processes the whole prompt and mainly affects TTFT. Decode generates tokens autoregressively and mainly affects TPOT and end-to-end latency.

### What I Did

- Added server-reported `prompt_tokens` to benchmark outputs.
- Added synthetic prompt buckets to control prompt length.
- Added prompt salt and varied prompts to reduce accidental prefix-cache reuse.
- Swept prompt length to study prefill.
- Swept output length from 16 to 256 tokens to study decode.

### Result

For TP=1, salted prompt TTFT increased from about 71 ms at roughly 575 prompt tokens to about 201 ms at roughly 1984 prompt tokens. Decode latency scaled almost linearly with output length. TP=1 TPOT stayed near 15.5 ms/token, while TP=2 stayed near 8.8 ms/token.

### Interview Answer

```text
I separated prefill and decode experimentally. For prefill, I swept synthetic prompt lengths with varied salted prompts to avoid prefix-cache reuse and observed TTFT rising as prompt tokens increased. For decode, I fixed the prompt and swept output length from 16 to 256 tokens, and latency scaled almost linearly with generated tokens. This showed that prompt length mainly affects TTFT through prefill, while output length mainly affects TPOT and total latency through decode.
```

## Story 4: KV Cache Pressure And Prefix Cache

### Problem

KV cache determines how many active sequences and tokens the server can keep during decoding. Pressure may appear as long waiting queues and high tail latency before requests actually fail.

### What I Did

- Ran long-prompt, high-concurrency workloads.
- Sampled vLLM `/metrics` during benchmarks.
- Tracked KV usage, running requests, waiting requests, capacity waiting, and preemptions.
- Compared repeated prompts with varied and salted prompts.
- Computed prefix-cache hit ratio from vLLM metrics.

### Result

In the KV pressure time-series run, TP=1 reached about 97% max KV usage, max waiting 27, and p99 TTFT around 5.4 s. TP=2 had lower reported KV usage but still had waiting queues and similar tail latency, showing that no single metric explains the whole system.

Prefix cache made repeated prompts much faster. Repeated synthetic prompts reached about 96-99% prefix-cache hit ratio, while salted varied prompts were about 2%. The TTFT difference confirmed that prefix cache should be measured intentionally rather than accidentally mixed into raw prefill tests.

### Interview Answer

```text
I tested KV-cache pressure by combining long prompts with high concurrency. The requests did not fail, but p99 TTFT rose to several seconds and vLLM metrics showed waiting queues. This taught me that cache or scheduler pressure often appears first as tail latency, not as OOM. I also measured prefix-cache behavior using vLLM metrics. Repeated prompts had around 96-99% hit ratio and much lower TTFT, while salted varied prompts had near-zero hit ratio and exposed the uncached prefill cost.
```

### 中文面试表达

```text
我会区分普通 KV cache 和 prefix cache。普通 KV cache 是同一个请求内部的复用：prefill 阶段写入 prompt 的 K/V，decode 阶段每生成一个新 token 都复用历史 K/V，主要提升 decode 速度。Prefix cache 是不同请求之间的复用：如果多个请求有相同的 prompt 前缀，框架可以复用这段前缀的 prefill 计算结果，从而降低 TTFT。

我在 vLLM 实验里专门做了对比。repeated prompt 的 prefix-cache hit ratio 大约 96-99%，TTFT 很低；salted varied prompt 的 hit ratio 只有约 2%，TTFT 明显更高，更接近真实 uncached prefill 成本。这个实验告诉我，prefix cache 是很有价值的优化，但 benchmark 时必须把它和普通 prefill 性能分开，否则容易把缓存命中误认为模型 prefill 本身很快。
```

### 面试官可能追问

- 普通 KV cache 和 prefix cache 的区别是什么？
- prefix cache 为什么能降低 TTFT？
- 什么样的业务场景 prefix cache 收益最大？
- 为什么 repeated prompt benchmark 可能不可信？
- 你怎么验证 prefix cache 真的命中了？

## Story 5: Scheduler Parameter Tradeoff

### Problem

Scheduler knobs are tradeoffs. A setting that improves p99 latency or throughput can worsen TTFT or increase preemptions.

### What I Did

- Held the slow workload fixed: synthetic long prompt, output 64, requests 96, concurrency 32.
- Compared default behavior with `--max-num-seqs 64`.
- Swept `--max-num-batched-tokens` through 4096, 6144, and 8192.
- Tracked TTFT, p99 latency, TPOT, max waiting, KV usage, and preemptions.

### Result

Increasing `max_num_batched_tokens` reduced max waiting and p99 latency, and improved TPOT. However, it also increased average TTFT. On TP=1, the largest tested value introduced preemptions.

### Interview Answer

```text
I swept vLLM's max_num_batched_tokens on a fixed long-prompt high-concurrency workload. Larger batched-token budgets reduced waiting queues, improved TPOT, and lowered p99 latency, but they also worsened average TTFT. On TP=1, the largest setting introduced preemptions. So I would describe max_num_batched_tokens as a scheduler tradeoff knob rather than a monotonic optimization. It changes how much prefill/decode work the engine admits per batch, which shifts latency between first-token delay, decode progress, and tail completion time.
```

## Story 6: Metrics And Profiling

### Problem

Client-side latency can tell me that a request is slow, but not why it is slow. For inference framework work, I need server-side metrics to connect latency symptoms to scheduler, KV cache, prefix cache, and preemption behavior.

### What I Did

- Added metrics snapshot collection around vLLM benchmarks.
- Computed prefix-cache hit ratio from counter deltas.
- Added metrics time-series sampling during long-running pressure tests.
- Tracked KV cache usage, running requests, waiting requests, capacity waiting, and preemptions.

### Result

The prefix-cache experiment used counter deltas to show repeated prompts had about 96-99% hit ratio, while salted varied prompts were about 2%.

The KV pressure experiment used time-series gauge sampling to show that p99 TTFT around 5-6 seconds came with waiting queues of 27-29 requests. TP=1 also reached about 97% peak KV usage.

### 中文面试表达

```text
我做推理 benchmark 时不会只看客户端 latency。客户端指标，比如 TTFT、TPOT、p95/p99 latency 和吞吐，能告诉我用户侧发生了什么；但如果要分析推理框架，就需要服务端 metrics 来解释原因，比如 KV cache usage、running/waiting requests、preemptions、prefix cache hits 和 queries。

我会区分 counter 和 gauge。对于 prefix_cache_hits_total、prefix_cache_queries_total 这种 counter，我用前后差值算 hit ratio。对于 kv_cache_usage_perc、num_requests_waiting 这种 gauge，我不会只做前后 snapshot，因为压测结束后数值会回到 idle，容易错过峰值。所以我写了 time-series 采样脚本，在 benchmark 过程中持续采 `/metrics`。在 KV pressure 实验里，我就是这样观察到 p99 TTFT 到 5-6 秒时，waiting requests 峰值达到 27-29，TP=1 的 KV usage peak 接近 97%。
```

### 面试官可能追问

- 客户端 latency 和服务端 metrics 分别解决什么问题？
- counter 和 gauge 的区别是什么？
- 为什么 gauge 需要 time-series 采样？
- 如果 TTFT 很高，你会看哪些 metrics？
- 什么时候需要进一步用 Nsight 或 torch.profiler？

## Story 7: Context-Length Admission

### Problem

In LLM serving, a request can be rejected even if the prompt itself fits, because the server must reserve room for the requested output tokens.

### What I Did

- Served Qwen2.5-7B-Instruct with `--max-model-len 2048`.
- Used a synthetic prompt with 1968 server-reported prompt tokens.
- Tested requested output lengths of 80 and 81 tokens.

### Result

- 1968 prompt tokens + 80 requested output tokens = 2048, accepted.
- 1968 prompt tokens + 81 requested output tokens = 2049, rejected.

### 中文面试表达

```text
我在 vLLM 里验证过 context-length admission。max_model_len 限制的是单个请求的总序列长度，不是只限制 prompt 长度。服务端在生成前不知道模型会不会提前停止，所以必须按 requested max_tokens 做预算。

我的实验里 max_model_len=2048，一个 synthetic prompt 的 server-reported prompt_tokens 是 1968。当 requested max_tokens=80 时，总预算是 1968+80=2048，请求成功；当 requested max_tokens=81 时，总预算变成 2049，超过 max_model_len，所以请求被拒绝。这个规则也和 KV cache 管理相关，因为每个可能生成的位置都需要上下文和 cache 容量。
```

### 面试官可能追问

- 为什么看 requested max_tokens，而不是实际生成 token 数？
- prompt 本身没超过 max_model_len，为什么仍然可能被拒？
- 这个规则和 KV cache 有什么关系？
- 为什么 benchmark 要记录 server-reported prompt_tokens？

## Story 8: What vLLM Does

### Problem

A model checkpoint is not enough to serve real traffic. An inference framework must handle APIs, batching, scheduling, KV cache management, GPU execution, streaming, and observability.

### Beginner Explanation

vLLM is not a model. It is a serving framework around a model.

```text
model checkpoint = weights
vLLM = serving system
```

Important pieces:

- OpenAI-compatible API server,
- tokenizer and detokenizer,
- scheduler and continuous batching,
- KV cache manager and PagedAttention,
- GPU model executor / workers,
- tensor parallel support,
- prefix cache,
- `/metrics` observability.

### 中文面试表达

```text
vLLM 不是模型本身，而是 LLM 推理服务框架。模型 checkpoint 只提供权重、tokenizer 和 config；vLLM 负责把它变成在线服务，包括 OpenAI-compatible API server、tokenization、scheduler、continuous batching、KV cache 管理、PagedAttention、GPU worker/model executor、streaming response 和 metrics。

我理解 vLLM 的核心价值有两块：一是调度和 continuous batching，让多请求并发时 GPU 更忙、吞吐更高；二是 PagedAttention/KV cache 管理，把 KV cache 按 block/page 方式管理，减少长短请求混合时的显存浪费和碎片。再加上 `/metrics`，可以观察 waiting requests、KV usage、prefix cache hit ratio 和 preemptions，方便做 scheduler 和 cache system 优化。
```

### 面试官可能追问

- vLLM 和模型 checkpoint 的区别是什么？
- vLLM 为什么要做 continuous batching？
- PagedAttention 解决了什么问题？
- vLLM 的 `/metrics` 对性能优化有什么用？
- vLLM 和 SGLang 的侧重点可能有什么不同？

## Story 9: PagedAttention

### Problem

In LLM serving, KV cache can dominate GPU memory. Requests have different prompt lengths and output lengths, so reserving large contiguous KV regions can waste memory and limit concurrency.

### Beginner Explanation

PagedAttention manages KV cache like pages or blocks.

```text
logical sequence tokens
-> logical blocks
-> block table
-> physical KV cache blocks
```

The request sees a continuous sequence, but the physical KV blocks in GPU memory do not have to be contiguous.

### Why It Helps

- Allocates KV blocks on demand as sequences grow.
- Reduces memory waste from over-reserving large contiguous regions.
- Releases blocks when requests finish.
- Limits waste mostly to the last partially filled block.
- Makes prefix sharing easier because physical blocks can be reused.

### 中文面试表达

```text
PagedAttention 的核心不是改 attention 的数学公式，而是改 KV cache 的内存管理方式。它借鉴操作系统分页思想，把每个序列的 KV cache 切成固定大小的 block，通过 block table 把逻辑 token block 映射到 GPU 显存里的物理 KV block。这样序列在逻辑上是连续的，但物理存储可以不连续。

这样做的好处是，请求不需要一开始就预留一大段连续 KV cache，而是随着 decode 增长按需分配 block；请求结束后 block 可以归还；显存浪费主要限制在最后一个没填满的 block。对于长短请求混合、高并发 serving，这能提高 KV cache 利用率，也让 scheduler 能根据可用 block 做 admission 和 waiting 决策。
```

### 面试官可能追问

- PagedAttention 和普通 attention 的区别是什么？
- 为什么说它像操作系统分页？
- block table 是做什么的？
- 它怎么减少 KV cache 碎片和浪费？
- 它和 prefix cache 有什么关系？

## Story 10: vLLM vs SGLang

### Problem

Different inference frameworks may expose similar serving APIs, but their design emphasis can be different. For an inference framework role, I should be able to explain why I started with vLLM and why SGLang is a natural next framework to study.

### Beginner Explanation

Both vLLM and SGLang are high-performance LLM serving frameworks. Both care about batching, KV cache, multi-GPU execution, prefix caching, and serving throughput.

The beginner distinction:

```text
vLLM = general high-throughput LLM serving engine
SGLang = serving/runtime system with strong structured generation and prefix-reuse focus
```

### What I Have Done

- Used vLLM first because it has an OpenAI-compatible server, Docker image, PagedAttention, continuous batching, tensor parallel, prefix cache, and `/metrics`.
- Benchmarked vLLM on `pink` with Qwen2.5-7B-Instruct.
- Measured TTFT, TPOT, latency, throughput, KV pressure, prefix-cache hit ratio, and scheduler knobs.

### Next Natural Step

Run SGLang on the same model and server, then compare:

- same benchmark client flow,
- TTFT/TPOT/latency/throughput,
- prefix-heavy vs salted varied prompts,
- scheduler/KV cache metrics if available,
- behavior under long prompt and high concurrency.

### 中文面试表达

```text
我会把 vLLM 和 SGLang 都看成高性能 LLM serving framework，但侧重点不完全一样。vLLM 更像通用高吞吐 serving engine，核心亮点是 PagedAttention、continuous batching、OpenAI-compatible API、tensor parallel、prefix cache 和 metrics，所以我先用它搭了 benchmark lab，系统测 TTFT、TPOT、吞吐、KV pressure 和 scheduler 参数。

SGLang 也支持高性能 serving，但它更强调结构化生成、程序化 LLM workflow 和 prefix-heavy 场景。它的 RadixAttention 会把可复用的 KV cache 前缀组织到 radix tree 里，方便跨请求和复杂 workflow 复用共享前缀。简单说，vLLM 的 PagedAttention 更偏 KV cache 存储管理，SGLang 的 RadixAttention 更偏共享前缀复用。下一步我会在同一台双 4090 机器上跑 SGLang，用同样 workload 对比 prefix cache、TTFT、TPOT 和 scheduler 行为。
```

### 面试官可能追问

- vLLM 和 SGLang 都能 serving，为什么还要比较？
- PagedAttention 和 RadixAttention 的区别是什么？
- 什么 workload 更适合 SGLang？
- 如果让你评估一个新推理框架，你会怎么设计 benchmark？
- vLLM 已经有 prefix cache，SGLang 的 RadixAttention还有什么值得学？

Sources:

- vLLM documentation: https://docs.vllm.ai/
- SGLang documentation: https://docs.sglang.ai/
