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
