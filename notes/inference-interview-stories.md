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
