# Prefill-Decode Disaggregation

Prefill-decode disaggregation, often shortened as PD disaggregation, separates the prefill phase and decode phase of LLM inference onto different workers, instances, or clusters.

Beginner mental model:

```text
normal serving:
same engine/GPU pool does prefill and decode together

PD disaggregation:
prefill workers process prompts and produce KV cache
decode workers consume KV cache and generate output tokens
```

The key reason is that prefill and decode stress the system differently.

## Prefill vs Decode Resource Shape

Prefill:

- processes the whole input prompt,
- creates the initial KV cache,
- is compute-intensive,
- often benefits from large matrix operations and high parallelism,
- mainly affects TTFT.

Decode:

- generates one token at a time,
- repeatedly reads and appends KV cache,
- is memory/KV-cache intensive,
- often has smaller per-step compute but many sequential steps,
- mainly affects TPOT and inter-token latency.

This difference creates a scheduling problem when both phases share one engine.

## Why Co-Location Can Be Inefficient

In a unified engine, prefill and decode are scheduled together.

That can cause interference:

```text
large prefill batch
-> occupies compute for a while
-> ongoing decode requests wait
-> inter-token latency gets worse
```

Or:

```text
many decode requests
-> keep reading KV cache and occupying scheduler capacity
-> new long-prefill requests wait
-> TTFT gets worse
```

So one phase can hurt the other phase's latency.

## What PD Disaggregation Does

PD disaggregation splits responsibilities:

```text
prefill worker:
receive request prompt
run prefill
produce prompt KV cache
send KV cache to decode worker

decode worker:
receive KV cache
run autoregressive decode
stream output tokens
```

This allows each phase to be tuned separately.

For example:

- prefill workers may use one parallelism strategy,
- decode workers may use another,
- prefill capacity can scale based on prompt load,
- decode capacity can scale based on output-token load.

## KV Cache Transfer

The hard part is not only splitting compute. The hard part is moving KV cache.

After prefill, the decode worker needs the generated KV cache:

```text
prompt tokens
-> prefill worker builds KV cache
-> KV cache transfer
-> decode worker starts generating
```

This creates new systems questions:

- How large is the KV cache?
- Is the network fast enough?
- Can transfer overlap with compute?
- Is transfer done GPU-to-GPU, CPU-mediated, RDMA, or through a cache layer?
- What happens when decode workers are overloaded?
- How does the scheduler choose which decode worker receives the KV cache?

If KV transfer is slow, PD disaggregation can lose its benefit.

## Mooncake Mental Model

Mooncake is described as a KVCache-centric disaggregated architecture for LLM serving.

Beginner version:

```text
Mooncake separates prefill and decode clusters
and treats KV cache as a first-class distributed resource
```

It also uses underutilized CPU, DRAM, SSD, and NIC resources to build a disaggregated KV cache pool.

The key idea is not just:

```text
split prefill and decode
```

It is:

```text
make KV cache placement, transfer, and reuse central to serving architecture
```

This connects to long-context serving because long prompts produce huge KV caches.

## DistServe Mental Model

DistServe focuses on disaggregating prefill and decoding to improve goodput under latency objectives.

The important idea:

```text
prefill and decode have different resource needs
so colocating them couples their scheduling and parallelism choices
```

By separating them, the system can optimize TTFT and TPOT-related objectives more independently.

## vLLM And SGLang Support

vLLM documentation describes disaggregated prefilling as running prefill and decode phases in different vLLM instances. This allows different parallel strategies for tuning TTFT and inter-token latency separately.

SGLang documentation describes PD disaggregation as separating compute-intensive prefill from memory-intensive decode, with profiling guidance for prefill and decode workers separately.

The exact command-line options and maturity can change quickly, so always check framework docs before running experiments.

## When PD Disaggregation Helps

It is more likely to help when:

- prompts are long,
- prefill causes decode stalls,
- decode latency SLO is tight,
- workload mix has uneven prompt/output lengths,
- prefill and decode need different parallelism strategies,
- the cluster has fast interconnect for KV cache transfer,
- the serving system can schedule KV placement well.

It may help less or hurt when:

- prompts are short,
- KV transfer overhead is large,
- interconnect is slow,
- the system is small and simple,
- GPU utilization is already good,
- scheduler overhead becomes too complex,
- decode workers wait for KV transfers.

## Metrics To Track

Do not evaluate PD disaggregation only by throughput.

Track:

- TTFT,
- TPOT / ITL,
- p95/p99 latency,
- prefill queue length,
- decode queue length,
- KV cache transfer time,
- KV cache transfer bandwidth,
- GPU utilization per worker type,
- KV cache memory usage,
- goodput under SLO.

The central question:

```text
Does separating prefill and decode reduce phase interference enough
to outweigh KV transfer and scheduling overhead?
```

## Interview Sentence

```text
Prefill-decode disaggregation 是把 LLM 推理的 prefill 和 decode 两个阶段拆到不同 worker 或集群上。原因是这两个阶段资源特性不同：prefill 处理整个 prompt，偏计算密集，主要影响 TTFT；decode 每次生成一个 token，不断读写 KV cache，偏内存和 KV cache 密集，主要影响 TPOT/ITL。放在同一个 engine 里时，大 prefill batch 可能阻塞 decode，导致 inter-token latency 变差；大量 decode 也可能让新的 prefill 排队，导致 TTFT 变差。

PD 分离后，prefill worker 负责处理 prompt 并生成 KV cache，decode worker 接收 KV cache 后继续生成 token。这样可以分别给 prefill 和 decode 配不同并行策略、不同资源规模和不同调度策略。但难点是 KV cache transfer：prefill 生成的 KV cache 必须高效传到 decode worker，如果网络或传输层太慢，收益会被抵消。

Mooncake 这类 KVCache-centric 架构进一步把 KV cache 当成一等资源，关注 KV 的放置、传输、复用和分布式缓存池。面试里我会说，PD disaggregation 的核心 tradeoff 是减少 prefill/decode phase interference，同时控制 KV transfer 和调度复杂度；评估时要看 TTFT、TPOT/ITL、p99、KV transfer time、GPU utilization 和 goodput under SLO。
```

## Sources

- SGLang PD disaggregation docs: https://docs.sglang.ai/advanced_features/pd_disaggregation.html
- vLLM disaggregated prefilling docs: https://docs.vllm.ai/en/stable/features/disagg_prefill/
- DistServe paper: https://arxiv.org/abs/2401.09670
- Mooncake paper: https://arxiv.org/pdf/2407.00079
- Mooncake docs: https://kvcache-ai.github.io/Mooncake/
- Mooncake GitHub: https://github.com/kvcache-ai/Mooncake
