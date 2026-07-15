# Batching And Scheduling

Batching is how an inference server turns many small requests into larger GPU workloads. Scheduling decides which requests enter the batch and when they leave.

## Beginner Mental Model

An LLM server is not just running one user's request. It is usually serving many users at the same time.

If the server runs requests one by one, the GPU may be underused:

```text
request A -> GPU
request B waits
request C waits
request D waits
```

Batching tries to run several requests together:

```text
request A + request B + request C + request D -> GPU batch
```

This usually improves throughput because GPUs are good at parallel work. The tradeoff is that some requests may wait a little so the server can form efficient batches.

The scheduler is the component that makes this decision:

```text
Which requests should run now?
Which requests should wait?
How many prompt tokens can be prefilling in this step?
How many decode tokens can be generated in this step?
Is there enough KV cache capacity?
```

So the core tradeoff is:

```text
larger/more efficient batches -> better throughput
but possibly worse TTFT or tail latency
```

## Why LLM Scheduling Is Special

LLM requests are variable length in two dimensions:

- prompt length can vary from a few tokens to thousands of tokens,
- output length is not always known in advance.

They also have two different phases:

- prefill does a large prompt forward pass,
- decode generates one token per active sequence per step.

This makes scheduling harder than ordinary fixed-shape batching. The server has to mix new prefill work with ongoing decode work while staying inside GPU memory and KV cache limits.

## Static vs Continuous Batching, Intuition

Static batching is like waiting for a bus to fill up, driving the whole bus to the destination, then starting the next bus.

That is simple, but bad for LLMs because some passengers finish early and some need many more decode steps.

Continuous batching is more like a subway system:

```text
finished requests leave
new requests enter
active decode continues
```

This keeps the GPU busier and avoids waiting for every request in a fixed batch to finish.

## Static Batching

Static batching waits to form a fixed batch, runs it, and then returns results. It is simple, but bad for variable-length LLM generation because one slow request can hold up the rest.

## Continuous Batching

Continuous batching keeps the decode loop running while adding new requests and removing finished requests. This is one of the central ideas behind modern LLM serving systems.

```text
time step 1: A B C
time step 2: A B C D
time step 3: A C D
time step 4: C D E
```

## Metrics

- TTFT: time to first token, heavily affected by prefill and queueing.
- TPOT: time per output token, mostly decode-loop behavior.
- Throughput: total generated tokens per second.
- Queue time: time spent waiting before execution.
- Failure rate: requests that timeout or fail under load.
- Tail latency: p95/p99 TTFT and p95/p99 latency, often the first place scheduler pressure appears.

## vLLM Baseline Observation

At concurrency 64 on Qwen2.5-7B-Instruct:

- TP=1 p99 TTFT was about 252 ms and p99 latency was about 880 ms.
- TP=2 p99 TTFT was about 212 ms and p99 latency was about 822 ms.

This is why scheduler optimization should not be judged only by average latency or tokens/s. Queueing, batch admission, and long-running decode steps usually show up in the tail first.

With long salted prompts, the tail becomes much worse. At concurrency 32:

- TP=1 synthetic_768, output 64: p99 TTFT about 4.72 s.
- TP=1 synthetic_896, output 64: p99 TTFT about 5.48 s.
- TP=2 synthetic_768, output 64: p99 TTFT about 5.21 s.
- TP=2 synthetic_896, output 64: p99 TTFT about 6.01 s.

This is the interview-relevant link between scheduler and KV cache: even without failures, the scheduler may keep requests waiting because long sequences consume more cache and execution budget.

Time-series metrics made this clearer. In a synthetic_896/output64/concurrency32 run with 96 requests:

- TP=1 max running requests: 32; max waiting requests: 27; p99 TTFT about 5.41 s.
- TP=2 max running requests: 32; max waiting requests: 29; p99 TTFT about 5.94 s.
- Both waiting queues were labeled `reason="capacity"`.

This suggests the next scheduler experiment should tune `max_num_seqs` and `max_num_batched_tokens`, because the waiting queue is a direct scheduler signal.

## Scheduler Knob: max_num_batched_tokens

`max_num_batched_tokens` limits how many tokens the scheduler can include in a batch step. It controls how much total prefill/decode work can be admitted together.

In the vLLM sweep, larger values improved some throughput-oriented metrics:

- lower max waiting,
- better TPOT,
- lower p99 latency.

But they also worsened average TTFT. On TP=1, the largest tested value caused preemptions.

The beginner interpretation is:

```text
larger token budget:
more work admitted per batch
better decode progress / lower completion tail
but new requests may wait longer for first token
```

This is why scheduler tuning is not a simple "increase the knob" problem. It is a workload-dependent tradeoff.

## What Scheduler Pressure Looks Like

Scheduler pressure can appear as:

- rising TTFT,
- rising p95/p99 latency,
- more waiting requests,
- waiting reason labeled as capacity,
- preemptions,
- throughput flattening even as concurrency increases.

It does not have to appear as an immediate error or OOM.

## Interview Answer

```text
Batching groups multiple requests into larger GPU workloads to improve utilization. Static batching is simple but inefficient for LLMs because requests have variable output lengths. Modern LLM servers use continuous batching: finished requests leave, new requests enter, and active decode continues step by step. The scheduler decides which requests run, which wait, and how much prefill/decode work enters each batch under KV-cache and token-budget limits. This improves throughput, but it can trade off against TTFT and tail latency.
```

## Experiment Questions

- At what concurrency does throughput stop improving?
- At what concurrency does TTFT become unacceptable?
- Does a longer prompt hurt only TTFT, or also TPOT?
- Does a longer output hurt other users through scheduler pressure?
