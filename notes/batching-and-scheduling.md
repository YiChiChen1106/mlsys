# Batching And Scheduling

Batching is how an inference server turns many small requests into larger GPU workloads. Scheduling decides which requests enter the batch and when they leave.

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

## Experiment Questions

- At what concurrency does throughput stop improving?
- At what concurrency does TTFT become unacceptable?
- Does a longer prompt hurt only TTFT, or also TPOT?
- Does a longer output hurt other users through scheduler pressure?
