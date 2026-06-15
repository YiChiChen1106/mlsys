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

## Experiment Questions

- At what concurrency does throughput stop improving?
- At what concurrency does TTFT become unacceptable?
- Does a longer prompt hurt only TTFT, or also TPOT?
- Does a longer output hurt other users through scheduler pressure?
