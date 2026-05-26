# Bandwidth-Bound vs Compute-Bound

## Bandwidth-Bound

A workload is bandwidth-bound when runtime is dominated by moving data from memory.

Example:

```text
sum(x)
```

For each `float32` element:

```text
read 4 bytes
do one addition
```

The math is light, so the bottleneck is usually reading data from GPU memory.

Memory cue:

```text
bandwidth-bound = waiting for data
```

## Compute-Bound

A workload is compute-bound when runtime is dominated by arithmetic.

Example:

```text
matrix multiplication
```

Each loaded value can participate in many multiply-add operations, so compute throughput can become the bottleneck.

Memory cue:

```text
compute-bound = waiting for math
```

## How To Tell

Signs of bandwidth-bound behavior:

- runtime scales roughly with bytes read/written,
- effective GB/s approaches hardware limits,
- adding more arithmetic-light optimization does not help much.

Signs of compute-bound behavior:

- arithmetic intensity is high,
- Tensor Cores or FLOPS utilization matter,
- data reuse is important.

## Vector Sum Result

On RTX 4090:

```text
N=52,428,800 float32 values
data read ~= 209 MB
best two-stage Triton ~= 0.2844 ms
effective bandwidth ~= 737 GB/s
```

This is why vector sum reduction is treated as a memory-bandwidth-bound kernel.
