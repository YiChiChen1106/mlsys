# GPU Kernel Optimization

## One Sentence

GPU kernel optimization is the practice of mapping tensor operations to GPU programs that use memory bandwidth and compute resources efficiently.

## Basic Vocabulary

- **Operator**: a tensor operation such as sum, matmul, softmax, or layernorm.
- **Kernel**: the GPU program that implements an operator.
- **Program / block**: a unit of parallel work.
- **Memory bandwidth**: how fast data can be moved from GPU memory.
- **Throughput**: how much work is completed per second.
- **Latency**: how long one operation takes.

## Optimization Loop

```text
1. Establish PyTorch baseline.
2. Write a correct custom kernel.
3. Benchmark across target shapes.
4. Identify whether the workload is bandwidth-bound or compute-bound.
5. Sweep parameters.
6. Keep the fastest correct version.
```

## Common Levers

- `BLOCK_SIZE`
- `num_warps`
- memory access pattern
- reducing intermediate writes
- using partial sums
- avoiding unnecessary atomics
- reducing kernel launches when useful

## Current Example

Vector sum reduction is mostly memory-bandwidth-bound:

```text
Each float32 element needs one 4-byte read and roughly one addition.
The addition is cheap; reading the whole tensor dominates runtime.
```
