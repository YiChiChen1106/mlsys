# Vector Sum Reduction

## Problem

Compute the sum of all elements in a 1D tensor:

```text
input:  x.shape = (N,)
output: scalar = sum(x)
```

Benchmark shapes:

```text
1,638,400
3,276,800
6,553,600
13,107,200
26,214,400
52,428,800
```

## Why This Is MLsys-Relevant

Vector sum is small, but it teaches the core pattern behind many ML operators:

- reduction,
- memory bandwidth,
- partial sums,
- shape-dependent benchmarking,
- hardware-dependent tuning.

## Implementation Path

### 1. PyTorch Baseline

File: `projects/gpu-mode-vector-sum/baseline.py`

Use `torch.sum(x)` and measure effective bandwidth.

### 2. Mixed Triton Version

File: `projects/gpu-mode-vector-sum/vector_sum_triton.py`

Flow:

```text
x[N]
-> Triton partial sums
-> torch.sum(partial)
-> scalar
```

### 3. Block Size Sweep

File: `projects/gpu-mode-vector-sum/sweep_block_size.py`

Sweep:

```text
512, 1024, 2048, 4096, 8192
```

Best observed on RTX 4090 was usually `BLOCK_SIZE=8192`.

### 4. Full Two-Stage Triton

File: `projects/gpu-mode-vector-sum/vector_sum_two_stage.py`

Flow:

```text
x[N]
-> Triton partial sums
-> Triton final sum
-> scalar
```

This removes the PyTorch second-stage reduction.

## Key Results

Comparison on RTX 4090:

| N | torch.sum | mixed | two_stage |
| ---: | ---: | ---: | ---: |
| 1,638,400 | 0.0174 ms / 376.12 GB/s | 0.0175 ms / 375.55 GB/s | 0.0171 ms / 383.32 GB/s |
| 3,276,800 | 0.0270 ms / 485.51 GB/s | 0.0267 ms / 490.26 GB/s | 0.0263 ms / 497.80 GB/s |
| 6,553,600 | 0.0465 ms / 563.53 GB/s | 0.0459 ms / 571.52 GB/s | 0.0453 ms / 578.54 GB/s |
| 13,107,200 | 0.0848 ms / 618.32 GB/s | 0.0843 ms / 622.03 GB/s | 0.0836 ms / 626.80 GB/s |
| 26,214,400 | 0.1609 ms / 651.56 GB/s | 0.1607 ms / 652.53 GB/s | 0.1600 ms / 655.31 GB/s |
| 52,428,800 | 0.2926 ms / 716.84 GB/s | 0.2855 ms / 734.65 GB/s | 0.2844 ms / 737.47 GB/s |

## Concepts

### Memory-Bandwidth-Bound

Each `float32` element costs roughly:

```text
read 4 bytes
do one addition
```

The arithmetic is cheap, so reading the tensor dominates runtime.

### Partial Sum

Instead of one program summing all elements:

```text
many programs each sum one chunk
then partial sums are reduced
```

This increases parallelism and avoids one serial path.

### Block Size Sweep

`BLOCK_SIZE` controls how many elements each program processes.

Larger `BLOCK_SIZE`:

- fewer programs,
- fewer partial sums,
- cheaper second stage,
- but potentially less parallelism and more pressure inside each program.

So it must be measured, not guessed.

### Two-Stage Reduction

The two-stage Triton version improves slightly over the mixed version because it removes the PyTorch second-stage call, but the gain is limited because the second stage is small.

For `N=52,428,800` and `BLOCK_SIZE=8192`:

```text
num_blocks = 6400
```

The first stage reads 52M values, while the second stage reads only 6400 values.

## Interview Answer

```text
I optimized vector sum reduction by splitting the input into blocks, computing partial sums in Triton, sweeping block size, and then replacing the PyTorch second-stage sum with a Triton final reduction. The workload is mainly memory-bandwidth-bound because each float32 element only needs one read and one addition, so the key was maximizing effective bandwidth while preserving correctness.
```

## Review Notes - 2026-05-26

### Reduction

Reduction means combining many values into fewer values, often one scalar.

Examples:

```text
sum(x): x[N] -> scalar
max(x): x[N] -> scalar
```

An elementwise operation is different:

```text
x + 1: x[N] -> y[N]
```

The output shape stays aligned with the input, so it is not usually called a reduction.

### Why Vector Sum Is Bandwidth-Bound

For each `float32` element:

```text
read 4 bytes
do a small amount of addition
```

The arithmetic is cheap. The expensive part is reading the full tensor from GPU memory, so performance is best understood through effective memory bandwidth.

### Why Partial Sums Matter

A Triton program reduces one chunk of the input:

```text
x chunk -> one partial sum
```

Many programs run in parallel, producing many partial sums. This gives the GPU enough independent work to use memory bandwidth effectively.

### Why Sweep BLOCK_SIZE

`BLOCK_SIZE` controls how much input each Triton program handles.

Larger `BLOCK_SIZE`:

- fewer programs,
- fewer partial sums,
- cheaper final reduction,
- but possibly less parallelism.

Smaller `BLOCK_SIZE`:

- more programs,
- more parallelism,
- more partial sums,
- more second-stage work.

The best value is hardware- and shape-dependent, so it should be measured.

### Why Two-Stage Triton Only Helps A Little

The first stage reads all `N` input values. The second stage reads only the partial sums.

For the largest benchmark:

```text
N = 52,428,800
BLOCK_SIZE = 8192
num_blocks = 6400
```

The first stage reads tens of millions of values, while the second stage reads only thousands. Therefore replacing `torch.sum(partial)` with a Triton final reduction removes some overhead, but it does not change the main bottleneck.

### Stronger Interview Version

```text
I optimized vector sum reduction by splitting the input into blocks, computing partial sums in parallel, and sweeping BLOCK_SIZE for the target GPU. The final two-stage Triton version replaces the PyTorch second-stage sum with a Triton kernel. The speedup over torch.sum is small because torch.sum is already highly optimized, and the main bottleneck is reading the full input tensor from memory, not reducing the much smaller partial-sum array.
```
