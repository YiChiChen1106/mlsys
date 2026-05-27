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

## Competition Feedback Loop

For GPU MODE `vectorsum_v2`, the local RTX 4090 benchmark is useful for correctness and sanity checks, but A100 ranking must come from the official Popcorn runner.

Workflow:

```text
write single-file submission.py
-> verify correctness locally
-> sanity benchmark locally
-> submit mode=test on A100
-> submit mode=benchmark on A100
-> submit mode=leaderboard on A100
-> use leaderboard result to choose the next optimization
```

The first submission should optimize for getting a correct baseline onto the board. After that, the next round should be driven by measured A100 results rather than only local 4090 timing.

### First A100 Result

The first Popcorn CLI submission used the cached two-stage Triton implementation.

Result on 2026-05-27:

```text
leaderboard: vectorsum_v2
gpu: A100
rank: 3
submission_id: 783079
score: 137.626 us
```

This confirms that the same conceptual structure from the local learning project can produce a competitive official submission. Further optimization should target the small remaining gap to the current rank 1 score of about `135.339 us`.

### A100 BLOCK_SIZE Sweep

The first coarse A100 sweep tested:

```text
4096, 8192, 16384, 32768
```

Result:

```text
4096: 145 us mean, 139 us best
8192: 144 us mean, 137 us best
16384: 147 us mean, 138 us best
32768: 147 us mean, 140 us best
```

`BLOCK_SIZE=8192` remained the best candidate. The next tuning lever should be `num_warps` or a different reduction strategy, not a wider coarse block-size sweep.

### num_warps Sweep

In Triton, `num_warps` controls how many warps cooperate inside one program instance.

For vector sum, the first experiment keeps:

```text
BLOCK_SIZE = 8192
FINAL_BLOCK_SIZE = 8192
```

and changes:

```text
partial kernel num_warps = 4, 8, 16
final kernel num_warps = 8
```

This isolates the main read-heavy stage. The final stage reads only the partial sums, so it is less likely to dominate the total time.

First A100 benchmark-mode result:

```text
wp4_wf8: 146 us mean, 140 us best
wp8_wf8: 145 us mean, 141 us best
wp16_wf8: 145 us mean, 139 us best
```

This did not improve the current `137.626 us` leaderboard score.

### Atomic-Add Version

The atomic-add version changes the second stage:

```text
two-stage:
x[N] -> partial[num_blocks] -> output[0]

atomic:
zero output[0]
x[N] chunks -> atomic_add(output[0], partial)
```

Potential benefit:

- no partial buffer allocation or reuse logic,
- no final reduction over partial sums.

Potential cost:

- many programs contend on the same scalar output address,
- floating-point atomic order is nondeterministic.

So atomic-add is an experiment, not automatically an improvement. It must be judged by the official A100 benchmark.

First A100 benchmark-mode result:

```text
atomic: 144 us mean, 137 us best
```

The atomic version was the best of this batch by mean time, but still not strong enough to justify a new leaderboard submission.

### CUDA Inline Attempt

CUDA inline means embedding CUDA C++ code inside the Python submission and compiling it with `torch.utils.cpp_extension.load_inline`.

The first version used the same high-level two-stage structure:

```text
x[N] -> partial[num_blocks] -> output[0]
```

but implemented each block reduction manually:

```text
thread local sum
-> __shfl_down_sync inside each warp
-> shared memory for warp sums
-> final warp reduction
```

Local result:

```text
RTX 4090 correctness: pass
largest shape: 0.2846 ms, 736.95 GB/s
```

A100 Popcorn result:

```text
rejected: "Your code contains work on another stream"
```

Passing the Python current CUDA stream explicitly into the C++ launcher did not fix the Popcorn rejection.

Lesson:

```text
Local CUDA correctness is necessary but not sufficient. The official runner may enforce stream rules that a custom C++ extension does not satisfy, even when the kernel is correct locally.
```

Until there is an accepted CUDA inline template for this leaderboard, Triton remains the safer submission path.

Update: an accepted CUDA inline template exists. The working pattern matches the official `vectoradd_py` example:

```text
load_inline(functions=[...])
no custom PYBIND11_MODULE
CUDA wrapper returns torch::Tensor
```

The resulting `submission_cuda_inline_v2.py` passed A100 benchmark mode:

```text
cuda_inline_v2: 147 us mean, 139 us best
```

This is slower than the Triton score, but it proves CUDA inline can be used if written in the runner-friendly template shape.

The first CUDA config sweep tested:

```text
THREADS x ITEMS_PER_THREAD
128x32, 128x64, 256x64, 512x16, 512x32
```

A100 benchmark-mode result:

```text
t128_i32: 154 us mean
t128_i64: 154 us mean
t256_i64: 152 us mean
t512_i16: 156 us mean
t512_i32: 152 us mean
```

This did not improve the accepted CUDA inline baseline. It suggests the next step should change the reduction strategy, not just the block shape.

One measurement caution: the runner reported different A100 labels across runs (`A100-SXM4-80GB` vs `A100 80GB PCIe`), so clean sweeps should include a same-batch baseline candidate.

### CUDA Atomic Direction

The accepted-template CUDA atomic version changes the structure from:

```text
two-stage:
x[N] -> partial[num_blocks] -> output[0]
```

to:

```text
atomic:
zero output[0]
each block computes block_sum
atomicAdd(output[0], block_sum)
```

The tradeoff:

```text
less final-stage work
more contention on one scalar address
```

First candidate set:

```text
t256_i32: 8192 elements/block, about 6400 atomic adds
t256_i64: 16384 elements/block, about 3200 atomic adds
t512_i32: 16384 elements/block, about 3200 atomic adds
t256_i128: 32768 elements/block, about 1600 atomic adds
```

RTX 4090 correctness passed for all variants.

A100 benchmark-mode result:

```text
t256_i32: 148 us mean, 141 us best
t256_i64: 150 us mean, 138 us best
t512_i32: 149 us mean, 144 us best
t256_i128: 153 us mean, 142 us best
```

This did not beat the current `137.626 us` leaderboard baseline. The best sample, `138 us`, shows atomic can get close, but the mean is too slow for a safe leaderboard submission.

### Persistent / Grid-Stride Two-Stage

This version fixes the number of blocks and lets each block loop over multiple chunks:

```text
GRID_BLOCKS = 256/512/1024/2048
each block processes multiple ELEMENTS_PER_CHUNK windows
partial count = min(GRID_BLOCKS, ceil(N / ELEMENTS_PER_CHUNK))
```

The key code shape is:

```text
for (block_start = blockIdx.x * ELEMENTS_PER_CHUNK;
     block_start < n_elements;
     block_start += gridDim.x * ELEMENTS_PER_CHUNK)
```

This reduces partial count without using atomic contention.

Initial A100 result:

```text
g256: 157 us mean, 152 us best
g512: 145 us mean, 139 us best
```

`g512` is already competitive with the accepted CUDA inline baseline, but not yet better than the Triton leaderboard score.
