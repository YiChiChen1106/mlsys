# vectorsum_v2 A100 CUDA Inline v1

## Goal

Try a hand-written CUDA C++ two-stage reduction through `torch.utils.cpp_extension.load_inline`.

This is the first CUDA inline baseline, not an optimized final submission.

## Kernel Structure

The implementation lives in:

```text
projects/gpu-mode-vectorsum-v2/submission_cuda_inline.py
```

Constants:

```text
THREADS = 256
ITEMS_PER_THREAD = 32
ELEMENTS_PER_BLOCK = 8192
```

Flow:

```text
stage 1:
x[N] -> partial[num_blocks]

stage 2:
partial[num_blocks] -> output[0]
```

Inside each block:

```text
thread local sum
-> warp reduction with __shfl_down_sync
-> shared memory for inter-warp sums
-> warp reduction again
-> one partial sum per block
```

## Local Correctness

The CUDA inline file passed local structure tests:

```text
python -m pytest test_submission_cuda_inline.py test_sweep_a100_block_size.py test_sweep_a100_num_warps.py test_submission_atomic.py
10 passed
```

It also compiled and passed correctness in the RTX 4090 container on `pink-outer`:

```text
correctness size=1023 pass
correctness size=1024 pass
correctness size=1025 pass
correctness size=2048 pass
correctness size=4096 pass
```

## RTX 4090 Sanity Benchmark

```text
size=1,638,400   0.0203 ms  322.22 GB/s
size=3,276,800   0.0283 ms  462.76 GB/s
size=6,553,600   0.0489 ms  536.38 GB/s
size=13,107,200  0.0865 ms  606.28 GB/s
size=26,214,400  0.1627 ms  644.61 GB/s
size=52,428,800  0.2846 ms  736.95 GB/s
```

This is not better than the current Triton baseline, but it is close on the largest shape.

## A100 Popcorn Result

Popcorn benchmark mode rejected the CUDA inline version:

```text
Application error: Server returned status 500 Internal Server Error:
Your code contains work on another stream. This is not allowed and may result in your disqualification.
```

An attempted fix explicitly passed `torch.cuda.current_stream(x.device).cuda_stream` from Python into the C++ launcher and used that stream for both kernel launches. Popcorn still returned the same stream-checker error.

## Interpretation

The CUDA inline kernel is locally correct, but this submission path is not accepted by the current `vectorsum_v2` A100 Popcorn runner.

The practical conclusion is:

```text
CUDA inline via torch.utils.cpp_extension.load_inline is not a viable next submission path unless we find the exact stream contract expected by Popcorn or a known accepted CUDA inline template for this leaderboard.
```

For now, continue optimization in Triton or use CUDA inline only as a learning exercise outside the official submission path.

## Next Step

Return to Triton-focused optimization:

- try vectorized/block-pair reduction patterns in Triton,
- inspect public winning-style approaches if available,
- avoid spending A100 submissions on `load_inline` until the stream issue is resolved.
