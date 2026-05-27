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

Popcorn benchmark mode rejected v1:

```text
Application error: Server returned status 500 Internal Server Error:
Your code contains work on another stream. This is not allowed and may result in your disqualification.
```

An attempted fix explicitly passed `torch.cuda.current_stream(x.device).cuda_stream` from Python into the C++ launcher and used that stream for both kernel launches. Popcorn still returned the same stream-checker error.

## Interpretation

The CUDA inline kernel is locally correct, but this submission path is not accepted by the current `vectorsum_v2` A100 Popcorn runner.

The practical conclusion is:

```text
CUDA inline via a custom PYBIND11_MODULE wrapper is not a viable next submission path unless we find the exact stream contract expected by Popcorn.
```

## CUDA Inline v2

After checking the official `vectoradd_py` CUDA inline example, v2 was rewritten to match that template:

```text
load_inline(functions=[...])
no custom PYBIND11_MODULE
CUDA wrapper functions return tensors
kernel launches use the simple <<<blocks, threads>>> template style
```

File:

```text
projects/gpu-mode-vectorsum-v2/submission_cuda_inline_v2.py
```

Local tests:

```text
python -m pytest test_submission_cuda_inline_v2.py test_submission_cuda_inline.py test_sweep_a100_block_size.py test_sweep_a100_num_warps.py test_submission_atomic.py
13 passed
```

RTX 4090 correctness:

```text
correctness size=1023 pass
correctness size=1024 pass
correctness size=1025 pass
correctness size=2048 pass
correctness size=4096 pass
```

A100 Popcorn benchmark:

```text
147 ± 0.2 us
best 139 us
worst 154 us
```

This confirms that CUDA inline is a viable Popcorn submission path when written in the official template shape, but this first accepted CUDA baseline is slower than the current Triton leaderboard score of `137.626 us`.

## Next Step

Continue CUDA inline optimization from v2, not v1:

- sweep `THREADS` and `ITEMS_PER_THREAD`,
- compare two-stage vs accepted-template atomic,
- reduce overhead in the final stage,
- consider vectorized loads if the official runner accepts the code shape.

## CUDA Inline Config Sweep

Script:

```text
projects/gpu-mode-vectorsum-v2/sweep_a100_cuda_inline_config.py
```

Generated candidates:

| Candidate | THREADS | ITEMS_PER_THREAD | elements/block |
| --- | ---: | ---: | ---: |
| `t128_i32` | 128 | 32 | 4096 |
| `t128_i64` | 128 | 64 | 8192 |
| `t256_i64` | 256 | 64 | 16384 |
| `t512_i16` | 512 | 16 | 8192 |
| `t512_i32` | 512 | 32 | 16384 |

RTX 4090 correctness:

```text
all five generated variants passed sizes 1023, 1024, 1025, 2048, 4096
```

A100 benchmark mode:

| Candidate | mean | best | worst |
| --- | ---: | ---: | ---: |
| `t128_i32` | 154 us | 144 us | 162 us |
| `t128_i64` | 154 us | 145 us | 161 us |
| `t256_i64` | 152 us | 144 us | 159 us |
| `t512_i16` | 156 us | 152 us | 164 us |
| `t512_i32` | 152 us | 143 us | 157 us |

The runner reported `NVIDIA A100 80GB PCIe` for this sweep. Earlier `submission_cuda_inline_v2.py` benchmark output reported `NVIDIA A100-SXM4-80GB`, so absolute comparison to the earlier `147 us` result is imperfect.

Conclusion:

```text
No tested THREADS x ITEMS_PER_THREAD variant is leaderboard-worthy.
```

The next CUDA step should change the reduction strategy rather than only changing per-block shape. Also include `256x32` in the same future batch as a same-run baseline.
