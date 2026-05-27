# GPU MODE vectorsum_v2 A100

Single-file submission workspace for the GPU MODE `vectorsum_v2` leaderboard.

## Goal

Submit a correct first A100 entry, then iterate from real leaderboard feedback.

## Files

- `submission.py`: Popcorn-compatible Triton two-stage reduction.
- `submission_atomic.py`: Atomic-add comparison kernel.
- `submission_cuda_inline.py`: CUDA inline two-stage reduction experiment.
- `submission_cuda_inline_v2.py`: Official-template CUDA inline two-stage reduction experiment.
- `validate_submission.py`: Local correctness and sanity benchmark script.
- `sweep_a100_block_size.py`: Generate and benchmark `BLOCK_SIZE` variants.
- `sweep_a100_num_warps.py`: Generate and benchmark `num_warps` variants.
- `sweep_a100_cuda_inline_config.py`: Generate and benchmark CUDA inline `THREADS x ITEMS_PER_THREAD` variants.

## Local 4090 Validation

On `pink-outer`:

```bash
cd ~/gpu-mode/vectorsum-v2-a100
docker run --gpus all --rm \
  -v /mnt/hdd/users/cychi/gpu-mode/vectorsum-v2-a100:/workspace \
  -w /workspace \
  pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel \
  python validate_submission.py --benchmark
```

Observed on RTX 4090:

```text
benchmark size=1638400 0.0171 ms 384.07 GB/s
benchmark size=3276800 0.0265 ms 495.17 GB/s
benchmark size=6553600 0.0455 ms 576.36 GB/s
benchmark size=13107200 0.0840 ms 624.08 GB/s
benchmark size=26214400 0.1602 ms 654.49 GB/s
benchmark size=52428800 0.2843 ms 737.74 GB/s
```

## A100 Submission

```bash
popcorn-cli submit --no-tui --leaderboard vectorsum_v2 --gpu A100 --mode test submission.py
popcorn-cli submit --no-tui --leaderboard vectorsum_v2 --gpu A100 --mode benchmark --output benchmark-a100.json submission.py
popcorn-cli submit --no-tui --leaderboard vectorsum_v2 --gpu A100 --mode leaderboard --output leaderboard-a100.json submission.py
```

Observed A100 result:

```text
rank: 3
submission_id: 783079
score: 137.626 us
ranked benchmark: 138 +/- 0.1 us
best ranked sample: 136 us
```

If the CLI returns `401 Unauthorized`, run:

```bash
popcorn-cli reregister github
```

Then open the printed GitHub OAuth URL in a browser.

## A100 BLOCK_SIZE Sweep

Generate variants:

```bash
python sweep_a100_block_size.py
```

Run A100 benchmark mode for all variants:

```bash
python sweep_a100_block_size.py --run
```

First sweep result:

| BLOCK_SIZE | mean | best | worst |
| ---: | ---: | ---: | ---: |
| 4096 | 145 us | 139 us | 148 us |
| 8192 | 144 us | 137 us | 150 us |
| 16384 | 147 us | 138 us | 153 us |
| 32768 | 147 us | 140 us | 174 us |

`BLOCK_SIZE=8192` remains the best coarse candidate.

## A100 num_warps Sweep

Generate first-round variants:

```bash
python sweep_a100_num_warps.py \
  --partial-warps 4,8,16 \
  --final-warps 8 \
  --output-dir outputs/num-warps-sweep-first
```

Run A100 benchmark mode:

```bash
python sweep_a100_num_warps.py \
  --partial-warps 4,8,16 \
  --final-warps 8 \
  --output-dir outputs/num-warps-sweep-first \
  --run
```

First A100 result:

| Candidate | mean | best | worst |
| --- | ---: | ---: | ---: |
| `wp4_wf8` | 146 us | 140 us | 151 us |
| `wp8_wf8` | 145 us | 141 us | 148 us |
| `wp16_wf8` | 145 us | 139 us | 147 us |

None of these beat the existing leaderboard baseline.

## Atomic-Add Comparison

`submission_atomic.py` uses two launches:

```text
zero output[0]
each Triton program reduces one chunk and atomic_adds its partial sum into output[0]
```

This removes the partial buffer and final reduction kernel, but introduces contention on one global memory address.

Observed A100 benchmark mode:

| Candidate | mean | best | worst |
| --- | ---: | ---: | ---: |
| `atomic` | 144 us | 137 us | 148 us |

This was competitive, but not a clear leaderboard improvement over the current `137.626 us` score.

## CUDA Inline Experiment

`submission_cuda_inline.py` uses `torch.utils.cpp_extension.load_inline` to compile a CUDA C++ two-stage reduction:

```text
THREADS = 256
ITEMS_PER_THREAD = 32
ELEMENTS_PER_BLOCK = 8192
```

It passed RTX 4090 correctness and sanity benchmark:

```text
largest shape: 0.2846 ms, 736.95 GB/s
```

Popcorn A100 benchmark rejected it:

```text
Your code contains work on another stream.
```

Passing `torch.cuda.current_stream(x.device).cuda_stream` explicitly into C++ did not resolve the rejection, so this path is currently recorded as a local learning baseline rather than a viable leaderboard submission.

`submission_cuda_inline_v2.py` follows the official `vectoradd_py` inline-CUDA template more closely:

```text
load_inline(functions=[...])
no custom PYBIND11_MODULE
CUDA wrapper functions return tensors
```

This version passed Popcorn A100 benchmark mode:

| Candidate | mean | best | worst |
| --- | ---: | ---: | ---: |
| `cuda_inline_v2` | 147 us | 139 us | 154 us |

It is not leaderboard-worthy yet, but it establishes an accepted CUDA inline starting point.

### CUDA Inline Config Sweep

Generate first-round CUDA inline variants:

```bash
python sweep_a100_cuda_inline_config.py --output-dir outputs/cuda-inline-config-sweep
```

Run A100 benchmark mode:

```bash
python sweep_a100_cuda_inline_config.py \
  --output-dir outputs/cuda-inline-config-sweep \
  --run
```

First sweep result:

| Candidate | elements/block | mean | best | worst |
| --- | ---: | ---: | ---: | ---: |
| `t128_i32` | 4096 | 154 us | 144 us | 162 us |
| `t128_i64` | 8192 | 154 us | 145 us | 161 us |
| `t256_i64` | 16384 | 152 us | 144 us | 159 us |
| `t512_i16` | 8192 | 156 us | 152 us | 164 us |
| `t512_i32` | 16384 | 152 us | 143 us | 157 us |

The runner reported `NVIDIA A100 80GB PCIe` for this sweep, while earlier CUDA inline v2 ran on `A100-SXM4-80GB`, so future sweeps should include the baseline config in the same batch for cleaner relative comparison.
