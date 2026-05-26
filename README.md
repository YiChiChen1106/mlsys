# MLsys Learning Lab

This repository is a learning lab for ML systems, GPU kernels, and inference performance engineering.

The current focus is GPU kernel optimization through small, measurable operators. The first project is a GPU MODE style vector sum reduction implemented with PyTorch and Triton.

## Current Status

- Main learning thread: GPU kernel optimization
- First project: vector sum reduction
- Hardware used so far: `pink` with dual RTX 4090
- Best observed result: `two_stage` reached about `737 GB/s` on the largest benchmark

## Start Here

- [[START_HERE]]
- [[notes/mlsys-learning-hub]]
- [[notes/vector-sum-reduction]]

## What Is In This Repo

```text
notes/        concepts and learning notes
daily/        daily logs
experiments/  benchmark writeups
projects/     runnable code
templates/    reusable note templates
```

## Projects

### GPU MODE Vector Sum

Path: `projects/gpu-mode-vector-sum`

Learning path:

1. Measure `torch.sum` as a baseline.
2. Implement a Triton partial-sum reduction.
3. Sweep `BLOCK_SIZE`.
4. Implement a full two-stage Triton reduction.
5. Explain the result in terms of memory bandwidth.

Best observed result on `pink` with RTX 4090:

```text
N=52,428,800
torch.sum:  0.2926 ms, 716.84 GB/s
mixed:      0.2855 ms, 734.65 GB/s
two_stage:  0.2844 ms, 737.47 GB/s
```

## Environment

Remote server:

```text
Host: pink
GPU: 2 x NVIDIA GeForce RTX 4090, 24 GB
Container: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
PyTorch: 2.5.1+cu121
Triton: 3.1.0
```

Open the container:

```bash
ssh pink
cd ~/gpu-mode/vector-sum-reduction
docker start -ai gpumode-vector-sum
```

## Why This Exists

The goal is not just to submit kernels. The goal is to build MLsys intuition:

- what an operator is,
- how a tensor operation maps to GPU kernels,
- when a workload is memory-bandwidth-bound,
- how to benchmark and tune kernels,
- how to explain performance tradeoffs for interviews.
