# MLsys Learning Lab

This repository is a public learning lab for ML systems, GPU kernels, and inference performance engineering. It is also an Obsidian vault, so notes and code evolve together.

The lab currently has two tracks:

- GPU kernel optimization through small, measurable operators.
- LLM inference systems through serving benchmarks and framework experiments.

## Current Status

- Track 1: GPU kernel optimization
- Track 2: LLM inference systems
- Hardware used so far: `pink` with dual RTX 4090
- Kernel result so far: `two_stage` vector sum reached about `737 GB/s` on the largest benchmark
- Inference status: project scaffold created; first target is a vLLM baseline on `pink`

## Start Here

- [[START_HERE]]
- [[notes/mlsys-learning-hub]]
- [[notes/vector-sum-reduction]]
- [[notes/llm-inference-systems]]

## What Is In This Repo

```text
notes/        concepts and learning notes
daily/        daily logs
experiments/  benchmark writeups
projects/     runnable code
templates/    reusable note templates
```

## Projects

### Track 1: GPU MODE Vector Sum

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

### Track 2: LLM Inference Benchmark Lab

Path: `projects/llm-inference-benchmark-lab`

Learning path:

1. Run a vLLM OpenAI-compatible server on `pink`.
2. Measure TTFT, TPOT, throughput, memory, and failure rate.
3. Sweep concurrency, prompt length, output length, and GPU count.
4. Repeat with SGLang and llama.cpp.
5. Explain serving behavior in terms of scheduling, KV cache, and GPU memory pressure.

First target:

```text
vLLM baseline on 1 x RTX 4090, then 2 x RTX 4090
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

The goal is not just to submit kernels or run model servers. The goal is to build MLsys intuition:

- what an operator is,
- how a tensor operation maps to GPU kernels,
- when a workload is memory-bandwidth-bound,
- how to benchmark and tune kernels,
- how LLM serving frameworks trade latency, throughput, and memory,
- how KV cache and batching shape inference performance,
- how to explain performance tradeoffs for interviews.
