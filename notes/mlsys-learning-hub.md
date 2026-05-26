# MLsys Learning Hub

## Current Focus

MLsys is the layer between model math and real hardware.

For this project, the focus is:

```text
tensor operation -> GPU kernel -> benchmark -> profiling -> optimization
```

## Track 1: GPU Kernels

Current project: [[vector-sum-reduction]]

Key questions:

- What is an operator?
- What is a GPU kernel?
- Why can two implementations of the same math have different latency?
- How do memory bandwidth and compute throughput differ?
- Why do we tune `BLOCK_SIZE`, `num_warps`, and reduction strategy?

## Track 2: LLM Inference Systems

Future topics:

- prefill vs decode
- KV cache
- batching
- attention kernel bottlenecks
- RMSNorm / LayerNorm / Softmax
- quantization and memory footprint

## Track 3: Performance Engineering

Core loop:

```text
baseline
-> correctness check
-> benchmark
-> bottleneck hypothesis
-> parameter sweep
-> compare against baseline
-> write down why
```

## Project Ladder

1. Vector sum reduction.
2. LayerNorm / RMSNorm.
3. Softmax.
4. Matmul.
5. Attention-related kernels.
6. LLM serving experiments.

## Interview Themes

- Explain bandwidth-bound vs compute-bound.
- Explain reduction and partial sums.
- Explain why kernel performance depends on shape and hardware.
- Explain how you benchmarked a custom Triton kernel against PyTorch.
