# Start Here

This is the entry point for the MLsys learning project.

## Current Tracks

Two MLsys tracks are active:

```text
Track 1: GPU kernel optimization
Track 2: LLM inference systems
```

Current kernel project:

```text
projects/gpu-mode-vector-sum
```

Current inference project:

```text
projects/llm-inference-benchmark-lab
```

Current concept stack:

```text
PyTorch baseline
-> Triton kernel
-> partial sums
-> block size sweep
-> two-stage reduction
-> memory bandwidth analysis
```

Inference concept stack:

```text
OpenAI-compatible serving
-> TTFT / TPOT / throughput
-> prefill vs decode
-> KV cache
-> batching and scheduling
-> framework comparison
```

## Daily Routine

1. Open the Obsidian vault at `F:\codex_workspace\mlsys`.
2. Read [[notes/mlsys-learning-hub]].
3. Pick one active track for the session.
4. For kernel work, continue [[notes/vector-sum-reduction]].
5. For inference work, continue [[notes/llm-inference-systems]] and `projects/llm-inference-benchmark-lab/README.md`.
6. Run one small experiment.
7. Record the output under `experiments/`.

## Remote GPU Environment

```bash
ssh pink
cd ~/gpu-mode/vector-sum-reduction
docker start -ai gpumode-vector-sum
```

Inside the container:

```bash
python baseline.py
python vector_sum_triton.py
python sweep_block_size.py
python vector_sum_two_stage.py
python compare_versions.py
```

## Learning Goal

Build a project portfolio for ML systems and GPU performance engineering:

```text
Implemented and benchmarked custom Triton reduction kernels;
optimized memory bandwidth using partial sums, block-size sweep, and two-stage reduction.
```

Build a second portfolio thread for AI infra inference systems:

```text
Built and benchmarked an LLM inference serving lab;
measured vLLM/SGLang/llama.cpp latency, throughput, and memory behavior on dual RTX 4090 GPUs.
```

## Current Next Step

Inference track:

```text
ssh pink
nvidia-smi
```

Then inspect Docker/Python availability and choose the first vLLM environment.
