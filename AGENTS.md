# AGENTS.md

This repository is a public MLsys learning lab. It is also an Obsidian vault.

## Current Tracks

### Track 1: GPU Kernel Optimization

Focus: custom operators, Triton/CUDA kernels, bandwidth analysis, and profiling.

Owned areas:

- `projects/gpu-mode-*/`
- `notes/vector-sum-reduction.md`
- `notes/triton-basics.md`
- `notes/gpu-kernel-optimization.md`
- `experiments/vectorsum-*.md`

### Track 2: LLM Inference Systems

Focus: inference serving frameworks, KV cache, batching, scheduling, latency/throughput tradeoffs, and benchmark methodology.

Owned areas:

- `projects/llm-inference-benchmark-lab/`
- `notes/llm-inference-systems.md`
- `notes/kv-cache.md`
- `notes/batching-and-scheduling.md`
- `notes/serving-benchmarking.md`
- `experiments/inference-*.md`
- `templates/inference-experiment.md`

## Shared Files

These files are shared by both tracks. Edit them only for navigation, status, or clear cross-track context:

- `README.md`
- `START_HERE.md`
- `AGENTS.md`
- `notes/mlsys-learning-hub.md`
- `daily/YYYY-MM-DD.md`

When both tracks are active on the same day, split daily notes into:

```md
## Inference Track

## Kernel Track
```

## Agent Handoff Rules

Before working, read:

1. `README.md`
2. `START_HERE.md`
3. The relevant project README under `projects/`
4. The latest daily log under `daily/`
5. The relevant concept note under `notes/`

For inference work, start with:

- `projects/llm-inference-benchmark-lab/README.md`
- `notes/llm-inference-systems.md`
- `notes/serving-benchmarking.md`

## Hardware Context

Primary server:

```text
Host: pink
GPU: 2 x NVIDIA GeForce RTX 4090, 24 GB each
```

Use `pink` first. Renting GPUs is only needed for models or experiments that do not fit 2 x RTX 4090, or for data-center GPU comparisons.

## Git Hygiene

- Keep inference and kernel work on separate branches or worktrees when possible.
- Do not commit model weights, large datasets, large logs, generated caches, or benchmark dumps.
- Commit small learning milestones with descriptive messages.
- Keep public notes readable from GitHub, even when they also use Obsidian wiki links.

## Experiment Record Rule

Every meaningful experiment should have:

- The command that was run.
- Machine and GPU information.
- Framework and model versions.
- The measured metrics.
- A short interpretation of what changed and why it might have changed.
- The next concrete step.
