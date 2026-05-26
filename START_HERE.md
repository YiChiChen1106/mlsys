# Start Here

This is the entry point for the MLsys learning project.

## Current Track

GPU kernel optimization for ML systems.

Current project:

```text
projects/gpu-mode-vector-sum
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

## Daily Routine

1. Open the Obsidian vault at `F:\codex_workspace\mlsys`.
2. Read [[notes/mlsys-learning-hub]].
3. Continue the current project note: [[notes/vector-sum-reduction]].
4. Run one small experiment.
5. Record the output under `experiments/`.

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
