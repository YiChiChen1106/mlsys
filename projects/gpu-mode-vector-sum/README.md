# GPU MODE Vector Sum Reduction

This project studies vector sum reduction as a first GPU kernel optimization exercise for ML systems.

## Problem

```text
input:  x.shape = (N,)
output: scalar = sum(x)
```

Benchmark sizes:

```text
1,638,400
3,276,800
6,553,600
13,107,200
26,214,400
52,428,800
```

## Files

- `baseline.py`: PyTorch `torch.sum` baseline.
- `vector_sum_triton.py`: Triton partial-sum reduction with PyTorch final sum.
- `sweep_block_size.py`: `BLOCK_SIZE` sweep.
- `vector_sum_two_stage.py`: Triton partial sum plus Triton final sum.
- `compare_versions.py`: compares baseline, mixed, and full two-stage versions.

## Run

On `pink`:

```bash
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

## Best Result So Far

On RTX 4090:

```text
N=52,428,800
torch.sum      0.2926 ms  716.84 GB/s
mixed          0.2855 ms  734.65 GB/s
two_stage      0.2844 ms  737.47 GB/s
```

## What This Teaches

- Vector sum reduction is memory-bandwidth-bound.
- Partial sums expose parallelism.
- `BLOCK_SIZE` affects program count, second-stage cost, and occupancy.
- Parameter sweep is necessary because the best setting depends on shape and GPU.
- Moving the second stage from PyTorch to Triton helps, but only a little, because the second stage is tiny.

## Interview Summary

```text
I implemented and benchmarked custom Triton reduction kernels for vector sum. Starting from a PyTorch baseline, I used partial sums, block-size sweep, and a full two-stage Triton reduction to improve effective memory bandwidth on RTX 4090 from about 717 GB/s to about 737 GB/s on the largest input.
```
