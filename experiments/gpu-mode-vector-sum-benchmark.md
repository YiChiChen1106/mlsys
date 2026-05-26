# GPU MODE Vector Sum Benchmark

## Environment

```text
Host: pink
GPU: NVIDIA GeForce RTX 4090
Docker image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
PyTorch: 2.5.1+cu121
Triton: 3.1.0
```

## Commands

```bash
python baseline.py
python vector_sum_triton.py
python sweep_block_size.py
python vector_sum_two_stage.py
python compare_versions.py
```

## Baseline Results

```text
N=  1638400  torch.sum=0.0175 ms  bandwidth=374.37 GB/s
N=  3276800  torch.sum=0.0270 ms  bandwidth=485.60 GB/s
N=  6553600  torch.sum=0.0466 ms  bandwidth=562.94 GB/s
N= 13107200  torch.sum=0.0849 ms  bandwidth=617.69 GB/s
N= 26214400  torch.sum=0.1611 ms  bandwidth=650.76 GB/s
N= 52428800  torch.sum=0.2926 ms  bandwidth=716.62 GB/s
```

## Block Size Sweep Takeaway

`BLOCK_SIZE=8192` was generally best or close to best on RTX 4090.

Largest benchmark:

```text
N=52428800
torch.sum    0.2924 ms  717.12 GB/s
block=512    0.2950 ms  710.85 GB/s
block=1024   0.2897 ms  723.99 GB/s
block=2048   0.2871 ms  730.39 GB/s
block=4096   0.2859 ms  733.62 GB/s
block=8192   0.2855 ms  734.59 GB/s
```

## Version Comparison

```text
N=52428800
torch.sum      0.2926 ms  716.84 GB/s
mixed          0.2855 ms  734.65 GB/s
two_stage      0.2844 ms  737.47 GB/s
```

## Interpretation

The two-stage Triton version is the best observed version, but it only improves slightly over the mixed version because the second stage operates on partial sums, not the full input.

For the largest input with `BLOCK_SIZE=8192`:

```text
input elements: 52,428,800
partial sums: 6,400
```

Therefore, the main bottleneck remains the first-stage read of the original tensor.
