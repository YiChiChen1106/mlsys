# vectorsum_v2 A100 First Submission

## Goal

Get a correct first submission onto the GPU MODE `vectorsum_v2` A100 leaderboard.

## Submission

Path:

```text
projects/gpu-mode-vectorsum-v2/submission.py
```

Approach:

```text
x[N]
-> Triton partial sums with BLOCK_SIZE=8192
-> Triton final sum with FINAL_BLOCK_SIZE=8192
-> output[0]
```

The submission uses a module-level partial-sum cache so repeated benchmark calls reuse the intermediate buffer for the same shape.

## Local 4090 Correctness

Command:

```bash
docker run --gpus all --rm \
  -v /mnt/hdd/users/cychi/gpu-mode/vectorsum-v2-a100:/workspace \
  -w /workspace \
  pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel \
  python validate_submission.py
```

Result:

```text
correctness size=1023 pass
correctness size=1024 pass
correctness size=1025 pass
correctness size=2048 pass
correctness size=4096 pass
```

## Local 4090 Benchmark

Command:

```bash
python validate_submission.py --benchmark
```

Result:

```text
benchmark size=1638400 0.0171 ms 384.07 GB/s
benchmark size=3276800 0.0265 ms 495.17 GB/s
benchmark size=6553600 0.0455 ms 576.36 GB/s
benchmark size=13107200 0.0840 ms 624.08 GB/s
benchmark size=26214400 0.1602 ms 654.49 GB/s
benchmark size=52428800 0.2843 ms 737.74 GB/s
```

## A100 Results

Status:

```text
Blocked on Popcorn CLI GitHub authorization.
```

Next commands after authorization:

```bash
popcorn-cli submit --no-tui --leaderboard vectorsum_v2 --gpu A100 --mode test submission.py
popcorn-cli submit --no-tui --leaderboard vectorsum_v2 --gpu A100 --mode benchmark --output benchmark-a100.json submission.py
popcorn-cli submit --no-tui --leaderboard vectorsum_v2 --gpu A100 --mode leaderboard --output leaderboard-a100.json submission.py
```

## Next Step

Complete GitHub OAuth for the current Popcorn CLI `cli_id`, then submit to A100.
