# GPU MODE vectorsum_v2 A100

Single-file submission workspace for the GPU MODE `vectorsum_v2` leaderboard.

## Goal

Submit a correct first A100 entry, then iterate from real leaderboard feedback.

## Files

- `submission.py`: Popcorn-compatible Triton two-stage reduction.
- `validate_submission.py`: Local correctness and sanity benchmark script.

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
