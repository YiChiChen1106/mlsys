# vectorsum_v2 A100 BLOCK_SIZE Sweep

## Goal

Check whether changing `BLOCK_SIZE` can improve the current A100 rank 3 submission.

Current leaderboard baseline:

```text
BLOCK_SIZE = 8192
FINAL_BLOCK_SIZE = 8192
score = 137.626 us
rank = 3
submission_id = 783079
```

## Candidates

Largest official benchmark:

```text
N = 52,428,800
```

| BLOCK_SIZE | num_blocks | FINAL_BLOCK_SIZE |
| ---: | ---: | ---: |
| 4096 | 12800 | 16384 |
| 8192 | 6400 | 8192 |
| 16384 | 3200 | 4096 |
| 32768 | 1600 | 2048 |

## Local Correctness

All generated variants passed official small-size correctness on `pink-outer` RTX 4090:

```text
sizes: 1023, 1024, 1025, 2048, 4096
result: all pass
```

## A100 Benchmark Mode Results

Command pattern:

```bash
popcorn-cli submit --no-tui \
  --leaderboard vectorsum_v2 \
  --gpu A100 \
  --mode benchmark \
  --output outputs/block-size-sweep/benchmark-bsXXXX-a100.json \
  outputs/block-size-sweep/submission_bsXXXX.py
```

Results:

| BLOCK_SIZE | mean | best | worst |
| ---: | ---: | ---: | ---: |
| 4096 | 145 us | 139 us | 148 us |
| 8192 | 144 us | 137 us | 150 us |
| 16384 | 147 us | 138 us | 153 us |
| 32768 | 147 us | 140 us | 174 us |

## Conclusion

`BLOCK_SIZE=8192` remains the best of this first sweep. It has the lowest mean and best sample among the tested values.

No new leaderboard submission was made from this sweep because none of the variants clearly beat the existing rank 3 result.

## Next Step

The remaining gap to rank 1 is too small for coarse `BLOCK_SIZE` sweep alone. Next candidates:

- sweep `num_warps` while keeping `BLOCK_SIZE=8192`,
- compare two-stage Triton against an atomic-add version,
- try CUDA inline if Triton tuning plateaus.
