# vectorsum_v2 A100 num_warps and Atomic-Add Experiments

## Goal

Try the next tuning levers after the coarse `BLOCK_SIZE` sweep:

- keep `BLOCK_SIZE=8192`,
- sweep partial-stage `num_warps`,
- compare against an atomic-add reduction.

Current official baseline:

```text
rank: 3
submission_id: 783079
score: 137.626 us
```

## Candidates

First `num_warps` sweep:

| File | Partial num_warps | Final num_warps |
| --- | ---: | ---: |
| `submission_wp4_wf8.py` | 4 | 8 |
| `submission_wp8_wf8.py` | 8 | 8 |
| `submission_wp16_wf8.py` | 16 | 8 |

Atomic-add comparison:

| File | Strategy |
| --- | --- |
| `submission_atomic.py` | zero `output[0]`, then atomic-add one partial sum per program |

## Local Correctness

All four candidates passed small-size correctness on `pink-outer` RTX 4090:

```text
sizes: 1023, 1024, 1025, 2048, 4096
result: all pass
```

## RTX 4090 Sanity Benchmark

These numbers are only sanity checks. A100 ranking still requires Popcorn benchmark mode.

| Candidate | 1,638,400 | 3,276,800 | 6,553,600 | 13,107,200 | 26,214,400 | 52,428,800 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wp16_wf8` | 0.0170 ms | 0.0264 ms | 0.0454 ms | 0.0836 ms | 0.1604 ms | 0.2838 ms |
| `wp4_wf8` | 0.0172 ms | 0.0265 ms | 0.0457 ms | 0.0839 ms | 0.1603 ms | 0.2842 ms |
| `wp8_wf8` | 0.0169 ms | 0.0264 ms | 0.0454 ms | 0.0837 ms | 0.1603 ms | 0.2841 ms |
| `atomic` | 0.0166 ms | 0.0264 ms | 0.0457 ms | 0.0835 ms | 0.1599 ms | 0.2830 ms |

## A100 Status

The first Popcorn A100 benchmark attempt hit the hourly submission limit:

```text
Rate limit exceeded: 6/6 test submissions per hour. Try again in 1703s.
```

No new leaderboard submission has been made from these variants yet.

## Interpretation

`num_warps` controls how many warps cooperate inside one Triton program. For this reduction, it mostly affects the per-block reduction tree and scheduling, while `BLOCK_SIZE` controls how much data each program reads.

The atomic-add version is a useful comparison because it removes the cached partial buffer and final reduction. The tradeoff is that all programs update the same scalar output, so atomic contention may dominate on A100 even if the 4090 sanity benchmark looks competitive.

## Next Step

After the Popcorn limit resets:

```bash
python sweep_a100_num_warps.py \
  --partial-warps 4,8,16 \
  --final-warps 8 \
  --output-dir outputs/num-warps-sweep-first \
  --run

popcorn-cli submit --no-tui \
  --leaderboard vectorsum_v2 \
  --gpu A100 \
  --mode benchmark \
  --output outputs/atomic-a100.json \
  submission_atomic.py
```

Only submit to `mode=leaderboard` if a benchmark result clearly beats the current `137.626 us` baseline.
