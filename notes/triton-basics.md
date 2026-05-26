# Triton Basics

## One Sentence

Triton lets us write GPU kernels in Python-like code while still expressing parallel GPU work.

## Core Ideas

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
partial = tl.sum(values, axis=0)
tl.store(partial_ptr + pid, partial)
```

Meaning:

- `tl.program_id(0)` identifies which program is running.
- `tl.arange(0, BLOCK_SIZE)` creates vector offsets inside one program.
- `mask` prevents out-of-bounds loads.
- `other=0.0` uses the neutral value for sum.
- `tl.sum` reduces the vector inside one program.

## Important Habit

Always check correctness against PyTorch first:

```python
expected = torch.sum(x)
actual = vector_sum(x)
diff = (expected - actual).abs().item()
```

Small float32 differences are normal because reduction order changes.
