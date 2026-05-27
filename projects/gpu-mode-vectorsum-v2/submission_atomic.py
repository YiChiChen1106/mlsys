#!POPCORN leaderboard vectorsum_v2
#!POPCORN gpu A100

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

try:
    from task import input_t, output_t
except Exception:
    input_t = Any
    output_t = torch.Tensor


BLOCK_SIZE = 8192
NUM_WARPS = 8


@triton.jit
def _zero_output_kernel(output_ptr):
    tl.store(output_ptr, 0.0)


@triton.jit
def _atomic_sum_kernel(x_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.atomic_add(output_ptr, partial, sem="relaxed")


def custom_kernel(data: input_t) -> output_t:
    x, output = data
    n_elements = x.numel()
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    _zero_output_kernel[(1,)](output)
    _atomic_sum_kernel[(num_blocks,)](
        x,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    return output[0]
