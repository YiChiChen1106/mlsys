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
FINAL_BLOCK_SIZE = 8192
_partial_cache: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}


@triton.jit
def _partial_sum_kernel(x_ptr, partial_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    output_ptr,
    num_blocks: tl.constexpr,
    FINAL_BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, FINAL_BLOCK_SIZE)
    mask = offsets < num_blocks
    values = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    final = tl.sum(values, axis=0)
    tl.store(output_ptr, final)


def _device_index(x: torch.Tensor) -> int:
    if x.device.index is not None:
        return x.device.index
    return torch.cuda.current_device()


def _partial_buffer(x: torch.Tensor, num_blocks: int) -> torch.Tensor:
    key = (_device_index(x), num_blocks, torch.float32)
    partial = _partial_cache.get(key)
    if partial is None or partial.device != x.device:
        partial = torch.empty((num_blocks,), device=x.device, dtype=torch.float32)
        _partial_cache[key] = partial
    return partial


def custom_kernel(data: input_t) -> output_t:
    x, output = data
    n_elements = x.numel()
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial = _partial_buffer(x, num_blocks)

    _partial_sum_kernel[(num_blocks,)](
        x,
        partial,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    _final_sum_kernel[(1,)](
        partial,
        output,
        num_blocks,
        FINAL_BLOCK_SIZE=FINAL_BLOCK_SIZE,
    )
    return output[0]
