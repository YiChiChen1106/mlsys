#!POPCORN leaderboard vectorsum_v2
#!POPCORN gpu A100

from __future__ import annotations

from typing import Any

import torch
import triton
from torch.utils.cpp_extension import load_inline

try:
    from task import input_t, output_t
except Exception:
    input_t = Any
    output_t = torch.Tensor


THREADS = 256
ITEMS_PER_THREAD = 32
ELEMENTS_PER_BLOCK = THREADS * ITEMS_PER_THREAD
_partial_cache: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}


_CUDA_SOURCE = f"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace {{

constexpr int THREADS = {THREADS};
constexpr int ITEMS_PER_THREAD = {ITEMS_PER_THREAD};

__inline__ __device__ float warp_reduce_sum(float value) {{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {{
        value += __shfl_down_sync(0xffffffff, value, offset);
    }}
    return value;
}}

__inline__ __device__ float block_reduce_sum(float value) {{
    __shared__ float warp_sums[32];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    value = warp_reduce_sum(value);
    if (lane == 0) {{
        warp_sums[warp_id] = value;
    }}
    __syncthreads();

    value = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    if (warp_id == 0) {{
        value = warp_reduce_sum(value);
    }}
    return value;
}}

__global__ void partial_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ partial,
    int64_t n_elements
) {{
    const int tid = threadIdx.x;
    const int64_t block_start = static_cast<int64_t>(blockIdx.x) * THREADS * ITEMS_PER_THREAD;
    float local_sum = 0.0f;

    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {{
        const int64_t offset = block_start + tid + static_cast<int64_t>(item) * THREADS;
        if (offset < n_elements) {{
            local_sum += x[offset];
        }}
    }}

    const float block_sum = block_reduce_sum(local_sum);
    if (tid == 0) {{
        partial[blockIdx.x] = block_sum;
    }}
}}

__global__ void final_sum_kernel(
    const float* __restrict__ partial,
    float* __restrict__ output,
    int64_t num_blocks
) {{
    float local_sum = 0.0f;

    for (int64_t offset = threadIdx.x; offset < num_blocks; offset += THREADS) {{
        local_sum += partial[offset];
    }}

    const float final_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {{
        output[0] = final_sum;
    }}
}}

}}  // namespace

torch::Tensor vector_sum_stage1(torch::Tensor x, torch::Tensor partial, int64_t n_elements) {{
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(partial.device().is_cuda(), "partial must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(partial.scalar_type() == torch::kFloat32, "partial must be float32");

    const int64_t num_blocks = partial.size(0);
    partial_sum_kernel<<<num_blocks, THREADS>>>(
        x.data_ptr<float>(),
        partial.data_ptr<float>(),
        n_elements
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {{
        throw std::runtime_error(cudaGetErrorString(err));
    }}

    return partial;
}}

torch::Tensor vector_sum_stage2(torch::Tensor partial, torch::Tensor output, int64_t num_blocks) {{
    TORCH_CHECK(partial.device().is_cuda(), "partial must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(partial.scalar_type() == torch::kFloat32, "partial must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");

    final_sum_kernel<<<1, THREADS>>>(
        partial.data_ptr<float>(),
        output.data_ptr<float>(),
        num_blocks
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {{
        throw std::runtime_error(cudaGetErrorString(err));
    }}

    return output;
}}
"""


_CPP_SOURCE = """
#include <torch/extension.h>

torch::Tensor vector_sum_stage1(torch::Tensor x, torch::Tensor partial, int64_t n_elements);
torch::Tensor vector_sum_stage2(torch::Tensor partial, torch::Tensor output, int64_t num_blocks);
"""


_CUDA_MODULE = load_inline(
    name="vectorsum_cuda_inline_v2",
    cpp_sources=_CPP_SOURCE,
    cuda_sources=_CUDA_SOURCE,
    functions=["vector_sum_stage1", "vector_sum_stage2"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


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
    num_blocks = triton.cdiv(n_elements, ELEMENTS_PER_BLOCK)
    partial = _partial_buffer(x, num_blocks)

    _CUDA_MODULE.vector_sum_stage1(x, partial, n_elements)
    _CUDA_MODULE.vector_sum_stage2(partial, output, num_blocks)
    return output[0]
