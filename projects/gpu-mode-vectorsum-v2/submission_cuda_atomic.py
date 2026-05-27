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

__global__ void zero_output_kernel(float* __restrict__ output) {{
    output[0] = 0.0f;
}}

__global__ void atomic_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
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
        atomicAdd(output, block_sum);
    }}
}}

}}  // namespace

torch::Tensor vector_sum_atomic(torch::Tensor x, torch::Tensor output, int64_t n_elements, int64_t num_blocks) {{
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");

    zero_output_kernel<<<1, 1>>>(output.data_ptr<float>());
    atomic_sum_kernel<<<num_blocks, THREADS>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements
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

torch::Tensor vector_sum_atomic(torch::Tensor x, torch::Tensor output, int64_t n_elements, int64_t num_blocks);
"""


_CUDA_MODULE = load_inline(
    name="vectorsum_cuda_atomic_t256_i32",
    cpp_sources=_CPP_SOURCE,
    cuda_sources=_CUDA_SOURCE,
    functions=["vector_sum_atomic"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    x, output = data
    n_elements = x.numel()
    num_blocks = triton.cdiv(n_elements, ELEMENTS_PER_BLOCK)

    _CUDA_MODULE.vector_sum_atomic(x, output, n_elements, num_blocks)
    return output[0]
