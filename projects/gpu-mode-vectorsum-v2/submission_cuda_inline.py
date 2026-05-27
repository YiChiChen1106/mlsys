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


_CPP_SOURCE = r"""
#include <torch/extension.h>

void vector_sum_stage1(torch::Tensor x, torch::Tensor partial, int64_t n_elements, uint64_t stream);
void vector_sum_stage2(torch::Tensor partial, torch::Tensor output, int64_t num_blocks, uint64_t stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stage1", &vector_sum_stage1, "vector sum partial reduction");
    m.def("stage2", &vector_sum_stage2, "vector sum final reduction");
}
"""


_CUDA_SOURCE = f"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

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

void vector_sum_stage1(torch::Tensor x, torch::Tensor partial, int64_t n_elements, uint64_t stream) {{
    const int64_t num_blocks = partial.size(0);
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    partial_sum_kernel<<<num_blocks, THREADS, 0, cuda_stream>>>(
        x.data_ptr<float>(),
        partial.data_ptr<float>(),
        n_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}

void vector_sum_stage2(torch::Tensor partial, torch::Tensor output, int64_t num_blocks, uint64_t stream) {{
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    final_sum_kernel<<<1, THREADS, 0, cuda_stream>>>(
        partial.data_ptr<float>(),
        output.data_ptr<float>(),
        num_blocks
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
"""


_CUDA_MODULE = load_inline(
    name="vectorsum_cuda_inline_v1",
    cpp_sources=_CPP_SOURCE,
    cuda_sources=_CUDA_SOURCE,
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
    stream = torch.cuda.current_stream(x.device).cuda_stream

    _CUDA_MODULE.stage1(x, partial, n_elements, stream)
    _CUDA_MODULE.stage2(partial, output, num_blocks, stream)
    return output[0]
