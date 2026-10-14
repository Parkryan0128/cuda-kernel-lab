#include "cuda_lab/ops.hpp"
#include "cuda_lab/cuda_check.hpp"
#include <algorithm>
#include <cstdint>

namespace cuda_lab {
namespace {
void check_config(const ReduceConfig& c) {
  if (c.variant == ReduceVariant::r2) {
    require((c.block_threads == 128 || c.block_threads == 256) &&
            (c.accumulator_count == 1 || c.accumulator_count == 4) && c.max_blocks > 0,
            "R2 requires 128/256 threads, 1/4 accumulators, and a positive grid cap");
  } else {
    require(c.variant == ReduceVariant::r0 || c.variant == ReduceVariant::r1, "unknown reduction variant");
    require(c.block_threads == 256 && c.accumulator_count == 1 && c.max_blocks == 0,
            "R0/R1 require 256 threads, one input per thread, and no grid cap");
  }
}
size_t first_block_count(size_t n, const ReduceConfig& c) {
  const size_t blocks = ceil_div(n, static_cast<size_t>(c.block_threads));
  return c.variant == ReduceVariant::r2 ? std::min(blocks, static_cast<size_t>(c.max_blocks)) : blocks;
}
__device__ float warp_sum(float value) {
  // Every launched block contains whole warps, and all lanes execute every shuffle.
  for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}
template<int Threads>
__device__ float block_sum(float value) {
  __shared__ float warp_sums[Threads / 32];
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  value = warp_sum(value);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  // Only the first warp participates in the second reduction; it still has all 32 lanes.
  if (warp == 0) value = warp_sum(lane < Threads / 32 ? warp_sums[lane] : 0.0f);
  return value;
}
__global__ void reduction_r1(const float* x, float* out, size_t n) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const float sum = block_sum<256>(index < n ? x[index] : 0.0f);
  if (threadIdx.x == 0) out[blockIdx.x] = sum;
}
template<int Threads, int Accumulators>
__global__ void reduction_r2(const float* x, float* out, size_t n) {
  const size_t stride = static_cast<size_t>(gridDim.x) * Threads;
  float sums[Accumulators] = {};
  for (size_t base = static_cast<size_t>(blockIdx.x) * Threads + threadIdx.x;
       base < n; base += Accumulators * stride) {
    #pragma unroll
    for (int a = 0; a < Accumulators; ++a) {
      const size_t index = base + a * stride;
      if (index < n) sums[a] += x[index];
    }
  }
  float sum = 0;
  #pragma unroll
  for (int a = 0; a < Accumulators; ++a) sum += sums[a];
  sum = block_sum<Threads>(sum);
  if (threadIdx.x == 0) out[blockIdx.x] = sum;
}
__global__ void reduction_r0(const float* x, float* out, size_t n) {
  __shared__ float values[256];
  const unsigned lane = threadIdx.x;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + lane;
  values[lane] = index < n ? x[index] : 0.0f;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1) {
    if (lane < stride) values[lane] += values[lane + stride];
    __syncthreads();
  }
  if (lane == 0) out[blockIdx.x] = values[0];
}
}
size_t reduce_sum_workspace_bytes(size_t n, const ReduceConfig& c) {
  check_config(c);
  checked_mul(n, sizeof(float));
  const size_t blocks = first_block_count(n, c);
  return blocks <= 1 ? 0 : checked_mul(checked_mul(blocks, 2), sizeof(float));
}
int reduction_pass_count(size_t n, const ReduceConfig& c) {
  check_config(c);
  if (!n) return 0;
  int passes = 1; n = first_block_count(n, c);
  while (n > 1) { n = ceil_div(n, 256); ++passes; }
  return passes;
}
void reduce_sum(const float* x, float* out, size_t n, void* workspace, size_t bytes,
                cudaStream_t stream, const ReduceConfig& c) {
  const size_t needed = reduce_sum_workspace_bytes(n, c);
  require(out != nullptr, "reduction output is required, including for empty input");
  require(n == 0 || x != nullptr, "nonempty reduction requires input");
  require(!needed || (workspace && bytes >= needed), "reduction workspace too small");
  require(!needed || reinterpret_cast<uintptr_t>(workspace) % alignof(float) == 0,
          "reduction workspace must be float-aligned");
  if (!n) { CUDA_LAB_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream)); return; }
  const size_t first_blocks = first_block_count(n, c);
  validate_grid(first_blocks, 1, c.block_threads, 256 * sizeof(float));
  float* scratch = static_cast<float*>(workspace);
  const float* input = x;
  int pass = 0;
  do {
    const size_t blocks = pass == 0 ? first_blocks : ceil_div(n, 256);
    float* output = blocks == 1 ? out : scratch + (pass % 2) * first_blocks;
    const unsigned grid = static_cast<unsigned>(blocks);
    if (c.variant == ReduceVariant::r0) reduction_r0<<<grid, 256, 0, stream>>>(input, output, n);
    else if (c.variant == ReduceVariant::r1 || pass > 0) reduction_r1<<<grid, 256, 0, stream>>>(input, output, n);
    else if (c.block_threads == 128 && c.accumulator_count == 1) reduction_r2<128,1><<<grid,128,0,stream>>>(input,output,n);
    else if (c.block_threads == 128) reduction_r2<128,4><<<grid,128,0,stream>>>(input,output,n);
    else if (c.accumulator_count == 1) reduction_r2<256,1><<<grid,256,0,stream>>>(input,output,n);
    else reduction_r2<256,4><<<grid,256,0,stream>>>(input,output,n);
    CUDA_LAB_CHECK(cudaGetLastError());
    input = output; n = blocks; ++pass;
  } while (n > 1);
}
}
