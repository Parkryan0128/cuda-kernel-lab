#include "cuda_lab/ops.hpp"
#include "cuda_lab/cuda_check.hpp"

namespace cuda_lab {
namespace {
template<bool AddInput>
__global__ void epilogue(float* y, const float* bias, size_t count, int n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float value = (AddInput ? y[i] : 0.0f) + (bias ? bias[i % n] : 0.0f);
    y[i] = bias ? fmaxf(value, 0.0f) : value;
  }
}
template<bool AddInput>
void launch(float* y, const float* bias, int m, int n, cudaStream_t stream) {
  require(m >= 0 && n >= 0, "epilogue dimensions must be nonnegative");
  const size_t count = checked_mul(static_cast<size_t>(m), n);
  checked_mul(count, sizeof(float));
  if (!count) return;
  require(y != nullptr, "epilogue output is required");
  if (AddInput) require(bias != nullptr, "bias is required");
  const size_t blocks = ceil_div(count, 256);
  validate_grid(blocks, 1, 256);
  epilogue<AddInput><<<static_cast<unsigned>(blocks), 256, 0, stream>>>(y, bias, count, n);
  CUDA_LAB_CHECK(cudaGetLastError());
}
}
void bias_relu_inplace(float* y, const float* bias, int m, int n, cudaStream_t stream) {
  launch<true>(y, bias, m, n, stream);
}
void fill_zero_or_bias(float* y, const float* bias, int m, int n, cudaStream_t stream) {
  launch<false>(y, bias, m, n, stream);
}
}
