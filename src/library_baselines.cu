#include "cuda_lab/ops.hpp"
#include "cuda_lab/cuda_check.hpp"
#include <cub/cub.cuh>
#include <limits>

namespace cuda_lab {
size_t cub_workspace_bytes(size_t n, cudaStream_t stream) {
  checked_mul(n, sizeof(float));
  require(n <= static_cast<size_t>(std::numeric_limits<int>::max()), "CUB adapter supports int-sized item counts");
  if (!n) return 0;
  size_t bytes = 0;
  CUDA_LAB_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, static_cast<const float*>(nullptr),
                                      static_cast<float*>(nullptr), static_cast<int>(n), stream));
  return bytes;
}
void cub_sum(const float* x, float* out, size_t n, void* workspace, size_t bytes, cudaStream_t stream) {
  checked_mul(n, sizeof(float));
  require(n <= static_cast<size_t>(std::numeric_limits<int>::max()), "CUB adapter supports int-sized item counts");
  require(out != nullptr && (!n || x), "CUB input/output missing");
  if (!n) { CUDA_LAB_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream)); return; }
  require(workspace && bytes > 0, "CUB workspace is required; query capacity before launch");
  CUDA_LAB_CHECK(cub::DeviceReduce::Sum(workspace, bytes, x, out, static_cast<int>(n), stream));
}
BlasHandle::BlasHandle(cudaStream_t stream) : stream_(stream) {
  CUDA_LAB_BLAS(cublasCreate(&handle_));
  try {
    CUDA_LAB_BLAS(cublasSetStream(handle_, stream));
    CUDA_LAB_CHECK(cudaMalloc(&workspace_, workspace_bytes));
    CUDA_LAB_BLAS(cublasSetWorkspace(handle_, workspace_, workspace_bytes));
    CUDA_LAB_BLAS(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH));
    CUDA_LAB_BLAS(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
    CUDA_LAB_BLAS(cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED));
  } catch (...) { cublasDestroy(handle_); if (workspace_) cudaFree(workspace_); handle_ = nullptr; throw; }
}
BlasHandle::~BlasHandle() { if (handle_) cublasDestroy(handle_); if (workspace_) cudaFree(workspace_); }
int BlasHandle::version() const { int v = 0; CUDA_LAB_BLAS(cublasGetVersion(handle_, &v)); return v; }
void BlasHandle::gemm(const float* a, const float* b, float* c, int m, int n, int k) {
  require(m >= 0 && n >= 0 && k >= 0, "cuBLAS dimensions must be nonnegative");
  checked_mul(checked_mul(static_cast<size_t>(m), n), sizeof(float));
  checked_mul(checked_mul(static_cast<size_t>(m), k), sizeof(float));
  checked_mul(checked_mul(static_cast<size_t>(k), n), sizeof(float));
  if (!m || !n) return;
  require(c != nullptr, "cuBLAS output is required");
  if (!k) { fill_zero_or_bias(c, nullptr, m, n, stream_); return; }
  require(a && b, "cuBLAS inputs are required");
  const float alpha = 1.0f, beta = 0.0f;
  // A_rm B_rm = C_rm <=> B_cm A_cm = C_cm, with transposed dimensions.
  CUDA_LAB_BLAS(cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
      &alpha, b, CUDA_R_32F, n, a, CUDA_R_32F, k, &beta, c, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
}
}
