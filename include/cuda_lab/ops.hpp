#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>

namespace cuda_lab {
enum class ReduceVariant { r0, r1, r2 };
struct ReduceConfig {
  ReduceVariant variant = ReduceVariant::r0;
  int block_threads = 256;
  int accumulator_count = 1;
  int max_blocks = 0; // R2 only; resolved by the caller from SM count.
};
enum class GemmVariant { g0, g1, g2 };
struct GemmConfig {
  GemmVariant variant = GemmVariant::g0;
  int bm = 16, bn = 16, bk = 16;
};
size_t reduce_sum_workspace_bytes(size_t n, const ReduceConfig& config);
int reduction_pass_count(size_t n, const ReduceConfig& config);
void reduce_sum(const float* x, float* out, size_t n, void* workspace, size_t bytes,
                cudaStream_t stream, const ReduceConfig& config);
void gemm(const float* a, const float* b, float* c, int m, int n, int k,
          cudaStream_t stream, const GemmConfig& config);
void gemm_bias_relu(const float* a, const float* b, const float* bias, float* y,
                    int m, int n, int k, cudaStream_t stream, const GemmConfig& config);
void bias_relu_inplace(float* y, const float* bias, int m, int n, cudaStream_t stream);
void fill_zero_or_bias(float* y, const float* bias, int m, int n, cudaStream_t stream);
size_t cub_workspace_bytes(size_t n, cudaStream_t stream);
void cub_sum(const float* x, float* out, size_t n, void* workspace, size_t bytes, cudaStream_t stream);
class BlasHandle {
 public:
  explicit BlasHandle(cudaStream_t stream);
  ~BlasHandle();
  BlasHandle(const BlasHandle&) = delete;
  BlasHandle& operator=(const BlasHandle&) = delete;
  void gemm(const float* a, const float* b, float* c, int m, int n, int k);
  int version() const;
  static constexpr size_t workspace_bytes = 4 * 1024 * 1024;
 private:
  cublasHandle_t handle_ = nullptr;
  cudaStream_t stream_ = nullptr;
  void* workspace_ = nullptr;
};
}
