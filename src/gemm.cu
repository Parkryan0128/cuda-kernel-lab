#include "cuda_lab/ops.hpp"
#include "cuda_lab/cuda_check.hpp"

namespace cuda_lab {
namespace {
template<int BM, int BN, int BK, bool Fused>
__global__ void gemm_g2(const float* a, const float* b, const float* bias, float* c, int m, int n, int k) {
  constexpr int Rows = BM / 16, Cols = BN / 16;
  __shared__ float tile_a[BM][BK], tile_b[BK][BN];
  const unsigned tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
  const size_t row_base = static_cast<size_t>(blockIdx.y) * BM;
  const size_t col_base = static_cast<size_t>(blockIdx.x) * BN;
  float sums[Rows][Cols] = {};
  for (size_t base = 0; base < static_cast<size_t>(k); base += BK) {
    // Linear cooperative loads are coalesced across each warp, including tile tails.
    for (unsigned i = tid; i < BM * BK; i += 256) {
      const unsigned row = i / BK, col = i % BK;
      tile_a[row][col] = row_base + row < static_cast<size_t>(m) && base + col < static_cast<size_t>(k)
                          ? a[(row_base + row) * k + base + col] : 0;
    }
    for (unsigned i = tid; i < BK * BN; i += 256) {
      const unsigned row = i / BN, col = i % BN;
      tile_b[row][col] = base + row < static_cast<size_t>(k) && col_base + col < static_cast<size_t>(n)
                          ? b[(base + row) * n + col_base + col] : 0;
    }
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < BK; ++t) {
      float av[Rows], bv[Cols];
      #pragma unroll
      for (int r = 0; r < Rows; ++r) av[r] = tile_a[ty + r * 16][t];
      #pragma unroll
      for (int col = 0; col < Cols; ++col) bv[col] = tile_b[t][tx + col * 16];
      #pragma unroll
      for (int r = 0; r < Rows; ++r) {
        #pragma unroll
        for (int col = 0; col < Cols; ++col) sums[r][col] = fmaf(av[r], bv[col], sums[r][col]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int r = 0; r < Rows; ++r) {
    const size_t row = row_base + ty + r * 16;
    #pragma unroll
    for (int col = 0; col < Cols; ++col) {
      const size_t column = col_base + tx + col * 16;
      if (row < static_cast<size_t>(m) && column < static_cast<size_t>(n)) {
        float value = sums[r][col];
        if constexpr (Fused) value = fmaxf(value + bias[column], 0.0f);
        c[row * n + column] = value;
      }
    }
  }
}
__global__ void gemm_g1(const float* a, const float* b, float* c, int m, int n, int k) {
  __shared__ float tile_a[16][16], tile_b[16][16];
  const unsigned tx = threadIdx.x, ty = threadIdx.y;
  const size_t row = static_cast<size_t>(blockIdx.y) * 16 + ty;
  const size_t col = static_cast<size_t>(blockIdx.x) * 16 + tx;
  float sum = 0;
  for (size_t base = 0; base < static_cast<size_t>(k); base += 16) {
    tile_a[ty][tx] = row < static_cast<size_t>(m) && base + tx < static_cast<size_t>(k) ? a[row * k + base + tx] : 0;
    tile_b[ty][tx] = base + ty < static_cast<size_t>(k) && col < static_cast<size_t>(n) ? b[(base + ty) * n + col] : 0;
    __syncthreads();
    #pragma unroll
    for (int t = 0; t < 16; ++t) sum = fmaf(tile_a[ty][t], tile_b[t][tx], sum);
    __syncthreads();
  }
  if (row < static_cast<size_t>(m) && col < static_cast<size_t>(n)) c[row * n + col] = sum;
}
__global__ void gemm_g0(const float* a, const float* b, float* c, int m, int n, int k) {
  const size_t row = static_cast<size_t>(blockIdx.y) * 16 + threadIdx.y;
  const size_t col = static_cast<size_t>(blockIdx.x) * 16 + threadIdx.x;
  if (row >= static_cast<size_t>(m) || col >= static_cast<size_t>(n)) return;
  float sum = 0;
  for (int t = 0; t < k; ++t) sum = fmaf(a[row * k + t], b[static_cast<size_t>(t) * n + col], sum);
  c[row * n + col] = sum;
}
void check_config(const GemmConfig& config) {
  if (config.variant == GemmVariant::g2) {
    require(config.bm == 64 && ((config.bn == 64 && (config.bk == 16 || config.bk == 32)) ||
                              (config.bn == 128 && config.bk == 16)), "unsupported G2 tile");
  } else {
    require((config.variant == GemmVariant::g0 || config.variant == GemmVariant::g1) &&
            config.bm == 16 && config.bn == 16 && config.bk == 16, "unsupported G0/G1 configuration");
  }
}
template<bool Fused>
void launch(const float* a, const float* b, const float* bias, float* c, int m, int n, int k,
            cudaStream_t stream, const GemmConfig& config) {
  check_config(config);
  if (Fused) require(config.variant == GemmVariant::g2, "fusion requires G2");
  require(m >= 0 && n >= 0 && k >= 0, "GEMM dimensions must be nonnegative");
  checked_mul(checked_mul(static_cast<size_t>(m), n), sizeof(float));
  checked_mul(checked_mul(static_cast<size_t>(m), k), sizeof(float));
  checked_mul(checked_mul(static_cast<size_t>(k), n), sizeof(float));
  if (!m || !n) return;
  require(c != nullptr, "GEMM output is required");
  if (Fused) require(bias != nullptr, "fusion requires bias");
  if (!k) { fill_zero_or_bias(c, Fused ? bias : nullptr, m, n, stream); return; }
  require(a && b, "nonzero K requires both GEMM inputs");
  const size_t gx = ceil_div(static_cast<size_t>(n), config.bn), gy = ceil_div(static_cast<size_t>(m), config.bm);
  const size_t shared = config.variant == GemmVariant::g0 ? 0 : (config.bm + config.bn) * config.bk * sizeof(float);
  validate_grid(gx, gy, 256, shared);
  const dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(gy));
  if (config.variant == GemmVariant::g0) gemm_g0<<<grid, dim3(16, 16), 0, stream>>>(a, b, c, m, n, k);
  else if (config.variant == GemmVariant::g1) gemm_g1<<<grid, dim3(16, 16), 0, stream>>>(a, b, c, m, n, k);
  else if (config.bn == 128) gemm_g2<64,128,16,Fused><<<grid, dim3(16,16), 0, stream>>>(a,b,bias,c,m,n,k);
  else if (config.bk == 32) gemm_g2<64,64,32,Fused><<<grid, dim3(16,16), 0, stream>>>(a,b,bias,c,m,n,k);
  else gemm_g2<64,64,16,Fused><<<grid, dim3(16,16), 0, stream>>>(a,b,bias,c,m,n,k);
  CUDA_LAB_CHECK(cudaGetLastError());
}
}
void gemm(const float* a, const float* b, float* c, int m, int n, int k,
          cudaStream_t stream, const GemmConfig& config) {
  launch<false>(a, b, nullptr, c, m, n, k, stream, config);
}
void gemm_bias_relu(const float*, const float*, const float*, float*, int, int, int,
                    cudaStream_t, const GemmConfig&) {
  throw std::invalid_argument("fused G2 is not available in the baseline build");
}
}
