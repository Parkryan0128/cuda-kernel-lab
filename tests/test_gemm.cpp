#include "test_support.hpp"
#include <array>
#include <limits>
using namespace lab_test;
namespace {
void check(int m, int n, int k, const GemmCase& config, unsigned seed, float scale = 1, bool exact = false) {
  SCOPED_TRACE(config.name + " " + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k));
  Stream stream; BlasHandle blas(stream);
  auto a = random_values(static_cast<size_t>(m) * k, seed, scale);
  auto b = random_values(static_cast<size_t>(k) * n, seed + 1, scale);
  auto bias = random_values(n, seed + 2, scale * scale);
  if (exact) {
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(static_cast<int>(i % 3) - 1);
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  }
  DeviceBuffer<float> da(a.size()), db(b.size()), dc(static_cast<size_t>(m) * n), dbias(bias.size());
  upload(a, da, stream); upload(b, db, stream); upload(bias, dbias, stream);
  if (dc.size()) CUDA_LAB_CHECK(cudaMemsetAsync(dc.get(), 0xff, dc.bytes(), stream));
  if (config.blas) blas.gemm(da.get(), db.get(), dc.get(), m, n, k);
  else if (config.fused) gemm_bias_relu(da.get(), db.get(), dbias.get(), dc.get(), m, n, k, stream, config.config);
  else gemm(da.get(), db.get(), dc.get(), m, n, k, stream, config.config);
  const auto result = download(dc, stream);
  for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
    const double ref = reference_gemm_element(a.data(), b.data(), config.fused ? bias.data() : nullptr, row, col, n, k);
    expect_finite_near(result[static_cast<size_t>(row) * n + col], ref, exact ? 0 : gemm_tolerance(ref, scale, scale));
    if (::testing::Test::HasFailure()) return;
  }
}
}
TEST(GemmBoundary, EmptyAndKTails) {
  for (const auto& config : gemm_configs()) for (const auto& shape : std::vector<std::array<int, 3>>{
    {0,7,13},{7,0,13},{7,13,0},{1,1,1},{7,13,5},{15,17,31},{31,33,65},{63,65,17},{127,129,33}}) {
    check(shape[0], shape[1], shape[2], config, 123, 1, true);
    check(shape[0], shape[1], shape[2], config, 7);
  }
}
TEST(GemmBoundary, RandomScalesAndSeeds) {
  for (const auto& config : gemm_configs()) for (unsigned seed : {7, 123, 8128})
    for (float scale : {1.0f / 256, 1.0f, 256.0f}) check(65, 129, 127, config, seed, scale);
}
TEST(GemmBoundary, RejectsInvalidArguments) {
  Stream stream; BlasHandle blas(stream); DeviceBuffer<float> x(1);
  EXPECT_THROW(gemm(nullptr, nullptr, nullptr, -1, 0, 0, stream, {}), std::invalid_argument);
  EXPECT_THROW(gemm(nullptr, x.get(), x.get(), 1, 1, 1, stream, {}), std::invalid_argument);
  EXPECT_THROW(gemm(nullptr, nullptr, nullptr, 1, 1, 0, stream, {}), std::invalid_argument);
  EXPECT_THROW(blas.gemm(nullptr, x.get(), x.get(), 1, 1, 1), std::invalid_argument);
  EXPECT_THROW(bias_relu_inplace(x.get(), nullptr, 1, 1, stream), std::invalid_argument);
  EXPECT_NO_THROW(gemm(nullptr, nullptr, nullptr, 0, 7, 13, stream, {}));
}
