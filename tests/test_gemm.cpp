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
  if (config.epilogue) bias_relu_inplace(dc.get(), dbias.get(), m, n, stream);
  const auto result = download(dc, stream);
  for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
    const double ref = reference_gemm_element(a.data(), b.data(), (config.fused || config.epilogue) ? bias.data() : nullptr, row, col, n, k);
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
  GemmConfig g2{GemmVariant::g2,64,64,32};
  EXPECT_THROW(gemm_bias_relu(nullptr, nullptr, nullptr, x.get(), 1, 1, 0, stream, g2), std::invalid_argument);
  EXPECT_THROW(gemm_bias_relu(x.get(), x.get(), x.get(), x.get(), 1, 1, 1, stream, {}), std::invalid_argument);
  EXPECT_THROW(gemm(x.get(), x.get(), x.get(), 1, 1, 1, stream, {GemmVariant::g2,64,128,32}), std::invalid_argument);
  EXPECT_THROW(gemm(x.get(), x.get(), x.get(), std::numeric_limits<int>::max(), 1, 1, stream, {}), std::invalid_argument);
}
TEST(GemmBoundary, ExactIdentityZeroOneHotAndBufferReuse) {
  Stream stream; BlasHandle blas(stream);
  DeviceBuffer<float> da(129 * 129), db(129 * 129), dc(129 * 129), dbias(129);
  for (const auto& config : gemm_configs()) {
    SCOPED_TRACE(config.name);
    for (int dim : {129, 17, 65}) for (int pattern : {0,1,2}) {
      const int m = dim, n = dim - 2, k = dim;
      std::vector<float> a(static_cast<size_t>(m) * k, 0), b(static_cast<size_t>(k) * n), bias(n);
      if (pattern == 1) for (int r = 0; r < m; ++r) a[static_cast<size_t>(r) * k + r] = 1;
      if (pattern == 2) a[a.size() - 1] = 1;
      for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>(static_cast<int>(i % 13) - 6);
      for (int col = 0; col < n; ++col) bias[col] = static_cast<float>((col % 5) - 2);
      upload(a, da, stream); upload(b, db, stream); upload(bias, dbias, stream);
      CUDA_LAB_CHECK(cudaMemsetAsync(dc.get(), 0xff, dc.bytes(), stream));
      if (config.blas) blas.gemm(da.get(), db.get(), dc.get(), m, n, k);
      else if (config.fused) gemm_bias_relu(da.get(), db.get(), dbias.get(), dc.get(), m, n, k, stream, config.config);
      else gemm(da.get(), db.get(), dc.get(), m, n, k, stream, config.config);
      if (config.epilogue) bias_relu_inplace(dc.get(), dbias.get(), m, n, stream);
      const auto result = download(dc, stream);
      for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
        float expected = pattern == 1 ? b[static_cast<size_t>(row) * n + col] :
                         pattern == 2 && row == m - 1 ? b[static_cast<size_t>(k - 1) * n + col] : 0;
        if (config.fused || config.epilogue) expected = std::max(0.0f, expected + bias[col]);
        ASSERT_EQ(result[static_cast<size_t>(row) * n + col], expected);
      }
      // Store guards leave the unused tail of the allocation untouched.
      for (size_t i = static_cast<size_t>(m) * n; i < result.size(); ++i) ASSERT_TRUE(std::isnan(result[i]));
    }
  }
}
TEST(GemmLarge, FullLibraryAndSampledDoubleReference) {
  const int m = 257, n = 509, k = 127;
  Stream stream; BlasHandle blas(stream);
  auto a = random_values(m * k, 8128), b = random_values(k * n, 8129), bias = random_values(n, 8130);
  DeviceBuffer<float> da(a.size()), db(b.size()), dbias(bias.size()), dc(m * n), ref(m * n);
  upload(a, da, stream); upload(b, db, stream); upload(bias, dbias, stream);
  for (const auto& config : gemm_configs()) {
    SCOPED_TRACE(config.name);
    blas.gemm(da.get(), db.get(), ref.get(), m, n, k);
    if (config.fused || config.epilogue) bias_relu_inplace(ref.get(), dbias.get(), m, n, stream);
    if (config.blas) blas.gemm(da.get(), db.get(), dc.get(), m, n, k);
    else if (config.fused) gemm_bias_relu(da.get(), db.get(), dbias.get(), dc.get(), m, n, k, stream, config.config);
    else gemm(da.get(), db.get(), dc.get(), m, n, k, stream, config.config);
    if (config.epilogue) bias_relu_inplace(dc.get(), dbias.get(), m, n, stream);
    const auto expected = download(ref, stream), actual = download(dc, stream);
    for (size_t i = 0; i < actual.size(); ++i) {
      expect_finite_near(actual[i], expected[i], 2 * gemm_tolerance(expected[i]));
      if (::testing::Test::HasFailure()) return;
    }
    for (size_t j = 0; j < 128; ++j) {
      size_t i = j * (actual.size() - 1) / 127;
      const double value = reference_gemm_element(a.data(), b.data(), (config.fused || config.epilogue) ? bias.data() : nullptr, i / n, i % n, n, k);
      expect_finite_near(actual[i], value, gemm_tolerance(value));
      expect_finite_near(expected[i], value, gemm_tolerance(value));
    }
  }
}
