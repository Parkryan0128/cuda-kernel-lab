#include "test_support.hpp"
#include <limits>
using namespace lab_test;
namespace {
void check(const std::vector<float>& x, const ReductionCase& c, bool exact = false, float scale = 1) {
  SCOPED_TRACE(c.name + " n=" + std::to_string(x.size()));
  Stream stream;
  DeviceBuffer<float> input(x.size()), output(1);
  const size_t bytes = c.cub ? cub_workspace_bytes(x.size(), stream) : reduce_sum_workspace_bytes(x.size(), c.config);
  DeviceBuffer<unsigned char> scratch(bytes); // Deliberately uninitialized: initcheck must detect bad scratch reads.
  upload(x, input, stream);
  const float sentinel = std::numeric_limits<float>::quiet_NaN();
  CUDA_LAB_CHECK(cudaMemcpyAsync(output.get(), &sentinel, sizeof(float), cudaMemcpyHostToDevice, stream));
  if (c.cub) cub_sum(input.get(), output.get(), x.size(), scratch.get(), bytes, stream);
  else reduce_sum(input.get(), output.get(), x.size(), scratch.get(), bytes, stream, c.config);
  const auto result = download(output, stream);
  const auto expected = reference_sum(x);
  expect_finite_near(result[0], expected.sum, exact ? 0 : reduction_tolerance(expected, scale));
}
}
TEST(ReductionBoundary, SizesAndExactPatterns) {
  for (const auto& c : reduction_configs()) {
    for (size_t n : {0u,1u,2u,31u,32u,33u,127u,128u,129u,255u,256u,257u,511u,512u,513u,1023u,1024u,1025u,65535u,65536u,65537u}) {
      check(std::vector<float>(n, 0), c, true);
      check(std::vector<float>(n, 1), c, true);
      std::vector<float> alternating(n);
      for (size_t i = 0; i < n; ++i) alternating[i] = i % 2 ? -1.0f : 1.0f;
      check(alternating, c, true);
      if (n) { std::vector<float> one_hot(n, 0); one_hot[n - 1] = 7; check(one_hot, c, true); }
    }
  }
}
TEST(ReductionBoundary, RandomAndCancellation) {
  for (const auto& c : reduction_configs()) for (unsigned seed : {7, 123, 8128}) {
    for (float scale : {1.0f / 256, 1.0f, 256.0f}) check(random_values(65537, seed, scale), c, false, scale);
    std::vector<float> values(1025);
    for (size_t i = 0; i < values.size(); ++i) values[i] = i % 3 == 0 ? 1e6f : (i % 3 == 1 ? -1e6f : 0.01f);
    check(values, c);
  }
}
TEST(ReductionBoundary, GridStrideTransitions) {
  for (const auto& c : reduction_configs()) {
    if (c.config.variant != ReduceVariant::r2) continue;
    const size_t stride = static_cast<size_t>(c.config.block_threads) * c.config.max_blocks;
    for (size_t boundary : {stride, stride * c.config.accumulator_count}) for (int offset : {-1,0,1}) {
      const size_t n = boundary + offset;
      check(std::vector<float>(n, 1.0f), c, true);
      check(random_values(n, 8128), c);
    }
  }
}
TEST(ReductionBoundary, ReusesBuffersWithChangedValuesAndSize) {
  Stream stream;
  DeviceBuffer<float> input(65537), output(1);
  for (const auto& c : reduction_configs()) {
    SCOPED_TRACE(c.name);
    const size_t bytes = c.cub ? cub_workspace_bytes(input.size(), stream) : reduce_sum_workspace_bytes(input.size(), c.config);
    DeviceBuffer<unsigned char> scratch(bytes);
    for (size_t n : {65537u, 255u, 65536u, 1u, 0u, 1025u}) {
      auto x = random_values(n, static_cast<unsigned>(n + 7)); upload(x, input, stream);
      if (c.cub) cub_sum(input.get(), output.get(), n, scratch.get(), bytes, stream);
      else reduce_sum(input.get(), output.get(), n, scratch.get(), bytes, stream, c.config);
      const auto r = reference_sum(x); expect_finite_near(download(output, stream)[0], r.sum, reduction_tolerance(r));
    }
  }
}
TEST(ReductionLarge, RandomTails) {
  for (const auto& c : reduction_configs()) for (size_t n : {1048576u, 1048593u, 16777229u}) check(random_values(n, 123), c);
}
TEST(ReductionBoundary, RejectsInvalidArguments) {
  Stream stream; DeviceBuffer<float> x(257), out(1);
  EXPECT_THROW(reduce_sum(nullptr, out.get(), 1, nullptr, 0, stream, {}), std::invalid_argument);
  EXPECT_THROW(reduce_sum(x.get(), nullptr, 0, nullptr, 0, stream, {}), std::invalid_argument);
  EXPECT_THROW(reduce_sum(x.get(), out.get(), 257, nullptr, 0, stream, {}), std::invalid_argument);
  EXPECT_THROW(reduce_sum_workspace_bytes(std::numeric_limits<size_t>::max(), {}), std::invalid_argument);
  ReduceConfig bad; bad.block_threads = 7;
  EXPECT_THROW(reduce_sum_workspace_bytes(1, bad), std::invalid_argument);
  EXPECT_EQ(reduction_pass_count(0, {}), 0);
  EXPECT_EQ(reduction_pass_count(1, {}), 1);
  EXPECT_EQ(reduction_pass_count(256, {}), 1);
  EXPECT_EQ(reduction_pass_count(257, {}), 2);
  EXPECT_EQ(reduction_pass_count(65537, {}), 3);
}
