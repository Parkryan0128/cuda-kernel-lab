#include "test_support.hpp"
using namespace lab_test;
TEST(Reference, DoubleBeforeMultiplication) {
  float a[] = {4097}, b[] = {4097};
  EXPECT_EQ(reference_gemm_element(a, b, nullptr, 0, 0, 1, 1), 16785409.0);
}
TEST(Reference, CancellationAndTolerance) {
  auto r = reference_sum({16777216.0f, 1.0f, -16777216.0f});
  EXPECT_EQ(r.sum, 1.0); EXPECT_EQ(r.absolute_sum, 33554433.0);
  EXPECT_GT(reduction_tolerance(r), 0);
}
