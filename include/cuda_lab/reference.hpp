#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace cuda_lab {
struct ReductionReference { double sum = 0, absolute_sum = 0; };
inline ReductionReference reference_sum(const std::vector<float>& x) {
  ReductionReference result;
  for (float v : x) { result.sum += static_cast<double>(v); result.absolute_sum += std::abs(static_cast<double>(v)); }
  return result;
}
inline double reduction_tolerance(const ReductionReference& r, double scale = 1) {
  return 1e-6 * std::abs(scale) + 5e-6 * r.absolute_sum;
}
inline double reference_gemm_element(const float* a, const float* b, const float* bias,
                                     size_t row, size_t col, int n, int k) {
  double value = 0;
  for (int t = 0; t < k; ++t) value += static_cast<double>(a[row * k + t]) * static_cast<double>(b[static_cast<size_t>(t) * n + col]);
  return bias ? std::max(0.0, value + static_cast<double>(bias[col])) : value;
}
inline double gemm_tolerance(double reference, double scale_a = 1, double scale_b = 1) {
  return 1e-4 * std::abs(scale_a * scale_b) + 1e-4 * std::abs(reference);
}
inline std::vector<float> random_values(size_t count, unsigned seed, float scale = 1) {
  // Integer-to-float conversion is explicit so the input stream is reproducible across standard libraries.
  std::mt19937 generator(seed);
  std::vector<float> result(count);
  for (auto& value : result) value = (static_cast<float>(generator() >> 8) * (1.0f / 8388608.0f) - 1.0f) * scale;
  return result;
}
}
