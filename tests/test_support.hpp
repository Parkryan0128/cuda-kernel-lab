#pragma once
#include "cuda_lab/ops.hpp"
#include "cuda_lab/cuda_check.hpp"
#include "cuda_lab/reference.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace lab_test {
using namespace cuda_lab;
struct ReductionCase { std::string name; ReduceConfig config; bool cub = false; };
inline std::vector<ReductionCase> reduction_configs() {
  std::vector<ReductionCase> result = {{"r0", {}}, {"r1", {ReduceVariant::r1,256,1,0}}, {"cub", {}, true}};
  for (int block : {128,256}) for (int accumulators : {1,4}) for (int multiplier : {4,8})
    result.push_back({"r2_B" + std::to_string(block) + "_A" + std::to_string(accumulators) + "_grid" + std::to_string(multiplier),
                      {ReduceVariant::r2,block,accumulators,multiplier * device_properties().multiProcessorCount}});
  return result;
}
struct GemmCase { std::string name; GemmConfig config; bool blas = false; bool fused = false; bool epilogue = false; };
inline std::vector<GemmCase> gemm_configs() {
  std::vector<GemmCase> result = {{"g0", {}}, {"g1", {GemmVariant::g1,16,16,16}}, {"cublas", {}, true}};
  for (const auto& tile : std::vector<GemmConfig>{{GemmVariant::g2,64,64,16},{GemmVariant::g2,64,64,32},{GemmVariant::g2,64,128,16}}) {
    result.push_back({"g2_" + std::to_string(tile.bm) + "x" + std::to_string(tile.bn) + "x" + std::to_string(tile.bk), tile});
    result.push_back({"fused_" + std::to_string(tile.bm) + "x" + std::to_string(tile.bn) + "x" + std::to_string(tile.bk), tile, false, true});
    result.push_back({"unfused_" + std::to_string(tile.bm) + "x" + std::to_string(tile.bn) + "x" + std::to_string(tile.bk), tile, false, false, true});
  }
  result.push_back({"library_bias_relu", {}, true, false, true});
  return result;
}
template<class T> void upload(const std::vector<T>& source, DeviceBuffer<T>& target, cudaStream_t stream) {
  if (!source.empty()) CUDA_LAB_CHECK(cudaMemcpyAsync(target.get(), source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
}
template<class T> std::vector<T> download(const DeviceBuffer<T>& source, const Stream& stream) {
  std::vector<T> out(source.size());
  if (!out.empty()) CUDA_LAB_CHECK(cudaMemcpyAsync(out.data(), source.get(), source.bytes(), cudaMemcpyDeviceToHost, stream));
  stream.synchronize(); return out;
}
inline void expect_finite_near(float actual, double reference, double tolerance) {
  ASSERT_TRUE(std::isfinite(actual)) << "non-finite output";
  EXPECT_LE(std::abs(static_cast<double>(actual) - reference), tolerance) << "actual=" << actual << " reference=" << reference;
}
}
