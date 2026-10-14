#include "benchmark.hpp"
#include "cuda_lab/cuda_check.hpp"
#include "cuda_lab/reference.hpp"
#include <cub/version.cuh>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace lab_bench {
using namespace cuda_lab;
namespace fs = std::filesystem;
namespace {
std::string quote(const std::string& value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
long long number(const std::string& value) {
  size_t used = 0; auto result = std::stoll(value, &used);
  require(used == value.size(), "invalid integer argument"); return result;
}
int integer(const std::string& value) {
  auto result = number(value); require(result >= 0 && result <= std::numeric_limits<int>::max(), "integer argument out of range");
  return static_cast<int>(result);
}
void h2d(float* device, const float* host, size_t count, cudaStream_t stream) {
  if (count) CUDA_LAB_CHECK(cudaMemcpyAsync(device, host, count * sizeof(float), cudaMemcpyHostToDevice, stream));
}
void d2h(float* host, const float* device, size_t count, cudaStream_t stream) {
  if (count) CUDA_LAB_CHECK(cudaMemcpyAsync(host, device, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
}
std::vector<std::string> variants(const Options& o) {
  if (o.variant != "all") return {o.variant};
  if (o.op == "reduce") return {"r0", "r1", "r2", "cub"};
  if (o.op == "gemm") return {"g0", "g1", "g2", "cublas"};
  return {"unfused", "fused", "library"};
}
std::vector<Shape> shapes(const Options& o) {
  if (o.suite.empty()) return {{o.elements, o.m, o.n, o.k}};
  require(o.suite == "standard" || o.suite == "representative" || o.suite == "tuning", "unknown suite");
  if (o.op == "reduce") {
    std::vector<size_t> sizes;
    if (o.suite == "tuning") sizes = {1048576, 16777216};
    else if (o.suite == "representative") sizes = {1024, 16777216};
    else sizes = {256,1024,4096,16384,65536,262144,1048576,1048593,4194304,16777216,16777229,67108864};
    std::vector<Shape> result; for (auto n : sizes) result.push_back({n,0,0,0}); return result;
  }
  if (o.op == "fusion") return {{0,128,512,64},{0,512,1024,128},{0,1024,1024,1024}};
  if (o.suite == "tuning") return {{0,512,512,512},{0,1024,1024,1024}};
  if (o.suite == "representative") return {{0,128,128,128},{0,1024,1024,1024}};
  return {{0,128,128,128},{0,512,512,512},{0,1024,1024,1024},{0,2048,2048,2048},
          {0,1024,2048,512},{0,257,509,127},{0,128,512,64}};
}
struct Validation {
  std::string scope;
  size_t compared = 0, fp64_samples = 0;
  double max_abs_error = 0, max_normalized_error = 0;
  void compare(float value, double reference, double tolerance) {
    if (!std::isfinite(value) || !std::isfinite(reference)) throw std::runtime_error("non-finite correctness value");
    double error = std::abs(static_cast<double>(value) - reference);
    max_abs_error = std::max(max_abs_error, error);
    max_normalized_error = std::max(max_normalized_error, error / tolerance);
    ++compared;
    if (error > tolerance) {
      std::ostringstream msg; msg << "correctness failure: value=" << value << " reference=" << reference << " error=" << error << " tolerance=" << tolerance;
      throw std::runtime_error(msg.str());
    }
  }
};
class Case {
 public:
  Case(const Options& o, const Shape& s, const std::string& variant)
      : o_(o), s_(s), variant_(variant),
        na_(o.op == "reduce" ? s.elements : checked_mul(s.m, s.k)),
        nb_(o.op == "reduce" ? 0 : checked_mul(s.k, s.n)),
        ny_(o.op == "reduce" ? 1 : checked_mul(s.m, s.n)), nz_(o.op == "fusion" ? s.n : 0),
        ha_(na_), hb_(nb_), hy_(ny_), hz_(nz_), a_(na_), b_(nb_), y_(ny_), bias_(nz_), reference_(o.op == "reduce" ? 0 : ny_) {
    rconfig_ = {variant == "r2" ? ReduceVariant::r2 : variant == "r1" ? ReduceVariant::r1 : ReduceVariant::r0,
                variant == "r2" ? o.block : 256, variant == "r2" ? o.accumulators : 1,
                variant == "r2" ? o.grid_multiplier * device_properties().multiProcessorCount : 0};
    gconfig_ = variant == "g0" ? GemmConfig{} : variant == "g1" ? GemmConfig{GemmVariant::g1,16,16,16} : GemmConfig{GemmVariant::g2,o.bm,o.bn,o.bk};
    if (o.op == "reduce") {
      workspace_bytes_ = variant == "cub" ? cub_workspace_bytes(na_, stream_) : reduce_sum_workspace_bytes(na_, rconfig_);
      scratch_ = std::make_unique<DeviceBuffer<unsigned char>>(workspace_bytes_);
      auto input = random_values(na_, o.seed);
      reduction_reference_ = reference_sum(input);
      std::copy(input.begin(), input.end(), ha_.get());
    } else {
      blas_ = std::make_unique<BlasHandle>(stream_);
      if (variant == "cublas" || variant == "library") workspace_bytes_ = BlasHandle::workspace_bytes;
      const auto a = random_values(na_, o.seed), b = random_values(nb_, o.seed + 1), bias = random_values(nz_, o.seed + 2);
      std::copy(a.begin(), a.end(), ha_.get()); std::copy(b.begin(), b.end(), hb_.get()); std::copy(bias.begin(), bias.end(), hz_.get());
    }
    copy_in(); stream_.synchronize();
    if (o.op != "reduce") {
      blas_->gemm(a_.get(), b_.get(), reference_.get(), s.m, s.n, s.k);
      if (o.op == "fusion") bias_relu_inplace(reference_.get(), bias_.get(), s.m, s.n, stream_);
      library_reference_.resize(ny_);
      d2h(library_reference_.data(), reference_.get(), ny_, stream_); stream_.synchronize();
    }
  }
  void copy_in() { h2d(a_.get(), ha_.get(), na_, stream_); h2d(b_.get(), hb_.get(), nb_, stream_); h2d(bias_.get(), hz_.get(), nz_, stream_); }
  void copy_out() { d2h(hy_.get(), y_.get(), ny_, stream_); }
  void operation() {
    if (o_.op == "reduce") {
      if (variant_ == "cub") cub_sum(a_.get(), y_.get(), na_, scratch_->get(), workspace_bytes_, stream_);
      else reduce_sum(a_.get(), y_.get(), na_, scratch_->get(), workspace_bytes_, stream_, rconfig_);
    } else if (o_.op == "gemm") {
      if (variant_ == "cublas") blas_->gemm(a_.get(), b_.get(), y_.get(), s_.m, s_.n, s_.k);
      else gemm(a_.get(), b_.get(), y_.get(), s_.m, s_.n, s_.k, stream_, gconfig_);
    } else if (variant_ == "fused") {
      gemm_bias_relu(a_.get(), b_.get(), bias_.get(), y_.get(), s_.m, s_.n, s_.k, stream_, gconfig_);
    } else {
      if (variant_ == "library") blas_->gemm(a_.get(), b_.get(), y_.get(), s_.m, s_.n, s_.k);
      else gemm(a_.get(), b_.get(), y_.get(), s_.m, s_.n, s_.k, stream_, gconfig_);
      bias_relu_inplace(y_.get(), bias_.get(), s_.m, s_.n, stream_);
    }
  }
  void job() {
    if (o_.mode == "e2e") { copy_in(); operation(); copy_out(); }
    else if (o_.mode == "h2d") copy_in();
    else if (o_.mode == "d2h") copy_out();
    else operation();
  }
  double elapsed_ms(int repetitions) {
    stream_.synchronize();
    if (o_.mode == "e2e") {
      auto start = std::chrono::steady_clock::now(); job(); stream_.synchronize();
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    CUDA_LAB_CHECK(cudaEventRecord(start_, stream_));
    for (int i = 0; i < repetitions; ++i) job();
    CUDA_LAB_CHECK(cudaEventRecord(stop_, stream_)); CUDA_LAB_CHECK(cudaEventSynchronize(stop_));
    float ms = 0; CUDA_LAB_CHECK(cudaEventElapsedTime(&ms, start_, stop_)); return ms;
  }
  Validation validate(bool launch) {
    if (launch) operation();
    copy_out(); stream_.synchronize(); Validation result;
    if (o_.op == "reduce") {
      result.scope = "full_cpu_fp64_sum";
      result.compare(hy_.get()[0], reduction_reference_.sum, reduction_tolerance(reduction_reference_)); return result;
    }
    const bool full_reference = s_.m <= 129 && s_.n <= 129 && s_.k <= 129;
    result.scope = full_reference ? "full_cpu_fp64" : "full_cublas_fp32_plus_sampled_cpu_fp64";
    if (!full_reference) for (size_t i = 0; i < ny_; ++i) result.compare(hy_.get()[i], library_reference_[i], 2 * gemm_tolerance(library_reference_[i]));
    const size_t samples = full_reference ? ny_ : std::min<size_t>(128, ny_);
    for (size_t j = 0; j < samples; ++j) {
      const size_t i = full_reference || samples == 1 ? j : (j * (ny_ - 1)) / (samples - 1);
      const double expected = reference_gemm_element(ha_.get(), hb_.get(), nz_ ? hz_.get() : nullptr, i / s_.n, i % s_.n, s_.n, s_.k);
      result.compare(hy_.get()[i], expected, gemm_tolerance(expected));
      result.compare(library_reference_[i], expected, gemm_tolerance(expected)); ++result.fp64_samples;
    }
    return result;
  }
  void warmup() { for (int i = 0; i < o_.warmup; ++i) job(); stream_.synchronize(); }
  void profile() {
    warmup(); nvtxRangePushA("measured_operation");
    job(); stream_.synchronize(); nvtxRangePop();
  }
  size_t input_bytes() const { return (na_ + nb_ + nz_) * sizeof(float); }
  size_t output_bytes() const { return ny_ * sizeof(float); }
  size_t workspace() const { return workspace_bytes_; }
  int passes() const { return o_.op == "reduce" && variant_ != "cub" ? reduction_pass_count(na_, rconfig_) : -1; }
  std::string config() const {
    if (variant_ == "cub") return "cub_default_run_to_run";
    if (variant_ == "cublas" || variant_ == "library") return "cublas_fp32_pedantic";
    if (o_.op == "reduce") return "B" + std::to_string(rconfig_.block_threads) + "_A" + std::to_string(rconfig_.accumulator_count) + "_cap" + std::to_string(rconfig_.max_blocks);
    return std::to_string(gconfig_.bm) + "x" + std::to_string(gconfig_.bn) + "x" + std::to_string(gconfig_.bk);
  }
 private:
  Options o_; Shape s_; std::string variant_;
  size_t na_, nb_, ny_, nz_, workspace_bytes_ = 0;
  Stream stream_; Event start_, stop_;
  PinnedBuffer<float> ha_, hb_, hy_, hz_;
  DeviceBuffer<float> a_, b_, y_, bias_, reference_;
  std::unique_ptr<DeviceBuffer<unsigned char>> scratch_;
  std::unique_ptr<BlasHandle> blas_;
  ReduceConfig rconfig_; GemmConfig gconfig_;
  ReductionReference reduction_reference_;
  std::vector<float> library_reference_;
};
void write_validation(std::ostream& out, const std::string& id, const char* phase, const Validation& v) {
  out << "{\"case\":" << quote(id) << ",\"phase\":" << quote(phase) << ",\"passed\":true,\"scope\":" << quote(v.scope)
      << ",\"compared_values\":" << v.compared << ",\"fp64_output_samples\":" << v.fp64_samples
      << ",\"max_abs_error\":" << v.max_abs_error << ",\"max_normalized_error\":" << v.max_normalized_error << "}\n";
}
}
Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--help") {
      std::cout << "CUDA Kernel Lab\n--op reduce|gemm|fusion --variant r0|r1|r2|cub|g0|g1|g2|cublas|unfused|fused|library|all\n"
                   "--mode device|e2e|h2d|d2h --suite standard|representative|tuning\n"
                   "--elements N --m M --n N --k K --output DIR\n"
                   "--block 128|256 --accumulators 1|4 --grid-multiplier 4|8 --bm 64 --bn 64|128 --bk 16|32\n"
                   "--samples 30 --warmup 10 --target-ms 20 --max-repetitions 10000 --seed 2027 --profile-once\n";
      std::exit(0);
    }
    if (key == "--profile-once") { o.profile_once = true; continue; }
    require(i + 1 < argc, "missing option value"); std::string value = argv[++i];
    if (key == "--op") o.op = value; else if (key == "--variant") o.variant = value;
    else if (key == "--mode") o.mode = value; else if (key == "--suite") o.suite = value; else if (key == "--output") o.output = value;
    else if (key == "--elements") { auto n = number(value); require(n >= 0, "negative reduction size"); o.elements = static_cast<size_t>(n); }
    else if (key == "--m") o.m = integer(value); else if (key == "--n") o.n = integer(value); else if (key == "--k") o.k = integer(value);
    else if (key == "--samples") o.samples = integer(value); else if (key == "--warmup") o.warmup = integer(value);
    else if (key == "--target-ms") { size_t used = 0; o.target_ms = std::stod(value, &used); require(used == value.size(), "invalid target time"); }
    else if (key == "--max-repetitions") o.max_repetitions = integer(value); else if (key == "--seed") o.seed = integer(value);
    else if (key == "--block") o.block = integer(value); else if (key == "--accumulators") o.accumulators = integer(value);
    else if (key == "--grid-multiplier") o.grid_multiplier = integer(value);
    else if (key == "--bm") o.bm = integer(value); else if (key == "--bn") o.bn = integer(value); else if (key == "--bk") o.bk = integer(value);
    else throw std::invalid_argument("unknown option: " + key);
  }
  require(o.op == "reduce" || o.op == "gemm" || o.op == "fusion", "unknown operation");
  require(o.mode == "device" || o.mode == "e2e" || o.mode == "h2d" || o.mode == "d2h", "unknown timing mode");
  require(o.samples > 0 && o.warmup > 0 && o.max_repetitions > 0 && std::isfinite(o.target_ms) && o.target_ms > 0, "invalid sampling settings");
  require(o.grid_multiplier == 4 || o.grid_multiplier == 8, "grid multiplier must be 4 or 8");
  const auto allowed = o.op == "reduce" ? std::vector<std::string>{"r0","r1","r2","cub","all"} : o.op == "gemm" ?
      std::vector<std::string>{"g0","g1","g2","cublas","all"} : std::vector<std::string>{"unfused","fused","library","all"};
  require(std::find(allowed.begin(), allowed.end(), o.variant) != allowed.end(), "variant does not belong to operation");
  require(!o.profile_once || (o.suite.empty() && o.variant != "all"), "profile-once requires one explicit case and variant");
  return o;
}
int run(const Options& o) {
  CUDA_LAB_CHECK(cudaSetDevice(0)); const auto& p = device_properties();
  fs::create_directories(o.output);
  const fs::path directory(o.output);
  require(!fs::exists(directory / "samples.csv") && !fs::exists(directory / "metadata.json"), "output directory already contains a run; choose a fresh directory");
  auto id = directory.filename().string();
  int runtime = 0, driver = 0;
  CUDA_LAB_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_LAB_CHECK(cudaDriverGetVersion(&driver));
  Stream info_stream; BlasHandle info_blas(info_stream);
  std::ofstream metadata(directory / "metadata.json");
  metadata << "{\n\"run_id\":" << quote(id) << ",\n\"gpu\":" << quote(p.name) << ",\n\"compute_capability\":" << quote(std::to_string(p.major) + "." + std::to_string(p.minor))
      << ",\n\"sm_count\":" << p.multiProcessorCount << ",\n\"l2_bytes\":" << p.l2CacheSize << ",\n\"vram_bytes\":" << p.totalGlobalMem
      << ",\n\"cuda_runtime\":" << runtime << ",\n\"driver_api_version\":" << driver << ",\n\"cuda_header_version\":" << CUDART_VERSION
      << ",\n\"cub_version\":" << CUB_VERSION << ",\n\"cublas_version\":" << info_blas.version()
      << ",\n\"math_mode\":\"FP32 pedantic, no TF32/Tensor Core comparison\",\n\"cub_policy\":\"DeviceReduce::Sum traditional API, default run-to-run determinism\""
      << ",\n\"stream\":\"one explicit nonblocking stream per case\",\n\"cache_policy\":\"warm repeated buffers; no explicit flush\""
      << ",\n\"timing\":\"CUDA events for device/H2D/D2H; steady_clock with final stream sync for E2E\""
      << ",\n\"samples\":" << o.samples << ",\n\"warmup\":" << o.warmup << ",\n\"target_batch_ms\":" << o.target_ms
      << ",\n\"max_repetitions\":" << o.max_repetitions << ",\n\"profile_once\":" << (o.profile_once ? "true" : "false") << "\n}\n";
  std::ofstream csv(directory / "samples.csv"), correctness(directory / "correctness.jsonl"), skipped(directory / "skipped.jsonl");
  require(metadata.good() && csv.good() && correctness.good() && skipped.good(), "could not open output files");
  csv << "run_id,operation,variant,mode,n_elements,m,n,k,seed,distribution,scale,config,workspace_bytes,pass_count,input_bytes,output_bytes,working_set_bytes,sample,repetitions,total_ms,per_operation_us,correctness\n";
  csv << std::setprecision(12); correctness << std::setprecision(12);
  for (const auto& shape : shapes(o)) for (const auto& variant : variants(o)) {
    const std::string case_id = o.op + "/" + variant + "/" + o.mode + "/" + (o.op == "reduce" ? std::to_string(shape.elements) : std::to_string(shape.m) + "x" + std::to_string(shape.n) + "x" + std::to_string(shape.k));
    if (variant == "g0" && static_cast<double>(shape.m) * shape.n * shape.k > 1073741824.0) {
      skipped << "{\"case\":" << quote(case_id) << ",\"reason\":\"G0 M*N*K > 1024^3 skip policy\"}\n"; continue;
    }
    std::cout << case_id << std::endl;
    try {
      Case test(o, shape, variant);
      write_validation(correctness, case_id, "before", test.validate(true)); correctness.flush();
      if (o.profile_once) { test.profile(); write_validation(correctness, case_id, "after", test.validate(false)); continue; }
      test.warmup(); int repetitions = 1;
      if (o.mode != "e2e") {
        const double pilot_ms = test.elapsed_ms(10) / 10.0;
        const double estimate = std::ceil(o.target_ms / std::max(pilot_ms, 1e-9));
        repetitions = static_cast<int>(std::clamp(estimate, 1.0, static_cast<double>(o.max_repetitions)));
      }
      for (int sample = 0; sample < o.samples; ++sample) {
        const double ms = test.elapsed_ms(repetitions);
        csv << id << ',' << o.op << ',' << variant << ',' << o.mode << ',' << (o.op == "reduce" ? shape.elements : 0) << ','
            << (o.op == "reduce" ? 0 : shape.m) << ',' << (o.op == "reduce" ? 0 : shape.n) << ',' << (o.op == "reduce" ? 0 : shape.k) << ','
            << o.seed << ",\"uniform[-1,1)\",1," << test.config() << ',' << test.workspace() << ',' << test.passes() << ','
            << test.input_bytes() << ',' << test.output_bytes() << ',' << (test.input_bytes() + test.output_bytes() + test.workspace()) << ','
            << sample << ',' << repetitions << ',' << ms << ',' << ms * 1000 / repetitions << ",passed\n";
      }
      write_validation(correctness, case_id, "after", test.validate(false)); csv.flush(); correctness.flush();
    } catch (const std::exception& error) {
      correctness << "{\"case\":" << quote(case_id) << ",\"passed\":false,\"error\":" << quote(error.what()) << "}\n"; throw;
    }
  }
  require(csv.good() && correctness.good() && skipped.good(), "output write failed");
  return 0;
}
}
