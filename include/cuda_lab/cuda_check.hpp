#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cuda_lab {
inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(file) + ':' + std::to_string(line) + " " + expression + ": " + cudaGetErrorString(status));
}
inline void check_cublas(cublasStatus_t status, const char* expression, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(file) + ':' + std::to_string(line) + " " + expression + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
}
#define CUDA_LAB_CHECK(expr) ::cuda_lab::check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUDA_LAB_BLAS(expr) ::cuda_lab::check_cublas((expr), #expr, __FILE__, __LINE__)

inline size_t checked_mul(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b) throw std::invalid_argument("size multiplication overflow");
  return a * b;
}
inline size_t ceil_div(size_t n, size_t d) { return n / d + (n % d != 0); }
inline void require(bool condition, const char* message) { if (!condition) throw std::invalid_argument(message); }

template<class T> class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count = 0) : count_(count) {
    if (count) CUDA_LAB_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), checked_mul(count, sizeof(T))));
  }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}
  T* get() const { return data_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }
 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};
template<class T> class PinnedBuffer {
 public:
  explicit PinnedBuffer(size_t count) : count_(count) {
    if (count) CUDA_LAB_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data_), checked_mul(count, sizeof(T))));
  }
  ~PinnedBuffer() { if (data_) cudaFreeHost(data_); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  T* get() const { return data_; }
  size_t bytes() const { return count_ * sizeof(T); }
 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};
class Stream {
 public:
  Stream() { CUDA_LAB_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~Stream() { if (stream_) cudaStreamDestroy(stream_); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  operator cudaStream_t() const { return stream_; }
  void synchronize() const { CUDA_LAB_CHECK(cudaStreamSynchronize(stream_)); }
 private:
  cudaStream_t stream_ = nullptr;
};
class Event {
 public:
  Event() { CUDA_LAB_CHECK(cudaEventCreate(&event_)); }
  ~Event() { if (event_) cudaEventDestroy(event_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  operator cudaEvent_t() const { return event_; }
 private:
  cudaEvent_t event_ = nullptr;
};
inline const cudaDeviceProp& device_properties() {
  // This lab deliberately supports one device selected before first use.
  static const cudaDeviceProp properties = [] {
    cudaDeviceProp p{}; int device = 0;
    CUDA_LAB_CHECK(cudaGetDevice(&device));
    CUDA_LAB_CHECK(cudaGetDeviceProperties(&p, device));
    return p;
  }();
  return properties;
}
inline void validate_grid(size_t x, size_t y, size_t threads, size_t shared = 0) {
  const auto& p = device_properties();
  require(x <= static_cast<size_t>(p.maxGridSize[0]) && y <= static_cast<size_t>(p.maxGridSize[1]), "grid exceeds device limits");
  require(threads <= static_cast<size_t>(p.maxThreadsPerBlock) && shared <= p.sharedMemPerBlock, "block exceeds device limits");
}
}
