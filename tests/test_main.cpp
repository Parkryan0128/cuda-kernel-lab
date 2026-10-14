#include "test_support.hpp"
#include <iostream>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  try {
    CUDA_LAB_CHECK(cudaSetDevice(0));
    std::cout << "GPU: " << cuda_lab::device_properties().name << '\n';
    const int status = RUN_ALL_TESTS();
    if (::testing::UnitTest::GetInstance()->test_to_run_count() == 0) {
      std::cerr << "Refusing an empty GPU test selection\n"; return 2;
    }
    CUDA_LAB_CHECK(cudaDeviceSynchronize());
    return status;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
