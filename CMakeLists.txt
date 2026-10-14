cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
project(cuda_kernel_lab VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit 13.0 REQUIRED)
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE LAB_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
configure_file(cmake/build_info.hpp.in generated/cuda_lab/build_info.hpp @ONLY)

add_library(cuda_lab STATIC src/reduction.cu src/gemm.cu src/epilogue.cu src/library_baselines.cu)
target_include_directories(cuda_lab PUBLIC include "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(cuda_lab PUBLIC CUDA::cudart CUDA::cublas)
if(WIN32)
  target_compile_definitions(cuda_lab PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
target_compile_options(cuda_lab PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--ptxas-options=-v>)
if(MSVC)
  target_compile_options(cuda_lab PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(cuda_lab PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

add_executable(cuda_lab_bench bench/main.cpp bench/benchmark.cpp)
target_link_libraries(cuda_lab_bench PRIVATE cuda_lab)
# CUDA 13 packages CCCL under include/cccl on Linux. nvcc adds that
# directory itself, but the host-only metadata code also reads CUB_VERSION.
find_path(CUDA_LAB_CCCL_INCLUDE_DIR cub/version.cuh
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
  PATH_SUFFIXES cccl
  REQUIRED)
target_include_directories(cuda_lab_bench PRIVATE bench "${CUDA_LAB_CCCL_INCLUDE_DIR}")

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    URL https://codeload.github.com/google/googletest/zip/refs/tags/v1.17.0
    URL_HASH SHA256=40d4ec942217dcc84a9ebe2a68584ada7d4a33a8ee958755763278ea1c5e18ff
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(googletest)
  add_executable(cuda_lab_tests tests/test_main.cpp tests/cpu_reference.cpp tests/test_reduction.cpp tests/test_gemm.cpp)
  target_link_libraries(cuda_lab_tests PRIVATE cuda_lab GTest::gtest)
  target_include_directories(cuda_lab_tests PRIVATE tests)
  add_test(NAME cuda_correctness COMMAND cuda_lab_tests)
  set_tests_properties(cuda_correctness PROPERTIES TIMEOUT 300)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME host_results COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_results.py")
  endif()
endif()
