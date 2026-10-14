#pragma once
#include "cuda_lab/ops.hpp"
#include <string>
#include <vector>

namespace lab_bench {
struct Options {
  std::string op = "reduce", variant = "r0", mode = "device", suite, output = "results/local";
  size_t elements = 1048576;
  int m = 512, n = 512, k = 512;
  int samples = 30, warmup = 10, max_repetitions = 10000;
  double target_ms = 20;
  unsigned seed = 2027;
  int block = 256, accumulators = 4, grid_multiplier = 8;
  int bm = 64, bn = 64, bk = 16;
  bool profile_once = false;
};
struct Shape { size_t elements = 0; int m = 0, n = 0, k = 0; };
Options parse_options(int argc, char** argv);
int run(const Options& options);
}
