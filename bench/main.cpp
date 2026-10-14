#include "benchmark.hpp"
#include <iostream>
int main(int argc, char** argv) {
  try { return lab_bench::run(lab_bench::parse_options(argc, argv)); }
  catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
