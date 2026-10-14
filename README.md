# CUDA Kernel Lab

CUDA C++ implementations of **FP32 reduction, GEMM, and fused bias/ReLU**, with correctness tests, CUB/cuBLAS comparisons, and profiler evidence on an RTX 4070 SUPER.

The project follows a small progression from readable baselines to bounded optimizations. Its outcome includes where those optimizations help, where they regress, and why GPU-only gains can disappear once transfers are included.

## Results at a glance

| Operation and scope | Measured result | Important limit |
|---|---|---|
| GEMM, 1024³, device | G0 **907.91 µs → G2 210.98 µs**, a **4.30×** baseline speedup | cuBLAS: **128.74 µs**; G2 reaches about 61% of its throughput |
| Reduction, 67,108,864 floats, device | R0 **831.07 µs → R2 588.50 µs**, a **1.41×** baseline speedup | At 16,777,216 floats, repeated R2/CUB measurements are near parity |
| GEMM+bias+ReLU, 128×512×64, device | Unfused **10.76 µs → fused 7.75 µs**, a **1.39×** speedup | E2E: **81.40 → 80.25 µs**, within overlapping process variation |
| Small GEMM, 128³, device | G0 **6.47 µs**, G2 **10.40 µs** | The optimized tile is slower here; the regression is retained |

Reduction/GEMM headline values are medians of 30 batch means from the standard campaign. Fusion values are medians of three independent process medians. FP32 library comparisons use pedantic cuBLAS math; no TF32 or Tensor Core speedup is implied. The [measurement rules and full reproduction guide](docs/reproduction.md) explain the scopes and limitations.

![FP32 GEMM results](docs/figures/gemm.png)

## Implemented scope

| Operation | Progression | Reference |
|---|---|---|
| Sum reduction | R0 shared tree → R1 warp shuffle → R2 capped grid-stride loop with independent accumulators | CUB `DeviceReduce::Sum` |
| Row-major GEMM | G0 one output/thread → G1 shared tiling → G2 shared + register tiling | `cublasGemmEx`, FP32 pedantic |
| Bias/ReLU fusion | G2 + combined epilogue versus fused G2 versus cuBLAS + the same epilogue | FP64 checks and library comparison |

Fixed defaults were chosen from **8 R2 configurations and 3 G2 tiles** using only the declared tuning shapes: R2 uses 128 threads, 4 accumulators, and a 4×SM grid cap; G2 uses a 64×64×32 tile. There is no per-shape autotuner.

The API handles empty inputs, K=0, tail blocks/tiles, scratch reuse, invalid dimensions/configurations, and byte-count overflow. One explicit nonblocking stream is used for operations and timing. Workspace, pinned buffers, and library handles are allocated before measurement.

## Validation and evidence

- **13 GoogleTest functions passed**, covering all retained candidates, exact patterns, random seeds/scales, tails, buffer reuse, and API rejection paths.
- **Compute Sanitizer:** memcheck, racecheck, initcheck, and synccheck each executed **9 boundary test functions** with zero reported errors; racecheck also reported zero warnings.
- **200 release case series / 6,000 raw samples** from 51 processes, each with correctness checks before and after measurement.
- **22 tuning case series**, **27 hardware-counter reports**, and **3 marked CUDA timelines**.
- **Five reproducible figures**, three case studies, raw samples, environment metadata, and an implementation walkthrough.

Small GEMMs use complete CPU FP64 references. Larger cases use full cuBLAS FP32 comparisons plus up to 128 FP64 output checks for both paths. The evidence does not claim a full FP64 reference for large matrices. See [validation logs](results/validation/validation.json), [benchmark summaries](results/release/summary.json), and [profile extracts](profiles/summary.json).

## Build and run

Requirements: a CUDA-capable NVIDIA GPU, CUDA Toolkit 13.x, a compatible C++17 host compiler, CMake 3.24+, Ninja, and Python 3. GoogleTest 1.17.0 is fetched from a pinned archive. Published GPU results use native Windows, CUDA 13.0.2, MSVC 19.39, and architecture 89.

From an x64 Visual Studio Developer PowerShell on Windows:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=--use-local-env
cmake --build build -j 8
ctest --test-dir build --output-on-failure
build/cuda_lab_bench.exe --op gemm --variant all --suite standard --mode device --output results/local-gemm
python scripts/summarize.py results/local-gemm
```

On Linux, omit `-DCMAKE_CUDA_FLAGS=--use-local-env` and the `.exe` suffix. Linux GPU execution is not part of the published validation. For sanitizer prerequisites, offline dependency setup, complete benchmark coverage, and profiling commands, follow [the reproduction guide](docs/reproduction.md).

Fresh output directories are required; measured data is not silently overwritten. `--help` lists single-case and configuration options. The full release runner includes transfer-only measurements and three independent-process repetitions of representative device/E2E comparisons.

## Read the experiments

1. [Reduction: fewer barriers and fewer partials](docs/case-studies/reduction.md) — pass scheduling, small-input overhead, L2 versus DRAM, and CUB parity.
2. [GEMM: register reuse versus occupancy](docs/case-studies/gemm.md) — tile selection, profiler observations, cuBLAS gap, and small-shape regression.
3. [Fusion: device gain versus end-to-end cost](docs/case-studies/fusion.md) — launch count, intermediate traffic, and the limits of transfer-inclusive gains.

The [architecture guide](docs/architecture.md) walks through indexing, synchronization, scratch lifetime, cuBLAS layout mapping, numerical tolerances, and interview questions. The [original reviewed implementation plan](docs/implementation-plan.md) is preserved as design history; its planning-stage checklist is not the current project status.

## Repository map

```text
include/cuda_lab/   Public operation contracts, RAII helpers, CPU references
src/               Custom CUDA kernels and CUB/cuBLAS adapters
bench/             Explicit-stream timing, validation, and raw export
tests/            GPU correctness and host result-parser checks
scripts/           Validation, bounded sweeps, profiling, summaries, plots
results/           Raw tuning/release samples and verification logs
profiles/          Public counter exports and marked timeline extracts
docs/             Case studies, figures, architecture, reproduction guide
```

CI compiles CUDA and runs host checks on a CPU runner; it does not claim GPU execution. Binary profiler containers remain local and ignored by Git; public CSV/JSON evidence and collection commands are included.

## Deliberate boundaries

This is a one-GPU FP32 learning and performance-analysis project. It is not a general BLAS implementation. Prefix scan, Tensor Cores, mixed precision, cuBLASLt fusion, CUDA Graphs, asynchronous transfer overlap, multi-GPU execution, and framework bindings are outside v1. Performance was measured on one interactive Windows system with dynamic clocks and warm buffers. No CPU crossover or cross-GPU generalization is claimed.

The implemented v1 scope and GPU validation are complete. Understanding and explaining these results remains a separate task from generating working code; start with one case study and reproduce its measurement.
