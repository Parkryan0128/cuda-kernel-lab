# Reproducing the results

## Verified environment

| Component | Measured setup |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER, compute capability 8.9 |
| SMs / L2 / VRAM | 56 / 50,331,648 bytes (48 MiB) / 12,878,086,144 bytes |
| CPU | AMD Ryzen 5 7600X, 6 cores |
| OS | Native Windows 11, build 26200, WDDM; not WSL |
| Driver | 596.36 |
| Toolkit | CUDA redistribution 13.0.2; nvcc 13.0.88; header version 13000 |
| Loaded runtime | CUDA 13.2, API value 13020, supplied by the Windows display driver |
| CUB / cuBLAS | 3.0.1 / 13.1.0.3 (`cublasGetVersion`: 130100) |
| Host compiler | MSVC 19.39.33523, toolset 14.39.33519 |
| Windows SDK | 10.0.22621.0 |
| CMake / Ninja | 3.31.6 / 1.11.1 |
| GoogleTest | 1.17.0, SHA256-pinned archive in CMake |
| Nsight Compute / Systems | 2025.3.1.0 / 2025.3.2.474 |
| Build | Release, C++17/CUDA C++17, sm_89, `-lineinfo`, no `-G`, no fast math |
| Measurement conditions | Interactive display active; default dynamic clocks/power; warm repeated buffers |

The exact library/API version values, per-process timestamps, commands, relevant environment overrides, and pre/post `nvidia-smi` telemetry are recorded under [`results/release`](../results/release). Telemetry is a pair of observations around a run, not continuous sampling. The desktop and Codex remained active; the machine was not an isolated performance server. Python dependency setup also overlapped the first standard reduction process, so small latency differences should not be treated as controlled hardware limits. Representative comparisons use the later three-process repetitions.

The Toolkit/runtime version difference is intentional reporting, not a mislabeled compiler: starting with CUDA 13 on Windows, `cudaRuntimeGetVersion` reports the runtime packaged with the display driver. NVIDIA documents this in its [version-management API reference](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html). Both the compiled header version and actual runtime value are saved.

Native Windows was selected after checking that WSL was unavailable. A locally extracted toolchain was used during development. Reproduction requires the listed tools on `PATH`; the local tool downloads are not vendored into this repository. Linux build instructions and CPU-hosted CI are included, but the published GPU execution and sanitizer evidence is for this Windows system only.

## Build and correctness

Install CUDA 13.x and a compatible C++17 host compiler. On Windows, open an x64 Developer PowerShell for Visual Studio 2022 with the compiler and Windows SDK initialized; add CUDA DLL and tool directories to `PATH`. Use Ninja for the following single-configuration build:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_FLAGS=--use-local-env
cmake --build build -j 8
ctest --test-dir build --output-on-failure
python tests/test_results.py
```

On Linux with a compatible Toolkit/compiler, omit the Windows-specific `CMAKE_CUDA_FLAGS` option and use the executables without `.exe`. Architecture 89 matches the measured GPU; override it for another supported GPU and treat its measurements as a separate campaign. No other GPU has been validated here.

CMake downloads GoogleTest 1.17.0 with an SHA256 check. For an offline build, download and verify that archive separately, extract it, and provide `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/absolute/path/to/googletest-1.17.0`. This override was needed in the recorded Windows sandbox because CMake's Schannel backend could not initialize credentials; Python downloaded the same verified archive. TLS verification was not disabled.

Run the full GPU validation campaign into a **new** directory:

```powershell
python scripts/validate_gpu.py --exe build/cuda_lab_tests.exe --output results/validation-new
```

It runs all 13 GoogleTests, then each of `memcheck`, `racecheck`, `initcheck`, and `synccheck` with `--gtest_filter=*Boundary.*`. Each sanitizer selects 9 test functions, including loops over every retained custom configuration. These counts are test functions, not a claim of only 9 individual shapes. A zero-test selection, nonzero exit, failed assertion, or missing clean sanitizer summary fails the campaign. In the measured run, all four tools passed; racecheck reported zero errors and zero warnings.

On Windows, profiler counters must be accessible. Follow [NVIDIA's counter permissions instructions](https://developer.nvidia.com/ERR_NVGPUCTRPERM). Compute Sanitizer may also require the administrator setting in the Toolkit's `EnableDebuggerInterface.bat`. Those settings were enabled before this project's validation. A normal CUDA run alone is not proof that either tool works.

## Benchmark campaigns

Run one bounded tuning sweep, then freeze the defaults before evaluating the standard suite:

```powershell
python scripts/run_benchmarks.py --exe build/cuda_lab_bench.exe --phase tuning --output results/tuning-new
python scripts/summarize.py results/tuning-new
python scripts/run_benchmarks.py --exe build/cuda_lab_bench.exe --phase release --output results/release-new
python scripts/summarize.py results/release-new
```

The implemented defaults are R2: 128 threads, 4 accumulators, 4×SM grid cap; G2: 64×64×32. The sweep exposes explicit CLI overrides for all eight R2 and three G2 candidates. It does not rewrite defaults or choose different settings for individual evaluation shapes.

The release runner starts 51 separate processes and produces 200 case series, each with 30 samples:

| Required coverage | Collected cases |
|---|---|
| Reduction, device | R0/R1/R2/CUB × 12 fixed sizes |
| GEMM, device | G0/G1/G2/cuBLAS × 7 fixed shapes; G0 at 2048³ is explicitly skipped |
| Fusion, device | Unfused/fused/library × 3 fixed shapes |
| Representative reduction | R2/CUB at 1,024 and 16,777,216 elements, device and E2E, in 3 independent processes per path/mode |
| Representative GEMM | G2/cuBLAS at 128³ and 1024³, device and E2E, in 3 independent processes per path/mode |
| Fusion repetitions | All 3 paths/shapes, device and E2E, in 3 independent processes per path/mode |
| Transfer-only | H2D and D2H separately for both reduction representatives, both GEMM representatives, and all 3 fusion shapes |

The fixed device reduction sizes and GEMM shapes are defined in `bench/benchmark.cpp`. The largest reduction input is 256 MiB. Allocation/reference work is outside the timer. H2D for GEMM copies A and B; fusion adds bias. D2H copies the complete output (one float for reduction). Host buffers are pinned and reused; no overlap or multi-stream pipeline is claimed.

Each output directory includes metadata, raw samples, before/after correctness evidence, explicit skips, process output, exact arguments, exit status, and telemetry. Existing run outputs are not overwritten. `--resume` skips only completed jobs with identical arguments and source digest. An incomplete job is preserved and requires a fresh campaign directory.

For a single case:

```powershell
build/cuda_lab_bench.exe --op reduce --variant r2 --elements 16777216 --mode device --output results/local-reduce
build/cuda_lab_bench.exe --op gemm --variant g2 --m 257 --n 509 --k 127 --mode e2e --output results/local-gemm
build/cuda_lab_bench.exe --op fusion --variant fused --m 128 --n 512 --k 64 --mode device --output results/local-fusion
build/cuda_lab_bench.exe --help
```

Use fresh names on a second invocation. `--variant all` selects all applicable paths for an operation. `--suite standard`, `representative`, and `tuning` use fixed cases. Inputs use explicit integer-to-float conversion from `mt19937`, seed 2027 for performance, and unit scale. Tests use additional seeds/scales.

## Profiles and figures

Do not profile concurrently with ordinary benchmarking. Counter collection deliberately uses kernel replay, cache flushing (`--cache-control all`), and unchanged clock control (`--clock-control none`). It includes the kernels within the `measured_operation` NVTX range and every reduction pass:

```powershell
python scripts/profile.py --exe build/cuda_lab_bench.exe --tool ncu --output profiles/counters-new
python scripts/profile.py --exe build/cuda_lab_bench.exe --tool ncu --detail --output profiles/detail-new
python scripts/profile.py --exe build/cuda_lab_bench.exe --tool nsys --output profiles/timelines-new
python scripts/summarize_profiles.py profiles
```

The first collection gathers 18 selected reports; the detail collection adds explicit DRAM, shared-bank, barrier, and scoreboard counters for 9 comparisons. Nsight Systems records one reduction E2E trace and a fused/unfused E2E pair. It sets `NSYS_NVTX_PROFILER_REGISTER_ONLY=0` because the marker uses an ordinary NVTX string; without that option, this Nsight version produced no capture. CPU sampling/context-switch collection is disabled. The raw CUDA API, kernel, and memcpy timelines are still available.

The committed CSV exports and [`profiles/summary.json`](../profiles/summary.json) are the public evidence. Binary `.ncu-rep`, `.nsys-rep`, and `.sqlite` files remain local and ignored by Git; the commands recreate them. Binary profiler containers can include unrelated machine/environment details, so the public timeline extract contains only the marked operation's events. Numeric report summaries preserve metric names and units. Missing metrics are not represented as zero.

To regenerate all five figures from the recorded data:

```powershell
python -m pip install -r scripts/requirements.txt
python scripts/plot_results.py --results results/release --tuning results/tuning --output docs/figures
```

`summary.json` is derived from raw samples, and PNG/SVG figures are derived from summaries. Do not edit raw values to change a graph. The reduction/GEMM figures show the standard campaign. Fusion and device/E2E comparisons use the median of three process medians, with process ranges labeled where shown. All reported p10/p90 statistics elsewhere describe batch means, not request-tail latency.

## Evidence boundaries

- Published source/provenance is explained in [`results/provenance.json`](../results/provenance.json).
- The [architecture guide](architecture.md) documents exact versus tolerance-based tests, numerical domains, and unsupported aliases.
- The [case studies](case-studies/reduction.md) distinguish observations from explanations and retain negative results.
- CI compiles CUDA and runs host result-parser checks on a CPU runner. It does **not** certify GPU execution or sanitizer results.
- Full GPU performance measurements were collected on one physical system, with one performance seed and warm reused buffers. No cross-GPU, cold-cache, production-workload, or CPU-versus-GPU crossover claim is made.
