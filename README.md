# CUDA Kernel Performance Lab

> **Status: design and implementation plan.** Only this README exists. The source tree, APIs, commands, tests, and reports below describe planned deliverables. No kernels have been implemented or benchmarked, and no performance results are claimed.
>
> Design baseline: October 13, 2026. Primary focus: preparation for GPU software, kernel, and performance engineering internships.

Implement FP32 reduction and matrix multiplication in CUDA C++, then explain how memory access, synchronization, data reuse, and register usage affect performance. Finish with a small GEMM + bias + ReLU fusion experiment to connect kernel optimization to the cost of a complete operation.

The deliverable is a compact collection of **understandable kernels, reproducible measurements, fair library comparisons, and evidence-backed performance reports**.

For a quick review, start with [scope](#1-v1-scope), [implementation stages](#6-reduction-implementation), [measurement methodology](#11-measurement-methodology), [work budget](#18-milestones-and-work-budget), and [completion criteria](#19-completion-and-stopping-criteria).

## 1. V1 scope

| Area | Required deliverable | Scope boundary |
| --- | --- | --- |
| Reduction | FP32 sum, three custom variants, CUB comparison | One operation; no generic reduction framework |
| GEMM | FP32 `C = A * B`, three custom variants, cuBLAS comparison | Stop at register tiling |
| Fusion | Add column bias and ReLU to the final GEMM store | One epilogue; no neural network framework |
| Correctness | CPU FP64 references, boundary tests, numerical checks, Compute Sanitizer | Explicitly bounded input domain and test suite |
| Measurement | Device-resident, transfer-only, and transfer-inclusive timing | Small command-line harness |
| Analysis | Nsight Compute evidence and Nsight Systems timelines | Representative cases on one GPU |
| Presentation | Three case studies, raw data, four to six figures, reproduction instructions | Explain existing results before adding operations |

**V1 consists of two core operations, one small fusion extension, and shared validation and measurement tools.**

Prefix scan, softmax, attention, and convolution are deferred. Tensor Core kernels, handwritten PTX, multi-GPU execution, and production library packaging are also outside v1.

The target is a finished performance engineering project that can be explained in an interview. Kernel count is not a success metric.

## 2. Why this scope fits the existing background

The project assumes prior experience with distributed backends, work queues, failure handling, data systems, storage engines, and memory management. Its incremental value is evidence of four additional capabilities:

1. Applying systems reasoning to GPU execution and memory hierarchies.
2. Writing parallel C++/CUDA code with explicit synchronization and boundary handling.
3. Investigating performance through hypothesis, measurement, modification, and remeasurement.
4. Comparing a custom implementation with established libraries under documented conditions.

Reduction introduces cooperative execution and input throughput. GEMM adds data reuse and arithmetic throughput. Fusion tests whether removing intermediate memory accesses and a kernel launch improves the complete operation. The actual bottleneck remains a measurement question for each implementation and shape.

This is a project-selection judgment for the stated GPU software direction. It does not establish firmware, driver, or hardware bring-up experience.

NVIDIA's [AI and DL Kernel Libraries internship posting](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Software-Engineer-Intern--AI-and-DL-Kernel-Libraries---2027_JR2026307) identifies CUDA fundamentals, linear algebra, profiling, and optimization as relevant skills. It provides role context, not confirmation of a September 2027 opening or a hiring outcome.

## 3. Questions the finished project must answer

- Why do two correct implementations of the same operation run at different speeds?
- How does problem size change the importance of launch overhead and useful GPU work?
- What changes when a shared-memory reduction tree is replaced with warp shuffles?
- What do shared-memory tiling and register tiling reuse, respectively?
- When does additional register usage help, and when does it hurt?
- Why can kernel improvements have little effect on transfer-inclusive latency?
- Which shapes benefit from bias/ReLU fusion, and which do not?
- Which explanations for the gap to CUB/cuBLAS are supported by measurements?

Answer these questions using code, recorded experiments, and profiler evidence. Do not infer a CPU-versus-GPU crossover from GPU-only measurements; the CPU reference in v1 is a correctness oracle, not an optimized CPU performance baseline.

## 4. Hardware and development environment

### 4.1 Previously verified host information

The following was read from the Windows host using `nvidia-smi` during preparation of the original plan. These values are an observation from that session, not a substitute for recording each benchmark environment.

| Item | Observed value or status |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| Reported total VRAM | 12,282 MiB, approximately a 12 GB-class device |
| Compute capability | 8.9 |
| Host driver | 596.36 |
| Host shell | Windows PowerShell |
| CUDA Toolkit / nvcc | Installation and operation unverified |
| WSL2 / Linux | Installation and operation unverified |
| Nsight Compute / Systems | Installation, operation, and counter access unverified |

Use this GPU for the initial implementation. Purchasing hardware or renting cloud GPUs is not part of the plan. A visible GPU and installed driver do not confirm a working CUDA build or profiling environment.

### 4.2 Default development path

- Use the existing GPU with **Ubuntu under WSL2, CUDA C++, and CMake**.
- If WSL profiling fails, check supported versions and Windows host counter permissions first. Consider native Windows on the same GPU if necessary.
- Choose one working environment for final measurements. Do not combine WSL and native results into one series.
- Use C++17 and CUDA C++17. Use Python for metadata collection, summaries, and plotting.
- Target `sm_89` initially. Supporting other architectures is optional future work.
- Pin the actual verified Toolkit, host compiler, CMake, CUB/CCCL, cuBLAS, and profiler versions.
- Start with the CUB version bundled with the selected Toolkit. A separate CCCL upgrade is not a project requirement.

WSL uses the NVIDIA driver installed on the Windows host. Follow the Toolkit installation path appropriate to WSL rather than installing a separate Linux display driver inside it. See the [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

Check the [Nsight Compute system requirements](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html) and [performance counter permissions guide](https://developer.nvidia.com/ERR_NVGPUCTRPERM) for the selected environment.

### 4.3 Environment gate

Before beginning optimization work:

- [ ] Compile and run a small CUDA sample.
- [ ] Record device properties, including SM count and L2 capacity.
- [ ] Execute one CUB operation and one cuBLAS operation successfully.
- [ ] Run Compute Sanitizer on a small sample.
- [ ] Collect duration, memory, and occupancy information with Nsight Compute.
- [ ] Capture CUDA API, kernel, and memcpy activity with Nsight Systems.

Profiler access is part of the project requirements. Resolve it early rather than discovering at the reporting stage that evidence cannot be collected.

## 5. Data and API contracts

### 5.1 Shared contract

| Property | V1 requirement |
| --- | --- |
| Arithmetic | FP32 input, output, and custom GPU accumulation |
| CPU reference | Convert the actual FP32 inputs to double, then compute in FP64 |
| Layout | Contiguous device buffers with explicit dimensions |
| Aliasing | Public operation inputs, output, and scratch do not overlap |
| Execution | Asynchronous submission to one caller-provided CUDA stream |
| Allocation | Allocate input, output, and scratch before the operation |
| Completion | The caller synchronizes before consuming results or reusing memory elsewhere |
| Numeric domain | Finite inputs within the documented fixtures, without overflow |
| Host errors | Reject invalid dimensions, unsupported configurations, insufficient scratch, and byte-count overflow |
| Exclusions | General strides, arbitrary views, NaN/Inf semantics, multi-GPU execution |

The operation does not scan every device input for finiteness. Numeric constraints are caller preconditions enforced by the fixture generator in tests. General guarantees for subnormal handling and underflow are outside v1.

### 5.2 Planned interface

These declarations describe the intended interface; they are not implemented code.

```cpp
// Small value types select only the finite configurations listed below.
// ReduceConfig contains variant, block_threads, accumulator_count, max_blocks.
// GemmConfig contains variant and a supported tile configuration.
// Resolve SM-dependent max_blocks before querying workspace or launching.

size_t reduce_sum_workspace_bytes(size_t n, const ReduceConfig& config);
void reduce_sum(const float* x, float* out, size_t n,
                void* workspace, size_t workspace_bytes,
                cudaStream_t stream, const ReduceConfig& config);

void gemm(const float* a, const float* b, float* c,
          int m, int n, int k,
          cudaStream_t stream, const GemmConfig& config);

void gemm_bias_relu(const float* a, const float* b,
                    const float* bias, float* y,
                    int m, int n, int k,
                    cudaStream_t stream, const GemmConfig& config);
```

Configuration types are small records, not an autotuning or plugin framework. Workspace queries and launches must use the **same input size and immutable resolved configuration**. This matters because the reduction grid changes the scratch requirement.

Operation definitions:

- Reduction computes `out[0] = sum(x[0:n])`. For `n = 0`, enqueue a write of zero; the output scalar must still be valid.
- GEMM uses row-major `A[M,K]`, `B[K,N]`, and `C[M,N]`, with `alpha = 1` and `beta = 0`. It never reads old C.
- Fusion computes `Y[i,j] = max(0, sum_k(A[i,k] * B[k,j]) + bias[j])`, where bias has N elements.
- After validating dimensions and configuration, `M = 0` or `N = 0` is a successful no-op. No matrix or bias pointer is dereferenced.
- For positive M and N with `K = 0`, GEMM writes zeros and fusion writes `ReLU(bias[j])`. A and B are not read.
- Negative dimensions and overflow in element/byte counts are errors. Use checked 64-bit or `size_t` indexing and offset calculations.
- Validate the resulting grid, block, and shared-memory requirements against the selected device's limits before launch. Reject unsupported shapes explicitly; do not silently narrow dimensions or promise arbitrary shapes just because their byte count fits in memory.
- A pointer may be null only when the corresponding storage is not accessed. Nonempty required output, input, bias, and scratch buffers must be valid.
- Require scratch alignment suitable for its stored types; storage returned by `cudaMalloc` satisfies the intended use. Keep scratch and all operands alive until stream completion.
- Pointer ownership, arbitrary alias detection, and allocation-size introspection are caller responsibilities. Do not build an allocator to infer them.
- Reusing scratch on the same stream is allowed in stream order. Concurrent operations on different streams may not share it without an explicit dependency.

Use `std::invalid_argument` for host contract violations and diagnostic `std::runtime_error` exceptions for CUDA/library errors. Asynchronous errors can also surface at the caller's completion synchronization. Device code does not use C++ exceptions.

The custom fused API accepts G2 configurations only. Reject G0/G1 configurations explicitly. The library and unfused paths are separate benchmark adapters.

## 6. Reduction implementation

### R0: shared-memory tree baseline

**Purpose:** establish a simple, valid parallel baseline.

- Start with 256 threads per block and one input element per thread.
- Threads beyond the input range contribute zero.
- Store values in shared memory and perform a tree reduction.
- Write one partial sum per block.
- Reduce those partial sums in additional launches on the same stream until the result is one device scalar.
- Use disjoint scratch regions for the input and output of each pass. Ping-pong between them when needed; never overwrite partials another block may still read.
- Make all participating threads reach the required barriers, including threads in partial blocks.
- Handle `n = 1` and the final scalar store explicitly; do not construct a recursion that fails to shrink.

Do not artificially weaken the baseline with redundant transfers or a global atomic for every input element.

### R1: warp shuffle reduction

**Purpose:** isolate the effect of reducing shared-memory exchanges and block barriers.

- Initially preserve R0's input partition and 256-thread block size.
- Reduce within each warp using `__shfl_down_sync`.
- Write one sum per warp to shared memory.
- Synchronize the block before the first warp reads these sums.
- The first warp reduces the warp sums; lanes beyond the number of warp sums contribute zero.
- Input-tail threads still participate with zero-valued inputs.
- Use a full-warp mask only where all lanes named by that mask execute the shuffle with valid participating source lanes.

Follow the execution and synchronization rules in the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html).

### R2: thread-local accumulation with a bounded grid

**Purpose:** balance input processing, available parallelism, and the cost of partial sums.

- Use a grid-stride traversal where adjacent threads access adjacent elements.
- Accumulate locally in FP32 before applying R1's block reduction.
- Bound the first-pass grid relative to the device's SM count.
- Test one versus four independent local accumulators, combining them before block reduction. This is the meaning of `accumulator_count`; it does not mean a thread reads only that many values in total.
- Keep tail handling, small-input handling, and the final device reduction.
- Use R1-style follow-up passes with 256 threads and one partial input per thread across all R2 candidates, isolating the first-pass changes.
- Test at most eight configurations: block size `{128, 256}` × accumulator count `{1, 4}` × maximum first-pass blocks `{4 * SM_count, 8 * SM_count}`.
- Bound the actual grid by useful input work as well as `max_blocks`.
- Check whether long accumulation paths change numerical error or instruction dependencies.

Validate every candidate before timing it. Select one documented default; do not create a shape-by-shape exhaustive autotuner.

For nonempty input, make the R2 traversal explicit: `grid = min(ceil_div(n, B), max_blocks)` and `stride = grid * B`, where B is the first-pass block size. Thread `(block, lane)` assigns index `block*B + lane + (t*A + a)*stride` to local accumulator a, with `a` in `[0,A)` and successive nonnegative t. Guard each individual load against n. This partitions input indices without changing the set of visited values when the accumulator count A changes. Use checked arithmetic for launch sizing and index calculations.

The workspace query must use this same pass schedule. A pass producing one partial writes directly to `out`; otherwise, two disjoint scratch regions sized for the first-pass partial count are a sufficient conservative allocation for the custom schedules. Later passes alternate regions until the final scalar is written. Empty/scalar paths must agree with the workspace query and terminate without unnecessary recursion.

### Library comparison and deliverables

Compare against `cub::DeviceReduce::Sum` with FP32 input and output. Query and allocate temporary storage before measurement, and record the bundled version and API used. Check the [CUB DeviceReduce documentation](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html) against the installed headers; that link tracks a development documentation tree.

Use the explicit temporary-storage overload and pass the benchmark stream explicitly. Record the selected API's determinism behavior; keep its ordinary run-to-run policy for the primary comparison where supported. Do not silently substitute a non-deterministic atomic mode or a stronger cross-GPU policy, since these can select different algorithms. No newer CUB API is required solely to expose these options.

Measure **all passes of the custom reduction** against the complete CUB operation. Keep the final result on the device in both paths. Never finish partial sums on the CPU while presenting the measurement as a complete GPU reduction.

Record correctness, latency by size, CUB-relative throughput, scratch size, and launch count. Custom variants must implement their own reduction; CUB is a separate baseline.

## 7. GEMM implementation

### G0: one thread per output

**Purpose:** establish the computation and row-major indexing.

- One thread computes one `C[i,j]` using FP32 accumulation across K.
- Map thread x to output columns.
- Begin with small rectangular matrices and dimensions that are not tile multiples.
- Avoid unnecessary transposes, transfers, or deliberately poor indexing in the baseline.

### G1: shared-memory tiling

**Purpose:** reuse A and B values across threads in a block.

- Start with a `16 x 16` output tile and a `16 x 16` thread block.
- Cooperatively load A and B tiles into shared memory.
- Synchronize after loading, accumulate the tile product, then synchronize before overwriting shared memory for the next K tile.
- Zero-fill out-of-range loads and guard output stores.
- Do not return boundary threads early when other threads will execute a block barrier.

### G2: register tiling

**Purpose:** reuse staged values across multiple outputs owned by each thread.

| Parameter | Initial configuration |
| --- | --- |
| Output block tile | `BM = 64`, `BN = 64` |
| K tile | `BK = 16` |
| Thread block | `16 x 16`, or 256 threads |
| Outputs per thread | `TM = 4`, `TN = 4`, or 16 FP32 accumulators |
| Output row | `block_row + ty + r * 16` |
| Output column | `block_col + tx + c * 16` |
| Initial r and c range | `0..3` |

Design cooperative global loads separately from output ownership so that adjacent loading threads access adjacent elements. Each thread loads selected shared-memory values into registers and reuses them across its accumulators.

- Inspect compiler-reported register usage and local-memory spills.
- Investigate suspected shared-memory bank conflicts through both address mapping and profiler evidence.
- Evaluate occupancy together with latency and data reuse. Higher occupancy is not an independent optimization objective.
- Test at most three `(BM,BN,BK)` configurations: `(64,64,16)`, `(64,64,32)`, and `(64,128,16)`.
- The third configuration uses `TM = 4`, `TN = 8` with the same 256-thread layout and must pass its own tests.
- Choose one default and retain another only if it helps explain a measured trade-off.

**Stop v1 GEMM optimization at G2.** Handwritten PTX, asynchronous copy pipelines, double buffering, and Tensor Core implementations belong to a separately scoped follow-up.

### cuBLAS comparison contract

- Compute the same FP32 `C = A * B`, with `alpha = 1`, `beta = 0`.
- Use `cublasGemmEx`, FP32 storage, and `CUBLAS_COMPUTE_32F_PEDANTIC`.
- Configure one handle once per benchmark stream: set the stream first, then any explicit workspace, and record math mode and relevant environment overrides. `cublasSetStream` resets user-provided workspace, so do not call it after `cublasSetWorkspace` unless the workspace is reapplied. Exclude configuration and initialization from timing; the default library workspace pool is acceptable after warm-up.
- Label results as the specified FP32 comparison. They are not a claim about the fastest mixed-precision library configuration.
- Avoid a separate matrix transpose. For positive dimensions, call column-major GEMM with both operations `CUBLAS_OP_N`, operands B then A, dimensions `(N,M,K)`, and leading dimensions `(N,K,N)`.
- Validate this mapping on asymmetric matrices. Handle empty dimensions in the adapter before calling cuBLAS.

The mapping follows `C^T = B^T * A^T`. Layout and compute-type semantics are documented in the [cuBLAS reference](https://docs.nvidia.com/cuda/cublas/index.html). Matching precision does not imply matching arithmetic order or bitwise results.

## 8. Bounded extension: GEMM + bias + ReLU

The operation is:

```text
Inputs:  A[M,K], B[K,N], bias[N]
Output:  Y[M,N]
Y[i,j] = max(0, (A * B)[i,j] + bias[j])
```

Implement three paths:

| Path | Implementation | Purpose |
| --- | --- | --- |
| Custom unfused | G2 GEMM, then one combined bias/ReLU kernel | Direct control for the fusion experiment |
| Custom fused | G2 with bias/ReLU applied before the final store | Remove an intermediate round trip and one launch |
| Library pipeline | FP32 cuBLAS GEMM, then the same bias/ReLU kernel | Practical library-based comparison |

The unfused path can write GEMM output to Y and apply the pointwise epilogue in place. That internal operation has its own in-place contract; public A/B/bias inputs still do not overlap Y.

Do not split bias and ReLU into two separate kernels to inflate the fusion benefit. A cuBLASLt fused epilogue is outside v1, so the library pipeline is not a comparison against every possible vendor fused implementation.

Hold the custom GEMM tile configuration, inputs, and precision fixed when comparing fused and unfused paths. Verify the same mathematical result within the numeric contract, rather than requiring general bitwise equality.

Investigate:

- Whether added instructions or register pressure slow the GEMM portion.
- Whether small K or small outputs make launch removal more valuable.
- How much of a large GEMM's total time the epilogue accounts for.
- Whether device-resident gains survive H2D and D2H transfers.

Fusion does not have to win on every shape. This is the final feature addition in v1.

## 9. Correctness plan

### 9.1 Separate structural errors from floating-point error

Floating-point summation depends on evaluation order. General CPU FP64, CUDA FP32, and library results need not be bitwise equal. See NVIDIA's [Floating Point and IEEE 754 guide](https://docs.nvidia.com/cuda/floating-point/index.html).

Use two categories of fixtures:

1. **Exact fixtures:** zeros, one-hot inputs, identity matrices, and small integers whose intermediate operations remain exactly representable. These catch missing work, duplicate work, and indexing errors.
2. **Numerical fixtures:** seeded FP32 random inputs, cancellation cases, and controlled scaling, checked against CPU double references.

Record absolute error, normalized error, and the worst failing index. Unexpected NaN/Inf is an immediate failure. Do not depend exclusively on relative error near a zero reference.

Test host contract violations separately: negative dimensions, overflow, unsupported configurations, insufficient scratch, and required null pointers. Test valid empty cases separately from invalid arguments.

### 9.2 Reduction fixtures

Boundary sizes:

```text
0, 1, 2, 31, 32, 33,
127, 128, 129, 255, 256, 257,
511, 512, 513, 1023, 1024, 1025
```

Larger correctness sizes: `2^20`, `2^20 + 17`, and `2^24 + 13`.

Add targeted exact/one-hot tests at `256^2 - 1`, `256^2`, and `256^2 + 1`, where the R0/R1 pass count changes. For each R2 configuration, test around `B*max_blocks` and `A*B*max_blocks` with offsets `-1, 0, +1`; these exercise grid capping and a partially filled accumulator group. These are small targeted structural tests, not another full benchmark sweep.

Include:

- All zeros and single positive/negative values.
- One-hot inputs at the first, middle, last, and tile-boundary positions.
- Integer fixtures satisfying `sum(abs(x)) <= 2^24` so that all integer partial sums remain representable.
- `U(-1,1)`, `U(0,1)`, cancellation patterns, and mixtures of magnitudes.
- Input scale factors `{2^-8, 1, 2^8}` for bounded scaling tests.
- Repeated calls using the same output and scratch allocations but changing input values, including the last element and zero/nonzero patterns. Verify after each call to catch stale partials or output. Also exercise changing input sizes within the same sufficiently large allocations, re-querying the required workspace for each size/configuration.

Validate R0, R1, R2, and CUB independently against a CPU double sum. Agreement with CUB alone is insufficient.

For a fixture created by scaling base inputs by a positive factor `s`, use the starting acceptance rule:

```text
ref   = sum(double(x[i]))
scale = sum(abs(double(x[i])))
abs(gpu - ref) <= 1e-6 * s + 5e-6 * scale
```

This is a fixture-level acceptance threshold, not a universal forward-error theorem. It can be too permissive to detect one omitted element in a large array; exact and one-hot tests are therefore mandatory.

Do not relax tolerances silently after seeing results. A revision requires a saved failing input, analysis of the accumulation path, and a documented change before final benchmarking.

### 9.3 GEMM and fusion fixtures

Include valid zero/one dimension cases and these `(M,N,K)` shapes:

```text
(7,13,5), (15,17,31), (31,33,65),
(63,65,17), (127,129,33)
```

Also include:

- Zeros, identity, one-hot, and small signed integer matrices.
- Asymmetric matrices to detect swapped operands or incorrect leading dimensions.
- At least three fixed random seeds using `U(-1,1)`.
- Controlled scale factors `sA,sB` chosen from `{2^-8, 1, 2^8}` on small fixtures.
- Positive, negative, and zero bias; scale bias by `sA*sB` in scaled fixtures.
- Values on both sides of the ReLU threshold.
- Repeated writes to a reused output buffer with alternating inputs and bias, checking each result. Include a nonzero result followed by a zero result to catch stale output and unintended accumulation.

Use full CPU FP64 references when every dimension is at most 129. Convert operands to double **before multiplication**. Apply bias and ReLU in double for the fusion reference.

For ordinary and scaled fixtures, define:

```text
S = abs(sA * sB)           # S = 1 for unscaled fixtures
atol = 1e-4 * S
rtol = 1e-4

abs(gpu[i,j] - ref[i,j]) <= atol + rtol * abs(ref[i,j])

Also report:
max_absolute_error
relative_L2_error = norm(gpu - ref) / max(norm(ref), 1e-12)
worst_element_index
```

For cancellation stress cases, also record `sum_k(abs(double(A[i,k]) * double(B[k,j])))` for examined outputs. A failed tolerance remains a correctness failure pending analysis; cancellation is not an automatic exemption.

For large performance cases:

1. Compare the complete custom output with the complete library output. Use a documented pairwise allowance of `2 * (atol + rtol * abs(library[i,j]))`, acknowledging that both are FP32 results.
2. Independently compute up to 128 CPU FP64 output elements selected from corners, tile boundaries, and fixed-seed random positions.
3. Check both custom and library values at those positions against the normal CPU-reference tolerance.
4. Label this **full library cross-check plus sampled CPU reference**. Do not describe it as full FP64 validation of the large matrix.

The larger pairwise allowance is a diagnostic comparison policy, not an accuracy guarantee or a replacement for CPU-reference tests.

### 9.4 Memory and synchronization checks

Run the following on a small boundary suite covering every custom configuration and the epilogue:

- `memcheck`: invalid accesses, out-of-bounds accesses, and alignment errors.
- `racecheck`: shared-memory hazards.
- `initcheck`: uninitialized device global-memory reads with the default configuration.
- `synccheck`: invalid synchronization usage.

Run memcheck first. Newer tool versions can expose additional address-space checks; record the options actually supported and used. See the [Compute Sanitizer guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

Use deliberate output sentinels in structural tests to detect missing stores. Keep a separate initcheck path without blanket initialization of scratch, so missing initialization is not hidden by the test harness.

Sanitizer runs are separate from performance runs. A clean report does not prove the absence of every possible race; review barrier placement, scratch lifetimes, and stream ordering in the code as well.

## 10. Fixed benchmark suite

### 10.1 Reduction

```text
256, 1024, 4096, 16384,
65536, 262144, 1048576, 1048593,
4194304, 16777216, 16777229, 67108864
```

Small inputs expose launch and final-pass overhead. Medium inputs expose cache and parallelism changes. For large inputs, check the working set against measured L2 capacity before making DRAM-related claims.

The largest FP32 input is 256 MiB. Calculate total live memory, including scratch and reference buffers, before allocating. Use one fixed performance seed and additional seeds for correctness.

### 10.2 GEMM

| M | N | K | Purpose |
| ---: | ---: | ---: | --- |
| 128 | 128 | 128 | Small operation and launch overhead |
| 512 | 512 | 512 | Medium case |
| 1024 | 1024 | 1024 | Common comparison case |
| 2048 | 2048 | 2048 | Larger optimized/library comparison |
| 1024 | 2048 | 512 | Rectangular reuse pattern |
| 257 | 509 | 127 | Tail handling and irregular dimensions |
| 128 | 512 | 64 | Small K and fusion candidate |

Run G0 initially only where the checked product `M*N*K <= 1024^3`. Mark excluded cases `skipped_by_policy`; compare G1/G2/cuBLAS there. Start pilot runs with small cases so a slow baseline does not create unnecessarily long work on the display GPU.

Never report a speedup against a variant that was not measured on that case.

### 10.3 Fusion

```text
(M,N,K) = (128,512,64), (512,1024,128), (1024,1024,1024)
```

Use all three paths from Section 8 on these cases.

### 10.4 Bounded tuning set

Use reduction sizes `2^20` and `2^24`, and GEMM shapes `(512,512,512)` and `(1024,1024,1024)`, to select defaults. Evaluate the fixed default on the remaining cases without retuning each one. Record if any departure from this policy becomes necessary.

The full suite remains the reported result, including regressions and skipped cases.

### 10.5 Required measurement coverage

| Measurement | Required cases | Compared paths |
| --- | --- | --- |
| Device-resident reduction | All Section 10.1 sizes | R0, R1, R2, CUB |
| Device-resident GEMM | All Section 10.2 shapes | G0, G1, G2, cuBLAS, subject to the G0 skip policy |
| Device-resident fusion | All three Section 10.3 shapes | Unfused, fused, library |
| Reduction E2E | `1024` and `2^24` elements | Selected R2 default and CUB |
| GEMM E2E | `(128,128,128)` and `(1024,1024,1024)` | Selected G2 default and cuBLAS |
| Fusion E2E | All three Section 10.3 shapes | Unfused, fused, library |
| Transfer-only | The input/output transfer sequences for those E2E cases | One measurement per distinct sequence, independent of kernel variant |

Keep H2D and D2H results separate. Measure the same sequence of copies used by its E2E case, including separate A/B/bias buffers. Do not require the Cartesian product of every timing mode, tuning candidate, and input size. For the three independent process reruns, use the E2E cases above and their corresponding device-resident comparisons as the representative set.

## 11. Measurement methodology

### 11.1 Separate three timing scopes

| Mode | Included work | Excluded work |
| --- | --- | --- |
| `device` | Complete GPU operation, including every reduction pass or unfused kernel | Transfers, allocation, generation, validation |
| `transfer` | H2D or D2H using preallocated pinned host buffers | Allocation, kernels, generation |
| `e2e` | H2D, complete operation, D2H, and completion synchronization | Process/context startup, allocation, generation |

E2E means **one request with reusable preallocated buffers**. It is not cold-start application latency.

- Measure device and transfer scopes using CUDA events on the operation's stream; synchronize the stop event before reading elapsed time.
- Use one explicit non-default stream for custom kernels, CUB, cuBLAS, and copies. Keep required dependencies on that stream and verify the actual stream assignment in the timeline.
- For E2E, ensure previous work is finished before starting a host monotonic clock, then stop it after the final stream synchronization.
- Reduction transfers its input and a single output scalar. GEMM transfers A, B, and the full output. Fusion additionally transfers bias.
- Use the same transfer policy for custom and library paths.
- Submit one E2E request at a time. Transfer/compute overlap and throughput pipelines are outside v1.
- Record pinned-buffer usage. Excluding pinned allocation does not support claims about allocation-inclusive cost.

### 11.2 Sampling procedure

1. Prepare context, stream, buffers, handles, and scratch.
2. Validate the selected case/configuration before timing. Mark failures explicitly.
3. Perform at least ten warm-up operations per case/variant.
4. Use pilot timing to choose repetitions, and record the choice.
5. For device and transfer modes, target approximately 20 ms per sample by repeating operations in stream order. Cap repetitions at 10,000.
6. Keep 30 raw samples by default, including batch duration and per-operation average. Mark samples that remain short at the repetition cap.
7. For E2E, each of the 30 samples completes one actual request.
8. Report the median and p10/p90. Quantiles of batch averages are not per-request tail-latency quantiles.
9. Recheck representative results in three independent process runs and preserve each run's median.
10. Interleave or seed-randomize variant order to reduce temperature and ordering bias.
11. Check the output again after the measured sequence, outside the timed region.

For very short kernels, host submission gaps can appear inside the CUDA-event interval. Describe this as device-stream operation timing, not perfectly isolated instruction execution. Use the timeline to investigate small-case behavior. CUDA Graphs remain follow-up work.

Avoid concurrent games, video processing, or unrelated CUDA jobs during final runs. Record clock, power, temperature, and utilization when available; otherwise record `unavailable`. Clock locking is not a completion requirement.

### 11.3 Cache policy

The default policy is **warm, repeated-buffer**.

- Reuse the same buffers and record working-set bytes and L2 capacity.
- Small-case effective throughput may reflect caching rather than DRAM traffic.
- Analyze DRAM behavior using sufficiently large inputs and measured traffic.
- Do not build a cache-flushing subsystem for v1.
- Record profiler replay, cache-control, and clock-control settings separately from ordinary benchmark conditions.

Nsight Compute may alter cache and timing behavior through replay and collection. Its host/event measurements must not replace unprofiled benchmark timing. See the [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

### 11.4 Always outside timed regions

Keep context/handle creation, `cudaMalloc`/`cudaFree`, CUB workspace queries, random generation, CPU references, validation, logging, CSV/JSON writes, and plotting outside measurement. Do not add per-pass device synchronization inside a multi-pass reduction; stream ordering already provides the dependency.

## 12. Metrics and interpretation

Use the median per-operation elapsed time for summary throughput. Keep raw samples so the calculation can be reproduced.

### Reduction

```text
latency_us = elapsed_seconds * 1e6
input_GB_per_s = (N * sizeof(float)) / elapsed_seconds / 1e9
speedup_vs_R0 = median_R0 / median_custom
fraction_of_CUB_throughput = median_CUB / median_custom
```

Input GB/s excludes scratch traffic and the final store. It is an input-processing metric, not measured DRAM bandwidth. If reporting logical traffic across passes, calculate and label that separately.

### GEMM

```text
GFLOP_per_s = (2.0 * M * N * K) / elapsed_seconds / 1e9
speedup_vs_G0 = median_G0 / median_custom
fraction_of_cuBLAS_throughput = median_cuBLAS / median_custom
```

Use `2MNK` as the conventional GEMM work estimate. For fusion, label the analogous rate GEMM-equivalent GFLOP/s; it does not count every epilogue operation. Empty cases have no meaningful throughput result.

### Fusion

```text
fusion_speedup = median_custom_unfused / median_custom_fused
latency_saved_us = median_custom_unfused - median_custom_fused
```

Removing an intermediate FP32 matrix store and reload removes approximately `8*M*N` logical bytes relative to the unfused flow. Cache behavior can make the observed DRAM reduction different.

### Reporting rules

- Keep GB (`10^9` bytes) distinct from GiB (`2^30` bytes). See [CUDA Best Practices: effective bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#effective-bandwidth-calculation).
- A library-relative throughput fraction of 0.5 means half the library's throughput for that case and timing scope.
- Compare the same operation, dimensions, inputs, precision, cache policy, and timing scope.
- Show per-case results before any aggregate summary. Preserve slower cases.
- Identify the tuning set and the evaluation set.
- Keep profiler-derived observations separate from unprofiled performance numbers.
- Never describe naive-baseline speedup as speedup over cuBLAS or CUB.

## 13. Profiling and experiment design

### 13.1 Nsight Compute: kernel behavior

Profile representative cases rather than every metric on every input:

- Reduction: one small case and one case larger than L2.
- GEMM: `1024^3`, one rectangular case, and one tail case.
- Fusion: one small-K case and `1024^3`.

Inspect duration, memory traffic/cache behavior, register and shared-memory usage, occupancy, and compute activity. Discover metric names supported by the installed tool/GPU and save the exact collection command.

| Change | Hypothesis | Evidence to inspect | Possible counterexample |
| --- | --- | --- | --- |
| R0 to R1 | Fewer shared exchanges and barriers | Instructions, shared accesses, duration | Input traffic or launch overhead dominates |
| R1 to R2 | Fewer partials and follow-up costs | Grid, pass count, traffic, duration | Longer dependency chains or less parallelism |
| G0 to G1 | Greater reuse of A/B | Memory workload, cache, duration | Staging/barriers dominate small cases |
| G1 to G2 | Greater reuse within each thread | Registers, spills, compute, duration | Register pressure or bank conflicts |
| Unfused to fused | Fewer launches and intermediate accesses | Complete latency, launch count, traffic | Added epilogue cost or GEMM dominance |

When useful, use a roofline corresponding to the FP32 execution path. Do not use Tensor Core peak throughput as the ceiling for a CUDA-core FP32 kernel. A roofline is supporting evidence, not proof of a bottleneck by itself. See [Nsight Compute analysis features](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html).

### 13.2 Nsight Systems: operation timeline

- Add NVTX ranges for H2D, the operation, and D2H.
- Verify the actual number of reduction passes and unfused/fused launches.
- Inspect unexpected synchronization, launch gaps, and transfer contribution.
- Preserve one representative reduction E2E trace and a before/after fusion pair.
- Use timelines to explain behavior; obtain final timing without profilers attached.

See the [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

### 13.3 Experiment log

For each meaningful change, record:

```text
Problem, input, and previous variant:
Hypothesis and prior evidence:
Single intended change:
Correctness result:
Unprofiled before/after timing and variability:
Profiler observations:
Cases that did not improve:
Keep, revise, or discard, with reasons:
```

Aim to explain three optimization transitions and at least one non-improvement, regression, or remaining limitation. The findings must reflect actual experiments; a prescribed number of successful speedups is not a release requirement.

If adapting an implementation from a paper, sample, or tutorial, identify the source and distinguish the borrowed algorithm from the project's own implementation choices and experiments.

## 14. Planned repository structure

```text
cuda-kernel-performance-lab/
├── README.md
├── CMakeLists.txt
├── CMakePresets.json
├── include/cuda_lab/
│   ├── ops.hpp
│   └── cuda_check.hpp
├── src/
│   ├── reduction.cu          # R0/R1/R2 and workspace sizing
│   ├── gemm.cu               # G0/G1/G2 and the fused G2 epilogue
│   ├── epilogue.cu           # Standalone in-place bias/ReLU
│   └── library_baselines.cu  # CUB and cuBLAS adapters
├── bench/
│   ├── main.cpp
│   ├── benchmark.cu          # Events, E2E timing, and sampling
│   └── cases.hpp             # Small fixed case definitions
├── tests/
│   ├── cpu_reference.cpp
│   ├── test_reduction.cu
│   └── test_gemm.cu
├── scripts/
│   ├── collect_env.py
│   ├── summarize.py
│   └── plot_results.py
├── docs/
│   ├── environment.md
│   ├── reduction.md
│   ├── gemm.md
│   ├── fusion.md
│   └── experiment-log.md
├── results/
│   └── <run-id>/
│       ├── environment.json
│       ├── correctness.json
│       ├── samples.csv
│       ├── summary.csv
│       └── figures/
├── profiles/                 # Selected reports; large files as release artifacts
└── .github/workflows/
    └── ci.yml
```

Files may be combined where that improves readability. Keep variant differences traceable and benchmark logic separate from kernels. Avoid generic registries, plugin systems, extensive template frameworks, web services, and databases.

## 15. Shared tooling to implement

- Small move-only RAII wrappers for device buffers, streams, and events.
- CUDA/cuBLAS diagnostics with the failing API and source location.
- Launch-error checks and asynchronous-error checks at completion.
- A small CLI for operation, variant/configuration, size, timing mode, seed, samples, and output directory.
- Fixed correctness and performance suites.
- CPU references and deterministic fixture generation.
- Raw sample export plus Python summaries and plots.
- Run IDs connecting metadata, correctness, and measurements.

Use GoogleTest with CTest. A small custom benchmark harness provides explicit control over CUDA events and E2E timing. Pin dependencies and prioritize readable code over abstraction or line count.

## 16. Intended usage: commands are not implemented yet

These are planned Linux/WSL commands. The referenced executables and scripts do not exist at the design stage. Verify and update every command once implementation exists.

### Build and tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Use optimized Release kernels with `-lineinfo`, and preserve compiler register-usage reports. Do not benchmark a device-debug `-G` build. Do not enable `--use_fast_math` in the default comparison build.

### Representative benchmarks

```bash
./build/cuda_lab_bench --op reduce --variant all --suite standard --mode device --output results/reduce-device
./build/cuda_lab_bench --op gemm --variant all --suite standard --mode device --output results/gemm-device
./build/cuda_lab_bench --op fusion --variant all --suite standard --mode e2e --output results/fusion-e2e
python scripts/summarize.py results/
python scripts/plot_results.py results/
```

Define `all` as `r0,r1,r2,cub` for reduction; `g0,g1,g2,cublas` for GEMM; and `unfused,fused,library` for fusion. Apply the documented G0 skip policy. `--suite standard` uses the fixed cases from Section 10 and the selected default configurations.

These commands illustrate the interface; they are not a complete release-validation script. Final reproduction instructions must include every required row in Section 10.5.

### Compute Sanitizer

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/cuda_lab_tests --gtest_filter='Boundary.*'
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/cuda_lab_tests --gtest_filter='Boundary.*'
compute-sanitizer --tool initcheck --error-exitcode 1 ./build/cuda_lab_tests --gtest_filter='Boundary.*'
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/cuda_lab_tests --gtest_filter='Boundary.*'
```

Create a `Boundary.*` suite that actually selects every custom configuration's small-input, tail, and synchronization tests. Treat a zero-test selection as an invocation error rather than a passing sanitizer run.

### Profilers

```bash
mkdir -p profiles
ncu --set basic --export profiles/gemm-g2 ./build/cuda_lab_bench --op gemm --variant g2 --m 1024 --n 1024 --k 1024 --profile-once
nsys profile --trace=cuda,nvtx --output profiles/fusion-e2e ./build/cuda_lab_bench --op fusion --variant fused --m 128 --n 512 --k 64 --mode e2e --profile-once
```

`--profile-once` is a planned project flag, not a profiler flag. It runs one marked operation after setup/warm-up instead of the statistical loop. Setup activity may still be traced; distinguish the target with NVTX ranges and kernel names. Include every pass when analyzing a reduction.

The basic `ncu` command is a starting point, not the full analysis collection. Select supported sections for memory, occupancy, and compute analysis, and save their exact settings. See the [Nsight Compute CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

## 17. Data and reproducibility

### 17.1 Environment metadata

```text
run_id, timestamp, git_commit, git_dirty
GPU name, compute capability, SM count, L2 bytes, VRAM bytes
driver, CUDA runtime, CUDA Toolkit, CUB/CCCL, and cuBLAS versions
Nsight Compute and Nsight Systems versions
OS, WSL/native, CPU model, host compiler, build flags
math mode, precision, CUB determinism policy, relevant environment overrides
stream policy, cache policy, profiler replay/clock/cache settings
power, clock, temperature, and background-load notes
```

### 17.2 Benchmark rows

```text
run_id, operation, variant, mode
n_elements OR m/n/k, seed, distribution, input scale
block configuration, tile configuration, workspace bytes, pass count
sample index, repetitions, total elapsed ms, per-operation us
input bytes, output bytes, correctness status, skip reason
```

Compute summaries from raw samples. Do not modify raw data to match a graph. Profiler reports use separate run IDs and link to unprofiled measurements from the same commit.

### 17.3 CI boundary

- CPU runners check host utilities, result parsing, formatting, and CUDA build-only checks when a supported Toolkit is available.
- CPU-only CI must not claim GPU execution or sanitizer coverage.
- Preserve local GPU correctness, sanitizer, and representative benchmark logs.
- Do not use a fixed speedup threshold as a CI pass/fail condition.
- Self-hosted GPU CI is optional follow-up work.

## 18. Milestones and work budget

These are planning budgets for scope control, not guaranteed completion times. CUDA experience and environment issues can change them.

| Phase | Work | Exit evidence | Focused work |
| --- | --- | --- | ---: |
| 0 | Environment, sample, and profiler checks | Working tool logs | 4–6 h |
| 1 | Minimal harness and CPU references | Small-operation correctness and timing | 10–14 h |
| 2 | R0/R1/R2 and CUB | Reduction tests, plots, initial analysis | 14–20 h |
| 3 | G0/G1/G2 and cuBLAS | GEMM tests, plots, initial analysis | 28–38 h |
| 4 | Bias/ReLU fusion | Three-path results and timelines | 8–12 h |
| 5 | Reruns, reports, and cleanup | Reproducible final artifacts | 16–22 h |

The focused-work total is **80–112 hours**. Budget approximately **100–140 hours** including learning and debugging, or roughly 10–14 weeks at ten hours per week.

Implementation order:

1. Establish R0 correctness and measurement before R1.
2. Keep evidence for each R1/R2 transition.
3. Validate G0 and the cuBLAS layout mapping before G1.
4. Pass G1 tail tests before G2.
5. Choose a stable G2 configuration before fusion.
6. Finish with reproduction and explanation rather than another feature.

The phase-1 GPU sample exists only to verify the harness. It does not become a separate vector-operation project. If work finishes early, use the remaining time to improve the evidence and explanation.

## 19. Completion and stopping criteria

### 19.1 Functional completion

- [ ] R0/R1/R2 and the CUB adapter meet their contracts.
- [ ] G0/G1/G2 and the FP32 cuBLAS adapter meet their contracts.
- [ ] Custom unfused, custom fused, and library fusion paths compute the same defined operation.
- [ ] All retained configurations pass boundary, exact, and numerical fixtures.
- [ ] Representative large cases pass the documented validation procedure.
- [ ] The four sanitizer tools pass on the small boundary suite with a nonzero selected test count.
- [ ] Workspace/configuration consistency, pass-count boundaries, changed-input buffer reuse, and host error cases are tested.

### 19.2 Performance analysis completion

- [ ] Raw samples, summaries, and environment metadata cover the measurement matrix in Section 10.5.
- [ ] Reduction timing includes all passes.
- [ ] GEMM comparisons match layout, precision, and math-mode conditions.
- [ ] Device-resident and transfer-inclusive results are separate.
- [ ] Representative findings are checked in three independent runs.
- [ ] Nsight Compute evidence and Nsight Systems timelines are preserved.
- [ ] At least one non-improvement or remaining limitation is explained.
- [ ] A reader can determine what each measurement includes and excludes.

### 19.3 Portfolio completion

- [ ] Reduction, GEMM, and fusion each have a focused case study.
- [ ] Four to six figures identify input, hardware, units, precision, and timing scope.
- [ ] The finished README leads with actual status, results, and reproduction commands.
- [ ] Planned and completed work are clearly distinguished.
- [ ] The author can trace one kernel's indexing and synchronization on a small input.
- [ ] The author can explain a before/after result using code and a graph in five minutes.

### 19.4 Performance goals and stop rules

Aim for R2 to improve over R0 on useful large inputs, and G2 to improve over G0/G1 on useful matrix shapes. Do not promise a particular CUB percentage, cuBLAS percentage, or speedup before measuring.

If expected improvements never appear, first investigate timing errors, indexing, spills, and launch configuration. Once the implementation and measurement are understood, a disappointing result can still be reported honestly with an explanation.

Stop tuning a bottleneck after two or three evidence-driven attempts yield changes smaller than repeat-run variability. Do not expand scope indefinitely to reach an arbitrary number.

If time is constrained, reduce optional tile candidates, extra figures, or supplementary telemetry first. Preserve correctness, library comparisons, E2E distinctions, and profiler evidence. Omitting G2 or fusion means a deliberately reduced interim release, not completion of this v1 plan.

## 20. Final reports and figures

### Reduction report

- R0/R1/R2 algorithms and pass structure.
- Latency and input throughput by size.
- CUB-relative throughput and small-input behavior.
- Observed effects of shuffles and thread-local accumulation.

### GEMM report

- A small diagram of thread output ownership and cooperative loads.
- Shared-memory reuse versus register reuse.
- G0/G1/G2/cuBLAS results by shape.
- Register pressure, occupancy, spills, memory traffic, and their relationship to latency.

### Fusion report

- Unfused and fused computation/memory flows.
- Latency and absolute savings for the three fixed shapes.
- Device-resident versus E2E effects.
- Cases with limited benefit and the gap to the library pipeline.

A useful figure set is two reduction plots, two GEMM plots, one fusion comparison, and one device-versus-E2E comparison. Each figure must trace back to raw samples. Do not create placeholder performance bars or invented speedup values.

## 21. Resume and interview evidence

Before completion, describe the work as in progress. After completion, claim only implemented and verified capabilities.

These are templates for future use, not current accomplishments:

> Implemented and profiled CUDA C++ reduction and FP32 GEMM kernels using warp-level primitives, shared-memory tiling, and register tiling; validated against CPU references and benchmarked against CUB and cuBLAS on an RTX 4070 SUPER.

> Reduced [specific operation] latency from [X] to [Y] for [shape and timing scope], supported by profiler evidence identifying [observed cause]; evaluated transfer-inclusive performance and cases with limited benefit.

Include the baseline and workload when quoting a speedup.

Be prepared to explain:

- Warp/block roles and dependencies between reduction passes.
- Why a conditional return can invalidate block synchronization.
- Shuffle participation masks and tail handling.
- Row-major GEMM mapping into cuBLAS.
- The relationship between register tiling, register pressure, and occupancy.
- Floating-point differences between evaluation orders and precisions.
- Why all passes belong inside operation timing.
- Pinned buffers, host transfers, and device-resident measurements.
- Which profiler observations support a hypothesis and which leave uncertainty.
- Why v1 stops at this scope.

AI assistance may support drafts, fixture ideas, and documentation lookup. The author must be able to explain and manually trace the accepted indexing, synchronization, numerical assumptions, and optimization decisions.

## 22. Extensions after v1

Select at most one as a new project goal after completing v1.

| Direction | Extension | Additional evidence |
| --- | --- | --- |
| Deep learning kernels | FP16/BF16 Tensor Core GEMM or CUTLASS comparison | Tensor execution and precision/throughput trade-offs |
| GPU primitives | Prefix scan, then stream compaction | Prefix dependencies across blocks |
| Pipeline performance | Pinned buffers and multiple streams | Transfer/compute overlap and throughput versus latency |
| Very small operations | CUDA Graphs comparison | Execution strategies for launch overhead |

A second GPU, Jetson support, PyTorch integration, Triton, attention, or distributed training does not become part of v1 by default.

## 23. Current status and review boundary

| Item | Status |
| --- | --- |
| Project scope and implementation guide | Written in English and reviewed for consistency |
| Host GPU identity, memory, and compute capability | Previously verified through read-only host queries |
| CUDA development and profiling environment | Unverified |
| Kernels, tests, and benchmark harness | Not started |
| Performance results and profiler reports | None |

The final pre-implementation review covered scope, operation contracts, reduction traversal and scratch scheduling, row-major cuBLAS mapping, library configuration, numerical validation, pass-count boundaries, buffer reuse, timing coverage, metric definitions, and completion criteria. Relevant NVIDIA documentation was checked. No unresolved design blocker was identified for beginning the phase-0 environment gate; tool compatibility, numeric thresholds on the chosen builds, and performance still require implementation-time verification. This is a review of the plan, not certification of code correctness or achievable performance.

**The next implementation step is the phase-0 environment gate. No package installation, kernel implementation, or GPU benchmark execution was performed while writing or reviewing this README.**
