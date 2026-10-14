# Implementation and reasoning guide

The scope is FP32 sum reduction, row-major FP32 GEMM, and GEMM + column bias + ReLU. There is no prefix scan, Tensor Core path, multi-GPU scheduler, framework binding, or shape-specific autotuner. The [original plan](implementation-plan.md) records the design decisions made before implementation.

## Operation contracts

The public API is in [`include/cuda_lab/ops.hpp`](../include/cuda_lab/ops.hpp). All device operations enqueue work on the caller's explicit stream. Inputs, outputs, and scratch must be device-accessible, adequately sized, and non-overlapping, except the explicitly in-place epilogue. The caller owns their lifetime and synchronizes before reading results or reusing buffers on another stream. Finite FP32 inputs are the supported numerical domain; NaN/Inf propagation is not part of the contract.

Dimensions are nonnegative signed integers for GEMM and `size_t` for custom reduction. Byte multiplication is checked for overflow. Launch grids, thread counts, and static shared-memory requirements are checked against device limits. Invalid configuration, required null pointers, and short custom scratch buffers throw `std::invalid_argument`. CUDA/cuBLAS errors throw diagnostics with the failed expression and source location. The CUB adapter limits counts to `INT_MAX`; CUB itself validates insufficient nonzero scratch capacity. CUDA cannot infer a pointer's allocation length from this API: passing an undersized input/output allocation violates the contract.

The lab uses one GPU, selected before the first cached device-property lookup. Switching devices afterward is unsupported. RAII wrappers release buffers, streams, events, and library handles. Destructors do not throw. No allocation, workspace query, stream creation, or library handle creation belongs in the measured operation.

| Edge case | Behavior |
|---|---|
| Reduction `N=0` | Enqueue a zero into the required scalar output; no reduction kernel |
| Reduction `N=1` | One complete block reduction; no recursive loop |
| GEMM `M=0` or `N=0` | No operation; input and output pointers may be null |
| GEMM `K=0` | Write zeros without reading A or B |
| Fused GEMM `K=0` | Write `max(bias[col],0)` |
| Tail tile/block | Guard every global load/store; inactive lanes contribute zero |

## Reduction

### R0: establish a simple baseline

Each 256-thread block consumes at most 256 contiguous values. All threads write a shared-memory element, including zero for out-of-range lanes. A binary tree reduces 256 to one with a block barrier at each stage. There is no early return before a barrier. Each block writes exactly one partial sum.

The host enqueues complete passes until only one partial remains. The final pass writes the caller's scalar directly. There are no global floating-point atomics and no CPU participation in combining partials.

### R1: change the within-block reduction

R1 preserves R0's partition and pass schedule. A shuffle tree reduces each 32-lane warp. Lane zero writes that warp's sum to shared memory. One block barrier makes the warp sums visible, and the first warp reduces them. Its unused lanes start from zero. Every shuffle uses a full mask because all 32 lanes of each participating warp execute it.

Only 8 shared values are needed for a 256-thread block. The important comparison is R0 versus R1 with the same first-pass grid, rather than attributing every R2 improvement to shuffles alone.

### R2: reuse threads across the input

The first pass uses 128 or 256 threads, 1 or 4 independent accumulators, and a grid capped at 4 or 8 blocks per SM. For block width `B`, actual grid `G`, accumulator count `A`, thread index `b*B+lane`, and iteration `t`, accumulator `a` visits:

```text
index = b*B + lane + (t*A + a) * (G*B)
```

This partitions every valid input index exactly once. Adjacent lanes load adjacent values. Each independent accumulator has its own dependency chain; they are combined before the same warp/block reduction as R1. Every load has a bound check.

R2's later passes are R1 with 256 threads and one partial per thread. On the measured 56-SM GPU, a 4×SM cap is 224 partials and needs one final block. An 8×SM cap is 448 partials and needs two follow-up passes. This is why a larger first-pass grid can hurt a short full reduction even when its first kernel has more parallelism.

Workspace is two float arrays, each sized to the first pass's partial count. Passes alternate arrays, and a one-block pass writes the final output instead. No scratch initialization is required: each live partial is written before it is read. Tests reuse the allocation with changed input values and changed sizes, including zero.

### CUB reference

The adapter uses `cub::DeviceReduce::Sum` with float input/output and its traditional API's default run-to-run determinism policy. Capacity is queried before timing. The operation uses the caller's scratch and stream. This is not a claim of bitwise equality across devices or CUDA versions; accumulation order differs from the custom kernels.

## GEMM

All custom variants implement `C = A*B`, with contiguous row-major A (`M×K`), B (`K×N`), and C (`M×N`), using FP32 `fmaf`. There is no `--use_fast_math`.

| Variant | Thread block | Work per thread | Shared tile |
|---|---|---|---|
| G0 | 16×16 | One output, direct global loads over K | None |
| G1 | 16×16 | One output | A 16×16, B 16×16 |
| G2 | 16×16 | 4×4 or 4×8 register accumulators | A BM×BK, B BK×BN |

G1 cooperatively loads both tiles, zero-fills missing rows/columns/K elements, synchronizes, computes 16 products, and synchronizes before overwriting the tiles. Out-of-range output threads still participate in every barrier.

For G2, each block computes BM×BN outputs. Thread `(tx,ty)` owns:

```text
row = block_row + ty + r*16
col = block_col + tx + c*16
```

The candidate tiles are 64×64×16, 64×64×32, and 64×128×16. Loads use the linear thread index `ty*16+tx` to cover the shared tiles in contiguous chunks. Per-K A and B values are loaded into registers and reused across the thread's output matrix. All stores are independently guarded.

Larger tiles can reduce loading overhead but increase registers or shared memory. Occupancy alone is not the objective. Shared-memory bank behavior must be checked with the address mapping and counters; do not infer an improvement simply because a tile is larger.

### cuBLAS mapping and precision

cuBLAS consumes column-major matrices. The adapter interprets row-major storage as transposed column-major storage and computes `Cᵀ = Bᵀ Aᵀ` by calling:

```text
cublasGemmEx(handle, N, N,
             n, m, k,
             alpha=1, B, CUDA_R_32F, lda=n,
                      A, CUDA_R_32F, ldb=k,
             beta=0,  C, CUDA_R_32F, ldc=n,
             CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT)
```

Here `N` means `CUBLAS_OP_N`, not the problem dimension. Asymmetric tests prevent a transposition error from hiding behind square matrices. The handle uses pedantic math, host scalar pointers, and atomics-not-allowed. A 4 MiB workspace is allocated and bound **after** setting the stream, because changing a cuBLAS stream resets its workspace configuration. Library timings include its complete operation; setup is excluded.

## Fusion

Three paths implement the same `Y[row,col] = max((A*B)[row,col] + bias[col], 0)`:

1. Custom unfused: G2, then a single combined bias/ReLU kernel in place.
2. Custom fused: G2 applies bias/ReLU to each accumulator before its final store.
3. Library: pedantic FP32 cuBLAS GEMM, then that same combined epilogue kernel.

Fusion removes one launch and approximately `8*M*N` logical intermediate bytes (one float store and one float reload). It does not necessarily remove that many DRAM bytes: the intermediate may hit cache. The cuBLAS comparison is not cuBLASLt fused-epilogue performance.

## Correctness and measurement

The GoogleTest suite combines exact zeros/ones/alternating signs, one-hot and identity patterns, deterministic random values at three scales and seeds, boundary dimensions, overflow/argument rejection, and changed-size buffer reuse. Output sentinels catch missed writes separately from uninitialized scratch checks.

CPU GEMM references convert both operands to `double` before multiplication. Small matrices use full FP64 references. Larger benchmark cases use a complete cuBLAS FP32 comparison plus up to 128 evenly spread FP64 output checks, including first/last output, for both implementations. A sampled check is not called a full FP64 validation. Non-finite values fail immediately.

Tolerances remain those specified in the plan:

```text
reduction: 1e-6*abs(input_scale) + 5e-6*sum(abs(input))
GEMM:      1e-4*abs(scale_A*scale_B) + 1e-4*abs(reference)
```

Large full FP32 library comparisons use twice the GEMM bound; sampled FP64 checks retain the original bound. Exact patterns use zero tolerance. These tests complement each other: the conservative reduction bound alone is not enough to catch every indexing error.

Device timings include every reduction pass or both unfused kernels. A single explicit nonblocking stream holds kernels, library calls, transfers, and timing events. Device and transfer samples use CUDA events around a calibrated repeated batch. E2E uses a host monotonic clock around one H2D→operation→D2H sequence and final stream synchronization, with preallocated pinned buffers.

Defaults are 10 warmups, 30 samples, a 20 ms target batch, and at most 10,000 repetitions. The final output is revalidated. Reported p10/p90 are percentiles of **batch means**, not individual-request tail latency. E2E is one job per sample. Short CUDA-event intervals can include host submission gaps, especially on WDDM.

`scripts/summarize.py` rejects incomplete/duplicate sample sets, failed processes, missing final validation, and inconsistent units. GPU-only rates are labeled input GB/s for reduction and conventional `2MNK` GFLOP/s for GEMM. Input GB/s is not measured DRAM bandwidth. Transfer timings are reported separately and are not subtracted from another run to invent an exact overhead breakdown.

## Questions to be able to answer in an interview

- Why does a padded lane still reach the block barrier?
- Why is the shuffle mask safe for a 1-element input?
- Why can an 8×SM grid need one more reduction pass than a 4×SM grid?
- Which G2 values live in registers, shared memory, and global memory?
- Why is B passed before A to cuBLAS?
- Which costs are outside the E2E timer, and which are inside?
- What differs between unprofiled warm-buffer timing and replayed, cache-flushed counter collection?
- Which shapes got slower, and how much variation remains between independent processes?
- What would a Tensor Core or cuBLASLt comparison change about the numerical contract?

Read the three case studies next to the relevant kernel source, then reproduce one measurement with a new output directory. The goal is to explain observed evidence, including limits, rather than memorize a headline speedup.
