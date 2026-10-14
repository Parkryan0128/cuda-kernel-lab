# Profiler evidence

- `measured/`: 18 counter reports covering small/large reduction, square/rectangular/tail GEMM, the alternate BK=16 tile, and all fusion paths.
- `detail/`: 9 follow-up counter reports with explicit DRAM, shared-bank, barrier, and long-scoreboard metrics.
- `timelines/`: 3 successful marked E2E captures, exported into the top-level `summary.json`.
- `summary.json`: extracted metric values with their original names/units, and only the marked operation's timeline events and CUDA API calls.

Each collection includes the environment and exact commands. `*-run/` contains the benchmark application's metadata and before/after validation; its `profile_once=true` flag prevents these runs from becoming ordinary timing samples. An empty `samples.csv` there is expected.

Counter collection uses kernel replay, all-cache flushing, and no requested clock adjustment. These conditions differ from the unprofiled warm-buffer benchmark. Profiled durations are not used as headline latency.

The final command in `measured/commands.json` records an initial Nsight Systems attempt that produced no report because the marker was an unregistered NVTX string. Its successful replacement is in `timelines/commands.json`, with `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`. The failed attempt's transient files are excluded from published evidence. A missing report was treated as a failure, not a successful trace.

Binary `.ncu-rep`, `.nsys-rep`, and `.sqlite` containers are retained locally and excluded from Git. They can contain broader environment details. The committed CSV counter exports and focused JSON timeline extraction provide the evidence used by the reports. Run `scripts/profile.py` to regenerate binaries and `scripts/summarize_profiles.py` to regenerate the focused summary. Do not infer a missing metric is zero.
