"""Collect selected hardware-counter reports and marked CUDA timelines."""
import argparse
import json
from pathlib import Path
import subprocess
from run_benchmarks import environment

def cases():
    for elements in (1024, 16777216):
        for variant in ("r0","r1","r2","cub"):
            yield f"reduce-{variant}-{elements}", ["--op","reduce","--variant",variant,"--elements",str(elements)]
    for variant in ("g0","g1","g2","cublas"):
        yield f"gemm-{variant}-1024", ["--op","gemm","--variant",variant,"--m","1024","--n","1024","--k","1024"]
    yield "gemm-g2-rect", ["--op","gemm","--variant","g2","--m","1024","--n","2048","--k","512"]
    yield "gemm-g2-tail", ["--op","gemm","--variant","g2","--m","257","--n","509","--k","127"]
    yield "gemm-g2-bk16", ["--op","gemm","--variant","g2","--m","1024","--n","1024","--k","1024","--bk","16"]
    for variant in ("unfused","fused","library"):
        yield f"fusion-{variant}", ["--op","fusion","--variant",variant,"--m","512","--n","1024","--k","128"]

def execute(command, log):
    with log.open("w", encoding="utf-8") as out:
        result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()); raise SystemExit(f"command failed: {command}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool", choices=("all","ncu","nsys"), default="all")
    parser.add_argument("--detail", action="store_true", help="selected memory-traffic/bank-conflict follow-up")
    args = parser.parse_args(); root = args.output.resolve(); exe = str(args.exe.resolve())
    root.mkdir(parents=True, exist_ok=False)
    (root / "environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    commands = []
    sections = ["SpeedOfLight","LaunchStats","Occupancy","MemoryWorkloadAnalysis","ComputeWorkloadAnalysis","SchedulerStats","WarpStateStats"]
    for name, flags in (cases() if args.tool != "nsys" else []):
        if args.detail and name not in {"reduce-r0-16777216","reduce-r1-16777216","reduce-r2-16777216", "gemm-g0-1024","gemm-g1-1024","gemm-g2-1024","gemm-g2-bk16","fusion-unfused","fusion-fused"}:
            continue
        print(name, flush=True)
        report = root / name
        command = ["ncu","--nvtx","--nvtx-include","measured_operation/","--replay-mode","kernel","--cache-control","all","--clock-control","none"]
        for section in sections:
            command += ["--section", section]
        if args.detail:
            command += ["--metrics", ",".join(("dram__bytes_read.sum","dram__bytes_write.sum",
                "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum","l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
                "smsp__warp_issue_stalled_barrier_per_warp_active.pct","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"))]
        command += ["--export",str(report),exe,*flags,"--profile-once","--output",str(root / (name + "-run"))]
        commands.append(command)
        (root / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
        execute(command, root / (name + ".log"))
        if not report.with_suffix(".ncu-rep").is_file():
            raise SystemExit(f"missing profiler report for {name}")
        execute(["ncu","--import",str(report.with_suffix(".ncu-rep")),"--page","raw","--csv"], root / (name + ".csv"))
    traces = [
        ("reduce-r2-e2e", ["--op","reduce","--variant","r2","--elements","16777216"]),
        ("fusion-unfused-e2e", ["--op","fusion","--variant","unfused","--m","128","--n","512","--k","64"]),
        ("fusion-fused-e2e", ["--op","fusion","--variant","fused","--m","128","--n","512","--k","64"]),
    ]
    for name, flags in (traces if args.tool != "ncu" else []):
        print(name, flush=True)
        command = ["nsys","profile","--trace=cuda,nvtx","--sample=none","--cpuctxsw=none",
                   "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
                   "--capture-range=nvtx","--nvtx-capture=measured_operation","--capture-range-end=stop",
                   "--output",str(root / name),exe,*flags,"--mode","e2e","--profile-once","--output",str(root / (name + "-run"))]
        commands.append(command)
        (root / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
        execute(command, root / (name + ".log"))
        execute(["nsys","export","--type=sqlite","--output",str(root / (name + ".sqlite")),str(root / (name + ".nsys-rep"))],root / (name + "-export.log"))

if __name__ == "__main__":
    main()
