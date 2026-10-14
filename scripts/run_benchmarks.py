"""Run the bounded tuning grid or the release matrix in separate processes.

No third-party Python dependencies. Every command, exit status, source revision,
and pre/post GPU telemetry record is saved next to the raw benchmark output.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def capture(command: list[str]) -> str:
    try:
        p = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=20)
        return (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"unavailable: {e}"

def source_digest() -> str:
    digest = hashlib.sha256()
    paths = [ROOT / "CMakeLists.txt"]
    for directory in ("src", "include", "bench"):
        paths.extend(sorted((ROOT / directory).rglob("*")))
    for path in sorted(p for p in paths if p.is_file()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()

def environment() -> dict:
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": capture(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(capture(["git", "status", "--porcelain", "--untracked-files=no"])),
        "source_sha256": source_digest(),
        "os": platform.platform(), "cpu": platform.processor(), "python": sys.version,
        "nvcc": capture(["nvcc", "--version"]), "host_compiler": capture(["cl"] if os.name == "nt" else ["c++", "--version"]),
        "cmake": capture(["cmake", "--version"]),
        "ncu": capture(["ncu", "--version"]), "nsys": capture(["nsys", "--version"]),
        "cuda_environment_overrides": {k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUDA_MODULE_LOADING", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG") if k in os.environ},
        "background_load_note": "Interactive Windows WDDM desktop; display and Codex active; clocks/power not locked. No concurrent benchmark processes started by this runner.",
    }

def telemetry() -> str:
    return capture(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used", "--format=csv"])

def jobs(phase: str):
    if phase == "tuning":
        for block in (128, 256):
            for accumulators in (1, 4):
                for multiplier in (4, 8):
                    yield f"r2-b{block}-a{accumulators}-g{multiplier}", ["--op", "reduce", "--variant", "r2", "--suite", "tuning", "--block", str(block), "--accumulators", str(accumulators), "--grid-multiplier", str(multiplier)]
        for bm, bn, bk in ((64,64,16), (64,64,32), (64,128,16)):
            yield f"g2-{bm}-{bn}-{bk}", ["--op", "gemm", "--variant", "g2", "--suite", "tuning", "--bm", str(bm), "--bn", str(bn), "--bk", str(bk)]
        return
    for op in ("reduce", "gemm", "fusion"):
        yield f"{op}-device-standard", ["--op",op,"--variant","all","--suite","standard","--mode","device"]
    for repetition in range(1,4):
        for mode in ("device", "e2e"):
            for op, variants in (("reduce",("r2","cub")), ("gemm",("g2","cublas")), ("fusion",("unfused","fused","library"))):
                for variant in variants:
                    yield f"repeat{repetition}-{op}-{variant}-{mode}", ["--op",op,"--variant",variant,"--suite","representative","--mode",mode]
    for mode in ("h2d","d2h"):
        for op, variant in (("reduce","r2"),("gemm","g2"),("fusion","fused")):
            yield f"{op}-{mode}", ["--op",op,"--variant",variant,"--suite","representative","--mode",mode]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--phase", choices=("tuning","release"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="skip only completed jobs with identical source digest and command")
    args = parser.parse_args()
    args.exe = args.exe.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = environment()
    env_path = args.output / "environment.json"
    if env_path.exists():
        old = json.loads(env_path.read_text())
        if not args.resume or old["source_sha256"] != env["source_sha256"]:
            raise SystemExit("existing campaign; --resume requires unchanged source")
    else:
        env_path.write_text(json.dumps(env, indent=2), encoding="utf-8")
    for name, flags in jobs(args.phase):
        target = args.output / name
        command = [str(args.exe), *flags, "--output", str(target)]
        record_path = target / "invocation.json"
        if record_path.exists() and args.resume:
            previous = json.loads(record_path.read_text())
            if previous.get("exit_code") == 0 and previous["command"][1:] == command[1:]:
                print(f"skip completed: {name}", flush=True); continue
            raise SystemExit(f"incomplete or different run at {target}; preserve it and use a fresh campaign")
        target.mkdir(parents=True, exist_ok=False)
        record = {"command": command, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "before": telemetry()}
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(name, flush=True)
        with (target / "stdout.log").open("w", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), after=telemetry())
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if result.returncode:
            print((target / "stdout.log").read_text(), file=sys.stderr)
            return result.returncode
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
