"""Run correctness and all four Compute Sanitizer tools; never accept zero tests."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
from run_benchmarks import environment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exe = args.exe.resolve(); root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    commands = [("correctness", [str(exe), f"--gtest_output=json:{root / 'correctness.json'}"])]
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        commands.append((tool, ["compute-sanitizer", "--tool", tool, "--error-exitcode", "1", str(exe),
                                "--gtest_filter=*Boundary.*", f"--gtest_output=json:{root / (tool + '-tests.json')}"]))
    records = []
    for name, command in commands:
        print(f"running {name}", flush=True)
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        with (root / f"{name}.log").open("w", encoding="utf-8") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        log = (root / f"{name}.log").read_text()
        test_count = re.search(r"\[==========\] Running (\d+) tests", log)
        valid = result.returncode == 0 and test_count and int(test_count[1]) > 0
        if name != "correctness":
            valid = valid and ("ERROR SUMMARY: 0 errors" in log or "0 hazards displayed (0 errors, 0 warnings)" in log)
        records.append({"tool":name,"command":command,"started_utc":started,"exit_code":result.returncode,
                        "tests":int(test_count[1]) if test_count else 0,"passed":bool(valid)})
        (root / "validation.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        if not valid:
            print(log); raise SystemExit(f"{name} did not pass")
        print(f"{name}: passed ({test_count[1]} tests)", flush=True)

if __name__ == "__main__":
    main()
