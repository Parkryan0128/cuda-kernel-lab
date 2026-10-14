"""Validate raw samples and derive median/p10/p90 batch-mean latency summaries."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

GROUP = ("operation","variant","mode","n_elements","m","n","k","seed","config")
def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = math.floor(position); hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)

def summarize_file(path: Path):
    metadata = json.loads((path.parent / "metadata.json").read_text())
    if metadata["profile_once"]:
        return []
    invocation = path.parent / "invocation.json"
    if invocation.exists() and json.loads(invocation.read_text()).get("exit_code") != 0:
        raise ValueError(f"incomplete or failed run: {path.parent}")
    correctness = [json.loads(line) for line in (path.parent / "correctness.jsonl").read_text().splitlines()]
    if not correctness or not all(entry["passed"] for entry in correctness):
        raise ValueError(f"missing or failed correctness evidence: {path}")
    validated = {(entry["case"], entry["phase"]) for entry in correctness}
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if None in row or row["correctness"] != "passed":
                raise ValueError(f"malformed or failed sample: {path}")
            total = float(row["total_ms"]); per_op = float(row["per_operation_us"]); repetitions = int(row["repetitions"])
            if repetitions < 1 or not math.isfinite(per_op) or per_op <= 0 or not math.isclose(total * 1000 / repetitions, per_op, rel_tol=1e-8):
                raise ValueError(f"invalid sample units: {path}")
            if row["mode"] == "e2e" and repetitions != 1:
                raise ValueError("E2E samples must contain one job")
            groups[tuple(row[key] for key in GROUP)].append(row)
    summaries = []
    for key, rows in groups.items():
        samples = metadata["samples"]
        if len(rows) != samples or sorted(int(row["sample"]) for row in rows) != list(range(samples)):
            raise ValueError(f"incomplete/duplicate sample series: {path}: {key}")
        row = rows[0]
        shape = row["n_elements"] if row["operation"] == "reduce" else "x".join(row[d] for d in ("m","n","k"))
        case = "/".join((row["operation"], row["variant"], row["mode"], shape))
        if not all((case, phase) in validated for phase in ("before", "after")):
            raise ValueError(f"missing before/after validation: {case}")
        values = [float(row["per_operation_us"]) for row in rows]
        summary = dict(zip(GROUP, key))
        summary.update(run_id=row["run_id"], samples=samples, repetitions=int(row["repetitions"]),
                       median_us=percentile(values,.5), p10_us=percentile(values,.1), p90_us=percentile(values,.9),
                       raw_file=path.as_posix(), workspace_bytes=int(row["workspace_bytes"]), pass_count=int(row["pass_count"]),
                       working_set_bytes=int(row["working_set_bytes"]), l2_bytes=metadata["l2_bytes"])
        seconds = summary["median_us"] / 1e6
        if row["mode"] == "device":
            if row["operation"] == "reduce":
                summary["input_GB_per_s"] = int(row["input_bytes"]) / seconds / 1e9
            else:
                summary["gemm_equivalent_GFLOP_per_s" if row["operation"] == "fusion" else "GFLOP_per_s"] = 2.0 * int(row["m"]) * int(row["n"]) * int(row["k"]) / seconds / 1e9
        elif row["mode"] in ("h2d","d2h"):
            summary["transfer_GB_per_s"] = int(row["input_bytes"] if row["mode"] == "h2d" else row["output_bytes"]) / seconds / 1e9
        summaries.append(summary)
    return summaries

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.rglob("samples.csv"))
    if not paths:
        raise SystemExit("no raw samples found")
    all_rows = []
    for path in paths:
        rows = summarize_file(path)
        for row in rows:
            row["raw_file"] = path.relative_to(args.root).as_posix()
        (path.parent / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        all_rows.extend(rows)
    (args.root / "summary.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"Validated {len(all_rows)} case series from {len(paths)} runs")
if __name__ == "__main__":
    main()
