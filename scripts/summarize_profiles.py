"""Extract auditable counters and timeline events; these are not benchmark samples."""
import argparse
import csv
import io
import json
from pathlib import Path
import sqlite3

METRICS = {
    "profile_duration": "gpu__time_duration.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "static_shared_memory": "launch__shared_mem_per_block_static",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "fma_pipe_active_pct": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput": "dram__bytes.sum.per_second",
    "dram_read": "dram__bytes_read.sum", "dram_write": "dram__bytes_write.sum",
    "shared_load_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "shared_store_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "barrier_stall_pct": "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "long_scoreboard_stall_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "local_spill_requests": "derived__local_spilling_requests",
}

def ncu_rows(path):
    raw = path.read_text(encoding="utf-8-sig")
    header = raw.index('"ID",')  # Nsight may prepend a section-deployment warning.
    rows = list(csv.DictReader(io.StringIO(raw[header:])))
    if len(rows) < 2:
        raise ValueError(f"no profiled kernels: {path}")
    units = rows[0]
    output = []
    for row in rows[1:]:
        if not row["ID"].isdigit():
            raise ValueError(f"unexpected profiler CSV row: {path}")
        item = {"kernel":row["Kernel Name"], "grid":row["Grid Size"], "block":row["Block Size"], "stream":row["Stream"], "metrics":{}}
        for label, metric in METRICS.items():
            value = row.get(metric)
            if value is not None and value not in ("", "no data"):
                item["metrics"][label] = {"value":float(value.replace(",", "")), "unit":units[metric], "metric":metric}
        output.append(item)
    return output

def timeline(path):
    with sqlite3.connect(path) as db:
        names = dict(db.execute("SELECT id,value FROM StringIds"))
        ranges = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='measured_operation'").fetchall()
        if len(ranges) != 1 or ranges[0][1] is None:
            raise ValueError("expected one complete measured_operation range")
        start, end = ranges[0]
        events = []
        for begin, finish, name, stream in db.execute("SELECT start,end,demangledName,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?", (start,end)):
            events.append({"kind":"kernel", "name":names[name], "start_us":(begin-start)/1000, "duration_us":(finish-begin)/1000, "stream":stream})
        for begin, finish, direction, count, stream in db.execute("SELECT start,end,copyKind,bytes,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=?", (start,end)):
            events.append({"kind":{1:"H2D",2:"D2H"}.get(direction,str(direction)), "bytes":count, "start_us":(begin-start)/1000, "duration_us":(finish-begin)/1000, "stream":stream})
        if not events or len({e["stream"] for e in events}) != 1:
            raise ValueError("empty trace or unexpected stream mismatch")
        api = [{"name":names[name], "duration_us":(finish-begin)/1000} for begin,finish,name in db.execute("SELECT start,end,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=?", (start,end))]
        return {"range_duration_us":(end-start)/1000,"events":sorted(events,key=lambda e:e["start_us"]),"cuda_api_calls":api}

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("root",type=Path); args = parser.parse_args()
    result = {"warning":"Replayed/instrumented observations; use results/release for ordinary performance numbers.", "counter_reports":{}, "timelines":{}}
    for path in sorted(args.root.rglob("*.csv")):
        if not path.name == "samples.csv":
            result["counter_reports"][path.relative_to(args.root).as_posix()] = ncu_rows(path)
    for path in sorted(args.root.rglob("*.sqlite")):
        result["timelines"][path.relative_to(args.root).as_posix()] = timeline(path)
    (args.root / "summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Extracted {len(result['counter_reports'])} counter reports and {len(result['timelines'])} timelines")

if __name__ == "__main__":
    main()
