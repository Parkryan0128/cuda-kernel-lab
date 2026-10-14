"""Render five reproducible figures from validated summaries (no raw-data editing)."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#9ca3af","#ea8a23","#1267a5","#26845b"]
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold","figure.dpi":140,"savefig.dpi":160})

def shape(row):
    return row["n_elements"] if row["operation"] == "reduce" else "x".join(row[d] for d in ("m","n","k"))

def save(fig, root, name, note):
    fig.text(.02,.015,note,fontsize=8,color="#475569")
    fig.tight_layout(rect=(0,.065,1,1))
    fig.savefig(root / (name + ".png"), facecolor="white")
    fig.savefig(root / (name + ".svg"), facecolor="white")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True); p.add_argument("--tuning",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    args = p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows = json.loads((args.results / "summary.json").read_text())
    tuning = json.loads((args.tuning / "summary.json").read_text())
    standard = [r for r in rows if "standard" in r["run_id"]]
    repeats = [r for r in rows if r["run_id"].startswith("repeat")]
    def med(op,variant,mode,case):
        values = [r["median_us"] for r in repeats if (r["operation"],r["variant"],r["mode"],shape(r)) == (op,variant,mode,case)]
        assert len(values) == 3
        return statistics.median(values),min(values),max(values)

    fig, axes = plt.subplots(1,2,figsize=(11,4.8))
    for variant,color in zip(("r0","r1","r2","cub"),COLORS):
        series = sorted([r for r in standard if r["operation"]=="reduce" and r["variant"]==variant],key=lambda r:int(r["n_elements"]))
        sizes = [int(r["n_elements"]) for r in series]
        axes[0].plot(sizes,[r["median_us"] for r in series],"o-",label=variant.upper(),color=color,markersize=4)
        axes[1].plot(sizes,[r["input_GB_per_s"] for r in series],"o-",label=variant.upper(),color=color,markersize=4)
    for ax in axes:
        ax.set_xscale("log",base=2); ax.set_yscale("log"); ax.set_xlabel("FP32 input elements"); ax.grid(alpha=.2)
        ax.axvline(50331648 / 4,color="#64748b",linestyle=":",alpha=.7)
    axes[0].set(ylabel="Full reduction latency (us)",title="All passes count")
    axes[1].set(ylabel="Input processing rate (GB/s)",title="A cache-aware interpretation")
    axes[0].legend(ncol=2,frameon=False)
    save(fig,args.output,"reduction","RTX 4070 SUPER | Warm repeated buffers | 30 batch samples per case | Dotted line: 48 MiB input equals L2 capacity; input GB/s is not DRAM bandwidth.")

    cases = ["128x128x128","512x512x512","1024x1024x1024","2048x2048x2048","1024x2048x512","257x509x127","128x512x64"]
    fig,ax = plt.subplots(figsize=(11,5))
    x = np.arange(len(cases)); width=.2
    for i,(variant,color) in enumerate(zip(("g0","g1","g2","cublas"),COLORS)):
        series = {shape(r):r for r in standard if r["operation"]=="gemm" and r["variant"]==variant}
        y = [series[c]["median_us"] if c in series else np.nan for c in cases]
        ax.bar(x+(i-1.5)*width,y,width,label=variant.upper(),color=color)
    ax.set_xticks(x, [c.replace("x","\nx",1) for c in cases]); ax.set_yscale("log")
    ax.set(ylabel="Device-stream latency (us, log scale)",title="FP32 GEMM: gains and regressions depend on shape")
    ax.legend(ncol=4,frameon=False); ax.grid(axis="y",alpha=.2)
    save(fig,args.output,"gemm","M x N x K | Median of 30 batch means | G0 at 2048^3 is skipped by the predeclared work limit | cuBLAS uses FP32 pedantic math.")

    cases = ["128x512x64","512x1024x128","1024x1024x1024"]
    fig,axes = plt.subplots(1,2,figsize=(11,4.8))
    for ax,mode in zip(axes,("device","e2e")):
        x=np.arange(len(cases)); width=.24
        for i,(variant,color) in enumerate(zip(("unfused","fused","library"),(COLORS[0],COLORS[2],COLORS[3]))):
            values=[med("fusion",variant,mode,c) for c in cases]
            y=np.array([v[0] for v in values]); lower=y-np.array([v[1] for v in values]); upper=np.array([v[2] for v in values])-y
            ax.bar(x+(i-1)*width,y,width,yerr=np.vstack((lower,upper)),capsize=3,label=variant,color=color)
        ax.set_xticks(x,[c.replace("x","\nx",1) for c in cases]); ax.set_yscale("log")
        ax.set(title="Device only" if mode=="device" else "Including pinned-memory transfers",ylabel="Latency (us, log scale)"); ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False)
    save(fig,args.output,"fusion","Bars: median of 3 independent process medians. Whiskers: min/max of those medians. Each process contains 30 samples; E2E uses one job per sample.")

    cases = [("reduce","r2","1024"),("reduce","r2","16777216"),("gemm","g2","128x128x128"),("gemm","g2","1024x1024x1024")]
    fig,ax=plt.subplots(figsize=(10,4.8)); x=np.arange(len(cases)); width=.2
    for i,(mode,label,color) in enumerate((("device","Device",COLORS[2]),("h2d","H2D only","#ad7020"),("d2h","D2H only","#7c69a8"),("e2e","End to end",COLORS[3]))):
        y=[]
        for op,var,c in cases:
            if mode in ("device","e2e"): value=med(op,var,mode,c)[0]
            else: value=next(r["median_us"] for r in rows if (r["operation"],r["mode"],shape(r))==(op,mode,c))
            y.append(value)
        ax.bar(x+(i-1.5)*width,y,width,label=label,color=color)
    ax.set_xticks(x,[f"{op.upper()}\n{c}" for op,var,c in cases]); ax.set_yscale("log"); ax.set(ylabel="Latency (us, log scale)",title="Transfer-inclusive costs change the practical result")
    ax.legend(ncol=4,frameon=False); ax.grid(axis="y",alpha=.2)
    ax.set_ylim(top=ax.get_ylim()[1]*2)
    save(fig,args.output,"transfers","Device/E2E: medians across 3 process medians. Copies: 30 batch means. These are separate measurements; bars are not an additive latency decomposition.")

    fig,axes=plt.subplots(1,2,figsize=(11,5.3),gridspec_kw={"width_ratios":[1.55,1]})
    for ax,op,title in zip(axes,("reduce","gemm"),("R2: bounded configuration sweep","G2: three tile candidates")):
        series=[r for r in tuning if r["operation"]==op]
        configs=sorted({r["config"] for r in series})
        cases=sorted({shape(r) for r in series},key=lambda x:int(x.split("x")[0]))
        for i,c in enumerate(cases):
            vals={r["config"]:r["median_us"] for r in series if shape(r)==c}; best=min(vals.values())
            ax.bar(np.arange(len(configs))+(i-.5)*.36,[vals[config]/best for config in configs],.36,label=c,color=("#1267a5","#76b2d9")[i])
        ax.axhline(1,color="#64748b",linestyle=":")
        ax.set_xticks(np.arange(len(configs)),[c.replace("_","\n") for c in configs],fontsize=8)
        ax.set(title=title,ylabel="Latency / best candidate for that shape"); ax.legend(frameon=False,fontsize=8); ax.grid(axis="y",alpha=.2)
        ax.set_ylim(top=ax.get_ylim()[1]*1.2)
    save(fig,args.output,"tuning","Tuning shapes only; 30 samples per candidate. Fixed defaults: R2 B128/A4/cap224 and G2 64x64x32. No per-shape dispatch selection.")
    print(f"Wrote five PNG/SVG figures to {args.output}")

if __name__ == "__main__":
    main()
