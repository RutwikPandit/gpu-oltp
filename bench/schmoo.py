#!/usr/bin/env python3
"""DRAM-BW schmoo: sweep op type x batch size, profile with Nsight Compute,
extract DRAM throughput (%SOL and GB/s) and key SOL metrics, tabulate + plot."""
import subprocess, csv, io, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NCU = "/usr/local/cuda/bin/ncu"
BIN = os.path.expanduser("~/rgi_prof2")
OUT = os.path.dirname(os.path.abspath(__file__))

METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
]
OPS = ["find", "insert"]
NS  = [1000, 10000, 100000, 1000000, 2000000]

def num(s):
    try: return float(str(s).replace(",", ""))
    except: return float("nan")

def run_one(op, n):
    cmd = [NCU, "--csv", "--launch-count", "1",
           "--kernel-name-base", "demangled",
           "--kernel-name", "regex:%s_device_func" % op,
           "--metrics", ",".join(METRICS), BIN, op, str(n)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith('"')]
    if not lines:
        sys.stderr.write("no csv for %s N=%d\n%s\n" % (op, n, p.stdout[-400:]))
        return {}
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    vals = {}
    for r in rows:
        vals[r.get("Metric Name","")] = num(r.get("Metric Value",""))
    return vals

results = {op: {"N": [], "dram_pct": [], "dram_gbps": [], "sm_pct": [],
               "l2_pct": [], "l1_pct": [], "mops": []} for op in OPS}

print("%-7s %-9s %-9s %-10s %-7s %-7s %-7s %-9s" %
      ("op","N","DRAM%","DRAM GB/s","SM%","L2%","L1%","Mop/s"))
for op in OPS:
    for n in NS:
        v = run_one(op, n)
        if not v: continue
        dur_ns = v.get("gpu__time_duration.sum", float("nan"))
        gbps   = v.get("dram__bytes.sum.per_second", float("nan")) / 1e9
        dram   = v.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", float("nan"))
        sm     = v.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", float("nan"))
        l2     = v.get("lts__throughput.avg.pct_of_peak_sustained_elapsed", float("nan"))
        l1     = v.get("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", float("nan"))
        mops   = n / (dur_ns/1e9) / 1e6 if dur_ns==dur_ns and dur_ns>0 else float("nan")
        results[op]["N"].append(n)
        results[op]["dram_pct"].append(dram); results[op]["dram_gbps"].append(gbps)
        results[op]["sm_pct"].append(sm); results[op]["l2_pct"].append(l2)
        results[op]["l1_pct"].append(l1); results[op]["mops"].append(mops)
        print("%-7s %-9d %-9.1f %-10.1f %-7.1f %-7.1f %-7.1f %-9.1f" %
              (op, n, dram, gbps, sm, l2, l1, mops))

# ---- plots ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
for op, c in zip(OPS, ["tab:blue","tab:purple"]):
    a1.plot(results[op]["N"], results[op]["dram_pct"], "o-", color=c, label=op)
    a2.plot(results[op]["N"], results[op]["dram_gbps"], "o-", color=c, label=op)
a1.axhline(80, color="tab:red", ls="--", alpha=0.6); a1.text(1000, 81, "~80% = BW-bound", color="tab:red", fontsize=8)
a1.set_xscale("log"); a1.set_xlabel("batch size (ops)"); a1.set_ylabel("DRAM throughput (% of SOL)")
a1.set_title("DRAM BW utilization vs batch size"); a1.grid(alpha=0.3, which="both"); a1.legend(); a1.set_ylim(0,100)
a2.axhline(272, color="k", ls=":", alpha=0.5); a2.text(1000, 274, "4060 peak ~272 GB/s", fontsize=8)
a2.set_xscale("log"); a2.set_xlabel("batch size (ops)"); a2.set_ylabel("achieved DRAM BW (GB/s)")
a2.set_title("Achieved DRAM bandwidth vs batch size"); a2.grid(alpha=0.3, which="both"); a2.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig9_dram_schmoo.png"), dpi=140)
print("wrote fig9_dram_schmoo.png")
