#!/usr/bin/env python3
"""Generate performance plots for the GPU-OLTP prototype (RTX 4060 / PCIe).
All numbers are measured in this session unless marked 'projected'."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- Fig 1: latency vs throughput batch sweep (RGI chain HT point lookups) ----
B   = [1, 8, 64, 512, 4096, 32768, 262144, 1048576]
lat = [12.14, 16.02, 22.76, 23.36, 19.98, 35.37, 209.93, 938.72]   # us  (measured 2026-06-08)
thr = [0.1, 0.5, 2.8, 21.9, 205.0, 926.3, 1248.7, 1117.0]          # Mop/s

fig, ax1 = plt.subplots(figsize=(7, 4.2))
ax1.set_xscale("log", base=2)
ax1.plot(B, thr, "o-", color="tab:blue", label="throughput (Mop/s)")
ax1.set_xlabel("batch size (requests per launch)")
ax1.set_ylabel("throughput (Mop/s)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(B, lat, "s--", color="tab:red", label="latency (us)")
ax2.set_yscale("log")
ax2.set_ylabel("per-batch latency (us, log)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax1.axvspan(1, 4096, alpha=0.08, color="green")
ax1.text(2, max(thr)*0.6, "launch-latency-bound\n(~16 us floor)", fontsize=8, color="green")
plt.title("Fig 1. Latency vs Throughput tradeoff (RGI lookups, 4060/PCIe)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_latency_throughput.png"), dpi=130)

# ---- Fig 2: CPU scaling vs GPU ceiling (point lookups) ----
threads = [1, 2, 4, 8, 16]
cpu_mops = [45.3, 69.4, 114.1, 224.1, 296.2]
gpu_ceiling = 1248.7
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(threads, cpu_mops, "o-", color="tab:orange", label="CPU hash index (per #threads)")
ax.axhline(gpu_ceiling, color="tab:blue", ls="--", label=f"GPU RGI ceiling ({gpu_ceiling:.0f} Mop/s)")
ax.axhline(296.2, color="tab:orange", ls=":", alpha=0.6)
ax.set_xlabel("CPU threads"); ax.set_ylabel("throughput (Mop/s)")
ax.set_title("Fig 2. Point-lookup throughput: CPU cores vs GPU (RGI)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_cpu_vs_gpu_throughput.png"), dpi=130)

# ---- Fig 3: bandwidth-bound aggregate scan, CPU vs GPU (50M rows) ----
labels = ["SUM(v)", "COUNT(v<c)"]
cpu_ms = [622.1, 586.2]
gpu_ms = [2.88, 2.77]
x = range(len(labels)); w = 0.35
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar([i - w/2 for i in x], cpu_ms, w, label="CPU Postgres (8 workers)", color="tab:orange")
ax.bar([i + w/2 for i in x], gpu_ms, w, label="GPU pushed-down aggregate", color="tab:blue")
ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("query time (ms, log)")
ax.set_title("Fig 3. Aggregate scan over 50M rows (lower is better)")
for i in x:
    ax.text(i - w/2, cpu_ms[i], f"{cpu_ms[i]:.0f}ms", ha="center", va="bottom", fontsize=8)
    ax.text(i + w/2, gpu_ms[i], f"{gpu_ms[i]:.1f}ms", ha="center", va="bottom", fontsize=8)
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_bandwidth_scan.png"), dpi=130)

# ---- Fig 4: throughput ladder (what batching + RGI unlock) ----
# all four bars measured 2026-06-08: per-op round-trip = 1/(toy single-op RT 75.5us);
# toy batched workload; RGI engine at a moderate batch (sweep B=4096); RGI raw kernel ceiling (sweep B=262144)
names = ["Unbatched\n(per-op RT)", "Toy engine\nbatched", "RGI engine\n(B=4096)", "RGI raw\nkernel ceiling"]
vals  = [0.0132, 42.8, 205.0, 1248.7]   # Mop/s
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(names, vals, color=["#bbb", "tab:gray", "tab:green", "tab:blue"])
ax.set_yscale("log"); ax.set_ylabel("find throughput (Mop/s, log)")
ax.set_title("Fig 4. Throughput ladder: batching + RGI index")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v, (f"{v*1000:.0f} k/s" if v < 1 else f"{v:.0f} M/s"),
            ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_throughput_ladder.png"), dpi=130)

print("wrote:", ", ".join(["fig1_latency_throughput.png","fig2_cpu_vs_gpu_throughput.png",
                            "fig3_bandwidth_scan.png","fig4_throughput_ladder.png"]))
