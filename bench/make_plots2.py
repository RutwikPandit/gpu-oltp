#!/usr/bin/env python3
"""Single clean latency-vs-throughput plot + per-query-type bars."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------- Fig 5: ONE latency-vs-throughput plot ----------
# x = latency per result (us, log), y = throughput (Mop/s, log)
# GPU RGI point-lookup batch sweep (measured, PCIe/4060)
gpu_lat = [12.14, 16.02, 22.76, 23.36, 19.98, 35.37, 209.93, 938.72]
gpu_thr = [0.1, 0.5, 2.8, 21.9, 205.0, 926.3, 1248.7, 1117.0]
# CPU hash index, best-case (per #threads): (latency us, throughput Mop/s)
cpu_pts = {"1 core": (0.0221, 45.3), "8 cores": (0.0357, 224.1), "16 cores": (0.0540, 296.2)}

fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(gpu_lat, gpu_thr, "o-", color="tab:blue", lw=2, ms=6, label="GPU (RGI), PCIe — batch sweep")
for i, b in enumerate([1,8,64,512,4096,32768,262144,1048576]):
    ax.annotate(f"B={b}", (gpu_lat[i], gpu_thr[i]), fontsize=7, color="tab:blue",
                xytext=(3,4), textcoords="offset points")

cx = [v[0] for v in cpu_pts.values()]; cy = [v[1] for v in cpu_pts.values()]
ax.plot(cx, cy, "s-", color="tab:orange", lw=2, ms=8, label="CPU hash index (1/8/16 cores)")
for name,(x,y) in cpu_pts.items():
    ax.annotate(name, (x,y), fontsize=8, color="tab:orange", xytext=(3,-10), textcoords="offset points")

# projected GPU + C2C: dispatch floor ~16us -> ~0.5us, same throughput envelope
ax.scatter([0.5], [900], marker="*", s=320, color="tab:green", zorder=5,
           label="GPU + C2C (projected)")
ax.annotate("GPU + C2C\n(projected: high throughput\nAT low latency)", (0.5, 900),
            fontsize=8, color="tab:green", xytext=(8,-6), textcoords="offset points")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("latency to result  (us, log)  ——  lower = better →", fontsize=10)
ax.set_ylabel("throughput  (Mop/s, log)  ——  higher = better ↑", fontsize=10)
ax.set_title("CPU vs GPU: throughput vs latency (point lookups)")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="lower right", fontsize=9)
ax.text(0.02, 60, "CPU: low latency,\ncapped throughput", fontsize=9, color="tab:orange")
ax.text(40, 2, "GPU on PCIe: high throughput\nbut ~16us dispatch floor", fontsize=9, color="tab:blue")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_latency_vs_throughput.png"), dpi=140)

# ---------- Fig 6: per-statement-type CPU vs GPU (100k rows) ----------
q = ["INSERT\n100k","SELECT\npoint","SELECT\nk=ANY(1k)","SCAN\ncount(*)","UPDATE\n1 row","UPDATE\nbulk 14k","DELETE\n1 row"]
cpu = [250.8, 2.80, 4.94, 6.79, 17.97, 107.0, 1.04]   # measured 2026-06-08
gpu = [127.0, 0.95, 430.3, 8.67, 1.64, 18.38, 0.89]   # (k=ANY uses ARRAY(SELECT) subplan -> snapshot fallback)
import numpy as np
x = np.arange(len(q)); w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 5.0))
b1 = ax.bar(x-w/2, cpu, w, label="CPU Postgres (heap+btree)", color="tab:orange")
b2 = ax.bar(x+w/2, gpu, w, label="GPU (RGI FDW)", color="tab:blue")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(q, fontsize=8)
ax.set_ylabel("statement time (ms, log) — lower better")
ax.set_title("Per-statement-type: CPU vs GPU (100k-row table)")
# mark GPU win
ax.annotate("GPU 5.8x faster", (5+w/2, 18.38), fontsize=9, color="tab:green",
            xytext=(0,18), textcoords="offset points", ha="center",
            arrowprops=dict(arrowstyle="->", color="tab:green"))
for bars in (b1,b2):
    for bb in bars:
        ax.text(bb.get_x()+bb.get_width()/2, bb.get_height(), f"{bb.get_height():.1f}",
                ha="center", va="bottom", fontsize=6.5)
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_per_query.png"), dpi=140)
print("wrote fig5_latency_vs_throughput.png, fig6_per_query.png")
