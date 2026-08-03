#!/usr/bin/env python3
"""Latency vs concurrency/batch: MEASURED GPU points + models. Point lookups.
Only the C2C curve is projected."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
OUT = os.path.dirname(os.path.abspath(__file__))

# MEASURED GPU per-batch latency (rgi_sweep), us
sweep_B   = [1, 8, 64, 512, 4096, 32768, 262144, 1048576]
sweep_lat = [12.14, 16.02, 22.76, 23.36, 19.98, 35.37, 209.93, 938.72]   # measured 2026-06-08

CEIL=1249e6; DISP_PCIE=16e-6; DISP_C2C=0.5e-6; CPU_CEIL=296.2e6; CPU_LAT0=22.07e-9
C = np.logspace(0, 6, 400)
lat_pcie = (DISP_PCIE + C/CEIL)*1e6
lat_c2c  = (DISP_C2C  + C/CEIL)*1e6
lat_cpu  = np.maximum(CPU_LAT0, C/CPU_CEIL)*1e6

def fmtB(b): return f"{b/1e6:.0f}M" if b>=1e6 else (f"{b/1e3:.0f}k" if b>=1e3 else str(int(b)))

fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.plot(C, lat_cpu,  color="tab:orange", lw=2,   label="CPU 16c — model (B/296 M/s, 22 ns floor)")
ax.plot(C, lat_pcie, color="tab:blue",  lw=1.5, alpha=0.6, label="GPU PCIe — model")
ax.plot(C, lat_c2c,  color="tab:green", lw=2, ls="--",     label="GPU C2C — model (PROJECTED)")
ax.scatter(sweep_B, sweep_lat, s=70, color="tab:blue", marker="o", edgecolor="k", zorder=5,
           label="GPU PCIe — MEASURED")
for b,l in zip(sweep_B, sweep_lat):
    ax.annotate(f"B={fmtB(b)}\n{l:.1f} us", (b,l), fontsize=7.5, color="tab:blue",
                xytext=(4,5), textcoords="offset points")
# CPU measured anchors: single-op latency and 16-core saturated point
ax.scatter([1],[CPU_LAT0*1e6], s=80, color="tab:orange", marker="s", edgecolor="k", zorder=5)
ax.annotate("CPU 1 op: 22 ns", (1, CPU_LAT0*1e6), fontsize=8, color="darkorange",
            xytext=(6,-12), textcoords="offset points")

# crossovers: GPU latency < CPU latency for B above these
slope=(1/CPU_CEIL - 1/CEIL); x_c2c=DISP_C2C/slope; x_pcie=DISP_PCIE/slope
for xv,col,tag in [(x_c2c,"tab:green","C2C"),(x_pcie,"tab:blue","PCIe")]:
    yv=(xv/CPU_CEIL)*1e6
    ax.scatter([xv],[yv], marker="*", s=300, color=col, edgecolor="k", zorder=6)
    ax.annotate(f"GPU faster than CPU\nfor B>{xv:.0f} ({tag})", (xv,yv), fontsize=8.5,
                color=col, fontweight="bold", xytext=(8, -28 if tag=="PCIe" else 12),
                textcoords="offset points", arrowprops=dict(arrowstyle="->",color=col))

ax.axvspan(50,2000, alpha=0.08, color="gray"); ax.text(60, 0.012, "typical OLTP\nin-flight", fontsize=8, color="gray")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("concurrent in-flight ops  (= GPU batch size)")
ax.set_ylabel("latency to result (us)")
ax.set_title("OLTP point-lookup LATENCY (SELECT WHERE k=? = RGI find vs CPU hash-index find)\n"
             "measured GPU points + models, RTX 4060   (lower = better)")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig12_latency.png"), dpi=140)
print("wrote fig12_latency.png; crossover C2C B~%.0f PCIe B~%.0f"%(x_c2c,x_pcie))
