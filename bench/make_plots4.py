#!/usr/bin/env python3
"""Nsight Compute Speed-of-Light breakdown for the RGI OLTP index kernels (4060)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
OUT = os.path.dirname(os.path.abspath(__file__))

metrics = ["Compute\n(SM)", "Memory", "L1/TEX", "L2", "DRAM\nBW"]
insert  = [52.77, 52.84, 52.84, 40.45, 13.55]   # measured 2026-06-08 (500k, kernel-only)
find    = [65.97, 64.51, 64.51, 15.38, 41.66]
x = np.arange(len(metrics)); w = 0.38
fig, ax = plt.subplots(figsize=(8.5, 5.0))
b1 = ax.bar(x-w/2, insert, w, label="INSERT kernel", color="tab:purple")
b2 = ax.bar(x+w/2, find,   w, label="FIND kernel",   color="tab:blue")
ax.axhline(100, color="k", lw=1, alpha=0.4); ax.text(4.3, 101, "peak (SOL)", fontsize=8)
ax.axhline(60, color="tab:red", ls="--", alpha=0.6); ax.text(0, 61.5, "<60% ⇒ latency-bound (per ncu)", fontsize=8, color="tab:red")
ax.set_ylim(0, 110); ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("% of Speed-of-Light (peak)")
ax.set_title("Nsight Compute SOL: RGI OLTP index kernels (RTX 4060, kernel-only)")
for bars in (b1,b2):
    for bb in bars:
        ax.text(bb.get_x()+bb.get_width()/2, bb.get_height()+1, f"{bb.get_height():.0f}", ha="center", fontsize=8)
ax.legend()
# highlight the DRAM-BW headroom point
ax.annotate("DRAM BW far from SOL\n(14–42%): NOT bandwidth-bound",
            (4, 41.66), fontsize=9, color="tab:green",
            xytext=(-150,25), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:green"))
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig8_sol_breakdown.png"), dpi=140)
print("wrote fig8_sol_breakdown.png")
