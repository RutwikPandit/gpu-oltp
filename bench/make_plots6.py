#!/usr/bin/env python3
"""Throughput vs concurrency: MEASURED data points + model curves.
Only the C2C curve is a projection; everything else is measured this session."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
OUT = os.path.dirname(os.path.abspath(__file__))

# --- MEASURED: GPU find throughput vs batch (rgi_sweep, kernel timing) ---
sweep_B   = [1, 8, 64, 512, 4096, 32768, 262144, 1048576]
sweep_thr = [0.1, 0.5, 2.8, 21.9, 205.0, 926.3, 1248.7, 1117.0]   # measured 2026-06-08
# --- MEASURED: GPU find throughput vs batch (Nsight Compute schmoo) ---
ncu_B   = [1000, 10000, 100000, 1000000, 2000000]
ncu_thr = [29.0, 268.9, 752.3, 791.1, 702.1]
# --- MEASURED: CPU hash index throughput at 1/2/4/8/16 threads ---
cpu_thr_x = [1, 2, 4, 8, 16]
cpu_thr_y = [45.3, 69.4, 114.1, 224.1, 296.2]

# --- MODELS (constants measured; C2C dispatch projected) ---
CEIL=1249e6; DISP_PCIE=16e-6; DISP_C2C=0.5e-6; CPU_PC=45.3e6; CPU_CEIL=296.2e6
C = np.logspace(0, 6, 400)
m_pcie = C/(DISP_PCIE + C/CEIL)/1e6
m_c2c  = C/(DISP_C2C  + C/CEIL)/1e6
m_cpu  = np.minimum(C*CPU_PC, CPU_CEIL)/1e6

fig, ax = plt.subplots(figsize=(9.5, 5.8))
# models
ax.plot(C, m_pcie, color="tab:blue",  lw=1.5, alpha=0.6, label="GPU PCIe — model")
ax.plot(C, m_c2c,  color="tab:green", lw=2, ls="--",     label="GPU C2C — model (PROJECTED)")
ax.plot(C, m_cpu,  color="tab:orange",lw=1.5, alpha=0.6, label="CPU 16c — model")
def fmtB(b):
    return f"{b/1e6:.0f}M" if b>=1e6 else (f"{b/1e3:.0f}k" if b>=1e3 else str(int(b)))
def fmtT(t):
    return f"{t:.1f}" if t<10 else f"{t:.0f}"
# measured points
ax.scatter(sweep_B, sweep_thr, s=70, color="tab:blue", marker="o", zorder=5,
           edgecolor="k", label="GPU PCIe — MEASURED (sweep)")
for b,t in zip(sweep_B, sweep_thr):
    ax.annotate(f"B={fmtB(b)}\n{fmtT(t)} M/s", (b,t), fontsize=7.5, color="tab:blue",
                xytext=(4,6), textcoords="offset points")
ax.scatter(ncu_B, ncu_thr, s=80, color="tab:cyan", marker="^", zorder=5,
           edgecolor="k", label="GPU PCIe — MEASURED (Nsight)")
ax.scatter(cpu_thr_x, cpu_thr_y, s=80, color="tab:orange", marker="s", zorder=5,
           edgecolor="k", label="CPU — MEASURED (1-16 cores)")
for x,t in zip(cpu_thr_x, cpu_thr_y):
    ax.annotate(f"{x}c: {t:.0f} M/s", (x,t), fontsize=7.5, color="darkorange",
                xytext=(4,-12), textcoords="offset points")
# reference horizontal lines with labels (easier than reading log y)
for yv,lbl in [(296,"CPU ceiling 296 M/s"),(1249,"GPU kernel ceiling 1249 M/s")]:
    ax.axhline(yv, color="gray", ls=":", lw=1, alpha=0.5)
    ax.text(1.1, yv*1.05, lbl, fontsize=7.5, color="gray")
ax.axvspan(50, 2000, alpha=0.08, color="gray"); ax.text(60, 0.2, "typical OLTP\nin-flight", fontsize=8, color="gray")
# crossover points where GPU model overtakes the CPU ceiling
slope = (1/CPU_CEIL - 1/CEIL)
x_pcie = (DISP_PCIE/slope)        # PCIe GPU == CPU ceiling
x_c2c  = (DISP_C2C /slope)        # C2C  GPU == CPU ceiling
cpu_ceil_mops = CPU_CEIL/1e6
for xv, col, tag in [(x_c2c, "tab:green", "C2C"), (x_pcie, "tab:blue", "PCIe")]:
    ax.scatter([xv], [cpu_ceil_mops], marker="*", s=300, color=col, edgecolor="k", zorder=6)
    ax.annotate(f"GPU > CPU here\nB\u2248{xv:.0f} ({tag})", (xv, cpu_ceil_mops),
                fontsize=8.5, color=col, fontweight="bold",
                xytext=(6, 14 if tag=="C2C" else -34), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=col))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("concurrent in-flight ops  (= GPU batch size)")
ax.set_ylabel("throughput (Mop/s)")
ax.set_title("OLTP point lookups (SELECT WHERE k=? = RGI find vs CPU hash-index find)\n"
             "measured points + models, RTX 4060")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig11_measured_vs_model.png"), dpi=140)
print("wrote fig11_measured_vs_model.png")
