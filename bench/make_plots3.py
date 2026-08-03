#!/usr/bin/env python3
"""Iso-query crossover: CPU vs GPU modeled as C2C dispatch + kernel duration.
A query issues B point-ops. Constants are measured this session except the C2C
doorbell (projected from GH200 ping-pong)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
OUT = os.path.dirname(os.path.abspath(__file__))

# --- measured constants ---
CPU_THR   = 296.2e6      # ops/s, 16-core hash index (measured 2026-06-08)
CPU_LAT0  = 22.07e-9     # single-op latency, s (measured 2026-06-08)
GPU_CEIL  = 1.249e9      # ops/s, RGI kernel throughput ceiling (measured 2026-06-08)
# --- dispatch costs ---
DISP_PCIE = 16e-6        # per-batch launch+dispatch floor on 4060/PCIe (measured)
DISP_C2C  = 0.5e-6       # persistent-kernel doorbell over C2C (projected, GH200)

B = np.logspace(0, 6, 400)
t_cpu  = np.maximum(CPU_LAT0, B / CPU_THR)
t_pcie = DISP_PCIE + B / GPU_CEIL
t_c2c  = DISP_C2C  + B / GPU_CEIL

# crossovers (CPU vs each GPU model), throughput-bound region
slope = (1/CPU_THR - 1/GPU_CEIL)
x_c2c  = DISP_C2C  / slope
x_pcie = DISP_PCIE / slope

fig, ax = plt.subplots(figsize=(9, 5.4))
ax.plot(B, t_cpu*1e6,  color="tab:orange", lw=2.5, label="CPU (16 threads, 296 Mop/s)")
ax.plot(B, t_pcie*1e6, color="tab:blue",  lw=2, ls=":", label="GPU = PCIe dispatch (16 us) + kernel")
ax.plot(B, t_c2c*1e6,  color="tab:green", lw=2.5, label="GPU = C2C dispatch (0.5 us) + kernel  [projected]")

ax.axvline(x_c2c, color="tab:green", ls="--", alpha=0.6)
ax.axvline(x_pcie, color="tab:blue", ls="--", alpha=0.4)
ax.set_xscale("log"); ax.set_yscale("log")

# shade win regions wrt the C2C model
ax.axvspan(1, x_c2c, alpha=0.07, color="tab:orange")
ax.axvspan(x_c2c, 1e6, alpha=0.07, color="tab:green")
ymin = (t_cpu.min()*1e6)*0.6
ax.text(1.4, ymin*1.1, "CPU wins\n(low-concurrency,\nlatency-critical)", color="tab:orange", fontsize=9)
ax.text(x_c2c*2.2, ymin*1.1, "GPU wins\n(many ops/query =\nbatch/concurrency)", color="tab:green", fontsize=9)
ax.annotate(f"C2C crossover\nB≈{x_c2c:.0f} ops", (x_c2c, DISP_C2C*1e6*1.2), color="tab:green",
            fontsize=9, xytext=(6,30), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:green"))
ax.annotate(f"PCIe crossover\nB≈{x_pcie:.0f} ops", (x_pcie, DISP_PCIE*1e6*1.05), color="tab:blue",
            fontsize=9, xytext=(6,18), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:blue"))

ax.set_xlabel("ops per query  B  (point-ops in one statement / concurrent in-flight)")
ax.set_ylabel("query latency (us, log) — lower better")
ax.set_title("Iso-query crossover: CPU vs GPU (modeled as C2C dispatch + kernel)")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_iso_query_crossover.png"), dpi=140)
print(f"crossover C2C B~{x_c2c:.0f}, PCIe B~{x_pcie:.0f}; wrote fig7_iso_query_crossover.png")
