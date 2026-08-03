#!/usr/bin/env python3
"""Realistic OLTP workload: throughput vs offered concurrency.
CPU (16 cores) vs GPU PCIe (measured 16us dispatch) vs GPU C2C (projected 0.5us).
A real server batches the C concurrent in-flight ops into one GPU dispatch window.
Constants measured this session except C2C dispatch (projected, GH200)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
OUT = os.path.dirname(os.path.abspath(__file__))

DISP_PCIE = 16e-6
DISP_C2C  = 0.5e-6
CPU_CORES = 16

# measured ceilings (Mop/s) and CPU per-core
# YCSB-C row measured 2026-06-08 (read = RGI find / CPU hash find).
# YCSB-A row remains a model estimate (mixed insert/update ceiling not re-measured).
WL = {
  "YCSB-C  (100% read)":      dict(gpu_ceil=1249e6, cpu_ceil=296e6, cpu_percore=45.3e6),
  "YCSB-A  (50% read/50% upd)":dict(gpu_ceil=784e6,  cpu_ceil=225e6, cpu_percore=32.0e6),
}
C = np.logspace(0, 6, 400)   # concurrent in-flight ops

def gpu_thr(C, ceil, disp):   # batch = C, one dispatch window
    return C / (disp + C/ceil)
def cpu_thr(C, ceil, percore):
    return np.minimum(C*percore, ceil)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
rows = []
for ax, (name, p) in zip(axes, WL.items()):
    tc  = cpu_thr(C, p["cpu_ceil"], p["cpu_percore"])/1e6
    tp  = gpu_thr(C, p["gpu_ceil"], DISP_PCIE)/1e6
    t2  = gpu_thr(C, p["gpu_ceil"], DISP_C2C )/1e6
    ax.plot(C, tc, color="tab:orange", lw=2.5, label="CPU (16 cores)")
    ax.plot(C, tp, color="tab:blue",  lw=2, ls=":", label="GPU PCIe (16us dispatch)")
    ax.plot(C, t2, color="tab:green", lw=2.5, label="GPU C2C (0.5us, projected)")
    ax.axvspan(50, 2000, alpha=0.08, color="gray")
    ax.text(60, 1.2, "typical OLTP\nin-flight", fontsize=8, color="gray")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("concurrent in-flight ops"); ax.set_ylabel("throughput (Mop/s)")
    ax.set_title(name); ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=8)
    # crossovers vs CPU ceiling
    slope = (1/p["cpu_ceil"] - 1/p["gpu_ceil"])
    xpcie = DISP_PCIE/slope; xc2c = DISP_C2C/slope
    rows.append((name, xc2c, xpcie,
                 gpu_thr(256,p["gpu_ceil"],DISP_PCIE)/1e6, gpu_thr(256,p["gpu_ceil"],DISP_C2C)/1e6,
                 p["cpu_ceil"]/1e6))
fig.suptitle("Realistic OLTP throughput vs concurrency: CPU vs GPU (PCIe & C2C)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig10_realistic_workload.png"), dpi=140)

print("%-28s %-12s %-12s %-14s %-14s %-10s" %
      ("workload","C2C xover","PCIe xover","GPU@256 PCIe","GPU@256 C2C","CPU ceil"))
for r in rows:
    print("%-28s %-12.0f %-12.0f %-14.1f %-14.1f %-10.0f" % r)
print("(xover = in-flight ops where GPU overtakes 16-core CPU; @256 = throughput Mop/s at 256 in-flight)")
print("wrote fig10_realistic_workload.png")
