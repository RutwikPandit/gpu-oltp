#!/usr/bin/env python3
"""Fig 17: GH200 roofline + practical random-access roof.

All constants are [measured] on Lambda GH200 480GB, 2026-07-07 unless
explicitly marked. Sources:
  - bench/prof_raw/membench_*_4g.txt
  - bench/prof_raw/ncu_rgi_*_details.csv
  - bench/gh200_campaign_results.md §7
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

# H100 SXM FP32 non-tensor peak, used for the classic roofline.
FP32_PEAK_GF = 67000.0

# Membench [measured]. Decimal GB/s, as printed by membench.
STREAM_READ_GBS = 3812.3
STREAM_TRIAD_GBS = 3567.8
RANDOM_32_GBS = 1191.7
RANDOM_64_GBS = 1402.0
RANDOM_128_GBS = 1404.0

# Nsight Compute [measured] on rgi_profile_one, same cooperative_find path.
B = np.array([32768, 262144, 1048576], dtype=float)
NCU_MEM_GBS = np.array([122.27, 638.84, 934.24])
NCU_DRAM_PCT = np.array([3.04, 15.88, 23.23])

# End-to-end launch/v2 ceilings [measured], converted with ~270 B/probe
# (bucket line + suffix line + key/out traffic; see results §7).
BYTES_PER_PROBE = 270.0
LAUNCH_GOPS = 4.962
V2_GOPS = 3.293
LAUNCH_GBS_EST = LAUNCH_GOPS * BYTES_PER_PROBE
V2_GBS_EST = V2_GOPS * BYTES_PER_PROBE

# The index does ~0 FP work. To show points on a classic GFLOP/s roofline,
# use one notional operation per probe: AI = 1 / bytes_per_probe. The labels
# state this is a placement convention, not a compute-throughput claim.
ai_index = 1.0 / BYTES_PER_PROBE
ai = np.logspace(-3, 2.2, 600)
stream_roof = np.minimum(FP32_PEAK_GF, STREAM_READ_GBS * ai)
random_roof = RANDOM_128_GBS * ai
ridge = FP32_PEAK_GF / STREAM_READ_GBS

fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.plot(ai, stream_roof, color="tab:blue", lw=2.2,
        label=f"classic streaming roof: {STREAM_READ_GBS:,.0f} GB/s read [measured]")
ax.plot(ai, random_roof, color="tab:green", lw=2.0, ls="--",
        label=f"practical random-128B roof: {RANDOM_128_GBS:,.0f} GB/s useful [measured]")
ax.axhline(FP32_PEAK_GF, color="0.35", lw=1.5, ls=":", label="FP32 peak ~67 TFLOP/s [spec]")
ax.axvline(ridge, color="tab:red", lw=1.4, ls=":")
ax.annotate(f"classic ridge\n{ridge:.1f} FLOP/B",
            (ridge, FP32_PEAK_GF), xytext=(-92, -50), textcoords="offset points",
            color="tab:red", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="tab:red"))

ax.scatter([ai_index], [LAUNCH_GBS_EST * ai_index], marker="o", s=95,
           color="tab:purple", edgecolor="k", zorder=5,
           label=f"RGI launch ceiling: {LAUNCH_GOPS:.2f} Gop/s, ~{LAUNCH_GBS_EST:,.0f} GB/s est.")
ax.scatter([ai_index], [V2_GBS_EST * ai_index], marker="D", s=85,
           color="tab:orange", edgecolor="k", zorder=5,
           label=f"persistent v2: {V2_GOPS:.2f} Gop/s, ~{V2_GBS_EST:,.0f} GB/s est.")
ax.annotate("index probes are ~0-FLOP;\npoints use 1 notional op/probe\nonly to show memory-roof placement",
            (ai_index, LAUNCH_GBS_EST * ai_index), xytext=(22, 38),
            textcoords="offset points", fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="0.25"))

for b, gbs, pct in zip(B, NCU_MEM_GBS, NCU_DRAM_PCT):
    ax.scatter([ai_index * 1.25], [gbs * ai_index], color="black", s=45, zorder=4)
    ax.annotate(f"B={int(b):,}\nNCU mem {gbs:.0f} GB/s\nDRAM {pct:.1f}%",
                (ai_index * 1.25, gbs * ai_index),
                xytext=(8, -4 if b < 1e6 else -30), textcoords="offset points",
                fontsize=7.8)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("arithmetic intensity (FLOP/B, log)")
ax.set_ylabel("GFLOP/s roofline scale (log)")
ax.set_title("Fig 17. GH200 roofline for GPU-resident index probes [measured 2026-07-07]\n"
             "The relevant roof is random-access bandwidth, not the classic FP32 ridge")
ax.grid(alpha=0.28, which="both")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig17_roofline.png"), dpi=150)
print("wrote fig17_roofline.png")
