#!/usr/bin/env python3
"""GH200 campaign figures (fig13-15). All constants [measured] 2026-07-07 on
Lambda GH200 480GB (H100 96GB sm_90, Grace 64x Neoverse-V2, C2C enabled,
driver 570.148.08, CUDA 12.8) unless labeled otherwise.
Source: bench/gh200_campaign_results.md"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------- Fig 13: the doorbell ladder + Fusco replication ----------
labels = ["two-flag\nvolatile+fence\n(our v1 protocol)",
          "CAS acq_rel\nunpinned\n(first attempt)",
          "CAS relaxed\n+ pinned\n(Fusco protocol)",
          "Fusco et al.\npublished\n(Alps, Fig.13)"]
vals   = [1.941, 1.182, 0.693, 0.833]   # us; DDR-flag config; 0.693 = mean(0.679, 0.706)
colors = ["tab:blue", "tab:blue", "tab:green", "0.45"]
fig, ax = plt.subplots(figsize=(8.5, 5.0))
bars = ax.bar(labels, vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.2f} us", ha="center", fontsize=10)
ax.axhline(1.0, color="tab:red", ls="--", alpha=0.7)
ax.text(2.55, 1.04, "1 us = pre-registered band-1 boundary\n(\"confirmed for ordinary serving\")",
        fontsize=8.5, color="tab:red")
ax.axhline(5.0, color="k", ls=":", alpha=0.4)
ax.set_ylim(0, 2.3)
ax.set_ylabel("CPU<->GPU full exchange latency (us)")
ax.set_title("Fig 13. NVLink-C2C doorbell on GH200 [measured 2026-07-07]\n"
             "Grace<->Hopper, flag in LPDDR5X; replication of Fusco et al. beats published by 15-18%")
ax.text(0.02, 0.02, "protocol deltas recovered from gh_benchmark source:\n"
        "relaxed CAS orders + pinned host thread (worth ~0.5 us)",
        transform=ax.transAxes, fontsize=8.5, va="bottom")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig13_doorbell_ladder.png"), dpi=140)
print("wrote fig13_doorbell_ladder.png")

# ---------- Fig 14: Grace two-regime scaling vs H100 ceiling ----------
T   = [1, 2, 4, 8, 16, 32, 64]
l3  = [108.6, 223.4, 451.4, 950.0, 1791.2, 3254.6, 6000.1]   # 2M keys / 50MB table
dram= [42.9,  85.4, 168.7, 322.9, 578.7,  917.2, 1187.1]     # 64M keys / 1.6GB table
GPU_CEIL = 4964.2
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(T, l3,   "s-", color="tab:orange", lw=2, ms=7,
        label="Grace, 50 MB table (fits ~114 MB L3)")
ax.plot(T, dram, "o-", color="tab:red", lw=2, ms=7,
        label="Grace, 1.6 GB table (DRAM-resident)")
ax.axhline(GPU_CEIL, color="tab:blue", ls="--", lw=2,
           label="H100 RGI find ceiling (4,964 Mop/s)")
for x, y in zip(T, l3):   ax.annotate(f"{y:,.0f}", (x, y), fontsize=7.5, xytext=(3,5),  textcoords="offset points", color="tab:orange")
for x, y in zip(T, dram): ax.annotate(f"{y:,.0f}", (x, y), fontsize=7.5, xytext=(3,-11), textcoords="offset points", color="tab:red")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(T); ax.set_xticklabels([str(t) for t in T])
ax.set_xlabel("Grace threads (Neoverse-V2, pinned)")
ax.set_ylabel("point-lookup throughput (Mop/s, log)")
ax.set_title("Fig 14. The two-regime result [measured 2026-07-07, GH200]\n"
             "cache-resident: Grace beats the GPU outright; memory-resident: GPU ceiling = 4.2x Grace")
ax.annotate("64T L3-resident: 6,000 Mop/s\nEXCEEDS the GPU ceiling\n-> GPU offload pointless here",
            (64, 6000), fontsize=8.5, color="tab:orange", ha="right",
            xytext=(-12, -52), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:orange"))
ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig14_grace_two_regime.png"), dpi=140)
print("wrote fig14_grace_two_regime.png")

# ---------- Fig 15: on-target crossover (memory-resident regime) ----------
# GPU measured (patched launch sweep, 64M keys)
gB   = [1, 8, 64, 512, 4096, 32768, 262144, 1048576]
gthr = [0.2, 0.7, 2.6, 20.8, 163.4, 1260.3, 3634.6, 4964.2]
CEIL = 4964.2e6
CPU_DRAM = 1187.1   # Mop/s, Grace 64T memory-resident
CPU_L3   = 6000.1
B = np.logspace(0, 6.2, 500)
def thr_model(D_us):   # dispatch D + B/ceiling
    return B / (D_us*1e-6 + B/CEIL) / 1e6
fig, ax = plt.subplots(figsize=(9.5, 5.8))
ax.plot(B, thr_model(0.7),  color="tab:green", lw=2, ls="--",
        label="doorbell dispatch D=0.7 us [measured primitive]")
ax.plot(B, thr_model(1.2),  color="tab:green", lw=1.4, ls=":",
        label="doorbell dispatch D=1.2 us [measured, fully ordered]")
ax.plot(B, thr_model(25.0), color="tab:blue", lw=1.4, alpha=0.7,
        label="launch dispatch ~25 us [measured plateau]")
ax.scatter(gB, gthr, s=75, color="tab:blue", marker="o", edgecolor="k", zorder=5,
           label="H100 launch sweep [measured]")
def fmtB(b): return f"{b/1e6:.0f}M" if b>=1e6 else (f"{b/1e3:.0f}k" if b>=1e3 else str(int(b)))
for b, t in zip(gB, gthr):
    ax.annotate(f"B={fmtB(b)}\n{t:,.0f}", (b, t), fontsize=7.5, color="tab:blue",
                xytext=(4, 5), textcoords="offset points")
ax.axhline(CPU_DRAM, color="tab:red", lw=2,
           label=f"Grace 64T, memory-resident ({CPU_DRAM:,.0f} Mop/s)")
ax.axhline(CPU_L3, color="0.5", ls=":", lw=1.5,
           label=f"Grace 64T, L3-resident ({CPU_L3:,.0f}) - GPU never wins")
# crossover stars vs the DRAM-regime CPU line
slope = (1/(CPU_DRAM*1e6) - 1/CEIL)
for D_us, col, tag in [(0.7, "tab:green", "D=0.7us"), (1.2, "tab:green", "D=1.2us")]:
    xs = D_us*1e-6 / slope
    ax.scatter([xs], [CPU_DRAM], marker="*", s=320, color=col, edgecolor="k", zorder=6)
    ax.annotate(f"GPU wins B>~{xs:,.0f}\n({tag})", (xs, CPU_DRAM), fontsize=8.5,
                color="tab:green", fontweight="bold",
                xytext=(6, 12 if D_us < 1 else -34), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=col))
ax.annotate("launch binding crosses ~30k [measured]", (32768, 1260), fontsize=8.5,
            color="tab:blue", xytext=(-150, 26), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:blue"))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("in-flight point operations per dispatch  B")
ax.set_ylabel("throughput (Mop/s, log)")
ax.set_title("Fig 15. On-target crossover, GH200 [measured 2026-07-07]\n"
             "64M-key working set (memory-resident both sides); every constant measured on this machine")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig15_gh200_crossover.png"), dpi=140)
print("wrote fig15_gh200_crossover.png")

# ---------- Fig 16: AMORTIZED per-op latency vs batch size (the single-plot answer) ----------
# Point lookups (= SELECT WHERE k = ?), GH200, 64M-key working set,
# memory-resident on both sides. Amortized latency = batch latency / B.
gB    = np.array([1, 8, 64, 512, 4096, 32768, 262144, 1048576])
glat  = np.array([4.25, 11.49, 24.25, 24.64, 25.06, 26.00, 72.13, 211.23])  # us [measured]
g_amort = glat * 1000.0 / gB                          # ns per op
KERN = 0.2014                                          # ns/op at the 4,964 Mop/s ceiling
CPU64 = 0.8425                                         # ns/op, Grace 64T, DRAM-resident [measured]
CPU1  = 23.32                                          # ns/op, Grace 1T, DRAM-resident [measured]
Bm = np.logspace(0, 6.2, 500)
d07 = 700.0/Bm + KERN                                  # primitive bound (model), D=0.7 us
d12 = 1200.0/Bm + KERN                                 # primitive bound (model), D=1.2 us

# persistent v2 (engine/rgi_persist2.cu) [measured end-to-end 2026-07-07,
# 64M-key table, uniform-random FIND]. Crossover region (B=256..2048) uses the
# statistically hardened 'cross' mode: 2 independent runs x 20 reps x 50,000
# batches/rep, submit thread on core 32 (core-0 pinning bimodalized reps);
# value shown = the WORSE (higher) of the two run means, sigma ~0.002 ns/op.
# Other points: original sweep run (runs agree <2% there).
v2B    = np.array([64, 256, 384, 512, 640, 768, 1024, 1536, 2048, 4096,
                   8192, 16384, 32768, 65536, 131072, 262144, 1048576])
v2sync = np.array([10.02, 9.92, np.nan, 10.27, np.nan, 10.49, 10.91, np.nan,
                   12.36, 14.38, 18.56, 21.27, 28.84, 39.77, 60.53, 101.71,
                   334.79])                                     # us/batch, W=1
v2pipe = np.array([4.298, 1.351, 1.030, 0.851, 0.830, 0.824, 0.808, 0.785,
                   0.778, 0.742, 0.733, 0.647, 0.498, 0.392, 0.339, 0.321,
                   0.303])                                      # ns/op, W<=64
v2sync_amort = v2sync * 1000.0 / v2B

fig, ax = plt.subplots(figsize=(9.5, 6.0))
ax.plot(Bm, d07, color="0.55", lw=1.4, ls="--",
        label="primitive bound (model): D=0.7 us + kernel ceiling")
ax.plot(Bm, d12, color="0.55", lw=1.1, ls=":",
        label="primitive bound (model): D=1.2 us + kernel ceiling")
ax.plot(gB, g_amort, "o-", color="tab:blue", lw=1.6, ms=7,
        label="GPU, launch dispatch [measured]")
m = ~np.isnan(v2sync_amort)
ax.plot(v2B[m], v2sync_amort[m], "s-", color="tab:purple", lw=1.6, ms=6,
        label="GPU, doorbell v2 sync (1 in flight) [MEASURED end-to-end]")
ax.plot(v2B, v2pipe, "D-", color="tab:green", lw=2.2, ms=6,
        label="GPU, doorbell v2 pipelined (<=64 in flight) [MEASURED end-to-end]")
ax.axhline(CPU64, color="tab:red", lw=2,
           label=f"Grace CPU, 64 threads saturated ({CPU64:.2f} ns/op)")
ax.axhline(CPU1, color="tab:red", lw=1.2, ls=":",
           label=f"Grace CPU, single thread ({CPU1:.1f} ns/op)")
# MEASURED crossover stars (intersections of the measured v2 series with CPU lines)
ax.scatter([640], [CPU64], marker="*", s=380, color="tab:green", edgecolor="k", zorder=6)
ax.annotate("pipelined v2 beats saturated 64T CPU\nfrom B = 640 [40/40 reps, 2 runs, end-to-end]\n(B=512 within ~1% of the line; model said ~1,100)",
            (640, CPU64), fontsize=8.5, color="tab:green", fontweight="bold",
            xytext=(-170, 40), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:green"))
ax.scatter([65536], [CPU64], marker="*", s=300, color="tab:purple", edgecolor="k", zorder=6)
ax.annotate("sync v2 crosses at B = 65k [measured]",
            (65536, CPU64), fontsize=8, color="tab:purple",
            xytext=(12, 26), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:purple"))
ax.annotate("launch binding crosses at B ~ 30k [measured]",
            (32768, g_amort[5]), fontsize=8, color="tab:blue",
            xytext=(30, 24), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="tab:blue"))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("batch size B (operations per dispatch)")
ax.set_ylabel("amortized latency per operation (ns/op, log)")
ax.set_title("Fig 16. Amortized per-operation latency: CPU vs GPU point lookups (SELECT WHERE k = ?)\n"
             "GH200 [measured 2026-07-07], 64M-key working set, memory-resident on both sides")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig16_amortized_latency.png"), dpi=140)
print("wrote fig16_amortized_latency.png")
