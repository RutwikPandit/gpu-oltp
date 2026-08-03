#!/usr/bin/env python3
"""Presentation-specific GH200 update figures.

These are simplified talk figures generated with the same matplotlib style as
the earlier campaign plots. They intentionally keep only the comparison the
slide needs, while the raw/full plots stay in bench/make_plots8.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = os.path.dirname(os.path.abspath(__file__))


def write_crossover():
    b = np.array([256, 512, 640, 1024, 4096, 32768, 65536, 1048576])
    gpu_ns = np.array([1.351, 0.851, 0.830, 0.808, 0.742, 0.498, 0.392, 0.303])
    cpu_ns = 0.842

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(b, gpu_ns, "D-", color="tab:green", lw=2.4, ms=7,
            label="GH200 persistent runtime [measured end-to-end]")
    ax.axhline(cpu_ns, color="tab:red", lw=2.2,
               label=f"Grace 64T DRAM baseline ({cpu_ns:.3f} ns/op)")
    ax.scatter([640], [0.830], marker="*", s=360, color="tab:green",
               edgecolor="k", zorder=5)
    ax.annotate("claimed crossover\nB ~= 640\n40/40 reps",
                (640, 0.830), fontsize=9, fontweight="bold",
                color="tab:green", xytext=(32, 42),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="tab:green"))
    ax.annotate("best runtime point\n0.303 ns/op at B=1M",
                (1048576, 0.303), fontsize=8.5, color="tab:green",
                xytext=(-160, 42), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="tab:green"))

    ax.set_xscale("log")
    ax.set_ylim(0.22, 1.55)
    ax.set_xlabel("batch size B (operations per dispatch)")
    ax.set_ylabel("amortized latency per operation (ns/op)")
    ax.set_title("End-to-end index probe cost: Grace CPU vs GH200 runtime\n"
                 "memory-resident ~1.6 GB index; lower is better")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig18_cpu_gpu_crossover_simple.png"), dpi=140)
    print("wrote fig18_cpu_gpu_crossover_simple.png")


def write_throughput():
    labels = ["Grace 64T\nDRAM baseline", "persistent\nruntime", "RGI launch\nceiling"]
    vals = [1.187, 3.3, 4.964]
    colors = ["tab:red", "tab:blue", "tab:green"]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars = ax.bar(labels, vals, color=colors, width=0.58)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12,
                f"{val:.2f} Gop/s", ha="center", fontsize=10)

    ax.annotate("2.8x over\nGrace DRAM",
                xy=(1, vals[1]), xytext=(0.32, 4.05),
                textcoords="data", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="0.2"))
    ax.annotate("runtime reaches\n~2/3 of kernel ceiling",
                xy=(1, vals[1]), xytext=(1.38, 2.45),
                textcoords="data", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.2"))

    ax.set_ylim(0, 5.8)
    ax.set_ylabel("point-lookup throughput (Gop/s)")
    ax.set_title("GH200 throughput summary [measured]\n"
                 "memory-resident 64 million-key / ~1.6 GB table")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig19_gh200_throughput_summary.png"), dpi=140)
    print("wrote fig19_gh200_throughput_summary.png")


if __name__ == "__main__":
    write_crossover()
    write_throughput()
