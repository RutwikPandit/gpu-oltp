# GH200 Measurement Campaign — Results (2026-07-07)

**The first on-target C2C measurements for the project. All numbers
`[measured]` on the machine below unless marked otherwise.**

**2026-07-07 update (second session): §4b adds persistent v2 — the
measured end-to-end dispatch path. Headline: pipelined v2 beats the
saturated 64-thread Grace baseline from B ≈ 640 onward (statistically
solid: 40/40 repetitions across two independent runs; B = 512 is
within noise of the line) — the fig16 "B\* ≈ 1,100" model claim is
replaced by a MEASURED crossover better than the model. §5 re-derived
accordingly. §7 adds the microarchitectural characterization.**

## Provenance

| | |
|---|---|
| Machine | Lambda cloud instance, NVIDIA GH200 480GB superchip |
| GPU | H100 side: 96 GB HBM3 (97,871 MiB visible), compute capability 9.0, 132 SMs |
| CPU | Grace: 64× Neoverse-V2 (aarch64), single socket |
| Memory | NUMA node0 = 452 GB LPDDR5X (Grace); NUMA node1 = 99 GB HBM3 (GPU, CPU-less node) |
| Coherence | `nvidia-smi -q`: **GPU C2C Mode: Enabled** |
| Software | Driver 570.148.08, CUDA 12.8 (nvcc V12.8.93), gcc 11.4, kernel 6.8.0-1013-nvidia-64k (**64 KB pages**), Ubuntu 22.04 |
| Build | `-arch=sm_90 -maxrregcount=64`, same flags as laptop otherwise |
| CPU runs | `OMP_PROC_BIND=close OMP_PLACES=cores` |
| Date | 2026-07-07, single session |

**Measurement patch (remote copy only, flagged for upstream):** RGI's
`launch_batch_kernel` calls `cudaGetDeviceProperties` on every launch;
on this driver stack that call costs **~875 µs** and dominated all
launch-path measurements (first sweep read ~880 µs flat at every batch
size). Patched on the remote copy only to cache launch geometry per
template instantiation (`cudaDeviceGetAttribute` once). All launch
numbers below are with the patch. **Action: report upstream to RGI.**

## 1. The doorbell (WP7 experiment 1 — the decisive number)

Bare 4-byte flag ping-pong, resident single-thread kernel, tight spin,
20,000 iterations after warmup (`engine/doorbell.cu`):

| Flag placement / allocation | Round trip |
|---|---|
| **`cudaHostAllocMapped` (both flags host-pinned)** | **1.740 µs** |
| `malloc` system memory (ATS coherent) | 2.656 µs |
| `cudaMallocManaged` | 2.628 µs |
| split: req preferred-on-GPU, ack preferred-on-host | 2.108 µs |

Reference points: same handshake on WSL2/WDDM laptop = 75.5 µs
(engine-level). Toy-engine full handshake on GH200 = 8.84 µs
(includes 1024-thread block syncs, 128 ns sleep quantization, payload).

**Verdict against the pre-registered decision boundaries
(plan/WP7_gbclass_campaign.md):** 1.74 µs falls in the **1–3 µs band →
the thesis holds for high-throughput transactional serving; the
"ordinary low-latency OLTP" claim is narrowed accordingly.** The
sub-microsecond projection (0.5 µs) was not met on this stack (3.5×);
per-direction cost ≈ 0.87 µs is consistent with published GH200
ping-pong data. Notably the engine's existing allocation choice
(mapped) is the fastest of the four modes.

## 1b. Replication of Fusco et al. Fig. 13 — SUCCESSFUL, slightly beats published

Protocol matched exactly against their released source
(github.com/luigifusco/gh_benchmark, `src/atomic_benchmarks.cuh`):
ONE single-byte system-scope CAS flag (`cuda::std::atomic<uint8_t>`),
Grace ping / Hopper pong, **`memory_order_relaxed` on both CAS
orders**, **host thread pinned** (their `pthread_setaffinity_np`; our
`sched_setaffinity` to core 0). `doorbell.cu` `cas_*` modes.

| Flag placement | relaxed + pinned (their protocol) | acq_rel, unpinned (first attempt) | Fusco et al. (Alps) |
|---|---|---|---|
| **Grace LPDDR5X** | **0.679–0.706 µs** | 1.18–1.23 µs | **0.833 µs** |
| Hopper HBM3 (managed, preferred-GPU) | 0.863–0.864 µs | 1.11 µs | — |
| pinned mapped | 0.763 µs | 1.23 µs | — |

**Verdict: fully replicated — and 15–18% faster than the published
number in the directly comparable configuration** (DDR flag: 0.68–0.71
vs 0.833 µs). The two protocol details recovered from their source
were worth ~0.5 µs: relaxed CAS orders (acq_rel compiles to barriered
atomics on this path) and host-thread pinning (unpinned spinning on a
64-core Grace adds scheduler jitter).

**Consequence for the model — the verdict tightens by one band:**
- Primitive doorbell floor: **~0.7 µs = sub-microsecond, at the
  pre-registered band-1 boundary ("confirmed for ordinary serving").**
  Honest caveat attached: a real dispatch protocol must order payload
  against the flag, which costs something between the relaxed
  primitive (0.68 µs) and the fully-ordered variant (1.11–1.28 µs).
  The engineering target for the persistent runtime's doorbell is
  therefore **0.7–1.2 µs**, bracketed by measurement on both ends.
- Recomputed crossovers vs. saturated 64-thread Grace
  (memory-resident): D=0.7 µs → **B\* ≈ 1,100**; D=1.2 µs →
  **B\* ≈ 1,870**. (Prior two-flag estimate ≈2,700 is superseded.)
- The future persistent runtime must use a CAS-flag doorbell with
  relaxed polling + explicit payload publication, not the two-flag
  volatile+fence handshake (measured 0.7 µs dearer).

## 2. Launch floor and H100 ceiling (RGI chain hashtable, patched)

`rgi_sweep`, find-only batches:

| B | 2M keys: latency / throughput | 64M keys: latency / throughput |
|---|---|---|
| 1 | 4.14 µs / 0.2 Mop/s | 4.25 µs / 0.2 Mop/s |
| 8 | 11.37 µs / 0.7 | 11.49 µs / 0.7 |
| 64 | 24.15 µs / 2.7 | 24.25 µs / 2.6 |
| 512 | 24.45 µs / 20.9 | 24.64 µs / 20.8 |
| 4,096 | 24.90 µs / 164.5 | 25.06 µs / 163.4 |
| 32,768 | 25.87 µs / 1,266.6 | 26.00 µs / 1,260.3 |
| 262,144 | 65.17 µs / 4,022.6 | 72.13 µs / 3,634.6 |
| 1,048,576 | 216.64 µs / 4,840.2 | 211.23 µs / **4,964.2** |

- True Linux launch floor: **4.1 µs at B=1**; a second plateau
  (~24–26 µs, B=64…32k) reflects full-grid execution cost.
- **H100 RGI find ceiling ≈ 4.9 Gop/s** — 3.9× the RTX 4060's
  1.25 Gop/s — and insensitive to table size (HBM-resident regime).
- All numbers still include the suffix-node hop (2-slice keys); the
  WP1 small-key fast path applies here too.

## 3. Grace CPU baseline — the two-regime finding

`cpu_sweep` (open-addressing hash, extended to 64 threads):

| Threads | 2M keys / 50 MB table (fits ~114 MB L3) | 64M keys / 1.6 GB table (DRAM regime) |
|---|---|---|
| 1 | 108.6 Mop/s (9.2 ns/op) | 42.9 Mop/s (23.3 ns/op) |
| 8 | 950.0 | 322.9 |
| 16 | 1,791.2 | 578.7 |
| 32 | 3,254.6 | 917.2 |
| 64 | **6,000.1** | **1,187.1** (53.9 ns/op) |

**The two-regime result (this is the honest headline for the CPU
side):**
- **Cache-resident working sets (≲100 MB): Grace wins outright** —
  6.0 Gop/s exceeds the H100 index ceiling (4.9 Gop/s). GPU offload
  is pointless in this regime and the write-up must say so.
- **Memory-resident working sets (the DB-scale regime): GPU ceiling =
  4.2× Grace** (4,964 vs 1,187 Mop/s) — strikingly, the same 4.2×
  ratio measured on the laptop (1,249 vs 296).

## 4. Persistent binding v1 on GH200 (`rgi_persist`, 2M keys)

1,188 resident blocks (9/SM × 132 SMs). Correctness: **1024/1024 on
persistent FIND and persistent INSERT+FIND.**

| B | launch | persistent v1 |
|---|---|---|
| 1 | 4.14 µs | 14.23 µs |
| 4,096 | 24.91 µs / 164 Mop/s | 41.40 µs / 99 Mop/s |
| 1,048,576 | 217.6 µs / 4,819 Mop/s | 325.9 µs / 3,218 Mop/s |

**Finding: the v1 persistent port is not doorbell-bound.** The bare
doorbell costs 1.74 µs, but v1's rendezvous costs 14.2 µs because of
(a) per-rendezvous payload staging through `cudaMemcpyAsync` +
`cudaStreamSynchronize` (unnecessary on C2C — the host can write
request payload directly into coherent memory), and (b) the grid-wide
arrival barrier across 1,188 blocks. Concrete headroom: ~8× between
the realized rendezvous and the primitive. This defines the next
persistent-runtime work item (zero-copy payload path + cheaper
completion), and until then **launch-per-batch remains the better
binding even on GH200** — now a measured statement on two platforms.

## 4b. Persistent v2 (`engine/rgi_persist2.cu`) — the MEASURED end-to-end path

The redesign motivated by §4's finding. Same 64M-key RGI chain hashtable
in HBM, uniform-random FIND keys in [1, 64M] (YCSB-C read shape),
prewritten 1M-entry request array with a rotating offset (payload
materialization excluded and symmetric with the CPU baseline, which also
reads prewritten arrays). Timed rendezvous = doorbell + probes +
completion. Validation: 4096/4096 at offset 0 and across the offset
wrap, 1024/1024 INSERT+FIND, 512/512 pipelined. Two consecutive runs
agreed to <2% at every point.

**Architecture (each stage chosen by measurement; see the file header
for the alternatives that lost):**
- One packed 64-bit doorbell word per batch, host-posted into a ring in
  managed-HBM (single 8 B store, no CUDA calls in the timed loop).
- 8 dispatcher blocks shard batches round-robin: leader polls the ring
  from L2, its 128 threads fan the word out to the participating serving
  blocks' private mailbox lines in parallel.
- 1,312 serving blocks; a leader polls only its own mailbox. Batch b is
  served by a rotating window of `nact = ceil(B/8)` blocks, so pipelined
  batches land on disjoint block sets; strided tiles give every 16-lane
  tile at most one op for B <= 10,496 (fixes v1's 16-serial-ops flaw).
- Requests AND results live in managed-HBM (consumer-side placement);
  moving `out[]` from mapped-host to HBM alone moved the pipelined
  crossover from B = 16k to B = 1k — per-op C2C stores were the last
  per-op serial cost. Completion = per-generation arrival counter, last
  arriver posts one tag byte to a mapped-host done line.

**Protocol floors (count = 0, no probes): sync rendezvous 6.2 µs;
pipelined dispatch cadence 0.24 µs/batch (W=64).** The NOOP sweep
(dispatch + wake + completion, zero probes) confirms the protocol is not
the limiter at the crossover: 0.45 ns/op of protocol at B = 1,024.

| B | sync (µs) | sync ns/op | pipelined ns/op | pipelined Mop/s |
|---|---|---|---|---|
| 64 | 10.0 | 156.6 | 4.30 | 233 |
| 256 | 9.9 | 38.8 | 1.37 | 729 |
| **512** | 10.3 | 20.1 | **0.832** | **1,202** |
| 768 | 10.5 | 13.7 | 0.802 | 1,248 |
| 1,024 | 10.9 | 10.7 | 0.788 | 1,269 |
| 4,096 | 14.4 | 3.51 | 0.742 | 1,348 |
| 32,768 | 28.8 | 0.880 | 0.498 | 2,009 |
| 65,536 | 39.8 | 0.607 | 0.392 | 2,551 |
| 1,048,576 | 334.8 | 0.319 | 0.303 | 3,299 |

(sync = one batch in flight, the per-batch latency number; pipelined =
up to 64 batches in flight on disjoint block sets, the peak-serving
throughput number. The fig16 model line implicitly assumes the pipelined
regime — the saturated 64T CPU baseline gets the same courtesy.)

**Statistical resolution of the crossover — DEFINITIVE
(`rgi_persist2 <N> cross` mode).** Two independent processes (one each
side of an instance reboot), 20 repetitions per point per process;
each repetition is an independent window-fill + **50,000** timed
batches; B values interleaved across reps so drift spreads over all
points. Two measurement fixes found on the way, both kept in the code:
(a) the submit thread must NOT be pinned to core 0 — kernel
housekeeping/IRQs land there and preempt the spin loop, producing
bimodal reps (0.78 vs 3+ ns/op); core 32 collapses run-to-run σ to
~0.002 ns/op. (b) An earlier 5-full-sweep-run analysis (2,000-iter
points, core-0 pinning; logs in `bench/prof_raw_box/rerun_1..4.log`)
gave medians consistent with the table below (0.847 at 512, 0.802 at
768) but with occasional 3–6× outlier runs — now explained by (a), and
superseded by:

| B | run A mean ± σ ns/op | run B mean ± σ | reps < 0.842 (of 40) |
|---:|---|---|---|
| 256 | 1.351 ± 0.004 | 1.337 ± 0.004 | 0 |
| 384 | 1.030 ± 0.001 | 1.019 ± 0.001 | 0 |
| 512 | 0.851 ± 0.001 | 0.837 ± 0.001 | 20 — **marginal, flips between runs** |
| **640** | **0.828 ± 0.001** | **0.830 ± 0.002** | **40/40, mean+2σ below** |
| 768 | 0.820 ± 0.002 | 0.824 ± 0.002 | 40/40 |
| 1,024 | 0.805 ± 0.002 | 0.808 ± 0.002 | 40/40 |
| 1,536 | 0.783 ± 0.002 | 0.785 ± 0.003 | 40/40 |
| 2,048 | 0.773 ± 0.003 | 0.778 ± 0.003 | 40/40 |

**Findings:**
1. **Measured end-to-end crossover vs saturated 64T Grace
   (0.842 ns/op): B ≈ 640 (pipelined) — below the CPU line in 40/40
   repetitions across two independent runs, mean+2σ clear of the
   line. B = 512 sits within ~1% of the line and flips between runs;
   we do not claim it.** The pre-registered model said B ≈ 1,100
   (D = 0.7 µs primitive); the measured path crosses EARLIER because
   the dispatch pipeline amortizes the doorbell to 0.24 µs/batch.
   Model confirmed; the measured number replaces it.
2. Sync (one-in-flight) crossovers: vs one Grace thread at B = 512
   (20 ns/op < 23.3); vs saturated 64T Grace at B = 65,536. A single
   in-flight batch cannot beat a saturated 64-core CPU at small B —
   the serving story requires overlap, and says so.
3. v2 pipelined throughput reaches 3.3 Gop/s at B = 1M — 2/3 of the
   launch-path ceiling (4.96 Gop/s); the remaining gap is the per-op
   protocol overheads (mailbox wake, rotation misalignment with L2).
4. v1's three bottlenecks, all confirmed by ablation on the box:
   payload staging (zero-copy mapped payload made it WORSE — every key
   deref became a 0.9 µs C2C round trip; HBM placement was the fix),
   grid-wide completion, and the 16-serial-ops tile mapping.

## 5. Recomputed crossovers (memory-resident regime, on-target numbers)

Per-op service times: Grace 64T = 0.842 ns; H100 = 0.201 ns;
marginal GPU advantage 0.641 ns/op.

| Dispatch path | B\* (GPU overtakes 64-thread Grace) |
|---|---|
| **Persistent v2, pipelined (§4b) — MEASURED end-to-end** | **B ≈ 640 (40/40 reps across 2 runs; B=512 marginal)** |
| Persistent v2, sync one-in-flight (§4b) — MEASURED | B = 65,536 |
| Launch binding, measured curve (~25 µs plateau) | ≈ 30,000 (GPU ahead at B=32,768) |
| CAS doorbell, relaxed primitive (0.7 µs, §1b) — model | ≈ 1,100 |
| CAS doorbell, fully ordered (1.2 µs, §1b) — model | ≈ 1,870 |
| (superseded) two-flag doorbell 1.74 µs — model | ≈ 2,700 |
| (for reference) original projection 0.5 µs | ≈ 780 |

Figures: `fig13_doorbell_ladder.png`, `fig14_grace_two_regime.png`,
`fig15_gh200_crossover.png`, `fig16_amortized_latency.png`
(regenerate via `bench/make_plots8.py`).

Honest reading: the model claim ("doorbell dispatch would cross at
B ≈ 1,100–1,870") is now superseded by a measured end-to-end result
that is BETTER than the model on the throughput side: **the pipelined
persistent runtime beats the saturated 64-thread Grace from
B ≈ 640 in-flight ops per dispatch [measured end-to-end, §4b;
40/40 repetitions across two independent runs]** — because the
dispatch pipeline amortizes the doorbell to 0.24 µs/batch, under the
0.7 µs primitive the model assumed. The honest caveats attached: (a)
this is the peak-serving regime — up to 64 batches in flight, i.e.
~40k total outstanding ops at B=640; a single in-flight batch crosses
at B ≈ 65k; (b) B = 512 is within ~1% of the CPU line and flips
between runs — the onset is 512–640, and 640 is what we claim;
(c) the two-regime rule stands — none of this pays for
cache-resident working sets (§3). The claim that survives, now
strengthened: **GPU-resident indexing pays on coherent hardware for
high-throughput serving of memory-resident working sets, from
sub-thousand-op dispatch batches upward.**

## 6. Campaign status vs. plan (WP7)

| Item | Status |
|---|---|
| 1. Doorbell microbenchmark (4 modes) | done |
| 2. On-target baselines + crossover | done (both regimes) |
| 3. Queue placement 2×2 | done (v2 ablations: ring/mailbox/done placement all measured) |
| 4. Payload-inline vs pointer | largely resolved by v2 (HBM-resident requests won; mapped zero-copy measured and rejected) |
| 5. Persistent vs launch under load | done (v1 + v2 redesign; v2 pipelined beats launch below B≈65k) |
| 6. Zipfian contention | open |
| 7. Full SQL stack on GH200 | open (needs Postgres install) |

Raw command lines and unabridged outputs: session transcript,
2026-07-07. Binaries remain on the instance at `~/work/bin/`.

## 7. Microarchitectural characterization (membench + Nsight)

**Provenance:** same GH200 instance and software stack as above, measured
2026-07-07. New additive files: `engine/membench.cu`,
`engine/rgi_profile_one.cu`, `bench/make_plots9.py`. Raw text/CSV
summaries copied into `bench/prof_raw/`; binary `.ncu-rep` and `.qdstrm`
reports remain on the box under `~/work/prof/`.

Nsight was not installed at handoff; installed `nsight-compute`
2025.1.1 and `nsight-systems` 2024.6.2 from the registered CUDA/Lambda
apt repos. `ncu` required `sudo`. `nsys` needed the bundled
`libbpf.so.1` on `LD_LIBRARY_PATH`; it produced a `.qdstrm` but the
importer was missing, so the v2 timeline artifact is the bounded run
stdout plus raw stream.

### 7.1 Hopper memory microbenchmarks (`engine/membench.cu`)

4 GiB `cudaMalloc` buffers, 1024 threads/block, block sweep
{33, 66, 132, 264, 528, 1056}; best value shown.

| Mode | Best result |
|---|---:|
| streaming read | **3,812 GB/s** |
| streaming write | **3,422 GB/s** |
| streaming copy (read+write bytes) | **3,448 GB/s** |
| streaming triad (2 reads + 1 write bytes) | **3,568 GB/s** |
| random 32 B useful granules | **1,192 GB/s** |
| random 64 B useful granules | **1,402 GB/s** |
| random 128 B useful granules | **1,404 GB/s** |

Interpretation: "90%+ bandwidth" is attainable for streaming kernels on
this box (read is ~95% of a 4.0 TB/s nominal HBM peak). The practical
roof for hash-index probes is much lower: **~1.4 TB/s useful random
bandwidth at 64-128 B granules**. That random roof, not the streaming
roof, is the relevant target for RGI find.

Pointer-chase latency, dependent single-thread chain, random
permutation: 16-64 KiB = 42-47 ns, 256 KiB = 186 ns, and large-HBM
points ranged roughly 310-655 ns depending on footprint/TLB behavior.
The stable takeaway is the expected cache cliff: sub-100 ns below L1,
~200 ns at the first cliff, and several hundred ns in HBM. The 2 GiB
sweep was stopped after the 512 MiB point to avoid burning the meter;
the older partial run reached 8 GiB but had the pre-fix ns print scale
bug (cycles were valid; printed ns/op was 1000x too small).

### 7.2 Nsight Compute on the actual RGI find path

Primary target was a new single-batch harness,
`engine/rgi_profile_one.cu`, to avoid `--launch-skip` through the full
sweep. It constructs the same 64M-key RGI chain table and launches the
same `kernels::batch_kernel<..., find_device_func<...>>` cooperative
find path. Sanity run at B=1M reproduced the banked ceiling:
**4,961.8 Mop/s** (211.33 us/batch) before profiling.

`ncu --set full`, one profiled find launch per B:

| B | NCU duration | Memory throughput | DRAM % peak | L2 hit | issue slots busy | active warps/sched | eligible warps/sched |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 72.93 us | 122 GB/s | 3.04% | 9.47% | 16.9% | 2.52 | 0.23 |
| 262,144 | 110.11 us | 639 GB/s | 15.88% | 6.10% | 55.6% | 8.49 | 1.28 |
| 1,048,576 | 293.63 us | 934 GB/s | 23.23% | 7.37% | 71.2% | 11.02 | 1.77 |

Additional B=1M details: L1/TEX hit rate 17.3%, L2 cache throughput
25.7% of peak, memory-pipes busy 62.6%, compute/SM throughput 69.5%,
and Nsight's rule text reports **5.0 cycles waiting on L1TEX
scoreboard dependencies**, about **32.2%** of the 15.4 cycles between
issued instructions.

Reading: the 1M find kernel is not close to streaming-HBM saturation
(23% DRAM peak), but it is already at **~67% of the measured random-128B
roof** if we estimate ~270 B/probe (4.96 Gop/s * 270 B ~= 1.34 TB/s vs
1.40 TB/s useful random roof). Nsight's lower 934 GB/s counter is still
consistent with a pointer-chasing, low-L2-hit workload because it counts
the profiled kernel's observed memory workload, not a full algorithmic
bytes/op model.

The operational knee is between B=262k and B=1M: issue-slot and memory
utilization keep rising through 1M, but the end-to-end launch sweep is
already near the 4.96 Gop/s ceiling at B=1M.

### 7.3 Persistent v2 timeline and the missing 1/3

`nsys profile` was run on `rgi_persist2 67108864` with a 180 s guard.
The binary exited cleanly; `nvidia-smi` showed no resident process
afterward. The importer was unavailable, but the run preserved the
same measured table as §4b: v2 pipelined reaches **3.29 Gop/s** at
B=1M, while launch reaches **4.96 Gop/s**.

At ~270 B/probe, v2 consumes an estimated **~0.89 TB/s**, about 63% of
the random-128B roof and 66% of the launch ceiling. The missing third
is therefore not explained by streaming bandwidth. The likely costs,
consistent with the v2 design and the Nsight results, are: mailbox and
dispatcher traffic, per-op lane-0 request staging plus shuffle, block
rotation/output-line misalignment, and completion bookkeeping. In other
words, v2 pays protocol work to buy sub-microsecond dispatch cadence.

### 7.4 Roofline / 90% answer

Figure: `fig17_roofline.png` (generated by `bench/make_plots9.py`).

The classic FP32 ridge point is **~17.6 FLOP/B** (67 TFLOP/s divided by
3.812 TB/s measured streaming read). RGI point lookup has essentially
0 FP arithmetic, so it is unreachable by construction and is the wrong
optimization target. For this workload, the right roof is the measured
random-access roof. Paths toward "90%+" are structural, not compiler
tuning: reduce bytes/op (WP1 small-key fast path removes the suffix
hop), increase memory-level parallelism, or make accesses larger/more
contiguous. A hash index cannot expect to look like the streaming
triad/read microbench.
