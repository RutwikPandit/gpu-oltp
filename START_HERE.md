# START HERE — GPU-OLTP Project: Single Source of Truth

**This file is the one document to read first.** It supersedes the older,
partly-inconsistent explanation docs (ledger in §14). If any other doc
disagrees with this one, this one is correct as of **2026-07-08**.
Raw measured data lives in `bench/gh200_campaign_results.md`; the forward
plan lives in `plan/`. Everything else is historical.

**Owner:** Rutwik Pandit (rpandit@andrew.cmu.edu), CMU MS ECE, ex-NVIDIA
GPU perf architect. **Advisors:** Andy Pavlo, Phil Gibbons.
**House style:** brutal honesty; every number tagged `[measured]` or
`[projected]`; negative results are results; never write "GPU beats CPU at
B=n" without "[measured end-to-end]" beside it.

---

## Table of contents
1. TL;DR (60 seconds)
2. The thesis and the cost model
3. Current headline result
4. Architecture — the SQL stack and the v1/v2 dispatch runtimes
5. Measured numbers (reference card)
6. Capabilities, correctness, and tests
7. Honest gaps
8. Roadmap (R1/R2/R3 and work packages)
9. Novelty and positioning
10. Repository map
11. Build and run
12. Machine, access, and dev-environment gotchas
13. Standing contracts (do not break)
14. Corrections & superseded-docs ledger
15. Design decisions (the why-ledger)
16. Using it: the SQL surface
17. Glossary (database ↔ GPU-architecture)

---

## 1. TL;DR (60 seconds)

A **PostgreSQL-integrated, GPU-resident transactional index**. A writeable
PostgreSQL 14 Foreign Data Wrapper maps a fixed `kv(k bigint, v bigint)`
foreign table onto a **RobustGPUIndexing (RGI)** `GPUChainHashtable`,
owned by a single shared background worker holding one CUDA context. Real
SQL `INSERT/SELECT/UPDATE/DELETE` executes against the GPU-resident index,
with atomic validate-then-apply commit, PK enforcement, read-your-writes,
and equality-predicate pushdown; differentially verified against a stock
Postgres heap table.

It is an **index / access-method prototype, not a full DBMS** (no MVCC/OCC,
no WAL, 32-bit values). Built on RGI, the warp-cooperative GPU index library
from Hyoungjoo Kim (Andy Pavlo's group). RGI's own VLDB submission §7 names
"persistent kernels + NVLink-C2C CPU→GPU query polling" as future work —
**this project is that layer**: the dispatch runtime + transactional
semantics between a real DBMS and RGI.

The campaign that defines the current state: on a **Lambda Cloud GH200**
we turned the previously *projected* NVLink-C2C dispatch numbers into
*measured* on-hardware results, and built the persistent-kernel dispatch
runtime the model assumed.

---

## 2. The thesis and the cost model

**GPU OLTP is a dispatch-cost problem.** A CPU point lookup is ~10–25 ns;
a GPU dispatch pays a fixed floor (kernel launch ≈ 4–25 µs on PCIe/Linux).
So the GPU only wins when many independent operations amortize one
dispatch. Coherent NVLink-C2C turns the CPU↔GPU dispatch into a
**sub-microsecond cache-line doorbell**, which moves the GPU-wins crossover
from tens-of-thousands of in-flight ops down to the low hundreds.

For a dispatch carrying `B` independent point ops:

```
T_gpu(B) = D + B / R_gpu           D     = fixed dispatch cost
T_cpu(B) = B / R_cpu               R_gpu = GPU index throughput
crossover  B* = D / (1/R_cpu - 1/R_gpu)
```

The whole point of the GH200 campaign was to replace the once-projected `D`
(and re-measure `R_gpu`, `R_cpu` on-target) with real numbers.

---

## 3. Current headline result

> On a Lambda Cloud GH200, the coherent doorbell measures **0.68 µs**
> (replicating and beating the published GH200 figure), and the persistent
> dispatch runtime built on top of it **beats a saturated 64-core Grace CPU
> from B ≈ 640 independent ops per dispatch** — for memory-resident working
> sets — measured end-to-end. It never wins for cache-resident sets.

Three qualifiers that are part of the claim, not footnotes:
- **B ≈ 640** is the pipelined crossover, statistically hardened (40/40
  repetitions across two runs, mean+2σ below the CPU line). **B = 512 is
  within ~1%, flips between runs, and is explicitly not claimed.**
- The win is for **memory-resident** working sets. For cache-resident sets
  (≲114 MB, fits Grace's L3) the CPU wins outright — the GPU offload is
  pointless there. This "two-regime rule" is stated wherever the claim is.
- The measured path is the **engine dispatch path** (CPU post → GPU
  dispatch → RGI probes → completion → CPU sees done), **not** full
  SQL-over-C2C. The SQL coalescer that would feed it is unbuilt (R1).

---

## 4. Architecture

### 4.1 The SQL stack

```
psql / any client
   │  SQL over foreign table kv(k bigint, v bigint)
   ▼
pg_rgi_fdw  (PostgreSQL 14 FDW)
   │  reads : pushdown of k='=' / k=ANY(const|$1 array), else full snapshot
   │  writes: buffered in a per-transaction HTAB (read-your-writes overlay)
   │  PRE_COMMIT: stage → validate → apply
   ▼  shared memory (two channels)
pg_gpu_service  (ONE background worker, one CUDA context, one shared index)
   │  - 2048-slot single-row ring (demo/test ops)
   │  - lock-guarded BULK region (all FDW ops), one bulk_lock serializes it
   ▼  C ABI  (rgi_oltp_engine.h — THE stability boundary)
rgi_oltp_engine.cu   (launch-per-flush production engine; 32-bit values)
   ▼
RGI gpu_chainhashtable<slab_allocator<128>, debra_reclaimer, tile=16>  in HBM
```

- **C ABI** (`rgi_oltp_engine.h`): `rgi_create(capacity,fill,pool)` /
  `destroy`; buffered `rgi_insert/update/delete`; `rgi_flush`;
  `rgi_flush_unique(*dup)`; `rgi_lookup`; `rgi_find_many`; `rgi_snapshot`;
  staging `rgi_stage_begin/del/upd/ins/commit(*dup)/abort`; paged
  `rgi_snapshot_begin/page`.
- **KV mapping:** 8-byte key = two uint32 RGI key-slices
  (`max_key_length = 2`); value = uint32 **row id** (values truncated to
  32 bits — this is why it's an index, not a tuple store);
  `RGI_INVALID = 0xFFFFFFFF` on absence.
- **Commit = validate-then-apply:** flush; validate ALL error conditions
  (intra-set dup + inserts-already-present) with **no mutation**; on
  conflict abort with index UNCHANGED; else apply (erase staged deletes,
  then one merged upsert of updates+inserts). Atomicity comes from
  validating before any mutation — **not** from "one launch."
- **Worker config:** `rgi_create(1<<22, fill=2.0, pool=0.20)`;
  `shared_preload_libraries='pg_rgi_fdw'`. Shared-mem constants:
  `GPU_SVC_BULK_CAP=262144`, `GPU_SVC_NSLOTS=2048`.

### 4.2 Persistent-kernel dispatch runtimes (research, not wired to SQL)

Both measure the doorbell/dispatch path. Both run their launch-mode sweep
*before* starting the resident kernel (a full-occupancy resident grid
blocks all other launches).

**v1 — `rgi_persist_engine.cu` (sm_89, PCIe prototype; INSERT+FIND only):**
two-level doorbell (block-0 polls a mapped `PersistCtl` line, republishes
to device `g_pub`; all blocks poll `g_pub`); grid-wide arrival barrier;
14.2 µs rendezvous — **not doorbell-bound** (payload staging + grid barrier
dominate). This finding motivated v2.

**v2 — `rgi_persist2.cu` (sm_90, GH200, the measured runtime):**

```
GRACE host                 NVLink-C2C          HOPPER resident kernel (1,320 CTAs)
                                                = 8 dispatchers + 1,312 serving
 post(b): pack 8-byte word    │      ┌─────────────┐   fan out to private
 {count|offset|gen|uniq}      │      │ 8 DISPATCHER│   per-block mailboxes
        │  store to           │      │ CTAs poll   │   (128 threads, one
        ▼  g_ring[b % 4096]───┼─────▶│ ring shard  │───atomic ordinal/blk)
 ┌──────────┐ (1 posted store,│      │ (b-1)%8     │        │
 │ g_ring   │  no CUDA call)  │      └─────────────┘        ▼
 │ HBM      │                 │                     ┌────────────────────┐
 └──────────┘                 │                     │ 1,312 SERVING CTAs │
                              │                     │ leader polls ONLY  │
 wait(b): 1 local read of     │                     │ its own mailbox    │
 done[gen] tag byte           │                     │ 8 tiles × 16 lanes │
        ▲                     │                     │ RGI cooperative_find│
 ┌──────────┐  1 tag byte     │   last arriver of   └────────────────────┘
 │ done[]   │◀────────────────┼─── generation posts        results → HBM
 │ (mapped  │                 │    done[gen] byte
 │  host)   │                 │
 └──────────┘
```

Key properties (all chosen by measuring the alternative that lost):
- **Consumer-side placement:** every polled location lives in its poller's
  memory — the ring in HBM (GPU polls it), the done line in mapped host
  memory (CPU polls it). Only the *writes* cross C2C. **Per batch, C2C is
  crossed exactly twice: one 8-byte descriptor out, one done byte back.**
- **Ring sharding:** one HBM ring of 4096 slots; dispatcher `d` owns
  batches where `(b-1)%8==d`, each with a private register cursor — no
  shared head/tail, no work-stealing. Independent readers by construction.
- **Block rotation:** batch `b` uses serving blocks starting at
  `(b·nact)%sgrid`, `nact=ceil(B/8)`, so consecutive pipelined batches run
  on disjoint block sets concurrently; strided tiles give ≤1 op/tile at
  small B (fixes v1's 16-serial-ops plateau).
- **Completion:** per-generation padded arrival counter; the last arriver
  (`==nact`) resets it, `__threadfence_system()`, writes one done byte.
- **Relaxed atomics + one explicit fence per boundary** (not acq_rel per
  op): the queues are single-producer/single-consumer, so the atomic only
  carries a "go" signal; payload ordering is a single `__threadfence_system`.
  On coherent C2C, single-location visibility is automatic, so relaxed is
  safe and ~0.5 µs cheaper than acq_rel.
- Constants: `NDISP=8`, `HOST_WINDOW=64`, `WINDOW_K=128`, `NSLOT=32`,
  `RING_K=4096`, `TILE=16`, `BLOCK_SIZE=128`, `TILES_PER_BLOCK=8`. Submit
  thread pinned to **core 32** (core 0 caught kernel IRQs → bimodal times).
  Default workload N = 64M keys.

---

## 5. Measured numbers (reference card)

All from the **Lambda Cloud GH200 480GB** campaign, 2026-07-07, unless
noted. Full tables + provenance: `bench/gh200_campaign_results.md`.

| Measurement | Value |
|---|---|
| Doorbell, Fusco CAS protocol (relaxed + pinned, LPDDR) | **0.68–0.71 µs** — beats published 0.833 µs by 15–18% |
| Doorbell, fully-ordered CAS (acq_rel) | 1.11–1.28 µs |
| Doorbell, two-flag volatile+fence (mapped, engine's mode) | 1.74–1.94 µs |
| Launch floor (patched RGI, B=1) | 4.1 µs |
| H100 RGI find ceiling (B=1M, 64M-key table) | **4,964 Mop/s** (~3.9× the RTX 4060's 1.25 Gop/s) |
| Grace 64T, memory-resident (1.6 GB table) | **1,187 Mop/s** (0.842 ns/op) |
| Grace 64T, cache-resident (50 MB, fits 114 MB L3) | **6,000 Mop/s** — exceeds the GPU ceiling |
| **v2 crossover vs 64T Grace, pipelined** | **B ≈ 640** [measured end-to-end; 40/40 reps; B=512 not claimed] |
| v2 crossover vs 64T Grace, sync (1 in flight) | B = 65,536 |
| v2 sync rendezvous floor / pipelined cadence | 6.2 µs / 0.24 µs per batch |
| v2 pipelined peak (B=1M) | 3.3 Gop/s (~2/3 of launch ceiling) |
| Streaming read BW / random 64–128 B "useful" roof | 3.81 TB/s / **~1.40 TB/s** (the hash-probe roof) |
| Nsight, RGI find B=1M | 23% DRAM peak, 7.4% L2 hit, ~32% scoreboard stalls, ~188 B/op → **MLP/latency-bound at ~67% of random roof** |

**Banked finding — RGI upstream bug:** `launch_batch_kernel` calls
`cudaGetDeviceProperties` per launch (~875 µs on this driver), which
dominated launch measurements until patched. Patched remote-only; report
upstream to Hyoungjoo (diff kept as `upstream/rgi_launch_geometry_cache.patch`).

---

## 6. Capabilities, correctness, and tests

**Provided by the FDW/worker transaction layer:**
- Per-transaction write buffering (HTAB in `TopTransactionContext`), applied
  only at `PRE_COMMIT`; ROLLBACK discards for free.
- **Atomic commit** (validate-then-apply; multi-chunk >262k rows are
  all-or-nothing); PK violation → `ERRCODE_UNIQUE_VIOLATION`, index unchanged.
- **Read-your-writes** (reads overlay the txn buffer); no dirty reads.
- **PK/UNIQUE** enforced both intra-txn and at commit; delete-then-reinsert
  and value-only updates are upserts (no false PK error).
- **Key-swap/chain safety:** overlapping old/new key sets in one statement
  are rejected (`ERRCODE_FEATURE_NOT_SUPPORTED`) rather than corrupted;
  non-overlapping renames (k=k+100) succeed.
- **Paged snapshot** (no truncation at the 262k bulk cap).
- **Pushdown** (read-only, key-'=' only): plan-time `k=const`,
  `k=ANY(const array)`; runtime `k=ANY($1)` for `PARAM_EXTERN`;
  subplan/InitPlan arrays fall back to full snapshot.

**Test suite** (`run_tests.sh` is the merge gate): `correctness.sql`
(differential vs heap oracle, EXCEPT both directions = 0), `txn_test.sql`,
`atomic_test.sql` (300,001-row multi-chunk all-or-nothing), `keyupd_test.sql`,
`keyswap_test.sql`, `pk_test.sql`, `snap_page_test.sql`.

---

## 7. Honest gaps

- **No OCC / no write-write detection / not serializable** — ≈Read-Committed
  + read-your-writes; lost updates possible under concurrency.
- **No durability / WAL / recovery** — GPU HBM is volatile; state dies with
  the postmaster.
- **No subtransactions / SAVEPOINT.**
- **32-bit values (row id)** — index/access-method, not a tuple store; no
  MVCC versions.
- **Fixed `kv(bigint,bigint)` schema**; no joins, secondary indexes, ranges,
  or aggregate pushdown; pushdown is key-'=' only.
- **Single worker + single `bulk_lock`** — all bulk ops serialize; scaling
  ceiling unmeasured.
- **v2 has no erase branch**; only the production launch path erases.
- **v2 is engine-level dispatch, not full SQL-over-C2C.**
- Open experiments: zipfian/skew through v2; full SQL stack on GH200; GH200
  atomic-commit performance curve.

---

## 8. Roadmap

Three results that matter; work packages in `plan/WP0..WP9`.

- **R1 — Measured multi-client SQL scaling** (throughput vs concurrent
  connections through real SQL). Needs **WP2** (coalescer that fuses many
  backends' ops into one dispatch; flips engine to `find<concurrent=true>`).
  **STATUS: OPEN — the top critical-path item.**
- **R2 — Defensible isolation** (OCC + write-write detection, killing the
  lost-update caveat). Needs **WP3** (per-key version validation + an ordered
  op-log buffer replacing the key-collapsed HTAB) + **WP4** (SAVEPOINT).
  **STATUS: OPEN.**
- **R3 — Hardware-validated dispatch model.** **STATUS: substantially DONE**
  by the GH200 campaign (doorbell measured, crossover B≈640, v2 built). The
  `plan/` files predate the campaign and read as if R3 is open — it isn't.

Other WPs: **WP0** git/test hygiene (repo is not yet under git); **WP1**
small-key fast path (keys <2³² as one slice → removes a 128 B dependent
load, ~2× space, roughly doubles the bandwidth-implied ceiling); **WP6** GPU
enumeration scan + aggregate pushdown; **WP8** row store / TAM / durability
(fall, design-first); **WP9** paper/artifact freeze (figs done).

---

## 9. Novelty and positioning

**Defensible claim (scoped):** *"first writeable PostgreSQL integration where
SQL DML executes against a GPU-resident warp-cooperative index, plus the
first implemented-and-measured OLTP dispatch runtime on cache-coherent
CPU-GPU hardware."* NOT "first GPU OLTP" (GPUTx/Gacco/Epic/LTPG are prior,
standalone) and NOT "first GPU index" (that's RGI).

- **vs PG-Strom:** PG-Strom accelerates OLAP and by its own docs **never**
  runs INSERT/UPDATE/DELETE on the GPU (the GPU is a read-mostly replica fed
  by a CPU redo-log applier). Our delta: **writes-on-GPU + persistent kernel
  + coherent C2C dispatch + warp-cooperative index.**
- **vs RGI:** RGI is the index; it has no transaction manager, commit/abort,
  CPU↔GPU queue, SQL, or persistence. This project builds exactly the layer
  RGI's §7 names as future work.
- **Two-regime honesty:** memory-resident high-throughput serving is the
  win; cache-resident and single-row-latency are explicitly the CPU's.

---

## 10. Repository map

```
gpu_oltp/
├── START_HERE.md                 ← this file (single source of truth)
├── engine/
│   ├── rgi_oltp_engine.{h,cu}    PRODUCTION engine: C ABI, staged commit
│   ├── rgi_persist2.cu           v2 dispatch runtime (GH200, measured)
│   ├── rgi_persist_engine.cu     v1 dispatch runtime (superseded by v2)
│   ├── doorbell.cu               doorbell microbench (mapped + Fusco CAS modes)
│   ├── membench.cu               streaming/random/pointer-chase BW bench
│   ├── rgi_sweep.cu              launch-mode batch sweep (floor/ceiling)
│   ├── rgi_profile_one.cu        single-launch Nsight harness for RGI find
│   ├── cpu_sweep.cpp             Grace/x86 CPU baseline (OpenMP, 1–64T)
│   └── gpu_oltp_engine.{h,cu}    older toy persistent engine (scaffold; OLAP scan)
├── pg_rgi_fdw/                   FDW + shared worker + SQL test suite  ← current
├── pg_gpu_fdw/                   superseded toy per-session FDW (OLAP demo only)
├── bench/
│   ├── gh200_campaign_results.md AUTHORITATIVE measured results
│   ├── make_plots8.py/9.py       GH200 figure generators (fig13–19)
│   ├── fig*.png                  result figures
│   └── prof_raw/ , prof_raw_box/ Nsight extracts (binary reports: gitignore)
├── plan/                         master plan + WP0–WP9 + GITHUB_CLEANUP_PLAN
├── archive/                      superseded docs (old summaries, handoffs,
│                                 the comp_arch explainer set, the chat log) — historical
├── demo/                         live psql demo scripts
├── SLIDES_GH200_UPDATE.md/pdf    the shareable deck (GH200 update talk) — in repo
├── internal/  (gitignored)       LOCAL ONLY: ARCHITECTURE_DEEP_DIVE.md
│                                 (in-depth arch talk) + SLIDES_1HR.* (full talk)
├── build_all.sh / build_*.sh     build scripts (paths need relativizing)
├── run_tests.sh                  the correctness gate
└── upstream/rgi_launch_geometry_cache.patch   the RGI bug fix (for Hyoungjoo)

../RobustGPUIndexing/             RGI dependency (Hyoungjoo Kim, Apache-2.0).
                                  NOT part of this repo — clone separately.
```

**Code-reading order** (fastest path to understanding the system):
1. `rgi_oltp_engine.h` — the C ABI, i.e. the whole system boundary in ~90 lines.
2. `engine/rgi_oltp_engine.cu` — the production engine (buffering, staged commit).
3. `pg_rgi_fdw/pg_gpu_service.{h,c}` — the worker + shared-memory protocol.
4. `pg_rgi_fdw/pg_rgi_fdw.c` — the FDW callbacks, pushdown, txn buffer.
5. `engine/rgi_persist2.cu` — the v2 dispatch runtime (the GH200 result).
6. `pg_rgi_fdw/*_test.sql` — what correctness actually means here.
7. `engine/gpu_oltp_engine.cu` — the toy engine, last (scaffold + OLAP scan).

---

## 11. Build and run

Laptop (dev): Windows + WSL2 Ubuntu 22.04, CUDA 12.3, RTX 4060 (sm_89),
PostgreSQL 14. GH200: aarch64, CUDA 12.8, H100 (sm_90).

```bash
# engine + FDW (laptop, from repo root inside WSL)
bash build_all.sh          # builds librgioltp.so + pg_rgi_fdw
bash run_tests.sh          # correctness/txn/atomic/keyupd/keyswap/pk/snap_page

# standalone benchmarks
bash build_bench.sh        # rgi_sweep, cpu_sweep, engine bench, ncu targets
bash build_persist.sh      # persistent engine + regenerates figures

# GH200 build flags (aarch64 + Hopper)
nvcc -std=c++17 -arch=sm_90 --expt-extended-lambda --expt-relaxed-constexpr \
     -maxrregcount=64 -I <RGI>/include <src>.cu -o <bin>
# CPU baseline: g++ -O3 -mcpu=native -fopenmp ; run OMP_PROC_BIND=close OMP_PLACES=cores
```

RGI include path is required for every engine build — point `-I` at a
separate RGI clone (do not vendor it; see §13).

---

## 12. Machine, access, and dev-environment gotchas

- **GH200 was Lambda Cloud** (NOT Vultr — Vultr was an abandoned early plan;
  `VULTR_API.txt` is a leftover credential to delete/revoke). GH200 480GB:
  H100 96 GB HBM3 sm_90 132 SMs + Grace 64× Neoverse-V2 (aarch64), 452 GB
  LPDDR5X, C2C enabled, driver 570.148.08 / CUDA 12.8, 64 KB pages. SSH key
  `rutwik_ed25519`. **The rented instance is being/was released — assume it
  is gone; all results are archived locally.**
- **WSL2/Windows quoting is treacherous.** Multi-line commands through
  Git-Bash → `wsl.exe` mangle; write scripts to files and
  `sed -i 's/\r$//'` them. Use `MSYS_NO_PATHCONV=1` for leading-slash args.
- **Persistent kernels pin SMs at 100%** — never leave one resident on a
  shared box; all binaries take a stop sentinel + `_exit`.
- Build scripts currently hardcode `/mnt/c/Users/rutwi/OneDrive/...` — must
  be relativized before another machine can build (cleanup Tier 5).

---

## 13. Standing contracts (do not break)

1. **The C ABI (`rgi_oltp_engine.h`) is the stability boundary** — extend,
   never break; FDW/worker build against it.
2. **RGI source is never modified in this repo.** Wrapper-level fixes first;
   the launch-bug fix stays a *patch* (`upstream/`), not a vendored copy.
   RGI is unpublished (under submission) — **coordinate with Hyoungjoo
   before publishing anything RGI-derived.**
3. **Serialized-channel invariants** (commit atomicity, snapshot consistency)
   hold only while the single `bulk_lock` serializes bulk ops — until WP2.
4. **`find<concurrent=false>`** is valid only under that serialization; WP2
   must flip it to `true` when reads overlap mutations.
5. **`[measured]` / `[projected]` on every number**, with machine + date.
   Never delete a superseded model line — relabel it.
6. **`run_tests.sh` must pass** before any work package is called done.

---

## 14. Corrections & superseded-docs ledger

**This file supersedes** the docs now moved to **`archive/`** (kept for
history; do not trust their numbers over this file): `PROJECT_SUMMARY.md`
(entirely pre-campaign — projected C2C, no GH200 pointer), `HANDOFF_GH200.md`
+ `HANDOFF_PERF_NSIGHT.md` + `GH200_RETRIEVAL.md` (ops handoffs, folded into
§11–12), `PRESENTATION.md` (superseded by the decks), `Summary.txt`,
`PRESENTATION_PLAN_GH200_UPDATE.md`, the whole `comp_arch_db_explainer/`
set (deep but internally inconsistent — its design-decisions, SQL-usage,
code-reading-order, and glossary content is folded into §15–17 + §10 here),
and the raw Cursor chat log. Their content is preserved; this file is the
active version.

**Still authoritative (not superseded):** `bench/gh200_campaign_results.md`
(raw measured data), `plan/` (forward plan + `GITHUB_CLEANUP_PLAN.md`), the
two `SLIDES_*` decks, and the source code.

**Known stale claims to fix (from the cleanup ledger):**
- **Vultr → Lambda:** 8 files still say "Vultr" and are wrong
  (`bench/gh200_campaign_results.md`, `make_plots8.py`, `make_plots9.py`,
  `HANDOFF_GH200.md`, `GH200_RETRIEVAL.md`,
  `bench/prof_raw_box/RETRIEVAL_MANIFEST.md`, `SLIDES_1HR.md`, and the
  cleanup plan). `SLIDES_GH200_UPDATE.md`'s "Lambda" is correct.
- **B = 512 vs 640:** the crossover claim is **B ≈ 640**; B=512 is not
  claimed. `bench/gh200_campaign_results.md` §4b still bolds the B=512 row —
  de-bold.
- **`FULL_PROJECT_EXPLAINER.md`** shows the measured B≈640 only in §8; its
  TL;DR/§33/Q&A/appendix still say doorbell is projected 0.5 µs / crossover
  ≈194. Historical — trust this file instead.
- Checked-in binaries (`.so/.o`) predate current sources — rebuild.

---

## 15. Design decisions (the why-ledger)

Do not re-litigate these; each was chosen deliberately. Format:
**decision — why — rejected alternative — cost/debt.**

1. **Integrate into real PostgreSQL** (not a standalone engine) — real
   parser/planner/catalog/wire protocol, and the PG-Strom comparison only
   works inside PG. *Rejected:* a standalone GPU engine (the GPUTx/Gacco
   route). *Cost:* PostgreSQL's per-tuple executor overhead sits on the SQL
   path (why engine-level and SQL-level numbers are reported separately).
2. **FDW first** — the fastest path to a *writeable* SQL surface with zero
   core patches. *Rejected:* Table Access Method (the "proper" storage AM
   with MVCC/visibility hooks — this is the **fall target**), CustomScan
   (can't do writes — the exact reason PG-Strom never writes on GPU), core
   patch. *Cost:* planner treats the table as opaque → pushdown is manual;
   no MVCC hooks → isolation is scoped.
3. **RGI chained hashtable, tile=16** — best insert ceiling, simplest
   per-bucket locking, concurrent mixed ops + safe reclamation.
   *Rejected:* cuckoo (deep-relocation fallback is a TODO in RGI),
   Masstree (buys ranges we don't expose, heavier per op), the toy
   open-addressing table (no var-len keys, no reclamation). *Cost:* 32-bit
   value; every 2-slice key pays a suffix-node hop (WP1 removes it).
4. **One shared background worker owns the only CUDA context** — PostgreSQL
   forks a process per connection, so a per-backend engine means N contexts
   / N copies / no sharing; one owner gives cross-session shared data +
   persistence (PG-Strom's "GPU Service" reached the same design).
   *Cost:* one worker + one `bulk_lock` serializes all bulk ops.
5. **Buffer writes, apply at commit (validate-then-apply)** — atomicity by
   validating every error condition before any mutation; batching for
   throughput; read-your-writes via an overlay. *Rejected:* per-row
   immediate apply (the toy FDW does this). *Cost:* the key-collapsed
   buffer can't represent overlapping renames in one statement (rejected
   with a clear error rather than corrupting).
6. **Persistent kernel is the architecture; launch-per-batch is the impl on
   PCIe** — WSL2/WDDM makes a resident kernel fragile, and the launch floor
   is the honest PCIe baseline. On GH200, v2 *is* the persistent runtime.
7. **v2: cross C2C once per batch, schedule on the GPU** — consumer-side
   placement (each polled line in its poller's memory); GPU dispatchers fan
   out; private per-block mailboxes. *Rejected (all measured losers):*
   mapped payload (0.9 µs C2C per key deref), one global polled word (serial
   re-reads), host-written mailboxes (host becomes the bottleneck).
8. **v2: relaxed atomics + one explicit fence per boundary** (not acq_rel per
   op) — the queues are single-producer/single-consumer, so the atomic only
   carries a "go" signal and coherence gives single-location visibility for
   free; ~0.5 µs cheaper than acq_rel on Grace.
9. **Keep the toy engine + `pg_gpu_fdw`** — origin of the doorbell protocol
   and the only OLAP bandwidth-scan demo. *Cost:* two engine ABIs coexist;
   both are clearly labeled superseded/scaffold.

---

## 16. Using it: the SQL surface

```sql
-- one-time (worker must be preloaded: shared_preload_libraries='pg_rgi_fdw')
CREATE EXTENSION pg_rgi_fdw;
CREATE SERVER rgi FOREIGN DATA WRAPPER pg_rgi_fdw;
CREATE FOREIGN TABLE kv (k bigint, v bigint) SERVER rgi;

INSERT INTO kv VALUES (1,100),(2,200);        -- buffered; applied atomically at COMMIT
SELECT v FROM kv WHERE k = 1;                 -- pushed down: one GPU find
SELECT * FROM kv WHERE k IN (1,2,3);          -- pushed down: batched find
PREPARE q(bigint[]) AS SELECT count(*) FROM kv WHERE k = ANY($1);  -- pushed ($1 = PARAM_EXTERN)
UPDATE kv SET v = 999 WHERE k = 1;            -- read-your-writes inside the txn
DELETE FROM kv WHERE k = 2;
BEGIN; INSERT INTO kv VALUES (9,9); ROLLBACK; -- discarded; GPU index untouched
```

- **Supported / pushed to GPU:** point + multi-key equality reads
  (`k=const`, `k IN (...)`, `k=ANY($1)`), bulk INSERT/UPDATE/DELETE, PK/UNIQUE
  enforcement, atomic commit, ROLLBACK, read-your-writes, paged full scan.
- **Not supported (falls back or errors):** ranges (`k>10`, `BETWEEN`),
  joins, aggregates on `kv` (except the toy `gpu_sum`/`gpu_count` demo
  funcs), `k=ANY(<subquery>)` (evaluated as a snapshot, not pushed),
  non-`bigint` types, values >32 bits, serializable isolation, SAVEPOINT, WAL.
- **Run:** `bash build_all.sh` then `bash run_tests.sh`. Data is **volatile**
  (lost on cluster restart) and **shared** across sessions via the worker.
- **Common gotcha:** if `CREATE EXTENSION` or queries fail, the worker isn't
  loaded — `shared_preload_libraries='pg_rgi_fdw'` requires a cluster restart.

---

## 17. Glossary (database ↔ GPU-architecture)

| Database term | What it is here / GPU-arch analogue |
|---|---|
| FDW (Foreign Data Wrapper) | PostgreSQL's plug-in storage interface; our entry point for routing SQL to the GPU |
| TAM (Table Access Method) | the deeper storage interface with MVCC hooks; the fall target, not used yet |
| Heap | PostgreSQL's normal row storage; we *replace* it with the GPU index for `kv` |
| Pushdown | moving a predicate (here `k=`) down to the storage layer so the GPU does the lookup instead of PostgreSQL scanning |
| Validate-then-apply | check all PK/UNIQUE errors before mutating; the source of commit atomicity |
| Read-your-writes | a txn sees its own uncommitted writes (via the buffer overlay); others don't |
| RGI | RobustGPUIndexing — the warp-cooperative GPU hash index (Hyoungjoo Kim); our storage substrate |
| Doorbell | the CPU→GPU "go" signal; on C2C a coherent cache-line write, not a driver call |
| Dispatch / launch floor | fixed per-batch cost to get work onto the GPU; the thing C2C shrinks |
| Persistent kernel | a resident GPU kernel that polls for work (vs. one launch per batch) |
| CTA / tile | CUDA thread block / a 16-lane sub-group that cooperatively does one index op |
| C2C (NVLink Chip-to-Chip) | Grace-Hopper's cache-coherent CPU↔GPU link; makes CPU and GPU act like two NUMA sockets |
| Memory- vs cache-resident | working set bigger than / smaller than the CPU's L3 — the two-regime boundary |
