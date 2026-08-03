# Primer Section 02 — The Toy Persistent-Kernel Engine Subsystem

Status of this document: written 2026-06-12 against the working tree at
`gpu_oltp/`. Every line number cited below was verified against the files as
read on that date. If the files have been edited since, re-verify line numbers
before acting on them; the *names* of functions and the *order* of statements
are the stable reference.

---

## 0. What this subsystem is, and — critically — what it is not

This subsystem is the **standalone, RGI-free prototype of the project's target
dispatch architecture**: a persistent CUDA kernel that is launched exactly once,
stays resident on the GPU for the lifetime of the process, and consumes request
batches from a zero-copy (mapped pinned-memory) ring via a doorbell handshake.
It consists of five files, ~745 lines total:

| File | Lines | Role |
|---|---|---|
| `engine/gpu_oltp_engine.h` | 95 | extern "C" host API; the ABI boundary of `libgpuoltp.so` |
| `engine/gpu_oltp_engine.cu` | 528 | the entire engine: doorbell protocol, hash table, three lock schemes, scan kernels, microbench |
| `diag.cu` | 71 | platform diagnostic: does zero-copy mapped-memory polling work on this WSL2/WDDM box at all? |
| `test_min.cu` | 18 | toolchain diagnostic: does `cuda_runtime.h` compile under nvcc 12.3 + MSVC 14.44 with zero STL includes? |
| `Makefile` | 33 | builds `libgpuoltp.so` (for the demo FDW) and `oltp_bench` (the microbench), plus a locking-study sweep target |

### What it is NOT

**This engine is not the SQL production path.** The production path is:
`pg_rgi_fdw` (Postgres FDW) → `pg_gpu_service` (background worker owning the
only CUDA context) → `librgioltp.so` (`engine/rgi_oltp_engine.cu`) → the RGI
chained hashtable, dispatched **launch-per-batch**, not via a persistent
kernel. That decision is recorded in the explainer (Part IV §29,
`comp_arch_db_explainer/FULL_PROJECT_EXPLAINER.md` lines 997–1008): on
WSL2/WDDM the persistent-kernel handshake *works* (this subsystem proves it)
but is the less stable and more overhead-prone binding — measured doorbell
rendezvous 75–90 µs vs. kernel launch 13–28 µs on this laptop — and it pins an
SM at 100% forever. So the production SQL path took the launch binding, and
this subsystem was retained as a separate artifact.

### What it IS, precisely (three roles)

1. **The architecture prototype for NVLink-C2C.** The Control-block doorbell
   protocol in `gpu_oltp_engine.cu` lines 42–49 and 172–219 is, byte for byte,
   the primitive that WP7 (the GB-class campaign,
   `plan/WP7_gbclass_campaign.md` item 1, lines 35–41) will port onto coherent
   C2C allocations. On PCIe the handshake rides `cudaHostAllocMapped` pinned
   memory; on GB-class the same five-field struct becomes a coherent cache
   line and the spin becomes a sub-microsecond ownership transfer
   (explainer Part XV §58, lines 1771–1773). Architecturally the protocol is
   an NVMe submission-queue/completion-queue pair: `batch_id` is the SQ tail
   doorbell, `done_id` is the CQ head.

2. **The vehicle for the single-op round-trip measurement.** The `BUILD_BENCH`
   main (`gpu_oltp_engine.cu` lines 519–523) times 2,000 back-to-back
   single-key lookups through the full CPU→GPU→CPU mapped-memory handshake.
   The recorded result is **75.5 µs per round trip on WSL2/WDDM (RTX 4060
   Laptop, measured 2026-06-08)** — the "PCIe baseline that C2C will cut", and
   the number that motivates the entire latency-budget model. WP5
   (`plan/WP5_persistent_v2_and_baremetal.md` lines 62–64) re-runs exactly
   this binary on bare-metal Linux, where low single-digit microseconds are
   expected because WDDM's doorbell path inflates the WSL2 number.

3. **Host of two reusable assets:**
   - the **three-lock-scheme hash table** (lock-free CAS / per-bucket spinlock
     / global spinlock, runtime-selectable, lines 94–168) built for the
     contention study — WP7 item 6 (lines 55–59) names it as "the
     controlled-experiment vehicle if RGI's results need explanation";
   - the **bandwidth-scan demo kernels** (lines 325–411): grid-stride
     block-reduction SUM/COUNT(<thr) over a GPU-resident 64-bit column,
     used by the demo FDW `pg_gpu_fdw` to back the OLAP supporting result
     (`gpu_scan_load` / `gpu_scan_agg_sum` / `gpu_scan_agg_count_lt` /
     `gpu_scan_kernel_ms` SQL functions; see `pg_gpu_fdw/pg_gpu_fdw.c`
     lines 45–48, 346–392).

### The three hard-won correctness fixes embedded in this code

The engine compiled cleanly on first build and then hung at 100% GPU
utilization. The hang decomposed into three independent bugs, each now fixed
*and documented in a comment at its fix site*. Future modifications that touch
these sites must preserve the fixes; each is dissected fence-by-fence in the
walkthrough below, but here is the index:

| Fix | Site | One-line mechanism |
|---|---|---|
| WDDM launch-queue flush | `gpu_oltp_engine.cu` 268–277 | `cudaEventRecord`+`cudaEventQuery` immediately after the persistent-kernel launch forces WDDM to submit the queued launch to the GPU; without it the kernel never starts and the host spins on `done_id` forever |
| Lost-doorbell race | `gpu_oltp_engine.cu` 187–196 and 214–216 | capture `c->batch_id` into a register **once, at batch start** (when the host is provably blocked) and use that single value for both `done_id` publication and the `seen` update; re-reading `batch_id` after signalling done races the host's next doorbell and silently drops a batch |
| Blocking-stream deadlock | `gpu_oltp_engine.cu` 259–263 | the persistent kernel must live on a `cudaStreamNonBlocking` stream; otherwise any later legacy-default-stream operation (e.g. the `cudaMemcpy` in `gpu_oltp_snapshot`) implicitly synchronizes with the never-terminating kernel and deadlocks |

Bug 2 is the one to internalize: it is **platform-independent**, it only
manifests under rapid back-to-back small submits (exactly OLTP's shape — single
large batches never hit it; 2,000 rapid single-op submits hit it reliably,
explainer §12 lines 500–510), and it would have shipped to GB-class unnoticed
if the single-op latency loop had not existed.

### Subsystem call graph

```
                       ┌──────────────────────────────────────────────┐
                       │  consumers                                   │
                       │                                              │
  oltp_bench           │  pg_gpu_fdw.c  (DEMO FDW — not pg_rgi_fdw)   │
  (BUILD_BENCH main,   │   - kv ops  -> gpu_oltp_{create,insert,      │
   .cu lines 475–527)  │                lookup,update,delete,snapshot}│
        │              │   - OLAP demo -> gpu_scan_{alloc,sum,        │
        │              │                count_lt,last_ms,free}        │
        │              └──────────────┬───────────────────────────────┘
        │   links statically          │ dlopens / links libgpuoltp.so
        ▼                             ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ engine/gpu_oltp_engine.cu        (API: engine/gpu_oltp_engine.h) │
  │                                                                  │
  │  host side                      device side                      │
  │  ─────────                      ───────────                      │
  │  gpu_oltp_create ──launch once─► persistent_kernel (1 block)     │
  │  gpu_oltp_submit ──doorbell───►   ├─ ht_lookup                   │
  │  gpu_oltp_set_scheme              ├─ do_write ─┬ ht_insert_lockfree
  │  gpu_oltp_snapshot (cudaMemcpy)   │            ├ ht_insert_bucketlock
  │  gpu_oltp_destroy (stop flag)     │            └ ht_insert_globallock
  │                                   └─ ht_delete                   │
  │  gpu_scan_alloc  ──ordinary────► scan_fill_kernel                │
  │  scan_reduce     ──launches────► scan_sum_kernel /               │
  │                                  scan_countlt_kernel             │
  └──────────────────────────────────────────────────────────────────┘
        ▲ calls only the CUDA runtime; zero RGI / zero Postgres includes

  diag.cu, test_min.cu: standalone binaries, no link relationship to the
  engine; they exist to validate the platform assumptions the engine makes.
```

Reading order for a new contributor: this overview → `gpu_oltp_engine.h`
(the contract) → `persistent_kernel` + `gpu_oltp_submit` (the handshake pair —
read them side by side) → the hash-table helpers → the scan module → the bench
→ `diag.cu` (the platform story) → `Makefile`.

---

## File: engine/gpu_oltp_engine.h (95 lines)

### Purpose

The single extern "C" header that defines the ABI of `libgpuoltp.so`. It is
deliberately C-compatible (`#ifdef __cplusplus extern "C"`, lines 18–20 and
91–93; only `<stdint.h>` included, line 16) because its primary consumer is
`pg_gpu_fdw.c`, plain C compiled by PGXS, which cannot see C++ or CUDA types.
Everything CUDA-flavored is hidden behind two opaque handles (`GpuOltpEngine`,
`GpuScanCol`).

### Position in the architecture

- Included by: `engine/gpu_oltp_engine.cu` (line 19 of the .cu — the
  implementation), `pg_gpu_fdw/pg_gpu_fdw.c` (its line 37), and the
  `BUILD_BENCH` main (same translation unit as the implementation).
- The `Makefile` lists it as a prerequisite of both build targets (lines 12,
  16), so editing it triggers rebuilds of the .so and the bench.
- It mirrors the design of the production wrapper header
  (`engine/rgi_oltp_engine.h`, documented elsewhere in this primer) — same
  opaque-handle + extern "C" pattern, different storage engine behind it.

### Design rationale

The file-top comment (lines 1–12) states the contract in three sentences worth
restating because they ARE the architecture: a single persistent kernel stays
resident and consumes request batches from a mapped zero-copy ring; the host
submits a batch and **blocks** until the kernel signals completion; on PCIe
the handshake rides mapped pinned memory and "on GB-class the same API can sit
on coherent C2C memory with a doorbell over C2C atomics" (lines 10–11). The
API is therefore synchronous by design — there is no async submit, no
completion callback, no multi-producer queue. That is a deliberate scoping
decision: the toy engine measures the *dispatch primitive*, not a scheduler.
(The multi-client coalescer is WP2, on the RGI path, not here.)

### Walkthrough

#### `enum gpu_oltp_op` (lines 22–28)

```c
enum gpu_oltp_op { GPU_OLTP_LOOKUP = 0, GPU_OLTP_INSERT = 1,
                   GPU_OLTP_UPDATE = 2, GPU_OLTP_DELETE = 3 };
```

The per-request opcode carried in the `types[]` array of a batch. The comment
on line 22 — "Numeric order matters: see store classification in the .cu" —
is **stale/forward-looking as of the current implementation**: the kernel's
dispatch (`gpu_oltp_engine.cu` lines 201–206) is an explicit `switch` on each
value, not a range test like `op >= GPU_OLTP_INSERT`. Treat the comment as a
warning that someone *may later* write such a range test (e.g., a "is this a
write?" predicate for a future read/write phase split); if you renumber these
constants, grep both the .cu and `pg_gpu_fdw.c` for range comparisons first.
Semantics to know: INSERT and UPDATE are **both upserts** at the device level
(both route to `do_write`, .cu lines 203–204); the distinction is preserved in
the API purely for benchmark labeling and FDW readability.

Pitfalls when modifying:
- An unknown opcode value is not an error: the kernel's `default:` arm
  (.cu line 206) returns `GPU_OLTP_NOTFOUND` for it. Adding a new op means
  adding a `case` in the kernel AND deciding its status semantics.
- The values cross the ABI: `pg_gpu_fdw.c` passes them as `uint32_t`.
  Renumbering requires rebuilding the FDW.

#### `enum gpu_oltp_scheme` (lines 30–35)

```c
enum gpu_oltp_scheme { GPU_OLTP_LOCKFREE = 0, GPU_OLTP_BUCKETLOCK = 1,
                       GPU_OLTP_GLOBALLOCK = 2 };
```

Selects the write-path concurrency protocol **for the whole engine**, batch
granularity (the kernel samples `c->scheme` once per batch, .cu line 196).
Exists solely for the contention study (Makefile `sweep` target runs all three
under uniform and zipfian-0.99 key distributions). `0` (lock-free) is the
default, set at engine creation (.cu line 256). Note the asymmetry documented
in the .cu walkthrough below: lookups are *always* lock-free and deletes are
*always* lock-free `atomicExch`, regardless of scheme — only inserts/updates
honor this enum. A scheme value outside 0–2 silently behaves as lock-free
(`default:` arm of `do_write`, .cu line 166).

#### `enum gpu_oltp_status` (lines 37–42)

```c
enum gpu_oltp_status { GPU_OLTP_OK = 0, GPU_OLTP_NOTFOUND = 1, GPU_OLTP_FULL = 2 };
```

Per-request result written into `out_status[]`. `OK` doubles as "found" for
lookups and "success" for writes. `FULL` is only producible by inserts/updates
and only after a probe of the **entire table** finds no `EMPTY`, no
`TOMBSTONE`, and no matching key (.cu lines 111, 123, 136) — i.e., it is a
hard capacity exhaustion signal, not a load-factor warning. There is no error
status: a malformed opcode maps to `NOTFOUND` (see above), and device-side
faults surface only as a CUDA error on a *later* host call. Callers (the FDW)
treat anything nonzero from a write as failure-worthy.

#### `typedef struct GpuOltpEngine GpuOltpEngine` (line 44)

Opaque handle. The full definition lives at .cu lines 61–72 and contains CUDA
types (`cudaStream_t`), which is exactly why it cannot appear here. One engine
== one persistent kernel == one hash table == one request ring. The API has no
notion of multiple tables; `pg_gpu_fdw` keeps a single process-global
`g_engine`.

#### `gpu_oltp_create` / `gpu_oltp_destroy` (lines 47–48)

```c
GpuOltpEngine *gpu_oltp_create(uint64_t capacity, int threads_per_block);
void           gpu_oltp_destroy(GpuOltpEngine *e);
```

`capacity` is rounded **up** to a power of two (line 46 comment; implemented
by `round_pow2`, .cu lines 222–224) because the table indexes with
`hash & mask`. `threads_per_block` sizes the single block of the persistent
kernel; `<= 0` defaults to 1024 (.cu line 228). Contract notes that matter to
callers:

- `gpu_oltp_create` is also the point where the persistent kernel launches and
  an SM becomes permanently busy. Creation is therefore the expensive,
  side-effectful call; the FDW creates lazily on first table touch
  (`pg_gpu_fdw.c` line 77).
- `gpu_oltp_destroy` is the ONLY way to retire the kernel (stop flag + wake +
  `cudaStreamSynchronize`, .cu lines 431–443). Killing the process without
  destroy is survivable (driver cleans up), but on a shared machine a leaked
  spinning kernel is hostile — WP5's campaign-script contract ("trap signals
  and set the stop sentinel on exit", `WP5_persistent_v2_and_baremetal.md`
  lines 105–107) exists because of this.
- Neither function is thread-safe; the whole API assumes a single host thread
  (see invariants summary).

#### `gpu_oltp_set_scheme` (line 51)

```c
void gpu_oltp_set_scheme(GpuOltpEngine *e, int scheme);
```

One volatile store into the mapped control block (.cu lines 281–283). Takes
effect at the **next batch** (kernel samples per batch). Not fenced and not
synchronized — calling it concurrently with an in-flight `gpu_oltp_submit`
from another thread is a data race; under the single-thread contract it is
trivially safe because submit blocks.

#### `gpu_oltp_submit` (lines 53–63)

```c
void gpu_oltp_submit(GpuOltpEngine *e,
                     const uint32_t *types, const uint64_t *keys,
                     const uint64_t *values, uint32_t *out_status,
                     uint64_t *out_values, uint32_t n);
```

The core entry point: submit `n` operations, block until all complete.
Contract details encoded in the signature and comment (lines 53–56):

- `types`, `keys` are mandatory inputs of length `n`. `values` may be NULL if
  the batch contains no inserts/updates (the implementation skips the staging
  memcpy when NULL, .cu line 294 — note it skips it for the WHOLE batch, so a
  mixed batch with any write MUST pass a full-length `values` array; positions
  for non-write ops are simply ignored).
- `out_status` and `out_values` may each independently be NULL; the result
  copy-back is skipped per-array (.cu lines 304–305). The kernel still writes
  the device-visible staging arrays unconditionally (well: guarded by
  pointer-truthiness of the *staging* pointers, which are always non-NULL —
  .cu lines 208–209 — so effectively unconditionally).
- `n` may exceed the internal ring capacity; submit transparently chunks into
  `MAX_BATCH`-sized doorbell rounds (see the chunking diagram in the .cu
  walkthrough). One API call may therefore be multiple kernel batches; there
  is no atomicity guarantee across the chunk boundary.
- Blocking is a **userspace spin** with no sleep and no timeout (.cu line
  301). If the kernel is wedged (e.g., bug 1's failure mode), submit hangs the
  calling thread forever. `diag.cu` exists because of this: it is the
  timeout-equipped version of the same handshake.

#### Single-op wrappers: `gpu_oltp_insert` / `gpu_oltp_lookup` / `gpu_oltp_update` / `gpu_oltp_delete` (lines 65–69)

Each is literally a batch of 1 through `gpu_oltp_submit` (.cu lines 310–323) —
which means **each single op pays one full doorbell round trip** (the 75.5 µs
on this box). This is intentional: these wrappers ARE the measurement
instrument for the dispatch floor, and they are what `pg_gpu_fdw` calls per
SQL row. Their return value is the `gpu_oltp_status` as an `int`. `lookup`
additionally writes the found value through `out_value` if non-NULL (and
leaves `*out_value` as the kernel's default `0` when not found — the kernel
zero-initializes `v` per request, .cu line 199).

#### `gpu_oltp_snapshot` (lines 71–76)

```c
uint64_t gpu_oltp_snapshot(GpuOltpEngine *e, uint64_t **out_keys, uint64_t **out_values);
```

Full-table scan support for `SELECT *`: copies the entire GPU-resident
keys/values arrays to host (two `cudaMemcpy`s of `capacity * 8` bytes each)
and compacts out EMPTY/TOMBSTONE slots in place. Returns the live count;
`*out_keys`/`*out_values` are `malloc`'d at **capacity** length (not trimmed
to the live count — over-allocation is deliberate simplicity) and the caller
must `free()` them. The line-75 comment is a real contract, not advice:
"call when no write batch is in flight." The memcpy reads device memory while
the persistent kernel could be mid-batch; there is no quiescing handshake, so
a concurrent write batch yields a torn snapshot (some slots pre-, some
post-write, and possibly a key visible with a stale value given the engine's
key-before-value publication order — see `ht_insert_lockfree` below). Under
the single-thread contract, "no in-flight batch" is automatic because submit
blocks. This function is also the reason fix 3 (non-blocking stream) exists:
its `cudaMemcpy` runs on the legacy default stream and would deadlock against
the resident kernel if the kernel's stream were blocking (.cu lines 259–262).

#### Scan API: `GpuScanCol`, `gpu_scan_alloc`, `gpu_scan_count`, `gpu_scan_sum`, `gpu_scan_count_lt`, `gpu_scan_last_ms`, `gpu_scan_free` (lines 78–89)

The OLAP demo surface. The block comment (lines 78–82) defines the modeled
scenario: "data already in HBM, CPU only issues the query" — a large column
lives in GPU memory, aggregates stream HBM on-GPU, and only a scalar crosses
PCIe. The comment's last sentence is load-bearing: "These are independent of
the persistent-kernel engine (no doorbell), so they use ordinary
launch+sync." `gpu_scan_alloc(n)` allocates and fills the column with
`v[i] = i+1` (so `sum` and `count_lt` have closed-form expected values —
that is the self-test); `gpu_scan_last_ms()` returns the CUDA-event-timed
duration of the most recent reduction kernel, which is what the SQL function
`gpu_scan_kernel_ms()` exposes so the demo can report kernel time separately
from end-to-end time. One latent hazard at the implementation level
(`cudaDeviceSynchronize` inside `gpu_scan_alloc` vs. a resident persistent
kernel) is dissected in the .cu walkthrough — it is the single place where the
"independent of the engine" claim is not quite airtight.

### Invariants summary (header-level contract)

1. **Single host thread.** No function in this API is thread-safe. Submit
   blocks; everything else assumes quiescence.
2. **One engine per process is the tested configuration.** Nothing prevents
   two engines structurally, but each pins a block on an SM forever and both
   would call `cudaSetDeviceFlags`; untested.
3. **Synchronous semantics.** When `gpu_oltp_submit` returns, all `n` results
   are in `out_status`/`out_values` and all table mutations are visible to
   subsequent batches and to `gpu_oltp_snapshot`.
4. **Intra-batch ordering is undefined.** Two ops on the same key in one batch
   execute on arbitrary threads in arbitrary order. Callers needing ordering
   (the FDW's update = delete+insert when the key changes) must split across
   batches — which the single-op wrappers do naturally.
5. **`values` must be full-length if any op in the batch writes** (the NULL
   skip is whole-batch).
6. **Snapshot requires no in-flight writes** (automatic under invariant 1).
7. **Capacity is a power of two ≥ 1 after rounding**; statuses and opcodes are
   stable ABI shared with `pg_gpu_fdw.c`.

### How to modify safely (header)

- Adding an op or status: append, never renumber (ABI with the FDW). Add the
  kernel `case` and define `default:` behavior consciously.
- Adding an async submit (a likely WP5/WP7 want, e.g., separate
  `submit_async`/`wait` so the host can overlap staging with GPU processing):
  the doorbell protocol already supports it — `done_id` is the completion
  signal — but the chunking loop and the result staging arrays are
  single-batch; an async API needs either double-buffered rings or a rule that
  only one batch may be outstanding. Read
  `plan/WP5_persistent_v2_and_baremetal.md` (the v2 binding on the RGI side
  already does two-level doorbells and is the better starting point) before
  growing this API; the toy header should stay the *minimal* C2C-portable
  surface that WP7 item 1 ports.
- Do not add C++ or CUDA types to this header; PGXS-compiled C consumers
  include it.

---

## File: engine/gpu_oltp_engine.cu (528 lines)

### Purpose

The entire engine in one translation unit, in five layers (file-top comment,
lines 1–18):

1. mapped control block + doorbell protocol (lines 41–49, 172–219, 285–308);
2. an open-addressing linear-probe hash table, GPU-resident, with three
   runtime-selectable write protocols (lines 51–168);
3. host API (lines 221–323, 413–443);
4. an independent bandwidth-scan/aggregate module (lines 325–411);
5. an `#ifdef BUILD_BENCH` microbenchmark main (lines 445–528).

The same file builds two artifacts (comment lines 11–13, mirrored by the
Makefile): `libgpuoltp.so` (no `BUILD_BENCH`) and `oltp_bench`
(`-DBUILD_BENCH`, gains `main`).

Line 15–17's NOTE is honest self-labeling: "scaffold... Single-block
persistent kernel (robust __syncthreads handshake); scale to a multi-block
cooperative-groups grid once correctness is confirmed." The multi-block
successor was in fact built on the RGI side
(`engine/rgi_persist_engine.cu`, 216 resident blocks, two-level doorbell —
WP5 "Current state", lines 19–23), so this file deliberately remains the
minimal, single-block reference implementation of the protocol.

### Position in the architecture

- Linked into `libgpuoltp.so`, consumed by `pg_gpu_fdw.c` (demo FDW only).
- Compiled standalone as `oltp_bench` for the locking sweep and the
  single-op round-trip measurement.
- Calls only the CUDA runtime and (host-side) libc/STL. No RGI headers, no
  Postgres headers. This isolation is the point: when WP7 ports the doorbell
  to GH200, this file plus the Makefile is the entire dependency closure.

### Design rationale

Three decisions shape everything:

- **Single block.** The persistent kernel is `<<<1, tpb>>>` (line 264). Within
  one block, `__syncthreads()` is a legal, cheap, fully-defined barrier, which
  makes the dispatch handshake (spin → barrier → broadcast → process →
  barrier → signal) trivially correct. A multi-block persistent grid needs
  cooperative launch or hand-rolled grid barriers (atomic arrival counters —
  what `rgi_persist_engine.cu` does); that complexity was deferred until the
  protocol itself was validated. Consequence: device-side throughput of this
  engine is bounded by one SM's worth of threads; its purpose is latency and
  protocol measurement, not peak Mops.
- **Zero-copy mapped staging, not cudaMemcpy.** Requests and results live in
  `cudaHostAllocMapped` pinned host memory that the GPU reads/writes directly
  over PCIe (lines 243–254). This removes every CUDA API call from the
  steady-state submit path — the host's per-batch work is memcpy into pinned
  staging + one volatile store; the GPU's is uncached PCIe reads. That is what
  makes the 75.5 µs number a measurement of the *interconnect + polling*
  path rather than of the CUDA driver, and what makes the protocol portable to
  C2C (where the same loads/stores become coherent cache traffic).
- **The table stores 8-byte keys/values in flat arrays with sentinel keys**
  (lines 27–28): `EMPTY_KEY = 0xFFFF...FF`, `TOMBSTONE_KEY = 0xFFFF...FE`.
  Flat open addressing (vs. RGI's chained 128-byte nodes) keeps the toy free
  of allocators and reclamation, at the cost of two reserved key values and
  tombstone accumulation. Anyone storing real data must guarantee keys never
  equal either sentinel (the FDW's `bigint` keys are sign-extended positive
  values, so they cannot).

### Walkthrough

#### `EMPTY_KEY`, `TOMBSTONE_KEY`, `MAX_BATCH` (lines 27–29)

```c
#define EMPTY_KEY     0xFFFFFFFFFFFFFFFFULL
#define TOMBSTONE_KEY 0xFFFFFFFFFFFFFFFEULL
#define MAX_BATCH     (1u << 20)   /* mapped ring capacity (requests/batch) */
```

- `EMPTY_KEY` is all-ones so the keys array can be initialized with a single
  `cudaMemset(..., 0xFF, ...)` (line 239) — memset writes bytes, and the only
  64-bit patterns reachable by byte-memset are repeats; all-ones is the
  classic choice. If you change `EMPTY_KEY`, line 239 silently breaks (memset
  cannot produce arbitrary 64-bit patterns); you would need a fill kernel.
- `TOMBSTONE_KEY` differs in the low byte only. Both sentinels are illegal
  user keys; nothing validates this at the API boundary — a user insert of key
  `0xFFFF...FE` would corrupt the state machine (it would look like a deleted
  slot to every probe). The FDW never produces them; the bench uses keys
  `1..nkeys`. If this engine ever takes adversarial input, add a host-side
  reject.
- `MAX_BATCH` = 1,048,576 requests sizes the five staging arrays:
  types 4 MiB + keys 8 MiB + vals 8 MiB + stat 4 MiB + oval 8 MiB =
  **32 MiB of pinned, mapped host memory per engine** (lines 250–254), plus
  the 64-byte control block. Pinned memory is a scarce, page-locked resource;
  raising `MAX_BATCH` multiplies this directly and large pinned allocations
  can fail or degrade the host. Lowering it only adds chunking rounds (submit
  already chunks). The `1u << 20` is `unsigned`, and every comparison against
  it is on `uint32_t` — safe, but if you push past `1u << 31` the arithmetic
  in submit (`chunk * sizeof(uint64_t)`, line 293) needs auditing for 32-bit
  overflow on the byte count (currently fine: `size_t` promotion happens at
  the multiply because `sizeof` is `size_t`).

#### `CUDA_CHECK` (lines 31–39)

Standard check-and-abort macro: prints `file:line` + `cudaGetErrorString` to
stderr and calls `abort()`. Two consequences worth knowing:

- Inside a Postgres backend (the FDW path), `abort()` takes down the backend
  with a core, and Postgres restarts in crash-recovery. That is the chosen
  failure mode for the *demo* FDW; the production service worker has graceful
  error paths instead. Do not "improve" this to return codes piecemeal — half
  the call sites have no error plumbing; if you need graceful errors, do it
  wholesale.
- One call site deliberately avoids the macro: `cudaEventQuery` at line 276,
  because `cudaErrorNotReady` is the *expected* result there (see fix 1
  below). When auditing "why isn't this wrapped", that comment (line 276) is
  the answer.

#### `struct Control` (lines 42–49) — the doorbell

```c
struct Control {
    volatile uint32_t batch_id;   /* host bumps to publish a new batch     */
    volatile uint32_t done_id;    /* kernel sets to batch_id when finished */
    volatile uint32_t count;      /* #requests in the current batch        */
    volatile uint32_t scheme;     /* gpu_oltp_scheme for write ops         */
    volatile uint32_t stop;       /* host sets to 1 to retire the kernel   */
    uint32_t _pad[11];            /* keep fields off one cache line each   */
};
```

This 64-byte struct IS the C2C primitive. Field roles:

- `batch_id` — monotonically increasing publication counter, written only by
  the host (lines 299, 435), read by the kernel (lines 182, 194). It is the
  submission-queue doorbell.
- `done_id` — completion counter, written only by the kernel (line 214), read
  only by the host (line 301). The protocol invariant is
  `done_id == batch_id` ⇔ idle; `done_id == batch_id - 1` ⇔ batch in flight.
  Single-writer per field is what keeps this protocol correct without any
  atomics: each 32-bit field has exactly one producer and one consumer.
- `count`, `scheme` — batch parameters, host-written before the doorbell,
  kernel-read after observing the doorbell (lines 195–196). They are
  protected by the release/acquire pairing described under
  `gpu_oltp_submit` and `persistent_kernel`.
- `stop` — one-shot shutdown flag (host line 433, kernel lines 182, 185).
- `volatile` on every field is doing two jobs: (host) it forces the compiler
  to emit each load/store and forbids hoisting the spin-loop load out of the
  loop; (device) it forces loads/stores to bypass per-thread register caching
  and compile to memory operations each iteration. `volatile` provides **no
  ordering** by itself on either side — ordering comes from the explicit
  fences (host `std::atomic_thread_fence`, device `__threadfence_system`) and,
  on x86, from TSO store ordering; the walkthroughs of submit and the kernel
  itemize this.
- `_pad[11]` rounds the struct to exactly 64 bytes (5×4 + 11×4). The line-48
  comment says "keep fields off one cache line each", which describes the
  intent imprecisely — the fields all share ONE 64-byte line; the padding's
  actual effect is to round the struct to a full cache line so the doorbell
  does not false-share with whatever the allocator places after it. On this
  PCIe path that barely matters (mapped accesses are uncached on the device).
  On **C2C it matters a lot**: host spinning on `done_id` and device writing
  `batch_id`-adjacent fields would ping-pong the single shared line. WP7's
  queue-placement experiments (items 1 and 3) will likely want `batch_id` and
  `done_id` on **separate** lines (each in its consumer's memory); if you make
  that change, this struct is the place, and the comment should be corrected
  at the same time. Do not reorder fields casually: nothing serializes this
  struct, but `gpu_oltp_set_scheme` and the FDW assume the struct layout via
  the same header recompile, so layout changes are safe only with a full
  rebuild of every consumer.

#### `struct Table` (lines 52–59)

```c
struct Table {
    uint64_t *keys;     /* device, length capacity, init EMPTY_KEY */
    uint64_t *values;   /* device, length capacity                 */
    int      *locks;    /* device, length capacity (bucket-lock)   */
    int      *glock;    /* device, length 1 (global-lock)          */
    uint64_t  capacity; /* power of two                            */
    uint64_t  mask;     /* capacity - 1                            */
};
```

Plain-old-data view of the device-resident table, passed to the kernel **by
value** (line 172: `Table t` parameter) — the pointers are device pointers,
the struct itself is copied into kernel parameters at launch. Consequence:
the kernel's view of `capacity`/`mask`/pointers is frozen at launch time;
the table **cannot be resized or rehashed while the persistent kernel is
resident**. Any resize design must stop/relaunch the kernel (cheap — see
`gpu_oltp_destroy`) or add an indirection through device memory. Memory cost:
`capacity * (8 + 8 + 4)` bytes + 4 (`glock`); the bench default
`cap = 2^24` costs 320 MiB of device memory. `locks` is allocated at full
`capacity` length even when the scheme is lock-free — 64 MiB of dead weight at
the default; acceptable for a study vehicle, and freeing it conditionally
would complicate runtime scheme switching.

#### `struct GpuOltpEngine` (lines 61–72)

The host-side handle: the `Table`, six mapped host/device pointer pairs
(`h_*`/`d_*` for the control block and the five staging arrays), the dedicated
non-blocking `stream`, `cur_batch` (the host's private mirror of `batch_id`,
incremented at line 298 so the host never needs to read back its own volatile
write), and `tpb`. The `h_*`/`d_*` pairing exists because, pre-UVA, mapped
memory had distinct host and device addresses; `cudaHostGetDevicePointer`
(line 247) returns the device alias. On every 64-bit platform with unified
virtual addressing the two are numerically equal, but keeping both is correct,
free, and portability-proof. `cur_batch` matters for the lost-doorbell story:
the HOST always knows exactly which batch it is waiting for (`cur_batch`,
line 301); the bug was on the KERNEL side, which had to be taught the same
discipline (capture once, line 194).

#### `hash64` (lines 75–79)

```c
__device__ __forceinline__ uint64_t hash64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}
```

This is the MurmurHash3 64-bit finalizer (fmix64) — a full-avalanche bijective
mixer. Why this one: it is 5 ALU ops, branch-free, and bijective (no two keys
collide in the hash itself; collisions come only from the `& mask`
truncation). Bijectivity matters for the bench's dense keys `1..nkeys`: a weak
hash on dense keys produces clustered probes and would corrupt the contention
study's comparability across schemes. If you change this function, change it
in lockstep with nothing — it is self-contained — but note any host-side
re-implementation (e.g., a future host-side partitioner) must match
bit-for-bit or every probe sequence changes.

#### `ht_lookup` (lines 82–91)

```c
__device__ uint32_t ht_lookup(const Table &t, uint64_t key, uint64_t *out)
```

Linear probe from `hash64(key) & mask`, up to `capacity` steps (line 84 —
the loop bound guarantees termination even on a pathological table with zero
EMPTY slots):

1. Read the slot key through a `volatile` cast (line 86) — forces a fresh
   load so a concurrent same-batch insert's CAS result can be observed rather
   than a register-cached stale value.
2. `cur == key` → read the value (also through volatile, line 87) and return
   `OK`.
3. `cur == EMPTY_KEY` → the probe chain ends; return `NOTFOUND` (line 88).
   TOMBSTONE does not end the chain — deleted slots are skipped implicitly by
   matching neither condition.

Memory-ordering reality check (important for modifiers): the volatile loads
give freshness, not ordering. The known race: `ht_insert_lockfree` publishes
the KEY first (CAS, line 101) and writes the VALUE after (line 105), so a
concurrent lookup in the *same batch* can match the key at line 87 and read a
stale/zero value. This cannot happen *across* batches (the batch-end
`__threadfence_system` at line 212 plus the host handshake order all writes
before the next batch starts). It is acceptable inside a batch because the
API declares intra-batch ordering undefined (header invariant 4). If you ever
need same-batch read-your-writes, the fix is to invert publication in the
insert (write value, `__threadfence()`, then CAS the key) — see the
`ht_insert_lockfree` pitfalls.

Worst-case cost note: after heavy deletes, chains contain long tombstone runs
and a miss on a chain with no EMPTY scans all `capacity` slots. The toy has no
tombstone compaction; the RGI path (DEBRA) is where real reclamation lives.

#### `ht_insert_lockfree` (lines 94–112) — the default write path

```c
__device__ uint32_t ht_insert_lockfree(const Table &t, uint64_t key, uint64_t val)
```

Step by step:

1. Probe from home (lines 95–97), reading the slot key via volatile (line 98).
2. **Key already present** (line 99): overwrite the value in place, then
   `__threadfence()`, return OK. The fence here is device-scope: it orders the
   `values[slot]` store before this thread's subsequent stores (its `stat[i]`
   status write back in the kernel loop, line 208), so any device observer
   that sees the status cannot miss the value. Host-visibility ordering does
   NOT rely on this fence — that is the batch-end `__threadfence_system`'s
   job (line 212).
3. **Slot free** (`EMPTY` or `TOMBSTONE`, line 100): try to claim it with
   `atomicCAS(keys[slot], cur, key)` (lines 101–103). Three outcomes:
   - `prev == cur`: we won the slot. Write the value, `__threadfence()`,
     return OK (lines 104–105).
   - `prev == key`: another thread inserted the SAME key into this slot
     between our read and our CAS. Treat as success and overwrite the value
     (last-writer-wins among same-key racers) — also lines 104–105.
   - otherwise: another thread claimed the slot with a DIFFERENT key. `--i`
     (line 108) re-evaluates **the same slot** on the next loop iteration —
     the re-read at line 98 will now see the other key and the probe moves on
     (or sees our key if a same-key racer landed there, hitting the line-99
     overwrite path).
4. Probed all `capacity` slots without success → `GPU_OLTP_FULL` (line 111).

Why this shape: CAS-claim on the key word is the minimal lock-free
open-addressing insert; the `--i` retry instead of `continue`-to-next-slot is
required for correctness (skipping the contested slot after losing a race to
a *same-key* writer would insert a duplicate key further down the chain).

Invariants this function maintains:
- A slot's key field transitions only EMPTY→key, TOMBSTONE→key (both via
  CAS here) — never spontaneously back (see the slot state machine diagram
  below).
- Duplicate keys never coexist: every probe checks `cur == key` *before*
  trying to claim a free slot, and the lost-CAS retry re-checks.

Pitfalls when modifying:
- **The key-before-value publication order** (CAS line 101 precedes value
  store line 105) is the engine's one real internal race, documented under
  `ht_lookup`. The safe inversion — stage the value, fence, then CAS — only
  works for the EMPTY/TOMBSTONE claim path (where the slot's value is not yet
  observable); for the overwrite path (line 99) torn key/value pairs are
  inherent to in-place 8-byte updates without versioning. Across batches none
  of this is observable; document any new intra-batch guarantee carefully
  before promising it.
- `atomicCAS` on `unsigned long long` requires the keys array be naturally
  aligned (cudaMalloc guarantees 256-byte alignment of the base; index
  arithmetic preserves 8-byte alignment).
- The `--i` with `uint64_t i` is safe at `i == 0` only because the loop
  increments before the next compare (`--i` makes it `0xFFFF...F`, then `++i`
  wraps to 0) — wraparound round-trip is well-defined for unsigned. Replacing
  `uint64_t` with a signed or narrower type changes this; don't.

#### `lock_acquire` / `lock_release` (lines 117–118)

```c
__device__ __forceinline__ void lock_acquire(int *l) { while (atomicCAS(l, 0, 1) != 0) { __nanosleep(64); } }
__device__ __forceinline__ void lock_release(int *l) { __threadfence(); atomicExch(l, 0); }
```

Test-and-set spinlock with `__nanosleep` backoff. The comment above (lines
114–116) names the SIMT hazard explicitly: when threads of one warp contend
for one lock, the spinning losers and the winning holder are warp-mates; on
pre-Volta SIMT the holder could be starved by reconvergence rules
(classic intra-warp spinlock deadlock). On sm_89 (Ada), Independent Thread
Scheduling makes this *live* but still slow — the holder makes progress but
the warp thrashes. `__nanosleep(64)` (Volta+; requires `-arch >= sm_70`)
yields the spinning thread so the scheduler can advance the holder. The
comment's verdict stands: "Acceptable for the contention *study*; lock-free /
warp-cooperative is the production answer" — RGI's per-bucket lock taken by
one lane with `shfl` broadcast is that answer.

Fence in `lock_release`: `__threadfence()` BEFORE `atomicExch(l, 0)` is the
release barrier — all stores made inside the critical section (key and value
plain stores in the bucket/global insert paths) become visible to other
device threads before the lock word reads 0. Without it, an acquirer could
see the lock free yet read pre-critical-section slot contents. There is no
matching explicit acquire fence in `lock_acquire`; the protocol leans on the
atomicCAS plus the dependency chain (and the device-scope visibility
guarantees of atomics). When porting this code to a weaker model or
hand-tuning, the conservative fix is `__threadfence()` after winning the CAS.

Pitfall: `__nanosleep` compiles only for sm_70+. The Makefile pins
`-arch=sm_89`; building for older arches fails here first.

#### `ht_insert_bucketlock` (lines 120–132)

```c
__device__ uint32_t ht_insert_bucketlock(const Table &t, uint64_t key, uint64_t val)
```

Takes the spinlock **at the key's home slot only** (`locks[home]`, line 122),
then probes with plain (non-atomic, non-volatile) reads and writes (lines
124–129), releases (line 130). Returns OK on overwrite or claim, FULL after a
full scan.

This is the "medium" point of the contention study, and it carries a
**known, deliberate unsoundness** that any modifier must understand before
trusting its results beyond the study's purpose: the lock protects the *home
bucket index*, but linear probing walks into slots that are other keys' home
slots. Two writers with different homes whose probe chains overlap hold
DIFFERENT locks and can both observe the same slot as free (plain read, line
126) and both store into it (line 128) — last writer wins, first key is
silently lost. Lock-free avoids this with CAS; global-lock avoids it by
serializing everything. The bucket scheme is therefore only strictly correct
at load factors / key distributions where chains never overlap — which the
study's writeup must (and does) treat as a *protocol cost* comparison, not a
correctness-equivalent alternative. A correct bucket-lock variant would lock
every bucket the probe traverses (deadlock-prone, ordered acquisition
needed) or CAS within the critical section. Do not "fix" this casually; the
three schemes exist to be *compared*, and changing one's cost profile
invalidates the sweep history. If WP7 item 6 reruns the study on GH200,
rerun all three from the same commit.

Also note: no volatile on the reads here — legal only because the slot bytes
it reads are either protected by ITS lock (the home slot) or racy-by-design
(the overlap case above). The `lock_release` fence publishes its writes.

#### `ht_insert_globallock` (lines 134–146)

Identical probe body to the bucket variant but bracketed by
`lock_acquire(t.glock)` / `lock_release(t.glock)` — one lock for the entire
table (lines 135, 144). With the kernel's 1024 threads all contending for a
single int, this is intentionally the pathological baseline: every write in a
batch serializes, and warp-mates of the holder burn issue slots spinning
(mitigated only by `__nanosleep`). It is correct (single writer at a time;
release fence publishes), just slow — which is the data point it exists to
produce. Reads (`ht_lookup`) still proceed lock-free concurrently, so even
under the global lock the engine is not a true single-writer system; a lookup
racing the locked writer can see key-without-value exactly as in the
lock-free case (the plain stores at line 142 are key first, value second —
wait, line 142 writes key then value left-to-right: `t.keys[slot] = key;
t.values[slot] = val;` — same key-before-value order, same caveat).

#### `ht_delete` (lines 148–159)

```c
__device__ uint32_t ht_delete(const Table &t, uint64_t key)
```

Probe (plain reads, line 152); on key match, `atomicExch` the key to
`TOMBSTONE_KEY` (lines 153–154), `__threadfence()` (line 155), return OK; on
EMPTY, return NOTFOUND (line 156). The value is left in place (a tombstone's
value is garbage by definition; snapshot filters tombstones, line 423).

Two properties to keep in mind:

- **Delete ignores the locking scheme.** `do_write` routes only
  INSERT/UPDATE; the kernel calls `ht_delete` directly (line 205) under all
  three schemes. Under bucket/global lock, a delete can therefore interleave
  with a locked writer: writer reads `cur == key` (will overwrite), delete
  tombstones the slot, writer's plain store resurrects the key with the new
  value. Net effect "delete lost to concurrent upsert" — a permissible
  linearization, but if the contention study ever adds delete-heavy mixes,
  this asymmetry becomes a confound. Document it in any such experiment.
- `atomicExch` rather than a plain store: guarantees the transition is a
  single atomic RMW so a racing CAS-claimer (lock-free insert) sees either
  `key` or `TOMBSTONE`, never a torn intermediate; the returned old value is
  discarded (deliberately — even if a racer changed the slot between the read
  at line 152 and the exch, exchanging to TOMBSTONE is the intended
  last-writer outcome for delete-vs-insert races within a batch).
- The `__threadfence()` at line 155 orders the tombstone store before this
  thread's status write, same rationale as in the inserts.

#### The hash-table slot state machine (synthesis of lines 94–159)

```
                 atomicCAS(EMPTY -> key)         [insert, line 101]
        ┌────────────────────────────────────────────────┐
        │                                                ▼
   ┌─────────┐                                      ┌─────────┐ ───┐ value overwrite
   │  EMPTY  │                                      │  key K  │    │ (key unchanged)
   │ (0xFF…F)│       (no transition back to EMPTY,  │         │ ◄──┘ lines 99,105,
   └─────────┘        ever — chains only grow)      └─────────┘      127–128, 141–142
                                                      │     ▲
                       atomicExch(key -> TOMBSTONE)   │     │  atomicCAS(TOMBSTONE -> key')
                       [delete, lines 153–154]        ▼     │  [insert reuse, line 101]
                                                 ┌───────────┐
                                                 │ TOMBSTONE │  (skipped by lookups,
                                                 │ (0xFF…E)  │   line 87/88 falls through;
                                                 └───────────┘   reusable by inserts,
                                                                 line 100; filtered by
                                                                 snapshot, line 423)
```

Invariants encoded by this machine:
1. EMPTY is a source state only. Once any key or tombstone occupies a slot,
   no code path ever writes EMPTY_KEY again. This is what makes the
   `cur == EMPTY_KEY → NOTFOUND` probe-termination rule (line 88) safe under
   concurrency: a chain can only get longer, so a reader can never
   early-terminate past a key that was present when its probe began.
2. Tombstones are permanent until reused by an insert. Sustained
   delete/insert churn with disjoint keys degrades probe lengths
   monotonically; capacity should be provisioned ≥ 2× live keys (the bench
   defaults: cap 2^24, keys 2^22 → load factor 0.25).
3. The pair (key CAS/exch atomic, value plain store) means slot *identity*
   transitions are atomic but key↔value *consistency* is only guaranteed at
   batch boundaries.

#### `do_write` (lines 161–168)

```c
__device__ uint32_t do_write(const Table &t, uint32_t scheme, uint64_t key, uint64_t val)
```

Three-way dispatch on the per-batch `scheme`: BUCKETLOCK → bucket variant,
GLOBALLOCK → global variant, anything else → lock-free (lines 163–166). Both
INSERT and UPDATE opcodes land here (kernel lines 203–204), making both
upserts; "UPDATE of a missing key" therefore inserts it. The FDW relies on
this (its UPDATE path is a plain `gpu_oltp_update`), and the bench's workload
generator emits UPDATE for its write mix knowing all keys were preloaded.
If real insert-vs-update semantics (e.g., UPDATE→NOTFOUND on miss) are ever
needed, this function is where the split happens — and `pg_gpu_fdw.c` plus
the bench's expected-status assumptions must be revisited together.

#### `persistent_kernel` (lines 172–219) — the GPU half of the doorbell

```c
__global__ void persistent_kernel(Control *c, Table t,
                                  const uint32_t *types, const uint64_t *keys,
                                  const uint64_t *vals, uint32_t *stat,
                                  uint64_t *oval)
```

Launched exactly once, `<<<1, tpb, 0, e->stream>>>` (line 264), with the
mapped-memory device pointers for the control block and the five staging
arrays, and the `Table` by value. The launch-time comment (line 171) states
the dispatch discipline: "Single block. tid 0 spins on the doorbell,
broadcasts via __syncthreads."

Per-thread state: `tid`, `nt` (block size), and `seen` — the register holding
the id of the last batch this kernel completed, initialized 0 to match the
host's `cur_batch = 0` at creation (line 257). `seen` is per-thread (every
thread has its own copy, all kept equal at line 216), but only thread 0's
copy is ever *consulted* (in the spin, line 182).

The eternal loop, iteration anatomy:

1. **Spin (tid 0 only), lines 181–183.**
   `while (c->batch_id == seen && c->stop == 0) { __nanosleep(128); }`
   Thread 0 polls two volatile fields of the mapped control block. Each
   volatile read compiles to an actual load; because the memory is mapped
   host memory, each load is an uncached read across PCIe (~sub-µs each).
   `__nanosleep(128)` throttles the polling rate — without it the spin
   saturates the PCIe read path and (more importantly on a laptop) burns
   power for nothing. 128 ns is a latency/cost compromise: it bounds added
   dispatch latency at ~one sleep quantum. The other 1023 threads of the
   block do NOT spin — they go straight to the barrier and sleep there, which
   is why only "one SM busy-polling" (explainer §29) and not 1024 threads
   hammering PCIe.

2. **Barrier broadcast, line 184.** `__syncthreads()` releases the whole
   block once tid 0 has observed the doorbell (or stop). The barrier is the
   broadcast mechanism: no flag in shared memory is needed because the
   *fact of passing the barrier* communicates "there is a batch (or stop)".
   This is the payoff of the single-block design — in a multi-block grid this
   one line becomes the hard part (the RGI persistent v2 replaces it with a
   block-0-republishes + atomic-arrival-counter scheme, WP5 lines 20–23).

3. **Stop check, line 185.** `if (c->stop != 0) return;` — every thread reads
   the volatile `stop` independently and the whole block returns together,
   ending the kernel (this is the only exit). It is checked AFTER the barrier
   so all threads agree on the iteration in which they exit (no thread can be
   left waiting at a barrier the others skipped — barrier divergence is UB).
   Modification rule: any new early-exit must preserve "all threads take the
   same branch around every `__syncthreads()`".

4. **The acquire fence + the batch capture, lines 187–196.** This is fix 2's
   site; the comment (lines 187–192) is the canonical statement of the race.
   Code order:

   ```c
   __threadfence_system();   /* line 193 */
   const uint32_t my_batch = c->batch_id;   /* line 194 */
   const uint32_t count    = c->count;      /* line 195 */
   const uint32_t scheme   = c->scheme;     /* line 196 */
   ```

   - The `__threadfence_system()` at **line 193** is the acquire side of the
     handshake (the in-line comment says exactly this: "acquire: count/payload
     written before the doorbell are now visible"). Mechanism, precisely:
     tid 0's spin already *observed* the new `batch_id`; the fence prevents
     this thread's (and, after the line-184 barrier, the block's) subsequent
     loads of `count`, `scheme`, and the payload arrays from being reordered
     before that observation. `__threadfence_system` is a system-scope fence —
     the only CUDA fence whose scope includes the host — making it the correct
     (and required) strength for ordering against host stores in mapped
     memory. Pairing: it pairs with the host's
     `std::atomic_thread_fence(release)` at line 297 (host: payload/count
     stores ordered BEFORE the `batch_id` store; device: `batch_id` load
     ordered BEFORE payload/count loads — the classic release/acquire
     message-passing pattern, with `batch_id` as the flag).
   - **`my_batch` capture at line 194 — fix 2 itself.** The kernel reads
     `c->batch_id` exactly once per batch, into a register, at a moment when
     the value is provably stable: the host cannot advance `batch_id` until
     it observes `done_id == cur_batch` (host line 301), and `done_id` has
     not been written yet — the host is guaranteed to be blocked in its spin.
     This single captured value is then used for BOTH the completion signal
     (line 214, `c->done_id = my_batch`) and the seen-update (line 216,
     `seen = my_batch`).

     The broken version this replaced: signal `done_id = batch` first, THEN
     re-read `c->batch_id` to update `seen`. Failure interleaving (explainer
     §12 lines 500–510, and Part XV §58's diagram, lines 1759–1770):

     ```
     kernel: done_id = N            host: sees done_id == N, returns from submit
                                    host: stages batch N+1, batch_id = N+1
     kernel: seen = c->batch_id     <- reads N+1, records it as ALREADY SEEN
     kernel: spin: batch_id(N+1) == seen(N+1) -> never wakes
     host:   spin: done_id(N) != cur_batch(N+1) -> never returns
     ```

     Mutual deadlock, with batch N+1 silently dropped. The race window is the
     gap between the kernel's `done_id` store becoming host-visible and its
     own re-read — tiny, so single large batches "never" hit it, while 2,000
     rapid single-op submits (the latency loop, line 521) hit it reliably.
     It is a pure protocol bug — no WDDM, no PCIe specifics — and carries to
     C2C unchanged, which is why the comment block at lines 187–192 must
     survive any refactor of this kernel.
   - `count` and `scheme` are also captured into registers once (lines
     195–196) so the op loop reads stable values even though the host could
     not legally change them mid-batch anyway (defense in depth + avoids
     repeated PCIe reads of volatile fields).

5. **The op loop, lines 198–210.** Block-stride distribution:
   `for (uint32_t i = tid; i < count; i += nt)`. Thread `tid` handles
   requests `tid, tid+nt, tid+2nt, ...` — contiguous threads touch
   contiguous requests, so the reads of `types[i]`, `keys[i]`, `vals[i]`
   (mapped host memory) coalesce into wide PCIe transactions per warp.
   Per request: zero-init `v` (line 199), `switch` on the opcode
   (lines 201–207) into the table helpers, then write results:
   `if (stat) stat[i] = rc; if (oval) oval[i] = v;` (lines 208–209). The
   NULL guards are vestigial safety (the engine always passes both arrays,
   lines 264–265); they cost one predicated branch and allow a future caller
   to pass nullptr staging. Note results are written to MAPPED memory —
   each `stat[i]`/`oval[i]` store is a PCIe write toward host DRAM; for big
   batches this is the engine's result "DMA", overlapped naturally with
   compute across warps.

6. **Publish + signal, lines 212–215.**

   ```c
   __threadfence_system();   /* line 212: publish results before signalling done */
   __syncthreads();          /* line 213 */
   if (tid == 0) { c->done_id = my_batch; __threadfence_system(); }  /* line 214 */
   __syncthreads();          /* line 215 */
   ```

   Fence-by-fence:
   - **Line 212** (`__threadfence_system`, executed by every thread): orders
     each thread's `stat[i]`/`oval[i]` stores before anything that thread does
     subsequently — in particular before tid 0's `done_id` store can be
     observed. System scope is required because the observer is the HOST
     (reading `h_stat`/`h_oval` after seeing `done_id`). A `__threadfence()`
     (device scope) would be insufficient: it does not order visibility to
     the CPU.
   - **Line 213** (`__syncthreads`): ensures every thread has *executed* its
     line-212 fence (i.e., all results are fenced) before tid 0 proceeds to
     signal. Without this barrier, tid 0 — which may finish its slice of the
     op loop early — could signal `done_id` while other threads' result
     stores are still in flight and unfenced. The combination
     (per-thread system fence, then block barrier, then signal) is the
     single-block equivalent of a grid-wide release.
   - **Line 214**: tid 0 stores `done_id = my_batch` (volatile, mapped → a
     PCIe write the host's spin will see), then issues ANOTHER
     `__threadfence_system()`. This trailing fence orders the `done_id` store
     before tid 0's *next-iteration* operations — specifically before any
     store or load of the next batch's processing could be reordered ahead of
     the completion signal. It also flushes the doorbell write promptly
     rather than letting it linger in a write buffer behind future traffic.
     (For host-side correctness the essential ordering is line 212's fence
     before line 214's store; the trailing fence is the belt to that
     suspender, and on C2C it becomes the line that bounds completion
     latency.)
   - **Line 215** (`__syncthreads`): keeps the non-zero threads from racing
     ahead into line 216/217 and the next iteration's barrier while tid 0 is
     still signalling — maintaining the all-threads-same-barrier-sequence
     invariant.

7. **Seen-update, lines 216–217.** `seen = my_batch;` in EVERY thread (each
   updates its private register — only tid 0's matters, but updating all is
   free and uniform), then a final `__syncthreads()` (line 217) so no thread
   enters the next iteration's line-184 barrier before all have completed
   this iteration's epilogue. The three barriers (213, 215, 217) look
   redundant at a glance; collapsing them is the classic tempting
   "optimization" that reintroduces barrier-mismatch UB or lets a fast
   thread's next-iteration work overlap the completion signal. Leave them
   unless you re-derive the whole iteration's happens-before graph.

The complete handshake, host and GPU columns, every fence at its line:

```
HOST  (gpu_oltp_submit, lines 285–308)     GPU  (persistent_kernel, lines 172–219)
─────────────────────────────────────      ─────────────────────────────────────────
memcpy types/keys[/vals] into mapped       tid0: while (batch_id == seen
staging                    (292–294)              && stop == 0) __nanosleep   (182)
h_ctrl->count = chunk      (295)                  [volatile reads over PCIe]
                                           others: waiting at barrier        (184)
atomic_thread_fence(release)  (297)
  └ compiler barrier (x86: stores
    already ordered, TSO); payload+count
    ordered BEFORE doorbell store
cur_batch += 1             (298)
h_ctrl->batch_id = cur_batch  (299) ─────► tid0: load batch_id != seen → exit spin
  [PCIe posted writes arrive in            __syncthreads()  ← block broadcast (184)
   issue order: count before batch_id]     if (stop) return                  (185)
                                           __threadfence_system()            (193)
spin: while (done_id != cur_batch) {}        └ acquire: doorbell observation
                          (301)                ordered before count/payload loads
  [volatile read each iteration;           my_batch = batch_id  [CAPTURE — fix 2;
   no pause instruction — pure burn]         host is provably blocked here]  (194)
                                           count, scheme → registers     (195–196)
                                           op loop: i = tid; i += nt    (198–210)
                                             reads types/keys/vals  [mapped, PCIe]
                                             writes stat[i], oval[i] [mapped, PCIe]
                                           __threadfence_system()  (all threads)
                                             └ results ordered before done   (212)
                                           __syncthreads()  ← all fences done(213)
              ◄──────────────────────────  tid0: done_id = my_batch          (214)
sees done_id == cur_batch → exit spin      tid0: __threadfence_system()      (214)
atomic_thread_fence(acquire)  (302)          └ doorbell ordered before next-batch
  └ done_id observation ordered                work
    before result loads                    __syncthreads()                   (215)
memcpy out_status ← h_stat (304)           seen = my_batch  (all threads)    (216)
memcpy out_values ← h_oval (305)           __syncthreads()                   (217)
off += chunk; loop or return (306)         → top of for(;;): tid0 spins again
```

Protocol invariants (the kernel side):

- I-K1: `done_id` is written by exactly one thread (tid 0) and only with a
  value previously read from `batch_id` while the host was blocked.
- I-K2: between observing `batch_id != seen` and writing `done_id`, the
  kernel performs no other write to the control block.
- I-K3: every `__syncthreads()` is executed by all `tpb` threads in the same
  order — no barrier is inside thread-divergent control flow.
- I-K4: the kernel never reads `types/keys/vals` beyond index `count - 1`,
  and trusts `count` blindly; the host-side chunker (line 291) is the sole
  enforcer of `count <= MAX_BATCH`. Any new writer of `c->count` (e.g., a
  second producer in a future multi-client design) inherits this obligation.

Pitfalls when modifying the kernel:

- Going multi-block: `__syncthreads` stops being a dispatch barrier. Use the
  RGI persistent v2's two-level scheme (block 0 polls and republishes to
  device memory; atomic arrival counter as the completion rendezvous) rather
  than cooperative-groups grid sync, which constrains occupancy and forbids
  concurrent kernels — and note WP5's contract that grid size must equal
  exact occupancy (per-block co-residency invariant, WP5 lines 83–85).
- Any added device-side caching of control fields (shared-memory mirror,
  read-once hoisting by removing `volatile`) must be re-audited against the
  fix-2 interleaving: the question to ask of every read of `batch_id` is
  "can the host have advanced it since the value I'm about to act on was
  read?".
- TDR: on WDDM, a kernel that monopolizes the GPU can trip the watchdog. This
  kernel survives because WSL2's dGPU path and `__nanosleep`-throttled
  polling keep it under the radar, but a modified version that busy-spins
  without sleeping invites a device reset that manifests as
  `cudaErrorLaunchTimeout` on some *later* call — a misery to debug. Keep the
  `__nanosleep`s.

#### `round_pow2` (lines 222–224)

`static uint64_t round_pow2(uint64_t x) { uint64_t p = 1; while (p < x) p <<= 1; return p; }`
Rounds up to the next power of two; exact powers map to themselves;
`round_pow2(0) == 1`. Edge case worth knowing: a caller passing capacity 0
gets a 1-slot table with `mask == 0` — every key probes slot 0 and the second
distinct insert returns FULL. Garbage-in, defined-out; no need to guard, but
do not "simplify" the `<` to `<=`.

#### `gpu_oltp_create` (lines 226–279) — including fixes 1 and 3

Step by step:

1. `calloc` the handle; default `tpb` to 1024 when the argument is ≤ 0
   (lines 227–228). 1024 is the maximum block size on this hardware; a value
   > 1024 passes through unchecked and the launch at line 264 fails — caught
   by `CUDA_CHECK(cudaGetLastError())` at line 266 → abort. If you want a
   friendlier failure, clamp here.
2. `CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost))` (line 230). Required
   (historically) before the CUDA context is created so that
   `cudaHostAllocMapped` allocations are device-mappable. **Ordering
   sensitivity:** if some other CUDA call has already initialized the primary
   context in this process — e.g., in `pg_gpu_fdw`, calling `gpu_scan_load()`
   (which does `cudaMalloc`) before the first kv-table access — this call may
   return `cudaErrorSetOnActiveProcess` on CUDA versions that enforce the old
   rule, and `CUDA_CHECK` aborts the backend. On modern 64-bit UVA platforms
   the flag is effectively redundant (all pinned memory is mappable), and
   newer runtimes tolerate the call; but the safe modification rule stands:
   keep `gpu_oltp_create` the first CUDA touch in the process, or demote this
   to a non-fatal check deliberately and test the FDW's scan-then-kv order.
3. Table allocation and init (lines 232–241): round capacity; `cudaMalloc`
   keys/values/locks/glock; `cudaMemset(keys, 0xFF, ...)` writes the all-ones
   `EMPTY_KEY` pattern into every slot (the byte-memset trick — see the
   macro notes); zero the lock arrays. `values` is deliberately left
   uninitialized — a value is only ever read after its key matched, and the
   key only becomes visible at/after a value store (modulo the documented
   intra-batch publication race).
4. Mapped staging (lines 243–254): the `map_alloc` lambda wraps the
   three-step idiom — `cudaHostAlloc(..., cudaHostAllocMapped)`, host-side
   `memset 0`, `cudaHostGetDevicePointer` — applied to the control block and
   the five arrays (32 MiB pinned total; see `MAX_BATCH`). The memset matters
   for the control block specifically: `batch_id = done_id = 0` must agree
   with the kernel's `seen = 0` and the host's `cur_batch = 0` (line 257) or
   the very first doorbell is lost.
5. Default scheme = LOCKFREE (line 256).
6. **Fix 3 — the non-blocking stream (lines 259–263).** The comment is the
   specification: "the persistent kernel runs forever, so it must NOT be
   synchronized-with by the legacy default stream. Otherwise a later
   default-stream cudaMemcpy (e.g. gpu_oltp_snapshot) would implicitly wait
   for the never-completing kernel and deadlock." Mechanism: CUDA's *legacy*
   default stream has implicit-synchronization semantics — work submitted to
   it does not begin until all preceding work in all *blocking* streams
   completes, and vice versa. A stream created with default flags is
   blocking; therefore a synchronous `cudaMemcpy` (which is issued as if on
   the legacy stream) would wait for the persistent kernel — which never
   completes. `cudaStreamCreateWithFlags(&e->stream, cudaStreamNonBlocking)`
   (line 263) opts the kernel's stream out of that relationship entirely:
   legacy-stream operations neither wait for it nor make it wait. This is
   what keeps `gpu_oltp_snapshot`'s memcpys (lines 418–419) and
   `scan_reduce`'s default-stream kernels and event syncs (lines 392–400)
   safe while the kernel is resident. The design lesson recorded in the
   explainer (§12 lines 516–518): "a persistent kernel poisons every
   implicit-sync path in the process; stream topology is part of the design."
   One residual hazard survives — `cudaDeviceSynchronize`, which waits on ALL
   streams regardless of blocking flags; see `gpu_scan_alloc` below.
7. Launch (lines 264–266): `persistent_kernel<<<1, e->tpb, 0, e->stream>>>`
   with the DEVICE aliases of the mapped pointers, then
   `CUDA_CHECK(cudaGetLastError())` to catch launch-configuration errors
   (this checks queueing, not execution — a persistent kernel's runtime
   errors surface only on later calls).
8. **Fix 1 — the WDDM launch-queue flush (lines 268–277).** The comment
   (lines 268–272) is the specification: "WDDM (Windows / WSL2) batches
   launches in a command queue that is only flushed at a synchronization
   point. Without this, the persistent kernel sits in the queue and never
   starts, so the host spins on done_id forever." Mechanism: on the WDDM
   driver model the CUDA driver accumulates GPU commands in a user-mode
   command buffer and submits to the kernel-mode driver lazily, normally when
   a synchronization API forces it. This host code's steady state never calls
   the CUDA API again (submit is pure memcpy + volatile stores + spins), so
   nothing would ever flush the buffer and the launch would sit there
   forever — host spinning on a doorbell no kernel is polling. (The original
   triage was extra-confusing because `nvidia-smi` showed 100% GPU
   utilization from an unrelated quirk — explainer §12 lines 495–496.) The
   fix:

   ```c
   cudaEvent_t launch_ev;                          /* line 273 */
   CUDA_CHECK(cudaEventCreate(&launch_ev));        /* line 274 */
   CUDA_CHECK(cudaEventRecord(launch_ev, e->stream)); /* line 275 */
   cudaEventQuery(launch_ev);                      /* line 276 */
   CUDA_CHECK(cudaEventDestroy(launch_ev));        /* line 277 */
   ```

   `cudaEventRecord` enqueues an event behind the launch;
   **`cudaEventQuery` is the flush** — querying event status forces the
   driver to submit pending work so the event can make progress. Line 276 is
   deliberately NOT wrapped in `CUDA_CHECK` (its comment says why): the query
   legitimately returns `cudaErrorNotReady` — the event is behind a kernel
   that will never finish, so it will never be "ready"; the call is made for
   its side effect only. Do not replace it with `cudaEventSynchronize`
   (deadlock: waits behind the eternal kernel) and do not let a lint pass
   wrap it in the abort macro. On Linux/TCC the whole block is a cheap no-op
   (the comment, line 272). When WP5/WP7 port to bare-metal Linux this stays
   harmless; when anyone debugs "host spins forever right after create" on a
   new Windows box, this block is the first thing to verify survived.
9. Return the handle (line 278). Note `cur_batch` was left 0 by `calloc` and
   set explicitly at line 257 — host and device both start at "batch 0 is
   complete".

#### `gpu_oltp_set_scheme` (lines 281–283)

One volatile store: `e->h_ctrl->scheme = (uint32_t)scheme;`. No fence: the
next submit's release fence (line 297) orders it before the next doorbell
anyway, and the kernel samples scheme per batch (line 196). Calling this
between batches is therefore exact; calling it from another thread during a
batch is a race (single-thread contract).

#### `gpu_oltp_submit` (lines 285–308) — the host half of the doorbell

The chunking loop:

```
n requests from the caller
        │
        ▼
   off = 0                                  (line 289)
┌───────────────────────────────────────────────────────────┐
│ chunk = min(n - off, MAX_BATCH)            (line 291)     │
│ memcpy(h_types, types+off, chunk*4)        (line 292)     │
│ memcpy(h_keys,  keys+off,  chunk*8)        (line 293)     │
│ if (values) memcpy(h_vals, values+off, …)  (line 294)     │
│ h_ctrl->count = chunk        [volatile]    (line 295)     │
│ atomic_thread_fence(release)               (line 297)     │
│ cur_batch += 1                             (line 298)     │
│ h_ctrl->batch_id = cur_batch [DOORBELL]    (line 299)     │
│ while (h_ctrl->done_id != cur_batch) {}    (line 301)     │
│ atomic_thread_fence(acquire)               (line 302)     │
│ if (out_status) memcpy(out_status+off, h_stat, chunk*4)   │
│ if (out_values) memcpy(out_values+off, h_oval, chunk*8)   │
│                                            (lines 304–305)│
│ off += chunk                               (line 306)     │
└───────────────┬───────────────────────────────────────────┘
                └── while (off < n)          (line 290)
```

Step-by-step with the memory-model reasoning:

1. **Staging (lines 292–294).** Plain `memcpy` into the pinned mapped arrays.
   `values` is skipped when NULL — the whole-batch skip documented in the
   header walkthrough. The staging arrays are reused every chunk; their
   previous contents are dead by protocol (the kernel never reads past
   `count`).
2. **`count` store (line 295).** Volatile, so the compiler emits it in
   program order relative to the other volatile control-block accesses.
3. **Release fence (line 297).** `std::atomic_thread_fence(std::memory_order_release)`.
   What it actually does on x86-64: emits **no instruction** — x86 TSO
   already guarantees store→store ordering — but acts as a full compiler
   reordering barrier, forbidding the compiler from sinking the memcpys or
   the `count` store below the `batch_id` store. The pairing is with the
   device fence at kernel line 193 (and, physically, with PCIe's posted-write
   ordering rule: writes from the CPU to the BAR/pinned region arrive in
   issue order, so the GPU cannot observe `batch_id` new but `count` old).
   Pedantic note for modifiers: mixing `atomic_thread_fence` with
   non-atomic/volatile accesses is not a formally-blessed C++11 pattern (the
   standard's fences pair with atomics), but it is the established idiom for
   external-device mailboxes; if you ever migrate this file to
   `cuda::atomic_ref` / libcu++ (`cuda::std::atomic` with
   `thread_scope_system`), migrate BOTH sides of the pairing at once — that
   is also the natural WP7 modernization, since GH200's coherence makes
   `cuda::atomic<.., thread_scope_system>` the first-class way to express
   this protocol.
4. **Doorbell (lines 298–299).** Increment the host-private `cur_batch`
   mirror, then publish it with a single volatile 32-bit store. 32-bit
   wraparound after 2^32 batches is theoretically possible
   (`batch_id == seen` again after exactly 2^32 submits with none in
   flight — at 75 µs/batch that is ~3.7 days of continuous single-op
   traffic); the protocol survives wraparound EXCEPT the pathological case of
   exactly-2^32 unobserved increments, which cannot happen here because the
   host blocks per batch. Don't change the field to 64-bit casually — field
   width is part of the C2C line-layout decision (WP7 item 1 writes "a 64 B
   request line").
5. **Completion spin (line 301).** `while (e->h_ctrl->done_id != e->cur_batch) {}`
   — a pure userspace spin on a volatile read of pinned memory (the GPU's
   `done_id` store arrives by PCIe write into host DRAM; the CPU's cached
   copy is invalidated by normal coherence since the write targets host
   memory). No `_mm_pause()`, no sleep, no timeout. This is deliberate for
   the measurement role (any backoff would pollute the single-op latency
   number) but means: (a) one host core is 100% busy for the duration of
   every batch — fine for a bench, noteworthy inside a Postgres backend;
   (b) a dead kernel hangs the process here forever (the failure signature of
   fix 1's bug). If you add a robustness timeout for production use, gate it
   so the bench path keeps the pure spin.
6. **Acquire fence (line 302).** Mirror of step 3: prevents the compiler from
   hoisting the result `memcpy`s above the spin's final `done_id`
   observation (x86 load→load is already ordered; this is again a
   compiler-barrier in practice). It pairs with the kernel's line-212 system
   fence: results were fenced before `done_id` was stored, so once the host
   observes `done_id`, the results in `h_stat`/`h_oval` are complete.
7. **Result copy-out (lines 304–305)** into the caller's arrays at offset
   `off`, each independently skippable by NULL.
8. **Advance (line 306)** and loop while requests remain.

Host-side invariants:

- I-H1: exactly one batch outstanding, ever (the spin enforces it). The
  kernel's fix-2 reasoning ("the host is blocked, so `batch_id` is stable at
  capture time") DEPENDS on this; an async or multi-threaded submit breaks
  the proof and requires re-deriving the protocol (this is precisely WP7
  item 3/4 territory — do it there, on purpose, not here by accident).
- I-H2: `cur_batch` strictly increases by 1 per chunk; `batch_id` is written
  by no one else (except `gpu_oltp_destroy`'s wake-bump, which happens only
  after `stop = 1`).
- I-H3: `count` is always ≤ `MAX_BATCH` (line 291) — the kernel's only
  bounds protection.
- I-H4: no CUDA API call appears anywhere in this function. That is a
  feature, not an omission (it is what fix 1's failure mode taught:
  steady-state submit must not depend on driver progress, and conversely the
  launch path must self-flush).

#### Single-op wrappers (lines 310–323)

Four near-identical bodies; e.g. lookup (lines 316–320) builds a stack batch
of one (`t`, `key`, statuses), passes `nullptr` for `values` (legal: no write
op in the batch), and unpacks `s`/`v`. Each call is one full doorbell round
trip by construction — these are simultaneously the FDW's row-at-a-time
operators and the latency probe. There is no fast path to add here without
changing what the 75.5 µs measures; resist batching inside the wrappers.

#### The scan module (lines 325–411) — bandwidth demo, independent of the doorbell

##### `struct GpuScanCol` + `g_scan_last_ms` (lines 326–327)

`{ uint64_t *d; uint64_t n; }` — a device column pointer and its length.
`g_scan_last_ms` is a file-scope static double holding the last reduction's
kernel-only time; written by `scan_reduce` (line 398), read by
`gpu_scan_last_ms()` (line 407). Process-global and unsynchronized — fine
under the single-thread contract, but it means two engines/columns share one
"last ms" slot.

##### `grid_for` (lines 329–332)

`(n + tpb - 1) / tpb` blocks, clamped to [1, 65535]. The 65535 clamp is the
legacy 1-D grid limit (modern x-dimension allows 2^31-1); harmless
conservatism because every scan kernel is grid-stride and covers any `n`
regardless of grid size.

##### `scan_fill_kernel` (lines 334–338)

Grid-stride fill `d[i] = i + 1`. The `i+1` (not `i`) gives closed-form
oracles: `sum = n(n+1)/2`, `count_lt(thr) = min(thr-1, n)` — the demo's
correctness is checkable by inspection at the SQL level.

##### `SCAN_TPB` and the reduction kernels: `scan_sum_kernel` (lines 342–356), `scan_countlt_kernel` (lines 357–371)

Identical structure, different accumulation (sum vs. predicate count):

1. Grid-stride accumulate into a per-thread register (lines 345–348 /
   360–363) — this is where the HBM streaming happens; with the grid capped
   at 4096 blocks (see `scan_reduce`), each thread strides over many
   elements, keeping the kernel bandwidth-bound.
2. Per-block shared-memory tree reduction (lines 349–354 / 364–369):
   `sm[SCAN_TPB]`, then halving strides with `__syncthreads()` between
   levels. Requires `blockDim.x` be a power of two — it is always launched
   with `SCAN_TPB = 256` (line 339); changing `SCAN_TPB` to a non-power-of-2
   silently corrupts results.
3. ONE `atomicAdd(out, sm[0])` per block (lines 355 / 370). The comment at
   lines 340–341 states the design point: one atomic per block, not per
   thread, so with ≤ 4096 blocks the atomic traffic is noise and the kernel
   is "bandwidth-bound, not atomic-bound". This is the kernel pair behind
   the OLAP supporting result (explainer §36) — the ~216× SQL-level
   aggregate win whose decomposition (≈3–4× bandwidth, rest executor
   overhead) the explainer insists on stating.

##### `gpu_scan_alloc` (lines 373–381) — and the residual deadlock hazard

Allocates the column (`cudaMalloc`, line 376 — `n * 8` bytes of device
memory; 50M rows = 400 MB, the demo's standard size), launches
`scan_fill_kernel` on the DEFAULT stream (line 378), then
**`cudaDeviceSynchronize()` (line 379)**.

That synchronize is the one place this module's "independent of the
persistent-kernel engine" claim (header lines 78–82) fails:
`cudaDeviceSynchronize` waits for ALL work on the device **in all streams,
including non-blocking ones** — including a resident `persistent_kernel`,
which never completes. Concretely: in a `pg_gpu_fdw` session that touches the
kv table first (lazily creating the engine → kernel resident) and then calls
`gpu_scan_load(...)`, line 379 blocks forever. The demo flows happened to
exercise scan-only or kv-only sessions, so this was never tripped, but it is
a real landmine. The fix, when someone needs both in one session: replace
lines 378–379 with a launch on a private non-blocking stream +
`cudaStreamSynchronize(that_stream)` (the same medicine as fix 3). Until
fixed, treat "engine resident ⇒ do not call gpu_scan_alloc" as an operating
rule. (`scan_reduce` does NOT have this problem — see next.)

##### `gpu_scan_count` (line 383)

Trivial accessor (`s ? s->n : 0`); exists so the FDW can report the loaded
row count without another kernel.

##### `scan_reduce` (lines 385–403)

The shared driver for both aggregates:

1. `cudaMalloc` an 8-byte accumulator, zero it via `cudaMemcpy` H2D
   (lines 387–388). Per-call alloc/free keeps the function stateless; at
   the demo's call rates this is irrelevant overhead, but it IS included in
   end-to-end timings (only the kernel is event-timed).
2. Grid: `grid_for(n, 256)` then clamp to 4096 (lines 390–391; the comment:
   "cap grid; grid-stride covers the rest"). 4096 blocks × 256 threads is
   ample to saturate this GPU's DRAM while keeping per-block atomics rare.
3. Time the kernel with a `cudaEventRecord(a)` / kernel / `cudaEventRecord(b)`
   / `cudaEventSynchronize(b)` bracket (lines 392–397), store elapsed ms
   into `g_scan_last_ms` (line 398).
4. Copy the scalar back (`cudaMemcpy` D2H, line 400), free, return.

Why this is safe while the persistent kernel is resident (unlike
`gpu_scan_alloc`): every blocking call here targets the LEGACY default
stream — the kernels, the events, the synchronous memcpys. The legacy
stream's implicit synchronization extends only to *blocking* streams; the
engine's stream is non-blocking (fix 3), so `cudaEventSynchronize(b)` waits
only for work preceding `b` in the legacy stream. The events/kernels make
progress because the GPU executes both the resident kernel's block and these
kernels' blocks concurrently (the resident kernel occupies one SM's worth;
the scan grid uses the rest). Note also these calls DO flush WDDM's queue —
which is why only `gpu_oltp_create` (whose steady-state successor never calls
CUDA again) needed the explicit fix-1 flush.

Error-handling nonuniformity worth knowing: lines 392–399 call the event APIs
unchecked (timing-path pragmatism), while allocation/memcpy are CUDA_CHECKed.

##### `gpu_scan_sum` / `gpu_scan_count_lt` / `gpu_scan_last_ms` (lines 405–407)

One-line wrappers over `scan_reduce(s, false, 0)` / `scan_reduce(s, true, thr)`
/ the static. NULL `s` is NOT guarded here (unlike `gpu_scan_count`) —
`scan_reduce` dereferences `s->n` at line 390; the FDW guards at its layer
(errors if no column loaded).

##### `gpu_scan_free` (lines 409–411)

NULL-safe free of the device column and the struct. Does not touch
`g_scan_last_ms` (a stale last-ms after free is readable; harmless).

#### `gpu_oltp_snapshot` (lines 413–429)

The full-table-scan support for `SELECT *` via the demo FDW:

1. `malloc` two host arrays of FULL capacity (lines 416–417) — for the
   default FDW capacity this is transient host memory of `2 * cap * 8`
   bytes; callers with big tables should know the high-water mark is
   capacity-, not cardinality-, proportional.
2. Two synchronous `cudaMemcpy` D2H of the entire keys and values arrays
   (lines 418–419). These run on the legacy default stream — **safe against
   the resident kernel only because of fix 3** (the .cu's own comment at
   lines 260–262 names this exact call site as the deadlock victim).
3. In-place compaction (lines 420–425): walk all slots; skip
   EMPTY/TOMBSTONE; copy live pairs to the front (`hk[n] = k; hv[n] = hv[i]`).
   Forward compaction is correct because `n <= i` always. Order of returned
   pairs is slot order — i.e., effectively unordered for callers.
4. Ownership transfer (lines 426–428): non-NULL out-pointers receive the
   (still capacity-sized) buffers, caller frees; NULL out-pointers cause an
   immediate free. Returns the live count.

Consistency caveat (restating the header contract with the mechanism): the
memcpys read device memory with no quiescing handshake against the kernel. A
concurrent write batch can produce a snapshot containing a key whose value
store had not yet landed (key-before-value publication), or half of a batch.
Under the single-host-thread contract there is never an in-flight batch
during snapshot, so this is theoretical until someone adds threads — at which
point the right fix is a drain: submit a zero-op batch and wait, THEN memcpy
(the doorbell protocol gives you quiescence for free: `done_id == batch_id`
and no submit in progress ⇒ kernel is in its spin, touching nothing but the
control block).

#### `gpu_oltp_destroy` (lines 431–443)

Shutdown sequence, order-critical:

1. `e->h_ctrl->stop = 1;` (line 433) — volatile store.
2. `std::atomic_thread_fence(release)` (line 434) — orders `stop` before the
   wake-bump, so the kernel cannot observe the bumped `batch_id` yet miss
   `stop` (it would then try to process a garbage batch: `count` is stale —
   actually last batch's count — and the staging arrays stale; harmless-ish
   but wrong; the fence plus the kernel's stop-check-before-capture order at
   line 185 prevents it).
3. `e->h_ctrl->batch_id = e->cur_batch + 1;` (line 435) — "wake the spinner
   so it exits". Strictly, the spin already polls `stop` (line 182's
   condition checks both), so the bump is redundant today; it is cheap
   insurance for any future spin variant that polls only the doorbell, and it
   bounds wake latency to one poll iteration regardless.
4. `cudaStreamSynchronize(e->stream)` (line 436) — now the kernel CAN exit
   (line 185), so this returns. (Also flushes WDDM, so even a hypothetical
   queued-but-never-started kernel resolves here.) Unchecked deliberately:
   destroy should proceed even if the device is in a bad state.
5. Free everything: device table arrays (437–438), the six pinned mapped
   buffers via `cudaFreeHost` (439–440), the stream (441), the handle (442).

Pitfalls: calling destroy while another thread is inside submit deadlocks or
worse (single-thread contract); calling any engine function after destroy is
use-after-free; on a SHARED machine, ensuring destroy runs on every exit path
(signal handlers included) is an explicit WP5 contract (lines 105–107) —
"never leave a spinning kernel on a shared box".

<!-- CONTINUED -->

