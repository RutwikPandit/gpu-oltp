# Section 03 — The RGI-Backed Engine and the Benchmark Binaries

This section documents the production GPU engine (`engine/rgi_oltp_engine.{h,cu}`,
compiled to `librgioltp.so`), the additive persistent-kernel binding
(`engine/rgi_persist_engine.cu`), the four standalone benchmark/profiling binaries
(`rgi_sweep.cu`, `rgi_prof.cu`, `rgi_prof2.cu`, `cpu_sweep.cpp`), and the three
build scripts that produce all of them (`build_all.sh`, `build_bench.sh`,
`build_persist.sh`). Every function, struct, kernel, and constant in those files
is walked through below with exact line numbers, the reasoning behind its shape,
the invariants it depends on, and what breaks if a modification violates them.

A reader of this section is assumed to have read (or to have available) the
adjacent primer sections covering the RGI library internals and the Postgres
integration (FDW + worker). The one-paragraph recap of each, for self-containment:

- **RGI (RobustGPUIndexing)** is the research group's header-only CUDA template
  library of GPU index structures. The engine instantiates its chained hashtable:
  `GpuHashtable::gpu_chainhashtable<simple_slab_allocator<128>, simple_debra_reclaimer<>, 16>`.
  Its execution model is warp-cooperative: a host-side call like `table.insert(...)`
  launches RGI's `batch_kernel`, inside which 16-lane tiles process requests one at
  a time, all lanes cooperating (ballot to find pending requests, `__ffs` to pick
  one, `shfl` to broadcast its key, cooperative node traversal). Nodes are 128-byte
  cache lines (one warp-coalesced load each). Deletion is safe under concurrent
  readers via DEBRA epoch-based reclamation; allocation is a 128-byte bitmap slab
  allocator. RGI has **no scan/enumeration API** and a hard `uint32_t` value type —
  two facts that shaped the wrapper, as documented below.

- **The Postgres side** (`pg_rgi_fdw` + the `pg_gpu_service` background worker) is
  plain C compiled by PGXS. It never includes a CUDA header. Everything it knows
  about the GPU is the ~20-function C ABI declared in `engine/rgi_oltp_engine.h`.
  The single background worker is the only process that ever touches CUDA; all
  backends reach it through a shared-memory bulk channel.

The files covered here are the layer between those two worlds, plus the
measurement apparatus that produced every published number in the project.

---

## 0. Map of this section

| File | Lines | Role |
|---|---|---|
| `engine/rgi_oltp_engine.h` | 93 | The C ABI — the stability boundary between CUDA and Postgres |
| `engine/rgi_oltp_engine.cu` | 345 | THE production engine behind SQL (`librgioltp.so`); also `rgi_bench` under `-DBUILD_BENCH` |
| `engine/rgi_persist_engine.cu` | 346 | v1 persistent-kernel binding (doorbell dispatch), additive, standalone bench |
| `engine/rgi_sweep.cu` | 51 | Launch-mode latency/throughput sweep (source of the floor/ceiling numbers) |
| `engine/rgi_prof.cu` | 30 | Minimal two-kernel ncu profiling target (SOL analysis) |
| `engine/rgi_prof2.cu` | 41 | Parameterized single-op ncu target (DRAM-bandwidth schmoo) |
| `engine/cpu_sweep.cpp` | 62 | The deliberately strong CPU baseline (OpenMP open-addressing table) |
| `build_all.sh` | 25 | Builds `librgioltp.so` + the Postgres extension (the production path) |
| `build_bench.sh` | 33 | Builds all standalone benchmarks into `~/gpu_bench` (WSL home) |
| `build_persist.sh` | 29 | Builds + runs the persistent-kernel bench, regenerates figures |

Reading order if modifying the engine: §1 (architecture) → §2 (key mapping) →
§3 (the header) → §4 (the engine, especially §4.18 `rgi_stage_commit`) → §7
(invariants). Reading order if working on dispatch/WP5: §1 → §5 (persistent
engine) → §6.1 (`rgi_sweep`) → §6.4 (`cpu_sweep`) → §8 (numbers card).

---

## 1. Architecture context: where these files sit and why they have this shape

### 1.1 The layer diagram

```
                     psql / any Postgres client
                               |
            =================== SQL ====================
                               |
   Postgres backend processes (one per connection, plain C)
     pg_rgi_fdw: planner hooks, scan/modify callbacks,
     transaction buffer, PRE_COMMIT staging protocol
                               |
              shared memory bulk channel (+ LWLock)
                               |
   pg_gpu_service background worker  (the ONLY CUDA process)
                               |
   ============ THE C ABI: rgi_oltp_engine.h =============   <-- stability boundary
                               |
   librgioltp.so  (rgi_oltp_engine.cu, nvcc-compiled)
     - host-side write buffering          (pend_k / pend_v)
     - validate-then-apply staged commit  (stg_* vectors)
     - live-key shadow set                (live)
     - chunked batched launches           (BATCH_MAX = 1M)
                               |
   RGI gpu_chainhashtable< slab<128>, debra<>, 16 >
     batch_kernel (ballot queue), 128B nodes, suffix nodes,
     per-bucket-head write locks, latch-free reads, DEBRA
                               |
                        RTX 4060 Laptop GPU
                  (WSL2 / WDDM driver model, sm_89)
```

Parallel to the production stack, and sharing the same RGI instantiation but not
the same process, sits the **persistent-kernel binding** (`rgi_persist_engine.cu`):
a standalone binary in which one resident kernel polls a mapped-memory doorbell
instead of being launched per batch. It is the prototype of the GB-class (C2C)
dispatch model and is deliberately additive — nothing in the production engine,
the FDW, or the worker references it.

### 1.2 The C ABI is the stability boundary — treat it as frozen

`rgi_oltp_engine.h` is compiled by two different toolchains: nvcc (building the
`.cu` that defines the functions) and the system gcc via PGXS (building the FDW
that calls them through `dlopen`/link). It therefore contains only `stdint.h`
types, an opaque `struct RgiEngine` forward declaration (line 23), and
`extern "C"` linkage guards (lines 19-21, 88-90). No CUDA type, no C++ type, no
RGI template ever appears in it. The explainer (§14) names this the portability
boundary: the FDW and worker would survive a wholesale engine swap (different
index structure, different GPU, eventually a C2C-native runtime) without
recompilation, because everything they know is these ~20 signatures.

Practical consequence for any modifier: **WP1 (small-key fast path), WP5 (persistent
v2), and WP6 (GPU enumeration) are all specified as wrapper-internal or additive
changes precisely so this header does not change.** WP1's contract section says
explicitly: "C ABI unchanged (`rgi_*` signatures identical)." If a planned change
appears to require an ABI change, the correct move is to add a new function (the
header has room; the worker dispatches on opcodes, not on header position), never
to alter an existing signature — the FDW's `.so` and the engine's `.so` are built
by different scripts at different times and an silent ABI skew produces stack
corruption, not a link error.

### 1.3 The dispatch thesis, in one paragraph, because every file here serves it

The project's claim is that OLTP-on-GPU is a **dispatch problem**: a single GPU
point operation costs ~12-23 µs of launch overhead on this box (WSL2/WDDM PCIe)
versus ~22 ns for a CPU L2-resident probe, so per-op dispatch loses by three
orders of magnitude — but the *same* GPU sustains 1,248.7 Mop/s when 262,144
lookups ride one launch. Every design choice in these files is downstream of
that: the engine buffers writes host-side and flushes them as one batched launch
(§4.6); reads flush first and then batch (§4.11, §4.12); the transaction commit
is shaped so the whole write set crosses the PCIe boundary as a handful of
launches (§4.18); the persistent engine exists to attack the floor itself by
replacing the launch with a doorbell (§5); `rgi_sweep` measures the floor and
ceiling (§6.1); `cpu_sweep` measures the opponent (§6.4). When modifying any of
these files, the first question to ask is: does this change add a launch, a
synchronization, or a PCIe crossing to a per-operation path? If yes, it is
probably wrong.

### 1.4 Two bindings of one dispatch primitive

The explainer's §29 framing is load-bearing for understanding why
`rgi_persist_engine.cu` exists as a separate binary rather than a mode of the
production engine: **persistent kernel is the architecture, launch-per-batch is
the implementation on this box.** On WSL2/WDDM, the doorbell rendezvous measures
75-90 µs versus 13-28 µs for a kernel launch — the persistent binding is 5-6x
*worse* on this platform (exactly as the earlier toy engine predicted), and a
resident kernel pins every SM block slot, so no other kernel in the process can
launch while it lives (§5.1, hazard note at `rgi_persist_engine.cu:30-33`). So
the production SQL path uses launch-per-batch, and the persistent binding is
kept compiling, validated for correctness (1024/1024 on persistent FIND and
INSERT), and ready for the bare-metal campaign (WP5) where the decision rule is:
if measured `D_doorbell < D_launch` on bare-metal Linux PCIe, the persistent
binding becomes the preferred binding on that platform.

---

## 2. The key mapping and the suffix-node tax (read this before touching any chunk helper)

### 2.1 The mapping

The SQL surface is a two-column table `(k bigint, v bigint)`. The engine maps it
as (header lines 7-8, engine lines 8-9):

```
SQL bigint key  (8 bytes)  ->  TWO uint32 RGI key-slices   (max_key_length = 2)
SQL bigint val  (8 bytes)  ->  ONE uint32 RGI value        (truncated to 32 bits)
```

The two-slice layout is realized with zero marshalling cost: a host array of
`uint64_t` keys is `cudaMemcpy`'d directly onto a device buffer that RGI reads
as `uint32_t` slices — little-endian layout makes `uint64[i]` exactly slices
`{2i, 2i+1}` with the low 32 bits first. Every call site passes `2` as the
key-length argument and `nullptr` as the per-request `d_key_lengths` array
(meaning "all keys are fixed length 2") — see `rgi_oltp_engine.cu:68, 74, 150, 204`
and `rgi_persist_engine.cu:139, 141, 270, 282`.

The value truncation is a scoping decision, not an accident (explainer §22, §27):
RGI's `value_type` is a hard `uint32_t` at the template level, so the prototype
is presented as a **key → row-id index** (an access method), never a tuple
store. `rgi_insert` casts at `rgi_oltp_engine.cu:137`; the cast appears again at
lines 143, 274, 275. Anyone widening values must fork or extend RGI — that work
is explicitly deferred to the TAM/row-store stage (WP8).

### 2.2 The suffix-node tax — the most important performance fact about this engine

Discovered by reading RGI's source (`gpu_chainhashtable.hpp`), confirmed in
profile data, documented in the explainer §13 and WP1: **any key with
`key_length > 1` takes RGI's suffix-node path.** The 128-byte bucket node stores
only a 32-bit *hash tag* of such a key; the actual key slices and the value live
in a separately slab-allocated 128-byte **suffix node** reached through a
pointer. Because this wrapper maps every 8-byte key to two slices, *every key in
the production engine today pays the suffix hop*:

```
FIND  (2-slice key):                          FIND (hypothetical 1-slice key):
  load bucket node (128B, coalesced)            load bucket node (128B)
  match 32-bit hash tag                         match key slice inline
  -> DEPENDENT load of suffix node (128B)       read value inline
     streq compare slices there                 done: ONE node load
     read value there
  done: TWO dependent node loads

INSERT (2-slice key): + one slab allocation per key (the suffix node)
SPACE : ~134 B per 16 B key/value pair
```

Three things to hold onto:

1. **All published numbers include this cost.** The 1,248.7 Mop/s find ceiling,
   the 12-23 µs floor, the ncu SOL profile (find 41.7% DRAM / 64.5% L1, insert
   13.6% DRAM) — all were measured with every key on the suffix path. The
   numbers are honest; the tax is *found headroom*, presented as such.
2. **The ncu profile is partly explained by it.** The find kernel's L1-heavy
   profile (64.5% L1 vs 15.4% L2) includes the second, dependent, L1-served
   128 B access per op; the insert kernel's latency-bound character includes
   the per-key slab allocation.
3. **WP1 is the fix**, and it is wrapper-only: keys with `(key >> 32) == 0`
   get stored/probed as ONE slice via RGI's per-request `d_key_lengths` array
   (currently `nullptr`). The change lands in exactly the three chunk helpers
   documented in §4.4, §4.5, §4.14 plus the persistent kernel (§5.3), keeping
   the staging layout (2 uint32 slots per key) unchanged. The correctness
   argument: the length rule is a pure function of the key value, so store and
   probe lengths always agree; the dangerous case (same value stored once as
   1-slice and once as 2-slice) is impossible under the rule. WP1's pitfall
   list (its lines 92-101) flags the erase path — a 1-slice-stored key erased
   with length 2 would miss *silently* — and requires a mixed-range SQL test
   straddling 2^32 including DELETE.

When reading the walkthroughs below, every `..., 2, nullptr, ...` argument pair
is this mapping; after WP1 each becomes `..., 2, d_lens, ...` where `d_lens[i]`
is 1 or 2 computed by the rule.

---

## 3. File: `engine/rgi_oltp_engine.h` (93 lines)

### Purpose

The plain-C linking boundary between the RGI CUDA wrapper (built as
`librgioltp.so` by nvcc) and the Postgres FDW/worker (built as plain C by PGXS,
which "never sees RGI's CUDA templates" — header comment, lines 1-5). It is the
complete catalogue of what the database side can ask the GPU side to do.

### Position in the architecture

Everything above this header is Postgres; everything below is CUDA. The header
also documents, in comments, the three behavioral contracts callers rely on:
the batching model (lines 10-12), the atomic-commit semantics (lines 56-66),
and the paged-snapshot protocol (lines 78-83). Those comments are normative —
the FDW and worker were written against them — so a change to engine behavior
that invalidates a header comment is an ABI-semantic break even if no signature
changes.

### Design rationale

Why ~20 small functions instead of a struct-of-function-pointers vtable or a
single command-union entry point: the worker already speaks an opcode protocol
over shared memory, so the engine ABI mirrors it one-to-one (one C function per
worker opcode family), which makes the worker's dispatch a trivial switch and
keeps every function independently testable from a small C harness. Why an
opaque handle (`typedef struct RgiEngine RgiEngine`, line 23): the engine struct
contains C++ standard-library members (§4.3) that must never be visible to a C
compiler.

### Walkthrough

#### Header comment (lines 1-13)

States the three facts a caller must know before reading any signature:
(a) the storage substrate — "RGI GPUChainHashtable (warp-cooperative, concurrent,
reclaiming)" (line 7); (b) the KV mapping — "8-byte key -> two uint32 RGI
key-slices; value -> uint32 row id" (line 8, see §2 above); (c) the batching
contract — "inserts/updates are buffered host-side and flushed as ONE RGI
batch_kernel launch (rgi_flush, or implicitly before a read). … the batch size
is the latency/throughput knob" (lines 10-12). The phrase "or implicitly before
a read" is the read-your-own-writes guarantee at engine level: no read API can
ever observe a state in which buffered writes are missing (§4.7 invariant).

#### `RgiEngine` (line 23)

`typedef struct RgiEngine RgiEngine;` — opaque. The definition lives only in
the `.cu` (lines 45-62). Pitfall: never move any field of the real struct into
this header "for convenience"; the layout includes `std::vector` and
`std::unordered_set` members whose size and layout differ across the two
compilers that include this header.

#### `rgi_create` (line 26)

`RgiEngine *rgi_create(uint32_t capacity, float fill_factor, float pool_ratio);`
Comment (line 25): capacity = expected #keys (sizes the table);
fill_factor/pool_ratio tune RGI. The worker calls this exactly once at server
start with capacity 4M. `fill_factor` is RGI's bucket multiplier (the production
call and every benchmark use 2.0f); `pool_ratio` sizes the slab allocator's pool
as a fraction of free GPU memory (0.5f in the production engine bench at
`rgi_oltp_engine.cu:327`, 0.4f in the standalone benches). Implementation: §4.4.

#### `rgi_destroy` (line 27)

Frees device buffers and host objects (§4.21). Only called at worker shutdown;
nothing in the system creates/destroys engines on a hot path.

#### `rgi_insert`, `rgi_update`, `rgi_delete` (lines 30-32)

The buffered write triple. Comment (line 29): "Buffered write ops (no GPU work
until a flush / read). value truncated to 32b." Important asymmetry, visible
only in the implementation and documented in §4.8-§4.10: insert and update are
pure host-side appends (identical bodies, because the eventual flush uses
upsert semantics), but **`rgi_delete` is NOT buffered** — it flushes and issues
an immediate single-key erase launch. The FDW's transactional path never uses
`rgi_delete` (deletes ride the staging API), so this per-op launch only costs
the legacy/demo path. A modifier adding delete buffering must preserve
operation ordering against buffered inserts of the same key (§4.10 pitfalls).

#### `rgi_flush` (line 35)

"Flush buffered writes to the GPU index (one batched launch per chunk)."
The parenthetical is the chunking discipline (§4.23): one launch per
`BATCH_MAX = 1M` entries, not literally one launch.

#### `rgi_flush_unique` (line 41)

The statement-level UNIQUE/PK-enforcing flush. The comment (lines 37-40) states
the full contract: checks (a) duplicate keys within the batch and (b) keys
already present in the index, **before applying**; on conflict nothing is
applied, `*dup_key` is set, returns 1; on success applies and returns 0. This is
the single-statement sibling of the transactional `rgi_stage_commit` and uses
the same validate-then-apply skeleton (§4.7).

#### `rgi_lookup` (line 44)

Single-key point read. "(flushes first)" — read-your-writes. Returns 1/0 found,
value through the out-pointer. Implementation is a batched find of size 1
(§4.11) — i.e., a single lookup pays the full dispatch floor; that is the point
the sweep quantifies.

#### `rgi_find_many` (lines 49-50)

The batched point-read used by SQL qual pushdown (`k = const`,
`k = ANY(array)` — comment lines 46-48). Parallel out-arrays `out_values[i]` /
`found[i]`; "One GPU find launch (chunked)". This is the function that turned
point SELECTs from CPU-wins to GPU-wins at SQL level.

#### `rgi_snapshot` (line 54)

Full-table snapshot for `SELECT *`: flushes, then batched-finds **all live
keys** on the GPU; returns row count; out-arrays are `malloc`'d and owned by the
caller (comment lines 52-53: "caller frees with free()"). The existence of a
"live key" notion the engine can enumerate is the host-side shadow set (§4.13).
Superseded in the worker by the paged variant below but kept for the bench and
any caller that wants one-shot semantics.

#### The staging API block (lines 56-76)

The header's largest comment (lines 56-66) is the atomicity argument, quoted
because every word was chosen after a real bug:

> "A commit is staged in full, then validated, then applied. Atomicity comes
> from validating ALL error conditions (PK/UNIQUE) BEFORE any mutation, then
> applying with operations that have no expected failure path (erase is a no-op
> when absent; insert-with-update_if_exists never fails). It is NOT a property
> of 'one launch'; partial application is impossible because apply cannot raise
> an expected error after validation succeeds."

The "NOT a property of one launch" sentence is there because the first
implementation got this wrong (it applied deletes, then PK-checked inserts
chunk by chunk, so a duplicate in chunk 2 left chunk 1 and all deletes applied
with no undo; an external reviewer caught it). The five functions:

- `rgi_stage_begin` (line 67) — reset staging.
- `rgi_stage_del` (line 68) — append keys to the staged delete set.
- `rgi_stage_upd` (line 69) — append key/value pairs to the staged update set.
- `rgi_stage_ins` (line 70) — append key/value pairs to the staged insert set.
- `rgi_stage_commit` (line 75) — validate, then apply, per the contract comment
  (lines 71-74): returns 1 + `*dup_key` on PK violation with the index left
  UNCHANGED; 0 on success; clears the staged set either way.
- `rgi_stage_abort` (line 76) — drop the staged set, index untouched.

Usage protocol (comment lines 64-66): begin; any number of stage calls in any
order (chunked — the worker's 4 MB channel forces large write sets through in
262,144-entry chunks); then exactly one of commit/abort. The engine does not
enforce the protocol with a state machine; the worker's PID-tagged staging and
single-lock-hold discipline enforce it one layer up. A modifier adding a second
concurrent stager at engine level would need to add that state machine here.

#### The paged snapshot pair (lines 84-86)

`rgi_snapshot_begin` (line 84): flush + **freeze** the live-key set, return
total count. `rgi_snapshot_page` (lines 85-86): emit up to `max` rows starting
at `off` against the frozen set; returns rows actually written, which can be
*less* than the span because "keys erased after freezing are skipped" (comment
line 83). The subtle cursor rule, spelled out at lines 80-83 and implemented in
the worker: the caller advances `off` by the SPAN (`min(max, total-off)`), not
by the returned row count — the return value only says how many of the span's
slots produced live rows. Advancing by the return value would silently re-read
or skip rows when concurrent erases punch holes in a page. This API exists
because the worker's fixed shared-memory window (262,144 entries) previously
caused silent truncation of large tables — a reviewer-caught bug; "no silent
truncation" in the comment (line 78) is the requirement it encodes.

### Invariants summary (header)

1. C-only types; opaque engine handle; `extern "C"` — the header must compile
   under both gcc-as-C and nvcc-as-C++ unchanged.
2. Every read API implies a flush (read-your-writes at engine level).
3. `rgi_stage_commit` leaves the index byte-identical to its pre-call state on
   any nonzero return.
4. Snapshot paging: caller advances by span, not by rows returned.
5. Values are 32-bit row-ids; keys are 8-byte and currently always two slices.

### How to modify safely

- Adding capability (WP6's `rgi_enumerate`, WP5's persistent opcodes if ever
  exposed): append new functions; never re-type or re-order existing ones.
- WP1 requires **no header change** — verify any WP1 patch leaves this file
  untouched.
- If a comment becomes false (e.g., WP6 removes the live set, changing the
  snapshot mechanism), update the comment in the same commit; the FDW authors
  treat these comments as the spec.

---

## 4. File: `engine/rgi_oltp_engine.cu` (345 lines)

### Purpose

THE production engine behind SQL. Every INSERT/UPDATE/DELETE/SELECT that a
Postgres client runs against `kv_rgi` ultimately executes inside this file,
via the worker calling the C ABI functions it defines. It is compiled two ways:
as `librgioltp.so` (the production shared library, by `build_all.sh`) and as
the standalone `rgi_bench` binary when `-DBUILD_BENCH` is set (by
`build_bench.sh`), in which case the `main` at lines 319-344 is included.

### Position in the architecture

Below the C ABI, above RGI. The file's responsibilities, in the order the
explainer (§14) lists them: (1) host-side write buffering realizing the batch
thesis; (2) one-time device staging-buffer allocation so no `cudaMalloc` ever
appears on a hot path; (3) batched point reads for pushdown; (4) the host-side
live-key shadow set that substitutes for RGI's missing scan API; (5) the paged
snapshot; (6) the validate-then-apply staged commit — "the engine half of
atomic commit."

### Design rationale

Three global decisions shape every function:

- **Synchronous, single-stream, default-stream CUDA.** Every chunk helper does
  `cudaMemcpy` (synchronous) → RGI host-API call (which launches `batch_kernel`)
  → `cudaDeviceSynchronize()`. No streams, no async copies, no events. This is
  deliberate bring-up engineering: the worker serializes all engine calls under
  one lock anyway (one outstanding bulk op system-wide), so overlap would buy
  nothing today, and synchronous semantics make the validate-then-apply
  ordering argument (§4.18) trivially true — when `find_chunk` returns, the
  results are in host memory, full stop. The known cost (H2D copy not
  overlapped with kernel) is part of the measured floor. WP2 (the coalescer)
  is where asynchrony would enter, and it enters in the worker, not here.
- **One set of reusable device staging buffers**, sized `BATCH_MAX`, allocated
  in `rgi_create`, freed in `rgi_destroy`. All traffic in both directions goes
  through them. This bounds device-side footprint at ~12 MB regardless of
  operation size, at the price of the chunking discipline below.
- **Host vectors as the unit of buffering and staging.** `std::vector` growth
  is amortized O(1) per append and the vectors live as long as the engine, so
  steady-state appends do not allocate. The engine is single-threaded by
  contract (only the worker thread calls it), so none of the host containers
  are synchronized.

### The chunking discipline (memorize this loop shape)

Every operation that can exceed `BATCH_MAX` uses the identical idiom:

```
total = <host vector size>
for (off = 0; off < total; off += BATCH_MAX) {
    chunk = min(total - off, BATCH_MAX)
    <copy chunk to device staging> ; <one RGI launch> ; <sync> [; <copy results back>]
}
```

```
host vector (any size, e.g. 2.5M entries)
+----------------------------+----------------------------+--------------+
|        chunk 0 (1M)        |        chunk 1 (1M)        | chunk 2 (.5M)|
+----------------------------+----------------------------+--------------+
        | memcpy H2D                  | memcpy H2D              | memcpy H2D
        v                             v                         v
device staging buffers (allocated ONCE, BATCH_MAX entries)
+--------------------------------------+
| d_keys : 2 * 1,048,576 uint32 (8 MB) |   <- reused by every chunk
| d_vals :     1,048,576 uint32 (4 MB) |
| d_out  :     1,048,576 uint32 (4 MB) |
+--------------------------------------+
        |  one RGI batch_kernel launch per chunk, then cudaDeviceSynchronize
        v
   gpu_chainhashtable  (the index itself; lives in slab pool + bucket array)
```

Occurrences of the loop: `rgi_flush` (94-97), `rgi_flush_unique` (119-122),
`rgi_find_many` (168-171), `rgi_snapshot` (185-188), `rgi_stage_commit`
validation (247-250), deletes (263-266), and apply (277-280). The persistent
engine repeats it with `PMAX_BATCH` (§5.6). If `BATCH_MAX` is ever changed,
note that nothing else needs to change — every loop derives from the constant —
but the 1M-entry dip in the throughput sweep (§8) suggests the optimum staging
size is below 1M, so a tuning pass is plausible.

### Walkthrough

#### File comment (lines 1-15)

Restates the storage substrate, the KV mapping, and — importantly — embeds the
canonical build command (lines 11-14): `nvcc -std=c++17 -arch=sm_89
--expt-extended-lambda --expt-relaxed-constexpr -maxrregcount=64 -Xcompiler
-fPIC -shared -I<RGI>/include rgi_oltp_engine.cu -o librgioltp.so`. The flags
matter (§9.4): `-maxrregcount=64` interacts with RGI's
`__launch_bounds__(128, 8)` to put the kernel exactly at Ada's 64K-register/SM
budget (128 threads x 8 blocks x 64 regs = 65,536) — the measured 63.6%/75.1%
occupancies are this configured ceiling, not an accident. Removing the flag
silently changes the performance profile of every published number.

#### `RGI_CK` (lines 28-36)

The CUDA error-check macro: on any non-success `cudaError_t` it prints
file:line and the error string to stderr and `abort()`s. Rationale for
abort-not-return: this library runs inside a Postgres background worker; a CUDA
error here (device lost, OOM, invalid context) means GPU state is unknown and
no recovery path exists in v1, so dying loudly and letting Postgres restart the
worker is strictly safer than limping with a corrupt index. Pitfall: `abort()`
in a Postgres process produces a crash-restart cycle of the whole cluster if
the error is persistent; a future hardening pass (post-prototype) would convert
this to an error return surfaced as a Postgres ERROR — but that requires every
ABI function to grow a failure path, i.e., an ABI change. Wrap *every* CUDA
runtime call in it; the file is consistent about this.

#### Type aliases (lines 38-40)

```cpp
using slab_t  = simple_slab_allocator<128>;
using debra_t = simple_debra_reclaimer<>;
using table_t = GpuHashtable::gpu_chainhashtable<slab_t, debra_t, 16>;
```

This is the single point where the RGI instantiation is chosen, and it is
repeated verbatim in all four CUDA benchmark files (`rgi_persist_engine.cu:49-51`,
`rgi_sweep.cu:14-16`, `rgi_prof.cu:11-13`, `rgi_prof2.cu:13-15`) — keep them in
lockstep or the benchmarks stop measuring the production configuration. The
three template arguments:

- `simple_slab_allocator<128>`: 128-byte slabs — one slab = one node = one
  cache line = one warp-coalesced load. The suffix nodes of §2.2 come from this
  same allocator.
- `simple_debra_reclaimer<>`: epoch-based reclamation; what makes `erase` safe
  under concurrent latch-free readers. Note for WP5: DEBRA's limbo-bag drain
  runs at `batch_kernel` exit — the launch-per-batch binding gets reclamation
  "for free" at every launch boundary, which is exactly what the never-exiting
  persistent kernel loses (§5.9).
- `16`: the cooperative-tile width (half-warp). Chosen by RGI's own tuning;
  changing it changes occupancy and every measured number.

Why chain hashtable and not cuckoo/Masstree (explainer §23, measured on this
GPU): chain insert 564 / find 1,450 / erase 499 Mop/s vs cuckoo 180 / 1,333 /
304; Masstree buys ordered scans the SQL surface does not yet expose. The swap
to Masstree for range pushdown is "a template parameter + ~50 wrapper lines, by
design" — these three lines are the template parameter in question.

#### `BATCH_MAX` (line 42)

`static constexpr uint32_t BATCH_MAX = 1u << 20;` — 1,048,576 entries of
device staging capacity, the chunking granularity for everything in this file.
Sizing logic: large enough that one chunk reaches the throughput-saturated
regime (saturation is at ~262K; 1M is comfortably past it), small enough that
the three staging buffers cost only ~12 MB of an 8 GB card. The sweep's
observed throughput dip at exactly B=1,048,576 (1,117.0 vs 1,248.7 Mop/s at
262K — cache/occupancy effects at full staging size) means the constant is not
sacred; what is sacred is that every loop and every buffer derives from it.

#### `RGI_INVALID` (line 43)

`0xFFFFFFFFu` — RGI's not-found sentinel for `find`. Consequence callers must
know: **a value of 0xFFFFFFFF is unrepresentable** (it is indistinguishable
from absent). The SQL layer stores row-ids, which never reach 2^32-1, so this
is documented and accepted. Tests must not use 0xFFFFFFFF as a value. The same
sentinel doubles as the persistent engine's stop sentinel for batch ids
(§5.2) — same bit pattern, two unrelated meanings; do not conflate them when
reading.

#### `struct RgiEngine` (lines 45-62) — the engine state, and its memory layout

```cpp
struct RgiEngine {
  slab_t   *host_alloc;                 // 46: RGI host-side allocator object
  debra_t  *host_reclaim;               // 47: RGI host-side reclaimer object
  table_t  *table;                      // 48: the chained hashtable handle
  uint32_t *d_keys;  /* 2*BATCH_MAX */  // 49: device key staging (8 MB)
  uint32_t *d_vals;  /* BATCH_MAX   */  // 50: device value staging (4 MB)
  uint32_t *d_out;   /* BATCH_MAX   */  // 51: device result staging (4 MB)
  std::vector<uint64_t> pend_k;         // 53: buffered insert/update keys
  std::vector<uint32_t> pend_v;         // 54: buffered insert/update values
  std::unordered_set<uint64_t> live;    // 55: live keys (snapshot enumeration)
  std::vector<uint64_t> stg_del_k;      // 57: staged transaction deletes
  std::vector<uint64_t> stg_upd_k, stg_upd_v;   // 58: staged updates
  std::vector<uint64_t> stg_ins_k, stg_ins_v;   // 59: staged inserts
  std::vector<uint64_t> snap_cache;     // 61: frozen live-key set for paging
};
```

The full state picture, host and device:

```
HOST (worker process heap)                      DEVICE (GPU global memory)
+---------------------------------------+      +------------------------------+
| RgiEngine                             |      | d_keys  [2 * 1,048,576 u32]  |
|  host_alloc  --> slab_t object -------+----> | slab pool (pool_ratio x free)|
|  host_reclaim--> debra_t object       |      |   - 128B chain nodes         |
|  table       --> table_t object ------+----> |   - 128B SUFFIX nodes (every |
|                                       |      |     key today, see §2.2)     |
|  d_keys, d_vals, d_out  (device ptrs) |      | bucket-head array            |
|                                       |      | d_vals  [1,048,576 u32]      |
|  pend_k  vector<u64>  \  write        |      | d_out   [1,048,576 u32]      |
|  pend_v  vector<u32>  /  buffer       |      +------------------------------+
|                                       |
|  live    unordered_set<u64>  <-- KEY-ONLY shadow of the index contents
|                                  (values NEVER stored host-side)
|  stg_del_k             \
|  stg_upd_k, stg_upd_v   |  transaction staging (one txn at a time,
|  stg_ins_k, stg_ins_v  /   owned by the worker's current commit)
|                                       |
|  snap_cache vector<u64>  <-- frozen copy of `live` during a paged snapshot
+---------------------------------------+
```

Field-by-field rationale and invariants:

- `host_alloc` / `host_reclaim` / `table` (46-48): heap-allocated because RGI's
  constructors do real work (device allocation) and the table constructor
  takes the other two **by reference** (line 84-85) — they must outlive the
  table and live at stable addresses. Destruction order in `rgi_destroy`
  (line 315) is the reverse: table, then reclaimer, then allocator. Violating
  that order makes the table destructor touch a freed allocator.
- `d_keys` is `2*BATCH_MAX` u32 because of the two-slice key mapping; `d_vals`
  and `d_out` are `BATCH_MAX` u32. Note `d_out` is distinct from `d_vals` so a
  find cannot clobber values being staged — though no current call path
  interleaves them, the separation costs 4 MB and removes a whole class of
  aliasing bugs.
- `pend_k`/`pend_v` (53-54): the write buffer. Parallel arrays, not a vector of
  pairs, because `pend_k.data()` and `pend_v.data()` are fed directly to
  `insert_chunk` with zero marshalling. Order is preserved (append-only),
  which matters: two buffered writes to the same key must apply in program
  order, and they do, because RGI processes a batch's duplicate keys in
  request order per its linearizability contract and the upsert semantics make
  the last write win. (Within ONE batch_kernel launch RGI does not guarantee
  inter-request ordering of same-key requests across tiles — which is why the
  FDW's transaction buffer collapses per-key state *before* anything reaches
  this engine; the engine-level buffer only sees one entry per key per
  statement in the transactional path. The legacy direct path can violate
  this; it is documented as test-only.)
- `live` (55): **the host-side key-only shadow set.** Exists solely because
  RGI has no scan/enumeration API (explainer §28). The contract has three
  clauses: (1) it shadows *keys only* — 8 bytes per live key, O(1) amortized
  maintenance per write; (2) **values always come from GPU truth** — every
  snapshot path re-finds the keys on the GPU and uses `d_out`, never a
  host-side value; (3) it may transiently contain keys the index no longer
  has (it is updated after the GPU operation, and snapshot paths defensively
  skip not-found keys at lines 191, 304). Alternatives rejected: shadowing
  values host-side would make reads CPU reads ("faking the whole point");
  writing a GPU enumeration kernel against RGI node internals is real work
  scheduled as **WP6**, which will delete this set entirely (after a
  one-release differential period where enumeration-vs-live-set equality is
  asserted). Until WP6 lands, every mutation path in this file MUST maintain
  `live` — grep for `e->live` when reviewing any engine patch; a missed
  update is an invisible bug that only snapshots expose.
- `stg_*` vectors (57-59): one transaction's staged write set. `stg_upd_v` and
  `stg_ins_v` are `vector<uint64_t>` (not u32) because they receive the ABI's
  u64 values verbatim; truncation happens once, at apply (lines 274-275).
  There is no PID/owner field here — single-stager ownership is enforced by
  the worker, not the engine (§3 staging block).
- `snap_cache` (61): the frozen key list for a paged snapshot. A `vector`
  (not a set) because pages index into it by offset (line 305). It persists
  between `rgi_snapshot_page` calls and is only rebuilt by the next
  `rgi_snapshot_begin`; there is no invalidation — a stale cache is harmless
  (erased keys are skipped; newly inserted keys are simply not in that
  snapshot, which is the intended frozen-view semantics).

#### `insert_chunk` (lines 65-70)

```cpp
static void insert_chunk(RgiEngine *e, const uint64_t *keys, const uint32_t *vals, uint32_t n)
```

The one place host writes become GPU state. Steps: (1) line 66 — synchronous
H2D copy of `n` u64 keys into `d_keys` (reinterpreted by RGI as `2n` u32
slices, §2.1); (2) line 67 — H2D copy of `n` u32 values into `d_vals`;
(3) line 68 — `e->table->insert<true>(e->d_keys, 2, nullptr, e->d_vals, n, 0,
/*update_if_exists=*/true)`: one `batch_kernel` launch. Template arg `<true>`
selects the concurrent/locking variant; positional args are (device key
slices, max_key_length=2, per-key lengths=nullptr i.e. all fixed at 2, device
values, count, offset 0, upsert=true); (4) line 69 — `cudaDeviceSynchronize()`.

The `update_if_exists=true` flag is the load-bearing choice in this file:
it makes every insert an **upsert**, which (a) lets `rgi_update` share the same
buffer and the same flush path as `rgi_insert`, and (b) makes the staged-commit
apply phase non-fallible — an insert that cannot collide cannot fail (§4.18).
Pitfall: changing this flag to false to "get real insert semantics" breaks
both `rgi_update` and the atomic-commit argument simultaneously. Real
uniqueness is enforced *above* this function, by validation.

Precondition: `n <= BATCH_MAX` — the function writes `n` entries into
fixed-size staging and does not check. All callers guarantee it via the
chunking loop; a new caller that does not will corrupt device memory past
`d_keys`. WP1 touchpoint: this function gains a per-key length computation
(`len[i] = (keys[i] >> 32) ? 2 : 1`), an H2D copy of the length array, and
passes it instead of `nullptr` — layout otherwise unchanged.

#### `find_chunk` (lines 72-77)

```cpp
static void find_chunk(RgiEngine *e, const uint64_t *keys, uint32_t *out, uint32_t n)
```

The one place GPU state becomes host-visible. Steps: H2D keys (73); line 74 —
`e->table->find<false, true>(e->d_keys, 2, nullptr, e->d_out, n)` — template
args `<false, true>` select the variant the production engine standardized on
(no-reclaim-context find with concurrent-safe traversal; the same pair appears
in every benchmark at `rgi_sweep.cu:39,41`, `rgi_prof.cu:26`,
`rgi_prof2.cu:34`); sync (75); D2H of `n` results into `out` (76), where
`out[i] == RGI_INVALID` means not found. Same `n <= BATCH_MAX` precondition,
same WP1 touchpoint as `insert_chunk`. Used by: `rgi_flush_unique` validation,
`rgi_lookup`, `rgi_find_many`, both snapshot paths, and `rgi_stage_commit`
validation — i.e., this is the hottest function in the production engine and
the direct beneficiary of WP1's removed suffix hop.

#### `rgi_create` (lines 80-90)

Steps: allocate the engine (81 — `new RgiEngine()`, value-initializing the
vectors/set); construct the slab allocator with `pool_ratio` (82); construct
the reclaimer (83); construct the table with allocator, reclaimer, `capacity`,
`fill_factor` (84-85) — this is where the bucket array and slab pool are
carved out of GPU memory; allocate the three staging buffers (86-88). No
further `cudaMalloc` happens for the life of the engine — "no cudaMalloc on
any hot path" is realized here. Pitfall: the function assumes a current CUDA
context (the worker initializes CUDA before calling); calling it from a
process that has never touched CUDA works (runtime implicit init) but calling
it from a *second* process while the worker holds the index gives you two
independent engines, not sharing — the entire worker architecture exists to
prevent that mistake.

#### `rgi_flush` (lines 92-101)

The plain (non-unique) flush. Walkthrough: read the pending count (93); the
chunking loop (94-97) pushes `pend_k/pend_v` through `insert_chunk` 1M at a
time; **after** all chunks land, fold every flushed key into `live` (98);
clear both pending vectors (99-100). Ordering note: `live` is updated after
the GPU writes complete, so a crash mid-flush leaves `live` missing keys the
index has — conservative in the safe direction for the only consumer
(snapshots under-enumerate rather than fabricate; and in practice RGI_CK
aborts the process on any CUDA failure anyway). Idempotent on empty buffers
(loop body never runs; clears are no-ops), which is why every read path can
call it unconditionally. Cost when empty: two size reads — effectively free,
so do not "optimize" the unconditional flush calls away.

#### `rgi_flush_unique` (lines 103-133)

The single-statement UNIQUE-enforcing flush; structurally a miniature of
`rgi_stage_commit` and worth reading first. Phases:

1. **Empty fast path** (105): zero pending → success.
2. **Validate (a): intra-batch duplicates** (108-115). A host
   `unordered_set` (`seen`, reserved at 2x for load factor, 109) absorbs each
   pending key; the first failed insertion (111) is a duplicate *within the
   statement*: report it via `*dup_key` (112), **discard the entire pending
   batch** (113 — clear both vectors), return 1. The discard is the contract:
   a failed statement's writes must not survive to contaminate a later flush.
3. **Validate (b): already-present keys** (118-128). Allocate a host result
   vector `tmp(total)` (118), batched-find ALL pending keys against the index
   via the chunking loop (119-122) — note this launches finds but **applies
   nothing**; any hit (`tmp[i] != RGI_INVALID`, 124) is a PK violation:
   report, discard, return 1 (125-127).
4. **Apply** (131): no conflict → delegate to `rgi_flush(e)`, which performs
   the upsert chunks and the `live` maintenance. Return 0.

Why validation order (a) before (b): (a) is free (host-only) and (b) costs GPU
launches; also (b)'s result array is only meaningful if each key appears once.
Subtlety: between (b) and apply, nothing else can run — the engine is
single-threaded under the worker's lock — so the check-then-act window is not
a race in this architecture. If the engine ever becomes multi-client at this
level (WP2 coalescer), this window is the first thing to re-audit.

#### `rgi_insert` (lines 135-138) and `rgi_update` (lines 140-144)

Both append `(key, (uint32_t)value)` to `pend_k`/`pend_v`. They are
*intentionally identical*: the flush path upserts, so an update is just an
insert that happens to hit an existing key, and "live-set add is idempotent"
(comment, 141) covers the bookkeeping. The distinction between INSERT and
UPDATE *semantics* (does the key have to exist? must it not exist?) lives in
the FDW's transaction buffer and the validation phases — never here. ~100 ns
per call; no CUDA.

#### `rgi_delete` (lines 146-153)

The non-transactional delete, and the file's one per-op-launch write path.
Steps: `rgi_flush` first (147) — the comment says why: "make sure key exists
in the index", i.e., a buffered insert of the same key must land before the
erase or the erase would be a no-op and the insert would resurrect the key at
the next flush. Then a single-key H2D (148-149), one erase launch (150 —
`erase<true, true>(d_keys, 2, nullptr, 1)`), sync (151), and `live` removal
(152). Erase of an absent key is a defined no-op in RGI — the property the
staged commit's apply phase relies on (§4.18). Performance: this function pays
the full dispatch floor per delete; it is acceptable only because the
transactional path batches deletes through staging instead. If a profiler
shows this function hot, the caller is using the wrong API.

#### `rgi_lookup` (lines 155-162)

Point read: flush (156), `find_chunk` with n=1 into a stack local initialized
to `RGI_INVALID` (157-158), translate sentinel to found/not-found (159-161).
The `out_value` null-check (160) lets callers probe existence cheaply. One
launch per call by construction — the SQL point-SELECT cost (0.95 ms wall,
dominated by Postgres layers, with ~20 µs of it this round trip) sits on top
of this.

#### `rgi_find_many` (lines 164-176)

The pushdown workhorse. Flush (166); allocate `tmp` with a guard for n=0
(167 — `n ? n : 1`, avoiding a zero-size vector data() edge); chunked finds
(168-171); then a host loop (172-175) splitting each `tmp[i]` into
`found[i] = (tmp[i] != RGI_INVALID)` and `out_values[i] = (uint64_t)tmp[i]`.
Note `out_values[i]` is written even when not found (it carries the sentinel,
widened); callers must consult `found[]`, not sniff values. Capacity contract:
the caller owns both output arrays sized `n`; the worker's bulk channel caps
`n` at 262,144 per request, well under `BATCH_MAX`, but the function itself
handles arbitrary `n` via chunking — keep it that way so the engine never
inherits the transport's limits.

#### `rgi_snapshot` (lines 178-199)

The one-shot full snapshot (`SELECT *` without paging; today mainly the
bench's path — the worker uses the paged pair). Steps: flush (179); size from
the live set (180); `malloc` both out-arrays with a 1-element floor for n=0
(181-182 — `malloc`, not `new`, because the caller frees with `free()` across
the C ABI); materialize the set into a contiguous vector (183) because
`find_chunk` needs a flat array; chunked finds over ALL live keys (185-188);
then the compaction loop (189-195): skip `RGI_INVALID` entries — "erased
after enumeration" (191) covers the legacy path's ability to erase without
perfect live-set discipline and any transient set/index divergence — and pack
surviving `(key, value-from-GPU)` pairs. Returns the packed count `live_n`,
which can be less than `live.size()`. Lines 196-197 tolerate null out-params
by freeing internally. The defining property, worth restating because it is
the live-set contract: **the keys come from the host shadow, the values come
from the GPU** — the snapshot is a batched read of GPU truth, not a dump of
host state. WP6 replaces the key source (host set → GPU enumeration kernel)
while keeping this exact ABI shape.

#### `erase_chunk` (lines 202-206)

The third chunk helper, placed with the staging code because the staged commit
is its only batched caller. Mirror of `insert_chunk` without values: H2D keys
(203), `erase<true, true>(d_keys, 2, nullptr, n)` (204), sync (205). Same
`n <= BATCH_MAX` precondition, same WP1 touchpoint (and WP1's pitfall list
singles this one out: an erase probing with the wrong slice length **misses
silently** — no error, key survives). Does NOT touch `live`; the caller does,
deliberately, so the helper stays a pure GPU operation.

#### `rgi_stage_begin` (lines 208-212) and `rgi_stage_abort` (line 214)

`rgi_stage_begin` clears all five staging vectors. `rgi_stage_abort` is
literally `rgi_stage_begin(e)` (214) — aborting a staged transaction IS
resetting the staging, because nothing staged has touched the GPU yet. That
one-liner is the cheapest possible proof of "ROLLBACK costs zero GPU work."
Note `clear()` keeps vector capacity, so a steady stream of transactions
reuses the same allocations.

#### `rgi_stage_del` (216-218), `rgi_stage_upd` (220-222), `rgi_stage_ins` (224-226)

Pure host-side appends of the caller's arrays into the staging vectors;
loops rather than bulk `insert()` for symmetry, no behavioral difference.
Callable any number of times in any order ("chunked" — the worker pushes
262,144-entry chunks through its 4 MB channel, so a 300,001-row commit arrives
as two stage calls per class). No validation here, by design: validation must
see the COMPLETE staged set, so it can only run at commit.

#### `rgi_stage_commit` (lines 228-286) — line-by-line

This is the most important function in the project: it is where the
correctness claim of the whole system ("a transaction either fully applies or
leaves the index byte-identical") is implemented, and it was rebuilt once
after an external reviewer demonstrated the first version could partially
apply. Read it with the header contract (header lines 56-66) open.

**Line 228** — signature: `int rgi_stage_commit(RgiEngine *e, uint64_t *dup_key)`.
Returns 0 on success; 1 on PK/UNIQUE violation with `*dup_key` set and the
index UNCHANGED. The staged set is cleared on every path.

**Line 230 — `rgi_flush(e);`** Pre-step: flush any *buffered* (non-staged)
writes first. Two reasons, the comment (229) gives the critical one: "make
sure prior buffered writes are visible before we validate against the index."
If a same-session statement-level write (legacy path) sat in `pend_k`, the
validation find below would miss it and a staged insert of the same key would
pass validation falsely. Second reason: the apply phase reuses `insert_chunk`
and the staging buffers; interleaving an unflushed pending batch after apply
would reorder writes across the commit boundary. In the production
transactional flow `pend_k` is empty here (the FDW routes everything through
staging), so this is a defensive no-op that makes the function correct even
for mixed-API callers.

**Line 232 — `const uint32_t ni = e->stg_ins_k.size();`** The staged-insert
count, hoisted because validation only concerns inserts. Deletes and updates
need no validation by construction: a staged delete of an absent key is legal
(no-op), and a staged update's "key must exist" semantics were already
enforced by the FDW's buffer against its read-your-writes view — and even if
an update key vanished concurrently, applying it as an upsert is the
documented last-writer-wins behavior at this isolation level (~Read
Committed + RYW; no write-write conflict detection until WP3's OCC).

**VALIDATE phase (lines 234-257) — mutates nothing.** The banner comment
(234) is normative: "every error condition is checked here." There are
exactly two error conditions in the system, both UNIQUE violations:

*Validation (a): intra-set duplicates, lines 236-244.* Host-only. A
`std::unordered_set<uint64_t> seen` reserved at `2*ni` (237-238) absorbs each
staged insert key in order (239-240); the first key whose insertion fails
(`.second == false`) is a duplicate *within this transaction's insert set* —
e.g., `INSERT (7,1); INSERT (7,2);` in one txn if the FDW buffer had not
already collapsed it (the engine does not assume the caller pre-deduplicated).
On hit: report through `dup_key` if non-null (241), `rgi_stage_abort(e)` (242)
— which clears all five vectors, leaving the index untouched because nothing
has been launched yet — and return 1 (243). Cost: O(ni) host time, zero GPU.

*Validation (b): already-present keys, lines 245-256.* The GPU half. A host
result vector `tmp(ni)` (246); the chunking loop (247-250) batched-finds ALL
staged insert keys against the index — for a 300,001-key insert set this is
two `find_chunk` calls (1M chunk size), i.e., two launches; **finds mutate
nothing.** Then the scan (251-256): any `tmp[i] != RGI_INVALID` means key
`stg_ins_k[i]` already exists in the committed index → report it (253), abort
the staging (254), return 1 (255). Because (a) already proved the insert keys
pairwise distinct, the first hit is well-defined and deterministic.

Note what validation does NOT check: it does not consult `live` (the shadow
set is for enumeration only, never for correctness decisions — GPU truth
decides existence), and it does not validate staged deletes/updates (see
line-232 note). If a future change adds a new fallible operation class to the
apply phase, the structural rule is: move its failure condition up here, or
the atomicity argument dies.

**APPLY phase (lines 259-282) — no expected failure path.** The banner (259)
is the other half of the contract. After validation succeeds, the function
performs only operations that cannot raise an *expected* error (a CUDA error
still aborts the process via RGI_CK — crash-stop, not partial-commit;
"expected failure" means PK violations, the only application-level error this
engine defines).

*Apply deletes, lines 261-268.* Chunked `erase_chunk` over `stg_del_k`
(262-266). Safe because "erase is a no-op for absent keys" (260) — a staged
delete can never fail. Then host bookkeeping: remove each deleted key from
`live` (267). Deletes go FIRST deliberately: a transaction that deletes key K
and re-inserts K (delete-then-reinsert collapses to an upsert in the FDW
buffer, but mixed shapes can still stage K in both sets across chunk
boundaries) must end with K present; erase-then-upsert gives that terminal
state, upsert-then-erase would not.

*Apply updates + inserts as one upsert stream, lines 269-282.* Lines 271-275
concatenate updates then inserts into combined arrays `ak`/`av` (reserving
exact capacity, 272-273; truncating values to u32 at 274-275 — the single
truncation point for staged values). Lines 276-280: the chunking loop drives
`insert_chunk` over the combined stream — with `update_if_exists=true`, an
update key (exists) overwrites and an insert key (validated absent) inserts;
neither can fail (269: "insert with update_if_exists never fails
(validated)"). Order within the stream (updates before inserts) is irrelevant
because validation proved the two key sets behave independently; the
concatenation exists to minimize launches (one chunked stream instead of two).
Line 281 folds every applied key into `live` — insertion is idempotent for
update keys already present.

**Line 284 — `rgi_stage_abort(e);`** Clear the staged set on the success path
too ("clears the staged set either way" — header line 74), so a stale commit
can never be replayed. **Line 285 — return 0.**

#### The staged-commit data flow, end to end

```
 Postgres backend                worker                engine (this file)            GPU
 ----------------                ------                ------------------            ---
 PRE_COMMIT callback
  classify txn buffer
  into del/upd/ins arrays
        |  LWLock(bulk_lock) acquired -- held across the WHOLE protocol
        |-- TXN_BEGIN ------------>  rgi_stage_begin     clear stg_* vectors
        |-- STAGE_DEL (<=262144) -->  rgi_stage_del      append stg_del_k
        |-- STAGE_UPD chunks ----->  rgi_stage_upd       append stg_upd_k/v
        |-- STAGE_INS chunks ----->  rgi_stage_ins       append stg_ins_k/v
        |-- COMMIT --------------->  rgi_stage_commit
        |                              rgi_flush(230)            (usually no-op)
        |                            VALIDATE
        |                              (a) host dup scan (236-244)      [no GPU]
        |                              (b) find ALL ins keys (245-256) --> find launch(es)
        |                                   any hit? -> abort staging, return dup ---+
        |                            APPLY  (only if validation passed)             |
        |                              erase chunks (261-268) ----------> erase launch(es)
        |                              upsert chunks (269-282) ---------> insert launch(es)
        |                              live set -= dels, += upds+inss               |
        |<-- ok / dup_key ----------  return 0 / 1  <--------------------------------+
        |  LWLock released
  ok  -> transaction commits
  dup -> ereport(ERROR, unique_violation)  -> Postgres aborts the txn
         (GPU index is byte-identical to pre-BEGIN; regression-tested with a
          300,001-row commit whose duplicate is planted in the LAST chunk)
```

The atomicity argument, compressed: the only fallible step (validation) runs
before the first mutation; every mutating step is non-fallible; therefore no
execution can stop between "some mutations applied" and "an error returned."
It is **not** "one kernel launch" — apply spans multiple launches — and the
header comment says so explicitly because the distinction is exactly what the
first, broken implementation got wrong.

#### `rgi_snapshot_begin` (lines 289-293)

Flush (290), then freeze: `snap_cache.assign(live.begin(), live.end())` (291)
copies the live-key set into the stable, indexable vector; return its size
(292). From this instant the page sequence is defined against an immutable key
list, so a multi-page `SELECT *` sees one consistent key population even if
later transactions mutate the index mid-scan (their effects show up only as
skipped rows, never as torn pages or duplicates). The worker holds `bulk_lock`
across begin + all pages, so in production nothing actually mutates mid-scan;
the freeze makes the engine correct even without that courtesy.

#### `rgi_snapshot_page` (lines 295-310)

Inputs: frozen-set offset `off`, page capacity `max`, caller-owned out-arrays.
Steps: bounds check against the frozen total (297-298); compute
`span = min(max, total - off)` (299); ONE `find_chunk` over
`snap_cache[off .. off+span)` (300-301) — note pages are capped by the
worker at 262,144 (< BATCH_MAX) so a single chunk suffices, and the function
correctly does not loop (a span above BATCH_MAX would violate `find_chunk`'s
precondition — if a future caller raises page size past 1M, add the chunk loop
here); compaction (302-308) skipping `RGI_INVALID` ("erased after the set was
frozen", 304); return rows written `m <= span`. The cursor contract from the
header (§3): the CALLER advances `off` by `span`, not by `m` — this function
cannot communicate `span` back (the caller computes the same min), and
advancing by `m` would re-scan the tail of a page that contained erased keys.

#### `rgi_destroy` (lines 312-316)

Null-tolerant (313). Frees the three device buffers (314), then host objects
in reverse construction order: table, reclaimer, allocator, engine (315). The
order matters (the table holds references to the other two — §4 struct notes).
Does not call `cudaDeviceReset()` — the worker process may have other CUDA
state (it does not today, but the engine should not assume).

#### The `BUILD_BENCH` microbenchmark (lines 319-344)

Compiled only with `-DBUILD_BENCH` (build_bench.sh line 17 → `rgi_bench`).
A smoke-test-plus-headline-number harness, NOT the precision instrument
(that is `rgi_sweep`): `main` takes N (default 5,000,000; line 326), creates
an engine at capacity N with fill 2.0 / pool 0.5 (327), then measures three
things through the FULL wrapper path (buffering, live set, chunking):

1. **Buffered insert + flush** (328-332): N `rgi_insert` calls then one
   `rgi_flush`, timed together — so the printed Mop/s includes the host-side
   vector appends and the live-set fold, i.e., it is the wrapper's insert
   number, deliberately lower than RGI's raw kernel ceiling.
2. **Full snapshot** (333-337): `rgi_snapshot` over all N rows, timed —
   exercises set materialization + chunked finds + compaction.
3. **Point lookup** (338-339): `rgi_lookup(12345)` correctness print.

`now_s` (321-324) is the steady-clock helper duplicated in every bench file.
WP1's task list extends this main to print bytes/entry via RGI's `validate()`
space stats before/after — when doing that, keep the three existing prints
stable; scripts grep them.

### Invariants summary (`rgi_oltp_engine.cu`)

1. **Single-threaded engine.** Exactly one thread (the worker's) calls any
   function; no host container is synchronized. Concurrency is the worker's
   lock's problem.
2. **Read-your-writes:** every read entry point (`rgi_lookup` 156,
   `rgi_find_many` 166, `rgi_snapshot` 179, `rgi_snapshot_begin` 290) calls
   `rgi_flush` first; `rgi_stage_commit` (230) and `rgi_delete` (147) also
   flush before acting.
3. **Chunk preconditions:** `insert_chunk`/`find_chunk`/`erase_chunk` require
   `n <= BATCH_MAX`; only the standard loop idiom may call them.
4. **Validate-before-mutate:** `rgi_stage_commit` performs no GPU mutation
   before line 259; every expected failure returns before line 259; the apply
   phase contains only non-fallible operations (no-op erase, validated upsert).
5. **Upsert everywhere:** every `insert` call in this file passes
   `update_if_exists=true`; nothing in the apply path may depend on
   insert-failure signaling.
6. **Live-set discipline:** every path that changes index membership updates
   `live` (flush 98, delete 152, commit 267/281); every consumer of `live`
   tolerates over-approximation by skipping `RGI_INVALID` finds (191, 304);
   `live` is never consulted for a correctness decision.
7. **Values from GPU truth:** no host structure ever stores a value; all
   snapshot/read values come from `d_out`.
8. **Sentinel:** `0xFFFFFFFF` is not a legal value (it is the not-found
   sentinel).
9. **Two-slice keys:** every RGI call passes `(…, 2, nullptr, …)` —
   uniformly, until WP1 replaces `nullptr` with a computed length array in
   ALL FOUR call sites (68, 74, 150, 204) plus the persistent kernel; partial
   adoption is silent data loss (a key stored 1-slice and probed 2-slice, or
   vice versa, simply misses).
10. **Staged set cleared on every commit/abort path** (242, 254, 284, 214) —
    no staged state survives a completed protocol round.

### How to modify safely (`rgi_oltp_engine.cu`)

- **WP1 (small-key fast path)** — the planned change to this file. Touch
  exactly `insert_chunk`, `find_chunk`, `erase_chunk` (add a device length
  array, computed by `len = (key >> 32) ? 2 : 1`); leave staging layout
  (2 slots/key), the C ABI, and all staging/commit logic untouched. Re-run
  `rgi_sweep`, `rgi_bench`, the ncu pair, and the full SQL suite; add the
  2^32-straddling SQL test with DELETE coverage. A >2% regression on either
  ceiling is stop-and-investigate per WP1's acceptance criteria.
- **WP6 (GPU enumeration)** — adds `rgi_enumerate` (new ABI function) and a
  parallel enumeration kernel in this file (grid-stride over buckets, one
  tile per chain, atomic output cursor); keeps the paged-snapshot ABI shape;
  deletes `live` only after a differential period proving
  enumeration == live-set. Until that lands, never remove a `live` update.
- **WP2/WP3 (coalescer, OCC)** — live above this file (worker / FDW), but
  OCC's version validation is designed to run inside the existing staged
  commit critical section; if it lands here, it must be added to the VALIDATE
  phase (before line 259), never to apply.
- Any new mutation path must (a) chunk via the standard idiom, (b) maintain
  `live`, (c) be either validated-before or non-fallible, (d) keep the upsert
  flag unless it also rewrites the atomicity argument in the header comment.
- Performance edits: do not add per-op launches; do not allocate device
  memory outside `rgi_create`; keep `-maxrregcount=64` in every build of this
  file or the published occupancy/SOL numbers stop being reproducible.

---

## 5. File: `engine/rgi_persist_engine.cu` (346 lines)

### Purpose

The v1 **persistent-kernel binding** for the same RGI chained hashtable: one
resident kernel that never exits, fed batches through a mapped-memory doorbell
instead of per-batch kernel launches. Built 2026-06-09 as a standalone binary
(`rgi_persist`) that benchmarks BOTH bindings (launch-per-batch and doorbell)
on the same table in the same process — the head-to-head that produced the
"persistent is 5-6x worse on WSL2/WDDM" finding. It is strictly **additive**:
"nothing in rgi_oltp_engine.cu / the FDW / the worker changes" (header
comment, lines 2-3), and it calls only RGI's **public device API**
(`cooperative_insert` / `cooperative_find`) — "RGI source is NOT modified"
(line 17).

### Position in the architecture

This is the GB-class dispatch model "prototyped on PCIe" (line 5). On a
coherent-interconnect machine (Grace-Hopper/Blackwell class), the doorbell
becomes a cache-line write the GPU observes through coherence at ~0.5 µs,
and the resident kernel becomes the primary binding; on this laptop
(WSL2/WDDM), the doorbell crosses PCIe through mapped memory at 75-90 µs and
loses to the 13-28 µs launch. The file exists so that when bare-metal or
GB-class hardware appears (WP5/WP7), the binding is already written,
validated, and instrumented — hardware days get spent measuring, not coding.

### Design rationale

The header comment (lines 1-37) is a compressed design document; its claims,
expanded:

- **One resident kernel at exact full occupancy (lines 6-7).** Blocks of a
  persistent grid rendezvous with each other every batch (the arrival
  counter); if even one block of the grid is not co-resident on an SM, the
  resident blocks wait forever for an arrival that cannot happen until they
  exit — **deadlock**. So the grid is sized to exactly what the occupancy
  calculator says fits (§5.5), never a block more. On the dev laptop this is
  216 resident blocks (the program prints the blocks/SM x SM-count
  factorization at startup, lines 202-203).
- **NVMe-style submission (lines 7-9):** payload is staged H2D on a copy
  stream BEFORE the doorbell rings — "payload DMA, doorbell MMIO." The
  doorbell line carries only control words; requests never travel through
  mapped memory.
- **TWO-LEVEL doorbell (lines 10-12):** mapped (zero-copy) memory reads from
  a kernel traverse the PCIe bus every poll. 216 blocks polling the mapped
  line would multiply that traffic 216x and hammer the host line. So ONLY
  block 0 polls the mapped line; it republishes `{batch_id, count}` into
  ordinary device memory (`g_pub`), which the other 215 blocks poll out of L2
  — cheap, on-die. Stop is signaled in-band as sentinel batch id
  `0xFFFFFFFF`.
- **Completion via arrival counter (lines 13-14):** each block atomically
  bumps `g_arrive` when done; block 0 waits for `gridDim.x` arrivals, resets
  the counter, fences, and writes `done_id` to the mapped line for the host.
- **v1 scope: INSERT + FIND only (lines 21-22).** Erase is excluded because
  DEBRA reclamation drains its limbo bags at `batch_kernel` **exit** — and
  this kernel never exits, so deferred frees would accumulate forever.
  Restructuring drains around batch quiescent points is the designed v2 item
  (WP5 design step 2; see §5.9).
- **Toy-engine lessons carried over verbatim (lines 24-29):** the three CUDA
  bugs the earlier toy engine hit are pre-fixed here: (1) WDDM launch flush
  (event record+query after launch, or the kernel never actually starts);
  (2) capture the batch id ONCE per batch (the lost-doorbell TOCTOU race);
  (3) non-blocking streams ONLY (a default-stream memcpy would synchronize
  against the never-ending kernel — deadlock by API semantics).
- **The co-residency hazard (lines 30-33):** while the persistent kernel is
  resident it occupies every block slot on every SM, so **no other kernel in
  the process can launch** — a launch would wait for a free slot forever and
  everything deadlocks. The bench therefore runs the launch-mode sweep FIRST
  (phase 1), then starts the resident kernel (phase 2). This ordering
  constraint propagates to WP5's bare-metal campaign script ("launch sweep
  first") and to any future integration: the production worker could not
  simply also launch ordinary kernels while a persistent grid is live.

### Walkthrough

#### Aliases and constants (lines 49-58)

Lines 49-51 repeat the production RGI instantiation exactly (§4 aliases —
same allocator, reclaimer, tile width, so the comparison is apples-to-apples).
Lines 52-53 add two device-side types the launch binding never needed:
`alloc_ctx_t = table_t::device_allocator_context_type` and
`alloc_inst_t = slab_t::device_instance_type` — the kernel constructs its own
allocator context per tile (line 95) because it IS the kernel now; in the
launch binding, RGI's `batch_kernel` did that internally.

- `TILE 16` (55): cooperative tile width; must equal the table's template
  tile parameter (the `16` in line 51) or cooperative calls misbehave.
- `BLOCK_SIZE 128` (56): matches RGI's `batch_kernel` geometry.
- `PMAX_BATCH (1u << 20)` (57): the persistent twin of `BATCH_MAX` — device
  request-buffer capacity and the host submit chunking granularity (§5.6).
  Same value, distinct constant: the two engines do not share headers, and a
  modifier resizing one must consciously decide about the other.
- `STOP_SENTINEL 0xFFFFFFFFu` (58): the in-band kernel-retirement batch id.
  Numerically identical to `RGI_INVALID` but unrelated in meaning (§4 note).

`CK` (60-68) is RGI_CK with `exit(1)` instead of `abort()` — fine for a
standalone bench; do not copy it back into library code (no core dump).

#### `struct PersistCtl` (lines 71-77)

```cpp
struct PersistCtl {
  volatile uint32_t batch_id;   // host bumps to publish a batch
  volatile uint32_t done_id;    // block 0 sets when the grid finished it
  volatile uint32_t count;      // requests in this batch
  volatile uint32_t stop;       // host sets 1 to retire the kernel
  uint32_t pad[12];             // pads the struct to 64 bytes
};
```

"The only thing the host and block 0 share" (line 70). It lives in
`cudaHostAllocMapped` pinned host memory; the device pointer (`d_ctl`) aliases
the same physical line. The 64-byte pad keeps the four control words on one
cache line / one PCIe transaction and prevents false sharing with anything
the allocator might place adjacent. `volatile` defeats compiler caching on
both sides; ordering is provided separately by explicit fences (§5.4) —
volatile alone orders nothing, a point the fence placement diagram below
makes precise. The protocol uses monotonically increasing `batch_id` /
`done_id` values (not a flag), so a stale read is always distinguishable from
a fresh one and the host's completion wait (`done_id != cur_batch`, line 231)
cannot be fooled by an old value.

#### `enum { POP_FIND = 0, POP_INSERT = 1 }` (line 79)

The per-request opcode, carried in `d_types` (one byte per request). v1's
whole opcode space; v2 adds erase (WP5).

#### `rgi_persistent_kernel` (lines 82-165) — the resident kernel

Signature (82-90): `__global__ void __launch_bounds__(BLOCK_SIZE, 8)` taking
the table and allocator instance **by value** (kernel parameters — RGI handles
are designed to be passed this way), the mapped control pointer `ctl`, the
two-word device publication buffer `g_pub`, the arrival counter `g_arrive`,
and the four request buffers (`d_types`, `d_keys` with 2 slices/key,
`d_vals`, `d_out`). The `__launch_bounds__(128, 8)` mirrors RGI's own kernel
so occupancy math matches the production binding.

*Tile setup (92-95).* `cg::block_tile_memory<BLOCK_SIZE>` shared storage, a
cooperative-groups block handle, a 16-wide tiled partition, and the per-tile
allocator context `alloc_ctx_t allocator{alloc_inst, tile}` — the preamble
RGI's `batch_kernel` performs, replicated here because this kernel replaces
it.

*Per-block doorbell state (97-98).* `__shared__ uint32_t s_batch, s_count`
(the value thread 0 broadcasts to its block) and the per-thread register
`seen = 0` — "which batch id this block last completed." Every thread
maintains `seen` (line 163 executes for all threads), so the level-2 wait
condition is uniform across the block.

*The outer infinite loop (100).* One iteration = one batch (or retirement).

*Level 1 — block 0, thread 0 polls the mapped line (102-109).*

- Line 103: spin `while (ctl->batch_id == seen && ctl->stop == 0)` with
  `__nanosleep(128)` between polls. Each read of `ctl->batch_id` is a PCIe
  round trip — this is the expensive poll that the two-level design confines
  to ONE thread in the whole grid. The nanosleep throttles bus traffic and
  frees scheduler slots.
- Line 104: `__threadfence_system()` — **the acquire fence.** It orders the
  doorbell read before all subsequent reads from this thread across the
  system scope, pairing with the host's release fence (line 228): if this
  thread observed the new `batch_id`, then the payload H2D (which the host
  completed BEFORE ringing) and `ctl->count` are guaranteed visible. Without
  it, the GPU could legally read a stale `count` or stale payload bytes.
- Line 105: resolve the batch id: `b = ctl->stop ? STOP_SENTINEL :
  ctl->batch_id` — stop folds into the same publication path as a normal
  batch, so the other 215 blocks need no second exit mechanism.
- Lines 106-108: republish to device memory **in the safe order**: write
  `g_pub[1] = count` first, `__threadfence()` (device-scope — L2 is the
  audience now, not the host), then `g_pub[0] = b`. Readers poll `g_pub[0]`;
  by the time they see the new batch id, the fence guarantees the count is
  already there. Writing in the other order (or dropping the fence) lets a
  level-2 reader pair the new batch id with the previous batch's count —
  silently processing the wrong number of requests.

*Level 2 — every block's thread 0 polls device memory (111-117).* The
do-while at line 113 reads `g_pub[0]` until it differs from `seen` (or is the
sentinel, which always breaks immediately — even block 0 takes this path for
uniformity), with a 64 ns sleep per poll. These polls hit L2 — on-die, no bus
traffic, which is the entire point of the two-level structure. Then thread 0
stores the batch id and count into shared memory (115-116) for the block.

*Capture-once + broadcast (118-121).* `__syncthreads()` (118) publishes the
shared values to all threads; line 119 copies `s_batch` into the register
`my_batch` with the comment "captured ONCE (race-fix pattern)" — this is toy
lesson 2 materialized. The kernel uses `my_batch` for everything downstream
(the completion signal at 159 and the `seen` update at 163); it never re-reads
`g_pub[0]` or `ctl->batch_id` within the batch. The race it prevents: after
block 0 signals done, the host may instantly publish batch N+1; a block that
re-read the current batch id at end-of-batch could capture N+1 and record it
as already-seen — dropping a batch and deadlocking the host's completion
wait. Single large batches never trip it; thousands of rapid small submits
(the OLTP shape) trip it reliably. Line 120: if `my_batch == STOP_SENTINEL`,
`return` — every block exits its infinite loop, the kernel terminates, and
the host's `cudaStreamSynchronize(kstream)` (line 244) completes. Line 121
snapshots the count.

*The ballot-queue execution loop (123-148).* This reproduces RGI's own
`batch_kernel` request-distribution pattern, grid-strided over the batch:

- Line 124: `span` rounds `count` up to a multiple of `BLOCK_SIZE`, so every
  thread of every block executes the same number of loop iterations —
  required because the loop body contains tile-cooperative operations
  (`ballot`, `shfl`) that need full-tile participation even when a thread has
  no task.
- Lines 125-126: grid-stride: thread `tid` covers requests
  `tid, tid + gridDim.x*BLOCK_SIZE, …`. With 216 blocks x 128 threads =
  27,648 threads, a 1M-request batch takes ~38 strides.
- Lines 127-131: each thread loads its request (predicated on `tid < count`):
  key pointer (`d_keys + 2*tid` — two slices), opcode, value; `out`
  initialized to the not-found sentinel.
- Lines 132-146: the ballot queue. `wq = tile.ballot(task)` is a 16-bit mask
  of lanes holding live requests; while any remain, `__ffs` picks lane `r`,
  `shfl` broadcasts that lane's key pointer/opcode/value to all 16 lanes, and
  the WHOLE TILE cooperatively executes that single request: line 139
  `table.cooperative_insert<true>(ck, 2, cv, tile, allocator, true)` for
  inserts (upsert semantics matching the production engine), line 141
  `table.cooperative_find<true, true>(ck, 2, tile, allocator)` for finds —
  the result is returned to every lane but only lane `r` keeps it (142).
  Lane `r` retires its task (144) and the ballot repeats. One-request-per-tile
  with all lanes cooperating is RGI's divergence-elimination trick: pointer
  traversals are uniform within a tile and concurrency comes from many tiles
  in flight, not from divergent threads.
- Line 147: predicated result writeback `d_out[tid] = out`. Inserts write the
  sentinel (their `out` is never set) — harmless, the host passes
  `out=nullptr` for insert-only batches.

Note the `2` in both cooperative calls: the same two-slice mapping as the
production engine, and WP1's persistent-side touchpoint — the kernel will
compute `len = (hi_slice == 0) ? 1 : 2` from the loaded key (WP1 design step
2) since there is no host-prepared length array here.

*Completion (150-163), with the fence placement that makes it correct:*

- Line 151: `__threadfence()` — publishes this thread's `d_out` writes to
  device scope BEFORE the block announces arrival. Without it, block 0 could
  observe all arrivals while some block's result writes are still in flight.
- Line 152: `__syncthreads()` — all threads of the block reach the fence
  before thread 0 arrives on the block's behalf.
- Lines 153-154: thread 0 of every block does `atomicAdd(g_arrive, 1)`.
- Lines 155-158 (block 0's thread 0 only): spin until
  `atomicAdd(g_arrive, 0)` (an atomic read) reaches `gridDim.x`; reset the
  counter with `atomicExch` for the next batch; then
  `__threadfence_system()` — **the release fence toward the host**: orders
  all device writes (every block's `d_out`, the counter reset) before the
  mapped-line write that follows.
- Line 159: `ctl->done_id = my_batch` — the completion doorbell, the single
  PCIe write the host is spinning on.
- Lines 162-163: `__syncthreads()` then `seen = my_batch` in every thread —
  the block arms itself for the next iteration. The level-1 spinner's next
  `ctl->batch_id == seen` comparison now correctly waits for a NEW batch.

#### The two-level doorbell, end to end, with every fence

```
HOST (persist_submit)                |  GPU
-------------------------------------+---------------------------------------------
 memcpyAsync types/keys/vals         |
   on cstream (223-225)              |
 cudaStreamSynchronize(cstream) 226  |   (payload now resident in d_types/d_keys/d_vals)
 h_ctl->count = c            (227)   |
 atomic_thread_fence(release)(228)   |   <-- pairs with kernel line 104
 h_ctl->batch_id = ++cur     (230)   o==== PCIe mapped line ====o
                                     |  BLOCK 0, THREAD 0 (level 1)
 spin: done_id != cur (231)          |   poll ctl->batch_id      (103)  [PCIe read/poll]
   .                                 |   __threadfence_system()  (104)  ACQUIRE
   .                                 |   g_pub[1] = count        (106)  [device mem]
   .                                 |   __threadfence()         (107)  count-before-id
   .                                 |   g_pub[0] = batch        (108)
   .                                 |  ALL 216 BLOCKS, THREAD 0 (level 2)
   .                                 |   poll g_pub[0] != seen   (113)  [L2, cheap]
   .                                 |   s_batch, s_count        (115-116)
   .                                 |  __syncthreads            (118)
   .                                 |  my_batch captured ONCE   (119)  race fix
   .                                 |  ballot-queue execution   (123-148)
   .                                 |  __threadfence()          (151)  d_out before arrival
   .                                 |  atomicAdd(g_arrive,1) x216 (154)
   .                                 |  block 0: wait 216 arrivals (156)
   .                                 |  atomicExch(g_arrive,0)   (157)
   .                                 |  __threadfence_system()   (158)  RELEASE toward host
 done_id == cur  -> exit spin        o==== ctl->done_id = my_batch (159) ====o
 atomic_thread_fence(acquire)(232)   |  seen = my_batch          (163)
 memcpyAsync d_out -> out (234)      |  loop to (100) for next batch
```

The four fences form two release/acquire pairs (host→GPU: 228/104; GPU→host:
158/232) plus two interior orderings (107: count-before-id within device
memory; 151: results-before-arrival). Every one of them is load-bearing;
when modifying this kernel, the rule is that any new data written before a
doorbell/arrival must be covered by the existing fence on that edge or get
its own.

#### `struct PersistEngine` (lines 168-177)

Host-side handle: the mapped control line in both address spaces
(`h_ctl`/`d_ctl`), the device publication pair `g_pub`, arrival counter
`g_arrive`, the four request buffers, **two streams** — `kstream` (the
kernel's home) and `cstream` (all copies) — "both NON-BLOCKING (toy lesson
3)" (174), the host's batch counter `cur_batch`, and the computed grid size.
The two-stream split is mandatory: the kernel occupies `kstream` forever, so
every copy must ride a stream with no implicit synchronization against it or
against the legacy default stream.

#### `persist_start` (lines 179-215)

Bring-up sequence, in order: enable mapped-host memory (180,
`cudaSetDeviceFlags(cudaDeviceMapHost)` — must precede any allocation);
allocate + zero the 64 B control line and get its device alias (181-183);
allocate + zero `g_pub` and `g_arrive` (184-187) — zeroing matters: the
kernel's initial `seen = 0` must match an initial published batch id of 0;
allocate the four request buffers at `PMAX_BATCH` capacity (188-191,
~13 MB total); create the two non-blocking streams (192-193).

Then the **exact-occupancy grid computation** (195-203):
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (197-198) asks the driver how
many blocks of THIS kernel at THIS block size actually fit per SM (accounting
for registers, shared memory, and the `__launch_bounds__`), and
`grid = bpm * multiProcessorCount` (201). This is the co-residency invariant
made executable: the grid is exactly what fits, derived at runtime, never
hardcoded — which is also why the binary ports to a different GPU (WP5
contract: "re-derive on the new GPU"). On the dev laptop the print at 202-203
reports 216 resident blocks. Pitfall: any change that alters the kernel's
register or shared-memory footprint silently changes `bpm`; that is fine
(the formula adapts) UNLESS someone replaces the computation with a constant.

Launch (205-207) on `kstream`, error check (208), then the **WDDM flush**
(210-214, toy lesson 1): record an event on `kstream` and `cudaEventQuery` it
— the query forces WDDM to submit the queued launch to the hardware. Without
it, on WDDM the launch can sit in a command buffer indefinitely while the
host spins on a doorbell the (never-started) kernel will never answer. On
bare-metal Linux this is a harmless no-op; never remove it.

#### `persist_submit` (lines 218-238)

One doorbell rendezvous per `PMAX_BATCH`-chunk (the persistent twin of the
chunking discipline):

1. Chunk loop (221-222): `c = min(n - off, PMAX_BATCH)`.
2. Payload staging (223-225): async H2D of types, keys (as u64 — same
   reinterpret-as-slices trick), and optionally values, all on `cstream`.
3. Line 226: `cudaStreamSynchronize(cstream)` — "payload resident BEFORE
   doorbell." This is the NVMe discipline: data first, then MMIO.
4. Ring (227-230): write `count`, host release fence (228 —
   `std::atomic_thread_fence(memory_order_release)`, pairing with kernel
   line 104), bump `cur_batch`, write `batch_id`. Order is count-then-id on
   the host side too — the GPU reads `count` only after seeing the new id.
5. Completion spin (231): `while (h_ctl->done_id != cur_batch) {}` — a raw
   userspace spin on the mapped line (no sleep; latency-measurement fidelity
   beats politeness in a bench). Then the host acquire fence (232).
6. Result drain (233-236): if the caller wants outputs, async D2H of `d_out`
   + stream sync.

**The measurement caveat lives here and must be stated whenever the numbers
are quoted:** the timed persistent sweep calls this function, so persistent
per-batch time INCLUDES step 2-3 (the H2D payload copy and its sync) every
rendezvous, while the launch-mode sweep (phase 1, and `rgi_sweep`) pre-stages
keys on the device once and times only the launch. At small B the doorbell
dominates and the comparison is fair; at large B the persistent numbers carry
a payload-copy penalty the launch numbers do not. The bench prints this
caveat itself (lines 342-343), and WP5 design step 3 adds a both-sides
pre-staged mode to make the large-B comparison clean. Do not "fix" the
existing mode away — the H2D-inclusive number is the honest end-to-end story
and both are wanted.

#### `persist_stop` (lines 240-245)

Sets `stop = 1`, release fence, then bumps `batch_id` (243) — the bump is
required because the level-1 spinner (kernel line 103) sleeps on
`batch_id == seen && stop == 0`; the bump guarantees wake-up even if the
spinner's compiler/hardware read pattern favored the first clause. Block 0
then publishes the sentinel, every block returns, and
`cudaStreamSynchronize(kstream)` (244) observes kernel termination. WP5's
borrowed-hardware etiquette item builds on this: campaign scripts must trap
signals and call this path so a spinning kernel is never left on a shared
box.

#### `main` — the dual-binding bench (lines 253-345)

*Setup (254-272):* N keys (default 2M), production-matching table
construction (256-258: pool 0.4, fill 2.0), key/value population `1..N`, a
SEPARATE set of device buffers (`dk/dv/dout`, 264-269) sized N for
launch-mode use — note these are distinct from the persistent engine's
buffers, and the keys are **pre-staged once** here (268-269), which is
exactly the asymmetry behind the large-B caveat. Base population via one
host-API insert (270-271).

*Batch ladder (274-276):* `{1, 8, 64, 512, 4096, 32768, 262144, 1048576}` —
identical to `rgi_sweep`'s ladder so results compose.

*Phase 1 — launch-mode floor (278-288).* MUST run before the resident kernel
exists (the co-residency hazard). Per batch size: iteration count scaled to
batch cost (281: 2000 / 200 / 20), one warmup find + sync (282), then the
timed loop of `table.find<false,true>(dk, 2, nullptr, dout, B)` launches with
ONE final sync (283-286) — per-batch time = total/iters. Syncing once outside
the loop measures sustained launch throughput; this matches `rgi_sweep`'s
methodology (§6.1) so the two binaries report consistent floors (12-28 µs
flat at small B on this box).

*Phase 2 — persistent mode (290-331).* `persist_start` (292), then the
correctness gate before any timing (294-315): 1024 persistent FINDs of known
keys with a stride-7 access pattern, validated element-wise (296-304,
expected "1024/1024 correct"); then 1024 persistent INSERTs of fresh keys
(N+1..N+1024 with values 7777+i) followed by persistent FINDs of those same
keys, validated (306-314). This insert-then-find round trip through the
doorbell is the v1 validation WP5 cites ("Validated 1024/1024 on persistent
FIND and INSERT"). Then the persistent floor sweep (317-330): same ladder,
slightly reduced iteration counts (323: 1000/100/10), one warm submit WITH
result drain (324), then the timed loop submitting with `out=nullptr` (327 —
"no D2H in timed loop", so the D2H result copy is excluded but the H2D
payload is not; see the caveat). `persist_stop` (331).

*Report (333-344):* a four-column table — per-batch µs and Mop/s for both
bindings side by side — followed by the printed caveat (342-343). Measured
outcome on WSL2/WDDM (2026-06-09): doorbell rendezvous 75-90 µs vs 13-28 µs
launch at small B — **persistent is 5-6x worse on this platform**, exactly as
the toy engine's 75.5 µs single-op round trip predicted. This is a finding
about WDDM-mapped-memory polling costs, not a defeat of the architecture; the
bare-metal expectation (low-single-digit µs doorbell) is stated only
conditionally, and WP5's decision rule settles it empirically.

### Invariants summary (`rgi_persist_engine.cu`)

1. **Exact co-residency:** grid size = occupancy calculator output x SM
   count, computed at runtime (197-201). Never hardcode; never launch any
   other kernel while the persistent grid is resident.
2. **Two-level doorbell:** only block 0 ever reads the mapped line; all other
   blocks read only device memory. Any new control word must follow the same
   republication path.
3. **Fence pairing:** host release (228) ↔ kernel system acquire (104);
   kernel system release (158) ↔ host acquire (232); device-scope
   count-before-id (107) and results-before-arrival (151). New shared data
   must ride an existing pair or add its own.
4. **Capture-once:** the batch id is read into `my_batch` exactly once per
   batch (119) and reused for completion and `seen`.
5. **Payload-before-doorbell:** the copy-stream sync (226) precedes the ring
   (230); requests never travel via mapped memory.
6. **Non-blocking streams only**, and the WDDM flush after launch (210-214)
   stays even on Linux.
7. **Monotonic batch ids**; `0xFFFFFFFF` reserved as stop; `done_id` compared
   by equality to the expected id, never by "changed."
8. **v1 opcode space is {FIND, INSERT} only.** No erase until DEBRA drains
   are restructured (WP5); dispatching `cooperative_erase` without a
   reclaimer context + quiescent-point drains is a use-after-free generator.
9. **Additive:** nothing in the production engine/FDW/worker may grow a
   dependency on this file until a platform's measurements justify making the
   persistent binding primary (WP5 contract).

### How to modify safely (`rgi_persist_engine.cu` — this is WP5's home)

- **v2 erase (WP5 design 1-2):** mirror `batch_kernel`'s reclaimer preamble
  (shared-memory buffer, `begin/end_critical_section` around each batch
  iteration); add `request_type_erase` dispatch via `cooperative_erase`; run
  `drain_all` every K batches (start K=16) at the existing arrival-counter
  rendezvous — it is already a global quiescent point. Verify first, by
  reading `simple_debra_reclaim.hpp`, that `drain_all` tolerates mid-kernel
  invocation; if it assumes kernel exit, fall back to periodic STOP/RELAUNCH
  epochs and document the compromise (WP5 risks section). Do not ship subtle
  reclamation races to hit a date.
- **Fair large-B mode (WP5 design 3):** add a pre-staged-payload submit
  variant for BOTH bindings; keep the H2D-inclusive mode and numbers.
- **Porting to new hardware (WP5/WP7):** `-arch` changes in the build line;
  the grid recomputes itself; re-validate the 1024/1024 gates before any
  timing; instrument scripts to set the stop sentinel in signal handlers.
- **WP1 interaction:** the per-request length rule lands inside the ballot
  queue (compute from the loaded high slice), keeping `d_keys` layout at 2
  slices/request.
- Never add a `cudaDeviceSynchronize` or default-stream operation anywhere in
  the host path while the kernel is resident — both deadlock by definition.

---
<!-- CONTINUED3 -->
