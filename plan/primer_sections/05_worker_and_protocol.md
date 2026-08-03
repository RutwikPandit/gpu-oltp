# Section 05 — The GPU Service Worker and Its Protocol, plus the Legacy Toy FDW

This section is the deep onboarding reference for:

| File | Role | Status |
|---|---|---|
| `pg_rgi_fdw/pg_gpu_service.h` | Client-side interface to the GPU service worker (opcodes, five entry points) | LIVE — production path |
| `pg_rgi_fdw/pg_gpu_service.c` | The background worker itself: shared memory, latch protocol, worker main loop, client API | LIVE — production path, WP2's primary modification target |
| `pg_gpu_fdw/pg_gpu_fdw.c` | The HISTORICAL per-backend FDW over the toy engine, plus the bandwidth-scan (OLAP demo) SQL functions | LEGACY for OLTP; bandwidth-scan functions still the OLAP demo path |
| `pg_gpu_fdw/pg_gpu_fdw.control`, `pg_gpu_fdw--1.0.sql`, `pg_gpu_fdw/Makefile` | Extension packaging for the legacy FDW | LEGACY |
| `run_gpu_db.sh` | One-command launcher for the live (kv_rgi) path | LIVE |
| `pg_rgi_fdw/enable_worker.sql` | The single ALTER SYSTEM that makes the worker exist | LIVE |

All paths in this document are relative to the repository root
`gpu_oltp/`. All line numbers refer to the files as of 2026-06-12 and are
exact; if a cited line does not contain what this document says it
contains, the file has changed since this primer was written and the
primer section must be regenerated before being trusted.

Reading order for a subagent with zero context:

1. Section 0 below (why a worker exists at all — without this, nothing
   in the code makes sense).
2. The PostgreSQL machinery explainer under the `pg_gpu_service.c`
   section (bgworkers, shared memory, LWLocks, latches — self-contained,
   assumes only C and general systems knowledge).
3. The function-by-function walkthrough of `pg_gpu_service.c`, with the
   four sequence diagrams.
4. The failure-path analysis and memory-ordering audit (required reading
   before touching the protocol).
5. The legacy FDW section, mostly so you never confuse `kv` (toy) with
   `kv_rgi` (real).

Related primer sections (not duplicated here): the planner/executor FDW
for `kv_rgi` itself (`pg_rgi_fdw/pg_rgi_fdw.c` — pushdown, transaction
buffer, PRE_COMMIT hook) is documented in its own section; the RGI engine
wrapper (`engine/rgi_oltp_engine.cu`) and the toy engine
(`engine/gpu_oltp_engine.cu`) are documented in the engine section. This
section cites specific engine lines only where the worker's correctness
depends on engine behavior.

---

## 0. Orientation: why a background worker exists at all

Three independent facts collide and force this architecture. None of
them is negotiable, so internalize all three before reading any code.

**Fact 1 — PostgreSQL is a process-per-connection system.** There are no
threads. The supervisor process (the "postmaster") `fork()`s a fresh OS
process (a "backend") for every client connection. Two psql sessions are
two processes with separate address spaces. Anything that must be shared
between sessions must live in System V / mmap shared memory that the
postmaster creates at startup and every child inherits.

**Fact 2 — CUDA contexts are per-process.** A CUDA context (the thing
that owns device allocations, streams, and the ability to launch
kernels) cannot be shared across `fork()`. If every backend created its
own context: (a) each would pay ~100+ ms context creation on first GPU
touch, (b) each would own a private copy of the index — N connections
would mean N disjoint GPU hash tables, and (c) an 8 GB GPU does not
survive N copies of a multi-GB index. CUDA IPC can share *allocations*
across processes but not engine state (host-side staging vectors, the
live-key set, DEBRA reclamation epochs), and MPS merely multiplexes N
contexts — wrong shape. (This decision ledger is
`comp_arch_db_explainer/FULL_PROJECT_EXPLAINER.md` §25; do not re-litigate
it without reading that first.)

**Fact 3 — the index must be shared and transactional.** The whole point
of the system is one GPU-resident index visible to all SQL sessions,
with real `unique_violation` errors and atomic multi-statement commits.

The only shape that satisfies all three: **exactly one long-lived
process owns the CUDA context and the engine**, and every backend talks
to it via shared memory + wakeup signals. PostgreSQL has a first-class
mechanism for exactly this kind of process: the *background worker*
(bgworker). `pg_gpu_service.c` registers one, names it
`pg_gpu_service`, and that process is the **only process in the entire
system that ever calls into CUDA on the OLTP path**. Every other process
(every backend) interacts with the GPU exclusively by writing request
descriptors into a shared-memory region and waking the worker.

This is also PG-Strom's production answer (their "GPU Service"),
arrived at independently — convergent evolution that the project cites
as evidence the shape is right. Architecturally, the worker is the
future home of the C2C doorbell and the WP2 coalescer: it is the process
that will fuse many backends' concurrent requests into one GPU dispatch
per window. Today (pre-WP2) it does something much simpler: it services
**one bulk operation at a time**, serialized by a single lock. That
serialization is deliberate, named, and measured — it is the known
ceiling WP2 exists to remove (`plan/WP2_coalescer_multiclient.md`,
"Current state").

Two extensions exist in this repository and they are very easy to
confuse:

- `pg_rgi_fdw` (directory `pg_rgi_fdw/`) — the REAL system. One shared
  library containing both the FDW (`pg_rgi_fdw.c`, documented elsewhere)
  and the service worker (`pg_gpu_service.c`, documented here). Backed
  by the RGI engine wrapper (`engine/rgi_oltp_engine.cu` →
  `libgpuoltp.so`'s RGI half). The SQL surface is the foreign table
  `kv_rgi`. Data is shared across sessions, PK-enforced,
  transactionally committed.
- `pg_gpu_fdw` (directory `pg_gpu_fdw/`) — the LEGACY sprint scaffold.
  A self-contained FDW over the *toy* engine
  (`engine/gpu_oltp_engine.cu`, persistent-kernel hash table). The SQL
  surface is a foreign table conventionally named `kv`. Per-backend
  engine, per-row dispatch, no transactions, no PK. It survives in the
  tree for two reasons: it is the historical baseline (the 13 kop/s
  number that motivates everything), and its four bandwidth-scan SQL
  functions (`gpu_load` / `gpu_sum` / `gpu_count_lt` / `gpu_kernel_ms`)
  remain the live OLAP demo path (the ~216x aggregate figure).

When this document says "the worker" it always means the
`pg_gpu_service` background worker from `pg_rgi_fdw`. The legacy FDW
has no worker — that absence is its defining limitation.

---

## 1. File: `pg_rgi_fdw/pg_gpu_service.h` (55 lines)

### Purpose

The complete client-side contract with the worker. Anything a backend
process (in practice: `pg_rgi_fdw.c`, plus the two demo SQL functions)
may do to the GPU index goes through one of the five functions declared
here, parameterized by the opcode enum. The header deliberately exposes
**no shared-memory structure** — `GpuServiceShmem` is private to
`pg_gpu_service.c` — so the FDW cannot reach around the protocol. Keep
it that way: WP2 will change the shared-memory layout radically, and the
fact that the FDW compiles against only these five prototypes is what
makes that change tractable.

### Position in the architecture

```
  pg_rgi_fdw.c  (FDW callbacks, txn buffer, PRE_COMMIT hook)
        |
        |  calls (this header's API)
        v
  pg_gpu_service.h  <-- the contract documented here
        |
        |  implemented by (client-API half, runs in EVERY backend)
        v
  pg_gpu_service.c  --- shared memory + latches ---> worker process
        |                                                  |
        |                                                  v
        |                                        rgi_oltp_engine.cu (C ABI)
        |                                                  |
        |                                                  v
        |                                        RGI GPUChainHashtable (CUDA)
```

Both `pg_rgi_fdw.c` and `pg_gpu_service.c` are compiled into the single
shared library `pg_rgi_fdw.so`. The client-API half of
`pg_gpu_service.c` executes inside ordinary backends; the worker half
(`gpu_service_main`) executes only inside the bgworker process. Same
object file, two execution contexts — keep that in mind whenever you
read a function in `pg_gpu_service.c` and ask "who runs this?".

### Walkthrough

#### The opcode enum (lines 8–15)

```c
enum {
    SVC_LOOKUP = 0, SVC_INSERT = 1, SVC_UPDATE = 2, SVC_DELETE = 3,   /* single-row */
    SVC_SNAPSHOT = 4, SVC_FIND_MANY = 5,                              /* bulk read  */
    SVC_INSERT_MANY = 6, SVC_UPDATE_MANY = 7, SVC_DELETE_MANY = 8,    /* bulk write (legacy) */
    SVC_SNAPSHOT_NEXT = 9,                                            /* next snapshot page */
    SVC_TXN_BEGIN = 10, SVC_TXN_STAGE_DEL = 11, SVC_TXN_STAGE_UPD = 12,
    SVC_TXN_STAGE_INS = 13, SVC_TXN_COMMIT = 14, SVC_TXN_ABORT = 15   /* atomic commit */
};
```

Sixteen opcodes in four families. Which transport each family uses, and
who calls it today:

| Opcodes | Transport | Live caller | Status |
|---|---|---|---|
| `SVC_LOOKUP/INSERT/UPDATE/DELETE` (0–3) | slot ring | `gpu_svc_insert()` / `gpu_svc_lookup()` SQL demo functions only | demo/test path; bypasses transactions AND the primary key |
| `SVC_SNAPSHOT` (4), `SVC_SNAPSHOT_NEXT` (9) | bulk channel | `gpu_svc_snapshot_all()` (full scans in `pg_rgi_fdw.c`) | live |
| `SVC_FIND_MANY` (5) | bulk channel | `gpu_svc_bulk()` (pushdown point reads / multi-gets) | live — the hot read path |
| `SVC_INSERT_MANY/UPDATE_MANY/DELETE_MANY` (6–8) | bulk channel | none in the live FDW | legacy; superseded by the TXN_* staged protocol, kept for compatibility |
| `SVC_TXN_BEGIN/STAGE_DEL/STAGE_UPD/STAGE_INS/COMMIT/ABORT` (10–15) | bulk channel | `gpu_svc_txn_commit()` (the PRE_COMMIT flush of the transaction buffer) | live — the hot write path |

Three things the enum's shape tells you:

1. The numbering is append-only history, not design: 0–8 are the
   original sprint protocol, 9 was added when snapshot truncation was
   discovered, 10–15 were added when the original chunked
   `INSERT_MANY` commit was shown to be non-atomic (a duplicate in
   chunk 2 left chunk 1 and all deletes applied — see explainer §18).
   Never renumber; the values are baked into the protocol's only
   versioning, which is "both halves come from the same .so".
2. The enum is anonymous and untyped — opcodes travel through shared
   memory as a plain `int` field (`bulk_op` / slot `op`). There is no
   protocol version field and no magic number in the shared region.
   This is safe today only because client and worker are the same
   compiled library loaded by the same postmaster. If WP2 adds fields,
   a mid-flight `pg_ctl restart` after replacing the .so is the only
   supported upgrade path (the shared region is re-initialized at
   postmaster start; see `gpu_svc_shmem_startup`).
3. There is no `SVC_SHUTDOWN`, no `SVC_STATS`, no `SVC_PING`. Worker
   liveness is inferred from `worker_pid` (see `gpu_svc_available`,
   and the failure-path analysis for why that inference is weaker than
   it looks). WP2's plan requires a stats query ("measured mean
   batch-per-dispatch") — that will be the first new opcode added in
   anger; follow the append-only rule.

#### `GPU_SVC_BULK_CAP` (line 18)

```c
#define GPU_SVC_BULK_CAP 262144
```

262,144 = 2^18 entries — the capacity of EACH of the two `uint64`
arrays in the shared bulk region (`bulk_keys`, `bulk_vals`), i.e.
2 MiB per array, 4 MiB for the pair. Every bulk request and every bulk
response is bounded by this. Consequences, all visible later in the .c:

- A staged commit larger than 262,144 rows is streamed in multiple
  `STAGE_*` chunks under one lock hold (`stage_stream_locked`,
  pg_gpu_service.c lines 390–401).
- A snapshot of a table larger than 262,144 rows is paged
  (`SVC_SNAPSHOT_NEXT`); before paging existed, results silently
  truncated at this cap — that bug is why opcode 9 exists and why
  `snap_page_test.sql` plants a >262,144-row table.
- A single `gpu_svc_bulk` call with `*count > GPU_SVC_BULK_CAP` is a
  hard `ereport(ERROR)` (pg_gpu_service.c lines 374–376) — the caller
  (`pg_rgi_fdw.c`'s FIND_MANY pushdown) must keep multi-get batches
  under the cap. SQL `k = ANY(...)` arrays of >262k keys would error;
  realistic multi-gets are a few thousand keys.

The constant lives in the header (not the .c) because callers need it
for sizing — e.g. `gpu_svc_snapshot_all` uses it as the initial palloc
size, and the FDW could in principle chunk its own requests against it.

#### `gpu_svc_available(void)` (line 21)

Declared here, defined at pg_gpu_service.c lines 285–289. Returns true
when the shared region exists and a worker has registered its pid and
latch. Every entry point checks it and `ereport(ERROR)`s with the
"is 'pg_rgi_fdw' in shared_preload_libraries?" hint if false — by far
the most common operator mistake (loading the extension via
`CREATE EXTENSION` alone does NOT start the worker; see the
`enable_worker.sql` section).

#### `gpu_svc_dispatch(int op, uint64 key, uint64 value, uint64 *out_value)` (line 25)

The single-row transport. Returns `found` (1/0); fills `*out_value` for
lookups. The header comment is explicit that this is "used by the
gpu_svc_insert/lookup test functions" — it is NOT a supported write
path. Documented in full at the .c walkthrough.

#### `gpu_svc_bulk(...)` (lines 33–37)

One bulk operation, one lock hold. `count` is in/out: in = number of
input rows, out = number of result rows. `ok`/`dup_key` carry
PK-violation results for the legacy `INSERT_MANY` and double as the
has-more flag for snapshots (a semantic overload documented — and
flagged as a trap — at the .c walkthrough). Input arrays may be NULL
where the op doesn't need them.

#### `gpu_svc_txn_commit(...)` (lines 45–48)

The atomic-commit entry point: ships an entire classified write set
(delete keys, update key/value pairs, insert key/value pairs) under ONE
lock hold, validate-before-mutate on the worker side. On a PK/UNIQUE
violation, `*ok = 0`, `*dup_key` = offending key, and the header
promises — and the engine delivers — that the GPU index is left
**unchanged** (no partial application). This promise is the load-bearing
sentence of the whole write path; the regression test for it
(`atomic_test.sql`) plants a duplicate in the LAST chunk of a
300,001-row commit and asserts byte-identical index state.

#### `gpu_svc_snapshot_all(uint64 **out_keys, uint64 **out_vals, uint32 *count)` (line 53)

Full-table snapshot with no truncation: pages through the frozen live
set under one lock hold; results are `palloc`'d in the caller's memory
context and grown with `repalloc` as needed. The caller does not free
them explicitly — PostgreSQL memory contexts reclaim them when the
query ends (standard PG idiom; see the machinery explainer).

### Invariants summary

- The header is the ONLY coupling surface between the FDW and the
  worker. No shared-memory types leak out.
- Opcode values are append-only.
- `GPU_SVC_BULK_CAP` bounds every single request and every single
  response; anything bigger must chunk (writes) or page (snapshots).
- `ok` means "success" for write ops but "has-more" for snapshot ops.
- Single-row ops (`gpu_svc_dispatch`) are demo-only and bypass both
  transactions and PK enforcement.

### How to modify safely

- Adding an opcode: append to the enum, add a `case` in
  `worker_do_bulk` (or the slot-ring switch), add/extend a client
  function here. Never reuse or renumber.
- WP2 will add per-backend request descriptors. The right move is to
  keep these five prototypes stable (the FDW does not need to know
  about slots) and change only their implementations — plus one new
  stats accessor for batch-per-dispatch reporting.
- If a prototype must change, grep `pg_rgi_fdw.c` for every caller in
  the same commit; there is no other consumer in the tree (verified:
  the legacy `pg_gpu_fdw` does not include this header).

---

## 2. File: `pg_rgi_fdw/pg_gpu_service.c` (509 lines)

### Purpose

Everything between "a backend wants a GPU operation" and "an RGI C-ABI
call executes in the one process that owns CUDA":

1. **Registration** (`_PG_init`, lines 488–509): at postmaster startup,
   reserve shared memory, reserve two LWLocks, register the bgworker.
2. **Shared memory** (`gpu_svc_shmem_startup`, lines 86–111): one
   ~4.3 MB struct holding the worker's identity, a 2,048-slot
   single-row request ring, and ONE bulk channel with two 2 MiB arrays.
3. **The worker** (`gpu_service_main`, lines 227–282): creates the
   engine once, then loops forever: drain the slot ring, service the
   bulk channel, sleep on its latch with a 50 ms timeout.
4. **The client API** (lines 285–469): the five functions from the
   header, implementing the post/latch/wait/consume handshake and the
   locking discipline that makes commits atomic and snapshots
   consistent.
5. **Two demo SQL functions** (`gpu_svc_insert` / `gpu_svc_lookup`,
   lines 471–485) exercising the slot ring.

### Position in the architecture

This file is the system's concurrency control, its IPC layer, and its
GPU-ownership boundary, all in 509 lines. Upstream of it: the FDW's
per-backend transaction buffer (writes never reach this file until
PRE_COMMIT; reads reach it per statement). Downstream of it: the RGI
engine wrapper's C ABI (`rgi_create`, `rgi_find_many`, `rgi_stage_*`,
`rgi_snapshot_*`, ...), every call of which executes in the worker
process only.

It is also WP2's primary target. WP2 replaces "one lock, one
outstanding bulk operation" with per-backend descriptors and a
worker-side read coalescer; every invariant this section derives must
be explicitly re-derived under that change (the "How to modify safely"
subsection at the end of this file's coverage is the charter for that).

### PostgreSQL machinery explainer

Self-contained. Read this once and the rest of the file is just C.

#### Background workers (bgworkers)

A bgworker is an extension-owned process that the postmaster forks and
supervises, exactly like it forks backends — except a bgworker serves
no client connection. Registration is a `BackgroundWorker` struct
handed to `RegisterBackgroundWorker()`:

- `bgw_library_name` + `bgw_function_name` — the postmaster `dlopen`s
  that library in the new process and calls that function as the
  worker's `main()`. Here: library `"pg_rgi_fdw"`, function
  `"gpu_service_main"` (lines 505–506). This is why
  `gpu_service_main` is `PGDLLEXPORT` (line 77) — it must be resolvable
  by name at runtime.
- `bgw_flags = BGWORKER_SHMEM_ACCESS` (line 500) — the worker attaches
  to the main shared-memory segment (and gets a PGPROC entry, hence a
  latch). Without this flag it would be an isolated process, useless
  here. Note the worker does NOT request
  `BGWORKER_BACKEND_DATABASE_CONNECTION`: it never reads catalogs,
  never runs SQL, never participates in transactions or snapshots. It
  is a pure shared-memory service. That is a feature — it can never
  deadlock against a backend through the lock manager, and it never
  holds back the xid horizon.
- `bgw_start_time = BgWorkerStart_RecoveryFinished` (line 501) — start
  only after WAL crash recovery completes, i.e. when the cluster is
  actually open for business.
- `bgw_restart_time = 5` (line 502) — if the worker exits (crash, GPU
  failure, `proc_exit(1)`), the postmaster restarts it 5 seconds later,
  forever. The consequences of a restart (fresh CUDA context, EMPTY
  index, stale in-flight requests) are analyzed in the failure-path
  section — they are the sharpest edge in this file.

Static registration (`RegisterBackgroundWorker`, as opposed to the
dynamic `RegisterDynamicBackgroundWorker`) is only legal while the
postmaster is processing `shared_preload_libraries` — which is why
`_PG_init` bails immediately unless
`process_shared_preload_libraries_in_progress` is true (line 492).

#### `shared_preload_libraries` and `_PG_init`

`_PG_init(void)` is the magic function PostgreSQL calls whenever a
shared library is loaded into a process. The same library can be loaded
two very different ways:

1. Via `shared_preload_libraries = 'pg_rgi_fdw'` in the server config:
   loaded into the **postmaster itself**, before shared memory is
   created and before any child exists. Only in this window can an
   extension ask for shared memory (`RequestAddinShmemSpace`) and
   LWLocks (`RequestNamedLWLockTranche`) — the main segment is sized
   exactly once, at creation — and register static bgworkers.
2. Via `CREATE EXTENSION` / `LOAD` / first use of a C function: loaded
   into one backend, far too late for any of the above.

`_PG_init` here (lines 488–509) does all its work in case 1 and nothing
in case 2. Operationally this means **the extension does not work until
`shared_preload_libraries` is set and the server restarted** — that is
the entire content of `enable_worker.sql`, and the error hint in every
client entry point exists because people forget.

On Linux (the deployment target — WSL2/Ubuntu per `run_gpu_db.sh`),
children are `fork()`ed: the postmaster runs `_PG_init` and the shmem
startup hook once, and every backend and the worker inherit the
library, the `gpu_svc` pointer, and the attached segment through fork.
(On EXEC_BACKEND/Windows builds each child re-runs the hook; the
`found` flag in `ShmemInitStruct` makes that idempotent. Nobody runs
this on native Windows, but the code is correct for it.)

#### The shmem startup hook and `ShmemInitStruct`

Extensions cannot just `malloc` shared memory. The dance is:

1. In `_PG_init` (preload window): `RequestAddinShmemSpace(size)` —
   "add `size` bytes to the main segment when you create it" (line
   494), and `RequestNamedLWLockTranche("gpu_service", 2)` — "carve me
   2 LWLocks, findable by name" (line 495).
2. Hook `shmem_startup_hook` (lines 496–497), chaining the previous
   hook — this is a single global function pointer shared by all
   extensions, so every extension must save and call its predecessor
   (line 90). Break the chain and you silently break pg_stat_statements
   or whatever else is preloaded.
3. When the postmaster creates the segment, the hook runs.
   `ShmemInitStruct("gpu_service", size, &found)` (lines 92–93) finds
   or allocates the named region; `found` says whether somebody (a
   prior hook run, in EXEC_BACKEND) already initialized it. First
   initializer takes `AddinShmemInitLock` (lines 91, 110) — the
   system-wide "extensions initializing shmem" mutex — zero-state
   races excluded.

Initialization on `!found` (lines 94–109): resolve the two LWLocks from
the named tranche (`GetNamedLWLockTranche("gpu_service")` returns an
array of 2 padded locks; `[0]` becomes `alloc_lock`, `[1]` becomes
`bulk_lock` — matching the comment at line 495), null the worker
identity, set the bulk channel and all 2,048 ring slots to `ST_FREE`.
Note what is NOT initialized: `bulk_op`, `bulk_count`, the 4 MiB
arrays — they are written fresh by every request, and the segment is
zero-filled by the OS anyway.

#### The shared-memory layout (ASCII)

`GpuServiceShmem` (defined at lines 48–67), one instance, name
`"gpu_service"`, total `MAXALIGN(sizeof(...))` ≈ 4.3 MB:

```
GpuServiceShmem  (shared memory segment "gpu_service", ~4.3 MB)
+--------------------------------------------------------------------------+
| worker identity                                                          |
|   Latch  *worker_latch     -- &MyProc->procLatch of the worker (line 235)|
|   pid_t   worker_pid       -- worker's pid; 0 until first start (line 236)|
+--------------------------------------------------------------------------+
| locks (pointers into the named-tranche array, NOT inline storage)        |
|   LWLock *alloc_lock       -- guards slot-ring allocation        [tr 0]  |
|   LWLock *bulk_lock        -- serializes ALL bulk ops (and, by   [tr 1]  |
|                               design, single-row ops too)                |
+--------------------------------------------------------------------------+
| THE BULK CHANNEL  (exactly ONE outstanding bulk op, ever)                |
|   pg_atomic_uint32 bulk_state   -- ST_FREE(0) -> ST_POSTED(1) -> ST_DONE(2)
|   int    bulk_op                -- opcode (enum in the header)           |
|   uint32 bulk_count             -- in: #input rows; out: #result rows    |
|   int    bulk_ok                -- out: writes: 1=ok 0=PK-violation;     |
|                                    snapshots: 1=MORE PAGES 0=done (TRAP:|
|                                    the struct comment at line 58 reads   |
|                                    backwards for snapshots; trust lines  |
|                                    138 and 455, not the comment)         |
|   uint64 bulk_dup               -- out: offending key on PK violation    |
|   uint64 bulk_off               -- VESTIGIAL: written (line 345), never  |
|                                    read by the worker; the snapshot      |
|                                    cursor lives worker-private (snap_off)|
|   int    bulk_req_pid           -- in: requester pid (stage ownership)   |
|   Latch *bulk_waiter            -- requester's latch (worker wakes it)   |
|   uint64 bulk_keys[262144]      -- 2 MiB  \  in: keys / out: result keys |
|   uint64 bulk_vals[262144]      -- 2 MiB  /  in: vals / out: result vals |
+--------------------------------------------------------------------------+
| THE SINGLE-ROW SLOT RING  (demo path ONLY — gpu_svc_insert/lookup)       |
|   GpuReqSlot slots[2048]        -- ~96 KiB total; each slot (~48 B):     |
|     pg_atomic_uint32 state      --   ST_FREE -> ST_POSTED -> ST_DONE     |
|     int    op                   --   SVC_LOOKUP/INSERT/UPDATE/DELETE     |
|     uint64 key, value           --   request payload                     |
|     uint64 out_value            --   response payload (lookups)          |
|     int    found                --   response: 1/0                       |
|     Latch *waiter               --   requester's latch                   |
+--------------------------------------------------------------------------+
```

Size arithmetic: 2 arrays x 262,144 x 8 B = 4,194,304 B dominate;
2,048 slots x ~48 B ≈ 96 KiB; header fields are noise. WP2's design
note (16 descriptors x 2 x 64k x 8 B ≈ 16 MB, or smaller windows) is a
direct replacement of the bulk-channel block of this diagram.

Loudly, again, because it is the most-misread part of the file: **the
slot ring serves ONLY the two demo SQL functions.** It bypasses the
transaction buffer, bypasses PK/UNIQUE enforcement, and writes straight
into the shared engine. It exists to demonstrate cross-session
sharing ("insert in session A, look up in session B" —
`svc_a.sql`/`svc_b.sql`) and for protocol smoke tests. The FDW never
touches it. Any benchmark that goes through `gpu_svc_insert` is
benchmarking the demo path, not the system.

#### LWLocks

LWLocks ("lightweight locks") are PostgreSQL's shared-memory
reader/writer mutexes: spinlock-then-sleep, no deadlock detector, no
fairness guarantees beyond a wait queue, and — critically for this
file — **automatically released when a backend errors out**
(`ereport(ERROR)` longjmps to the transaction abort path, which calls
`LWLockReleaseAll`). That auto-release is why the code can
`ereport(ERROR)` while holding `bulk_lock` (e.g. line 315, line 428)
without wedging the system: the lock frees, even though — see the
failure-path analysis — the *protocol state* it protected may not.

Both locks here are only ever taken `LW_EXCLUSIVE`. The reader/writer
capability of LWLocks is unused today; WP2's "reads concurrent, commits
exclusive" could in principle use `LW_SHARED` for reads, but the WP2
plan deliberately rejects that in favor of scheduling inside the
single-threaded worker (WP2 design point 4: "Readers-writer discipline
implemented IN THE WORKER LOOP ... this is scheduling, not locking"),
because the contention point should become the batching mechanism, not
a fancier lock.

Lock roles as they exist today:

- `bulk_lock` — THE serialization point of the entire system. Held for
  the full duration of every logical operation: one bulk call, one
  whole multi-chunk staged commit, one whole multi-page snapshot, and
  (deliberately) even one single-row demo op. Owning `bulk_lock` means
  "the engine's state cannot be changed by anyone else until I release
  it" — every correctness argument in this file reduces to that
  sentence.
- `alloc_lock` — guards the find-a-free-slot scan of the demo ring
  (lines 310–314). Historically the ring allowed concurrent allocation
  from many backends; now that `gpu_svc_dispatch` first takes
  `bulk_lock` (line 308), at most one backend allocates at a time and
  `alloc_lock` is redundant-but-harmless belt-and-suspenders. It is
  acquired strictly inside `bulk_lock` (consistent ordering — no
  deadlock possible).

#### Latches

A latch is PostgreSQL's process wakeup primitive: a per-process flag
(in that process's `PGPROC` entry, which lives in shared memory) plus a
self-pipe/signal mechanism. API:

- `SetLatch(latch)` — set the flag, wake the owner if it is sleeping.
  Callable from ANY process (the latch is in shared memory) and from
  signal handlers. Cheap, async-signal-safe.
- `WaitLatch(MyLatch, flags, timeout, wait_event)` — sleep until own
  latch is set, or timeout, or (with `WL_EXIT_ON_PM_DEATH`) the
  postmaster dies — in which case the process exits immediately, which
  is how every loop in this file avoids surviving a dead postmaster.
  The `wait_event` tag (`PG_WAIT_EXTENSION` everywhere here) is what
  shows up in `pg_stat_activity.wait_event` — a stuck backend waiting
  on the worker is visible as wait_event `Extension`.
- `ResetLatch(MyLatch)` — clear the flag.

The race-free usage pattern (used by every wait loop in this file) is:

```
while (condition not met)
    WaitLatch(...);          /* may wake spuriously or by timeout */
    ResetLatch(MyLatch);
    /* loop re-checks condition AFTER reset */
```

Re-checking after `ResetLatch` is what closes the lost-wakeup race: if
the peer sets the flag between the condition check and the reset, the
reset clears it, but the loop's next condition check sees the completed
state anyway. The inverse order (reset, then check, then sleep) is
also fine; checking only before reset is NOT. All three wait loops in
this file (lines 322–327, 354–359, 276–278) follow the safe pattern —
preserve it in any modification.

Every wait here also has a timeout (1 s / 5 s / 50 ms). The timeouts
are not load-bearing for correctness (the latch handshake is); they are
insurance: a lost `SetLatch` (e.g. a waiter pointer to a since-exited
process) degrades to polling instead of hanging forever. The worker's
50 ms timeout additionally means the worker is, at worst, a 20 Hz
poller even if no one ever latches it — relevant when reasoning about
worst-case request latency (≤50 ms added if a SetLatch is lost; ~0
normally).

Backends additionally call `CHECK_FOR_INTERRUPTS()` in their wait loops
(lines 326, 358): that is the hook where PostgreSQL services query
cancellation (`pg_cancel_backend`, Ctrl-C in psql) and termination —
it `ereport(ERROR)`s out of the loop, releasing LWLocks but leaving
shared protocol state behind. What exactly is left behind, per protocol
step, is the subject of the failure-path analysis. The worker has no
`CHECK_FOR_INTERRUPTS` (it is not a backend; it handles SIGTERM via its
own flag).

#### Memory barriers and PG atomics

`pg_atomic_uint32` read/write (`pg_atomic_read_u32` /
`pg_atomic_write_u32`) are **atomic but NOT ordering barriers** in
PostgreSQL's portability layer. Cross-process publication therefore
needs explicit fences:

- `pg_write_barrier()` — all stores before it become visible before
  any store after it.
- `pg_read_barrier()` — all loads after it happen after any load
  before it.

The protocol's rule, applied at every state transition: **payload
first, barrier, then state flag** on the producing side; **state flag
first, barrier, then payload** on the consuming side. The
barrier-by-barrier audit below walks every instance and flags the one
place the consuming half is missing (safe on x86, latent elsewhere).

### Walkthrough

Conventions: "backend" = an ordinary connection process running the
client-API half; "worker" = the bgworker process running
`gpu_service_main`. Quoted line numbers are from
`pg_rgi_fdw/pg_gpu_service.c`.

#### File-scope constants and types (lines 32–77)

```c
#define GPU_SVC_NSLOTS     2048          /* line 32 */
#define GPU_SVC_CAPACITY   (1u << 22)    /* line 33 */
#define GPU_SVC_FILL       2.0f          /* line 34 */
#define GPU_SVC_POOL       0.20f         /* line 35 */

enum { ST_FREE = 0, ST_POSTED = 1, ST_DONE = 2 };   /* line 37 */
```

- `GPU_SVC_NSLOTS = 2048` — demo ring size. 2,048 slots is wildly
  oversized for a demo path (each op holds `bulk_lock` anyway, so at
  most one is ever in flight); it is sized for a ring that once
  intended to be the real transport.
- `GPU_SVC_CAPACITY = 1<<22 = 4,194,304` — the key capacity passed to
  `rgi_create` (line 238). This is the shared index's nominal size:
  4M keys for ALL sessions combined.
- `GPU_SVC_FILL = 2.0` — RGI chained-hashtable fill factor. A chained
  table tolerates load factor > 1 (chains absorb overflow); 2.0 trades
  bucket-array memory for chain length. Changing it changes find
  latency vs memory footprint; it was chosen to match RGI's own
  benchmarking defaults.
- `GPU_SVC_POOL = 0.20` — fraction of total GPU memory handed to RGI's
  slab allocator pool at `rgi_create` (the `pool_ratio` argument; the
  engine constructs its `slab_t(pool_ratio)` with it,
  rgi_oltp_engine.cu line 81). 20% of an 8 GB card = ~1.6 GB of slab
  pool for nodes/keys. Raise it for bigger tables; remember the toy
  engine, scan columns, and CUDA overheads share the same physical
  memory.
- The `ST_*` state machine is shared by both transports: a slot or the
  bulk channel is FREE (claimable), POSTED (request published, worker
  must process), or DONE (response published, requester must consume).
  Legal transitions and who performs them:

```
            backend writes payload,            worker writes results,
            pg_write_barrier,                  pg_write_barrier,
            state=POSTED, SetLatch(worker)     state=DONE, SetLatch(waiter)
  ST_FREE ----------------------------> ST_POSTED ----------------------> ST_DONE
     ^                                                                      |
     |            backend consumes results (after pg_read_barrier),         |
     +----------------------------- state=FREE -----------------------------+
```

  Plus one off-protocol use: `gpu_svc_dispatch` writes `ST_DONE` into a
  FREE slot as a *reservation marker* (line 313) — covered below.

`GpuReqSlot` (lines 39–46) and `GpuServiceShmem` (lines 48–67) are as
drawn in the layout diagram above. File-scope statics (lines 69–71):

- `gpu_svc` — the per-process pointer to the shared struct, set in the
  postmaster by the startup hook and inherited by every child via fork.
  NULL in any process where the library was loaded without preload
  (which is how `gpu_svc_available` catches the misconfiguration).
- `prev_shmem_startup_hook` — hook chaining (see explainer).
- `got_sigterm` — `volatile sig_atomic_t` flag set by the worker's
  SIGTERM handler; the worker main loop's exit condition.

Worker-private statics (lines 124–126):

```c
static uint64 snap_total = 0;   /* rows in the frozen snapshot set */
static uint64 snap_off   = 0;   /* next snapshot page offset       */
static int    stage_owner = 0;  /* pid that opened current staging */
```

These live in the worker process's private memory (file-scope statics
are per-process; only the worker ever executes the functions that touch
them). They are protocol *session state*: the snapshot cursor and the
staging owner. Their privacy is load-bearing in two ways: (a) no
backend can corrupt them, (b) they are implicitly single-session — only
one snapshot cursor and one staging session can exist, which is exactly
the "one outstanding bulk op" assumption that `bulk_lock` enforces.
WP2 keeps snapshots paged, so the cursor must become per-descriptor
state when multiple snapshot sessions can interleave — see "How to
modify safely".

#### `gpu_svc_shmem_size` (lines 80–84)

`MAXALIGN(sizeof(GpuServiceShmem))` — the single size used both for the
reservation (`RequestAddinShmemSpace`, line 494) and the allocation
(`ShmemInitStruct`, line 93). Keeping them the same expression is the
invariant; if WP2 sizes descriptors dynamically (GUC-driven slot
count), compute the size ONCE in a helper exactly like this and call it
from both places — a mismatch is an out-of-memory at startup or silent
overlap with another extension's region.

#### `gpu_svc_shmem_startup` (lines 86–111)

Covered structurally in the machinery explainer. Function-level notes:

- Line 90: chains the previous hook FIRST, so this extension's region
  is allocated after (not instead of) others'. Order among extensions
  is their order in `shared_preload_libraries`.
- Lines 91/110: `AddinShmemInitLock` held across find-or-create plus
  first-touch initialization. This lock is system-provided,
  specifically for extension shmem init.
- Lines 94–109 run only for the segment creator (`!found`). Note that
  on a *worker restart* this does NOT run — the segment persists for
  the postmaster's lifetime. So after a worker crash/restart,
  `bulk_state` and slot states are whatever they were at the moment of
  death; the new worker inherits in-flight POSTED requests and will
  process them against a brand-new, EMPTY engine. (Failure-path
  analysis below.)
- Lines 97, 100–101: lock pointers are resolved from the named tranche
  every postmaster start and STORED IN SHARED MEMORY as pointers. This
  works because the main segment maps at the same address in all
  children of one postmaster (fork inheritance). Do not be tempted to
  store such pointers anywhere that outlives the postmaster.

#### `handle_sigterm` (lines 113–120)

The worker's SIGTERM handler (installed at line 232): set
`got_sigterm`, `SetLatch(&MyProc->procLatch)` to pop the worker out of
`WaitLatch`, preserve `errno` (signal-handler hygiene — `SetLatch` can
clobber it). Standard bgworker boilerplate; the main loop polls
`got_sigterm` (line 246). SIGTERM arrives on `pg_ctl stop`, postmaster
shutdown, or `pg_terminate_backend(worker_pid)`. The handler must stay
async-signal-safe: flag write + `SetLatch` only.

#### `worker_snapshot_page` (lines 129–139) — worker side

Emits one page of the current snapshot into the bulk arrays:

```c
uint32 m = rgi_snapshot_page(engine, snap_off, GPU_SVC_BULK_CAP,
                             gpu_svc->bulk_keys, gpu_svc->bulk_vals);  /* 132-133 */
uint64 span = min(snap_total - snap_off, GPU_SVC_BULK_CAP);            /* 134-135 */
snap_off += span;                                                      /* 136 */
gpu_svc->bulk_count = m;                                               /* 137 */
gpu_svc->bulk_ok = (snap_off < snap_total) ? 1 : 0;  /* 1 = more pages, line 138 */
```

Three subtleties:

1. **`m` (returned rows) and `span` (cursor advance) are tracked
   separately.** The engine's `rgi_snapshot_page` batch-finds the page's
   frozen keys on the GPU and compacts out keys that no longer resolve
   (in principle none can vanish mid-snapshot today, because
   `bulk_lock` excludes writers — but the engine API tolerates it), so
   a page can return `m < span` rows while the cursor still advances by
   `span` over the frozen key list. Conflating the two would skip or
   repeat keys.
2. **`bulk_ok` is overloaded as the has-more flag** (line 138). For
   every other op `bulk_ok` means success/failure. The shmem struct
   comment (line 58: "1 ok, 0 PK violation / has-more (snapshot)")
   reads as if 0 meant has-more — it is wrong/garbled; the truth is
   here and at the consumer (line 455: `more = gpu_svc->bulk_ok`).
   This is the single most likely place for a future modification to
   introduce an off-by-one-page bug. WP2 should split has_more into
   its own descriptor field and kill the overload.
3. The page offset comes from worker-private `snap_off`, NOT from the
   shared `bulk_off` the client dutifully zeroes (line 345). `bulk_off`
   is dead weight from an earlier protocol revision where the client
   drove the cursor; it can be deleted (both writes and the field) in
   any cleanup pass — verified: no reader exists anywhere in the tree.

#### `worker_do_bulk` (lines 141–225) — worker side; THE opcode dispatcher

Called by the main loop when `bulk_state == ST_POSTED`. Reads the
request from the shared channel, executes it against the engine,
writes results back into the same channel. Defaults first (lines
148–149): `bulk_ok = 1`, `bulk_dup = 0` — every op starts presumed
successful; failure paths overwrite.

Per opcode:

- **`SVC_SNAPSHOT` (lines 153–157):** `snap_total =
  rgi_snapshot_begin(engine)` freezes the live-key set inside the
  engine (the engine copies its host-side `live` set into
  `snap_cache`), resets `snap_off = 0`, emits page 0 via
  `worker_snapshot_page`. "Freeze" means the KEY LIST is fixed; the
  VALUES are fetched from GPU truth page by page. Consistency of the
  values across pages is therefore NOT the engine's doing — it is
  purely the fact that the client holds `bulk_lock` across all pages
  so no write op can run between them. Write that sentence into any
  WP2 redesign.
- **`SVC_SNAPSHOT_NEXT` (lines 158–159):** next page off the existing
  cursor. Note there is NO guard that a snapshot is actually open
  (`snap_total`/`snap_off` could be stale from a previous snapshot). A
  client that sends SNAPSHOT_NEXT without SNAPSHOT gets garbage pages
  of the previous frozen set. Today unreachable (the only caller,
  `gpu_svc_snapshot_all`, always opens first, under the lock); under
  WP2's concurrent world this becomes reachable and needs an owner/pid
  check exactly like staging has.
- **`SVC_FIND_MANY` (lines 161–174):** the hot read path. Input: `n`
  keys in `bulk_keys`. The worker pallocs two scratch arrays (`outv`,
  `fnd`, sized `n`), calls `rgi_find_many(engine, bulk_keys, outv,
  fnd, n)` — which engine-side flushes any pending buffered inserts
  first (rgi_oltp_engine.cu line 166: `rgi_flush(e)`), then runs the
  batched GPU find — and then compacts in place: found pairs are
  written back to `bulk_keys[0..m)` / `bulk_vals[0..m)` (lines
  168–170), `bulk_count = m`. Misses simply vanish from the response;
  the FDW treats absent keys as no row, and PostgreSQL re-checks quals
  anyway so over- or under-approximation here cannot produce wrong
  query results — only missing rows would, and compaction preserves
  exactly the found set. The palloc/pfree happen in the worker's
  default memory context — fine for now; a WP2 worker loop that runs
  forever fusing batches should pre-allocate scratch once instead
  (palloc churn per dispatch is measurable at high request rates).
  Note `rgi_find_many` launches the find with `find<false, true>` —
  the `concurrent=false` template path (rgi_oltp_engine.cu line 74).
  That is sound TODAY because `bulk_lock` guarantees no concurrent
  mutation; it is the engine coupling WP2 must flip (see "How to
  modify safely").
- **`SVC_TXN_BEGIN` (lines 176–181):** `rgi_stage_begin(engine)` —
  which unconditionally CLEARS all four staging vectors
  (rgi_oltp_engine.cu lines 208–212) — then records `stage_owner =
  bulk_req_pid`. The unconditional clear is the crash-hygiene
  property: if a previous backend died after staging 200k rows and
  never committed, those rows sit in the engine's staging vectors —
  and the very next transaction's BEGIN wipes them before staging its
  own. A crashed backend can therefore never poison a later commit
  with leftover staged rows. There is deliberately no "staging already
  open" error: BEGIN is a reset, not a handshake.
- **`SVC_TXN_STAGE_DEL` / `STAGE_UPD` / `STAGE_INS` (lines 182–193):**
  each first checks `stage_owner != bulk_req_pid` and NACKs
  (`bulk_ok = 0`) on mismatch — so even if a protocol bug let some
  other backend's stage chunk through the lock, it fails closed
  instead of splicing rows into a foreign transaction. On match,
  append the chunk into the engine's staging vectors (`rgi_stage_del/
  upd/ins` are plain `std::vector` appends, no GPU work — staging is
  pure host-side accumulation; ALL GPU work happens at COMMIT).
- **`SVC_TXN_COMMIT` (lines 194–201):** owner check, then
  `rgi_stage_commit(engine, &dup)`. Engine-side
  (rgi_oltp_engine.cu lines 228+), commit is validate-then-apply:
  flush pending buffered writes; VALIDATE = host hash-set scan for
  intra-commit duplicate insert keys + one batched GPU find over all
  staged insert keys against the index — both before ANY mutation; on
  any hit, drop staging and return the dup key (index untouched).
  APPLY = batched erases (erase of an absent key is a defined no-op)
  then batched upserts (insert with update_if_exists — no failure
  path). Nonzero return → `bulk_ok = 0`, `bulk_dup = dup` (line 198).
  Either way `stage_owner = 0` (line 199) — the staging session is
  closed; a retry must re-BEGIN.
- **`SVC_TXN_ABORT` (lines 202–205):** `rgi_stage_abort` (=
  `rgi_stage_begin`, i.e. clear vectors), `stage_owner = 0`. Note NO
  owner check on ABORT — anyone can abort the current staging. Today
  unreachable cross-pid (the lock again); under WP2 this asymmetry
  (guarded stage/commit, unguarded abort) must be revisited — an
  unguarded abort from pid B between pid A's stage chunks would be a
  denial-of-atomicity, though never a corruption (A's later chunks
  then NACK on owner mismatch... actually no: ABORT sets
  `stage_owner = 0`, and A's next STAGE_* sees `0 != A.pid` and NACKs,
  so A aborts cleanly — fail-closed holds, but the failure is
  spurious).
- **`SVC_INSERT_MANY` (lines 207–213), legacy:** buffer all pairs via
  `rgi_insert` (host-side append), then `rgi_flush_unique(&dup)` —
  which validates (intra-batch dup scan + batched find) BEFORE
  applying, and on conflict discards the pending batch and reports the
  key. So even the legacy op is validate-before-mutate *within
  itself* — what it cannot do, and why it was superseded, is atomicity
  ACROSS deletes+updates+inserts of one SQL transaction: the original
  commit protocol issued DELETE_MANY, then UPDATE_MANY, then chunked
  INSERT_MANY as separate ops, so a PK violation in inserts left the
  deletes applied. The TXN_* family exists because of exactly that
  bug. Do not build anything new on opcodes 6–8.
- **`SVC_UPDATE_MANY` (lines 214–216), legacy:** buffer + `rgi_flush`
  (upsert semantics, no validation — an "update" of an absent key
  inserts it).
- **`SVC_DELETE_MANY` (lines 218–219), legacy:** loop of single-key
  `rgi_delete` calls — each of which engine-side does a flush plus a
  ONE-KEY erase kernel launch (rgi_oltp_engine.cu lines 146–153). n
  deletes = n kernel launches; this is the only per-row GPU dispatch
  left in the worker, another reason the legacy ops are dead.
- **default (lines 221–223):** unknown opcode → `bulk_ok = 0`. This is
  what a torn/corrupted `bulk_op` degrades to — relevant in the
  failure-path analysis.

What `worker_do_bulk` does NOT do: barriers or state transitions. The
caller (main loop) owns the `ST_POSTED → ST_DONE` transition and its
write barrier. Keep that separation: handler = pure request→response
function over the channel; loop = protocol.

#### `gpu_service_main` (lines 227–282) — the worker process

The bgworker entry point (named in `_PG_init`, line 506).

Startup (lines 232–244):

```c
pqsignal(SIGTERM, handle_sigterm);            /* 232 */
BackgroundWorkerUnblockSignals();             /* 233: bgworkers start with
                                                 signals blocked; unblock
                                                 or SIGTERM never arrives */
gpu_svc->worker_latch = &MyProc->procLatch;   /* 235: publish identity */
gpu_svc->worker_pid = MyProcPid;              /* 236 */
engine = rgi_create(GPU_SVC_CAPACITY, GPU_SVC_FILL, GPU_SVC_POOL);  /* 238 */
if (!engine) { ereport(LOG, ...); proc_exit(1); }                   /* 239-243 */
ereport(LOG, "...worker started (pid %d), GPU engine ready");       /* 244 */
```

Notes:

- **The engine is created exactly once per worker lifetime**, here.
  This is THE CUDA context creation, the cudaMalloc of the staging
  buffers, the hashtable build, the slab-pool reservation (20% of GPU
  memory). Everything the index ever holds lives in this process's
  GPU allocations. **Data lifetime = worker lifetime ∩ postmaster
  lifetime**: GPU memory is volatile, there is no WAL, no
  checkpointing, no rebuild-from-heap. A worker restart or cluster
  restart yields an empty index. `run_gpu_db.sh` says this to the
  operator (lines 9–12 of the script); say it in every demo.
- Identity publication (235–236) has no write barrier before it and
  none is needed for correctness of the handshake (clients that read a
  stale NULL just error out with the preload hint and retry); but note
  `gpu_svc_available` can also return true while the worker is between
  `rgi_create` failure and `proc_exit` — a client posting in that
  window waits until the restarted worker (5 s later) drains it.
- On `rgi_create` failure (no GPU, driver hiccup, out of GPU memory):
  LOG + `proc_exit(1)` → postmaster restarts the worker every 5 s
  forever, logging each failure. The system degrades to "every kv_rgi
  query hangs until interrupt or until a worker finally starts" —
  there is no negative cache. Operationally: check the server log for
  "failed to create GPU engine".

The main loop (lines 246–279), verbatim structure:

```c
while (!got_sigterm)
{
    /* PHASE 1: drain the single-row slot ring (lines 250-267) */
    for (i = 0; i < GPU_SVC_NSLOTS; i++)
    {
        GpuReqSlot *s = &gpu_svc->slots[i];
        if (pg_atomic_read_u32(&s->state) != ST_POSTED) continue;
        switch (s->op)
        {
            case SVC_INSERT: rgi_insert(...); rgi_flush(...); s->found = 1; break;
            case SVC_UPDATE: rgi_update(...); rgi_flush(...); s->found = 1; break;
            case SVC_DELETE: rgi_delete(...);                 s->found = 1; break;
            default: /* lookup */ s->found = rgi_lookup(...); s->out_value = v;
        }
        pg_write_barrier();                          /* 264 */
        pg_atomic_write_u32(&s->state, ST_DONE);     /* 265 */
        if (s->waiter) SetLatch(s->waiter);          /* 266 */
    }
    /* PHASE 2: service the bulk channel, at most one op (lines 269-275) */
    if (pg_atomic_read_u32(&gpu_svc->bulk_state) == ST_POSTED)
    {
        worker_do_bulk(engine);
        pg_write_barrier();                                  /* 272 */
        pg_atomic_write_u32(&gpu_svc->bulk_state, ST_DONE);  /* 273 */
        if (gpu_svc->bulk_waiter) SetLatch(gpu_svc->bulk_waiter);  /* 274 */
    }
    /* PHASE 3: sleep (lines 276-278) */
    WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
              50L, PG_WAIT_EXTENSION);
    ResetLatch(MyLatch);
}
rgi_destroy(engine);     /* 280: orderly SIGTERM exit only */
proc_exit(0);            /* 281 */
```

Properties worth spelling out:

- **Single-threaded by construction.** Every engine call in the entire
  system happens on this one thread of this one process. There is no
  engine-internal locking requirement; serialization is the loop. WP2
  keeps this (its readers-writer discipline is loop scheduling, not
  threads).
- **Ring before bulk, every iteration.** A posted bulk op can be
  delayed by up to a full ring drain (2,048 demo ops). Irrelevant
  today (`bulk_lock` means a demo op and a bulk op cannot even be
  outstanding simultaneously — both transports are mutually serialized
  by the same lock, so at most ONE request of either kind exists at a
  time); the phase order only matters if someone removes the demo
  path's `bulk_lock` acquisition. Don't.
- **One bulk op per wakeup is NOT a throughput limit:** after PHASE 2
  completes and the loop reaches `WaitLatch`, the just-released client
  has already let the next client post and `SetLatch`; the latch being
  set makes `WaitLatch` return immediately. The loop then services it.
  So the loop processes back-to-back bulk ops at latch speed, not at
  20 Hz. (Verify against `WaitLatch` semantics: a set latch returns
  without sleeping.)
- The ring's per-op `rgi_flush` (lines 256–257) means each single-row
  demo INSERT/UPDATE costs a full upsert-kernel launch + sync — ~tens
  of µs. The demo path is deliberately naive.
- The `default:` case treats ANY unrecognized slot opcode as a lookup
  (lines 259–262). Garbage in a slot's `op` cannot crash the worker;
  it does a harmless find.
- SIGTERM is only checked at the top of the loop; a SIGTERM that lands
  mid-`worker_do_bulk` (e.g. during a 300k-row commit's GPU work) is
  honored after the op completes. `SetLatch` from the handler ensures
  the subsequent `WaitLatch` doesn't absorb the 50 ms.
- On exit, `rgi_destroy(engine)` frees GPU state in the orderly path
  only. A crash (segfault in the engine, CUDA abort) skips it; the
  context dies with the process anyway (the driver reclaims), so the
  leak is cosmetic — but in-flight protocol state is NOT reclaimed
  (failure-path analysis).

#### `gpu_svc_available` (lines 285–289) — backend side

```c
return (gpu_svc != NULL && gpu_svc->worker_pid != 0 && gpu_svc->worker_latch != NULL);
```

Three conditions, three distinct misconfigurations they catch:

- `gpu_svc == NULL` — the library was loaded without
  `shared_preload_libraries` (no startup hook ran in the postmaster, so
  this backend's static is NULL). The classic operator error.
- `worker_pid == 0` — preload happened (shmem exists, initialized to
  pid 0 at line 99) but the worker has never reached line 236: still
  starting, in recovery wait, or repeatedly failing `rgi_create`.
- `worker_latch == NULL` — same window, belt-and-suspenders.

What it does NOT catch: a worker that started once and has since died
(`worker_pid`/`worker_latch` are never cleared on exit — there is no
on-exit hook resetting them). "Available" really means "a worker has
existed at some point this postmaster lifetime". The protocol survives
this because POSTED requests persist in shared memory and the restarted
worker drains them (with the 5 s restart gap as added latency, against
an empty index). A stricter check (`kill(worker_pid, 0)` or a heartbeat
counter) is a known cheap improvement; today the timeouts +
`CHECK_FOR_INTERRUPTS` make the worst case "hangs until cancel or
restart", never "hangs forever uncancellably".

#### `gpu_svc_dispatch` (lines 291–334) — backend side; the single-row transport

Full anatomy, then the sequence diagram.

1. **Availability gate** (lines 297–299): `ereport(ERROR)` with the
   preload hint.
2. **Take `bulk_lock`** (line 308). The comment at lines 301–307 is the
   design statement, quoted because it is the invariant in the author's
   own words:

   > Hold bulk_lock for the whole single-row op so it cannot interleave
   > with a staged transaction commit or a paged snapshot (both also
   > hold bulk_lock). Without this, a single-row write could mutate the
   > engine between staged chunks / snapshot pages and break commit
   > atomicity / snapshot freshness. NB: the single-row WRITE helpers
   > (SVC_INSERT/UPDATE/DELETE) bypass PK and transaction semantics and
   > exist ONLY for the cross-session demo/tests (gpu_svc_insert); they
   > are not the supported write path (use SQL DML).

   This is why "even single-row ops take the big lock": the lock is not
   protecting the slot ring (the ring's atomics do that); it is
   protecting the ENGINE's quiescence during someone else's multi-step
   bulk sequence. A demo insert sneaking between snapshot pages would
   make page N+1 inconsistent with page N; sneaking between stage
   chunks, it would mutate state the commit's validation already
   reasoned about.
3. **Claim a slot** (lines 310–315): under `alloc_lock`, linear-scan
   the ring for `ST_FREE` and write **`ST_DONE`** into it as a claim
   marker. Why DONE and not POSTED? Because the worker acts on POSTED
   (line 253: `!= ST_POSTED → continue`) and the allocator seeks FREE —
   a slot in DONE is invisible to both, i.e. "reserved". The payload is
   not written yet; publishing POSTED now would hand the worker
   garbage. If no slot is free: `ereport(ERROR, "no free request
   slots")` (line 315) — reachable only via slot leaks (see failure
   paths), since the lock admits one dispatcher at a time.
4. **Publish** (lines 317–320): fill `op/key/value/out_value/found/
   waiter` (waiter = `MyLatch`, the backend's own latch), then
   `pg_write_barrier()` — payload becomes visible BEFORE — then
   `pg_atomic_write_u32(&s->state, ST_POSTED)`, then
   `SetLatch(gpu_svc->worker_latch)`.
5. **Wait** (lines 322–327): the canonical latch loop — while state is
   not DONE: `WaitLatch(... 1000 ms ...)`, `ResetLatch`,
   `CHECK_FOR_INTERRUPTS()`. The 1 s timeout makes a lost wakeup a 1 s
   hiccup, not a hang; `CHECK_FOR_INTERRUPTS` makes Ctrl-C work (with
   protocol-state consequences analyzed below).
6. **Consume** (lines 328–331): `pg_read_barrier()` — results read
   AFTER the DONE observation — copy out `found`/`out_value`, then
   release the slot with `state = ST_FREE` (plain atomic write; no
   barrier needed before FREE because the slot's payload is dead at
   that point and the next claimant fully rewrites it under its own
   publish barrier).
7. **Release `bulk_lock`** (line 332), return `found`.

Sequence diagram — single-row op (demo path):

```
 BACKEND B                                 WORKER W                    GPU
 =========                                 ========                    ===
 LWLockAcquire(bulk_lock)  ...... [excludes ALL other ops, bulk or demo]
 LWLockAcquire(alloc_lock)
   scan ring: slots[i].state==FREE
   slots[i].state = DONE          (claim marker: invisible to W and allocators)
 LWLockRelease(alloc_lock)
 slots[i] = {op,key,value,waiter=MyLatch}
 pg_write_barrier()
 slots[i].state = POSTED  ---------------> (visible on next ring scan)
 SetLatch(worker_latch)   ---------------> WaitLatch returns
 WaitLatch(MyLatch,1s) loop                ring scan finds POSTED:
   ResetLatch / CHECK_FOR_INTERRUPTS         INSERT: rgi_insert+rgi_flush --> upsert kernel
                                             LOOKUP: rgi_lookup ----------> find kernel
                                           s->found / s->out_value written
                                           pg_write_barrier()
            (sees DONE) <----------------- slots[i].state = DONE
 WaitLatch returns      <----------------- SetLatch(s->waiter)
 pg_read_barrier()
 read found / out_value
 slots[i].state = FREE
 LWLockRelease(bulk_lock)
```

Cost: one lock round-trip, one slot scan, two latch crossings, and one
or two kernel launches — per ROW. This is the architecture of the
13 kop/s ancestor, preserved as a demo. Its only legitimate uses are
`svc_a.sql`/`svc_b.sql`-style cross-session demonstrations and protocol
smoke tests.

#### `bulk_post_locked` (lines 338–363) — backend side; the bulk-channel handshake core

Every bulk opcode flows through this one static helper. Contract (per
the comment at lines 336–337): **the caller already holds
`bulk_lock`**; inputs and outputs travel via the shared bulk region;
on return the results are still sitting in the shared arrays (the
caller copies them out while still holding the lock) and `bulk_state`
has been returned to `ST_FREE`.

Steps:

1. Copy inputs into the shared window (lines 341–342): `memcpy` of up
   to `n` keys and (if non-NULL) `n` vals. For zero-input ops
   (TXN_BEGIN/COMMIT/ABORT, SNAPSHOT, SNAPSHOT_NEXT) both pointers are
   NULL and nothing is copied.
2. Fill the descriptor fields (lines 343–349): `bulk_op`, `bulk_count =
   n`, `bulk_off = off` (always 0 from every live call site —
   vestigial), `bulk_req_pid = MyProcPid` (the staging-ownership
   credential), reset `bulk_ok = 1` / `bulk_dup = 0`, `bulk_waiter =
   MyLatch`.
3. Publish (lines 350–352): `pg_write_barrier()`, `bulk_state =
   ST_POSTED`, `SetLatch(worker_latch)`. Same payload-barrier-flag
   discipline as the slot path. Note the state write is unconditional —
   it does not compare-and-swap from FREE. Under the lock that is fine
   (only one bulk client exists at a time) and it is also what makes a
   stale `ST_DONE` left by a cancelled predecessor self-heal: the next
   request simply overwrites it. The one case where unconditional
   overwrite is NOT fine — stale `ST_POSTED` with the worker actively
   reading — is reachable only through the cancellation window analyzed
   in the failure-path section.
4. Wait (lines 354–359): same canonical loop, 5 s timeout (bulk ops can
   legitimately take longer than demo ops — a 300k-row commit's
   validation find + apply kernels, or a large snapshot page).
5. Consume-side barrier + slot release (lines 360–362):
   `pg_read_barrier()`, then `bulk_state = ST_FREE`. Results
   (`bulk_ok/bulk_dup/bulk_count` and the arrays) remain valid for the
   caller because the caller still holds `bulk_lock` — nobody else can
   start a request and overwrite them. Setting FREE before the caller
   reads the arrays is therefore safe TODAY but is a lock-coupled
   subtlety: if WP2 removes the global lock, FREE must move to AFTER
   result extraction or the next requester races the copy-out.

#### `gpu_svc_bulk` (lines 365–386) — backend side; one-shot bulk ops

The public wrapper for single-request opcodes (today: `SVC_FIND_MANY`
from the FDW's pushdown; the legacy MANY ops; in principle a single
snapshot page, though no live caller does that). Steps: availability
gate (371–373), cap check (374–376: `n > GPU_SVC_BULK_CAP` →
`ereport(ERROR)` — the request must fit the window in ONE shot; this
function does not chunk), take `bulk_lock` (378), `bulk_post_locked`
(379), copy out `count`/`ok`/`dup_key` and up to `bulk_count` rows of
keys/vals into caller arrays (380–384, all optional via NULL), release
(385).

Caller-side sizing rule: `out_keys`/`out_vals` must each hold
`GPU_SVC_BULK_CAP` entries in the worst case — for FIND_MANY the result
count is ≤ the input count, so sizing to the input count is safe; for a
raw SNAPSHOT through this function the result is up to the full cap.

Sequence diagram — FIND_MANY (the hot read path; this exact flow
serves `SELECT ... WHERE k = $1` and `k = ANY($1)` after pushdown):

```
 BACKEND B                                 WORKER W                    GPU
 =========                                 ========                    ===
 LWLockAcquire(bulk_lock)
 memcpy n keys -> bulk_keys
 bulk_op=FIND_MANY, bulk_count=n,
   bulk_req_pid=MyProcPid, bulk_waiter=MyLatch
 pg_write_barrier()
 bulk_state = POSTED  -------------------->
 SetLatch(worker_latch) ------------------> WaitLatch returns
 WaitLatch(MyLatch,5s) loop                 bulk_state==POSTED:
   ResetLatch / CHECK_FOR_INTERRUPTS          worker_do_bulk:
                                                rgi_find_many:
                                                  rgi_flush (if pending) --> upsert kernel
                                                  H2D keys; find<false,true> --> find kernel (1 launch)
                                                  D2H results
                                                compact found pairs into
                                                  bulk_keys/bulk_vals[0..m)
                                                bulk_count = m
                                              pg_write_barrier()
            (sees DONE) <------------------ bulk_state = DONE
 WaitLatch returns      <------------------ SetLatch(bulk_waiter)
 pg_read_barrier(); bulk_state = FREE
 copy m pairs out (lock still held)
 LWLockRelease(bulk_lock)
```

One lock hold, one POSTED/DONE handshake, one find launch (plus at most
one flush launch) — **whether n is 1 or 262,144**. That invariance is
the design thesis in miniature: the cost of a dispatch is flat, so the
goal is maximizing rows per dispatch (explainer §55). WP2 generalizes
exactly this: many backends' FIND_MANYs fused into one launch.

#### `stage_stream_locked` (lines 390–401) — backend side; chunked staging

```c
static int stage_stream_locked(int op, const uint64 *keys, const uint64 *vals, uint32 n)
{
    for (off = 0; off < n; off += GPU_SVC_BULK_CAP)
    {
        uint32 c = min(n - off, GPU_SVC_BULK_CAP);
        bulk_post_locked(op, keys + off, vals ? vals + off : NULL, c, 0);
        if (!gpu_svc->bulk_ok) return 0;   /* worker rejected this stage chunk */
    }
    return 1;
}
```

Streams one class of staged rows (`STAGE_DEL` keys-only, or
`STAGE_UPD`/`STAGE_INS` key+val) through the 262,144-row window in as
many chunks as needed, all under the caller's single `bulk_lock` hold.
**Every chunk's `bulk_ok` is checked**; the only way a stage chunk
NACKs today is the worker's owner check (`stage_owner !=
bulk_req_pid`), which under the lock can only mean the worker
restarted mid-commit (restart resets `stage_owner` to 0 — file-scope
static in the new process). Returning 0 propagates to
`gpu_svc_txn_commit`'s abort path: the failure mode is a clean error,
never a partial commit. `n == 0` performs zero iterations and returns
1 — empty classes cost nothing, which is why callers can pass NULL
arrays with zero counts.

#### `gpu_svc_txn_commit` (lines 403–434) — backend side; THE atomic commit

Called exactly once per writing transaction, from `pg_rgi_fdw.c`'s
PRE_COMMIT callback, with the transaction buffer already classified
into three arrays (deletes, updates, inserts). Anatomy:

1. Availability gate (411–413).
2. `LWLockAcquire(bulk_lock, LW_EXCLUSIVE)` (line 417). The comment at
   lines 415–416: "One lock hold for the whole stage->commit: no other
   backend can interleave staging, so the staged set is unambiguously
   owned by this commit." This is invariant #1 of the system.
3. `SVC_TXN_BEGIN` (line 418) — worker drops any stale staging and
   records this pid as owner.
4. Stream all three classes (lines 419–422), short-circuiting:
   `bulk_ok` of BEGIN && stream(DEL) && stream(UPD) && stream(INS).
   Order is semantically irrelevant at this layer (staging is pure
   accumulation; the ENGINE imposes the apply order erase-then-upsert
   at commit), but keep the order stable anyway — tests and the
   explainer describe it.
5. On any staging failure (lines 423–429): post `SVC_TXN_ABORT` (drop
   the partial staging worker-side), release the lock, and
   `ereport(ERROR, "transaction staging failed (protocol error)")` —
   which aborts the whole PostgreSQL transaction. Nothing was applied:
   staging never mutates the index.
6. `SVC_TXN_COMMIT` (line 430): the worker validates then applies (see
   `worker_do_bulk`). Copy out `ok`/`dup_key` (431–432), release
   (433). The CALLER (`pg_rgi_fdw.c`) maps `ok == 0` to
   `ereport(ERROR, errcode(ERRCODE_UNIQUE_VIOLATION), ...)` so SQL
   sees a native-looking `duplicate key value violates unique
   constraint` and the transaction aborts with the index untouched.

Sequence diagram — full staged commit, lock span marked (the `║`
column shows the single uninterrupted `bulk_lock` hold; each arrow pair
is one complete `bulk_post_locked` POSTED/DONE handshake with its own
barriers and latches, elided for space):

```
 BACKEND B                                  WORKER W                       GPU
 =========                                  ========                       ===
 ║ LWLockAcquire(bulk_lock)   <=== LOCK HELD FOR THE ENTIRE SEQUENCE ===>
 ║ TXN_BEGIN ----------------------------->  rgi_stage_begin  (clears ANY stale
 ║                                           staging, even a dead pid's)
 ║            <----------------------------  stage_owner = B.pid; ok=1
 ║ STAGE_DEL chunk 1..ceil(nd/262144) ---->  owner check; vector append (no GPU)
 ║ STAGE_UPD chunk 1..ceil(nu/262144) ---->  owner check; vector append (no GPU)
 ║ STAGE_INS chunk 1..ceil(ni/262144) ---->  owner check; vector append (no GPU)
 ║   [any chunk NACKs (ok=0)?  -> TXN_ABORT -> release lock -> ereport(ERROR);
 ║    index untouched, staging dropped]
 ║ TXN_COMMIT ----------------------------->  rgi_stage_commit:
 ║                                            VALIDATE (mutates NOTHING):
 ║                                              host dup-scan of ins keys
 ║                                              batched find of ins keys ----> find kernel(s)
 ║                                              hit? drop staging, dup=k, FAIL
 ║                                            APPLY (no failure path exists):
 ║                                              batched erases  -------------> erase kernel(s)
 ║                                              batched upserts -------------> insert kernel(s)
 ║            <----------------------------  ok / dup_key; stage_owner = 0
 ║ read ok, dup_key
 ║ LWLockRelease(bulk_lock)   <=== LOCK RELEASED ===>
 ok=1 -> PostgreSQL txn commits
 ok=0 -> ereport(unique_violation) -> PostgreSQL txn aborts; GPU index is
         bit-identical to its pre-BEGIN state (regression: atomic_test.sql,
         300,001 rows, duplicate planted in the LAST chunk)
```

The atomicity argument, stated once, completely — these three
properties together ARE the proof, and all three must be re-derived if
any one of them changes (WP2 changes the first):

1. **Exclusion**: `bulk_lock` spans BEGIN→COMMIT, so no other
   backend's request of ANY kind (stage chunk, demo write, legacy
   bulk write, snapshot) executes between this commit's steps. The
   staged set the worker validates is exactly and only this
   transaction's writes, and the index state validation reasons about
   is the state apply will mutate.
2. **Ownership fail-closed**: `stage_owner` pid tagging means that even
   if exclusion were violated (protocol bug, lock removed), foreign
   stage/commit requests NACK rather than splice; and TXN_BEGIN's
   unconditional reset means a crashed backend's residue is wiped, not
   inherited.
3. **Validate-before-mutate**: the engine checks every failure
   condition (intra-set duplicates, existing keys) before the first
   mutating kernel; the apply phase consists only of operations with no
   failure mode (erase-absent = no-op; upsert cannot conflict). So
   "failed commit" and "partially applied" are disjoint by
   construction — NOT because commit is one kernel (it is several).

History note (why the protocol looks "too careful"): the first commit
implementation applied deletes, then PK-checked inserts chunk by chunk;
a duplicate in chunk 2 left chunk 1 and all deletes applied, with no
undo. An external reviewer constructed the case; the TXN_* protocol and
`atomic_test.sql` are the response. The care is scar tissue — keep it.

#### `gpu_svc_snapshot_all` (lines 436–469) — backend side; full-table reads

Used by the FDW whenever pushdown yields no keys (full scans,
non-pushable quals). Anatomy:

1. Availability gate (444–446).
2. palloc initial result arrays at one page (`cap = GPU_SVC_BULK_CAP`,
   lines 439, 448–449) in the CALLER's current memory context — for a
   scan that is the executor's per-query context, so the arrays die
   with the query; no explicit free, no leak.
3. `LWLockAcquire(bulk_lock)` (451) — held across ALL pages.
4. `SVC_SNAPSHOT` (452): worker freezes the live-key set, returns page
   0 and the has-more flag.
5. Page loop (453–465): record `pc = bulk_count` and `more = bulk_ok`
   FIRST (454–455) — both fields will be overwritten by the next
   handshake — then grow the result arrays geometrically
   (`while (m + pc > cap) cap *= 2; repalloc`, 456–460), `memcpy` the
   page in (461–462), and if `more`, post `SVC_SNAPSHOT_NEXT` (464).
6. Release (466); return arrays + count (468).

Sequence diagram — paged snapshot (lock span marked):

```
 BACKEND B                                  WORKER W                       GPU
 =========                                  ========                       ===
 palloc ks/vs (1 page)
 ║ LWLockAcquire(bulk_lock)   <=== LOCK HELD ACROSS ALL PAGES ===>
 ║ SNAPSHOT ------------------------------>  snap_total = rgi_snapshot_begin
 ║                                           (FREEZES the live-key list);
 ║                                           snap_off = 0
 ║                                           page 0: batched find ---------> find kernel
 ║            <----------------------------  bulk_count=m0, bulk_ok=more?
 ║ repalloc if needed; memcpy page 0
 ║ SNAPSHOT_NEXT (if more) --------------->  page 1: batched find ---------> find kernel
 ║            <----------------------------  bulk_count=m1, bulk_ok=more?
 ║   ... repeat until bulk_ok == 0 ...
 ║ LWLockRelease(bulk_lock)   <=== LOCK RELEASED ===>
 return ks, vs, m   (palloc'd; freed with the query's memory context)
```

Why the lock must span pages — the consistency argument, explicitly:
the engine freezes the KEY list at SNAPSHOT, but each page's VALUES are
fetched live from the GPU at page time. If a writer ran between page k
and page k+1 it could (a) change values the snapshot will read in later
pages while earlier pages carry pre-write values — a torn read of a
single logical snapshot — or (b) delete keys, making later pages
silently shrink relative to the frozen list (the engine tolerates this
by compaction, but the result would mix two logical points in time).
Holding `bulk_lock` across pages excludes ALL writers (commits hold the
same lock; demo writes hold the same lock; legacy bulk writes hold the
same lock), so every page reads the same engine state: a consistent
frozen view. The cost is equally explicit: a snapshot of a large table
holds the global lock for its full duration — a multi-page snapshot is
the longest lock hold in the system and the worst-case latency a
concurrent commit can observe. Pre-WP2 this is accepted; WP2 must
re-derive snapshot consistency under concurrent reads (frozen key list
helps; value stability must come from commit exclusivity: no commit may
interleave a snapshot's pages — see "How to modify safely").

`bulk_ok` here means has-more (1 = more pages), the documented overload
of the success flag. A PK-violation-style failure cannot occur on the
snapshot path, which is the only reason the overload is sound.

#### `gpu_svc_insert` / `gpu_svc_lookup` (lines 471–485) — SQL demo functions

`PG_FUNCTION_INFO_V1`-declared (lines 73–74) C functions, exposed by
`pg_rgi_fdw--1.0.sql` as `gpu_svc_insert(bigint, bigint)` and
`gpu_svc_lookup(bigint) returns bigint`. Thin wrappers over
`gpu_svc_dispatch`: insert posts `SVC_INSERT` and returns void; lookup
posts `SVC_LOOKUP` and returns NULL when not found (lines 483: `if
(!found) PG_RETURN_NULL()` — note the SQL function must NOT be declared
STRICT-with-NOT-NULL assumptions for that to be meaningful; it returns
SQL NULL for absent keys, the natural KV-get shape).

Both cast through `(uint64) PG_GETARG_INT64(...)` — SQL bigints are
signed; the index keys are unsigned; negative bigints map to huge
uint64s, round-tripping consistently. Same convention as the FDW.

Their purpose, one more time because misuse is cheap: demonstrate
cross-session sharing through the worker (insert in session A, lookup
in session B — `svc_a.sql` / `svc_b.sql`) and smoke-test the slot
transport. They bypass the FDW's transaction buffer (a
`gpu_svc_insert` inside an aborted transaction STAYS in the index) and
bypass PK enforcement (inserting an existing key silently upserts).
They share `bulk_lock`, so they cannot corrupt anyone else's commit —
they can only surprise the user who expected transactional semantics.

#### `_PG_init` (lines 488–509) — postmaster side; registration

Walked structurally in the machinery explainer; the checklist version:

```c
if (!process_shared_preload_libraries_in_progress) return;   /* 492: preload-only */
RequestAddinShmemSpace(gpu_svc_shmem_size());                /* 494 */
RequestNamedLWLockTranche("gpu_service", 2);                 /* 495: [0]=alloc,[1]=bulk */
prev_shmem_startup_hook = shmem_startup_hook;                /* 496: chain */
shmem_startup_hook = gpu_svc_shmem_startup;                  /* 497 */
/* worker registration, lines 499-508: */
worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
worker.bgw_restart_time = 5;                                 /* restart 5s after exit */
bgw_name = bgw_type = "pg_gpu_service";
bgw_library_name = "pg_rgi_fdw"; bgw_function_name = "gpu_service_main";
worker.bgw_notify_pid = 0;                                   /* nobody waits for startup */
RegisterBackgroundWorker(&worker);
```

Notes: `bgw_name`/`bgw_type` "pg_gpu_service" is what appears in `ps`
and `pg_stat_activity` (`backend_type`); `bgw_notify_pid = 0` means no
process is notified when the worker starts — clients discover it via
`worker_pid` in shmem. There is exactly ONE `_PG_init` in the
`pg_rgi_fdw.so` library even though the library is built from two .c
files — it lives here, and `pg_rgi_fdw.c` (which carries
`PG_MODULE_MAGIC`) must not define another; if a refactor splits the
files into separate libraries, the FDW half loses the worker unless
registration moves with it.

<!-- APPEND -->
