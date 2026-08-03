# Section 04 — The Foreign Data Wrapper

This section documents the PostgreSQL-facing layer of the GPU-OLTP prototype:
the foreign data wrapper extension `pg_rgi_fdw`. It is written for an engineer
who knows C and general systems programming but has never seen PostgreSQL
extension internals. Everything PostgreSQL-specific (the FDW callback
machinery, junk columns, memory contexts, transaction callbacks, PGXS) is
explained inline, in the order it is needed.

Files covered:

| File | Role |
|---|---|
| `pg_rgi_fdw/pg_rgi_fdw.c` | The FDW proper: planner hooks, scan/modify callbacks, the per-transaction write buffer, the commit protocol hook. The main subject. |
| `pg_rgi_fdw/pg_rgi_fdw.control` | Extension metadata consumed by `CREATE EXTENSION`. |
| `pg_rgi_fdw/pg_rgi_fdw--1.0.sql` | The SQL installed by `CREATE EXTENSION`: handler/validator functions, the FDW object, two test functions. |
| `pg_rgi_fdw/Makefile` | PGXS build that links the engine shared library and the CUDA runtime. |

Adjacent files referenced but documented elsewhere in the primer:

- `pg_rgi_fdw/pg_gpu_service.{c,h}` — the shared-memory transport to the GPU
  service background worker (`gpu_svc_bulk`, `gpu_svc_txn_commit`,
  `gpu_svc_snapshot_all`). This section quotes its contract where the FDW
  depends on it; the worker's own section documents its internals.
- `engine/rgi_oltp_engine.h` — the C ABI over the RGI CUDA engine. Included at
  `pg_rgi_fdw.c:41` but not called directly from this file; every engine
  interaction goes through the worker.

---

## File: pg_rgi_fdw.c

~640 lines of C. Built for PostgreSQL 14 (the file header at lines 1-16 says
so explicitly, and the `AddForeignUpdateTargets` signature at lines 460-466 is
the PG14+ one — this matters; see the walkthrough). It implements a writeable
foreign table with the fixed schema `kv(k bigint, v bigint)` whose storage is
an RGI `GPUChainHashtable` living in the GPU service worker's CUDA context.

### Purpose

`pg_rgi_fdw.c` is the translation layer between PostgreSQL's row-at-a-time
executor and the GPU engine's batch-at-a-time execution model. Its job
decomposes into four responsibilities, each visible as a region of the file:

1. **Planner integration with key pushdown** (lines 173-344). Convince the
   PostgreSQL planner to scan the foreign table at all
   (`rgiGetForeignRelSize`/`Paths`), and at plan time mine the query's WHERE
   clause for primary-key equality predicates (`k = 7`, `k IN (1,2,3)`,
   `k = ANY($1)`) so a point query becomes one batched GPU lookup instead of a
   full-table snapshot. Pushdown is best-effort and superset-safe: PostgreSQL
   re-evaluates every qual on every returned tuple, so the worst failure mode
   of *not* pushing down is a slow snapshot, never a wrong answer — provided
   the keys that *are* pushed come from genuine equality predicates (the
   `is_equal_op` guard at lines 211-218 enforces exactly that; see the
   walkthrough for why getting it wrong would be a missing-rows bug, not a
   superset).

2. **The per-transaction write buffer** (lines 53-171, plus the
   `ExecForeign*` callbacks at 503-596). DML against the foreign table never
   touches the GPU during the statement. Each `INSERT`/`UPDATE`/`DELETE` row
   becomes an entry in a per-backend hash table (`TxnEntry`, lines 59-71)
   living in the transaction's memory context. This is what makes the table
   *transactional*: `ROLLBACK` simply drops the buffer (the GPU never saw
   anything — no dirty reads by construction, no undo needed), and `COMMIT`
   flushes the entire write set as one validated, all-or-nothing batch. It is
   also what makes the table *fast* for multi-row statements: a 100k-row
   `INSERT ... SELECT` becomes one staged batch and two GPU kernel launches
   instead of 100,000 per-row PCIe round trips (the measured difference is
   ~13 kop/s vs ~790 kop/s — see the design-rationale subsection).

3. **Read-your-writes overlay** (inside `rgiBeginForeignScan`, lines
   346-429). Because writes are deferred, reads inside the same transaction
   must merge the GPU's committed state with the transaction's buffered,
   uncommitted state, or `INSERT (5,5); SELECT * WHERE k=5` would return
   nothing. Both read paths — point/multi-get and full snapshot — consult the
   buffer first and overlay it over worker results.

4. **The atomic commit hook** (lines 99-171). A transaction callback
   registered with PostgreSQL (`rgi_xact_cb`) fires at `PRE_COMMIT`,
   classifies the buffer into delete/update/insert arrays, and ships them to
   the worker via `gpu_svc_txn_commit`, which validates the whole write set
   (PK/UNIQUE) *before mutating anything* and applies it only if validation
   passes. A duplicate key surfaces as a standard PostgreSQL
   `unique_violation` error that aborts the entire transaction with the GPU
   index byte-identical to its pre-transaction state.

What this file deliberately is **not**:

- It is not a tuple store. The schema is hard-wired two `bigint` columns;
  values round-trip through the engine's `uint32_t` value type, so it is
  presented project-wide as a *key → row-id index prototype* (explainer §22).
- It is not concurrency-controlled between transactions. Isolation is
  approximately Read Committed plus read-your-writes: no dirty reads, no
  partial commits, but also no write-write conflict detection — two
  concurrent read-modify-write transactions can lose an update. This is
  documented, deliberate scoping; WP3 (OCC) closes it.
- It does not support `SAVEPOINT`/subtransactions. There is no
  `RegisterSubXactCallback` anywhere in the file — deliberately no lying stub
  either (see design rationale). WP4 closes it, after WP3's buffer redesign.

Anyone modifying this file is almost certainly doing WP3 or WP4 work. Read
`plan/WP3_occ_and_row_identity_buffer.md` and `plan/WP4_subtransactions.md`
after this section; the "How to modify safely" subsection at the end maps
their tasks onto specific functions and lines here.

### The FDW callback lifecycle

This subsection is a standalone explainer of PostgreSQL's foreign data
wrapper machinery — read it even if the goal is a one-line change, because
every function in this file only makes sense as a slot in this machinery.

#### What an FDW is

PostgreSQL's planner and executor know how to scan and modify *heap tables*
(its native row storage). A **foreign data wrapper** is the official
extension point for "a table whose storage lives somewhere else" — another
PostgreSQL server (`postgres_fdw`), a CSV file (`file_fdw`), or, here, a hash
table in GPU memory. The contract is a **vtable of callbacks**: the extension
provides one C function (the *handler*) that returns a palloc'd `FdwRoutine`
struct with ~15 function pointers filled in, and the planner/executor calls
through those pointers at well-defined points in a query's life. If a systems
analogy helps: `FdwRoutine` is to a foreign table what a `file_operations`
struct is to a character device — the kernel (PostgreSQL core) drives the
lifecycle; the driver (this file) implements the hooks.

The wiring from SQL to C happens in three layers, all visible in this
extension's files:

1. `pg_rgi_fdw--1.0.sql:3-11` declares two SQL-callable C functions
   (`pg_rgi_fdw_handler`, `pg_rgi_fdw_validator`) and creates the FDW object
   pointing at them.
2. `pg_rgi_fdw.c:612-632` implements the handler: it builds the `FdwRoutine`
   and returns it. `pg_rgi_fdw.c:634-638` implements the validator (a no-op
   here — this FDW takes no options).
3. The user then runs `CREATE SERVER rgi FOREIGN DATA WRAPPER pg_rgi_fdw;`
   and `CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;` — from
   that point, `kv_rgi` is, to every client and every SQL construct, just a
   table.

Callbacks the handler does *not* set (e.g. `ExplainForeignScan`,
`AnalyzeForeignTable`, `GetForeignJoinPaths`, `PlanDirectModify`) are simply
left NULL in the `makeNode`-zeroed struct; PostgreSQL treats NULL as "feature
not provided" and falls back to generic behavior (no custom EXPLAIN lines, no
foreign-side join pushdown, no direct modify — every UPDATE/DELETE goes
through the scan-then-modify path described below).

#### The scan path: GetForeignRelSize → GetForeignPaths → GetForeignPlan → Begin/Iterate/End

A `SELECT` against a foreign table flows through two phases.

**Plan time** (inside the optimizer, once per query — or once per prepared
statement when a generic plan is cached):

- `GetForeignRelSize(root, baserel, oid)` — the planner asks "how many rows
  should I expect?" so it can cost joins above this scan. There are no
  statistics for GPU-resident data, so this file hardcodes 1000
  (`pg_rgi_fdw.c:174-179`). Crude, but only join planning quality depends on
  it, and the test workloads are single-table.
- `GetForeignPaths(root, baserel, oid)` — the FDW must add at least one
  `ForeignPath` (a candidate access strategy with a cost) to the relation,
  or the planner cannot plan the query at all. This file adds exactly one
  cheap path (`pg_rgi_fdw.c:181-189`); there is no path differentiation
  (e.g. a cheaper parameterized path for point lookups) — pushdown is decided
  later, in `GetForeignPlan`, not via competing paths.
- `GetForeignPlan(root, baserel, oid, best_path, tlist, scan_clauses, outer_plan)`
  — the chosen path is turned into an executable `ForeignScan` plan node.
  This is the *only* plan-time point where the FDW sees the query's WHERE
  clauses (`baserel->baserestrictinfo`) and may stash private data into the
  plan. This file walks the restriction clauses, extracts constant equality
  keys, and rides them to the executor in the plan's `fdw_private` field
  (`pg_rgi_fdw.c:271-288`). Crucially it also keeps **all** clauses in the
  plan's qual list (`scan_clauses` → `qpqual`), which is what makes pushdown
  superset-safe — next paragraph.

**The recheck property (why pushdown cannot cause wrong results).** A
ForeignScan plan node has a qual list (`qpqual`); the executor re-evaluates
those expressions on **every tuple** the FDW returns and discards
non-matching tuples. An FDW that fully trusts its remote filtering may remove
pushed-down clauses from `qpqual` to skip the redundant evaluation
(`postgres_fdw` does this); this FDW deliberately does not
(`pg_rgi_fdw.c:284-285`: `scan_clauses` is passed through
`extract_actual_clauses` and kept as the plan qual, with the comment "keep
scan_clauses as qpqual so Postgres rechecks"). Consequence: the FDW is
allowed to return a **superset** of the matching rows — extra rows are
filtered above it — so any pushdown bug that *over*-returns is invisible to
users. The one failure mode recheck cannot repair is *under*-returning:
if the FDW returns fewer rows than match, the rows are simply missing from
the result. That asymmetry is the lens for reading every guard in the
pushdown code: each one exists to ensure the FDW only narrows the scan to a
key list when that key list provably covers all matching rows.

**Execution time** (inside the executor, once per scan; possibly repeated via
ReScan):

- `BeginForeignScan(node, eflags)` — open the scan. This file does all the
  real work here: it decides between the point-lookup path and the snapshot
  path, performs the GPU traffic, merges in the transaction buffer, and
  materializes the **entire result** into in-memory arrays
  (`pg_rgi_fdw.c:346-429`). One important convention: when `eflags` contains
  `EXEC_FLAG_EXPLAIN_ONLY` (plain `EXPLAIN` without `ANALYZE`), the executor
  builds the node but will never pull tuples — the callback must do nothing
  expensive and may leave its state NULL (line 356 returns early; line 437
  defensively handles the NULL state).
- `IterateForeignScan(node)` — return one tuple per call, or an empty slot to
  signal end-of-scan. Here it is a trivial cursor over the materialized
  arrays (`pg_rgi_fdw.c:431-447`).
- `ReScanForeignScan(node)` — restart the scan from the top (used by the
  inner side of nestloop joins, among others). Here: reset the cursor to 0
  *without* re-running the GPU query or re-overlaying the buffer
  (`pg_rgi_fdw.c:449-453`) — the materialized result is replayed as-is.
- `EndForeignScan(node)` — close, free resources. Here a no-op
  (`pg_rgi_fdw.c:454-457`): result arrays are palloc'd, and palloc'd memory
  belongs to a *memory context* (an arena) that the executor destroys
  wholesale at query end. This is the single most important PostgreSQL memory
  idiom for a newcomer: `palloc` ≈ arena-malloc into the "current" context;
  most cleanup is implicit via context destruction; explicit `pfree` is the
  exception. (The file *does* use raw `malloc`/`free` in one spot, for a
  scratch key array — lines 332, 363, 396 — see the walkthrough for why
  that's a latent-leak pitfall.)

ASCII sequence diagram — `SELECT v FROM kv_rgi WHERE k = 50000;` with
plan-time pushdown:

```
 psql            PostgreSQL core                pg_rgi_fdw.c             pg_gpu_service.c        GPU worker proc
  |                    |                             |                          |                      |
  |-- SQL text ------->|                             |                          |                      |
  |                    | parse / analyze             |                          |                      |
  |                    |                             |                          |                      |
  |                    |-- GetForeignRelSize ------->| rows := 1000       (174) |                      |
  |                    |-- GetForeignPaths --------->| add 1 ForeignPath  (181) |                      |
  |                    |-- GetForeignPlan ---------->| walk baserestrictinfo    |                      |
  |                    |                             |  k = 50000: OpExpr,      |                      |
  |                    |                             |  opname "=" verified,    |                      |
  |                    |                             |  int4 const widened to   |                      |
  |                    |                             |  int8 Const        (224) |                      |
  |                    |<-- ForeignScan plan --------| fdw_private=[50000]      |                      |
  |                    |    (qual kept for recheck)  | qpqual=[k=50000]   (286) |                      |
  |                    |                             |                          |                      |
  |                    | executor starts             |                          |                      |
  |                    |-- BeginForeignScan -------->| pushed != NIL:           |                      |
  |                    |                             |  keys=[50000]      (359) |                      |
  |                    |                             |  txn_buf overlay: miss   |                      |
  |                    |                             |                    (384) |                      |
  |                    |                             |-- gpu_svc_bulk --------->| fill bulk channel,   |
  |                    |                             |   (SVC_FIND_MANY,        | state=POSTED,        |
  |                    |                             |    wq=[50000])     (393) | SetLatch(worker) --->| rgi_find_many:
  |                    |                             |                          |                      |  1 warp-coop
  |                    |                             |                          |                      |  GPU find launch
  |                    |                             |                          |<--- state=DONE ------|
  |                    |                             |<-- (k,v) pairs ----------| SetLatch(client)     |
  |                    |                             | materialize result       |                      |
  |                    |                             | arrays, st->n=1    (397) |                      |
  |                    |-- IterateForeignScan ------>| virtual tuple #1   (431) |                      |
  |                    |   (executor RECHECKS the    |                          |                      |
  |                    |    qual k=50000 on it)      |                          |                      |
  |                    |-- IterateForeignScan ------>| empty slot => EOF        |                      |
  |                    |-- EndForeignScan ---------->| no-op              (454) |                      |
  |<-- 1 row ----------|                             |                          |                      |
```

Two things to internalize from this diagram. First, all GPU traffic happens
in `BeginForeignScan` — `Iterate` never blocks on the device; the scan is
fully materialized up front. Second, the recheck step: even though the FDW
already filtered to key 50000, the executor evaluates `k = 50000` again on
the returned tuple. That redundancy is the safety net the whole pushdown
design leans on.

#### The modify path: AddForeignUpdateTargets → PlanForeignModify → Begin → ExecForeignInsert/Update/Delete → End

DML against a foreign table is structured as *scan-then-modify*: the planner
builds a `ModifyTable` plan node whose input subplan is a scan producing the
rows to modify (for `UPDATE`/`DELETE`, a scan of the foreign table itself;
for `INSERT`, whatever produces the new rows — a VALUES list, a SELECT,
etc.). The executor pulls one row at a time from the subplan and calls the
FDW's per-row modify callback for each.

This raises the **row identity problem**, and understanding it is a
prerequisite for understanding half this file. For a heap table, the executor
identifies "which physical row does this UPDATE target" via the row's `ctid`
(physical block/offset address), carried through the plan as a hidden extra
column. A foreign table has no ctid — the FDW must declare what *it* needs to
identify a row. That is the job of:

- `AddForeignUpdateTargets(root, rtindex, target_rte, target_relation)` —
  called at plan time for the target relation of an UPDATE/DELETE. The FDW
  injects expressions into the plan as **junk columns**: columns that flow
  through the executor alongside the user-visible ones, are consumable by the
  modify node, and are stripped before results reach the client. (Mental
  model from CPU-land: a junk column is a tag riding with the row through the
  pipeline — like a physical-register tag riding with a µop so retirement
  knows its architectural destination — present for bookkeeping, invisible in
  the architectural result.) This FDW injects exactly one junk column,
  `rgi_key`: a `Var` referencing column 1 (`k`) of the target relation
  (`pg_rgi_fdw.c:460-466`). The key *is* the row identity — this is a
  hash-keyed store, there is no other notion of address.

  **PG14 signature alert:** before PostgreSQL 14 this callback received
  `(Query *parsetree, RangeTblEntry *, Relation)` and appended a junk TLE to
  the parse tree manually; PG14 changed it to
  `(PlannerInfo *root, Index rtindex, RangeTblEntry *, Relation)` using
  `add_row_identity_var()`. This file implements the PG14+ form. Code or
  examples copied from older FDW tutorials (most of the internet) will not
  compile or, worse, will mislead about the mechanism.

- `PlanForeignModify(root, plan, resultRelation, subplan_index)` — a hook to
  precompute plan-time private state for the modify (e.g. a remote SQL
  string in postgres_fdw). Here there is nothing to precompute; returns NIL
  (`pg_rgi_fdw.c:468-472`).

- `BeginForeignModify(mtstate, rinfo, fdw_private, subplan_index, eflags)` —
  executor-startup for the modify. Here: allocate per-statement state,
  resolve the junk column's attribute number in the subplan's targetlist
  (needed to fetch it per-row later), and register the transaction callback
  (`pg_rgi_fdw.c:474-489`).

- `ExecForeignInsert / ExecForeignUpdate / ExecForeignDelete
  (estate, rinfo, slot, planSlot)` — called once per row. `slot` carries the
  new row image (for INSERT/UPDATE); `planSlot` carries the subplan's output
  row *including junk columns* — the old key is fetched from `planSlot` via
  the attribute number resolved in Begin. In this FDW these callbacks **only
  mutate the in-memory transaction buffer** (lines 503-596) — no GPU traffic,
  no shared-memory traffic, ~hash-table-insert cost per row. Returning the
  slot (vs NULL) tells the executor the row was processed (it feeds
  `RETURNING` and the command's row count).

- `EndForeignModify(estate, rinfo)` — statement-end. Here a deliberate no-op
  (`pg_rgi_fdw.c:598-603`): the flush boundary is the *transaction*, not the
  statement. (The file's header comment at lines 4-5 says writes are flushed
  in `EndForeignModify` — that is **stale documentation**; the code comment
  at lines 601-602 and the actual mechanism say COMMIT. Trust the code.)

- `IsForeignRelUpdatable(rel)` — a capability bitmask the core consults to
  decide whether INSERT/UPDATE/DELETE against this table are allowed at all
  (`pg_rgi_fdw.c:605-609` returns all three bits).

The last piece is **transaction callbacks**, which are not FDW-specific but
generic extension machinery (`RegisterXactCallback`, from
`access/xact.h`): any loaded C extension can ask to be called at transaction
lifecycle events. The events used here:

- `XACT_EVENT_PRE_COMMIT` — fired *before* the transaction is irrevocably
  committed; an `ereport(ERROR)` thrown here still cleanly aborts the whole
  transaction. This is where the buffered write set is applied to the GPU
  (`pg_rgi_fdw.c:149-151`) — and the reason a GPU-side PK violation can
  surface as a normal SQL error. Hooking `XACT_EVENT_COMMIT` instead would be
  a serious bug: at that point the transaction is already committed and an
  ERROR escalates to a PANIC (database-wide crash-restart).
- `XACT_EVENT_ABORT` / `COMMIT` / parallel variants — used only to reset the
  static buffer pointer (`pg_rgi_fdw.c:152-157`); the buffer memory itself
  lives in `TopTransactionContext`, a memory context PostgreSQL destroys
  automatically at transaction end, so "drop the buffer" costs literally one
  pointer assignment.

ASCII sequence diagram — `BEGIN; INSERT INTO kv_rgi VALUES (7,70); COMMIT;`:

```
 psql            PostgreSQL core                pg_rgi_fdw.c               pg_gpu_service.c       GPU worker proc
  |                    |                             |                          |                      |
  |-- BEGIN ---------->| (no FDW involvement)        |                          |                      |
  |-- INSERT --------->| plan: ModifyTable(INSERT)   |                          |                      |
  |                    |   over a Values scan        |                          |                      |
  |                    |   (AddForeignUpdateTargets  |                          |                      |
  |                    |    NOT called: INSERT needs |                          |                      |
  |                    |    no row identity)         |                          |                      |
  |                    |-- BeginForeignModify ------>| st->op=CMD_INSERT  (479) |                      |
  |                    |                             | RegisterXactCallback     |                      |
  |                    |                             |   (once/backend)   (487) |                      |
  |                    |-- ExecForeignInsert ------->| txn_get_buf() creates    |                      |
  |                    |   slot=(7,70)               |  HTAB in TopTransaction- |                      |
  |                    |                             |  Context            (83) |                      |
  |                    |                             | HASH_ENTER key=7   (511) |                      |
  |                    |                             | not found => entry:      |                      |
  |                    |                             |  {v=70, inserted=T,      |                      |
  |                    |                             |   deleted=F, replace=F,  |                      |
  |                    |                             |   kc_*=F}          (521) |                      |
  |                    |                             |   *** NO GPU TRAFFIC *** |                      |
  |                    |-- EndForeignModify -------->| no-op (flush is at       |                      |
  |                    |                             |  COMMIT, not stmt) (598) |                      |
  |<-- INSERT 0 1 -----|                             |                          |                      |
  |                    |                             |                          |                      |
  |-- COMMIT --------->| CommitTransaction()         |                          |                      |
  |                    |-- XACT_EVENT_PRE_COMMIT --->| rgi_xact_cb        (149) |                      |
  |                    |                             | rgi_txn_apply:           |                      |
  |                    |                             |  classify buffer:        |                      |
  |                    |                             |  ins=[(7,70)], upd=[],   |                      |
  |                    |                             |  del=[]        (123-131) |                      |
  |                    |                             |-- gpu_svc_txn_commit --->| ONE bulk_lock hold:  |
  |                    |                             |              (134)       |  TXN_BEGIN           |
  |                    |                             |                          |  STAGE_INS [(7,70)]->| stage in worker
  |                    |                             |                          |  TXN_COMMIT -------->| VALIDATE:
  |                    |                             |                          |                      |  host dup-scan +
  |                    |                             |                          |                      |  batched GPU find
  |                    |                             |                          |                      |  over staged keys
  |                    |                             |                          |                      | APPLY (only if ok):
  |                    |                             |                          |                      |  batched erase +
  |                    |                             |                          |                      |  batched upsert
  |                    |                             |<-- ok=1 -----------------|                      |
  |                    |                             | txn_buf := NULL    (135) |                      |
  |                    |<-- (returns) ---------------|                          |                      |
  |                    | transaction commits;        |                          |                      |
  |                    | TopTransactionContext       |                          |                      |
  |                    | destroyed => buffer freed   |                          |                      |
  |                    |-- XACT_EVENT_COMMIT ------->| txn_buf=NULL (already)   |                      |
  |<-- COMMIT ---------|                             |              (152-157)   |                      |
```

The failure branch of the same diagram: if key 7 already existed on the GPU,
the worker's VALIDATE step finds it, applies **nothing**, and returns
`ok=0, dup=7`; `rgi_txn_apply` raises `ERRCODE_UNIQUE_VIOLATION`
(`pg_rgi_fdw.c:137-141`) from inside PRE_COMMIT; PostgreSQL converts the
in-flight COMMIT into an ABORT; the client sees
`ERROR: duplicate key value violates unique constraint on GPU table` followed
by `ROLLBACK` semantics — exactly the shape a heap table's deferred
constraint failure has. And the ROLLBACK branch: `XACT_EVENT_PRE_COMMIT`
never fires, `XACT_EVENT_ABORT` resets the pointer, the context destruction
frees the memory; total GPU work performed: zero.

#### Statement-versus-transaction timing summary

A compact reference for when each callback fires (this trips up everyone new
to FDWs):

| Callback | Fires | Per |
|---|---|---|
| GetForeignRelSize/Paths/Plan | plan time | query (or once per cached plan) |
| AddForeignUpdateTargets | plan time | UPDATE/DELETE target relation |
| PlanForeignModify | plan time | modified relation |
| BeginForeignScan / BeginForeignModify | executor startup | statement (per scan/modify node) |
| IterateForeignScan, ExecForeignInsert/Update/Delete | executor run | **row** |
| ReScanForeignScan | executor run | scan restart |
| EndForeignScan / EndForeignModify | executor shutdown | statement |
| rgi_xact_cb (PRE_COMMIT / COMMIT / ABORT) | transaction end | **transaction** |

The per-row callbacks are the hot path (hash-table operations only, by
design); the per-transaction callback is where all deferred GPU work
concentrates.

### Design rationale

Five decisions shape this file. Each is summarized here with its
alternatives; the project explainer (`comp_arch_db_explainer/
FULL_PROJECT_EXPLAINER.md`, Part IV) carries the full ledger.

**1. FDW as the integration rung (vs. Table Access Method, CustomScan, core
patch).** The four ways to put foreign execution under PostgreSQL SQL, in
increasing invasiveness: FDW (replace a table's storage behind callbacks;
writeable since PG 9.3; days of effort; *no* MVCC integration), TAM (replace
the heap itself, PG12+; MVCC visibility hooks; weeks-months), CustomScan
(replace plan *operators* — PG-Strom's route; read-only by construction,
which is precisely why PG-Strom never runs writes on GPU), and patching core
(unmaintainable). FDW was chosen as the fastest legitimate path to "real SQL,
real clients, real DML, zero core patches." Its two structural costs are both
visible in this file: planner opacity (the planner does not understand the
foreign side, hence the *manual* pushdown machinery at lines 191-344) and no
MVCC hooks (hence isolation scoped to RC+RYW and implemented entirely in user
space via the buffer — the FDW is outside PostgreSQL's visibility machinery
and cannot piggyback on snapshots). TAM is the stated fall-semester step;
everything below this file (worker protocol, engine C ABI) was shaped to
survive that move.

**2. Buffer writes; flush at the commit boundary (vs. dispatch per row).**
The first implementation dispatched each row to the GPU as the Exec callback
received it — the natural shape FDW callbacks suggest — and measured
≈87 µs/row (a PCIe round trip per row, ≈13 kop/s: the single worst number in
the project's history). The fix is this file's core idea: the per-row
callbacks write to a host-side hash table; the commit callback flushes the
net write set as one staged batch. Batch size therefore equals whatever the
transaction naturally carries — a 14k-row UPDATE commits as B=14k (deep in
the GPU's win region, ≈790 kop/s SQL-inclusive on the reference workload), a
single-row autocommit INSERT commits as B=1 (the latency floor; CPU
territory; stated honestly). The file header (lines 4-8) calls this the
latency/throughput tradeoff. Note the flush boundary is the *transaction*,
not the statement — `EndForeignModify` (598-603) intentionally does nothing —
because only at commit is the write set final (a later statement in the same
transaction may delete what an earlier one inserted), and because
transaction-atomicity then comes for free: one staged, validated batch is
naturally all-or-nothing.

**3. Validate-before-mutate commit (vs. apply-and-hope).** The commit
protocol's atomicity argument is *validation precedes any mutation*, not "one
kernel launch." The worker stages the full del/upd/ins set, runs PK/UNIQUE
validation (host dup-scan over staged inserts + one batched GPU find), and
only then applies — with apply composed exclusively of operations that have
no expected failure path (erase-of-absent is a defined no-op; upsert cannot
fail). The first implementation got this wrong (applied deletes, then
PK-checked inserts chunk-by-chunk, so a duplicate in chunk 2 left chunk 1 and
all deletes applied); an external reviewer caught it and the protocol was
rebuilt. The FDW-side residue of that contract is visible at lines 99-104
(the comment) and 134-141: a single fallible call, then a single error path.
WP3's contract explicitly preserves this: *the validate step must remain the
only fallible stage of commit* — OCC version checks belong inside it, never
in apply.

**4. Reject overlapping key-renames (vs. silently corrupting, vs.
re-architecting now).** The transaction buffer is keyed by key — it collapses
all operations on a key into one `TxnEntry`. That representation *cannot*
express a multi-row UPDATE that renames keys into each other (a swap
`1↔2`, or a chain `k=k+1` over adjacent keys): one row's new-key write and
another row's old-key tombstone land on the *same* entry and clobber each
other. Until a reviewer constructed the case, the code silently corrupted;
the options then were (a) keep corrupting, (b) re-architect the buffer around
row identity immediately, or (c) detect the overlap and raise
`feature_not_supported`, keeping non-overlapping renames (`k = k+100`) and
single-row renames working. (c) was chosen — "failing loudly on the
unrepresentable case" — implemented as the `kc_new`/`kc_old` flag pair and
two rejection branches (lines 552-557 and 569-574). (b) is exactly WP3's
buffer redesign: an ordered operation log computes net effects per *row*, for
which a swap is just two updates. When WP3 lands, the flags and both
rejection branches are deleted and the rejection tests flip to
success-expected.

**5. No SAVEPOINT support, and deliberately no stub.** Handling
`ROLLBACK TO SAVEPOINT` requires `RegisterSubXactCallback` plus per-savepoint
undo of the buffer. Against the key-collapsed buffer, per-savepoint undo
would require snapshotting/diffing entries — rework guaranteed to be thrown
away once WP3's op log lands (an op log undoes a savepoint by truncating to a
high-water mark). So the gap was left open *visibly*: there is no
SubXactCallback registration at all, rather than a registered do-nothing
callback that would make `ROLLBACK TO SAVEPOINT` appear to succeed while
silently keeping writes that should have been discarded. "A stub that lies is
worse than an absence." Today the failure mode is: buffered writes survive a
`ROLLBACK TO SAVEPOINT` that should have discarded them — documented in the
explainer's limitations (Part VIII) and closed by WP4 (2-4 days, hard
dependency on WP3).

A sixth, inherited decision worth restating because it explains otherwise
puzzling code: **one background worker owns the only CUDA context.**
PostgreSQL forks a *process* per connection; CUDA contexts are per-process;
therefore per-backend GPU access would mean N contexts and N index copies.
Hence every GPU interaction in this file is a `gpu_svc_*` call — a
shared-memory message to the single worker process — and never a direct
engine call, even though `rgi_oltp_engine.h` is included (line 41; the
include is for type visibility and is arguably vestigial). All bulk
operations serialize behind the worker's one lock; that serialization is the
known, named scaling ceiling, and it is also why the FDW can treat
`gpu_svc_txn_commit` as atomic: the worker holds the lock across the entire
stage/validate/apply sequence, so no other backend's operations interleave.

### Walkthrough

Every function in the file, in source order. Conventions used below: line
numbers refer to `pg_rgi_fdw/pg_rgi_fdw.c` as of this writing; "buffer" means
the per-transaction `TxnEntry` HTAB; "worker" means the GPU service
background worker reached via `pg_gpu_service.c`.

#### Module preamble (lines 1-47)

The header comment (lines 1-16) states the contract: a foreign table
`kv(k bigint, v bigint)` maps to an RGI `GPUChainHashtable`; writes are
buffered and flushed as one batched launch; built for PostgreSQL 14; one
engine per backend — **two of these claims are stale**. First, lines 4-5 say
writes flush in `EndForeignModify`; they flush at PRE_COMMIT
(`rgi_xact_cb` → `rgi_txn_apply`), and `EndForeignModify` is a documented
no-op (lines 598-603). Second, line 15 says "one engine per backend
(session-local data)"; the engine is *shared* — it lives in the GPU service
worker and all backends reach it through shared memory (the SQL file's
comment at `pg_rgi_fdw--1.0.sql:13` says "shared, multi-user index owned by
the bgworker", which is correct). Both staleness artifacts date from the
pre-worker, pre-commit-protocol revision. When touching this file, fix the
header rather than propagating it.

The includes (17-42) are standard FDW fare plus two project headers. Worth
knowing for modification work:

- `access/xact.h` — `RegisterXactCallback`, `XactEvent`. WP4 adds
  `RegisterSubXactCallback` from the same header.
- `foreign/fdwapi.h` — `FdwRoutine` and every callback typedef.
- `optimizer/*` — plan-time node manipulation (`add_path`,
  `create_foreignscan_path`, `make_foreignscan`, `extract_actual_clauses`,
  `add_row_identity_var` via `optimizer/appendinfo.h`).
- `utils/hsearch.h` — the dynahash HTAB used for the buffer.
- `utils/memutils.h` — `TopTransactionContext`.
- `rgi_oltp_engine.h` (41) — engine C ABI types; no direct engine calls
  remain in this file (all GPU access is via the worker), so this include is
  vestigial but harmless.
- `pg_gpu_service.h` (42) — the worker client API; the FDW's only window to
  the GPU.

`PG_MODULE_MAGIC` (44) is the ABI fingerprint every loadable PostgreSQL
module must embed exactly once — the server refuses to load a `.so` compiled
against a different major version. `PG_FUNCTION_INFO_V1` (46-47) declares the
two SQL-callable entry points with the version-1 calling convention; their
SQL declarations live in `pg_rgi_fdw--1.0.sql`.

#### `RgiScanState`, `RgiModifyState` (lines 50-51)

```c
typedef struct RgiScanState { uint64_t *keys; uint64_t *vals; uint64_t n, cur; } RgiScanState;
typedef struct RgiModifyState { AttrNumber key_junk; CmdType op; } RgiModifyState;
```

Per-statement executor state, hung off the nodes PostgreSQL provides for the
purpose (`node->fdw_state` for scans, set at line 428;
`rinfo->ri_FdwState` for modifies, set at line 488 — both are `void *` slots
the core never interprets).

- `RgiScanState`: the fully materialized scan result (`keys`/`vals`, palloc'd
  arrays of `n` rows) and a cursor (`cur`). Materializing the whole result in
  `BeginForeignScan` is a deliberate property, not laziness: it makes the
  scan immune to the buffer changing underneath it mid-statement. An
  `UPDATE kv_rgi SET ...` pipelines the foreign scan into the ModifyTable
  node, and each `ExecForeignUpdate` mutates the buffer *while the scan over
  that same table is still iterating* — if the scan consulted the buffer
  per-tuple it could see its own statement's writes (the classic Halloween
  problem). Snapshot-at-Begin sidesteps it entirely.
- `RgiModifyState`: `key_junk` is the attribute number of the `rgi_key` junk
  column inside the subplan's output row (resolved once in
  `BeginForeignModify`, used per-row in Update/Delete), and `op` records the
  statement type (set at 480, currently never read afterwards — `key_junk`
  validity already encodes what matters; harmless).

#### `TxnEntry` and the buffer's representational limit (lines 53-81)

```c
typedef struct TxnEntry
{
    int64 key;       /* hash key */
    int64 value;
    bool  deleted;   /* tombstone */
    bool  inserted;  /* created in this txn (PK-checked at commit) */
    bool  replace;   /* deleted earlier in THIS txn, then re-created */
    bool  kc_new;    /* NEW key of a key-changing UPDATE */
    bool  kc_old;    /* OLD key tombstoned by a key-changing UPDATE */
    bool  seen;      /* transient: matched a worker row during a scan */
} TxnEntry;
```

One entry per **key** touched by the transaction (dynahash entry; the first
field is the hash key, per `HASH_BLOBS` with `keysize = sizeof(int64)`). This
is the data structure WP3 replaces wholesale, so understanding precisely what
each flag means — and what the key-collapsed shape can and cannot represent —
is the core of onboarding to this file.

**`value` (62).** The row's current in-transaction value. Only meaningful
when the entry is live (`!deleted`). Set by Insert (521), value-Update (541),
key-change-Update on the new key (566); zeroed on the tombstoned old key
(577).

**`deleted` (63) — the tombstone.** True means: within this transaction, the
key currently has no live row. Set by Delete (594) and by the old-key half of
a key-changing Update (577); cleared whenever the key becomes live again
(Insert 521, both Update paths 541/566). Read paths treat
`deleted == true` as "key invisible" even if the GPU has a committed row for
it (385, 419) — that is how an uncommitted DELETE hides a committed row. At
commit, `deleted` (without `inserted`) classifies the entry into the delete
array (126).

**`inserted` (64) — created in this transaction.** True means: this
transaction claims to have *created* the live row under this key, so at
commit the key must be PK-checked against the GPU index (a committed row with
the same key is a `unique_violation`). Set by Insert (521) and by the new-key
half of a key-changing Update when the key was not already in the buffer
(564). Explicitly cleared when a value-Update (540) or Delete (593) creates
the entry — those operate on rows that must already exist on the GPU, and
classifying them as inserts would make commit PK-fail against the very rows
they target.

**`replace` (65-67) — the upsert escape hatch.** True means: this key *was
tombstoned earlier in this same transaction and then re-created*. The
sequence `DELETE k; INSERT k` (or `UPDATE ... SET k = freed-key`) is legal
even when `k` exists on the GPU: the commit's delete would remove the old row
and the insert recreate it. But a key-collapsed buffer cannot ship "delete
then insert" for one key as two operations (the entry is one slot), so the
classification collapses it into a single **upsert** (update-or-insert, no PK
check) — that is what `replace` marks. Set at Insert (519:
`e->replace = (found && e->deleted)`) and at the key-change new-key path
(563, same expression). At commit, `inserted && replace` routes the entry to
the *update* array instead of the PK-checked insert array (127-130). Without
this flag, `DELETE k; INSERT k; COMMIT` on a pre-existing key would
spuriously raise `unique_violation` (the staged insert would collide with the
committed row whose deletion is staged in the same batch — the worker
validates inserts against the *pre-commit* index state).

Known, documented artifact of this collapse (`WP3` plan, "Current state"):
`INSERT k` (where k pre-exists on GPU, so it *should* error) followed by
`DELETE k; INSERT k` in the same transaction does not error at statement 1 —
statement 1's PK check is deferred to commit, by which time the buffer has
collapsed the whole sequence into an upsert. The op-log buffer in WP3 is
expected to restore statement-1 erroring; the WP3 plan calls for deciding and
testing that semantics explicitly.

**`kc_new` / `kc_old` (68-69) — the rename-overlap detectors.** A
key-changing UPDATE (`UPDATE ... SET k = <different value>`) touches two
buffer entries: the new key (gets the row, with insert-like PK semantics) and
the old key (gets a tombstone). `kc_new` marks "this key was the destination
of a rename in this transaction"; `kc_old` marks "this key was the source of
a rename." Their only purpose is to detect, at the *next* rename, the overlap
the buffer cannot represent: if a rename's new key already carries `kc_old`
(some other row was renamed *away from* this key — its tombstone lives here)
or a rename's old key already carries `kc_new` (some other row was renamed
*into* this key — its live row lives here), then two distinct rows' state now
needs to occupy one entry, and the statement is rejected with
`ERRCODE_FEATURE_NOT_SUPPORTED` (552-557, 569-574). The flags are set at
565/576, cleared when an entry is created by a non-rename operation
(520, 540, 593, and partially 564/575). Both flags, both rejection branches,
and the explanatory comment block at 73-78 are deleted by WP3.

**`seen` (70) — transient scan scratch.** Only meaningful *during* one
snapshot-path scan: `rgiBeginForeignScan` first clears it on every buffer
entry (414), then sets it on entries that matched a GPU row (419), then
appends the still-unseen live entries as buffer-only inserts (422-424). It
prevents double-emitting a key that exists both on the GPU and in the buffer.
Write paths defensively reset it to false whenever they touch an entry
(521, 541, 566, 577, 594). It carries no meaning across statements.

The comment block at lines 73-78 is the in-source statement of the
representational limit (one paragraph, worth reading verbatim — it is the
"why" for `kc_new`/`kc_old`).

**Statics (80-81).**

```c
static HTAB *txn_buf = NULL;
static bool  xact_cb_registered = false;
```

`txn_buf` is the current transaction's buffer, or NULL if none exists yet (a
transaction that has performed no foreign-table DML allocates nothing).
`xact_cb_registered` makes `RegisterXactCallback` once-per-backend-lifetime —
PostgreSQL keeps callbacks registered across transactions, so registering per
statement would accumulate duplicates and fire the apply path N times.
Both are per-backend globals (each connection is its own process, so "static"
*is* "per session" — no thread-safety concerns exist in a PostgreSQL
backend).

#### `txn_get_buf` (lines 83-97)

```c
static HTAB *txn_get_buf(void)
```

Lazily creates the buffer on first write of the transaction.

Step by step:

1. If `txn_buf` is non-NULL, return it (86).
2. Otherwise build a `HASHCTL`: `keysize = sizeof(int64)` (the `key` field),
   `entrysize = sizeof(TxnEntry)`, `hcxt = TopTransactionContext` (88-92).
3. `hash_create("rgi txn buffer", 1024, &ctl, HASH_ELEM | HASH_BLOBS |
   HASH_CONTEXT)` (93-94). Flag meanings for a dynahash newcomer:
   `HASH_ELEM` = key/entry sizes are supplied; `HASH_BLOBS` = the key is raw
   bytes, hash it with the built-in byte hasher (vs. `HASH_STRINGS` or a
   custom function); `HASH_CONTEXT` = allocate everything in the supplied
   memory context instead of the default. 1024 is the initial bucket-count
   hint; dynahash grows unbounded beyond it.

Why `TopTransactionContext` is the load-bearing choice: PostgreSQL memory
contexts form a tree, and `TopTransactionContext` is destroyed automatically
at every transaction end — commit *or* abort, including aborts thrown from
arbitrary depths by `ereport(ERROR)`. The buffer therefore has **no leak path
and no explicit free anywhere in the file**: every exit from a transaction
reclaims it wholesale, and the callback (152-157) only needs to reset the
dangling static pointer. Any WP3/WP4 replacement structure (op log, per-key
index, savepoint stack, read set) must live in the same context for the same
reason — the WP4 plan calls this out explicitly ("do not malloc").

Pitfall for modifiers: `txn_get_buf` is only ever called from the
`ExecForeign*` write callbacks (506, 529, 586), which are always preceded in
the same transaction by `rgiBeginForeignModify` (which registers the xact
callback, 487). That ordering is what guarantees a non-NULL `txn_buf` always
has a registered callback to clear it. If a future change makes a *read* path
create buffer entries (WP3's read-set capture is exactly such a change — it
records versions on scans), the registration call must move or be added
there too, or the first transaction whose only foreign-table activity is a
read would leave `txn_buf` pointing into a destroyed context for the next
transaction to trip over.

#### `rgi_txn_apply` (lines 105-142)

```c
static void rgi_txn_apply(void)
```

The commit-time flush: classify the buffer into three operation arrays and
hand them to the worker's atomic commit protocol. Called from exactly one
place: the `XACT_EVENT_PRE_COMMIT` arm of `rgi_xact_cb` (150).

Step by step:

1. **Empty cases** (115-117): if `txn_buf` is NULL there is nothing to do; if
   it exists but has zero entries, reset the pointer and return. The
   zero-entry case is reachable: `rgiExecForeignDelete` calls `txn_get_buf()`
   (586) *before* its NULL-identity early return (590), so a DELETE whose
   junk key came up NULL creates the HTAB without ever populating it. The
   guard at 117 exists for exactly this kind of path.
2. **Allocate worst-case arrays** (119-121): five `palloc`s of `cnt` elements
   each (`ins_k/ins_v/upd_k/upd_v/del_k`). Each entry lands in at most one
   array, so `cnt` bounds all three counts. palloc'd at PRE_COMMIT time means
   they live in a transaction-lifetime context and are freed with it — no
   explicit frees, consistent with the file's memory discipline.
3. **Classification scan** (122-131) — the rules, which several other parts
   of the system (worker protocol, WP3's net-effect computation, the
   regression suite) depend on exactly:

   ```c
   if (e->deleted && e->inserted) continue;            /* (a) */
   else if (e->deleted)  del_k[nd++] = key;            /* (b) */
   else if (e->inserted && !e->replace) ins[ni++]=k,v; /* (c) */
   else                  upd[nu++] = k,v;              /* (d) */
   ```

   - **(a) `deleted && inserted` → emit nothing** (125). The row was created
     *and* destroyed inside this transaction; net effect on committed state
     is nil. Example: `INSERT (9,9); DELETE 9; COMMIT` — the GPU never hears
     about key 9.
   - **(b) `deleted` (and not inserted) → delete array** (126). A committed
     row was deleted (or renamed away: the `kc_old` tombstone takes this
     branch too, since 575 forces `inserted=false` on a freshly created
     old-key entry).
   - **(c) `inserted && !replace` → insert array** (127-128). A genuinely
     fresh key. This is the **only PK-checked class**: the worker validates
     every key in this array against the committed index (plus an intra-array
     duplicate scan) before applying anything.
   - **(d) everything else → update array** (129-130). Two sub-cases share
     the slot deliberately: a plain value update of a pre-existing row
     (`inserted=false, deleted=false`), and a delete-then-reinsert
     (`inserted=true, replace=true`) which must be applied as an upsert
     *without* PK checking (see the `replace` discussion above). The worker
     applies updates as upserts (insert-with-update-if-exists), which is what
     makes both sub-cases safe in one array.

   **Suspected latent bug, flagged here for whoever touches this next**
   (analysis from reading, not yet reproduced in a test): branch (a) ignores
   `replace`. Consider, against a key that exists committed on the GPU:
   `BEGIN; DELETE k; INSERT k; DELETE k; COMMIT;`. The first DELETE makes
   `{deleted=T, inserted=F}`; the INSERT makes
   `{deleted=F, inserted=T, replace=T}` (519-521); the second DELETE sets
   `deleted=T` while **preserving** `inserted=T` (593 clears flags only when
   the entry is newly created). The final entry
   `{deleted=T, inserted=T, replace=T}` hits branch (a) and emits *nothing* —
   so the committed row survives a transaction whose net effect should have
   been its deletion. The fix shape would be
   `if (e->deleted && e->inserted && !e->replace) continue;` with `replace`d
   tombstones falling through to (b) — but since WP3 deletes this whole
   classifier in favor of op-log net effects, the cheaper path may be a
   regression test pinning the correct semantics (heap-table differential,
   per the suite's pattern) that WP3's rewrite must pass. Either way: do not
   trust branch (a) blindly when extending this code.

4. **The fallible call** (134): `gpu_svc_txn_commit(del_k, nd, upd_k, upd_v,
   nu, ins_k, ins_v, ni, &ok, &dup)`. Per its contract
   (`pg_gpu_service.h:39-48`): stages the entire write set under one lock
   hold, validates all PK/UNIQUE conditions *before any mutation*, applies
   only if validation passes; on violation sets `*ok = 0`, `*dup_key` to the
   offending key, and leaves the GPU index unchanged. Order of application in
   the worker is deletes → upserts → inserts as batches, which is why
   delete-before-insert dependencies inside one commit (the `replace`
   collapse notwithstanding) are safe.
5. **Reset before raising** (135): `txn_buf = NULL` happens *before* the
   error check. Ordering is deliberate: if line 137 raises, PostgreSQL aborts
   the transaction, `TopTransactionContext` is destroyed (freeing the HTAB),
   and `rgi_xact_cb` fires again with `XACT_EVENT_ABORT` — by then the
   pointer is already NULL and nothing dangles or double-applies. If the
   reset came after the `ereport`, behavior would be identical only because
   the ABORT arm also NULLs it; the early reset removes the dependency.
6. **The error** (137-141):

   ```c
   ereport(ERROR,
           (errcode(ERRCODE_UNIQUE_VIOLATION),
            errmsg("duplicate key value violates unique constraint on GPU table"),
            errdetail("Key (k)=(%lld) already exists.", (long long) dup)));
   ```

   The SQLSTATE (`23505`) and the message/detail shapes mirror heap-table
   unique-violation errors (`duplicate key value violates unique constraint
   "<name>"` / `Key (k)=(...) already exists.`) so that client retry logic
   and the differential test suite see identical behavior. This wording
   parity is a stated project contract (WP3 plan, "Contracts": "Postgres
   error codes/wording match heap-table behavior exactly") — treat the
   strings as frozen API. The same error text appears at the
   statement-level duplicate checks (513-516, 558-562); all three sites must
   stay in lockstep.

Why this function raises from PRE_COMMIT and not earlier or later: earlier
(statement end) would forfeit transaction-scoped batching and would check
constraints against a state that other statements in the same transaction may
still change; later (`XACT_EVENT_COMMIT`) is past the point of no return —
an ERROR there is escalated to PANIC. PRE_COMMIT is the unique point where
the full write set is known *and* an error still aborts cleanly. This makes
PK enforcement on the GPU table behave like a deferred constraint
(`INITIALLY DEFERRED` on a heap table) for cross-statement cases, while the
in-buffer duplicate checks in the Exec callbacks (513, 558) restore
immediate, statement-time erroring for the common within-transaction case.

#### `rgi_xact_cb` (lines 144-161)

```c
static void rgi_xact_cb(XactEvent event, void *arg)
```

The transaction-lifecycle dispatcher, registered once per backend. Branches:

- `XACT_EVENT_PRE_COMMIT` (149-151) → `rgi_txn_apply()`. The comment on 150
  ("a PK violation here aborts the commit") is the crux: ERROR is still legal
  at this event.
- `XACT_EVENT_ABORT`, `XACT_EVENT_PARALLEL_ABORT`, `XACT_EVENT_COMMIT`,
  `XACT_EVENT_PARALLEL_COMMIT` (152-156) → `txn_buf = NULL`. Pure pointer
  hygiene; the memory is gone (or going) with `TopTransactionContext`. The
  COMMIT arm looks redundant given `rgi_txn_apply` already nulled it at 135 —
  it is belt-and-braces for paths where PRE_COMMIT did not run this
  extension's apply (e.g. nothing was buffered) but the pointer might be
  stale (in practice: only the empty-HTAB path makes this matter, see
  115-117).
- `default` (158-159): events not handled — notably `XACT_EVENT_PREPARE`
  (two-phase commit `PREPARE TRANSACTION` would buffer-apply nothing and
  then... in fact 2PC against this table is unhandled and would silently
  lose the write set at PREPARE; nobody has scoped 2PC, and no test covers
  it — out-of-scope like SAVEPOINT, just less loudly documented) and the
  pre-prepare/parallel-pre-commit variants.

What is **not** here, by deliberate omission: `RegisterSubXactCallback` /
subtransaction events. `ROLLBACK TO SAVEPOINT` therefore does *not* trim the
buffer — writes made after the savepoint survive the rollback and commit with
the transaction (silent wrong results, documented in the explainer Part VIII
and the WP4 plan). WP4 adds the subxact callback (start: push a high-water
mark; abort-sub: truncate the op log to the mark and rebuild the per-key
index; commit-sub: pop), which only becomes mechanical once WP3's ordered log
exists — against today's key-collapsed HTAB there is no cheap "state at
savepoint" to restore, which is precisely why a do-nothing stub was rejected
rather than registered.

#### `rgi_ensure_xact_cb` (lines 163-171)

```c
static void rgi_ensure_xact_cb(void)
```

Idempotent registration: `RegisterXactCallback(rgi_xact_cb, NULL)` guarded by
the `xact_cb_registered` static (166-169). Called from one site:
`rgiBeginForeignModify` (487) — i.e. registration happens lazily on the
backend's first DML against any RGI table, and the callback then stays
registered for the life of the backend, firing (cheaply: NULL check at 115)
on every subsequent transaction including ones that never touch the FDW.

Why lazy rather than in `_PG_init`: this module is loaded two ways — via
`shared_preload_libraries` (for the background worker) and on first use of
the FDW functions — and per-backend callback registration belongs with
per-backend state, not preload-time process setup. The invariant to preserve
when modifying: **registration must precede the first buffer entry of the
backend's lifetime**. Today `BeginForeignModify` → `ExecForeign*` ordering
guarantees it; WP3's read-set capture and WP4's savepoint stack must either
route through a path that calls `rgi_ensure_xact_cb` or call it themselves.

#### `rgiGetForeignRelSize` (lines 174-179)

```c
static void rgiGetForeignRelSize(PlannerInfo *root, RelOptInfo *baserel, Oid foreigntableid)
```

First planner callback for any query touching the table. Sets
`baserel->rows = 1000` (177) — a hardcoded row-count estimate, since the GPU
side exposes no statistics to `ANALYZE` (no `AnalyzeForeignTable` callback is
registered) — and `fdw_private = NULL` (178). The estimate only influences
costing of plans *above* this scan (join order, hash-vs-nestloop choices);
for the project's single-table workloads it is inert. If multi-table queries
ever matter, the honest improvement is asking the worker for
`hash_get_num_entries`-equivalent live-set size; until then this is a known
crudeness, not a bug.

#### `rgiGetForeignPaths` (lines 181-189)

```c
static void rgiGetForeignPaths(PlannerInfo *root, RelOptInfo *baserel, Oid foreigntableid)
```

Adds exactly one `ForeignPath` via
`create_foreignscan_path(root, baserel, NULL /*pathtarget*/, baserel->rows,
startup=1.0, total=1.0+rows, NIL /*pathkeys*/, NULL /*outer rel*/,
NULL /*extra plan*/, NIL /*fdw_private*/)` (184-188). The planner *must* be
given at least one path for the relation or planning fails outright — this
callback is mandatory in that sense. Notable simplifications:

- One path only, costed flat. There is no separate cheaper path for
  key-lookup queries, so EXPLAIN cost numbers never reflect pushdown — the
  pushdown decision is made unconditionally later in `GetForeignPlan`
  regardless of cost. (Contrast `postgres_fdw`, which builds parameterized
  paths so the planner can *choose* remote filtering. Here there is nothing
  to choose between, so the single-path shortcut is sound.)
- `pathkeys = NIL`: the FDW promises no output ordering (a hash table has
  none), so any `ORDER BY` above this scan gets an explicit Sort node.
- No parameterized paths: joins always treat this table as a full scan
  source from the planner's perspective; a nestloop with `kv_rgi` on the
  inner side will ReScan the materialized result rather than re-probing the
  GPU per outer row (see `rgiReScanForeignScan`).

#### `rgi_push_key` (lines 192-196)

```c
static void rgi_push_key(List **keys, int64 v)
```

Helper that appends one extracted key to the plan-time key list as an
`INT8OID` `Const` node:
`makeConst(INT8OID, -1, InvalidOid, sizeof(int64), Int64GetDatum(v), false /*isnull*/, true /*byval*/)`
(194-195). Why wrap plain integers in `Const` nodes at all: the list is going
to ride in the plan's `fdw_private` field (287), and everything in
`fdw_private` must be a `copyObject`-able node tree — plans get copied (e.g.
into the plan cache for prepared statements) and a raw C array would not
survive. Normalizing every key to int8 here is also what lets
`rgiBeginForeignScan` decode the list with a single `DatumGetInt64` (365)
regardless of the literal's original width.

#### `const_to_int64` (lines 198-209)

```c
static bool const_to_int64(Const *c, int64 *out)
```

Reads an integer `Const` of type int2/int4/int8 into an `int64`; returns
false for NULL constants (200) and for any other type (207). The existence of
the three-way switch (203-206) encodes a real-world planner behavior that
initially defeated pushdown entirely: **SQL integer literals are not widened
to the column's type when a cross-type operator exists.** `WHERE k = 50000`
against a `bigint` column parses the literal as `int4` and resolves the
operator to `int84eq(bigint, int)` — the Const stays `INT4OID`. A naive
extractor that only accepted `INT8OID` consts silently matches nothing and
every point query degrades to a full snapshot (correct results, catastrophic
performance, no error anywhere — the worst kind of regression). The
`DatumGetInt16/32/64` distinction also matters for correctness: a Datum
holding an int4 must be read with the int4 accessor; reading it as int64
would produce garbage on platforms where Datum carries sign-extension
differences.

Pitfall: `WHERE k = 50000::numeric` or `k = 1.0` resolves through numeric
comparison — `const_to_int64` returns false, extraction fails, and the query
falls back to the snapshot path. Correct (recheck-safe), just slow; if a
workload hits it, the fix is widening this switch, not bypassing the guard.

#### `is_equal_op` (lines 211-218)

```c
static bool is_equal_op(Oid opno)
```

Looks up the operator's catalog name via `get_opname(opno)` and accepts iff
it is the string `"="` (213-216; the returned name is pfree'd — catalog
lookups palloc copies).

This four-line function is a **correctness guard, not polish**, and the
asymmetry explained in the lifecycle subsection is why. Both extraction sites
(231 for `OpExpr`, 248 for `ScalarArrayOpExpr`) match on expression *shape* —
`<var> <op> <const>` — and shape alone cannot distinguish `k = 5` from
`k > 5` or `k <> 5`. If `k > 5` were pushed as a lookup of key 5, the FDW
would return at most the row with k=5; the executor's recheck would then
filter that row *out* (5 > 5 is false) and pass nothing — but the actual
matching rows (k=6,7,...) were never fetched. Recheck removes false
positives; it cannot resurrect false negatives. So pushing a non-equality
operator is a silent wrong-results bug, and this name check is the only thing
standing in the way.

Residual assumption worth knowing: matching by *name* assumes any operator
spelled `=` on these types is semantically equality. For the built-in integer
operators feeding a `bigint` column (`int8eq`, `int84eq`, `int82eq`) this
holds. A user-defined `=` operator with non-equality semantics could lie its
way through — a theoretical hole accepted for a prototype whose schema is
fixed. The principled alternative (checking the operator is the equality
member of the column's default btree/hash opclass, as `postgres_fdw` does) is
the upgrade path if the schema ever generalizes.

#### `rgi_extract_keys` (lines 224-269)

```c
static bool rgi_extract_keys(Expr *clause, List **keys)
```

Plan-time extraction of pushable keys from one restriction clause. Recognizes
exactly two shapes; anything else returns false and contributes no keys
(which is always safe — unextracted clauses just mean snapshot + recheck).

**Shape 1 — `OpExpr`: `k = <integer const>`** (227-241):

1. Must be binary and `is_equal_op` (231).
2. Normalize operand order (233): the planner may emit `5 = k` as
   `Const, Var`; one swap puts `Var` first.
3. The Var must be `varattno == 1` (234) — attribute numbers are 1-based, so
   this pins the predicate to the **first column, `k`**. A predicate on `v`
   (`v = 5`) correctly fails this test and stays recheck-only. Note what is
   *not* checked: `varno`/varlevelsup subtleties are implicitly handled
   because clauses come from `baserel->baserestrictinfo`, which only contains
   single-relation clauses for this relation.
4. The other side must be a `Const` convertible by `const_to_int64` (235) —
   so `k = v` (Var = Var) and `k = $1` (Var = Param) both correctly fail
   here at plan time (the latter gets a second chance at runtime; see
   `rgi_runtime_array_keys`).
5. On success, append the widened key (237) and return true.

**Shape 2 — `ScalarArrayOpExpr`: `k = ANY('{1,2,3}')`, i.e. `k IN (1,2,3)`**
(242-267). `IN`-lists over constants are folded by the parser into a single
`ScalarArrayOpExpr` with a constant array argument. Checks, in order:

1. `saop->useOr` must be true (248): `= ANY(...)` is the OR form;
   `= ALL(...)` has useOr=false and pushing its elements as a lookup
   union would be wrong (ALL of distinct values is almost always empty —
   another would-be missing/extra-rows bug avoided by one flag check).
   Operator name `=` verified as in shape 1.
2. Left operand: `Var`, attno 1; right operand: non-NULL `Const` (250-253).
3. The array is detoasted (`DatumGetArrayTypeP`, 254) and its **element
   type** checked against int2/int4/int8 (255-256) — same un-widened-literal
   reality as scalar consts: `k IN (1,2,3)` arrives as an `int4[]`.
4. `deconstruct_array` (257-258) explodes it into a Datum vector plus a NULL
   bitmap, and each non-NULL element is widened and appended (259-265).
   **NULL elements are skipped** (261), which matches SQL semantics:
   `k = ANY(ARRAY[1,NULL])` can only be satisfied by k=1 (the NULL comparison
   yields NULL, which is not true) — skipping is exact, not approximate.
5. Returns true even if every element was NULL (the key list is then empty
   for this clause — and if no other clause contributed keys,
   `rgiBeginForeignScan` sees `pushed == NIL`... with one subtlety: an
   all-NULL array yields `pushed == NIL`, which sends `BeginForeignScan`
   down the runtime-param probe and then the snapshot path; correct, just
   not the fast path for a degenerate query nobody writes).

What multiple clauses do: the caller (`rgiGetForeignPlan`, 278-282) runs this
over **every** restriction clause and unions the keys.
`WHERE k = 1 AND k = 2` therefore pushes `[1,2]`, fetches both rows, and the
recheck (which evaluates the *whole* qual list per tuple) filters both out —
correct empty result via harmless over-fetch, the superset property working
as designed. `WHERE k = 1 OR k = 2` is one `BoolExpr` clause, matches neither
shape, contributes nothing, snapshot path. (The parser usually rewrites that
specific OR into `k = ANY('{1,2}')` only via `IN`; a literal OR stays an OR.)

Not handled, deliberately: `BoolExpr` recursion, `k BETWEEN a AND b`, range
operators (no ordering on a hash index anyway), expressions over `k`
(`k + 0 = 5`), and `RowCompareExpr`. Each would be snapshot-and-recheck —
slow but correct — which is the designed failure mode for every unhandled
shape.

#### `rgiGetForeignPlan` (lines 271-288)

```c
static ForeignScan *rgiGetForeignPlan(PlannerInfo *root, RelOptInfo *baserel,
        Oid foreigntableid, ForeignPath *best_path, List *tlist,
        List *scan_clauses, Plan *outer_plan)
```

Converts the chosen path into the executable `ForeignScan` node; the FDW's
one plan-time chance to smuggle data to its executor callbacks.

Step by step:

1. Walk `baserel->baserestrictinfo` (278-282) — the planner's list of
   `RestrictInfo` wrappers around this relation's WHERE clauses — calling
   `rgi_extract_keys` on each bare clause. The return value is ignored
   (281, cast to void): extraction is best-effort, and the comment says why
   ("PG still rechecks").
2. `extract_actual_clauses(scan_clauses, false)` (284): strips the
   `RestrictInfo` wrappers to bare expressions (the executor wants
   expressions; `false` = exclude pseudoconstant clauses, which the core
   handles via one-time gating filters instead).
3. `make_foreignscan(tlist, scan_clauses, baserel->relid, NIL, pushed, NIL,
   NIL, outer_plan)` (286-287). Argument mapping, because the positional API
   is easy to misread: arg2 = `qpqual` (the clauses the executor will
   re-evaluate per returned tuple — **all** of them, none dropped); arg4 =
   `fdw_exprs` (NIL — expressions the executor would evaluate for the FDW;
   unused here, but note this is the mechanism a more thorough runtime
   pushdown would use, see the next function's pitfalls); arg5 =
   `fdw_private` = the pushed key list, which the executor hands back
   verbatim via `((ForeignScan *) node->ss.ps.plan)->fdw_private` (351).

Invariant established here, relied on everywhere downstream: **the plan's
qual list is complete**. Any future "optimization" that removes
pushed-equality clauses from `qpqual` to skip recheck converts every
extraction imprecision (and every buffer-overlay subtlety) from harmless
over-fetch into user-visible wrong results. Do not drop quals.

Plan-caching subtlety for modifiers: this function runs at *plan* time, and
plans are cached for prepared statements. A pushed Const list baked into a
generic plan is correct (consts are consts), but it is why parameterized
queries (`k = ANY($1)`) cannot be handled here at all — at plan time `$1` has
no value. Hence the split into plan-time extraction (this function) and
runtime evaluation (next function).

#### `rgi_runtime_array_keys` (lines 294-344)

```c
static bool rgi_runtime_array_keys(ForeignScanState *node, uint64_t **out_keys, uint32_t *out_n)
```

The runtime half of pushdown: when the plan carried no constant keys, probe
the plan's qual list for `k = ANY(<array parameter>)` and evaluate the array
*now*, inside `BeginForeignScan`, when parameter values exist. This is the
realistic application multi-get shape — a driver binding
`SELECT v FROM kv_rgi WHERE k = ANY($1::bigint[])` with a thousand keys — and
without this function every such query would snapshot the whole table.

Step by step (per qual clause, 299-342):

1. Shape filter mirroring the plan-time SAOP checks: `ScalarArrayOpExpr`,
   `useOr`, binary, operator named `=` (311-313), left side `Var` attno 1
   (314-315).
2. **The PARAM_EXTERN guard** (316-320) — the most important four lines in
   the function, and hard-learned:

   ```c
   /* Only evaluate EXTERNAL params ($1) standalone here — the real multi-get
    * case. InitPlan/PARAM_EXEC, SubPlans, etc. depend on executor state that
    * isn't ready in BeginForeignScan, so fall back to snapshot for those. */
   if (!(IsA(arrexpr, Param) && ((Param *) arrexpr)->paramkind == PARAM_EXTERN))
       continue;
   ```

   PostgreSQL has two parameter kinds. `PARAM_EXTERN` is a client-bound
   parameter (`$1`): its value sits in the EState's param list from the
   moment the executor starts and is safe to read any time. `PARAM_EXEC` is
   an *executor-internal* parameter: the channel through which InitPlans
   (one-shot subqueries like `k = ANY(SELECT ...)` hoisted by the planner)
   and correlated SubPlans deliver values — and those values are produced by
   *other plan nodes* that may not have run yet when this node's
   `BeginForeignScan` executes. The pre-guard version of this code called
   `ExecEvalExpr` on whatever array expression it found; evaluating a
   SubPlan-bearing expression from inside Begin dereferenced
   not-yet-initialized executor state and **segfaulted the backend** (a
   crash of the whole connection process, taking shared-memory cleanliness
   with it). The guard is therefore a crash fix, not an optimization choice.
   Queries with subquery-produced arrays fall back to the snapshot path —
   slow, correct, alive. Anyone extending runtime pushdown (e.g. to handle
   `k = $1` scalar params, which this function does NOT handle — only the
   array form) must keep evaluation strictly within what is initialized at
   Begin time, or move evaluation to first-Iterate, or use the proper
   `fdw_exprs` mechanism (expressions placed in `fdw_exprs` at plan time are
   initialized by the core into `festate` and evaluable safely — the
   "official" way postgres_fdw passes runtime values; adopting it is the
   clean fix if SubPlan arrays ever matter).

3. Evaluation (322-324): `ExecInitExpr` compiles the Param expression into
   an `ExprState`, `ExecEvalExpr` evaluates it against the node's expression
   context. For a PARAM_EXTERN this is a parameter-array fetch — cheap and
   side-effect-free.
4. NULL array → skip clause (325). Element type must be int2/int4/int8
   (327-328) — same un-widened-literal handling as everywhere
   (`'{1,2,3}'::int[]` arrives as int4[]).
5. `deconstruct_array` and per-element widening into a **`malloc`'d**
   `uint64_t` array (330-339), skipping NULL elements (335). `m` counts only
   non-NULLs, so the output may be shorter than `nelems`.
6. Empty array (`nelems <= 0`, 331) → skip; first successful clause wins —
   the function returns its keys (340-341) without examining further quals
   (so `k = ANY($1) AND k = ANY($2)` pushes only `$1`'s keys and lets
   recheck apply `$2` — superset-safe again).

Returns false if no clause qualified — caller falls through to snapshot.

Why `malloc` and not `palloc`: no strong reason survives inspection; the
caller frees it with `free()` at 396 either way. The asymmetry with the rest
of the file (palloc-everywhere) is a pitfall in itself — see the
`rgiBeginForeignScan` notes on the leak path.

#### `rgiBeginForeignScan` (lines 346-429)

```c
static void rgiBeginForeignScan(ForeignScanState *node, int eflags)
```

The workhorse of the read side: decides point-lookup vs snapshot, performs
all GPU traffic, applies the read-your-writes overlay, and materializes the
complete result into `RgiScanState`. After this function returns, the scan
never touches the worker or the buffer again.

**Setup (349-357).**

1. `EXEC_FLAG_EXPLAIN_ONLY` → return with `fdw_state` left NULL (356): plain
   `EXPLAIN` builds executor nodes without running them; doing GPU traffic
   here would make `EXPLAIN SELECT ...` mutate worker state and cost
   milliseconds. (`Iterate` guards the NULL at 437.)
2. Allocate zeroed `RgiScanState` (357).

**Key acquisition (351-371).** Two sources, tried in order:

- *Plan-time consts* (359-367): if `fsplan->fdw_private` is non-NIL, decode
  the Const list into a `malloc`'d `uint64_t` array (363-365; the int8
  normalization done by `rgi_push_key` pays off here — one `DatumGetInt64`
  decodes everything).
- *Runtime params* (368-371): otherwise probe `rgi_runtime_array_keys`.

Either path sets `have_keys`; neither firing means snapshot.

**Path A — point/multi-get with RYW overlay (373-398).** This implements
"check buffer first, then ask the worker for the rest":

1. Allocate result arrays `res_k/res_v` and a worker-query list `wq`, each
   sized `n` (`palloc`; the `n ? n : 1` guards palloc(0)) (378-380).
2. Per requested key (381-387): look it up in `txn_buf` (if any buffer
   exists, 384). Three outcomes:
   - **Buffer hit, live** (`e && !e->deleted`): emit `(key, e->value)`
     directly from the buffer (385) — an uncommitted INSERT/UPDATE is
     visible to its own transaction, and the GPU is never asked (whose
     answer for this key would be stale or absent anyway).
   - **Buffer hit, tombstoned** (`e && e->deleted`): emit nothing — an
     uncommitted DELETE hides the key even though the GPU still holds the
     committed row. Note this is the *absence* of an else-branch at 385,
     not explicit code.
   - **Buffer miss**: defer to the worker by appending to `wq` (386).
3. If any keys remain for the worker (388-395): one
   `gpu_svc_bulk(SVC_FIND_MANY, wq, NULL, &c, fk, fv, NULL, NULL)` call —
   `c` is in/out (in: query count; out: result count; misses are simply
   absent from the result, which is how `SELECT ... WHERE k = <absent>`
   returns zero rows). Results are appended after the buffer-sourced rows.
   **One worker round trip and one GPU launch regardless of key count** —
   the entire point of the multi-get path.
4. `free(kk)` (396) releases the malloc'd request array; results into
   `st` (397).

Bound worth knowing: `SVC_FIND_MANY` rides the bulk channel, whose capacity
is `GPU_SVC_BULK_CAP = 262144` rows per request (`pg_gpu_service.h:18`). A
pushed key list longer than that would overflow the channel — the FDW does
not chunk FIND_MANY requests (the commit path's staging does chunk;
the read path currently assumes < 262k keys per statement, fine for every
realistic `IN`-list but an unchecked edge).

**Path B — full-table snapshot with RYW overlay (399-426).** No usable keys:
fetch everything and merge.

1. `gpu_svc_snapshot_all(&ok, &ov, &c)` (410): the worker freezes its live
   key set under one lock hold and pages the complete committed table back
   (palloc'd into this backend; no 262k truncation — paging was added after
   review precisely to kill a silent truncation bug). `ok`/`ov` =
   keys/values, `c` = row count. ("ok" here is "output keys", an unfortunate
   name collision with the commit path's success flag.)
2. Capacity: `c + hash_get_num_entries(txn_buf)` (405, 411-413) — committed
   rows plus, at worst, every buffer entry being a buffer-only insert.
3. **Clear every entry's `seen` flag** (414) — the overlay scratch must not
   inherit state from a previous scan in the same transaction.
4. Merge loop over worker rows (415-421), per committed key:
   - in buffer and live → emit the **buffer's** value, mark `seen` (419):
     uncommitted UPDATE wins over committed value;
   - in buffer and tombstoned → emit nothing: uncommitted DELETE hides the
     row;
   - not in buffer → emit the worker row unchanged (420).
5. Append pass over the buffer (422-424): every live entry not `seen` —
   i.e. keys the committed table does not have — is a buffer-only insert and
   is appended. (`seen` is exactly the "already emitted via the merge loop"
   bit; without it, an uncommitted UPDATE of an existing key would emit
   twice.)

No ordering guarantee exists on the result (worker order, then buffer
iteration order) — correct, since the FDW declared no pathkeys.

**Shared tail (427-428):** cursor to 0, state hung on `node->fdw_state`.

Pitfalls and notes for modifiers:

- **The malloc leak path.** `kk` is `malloc`'d (363 or inside
  `rgi_runtime_array_keys` at 332) and freed at 396 — but if anything
  between allocation and 396 raises an ERROR (e.g. `gpu_svc_bulk` ereports
  on worker unavailability, or palloc fails), the longjmp unwinds past the
  `free` and the memory leaks permanently (malloc'd memory is invisible to
  memory-context cleanup; that is the whole reason PostgreSQL code prefers
  palloc). At ~8 bytes/key/error it is slow poison, not acute — but
  WP3 work in this function should take the opportunity to convert both
  allocation sites to palloc and delete the `free`.
- **Read paths do not register the xact callback and do not create the
  buffer** — they only observe `txn_buf` if write callbacks created it.
  WP3's read-set capture changes this contract (reads will record versions);
  see the `txn_get_buf` pitfall about callback registration ordering.
- **This is where WP3's version capture lands.** The plan extends
  FIND_MANY's reply to `(value, version)` per key, and this function must
  record `key → version-at-first-read` into the transaction's read set —
  with the RYW rule that keys first observed through the buffer (385) carry
  a sentinel and validate as part of the write set instead. Path B
  (snapshot) is explicitly *out* of OCC scope (scan read sets are not
  captured; scans are documented non-serializable in v1).
- The point path never touches `seen`; only Path B does. Any refactor that
  unifies the paths must preserve the reset-before-use discipline at 414.

#### `rgiIterateForeignScan` (lines 431-447)

```c
static TupleTableSlot *rgiIterateForeignScan(ForeignScanState *node)
```

Pure cursor replay; called once per tuple by the executor.

1. `ExecClearTuple(slot)` (436) resets the scan slot.
2. NULL state (EXPLAIN-only) or exhausted cursor → return the **empty** slot
   (437): an empty slot is the executor's end-of-scan signal, not NULL.
3. Otherwise fill the slot's column arrays directly:
   `tts_values[0] = k`, `tts_isnull[0] = false` (438-439); column 2 (`v`)
   only if the tuple descriptor actually has a second attribute (440-443) —
   the natts guard makes the FDW tolerate a one-column table definition
   (`CREATE FOREIGN TABLE t (k bigint)`), where writing `tts_values[1]`
   would be out-of-bounds.
4. `ExecStoreVirtualTuple(slot)` (445) marks the slot as holding a valid
   *virtual* tuple — values living in the slot's arrays with no materialized
   heap tuple behind them, the cheapest tuple representation an FDW can
   return. The executor copies/materializes only if something above needs
   it.

After this returns, the executor evaluates the plan's qual list (`qpqual`)
against the slot — the recheck — and either passes the tuple up or calls
Iterate again. NULLs never appear in returned tuples (both columns always
set non-null), consistent with the store having no NULL representation.

#### `rgiReScanForeignScan` (lines 449-453)

```c
static void rgiReScanForeignScan(ForeignScanState *node)
```

`st->cur = 0` (452) — replay the materialized result from the top, without
re-querying the worker or re-overlaying the buffer. ReScan fires when the
executor needs the same scan's output again: inner side of a nestloop, a
rewound cursor, `EXPLAIN ANALYZE` loops. Implications:

- A nestloop join probing `kv_rgi` N times costs one GPU round trip total,
  not N — materialization as accidental caching.
- The replayed data is a *statement-start* picture; if the same statement's
  own modify node changed the buffer in between (UPDATE self-join shapes),
  the rescan does not see those changes. For the supported query surface
  this is the Halloween-safe behavior, not a bug.
- NULL-state guard (`if (st)`, 452) covers EXPLAIN-only nodes.

#### `rgiEndForeignScan` (lines 454-457)

```c
static void rgiEndForeignScan(ForeignScanState *node)
```

Empty body; the comment (456) is the documentation: result arrays are
palloc'd in the executor's per-query memory context and vanish when the
executor destroys it. The malloc'd `kk` was already freed inline (396). If a
future change ever holds worker-side resources across a scan (e.g. a paged
cursor kept open), this is where they would be released — today there are
none by design (`gpu_svc_snapshot_all` completes its paging inside Begin,
under one lock hold).

<!-- CONTINUE -->


