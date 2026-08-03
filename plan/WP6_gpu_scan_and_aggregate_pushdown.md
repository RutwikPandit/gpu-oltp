# WP6 — GPU-Side Scan + Aggregate Pushdown

**Effort:** 1–2 weeks. **Priority:** capability (honesty of the scan
story + the most-requested missing SQL surface). Not on the R1–R3
critical path; schedule after WP2/WP3.
**Depends on:** WP0. Coordinates with WP2 (scan requests flow through
the coalescer's exclusive path like commits, or page under the
read-coalescing discipline — decide during implementation).

## Objective

(a) Replace the host-side live-key-set snapshot with a real **GPU
enumeration kernel** over the RGI table, and (b) push `count(*)`,
`sum(v)`, `min/max(k|v)` down to the GPU for `kv_rgi`, so simple
aggregates stop materializing every row through the executor.

Closes the standing review item "snapshot is not a real table-scan
abstraction" and removes the host shadow structure entirely.

## Current state

- `engine/rgi_oltp_engine.cu` keeps `std::unordered_set live` (every
  key, host-side); snapshot = freeze the set + batched GPU `find` of
  all keys, paged at 262,144 rows (`SVC_SNAPSHOT`/`SNAPSHOT_NEXT`).
  Values come from GPU truth; key *enumeration* is a host shadow.
- RGI has a debug single-warp `traverse_nodes` /
  `cooperative_traverse_nodes` (in `gpu_chainhashtable.hpp`) proving
  node-walking is expressible — the template to parallelize.
- Aggregates on `kv_rgi` today: paged snapshot → executor row-at-a-time
  (count(*) on 100 k rows ≈ 8.7 ms, mostly executor).
- A separate demo path (`pg_gpu_fdw`: `gpu_sum()` etc.) does on-GPU
  reduction over a synthetic column — kernels to crib from
  (`scan_sum_kernel`: grid-stride + shared-memory block reduction +
  one atomic per block), but it does NOT walk the RGI structure.

## Design

**Enumeration kernel** (new, in the engine wrapper — touches RGI
memory layout but not RGI source; isolate every layout assumption in
one header with a loud comment):

1. Grid-stride over `num_buckets`; one tile (16 lanes) per bucket:
   load head node, walk the chain (`has_next`), for each occupied slot
   emit `(key, value)`; suffix-backed entries (2-slice keys, until WP1
   reduces them) dereference the suffix node for the full key.
2. Output via a global atomic cursor into a device buffer
   (`atomicAdd(cursor, n_entries_in_node)` per node, then coalesced
   writes — one atomic per node, not per entry).
3. Exposed as `rgi_enumerate(out_keys, out_vals, cap) → n` plus a
   paged variant matching the existing snapshot ABI so the worker
   protocol does not change shape. Delete the `live` set after parity
   is proven (keep behind `#ifdef` for one release as the differential
   oracle: enumerate vs. live-set must match exactly).
4. **Consistency contract:** enumeration runs only while no mutation
   is in flight (worker schedules it exclusively, like commits). It is
   a point-in-time scan, same guarantee as today's frozen live set.

**Aggregate pushdown** (FDW planner work):

1. Implement `GetForeignUpperPaths` for `UPPERREL_GROUP_AGG` with no
   GROUP BY and a whitelist: `count(*)`, `sum/min/max(k|v)` with no
   quals or only pushable equality quals.
2. New worker op `SVC_AGGREGATE{which, qual_keys?}` → engine fuses
   enumeration + block reduction (crib the demo kernels) → returns one
   scalar; FDW emits a single-row result.
3. Everything else falls through to the existing scan path unchanged —
   the whitelist must be conservative (wrong-results bugs in planner
   hooks are the worst kind; when in doubt, do not push).

## Tasks

1. Enumeration kernel + parity test vs. live set (differential, on a
   500 k-row table with deletes — exercises chains and suffix nodes).
2. Swap snapshot path to enumeration; full suite green (notably
   `snap_page_test.sql` 300 k rows and `correctness.sql`).
3. Remove (or `#ifdef`) the live set; re-measure insert throughput —
   host-set maintenance was per-write overhead; record the gain.
4. Aggregate kernels + worker op + `GetForeignUpperPaths` whitelist.
5. Tests: `agg_test.sql` — count/sum/min/max vs. heap oracle on the
   same data (differential), with and without equality quals; EXPLAIN
   output asserted to show the pushdown path was taken.
6. Bench: count(*)/sum(v) on 1 M rows, before vs. after; update fig6
   table row or add a small `bench/agg_result.md`.

## Contracts

- RGI source untouched; node-layout assumptions isolated + commented.
- Scan consistency guarantee unchanged (point-in-time, exclusive).
- Planner hook is whitelist-only; any non-whitelisted shape must be
  byte-identical in behavior to today.

## Acceptance criteria

- [ ] Enumeration parity: 0 diffs vs. live set across insert/delete
      workloads (including >262 k rows, paged).
- [ ] Aggregates match heap oracle; EXPLAIN proves pushdown.
- [ ] count(*) at 1 M rows drops from executor-bound (~10 ms/100 k
      scale) to a few ms total; measured and recorded.
- [ ] Insert-path gain from deleting the live set recorded.

## Risks / pitfalls

- The enumeration kernel reads node metadata (count, has_next,
  suffix bits) — exactly the fields `validate_nodes_task` reads; use
  that task as the reference for correct field decoding, and pin the
  RGI commit hash you validated against in the header comment.
- Output buffer sizing: cap = current row count is unknowable on GPU;
  pass capacity from the worker (it tracks net row count once the
  live set dies — keep a simple host counter: inserts−deletes from
  applied commits, maintained in the worker; this is metadata, not a
  shadow of keys).
- `GetForeignUpperPaths` runs for every aggregate query — keep the
  whitelist check cheap and early-exit.
