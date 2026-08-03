# WP2 — Coalescer: Concurrent Reads and the Multi-Client Benchmark

**Effort:** 1–2 weeks. **Priority:** critical path — produces R1, the
single most important missing measurement.
**Depends on:** WP0. WP1 recommended first (ceilings final before curves).

## Objective

Replace "one lock, one outstanding bulk operation" with a worker that
**coalesces concurrent backends' read requests into one fused GPU batch
per dispatch window**, while commits remain exclusive. Then produce the
project's first honest multi-client scaling measurement: SQL throughput
vs. number of concurrent connections.

This converts the serialization point into the batching mechanism the
thesis requires ("implicit batching"), and it is the CPU-side prototype
of the C2C runtime's job.

## Current state

- `pg_rgi_fdw/pg_gpu_service.c`: all bulk ops — reads included —
  serialize behind one LWLock (`bulk_lock`). One 4 MB bulk window.
  Commit holds the lock across stage→validate→apply; paged snapshots
  hold it across pages. Single-row ring exists for demo functions.
- Consequence (documented): multi-client scaling is not measurable;
  a concurrent benchmark today measures the lock.
- Engine coupling (documented): finds launch on RGI's
  `concurrent=false` path, valid only under serialization.

## Design (staged, correctness-first)

**Stage A — concurrent read coalescing; writes stay exclusive.**

1. Shared memory: replace the single bulk window with
   `NCLIENT_SLOTS` (start: 16) fixed request descriptors, each with
   its own in/out arrays (size them: 16 × 2 × 64 k × 8 B ≈ 16 MB, or
   keep 4 MB windows and cap per-request rows at 32 k — decide by
   measuring typical pushdown batch sizes; multi-gets are ≤ a few
   thousand keys, so small windows are fine; snapshots keep paging).
2. Backend protocol unchanged in shape: claim a slot, fill, set
   state=POSTED, latch the worker, sleep. No global lock for reads.
3. Worker loop: gather ALL posted **read** descriptors
   (FIND_MANY + snapshot pages), concatenate keys into one staging
   buffer with per-request offsets, issue **one** `rgi_find_many`
   (one kernel launch), scatter results back per descriptor, latch
   each waiter. The dispatch window is "whatever is posted when the
   worker wakes" — no artificial delay in v1 (latency first); add an
   optional `coalesce_wait_us` GUC later for throughput experiments.
4. Writes/commits: a commit request takes a writer flag — the worker
   drains in-flight reads, services the entire staged commit
   exclusively (existing protocol untouched), then resumes reads.
   Readers-writer discipline implemented IN THE WORKER LOOP (it is
   single-threaded; this is scheduling, not locking).
5. Flip the engine to `find<concurrent=true>` — reads may now overlap
   nothing yet (worker still single-threaded, one GPU op at a time),
   but the coalesced batch may interleave with DEBRA state from prior
   erases; `true` is the safe setting and its cost is small. Document
   the flip per the standing contract.

**Stage B (optional, only if Stage A leaves GPU idle gaps):** double-
buffer — stage batch N+1's H2D while batch N's kernel runs, using the
two non-blocking streams already in the engine.

**Explicit non-goals:** multiple concurrent kernels; multi-worker;
relaxing commit exclusivity. Those are GB-class-runtime work.

## The benchmark (R1 deliverable)

1. Workload: pgbench custom scripts against `kv_rgi` —
   (a) 100% point SELECT (pushdown path), (b) 90/10 SELECT/UPDATE,
   (c) multi-get `k = ANY($1)` with 64 keys per query.
2. Sweep clients C ∈ {1, 2, 4, 8, 16, 32, 64}; report TPS and p50/p99
   latency. Same sweep against a heap table + btree as the CPU
   reference (same machine, same pgbench).
3. A/B: serialized path (pre-WP2, keep behind a build flag or git tag)
   vs. coalescer, same sweep — the delta IS the result.
4. Record average coalesced batch size per dispatch at each C (worker
   counts; expose via a stats function). This connects the curve to
   the crossover model: measured B-per-dispatch vs. C.
5. Output: `bench/multiclient_result.md` + a new figure
   (TPS vs. C, three curves: heap, serialized GPU, coalesced GPU),
   provenance-labeled.

## Contracts

- Commit atomicity and snapshot consistency must be re-derived under
  the new scheduling and stated in comments where the old `bulk_lock`
  reasoning lived. Extend `atomic_test.sql` with a concurrent variant:
  one session runs the 300 k-row failing commit in a loop while
  another hammers point reads; assert zero anomalies.
- `find<concurrent=true>` flip documented.
- Full suite green; new concurrency tests added.

## Acceptance criteria

- [ ] TPS scales with C on read workloads until GPU or executor
      saturation (any plateau explained with data, not guessed).
- [ ] Concurrent atomic_test variant passes 100 iterations.
- [ ] Measured mean batch-per-dispatch reported per C.
- [ ] A/B delta vs. serialized path published.

## Risks / pitfalls

- Latch storms at high C: batch the worker's reply latches (set all
  after scatter, not per-descriptor inside the loop).
- pgbench against an FDW: ensure prepared statements are used
  (`-M prepared`) so the runtime-param pushdown path (`PARAM_EXTERN`)
  is exercised — plain simple-protocol queries re-plan every time and
  measure the planner instead.
- Postgres connection scaling itself costs (process per connection);
  the heap-table reference curve absorbs that — always present the
  GPU curve next to it, never alone.
- Do NOT let snapshots starve under continuous read load: the worker
  services posted requests in claim order (FIFO over slot indices).
