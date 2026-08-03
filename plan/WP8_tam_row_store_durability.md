# WP8 — Table Access Method, Row Store, Durability Design (Fall)

**Effort:** the fall semester's architecture track. **Priority:** the
"becomes a real storage engine" arc; explicitly NOT summer work.
**Depends on:** WP3 (OCC — visibility design builds on versions),
WP6 (real scans). Advisor sign-off on scope (Master Plan §6.2).

This WP is intentionally a design-first package: each part begins with
a short design document for advisor review before code. A subagent
picking this up should produce the designs, not rush implementation.

## Part A — Row store (escape the uint32 row-id ceiling)

**Objective:** store real tuples so `kv_rgi`'s value column stops being
a 32-bit row identifier, and multi-column tables become possible.

**Design sketch to elaborate:**
- A GPU-resident **arena keyed by row-id**: fixed-width slots in v1
  (schema declared at table creation; N int64 columns), allocated by a
  bump cursor, free-listed on delete. The RGI index maps key → row-id
  (unchanged — this is exactly what it is built for); the arena holds
  the tuple.
- Reads: pushdown find returns row-ids; a second device step gathers
  tuples from the arena (fuse into the find kernel: one launch).
- Writes: tuple written to arena first, index insert second
  (validate-then-apply extends: arena writes to fresh slots are
  invisible until the index points at them — the arena append is
  naturally non-fallible, preserving the commit protocol's "no
  fallible step after validation" invariant).
- Variable-width/strings/NULLs: design only in v1 (slot directory +
  out-of-line heap — write the option analysis, do not build).

**Acceptance for Part A:** 3+-column foreign table round-trips through
SQL with the full existing test discipline (differential vs. heap);
commit protocol invariant re-derived in the design doc.

## Part B — Table Access Method (TAM) migration

**Objective:** move from FDW to the Table Access Method interface so
the GPU store participates in PostgreSQL's MVCC/visibility machinery —
the integration rung the whole FDW phase was staging toward.

**Design tasks (in order):**
1. A mapping document: every TAM callback
   (`scan_begin/getnextslot`, `tuple_insert/delete/update`,
   `tuple_satisfies_snapshot`, etc.) → what the engine/worker already
   provides vs. what is missing. The known hard part: visibility —
   TAM hands us snapshots and expects per-tuple visibility answers;
   design = WP3's version map grows xmin/xmax-like fields per row-id
   (arena metadata), checked host-side in v1.
2. Decide what the FDW keeps doing during transition (FDW and TAM can
   coexist on different tables — migration is additive, matching the
   project's pattern).
3. Skeleton TAM that delegates to the worker protocol; heap-parity
   differential tests from day one.

**Honest scoping note for the design doc:** TAM + real MVCC is a
semester, not a sprint; the deliverable that matters for the thesis is
the design + a working skeleton for point ops, not full parity.

## Part C — Durability / write-ahead log design ("stub for paper")

**Objective:** a credible design section, not an implementation.
- Commit records (the staged write set is already serialized — that IS
  the log record) stream worker → host NVMe with group commit;
  fsync off the GPU critical path (commit acknowledged after host
  fsync — measure added latency budget on paper).
- Recovery = replay committed staged batches into a fresh index
  (cold-start rebuild); checkpoint = periodic enumeration dump (WP6's
  kernel provides it).
- Deliverable: 2–3 pages + one latency-budget table; review with
  advisors before any code.

## Contracts

- Designs reviewed before implementation (each part).
- The C ABI and worker protocol may gain calls; existing ones stay.
- Differential-vs-heap remains the correctness method throughout.

## Acceptance criteria

- [ ] Three design documents (A, B, C) reviewed by advisors.
- [ ] Part A implemented to its acceptance bar.
- [ ] Part B skeleton: point insert/lookup through a TAM table.
- [ ] Part C: design doc merged into the paper draft (WP9).

## Risks / pitfalls

- Scope explosion is THE risk. The parts are ordered so each is
  independently presentable; cut from the bottom (C is a paper
  section; B can stop at skeleton; A is the must-build).
- Arena growth vs. the 8 GB laptop: design for tiered residency
  (GB-class Grace memory holds cold slots — connects to WP7 item 4)
  but implement single-tier.
