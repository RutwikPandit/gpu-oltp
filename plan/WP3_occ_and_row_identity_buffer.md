# WP3 — OCC Isolation + Row-Identity Transaction Buffer

**Effort:** 2–3 weeks (the largest summer WP). **Priority:** critical
path — produces R2. **Depends on:** WP2's worker protocol (design can
proceed in parallel; land after WP2 merges).

## Objective

Two coupled upgrades that together make "transactional" fully defensible:

1. **Optimistic concurrency control (OCC):** record per-key versions in
   read sets; validate at commit inside the existing staged critical
   section; abort with `serialization_failure` on conflict. Eliminates
   lost updates (the sharpest documented isolation gap).
2. **Row-identity operation-history buffer:** replace the key-collapsed
   `TxnEntry` hash with an ordered operation log. Removes the
   key-swap/chain rejection (currently `feature_not_supported`) and the
   deferred-validation artifact, and gives OCC its natural read-set
   carrier.

These are one WP because the buffer redesign determines where read-set
entries live; doing OCC first against the old buffer would be rework.

## Current state (precise)

- `pg_rgi_fdw/pg_rgi_fdw.c`: per-transaction `HTAB` keyed by key;
  `TxnEntry {value, deleted, inserted, replace, kc_new, kc_old, seen}`.
  Read-your-writes overlays this buffer on reads. Commit classifies
  entries into del/upd/ins arrays → `gpu_svc_txn_commit` →
  validate-then-apply in the worker. Overlapping multi-row key renames
  are detected and rejected. No write-write conflict detection: two
  concurrent read-modify-writes lose an update (documented).
- Known artifact: `INSERT k; DELETE k; INSERT k` where k pre-existed
  resolves as upsert rather than failing at statement 1.
- Reviewer-set scope (adopted): OCC read sets cover **point and
  multi-get reads only**. Full-scan/predicate read sets are explicitly
  out of scope; scans are not serializable in v1 and the docs say so.

## Design

**Versions.** Worker-side `key → uint64 version` map, host memory
(`std::unordered_map` in the engine wrapper, alongside the live set —
they can merge: `live: key → version`). Version bumps on every applied
mutation of that key. No GPU-side changes in v1: versions ride the
existing host metadata. (GPU-side version words are a GB-class/TAM
refinement; do not build them now.)

**Read-set capture.** Extend the read reply path
(`SVC_FIND_MANY` and the point path) to return `(value, version)` per
key. The FDW records `(key → version-at-first-read)` into the
transaction's read set. RYW rule: keys first observed through the
transaction's own buffer (uncommitted writes) carry a sentinel and are
validated as part of the write set instead.

**Commit-time validation (extends the existing validate step, same
critical section, before any mutation):**
1. Existing PK checks (intra-set dup + index existence) — unchanged.
2. NEW: for every read-set entry, current version == recorded version.
3. NEW: for every write-set key, current version == version at the
   time the row was identified (UPDATE/DELETE row identity counts as a
   read — the junk-column fetch must capture the version; this is the
   write-write detection).
4. Any mismatch → drop staging, return `conflict` status + first
   conflicting key → FDW raises
   `ERRCODE_T_R_SERIALIZATION_FAILURE` ("could not serialize access
   due to concurrent update" — match Postgres wording so client retry
   loops behave identically to Repeatable Read on heap tables).
5. Apply (unchanged) + bump versions of all written keys.

**Buffer redesign.** Ordered log of ops
`{seq, op ∈ {INS, UPD, DEL, RENAME(old,new)}, key(s), value}` plus a
per-key index (small hash → latest log entry) so RYW overlay stays
O(1). Commit classification walks the log to compute net effects —
which handles swaps/chains naturally (net effect of a swap is two
updates) — then feeds the same staged protocol. Delete
`kc_new/kc_old/replace` flags and the rejection branch; keep the
rejection TEST and flip its expectation to success.

## Tasks

1. Buffer: implement op log + per-key index; port RYW overlay and all
   exec callbacks; keep external behavior identical except swaps now
   succeed. Suite green (with `keyswap_test.sql` expectations updated:
   G and H now succeed with correct final states).
2. Versions in the engine wrapper (merge into live-set map); bump on
   apply; expose through `rgi_find_many` out-params (extend C ABI
   additively: new function `rgi_find_many_v`, keep old symbol).
3. Worker protocol: extend FIND_MANY reply with versions; extend
   TXN_STAGE messages with the read set; extend COMMIT validation.
4. FDW: read-set capture on pushdown reads and on UPDATE/DELETE row
   identification; serialization_failure mapping.
5. Tests (the R2 evidence):
   - `occ_lost_update_test`: two psql sessions, both
     `SELECT v WHERE k=1` then `UPDATE ... v=v+1`; exactly one
     commits, the other gets serialization_failure; final value +1
     each successful retry. Scripted with `psql &` jobs, 100 rounds.
   - `occ_write_write_test`: blind concurrent UPDATEs to same key →
     second committer aborts.
   - swap/chain tests now succeed end-to-end.
   - artifact test: `INSERT k(dup); DELETE k; INSERT k` — decide and
     document the semantics the op log gives (per-statement check
     against log+committed state now catches the first INSERT —
     verify it errors at statement 1 like a heap table).
6. Measure: OCC overhead on the read path (version copy) and abort
   rate vs. zipfian skew (reuse the workload generator approach;
   θ ∈ {0, 0.9, 0.99}) — this doubles as a paper figure.
7. Docs: isolation claim upgraded everywhere from "RYW, lost updates
   possible" to "single-statement reads + OCC validation on point/
   multi-get read sets; scans not serializable (stated)". Update
   explainer Part XIV table row.

## Contracts

- C ABI additive only. RGI untouched.
- The validate step must remain the ONLY fallible stage of commit;
  OCC checks happen there, never during apply.
- Postgres error codes/wording match heap-table behavior exactly.

## Acceptance criteria

- [ ] Lost-update test: 0 anomalies in 100 adversarial rounds.
- [ ] Full suite green; swaps succeed; PK semantics unchanged.
- [ ] Abort-rate-vs-skew curve recorded with provenance.
- [ ] Read-path overhead quantified (< ~10% on multi-get expected;
      if higher, profile before accepting).

## Risks / pitfalls

- **Version map memory:** 8 B/key on host — fine at 12 M keys; note
  the ceiling alongside the slab-pool ceiling.
- **Scan semantics:** a scan that feeds an UPDATE (full-table update)
  identifies every row — its read set is the whole table. Decision:
  row-identity versions ARE captured per row (it flows through the
  junk column naturally); document that full-table RMW transactions
  therefore validate against every touched key (correct, possibly
  abort-prone under load — measure, don't hide).
- **Snapshot reads inside a txn** still see latest committed state
  (no repeatable-read snapshot) — OCC validation does NOT add
  snapshot isolation for scans; say so precisely in docs.
