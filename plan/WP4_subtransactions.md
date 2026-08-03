# WP4 — Subtransaction / SAVEPOINT Support

**Effort:** 2–4 days. **Priority:** R2 completeness; closes the last
open finding from the external transaction-layer review.
**Depends on:** WP3's op-log buffer (hard dependency — implementing this
against the old key-collapsed buffer would be immediate rework).

## Objective

Make the GPU table behave like a heap table under `SAVEPOINT`,
`ROLLBACK TO SAVEPOINT`, `RELEASE`, and PL/pgSQL `BEGIN ... EXCEPTION`
blocks (which create subtransactions implicitly — the reason this
matters more than it looks: client frameworks and stored procedures use
them invisibly).

## Current state

- Only `RegisterXactCallback` is registered
  (`pg_rgi_fdw.c`, `rgi_ensure_xact_cb`). There is no
  `RegisterSubXactCallback`; a `ROLLBACK TO SAVEPOINT` silently keeps
  buffered writes that should have been discarded.
- Deliberately, no do-nothing stub was registered (a stub that lies is
  worse than an absence). The gap is documented in the explainer and
  was acknowledged in review.

## Design

With WP3's ordered op log this becomes almost mechanical — the log is
its own undo structure:

1. Register `RegisterSubXactCallback` alongside the xact callback.
2. On `SUBXACT_EVENT_START_SUB`: push the current log high-water mark
   (sequence number) onto a savepoint stack, keyed by subtransaction id.
3. On `SUBXACT_EVENT_ABORT_SUB`: truncate the op log back to that mark
   and rebuild/patch the per-key index for the truncated suffix
   (cheapest correct v1: rebuild the per-key index from the surviving
   log — transactions are small; optimize only if profiling says so).
   Discard read-set entries recorded after the mark as well (OCC
   correctness: reads made inside an aborted subxact must not
   constrain commit validation).
4. On `SUBXACT_EVENT_COMMIT_SUB`: pop the mark (entries simply remain).
5. Nested savepoints fall out of the stack discipline.

## Tasks

1. Implement callback + savepoint stack + truncation (FDW only; no
   worker/engine changes).
2. Tests (`subxact_test.sql`):
   - `SAVEPOINT s; INSERT (9,9); ROLLBACK TO s; COMMIT;` → key 9 absent.
   - Nested: two savepoints, roll back inner, commit outer → only
     outer's writes land.
   - PL/pgSQL: a DO block whose EXCEPTION clause catches a forced
     `unique_violation` → final state identical to the same script
     against a heap table (write the heap version in the same file and
     diff results — the differential pattern used by `correctness.sql`).
   - Interaction with WP3: read-set entries from an aborted subxact do
     not cause spurious serialization failures (read a key inside the
     savepoint, roll back, have another session update that key,
     commit — must succeed).
3. Docs: remove "no subtransactions" from every limitations list
   (explainer Part VIII, PROJECT_SUMMARY, deck limitations slide).

## Contracts

- FDW-local change only. No protocol, ABI, or worker changes.
- Heap-table-equivalent semantics is the acceptance bar, verified
  differentially, not asserted.

## Acceptance criteria

- [ ] All `subxact_test.sql` cases match heap-table results exactly.
- [ ] Full suite green.
- [ ] Limitations lists updated.

## Risks / pitfalls

- Subtransaction ids vs. nesting levels: use the callback's
  `mySubid/parentSubid` arguments, not a manual depth counter —
  PL/pgSQL can release out of order.
- Memory context: the savepoint stack must live in
  `TopTransactionContext` like the buffer (freed automatically on
  abort — do not malloc).
- Do not forget the read-set truncation (the subtle half; the write
  half is obvious).
