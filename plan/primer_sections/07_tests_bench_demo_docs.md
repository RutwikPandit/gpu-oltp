# Section 7 — Tests, Benchmarks, Figures, Demo, and the Document Map

This section documents the verification and evidence layer of the GPU-OLTP
project: every test file and what specific bug or guarantee it pins, the
benchmark suite and its provenance discipline, the figure-generation pipeline
(including its single biggest maintenance hazard), the live demo and its
framing rules, and a map of every document in and around the repository with
guidance on when a subagent should read each one.

All paths in this section are relative to the project root:

```
c:\Users\rutwi\OneDrive\Documents\CMU\Research_spring26\Research_spring26\gpu_oltp\
```

(WSL view: `/mnt/c/Users/rutwi/OneDrive/Documents/CMU/Research_spring26/Research_spring26/gpu_oltp/`.
All shell scripts in this project are WSL bash scripts and hardcode the
`/mnt/c/...` form — they must be run from inside WSL2, not from Windows.)

A one-line orientation for a subagent with zero context: the system under
test is a PostgreSQL 14 foreign data wrapper (`pg_rgi_fdw/`) backed by a
shared GPU-service background worker that owns a GPU-resident RGI
(RobustGPUIndexing) chained hashtable; SQL `INSERT/SELECT/UPDATE/DELETE`
against the foreign table `kv_rgi (k bigint, v bigint)` executes against
that GPU index. An earlier toy-engine FDW lives in `pg_gpu_fdw/`. Everything
below exists to prove that this system is (a) correct and (b) characterized
honestly.

---

## Contents

1. [The verification philosophy](#the-verification-philosophy)
2. [Test inventory](#test-inventory)
3. [run_tests.sh mechanics](#run_testssh-mechanics)
4. [Benchmark inventory](#benchmark-inventory)
5. [Figure pipeline](#figure-pipeline)
6. [Demo](#demo)
7. [Document map](#document-map)
8. [Re-measurement playbook](#re-measurement-playbook)
9. [Key numbers card](#key-numbers-card)

---

## The verification philosophy

The project's correctness story rests on two pillars, and a subagent must
understand both before touching any engine, worker, or FDW code, because the
test suite is the merge gate for every work package (Master Plan, standing
contract 6: "Tests are the gate. The full suite (`bash run_tests.sh`:
correctness, txn, pk, atomic, keyupd, keyswap, snap_page) must pass before
any WP is declared done. New behavior ships with new tests, preferably
adversarial ones." — `plan\00_MASTER_PLAN.md:83-86`).

### Pillar 1: Differential testing against a heap-table oracle

The primary correctness instrument is co-simulation against a golden model.
`pg_rgi_fdw\correctness.sql` creates two tables:

- `kv_ref` — a stock PostgreSQL heap table with a real btree primary key
  (`CREATE TABLE kv_ref (k bigint PRIMARY KEY, v bigint)`). This is the
  oracle: forty years of battle-tested storage code defines "correct."
- `kv_rgi` — the GPU foreign table
  (`CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi`). This is
  the device under test.

The *identical* operation stream is applied to both (bulk insert of 100k
rows, an UPDATE of every 7th key, a DELETE of every 13th key), and then the
two tables are compared with SQL set-difference in BOTH directions:

```sql
SELECT count(*) AS gpu_minus_ref FROM (SELECT * FROM kv_rgi EXCEPT SELECT * FROM kv_ref) d;
SELECT count(*) AS ref_minus_gpu FROM (SELECT * FROM kv_ref EXCEPT SELECT * FROM kv_rgi) d;
```

(`correctness.sql:30-31`). Both differences must be zero AND the row counts
must be equal. The two-direction check matters: `gpu_minus_ref = 0` alone
would pass if the GPU table silently dropped rows; `ref_minus_gpu = 0` alone
would pass if the GPU table fabricated or failed to delete rows. Zero in both
directions plus equal cardinality means the GPU table is bit-for-bit
indistinguishable from a real Postgres table over this workload. The
explainer frames this in hardware terms: "co-simulation against a golden
model" (`comp_arch_db_explainer\FULL_PROJECT_EXPLAINER.md`, section 40).

This pattern — same SQL on `kv_ref` and `kv_rgi`, then diff — is the template
for any new correctness test a subagent writes. It requires no expected-output
files, no test framework, and no knowledge of the GPU internals; the oracle
defines the answer.

### Pillar 2: An adversarially grown suite

The suite did not grow from happy paths. Several of its most important cases
encode bugs that an external expert reviewer constructed *before* any test
caught them (the "external review saga," documented as a strength in
`FULL_PROJECT_EXPLAINER.md` section 42). The review of the transaction layer
returned seven real findings; the most serious was that **commit was not
atomic**: deletes were applied before insert validation, and chunked
application meant a duplicate key arriving in a later chunk left earlier
chunks already applied, with no undo. No test exercised multi-chunk commits
or delete-then-failing-insert, so the bug was found by reading, not by
testing. The fix was the stage→validate→apply commit protocol, and the
regression tests that pin it are now permanent suite members:

- `atomic_test.sql` case A pins delete-then-failed-insert (the delete must
  not leak).
- `atomic_test.sql` case B pins the multi-chunk case: a 300,001-row commit
  spanning multiple `GPU_SVC_BULK_CAP` (262,144) chunks with the duplicate
  key staged LAST — so it lands in the final chunk, exactly where the old
  code would have already applied every earlier chunk before discovering the
  conflict.
- `keyupd_test.sql` cases D/E pin the reviewer's key-changing-UPDATE gate
  cases (rename onto a live key must fail with both rows preserved;
  delete-then-reinsert of the same key in one transaction must succeed).
- `keyswap_test.sql` cases G/H pin a *further* bug found in review round two:
  multi-row key swaps and rename chains corrupted the key-collapsed
  transaction buffer; the system now detects and cleanly rejects them rather
  than being silently wrong.
- `snap_page_test.sql` pins another review finding: `SELECT *` used to
  silently truncate at 262,144 rows (the shared-memory bulk cap); the paged
  snapshot fixed it, and this test holds the fix.

The lesson the suite encodes, and which every future subagent must continue:
when a bug is found — by review, by reading, or by accident — the failing
case becomes a permanent test before the fix is considered done.

### The reset mechanism: cluster restart IS the clean slate

Every test file in the gated suite assumes a **fresh, empty GPU index**. The
reset mechanism is a full PostgreSQL cluster restart, and it works for a
reason a subagent must internalize:

- The GPU index lives in GPU memory (HBM/GDDR), owned by the
  `pg_gpu_service` background worker that starts with the cluster (via
  `shared_preload_libraries = 'pg_rgi_fdw'`).
- GPU memory is **volatile** and the project has **no durability/WAL/recovery
  layer** (a documented limitation). When the postmaster restarts, the worker
  restarts, the CUDA context is recreated, and a brand-new empty engine is
  built.
- Therefore `pg_ctlcluster 14 main restart` is not a workaround — it is the
  semantically correct way to obtain a clean index. `run_tests.sh:13-14`
  states it directly: "Each test needs a clean GPU index. HBM is volatile, so
  a cluster restart (which restarts the GPU worker and recreates an empty
  engine) is the reset."

A critical asymmetry: the restart wipes GPU **data** but NOT the Postgres
**catalog**. The extension, the foreign server `rgi`, and the foreign table
`kv_rgi` are ordinary catalog objects on disk and survive restarts. This is
why `txn_test.sql` can use `kv_rgi` without creating it (the table definition
persists from `correctness.sql`, which runs first and creates extension +
server + table), and why most later test files begin with
`DROP FOREIGN TABLE IF EXISTS kv_rgi; CREATE FOREIGN TABLE ...` only as a
defensive re-declaration. The suite is therefore **order-dependent** in two
ways: catalog setup flows forward from `correctness.sql`, and the restart
between files resets the data. `plan\WP0_repo_and_test_hygiene.md:74-77`
warns explicitly: keep the per-file restart; "tests are order-dependent
without it."

Consequence for anyone running a single test ad hoc: restart the cluster
first, and if it is a brand-new cluster, run `correctness.sql` (or
`reload.sql`) once to create the catalog objects.

### Pillar 3 (for benchmarks): provenance discipline

The evaluation side has its own philosophy, stated as Master Plan standing
contract 5 (`plan\00_MASTER_PLAN.md:81-82`): "**Measured vs. projected
labeling** in every figure and document. A number without provenance is a
regression."

Concretely:

- Every figure constant currently in the plot scripts was **re-measured on
  2026-06-08/09** on the RTX 4060 Laptop GPU under WSL2, **on AC power**, in
  a single controlled session.
- The AC-power requirement is not pedantry: an earlier benchmark run was
  silently skewed by **battery-power DVFS** — every number, CPU and GPU,
  shifted by roughly 1.5–2× (e.g., CPU bulk INSERT 207→385 ms, GPU bulk
  UPDATE 12.5→18.4 ms, CPU point SELECT 0.75→7.5 ms — a whole-machine
  slowdown). The discovery is recorded in the session transcript
  (`cursor_cursor_assistance_request_docume.md:2256`) and elevated to a
  standing methodology note in `FULL_PROJECT_EXPLAINER.md` section 43
  ("Measurement validity"). **Any future re-measurement must be done on AC
  power, in one session, with the power state noted.**
- Exactly **one** projected input exists anywhere in the evidence chain: the
  **0.5 µs NVLink-C2C doorbell cost**, sourced from the GH200
  characterization literature (Fusco et al.) because no coherent hardware is
  on hand. Every figure and document that uses it labels it PROJECTED, and
  the sensitivity is pre-computed: if the real doorbell is 2–4× worse, the
  GPU-beats-CPU crossover moves from ~194 to ~400–800 in-flight ops — still
  inside the busy-server band, so the conclusion is robust to a pessimistic
  projection.
- Two independent timing instruments (CUDA-event timing in the sweep binary
  vs. Nsight Compute kernel durations) are kept in agreement as
  cross-validation.

### What the suite does NOT yet do (WP0)

`run_tests.sh` currently streams raw psql output; a human (or a grep) reads
the `expect_*` column aliases and the PASS/FAIL verdict lines. It does not
assert programmatically and never exits nonzero on a wrong answer.
`plan\WP0_repo_and_test_hygiene.md` task 4 schedules the upgrade: "Make
`run_tests.sh` exit nonzero on any test failure (currently it streams psql
output; add grep-based PASS/FAIL assertions per file and a final summary
line). This is the merge gate for every later WP." Until WP0 lands, a
subagent declaring the suite green must actually read the output against the
expected values tabulated in the next section — the embedded `expect_NNN`
aliases make this mechanical but not automatic.

---

## Test inventory

All test files live in `pg_rgi_fdw\` except where noted. The seven files in
the first group are the gated suite run by `run_tests.sh` (in this order);
the rest are manually invoked checks, worker/cross-session pairs, and setup
helpers. For each file: purpose, setup, then every case with its expected
result and the specific bug or guarantee it pins.

A recurring convention: files containing *expected* errors set
`ON_ERROR_STOP off` (or are run with `-v ON_ERROR_STOP=0`) so psql continues
past the deliberate failure; files where any error is a bug set
`\set ON_ERROR_STOP on`. The expected outputs are encoded in column aliases
(`AS rows_expect_5`), so the transcript is self-documenting.

### 1. correctness.sql — the differential oracle (gated, runs first)

**Purpose.** Prove the GPU table is operationally indistinguishable from a
stock Postgres heap table over a mixed 100k-row workload. This is the
project's primary correctness claim and the only test with a printed
`PASS`/`FAIL` verdict computed in SQL.

**Setup** (lines 4–11): `ON_ERROR_STOP on` (any error is a failure);
`CREATE EXTENSION IF NOT EXISTS pg_rgi_fdw`; `CREATE SERVER rgi`; fresh
`kv_ref` heap table with `PRIMARY KEY`; fresh `kv_rgi` foreign table. Because
this file runs first in the suite, it is also the file that (re)creates the
catalog objects every later file relies on.

**Cases.**

| # | Lines | Operation (applied to BOTH tables) | Expected | Guarantee pinned |
|---|-------|-----------------------------------|----------|------------------|
| op 1 | 13–15 | `INSERT ... SELECT g, g*7 FROM generate_series(1,100000)` | both succeed | bulk batched insert path (one staged commit through the worker) |
| op 2 | 17–19 | `UPDATE ... SET v = v + 1 WHERE k % 7 = 0` (14,285 rows) | both succeed | bulk UPDATE path: Postgres identifies rows via full scan, FDW buffers, commit batch-applies |
| op 3 | 21–23 | `DELETE ... WHERE k % 13 = 0` (7,692 rows) | both succeed | bulk DELETE path |
| cmp 1 | 25–27 | row counts | `ref_rows = gpu_rows` (= 92,308) | no lost or fabricated rows |
| cmp 2 | 29–31 | `EXCEPT` both directions | `gpu_minus_ref = 0` AND `ref_minus_gpu = 0` | bit-for-bit content equality (values included — catches wrong `v` after update, not just wrong key sets) |
| spot | 33–36 | point queries at k=7, k=13, k=100 on both | `gpu_k7 = ref_k7` (= 50, since 7%7=0 → 49+1); k=13 both NULL (deleted); `gpu_k100 = ref_k100` | the **pushed point-lookup path** returns the same answer as the full-scan path used by the EXCEPT — i.e., pushdown and snapshot agree |
| verdict | 38–45 | `CASE WHEN` over both diffs + counts | `PASS: GPU matches traditional Postgres` | single-line machine-greppable verdict (the future WP0 assertion hook) |

**Why the spot checks exist in addition to the EXCEPT:** the set-difference
comparison executes via the snapshot/scan path; the `WHERE k = 7` queries
exercise the qual-pushdown path (one GPU `find`). A pushdown bug that
returned stale or wrong single-row answers could in principle coexist with a
correct snapshot; the spot checks close that hole.

### 2. txn_test.sql — transaction semantics, single session (gated)

**Purpose.** Pin the four basic transactional guarantees on the GPU table:
rollback discards, commit applies, read-your-writes (RYW) inside a
transaction for all three DML types, and PK violation detected at COMMIT
aborts the transaction without applying anything.

**Setup** (line 2): none beyond a clean index — it opens with
`SELECT count(*) AS start_rows FROM kv_rgi;` which **must print 0**; a
nonzero value means the cluster restart did not happen and every subsequent
expectation in the file is invalid. The file relies on `kv_rgi` existing in
the catalog (created by `correctness.sql` on a prior run). `ON_ERROR_STOP`
is off (header comment, line 1) so the expected T4 error does not stop the
script.

**Cases.**

| Case | Lines | Script | Expected outputs | Guarantee pinned |
|------|-------|--------|------------------|------------------|
| T1 | 4–9 | `BEGIN; INSERT (1,10),(2,20); SELECT count(*); ROLLBACK; SELECT count(*)` | `in_txn_expect_2` = 2, `after_rollback_expect_0` = 0 | ROLLBACK discards the write buffer; **and** the in-transaction count proves RYW for INSERT (the rows are visible to the inserting transaction before commit, via the read-your-writes overlay, since nothing has touched the GPU yet) |
| T2 | 11–16 | `BEGIN; INSERT (1,10),(2,20),(3,30); SELECT v WHERE k=2; COMMIT; SELECT count(*)` | `k2_in_txn_expect_20` = 20, `after_commit_expect_3` = 3 | COMMIT batch-applies atomically at `PRE_COMMIT`; RYW point read inside the txn resolves from the buffer |
| T3 | 18–26 | `BEGIN; UPDATE k=1→999; SELECT; DELETE k=2; SELECT count(*); ROLLBACK;` then re-check | inside: `k1_in_txn_expect_999` = 999, `rows_in_txn_expect_2` = 2; after rollback: `k1_after_rollback_expect_10` = 10, `rows_after_rollback_expect_3` = 3 | RYW for UPDATE and DELETE (the overlay must subtract deleted keys from counts and override values), and rollback restores exactly the pre-txn state — the GPU index was never touched |
| T4 | 28–32 | `BEGIN; INSERT (1,111); COMMIT;` then `SELECT v WHERE k=1` | COMMIT raises a UNIQUE violation **at commit time**; `k1_expect_10` = 10 | PK enforcement happens at COMMIT (validate-then-apply), the failed commit applies nothing, and the pre-existing row's value survives untouched |

T4 is the within-file echo of the atomicity protocol: the duplicate is only
detectable at commit (the insert statement itself succeeds into the buffer),
and the post-error check proves the index is unchanged.

### 3. pk_test.sql — PK/UNIQUE enforcement and session health (gated)

**Purpose.** Pin the uniqueness contract for inserts in its three flavors
(conflict with an existing key, conflict within one statement/batch, no
conflict) and prove the session/connection survives a failed statement.

**Setup** (lines 2–4): recreate `kv_rgi`, load 1,000 rows
(`k, k*7` for k = 1..1000). `ON_ERROR_STOP` off.

**Cases.**

| Case | Lines | Script | Expected | Guarantee pinned |
|------|-------|--------|----------|------------------|
| dup vs existing | 5–6 | `INSERT INTO kv_rgi VALUES (500, 999);` | ERROR, unique violation | a key already resident on the GPU is detected at commit-time validation (k=500 exists with v=3500) |
| dup within one batch | 7–8 | `INSERT INTO kv_rgi VALUES (5000,1),(5000,2);` | ERROR, unique violation | **in-batch dedup**: neither row exists on the GPU yet; the conflict is between two rows of the same statement, so validation must dedup the staged set itself, not just probe the index |
| fresh keys | 9–11 | `INSERT (2000,11),(2001,22);` then count | success; `rows_after_inserts` = 1,002 | failed statements left no residue (still exactly 1,000 + 2; if the (5000,*) pair had half-applied, this would read 1,003) |
| UPDATE overwrites | 12–14 | `UPDATE ... SET v = 42 WHERE k = 500` | success; `v_at_500` = 42 | a value-only UPDATE of an existing key is NOT a uniqueness conflict (the PK check applies to *newly appearing* keys only) |
| session health | 15–17 | `INSERT (3000,33);` then count | success; `final_rows` = 1,003 | after an aborted statement the backend, its FDW state, and the worker channel are all still functional — errors do not wedge the shared ring or leak the bulk lock |

The session-health case is operationally important: the backend↔worker
channel is a shared-memory ring guarded by a lock; an error path that failed
to release state would deadlock every subsequent statement from every
backend. This is the cheapest test that would catch such a leak.

### 4. atomic_test.sql — commit atomicity, the stage→validate→apply protocol (gated)

**Purpose.** Pin the headline transaction-layer fix from external review:
commit must be all-or-nothing even when (a) deletes precede a failing insert
and (b) the write set spans multiple shared-memory chunks with the conflict
in the LAST chunk. Header comment (line 1): "Commit atomicity tests (the bug
fixed by stage->validate->apply)."

**Setup** (lines 3–6): recreate `kv_rgi`, insert 5 rows (k=1..5, v=k*7),
verify `start_expect_5` = 5. `ON_ERROR_STOP` off (expected PK errors).

**Cases.**

| Case | Lines | Script | Expected | Bug pinned |
|------|-------|--------|----------|------------|
| A: delete-then-failed-insert | 8–16 | `BEGIN; DELETE WHERE k=2; INSERT (3,999); COMMIT;` | COMMIT raises unique violation (k=3 exists); then `k2_expect_14` = 14 and `rows_expect_5` = 5 | **The delete-leak bug.** The pre-fix code applied deletes before validating inserts, so the delete of k=2 leaked even though the commit aborted — k=2 was gone after a "failed" commit. Correct behavior: validation of the WHOLE staged set happens before ANY mutation, so k=2 survives with its original value 14 (= 2×7), and the row count is unchanged. The script's own comment (lines 9–10) records the bug verbatim. |
| B: multi-chunk all-or-nothing | 18–25 | `BEGIN; INSERT ... generate_series(1000, 301000)` (300,001 fresh keys), then `INSERT (4,123)` (duplicate), `COMMIT;` | COMMIT raises unique violation; `rows_expect_5` = 5 | **The chunked-commit bug.** 300,001 rows exceed `GPU_SVC_BULK_CAP` = 262,144 (`pg_gpu_service.h:18`), so the staged set crosses the backend→worker shared-memory region in multiple chunks. The duplicate (k=4) is staged LAST — in the final chunk. Pre-fix, earlier chunks were already applied to the index when the conflict surfaced, leaving ~262k rows committed by an aborted transaction. Correct behavior: the worker stages all chunks, validates the complete set (index probe + in-batch dedup), and only then applies — so an abort leaves the index byte-identical to pre-BEGIN: exactly 5 rows. |
| C: same size, no duplicate | 27–31 | `BEGIN; INSERT ... generate_series(1000, 301000); COMMIT;` | success; `rows_expect_300006` = 300,006 (5 + 300,001) | the negative control: case B's rejection is not "large commits always fail." The identical multi-chunk volume without a conflict commits fully. Also exercises multi-chunk staging and apply end-to-end on the success path. |

Case B is the single most adversarial test in the suite and the one to study
when modifying anything in the commit path, the worker bulk channel, or the
chunking constant. Any change to `GPU_SVC_BULK_CAP` changes whether 300,001
rows still spans multiple chunks — if the cap is ever raised above 300,001,
case B silently stops testing the multi-chunk property. Master Plan standing
contract 3 makes the dependency explicit: commit atomicity and snapshot
consistency both lean on the single `bulk_lock` hold; "Any change to locking
must re-derive both properties and extend `atomic_test.sql` /
`snap_page_test.sql`" (`plan\00_MASTER_PLAN.md:73-76`).

### 5. keyupd_test.sql — key-changing UPDATE and delete-then-reinsert (gated)

**Purpose.** Pin "the reviewer's gate cases" (header comment, line 1) for
UPDATEs that change the key column — the operation that turns an UPDATE into
a delete+insert pair and therefore interacts with PK validation — plus the
upsert-shaped delete-then-reinsert pattern.

**Setup** (lines 3–6): recreate `kv_rgi`, insert exactly `(1,10),(2,20)`,
verify `start_expect_2` = 2. `ON_ERROR_STOP` off (case D's violation is
expected).

**Cases.**

| Case | Lines | Script | Expected | Guarantee pinned |
|------|-------|--------|----------|------------------|
| D | 8–11 | `UPDATE kv_rgi SET k = 2 WHERE k = 1;` | ERROR (unique violation, k=2 live); then `SELECT k,v ORDER BY k` shows `(1,10),(2,20)` unchanged, `rows_expect_2` = 2 | a key-changing UPDATE whose new key collides with a live row must FAIL — and fail *cleanly*: neither the source row (1,10) nor the target row (2,20) may be damaged. Pre-fix, key-changing UPDATEs were not globally prevalidated; this is one of the two cases the reviewer constructed. |
| E | 13–19 | `BEGIN; DELETE WHERE k=1; INSERT (1,99); COMMIT;` | success; `k1_expect_99` = 99, `rows_expect_2b` = 2 | delete-then-reinsert of the SAME key within one transaction must SUCCEED (upsert semantics). This is the dual of case D: naive validation that probes "does key 1 exist?" would wrongly reject the insert, because k=1 is live on the GPU — the validator must account for the staged delete that precedes the staged insert in the same write set. |
| F | 21–23 | `UPDATE kv_rgi SET k = 3 WHERE k = 2;` | success; final contents `(1,99),(3,20)` | a legal key change to a FREE key works — the rejection in D is conflict-driven, not a blanket ban on key-changing UPDATEs. |

Cases D and E together define the validation semantics precisely: the staged
write set is validated as an ordered/net effect, not as a bag of independent
probes. E is the case that breaks any "just check existence" shortcut.

### 6. keyswap_test.sql — multi-row key renames: rejection over corruption (gated)

**Purpose.** Pin the round-two review finding: multi-row key-changing
UPDATEs whose old and new key sets OVERLAP (swaps, chains) cannot be
represented by the key-collapsed transaction buffer, and pre-fix they
silently corrupted it. The chosen design is to **detect and reject** these
(`ERRCODE_FEATURE_NOT_SUPPORTED`) rather than be silently wrong, while
non-overlapping bulk renames remain supported. Header (lines 1–2):
"swaps/chains are rejected (not silently wrong); non-overlapping renames
succeed."

**Setup** (lines 3–6): recreate `kv_rgi`, insert `(1,10),(2,20)`,
`start_expect_2` = 2. `ON_ERROR_STOP` off.

**Cases.**

| Case | Lines | Script | Expected | Guarantee pinned |
|------|-------|--------|----------|------------------|
| G: swap | 8–12 | `BEGIN; UPDATE ... SET k = CASE WHEN k=1 THEN 2 ELSE 1 END WHERE k IN (1,2); COMMIT;` | ERROR; rows remain `(1,10),(2,20)` | a 1↔2 key swap (each new key equals the other row's old key) is rejected cleanly with ZERO corruption — both rows keep their original keys AND values. Pre-fix this scrambled the buffer. |
| H: chain | 14–18 | `BEGIN; UPDATE ... SET k = k + 1 WHERE k IN (1,2); COMMIT;` | ERROR (k=2 is simultaneously an old key and a new key); rows remain `(1,10),(2,20)` | rename chains — the generalization of swaps where the overlap is partial — are detected by the same old∩new overlap test. `k=k+1` is the canonical pattern a real application might issue. |
| I: non-overlapping rename | 20–25 | `BEGIN; UPDATE ... SET k = k + 100 WHERE k IN (1,2); COMMIT;` | success; contents `(101,10),(102,20)`, `rows_expect_2` = 2 | the negative control: when old keys {1,2} and new keys {101,102} are disjoint, the multi-row rename commits fully and values follow their rows. The rejection in G/H is precisely scoped to the unrepresentable overlap, not to multi-row renames in general. |

The architectural root cause (and why the fix is rejection, not support):
the transaction buffer is key-collapsed — it stores the net effect per key —
so it cannot represent row identity *through* overlapping renames (which row
ends up at k=2 after a swap?). Supporting swaps requires the row-identity /
operation-history buffer redesign scheduled as WP3
(`plan\WP3_occ_and_row_identity_buffer.md`). Until WP3 lands, G and H also
serve as the canary: if a refactor accidentally starts "accepting" swaps,
these tests catch the silent-corruption regression.

### 7. snap_page_test.sql — paged snapshot, no silent truncation (gated, runs last)

**Purpose.** Pin the review finding that `SELECT *` silently truncated at
the shared-memory bulk cap. The fix is a paged snapshot: the worker freezes
the live-key set and streams it back in bounded pages of at most
`GPU_SVC_BULK_CAP` rows. Header (line 1): "SELECT * must not silently
truncate at GPU_SVC_BULK_CAP."

**Setup** (lines 2–4): `ON_ERROR_STOP on` (no expected errors anywhere —
any error fails the file); recreate `kv_rgi`.

**Cases.**

| Case | Lines | Script | Expected | Guarantee pinned |
|------|-------|--------|----------|------------------|
| load | 6–7 | `INSERT ... generate_series(1,300000)` with v = k*3 | success | 300,000 > 262,144, so the subsequent scan必 must span at least two snapshot pages (and the load itself spans two staging chunks) |
| full count | 9–10 | `SELECT count(*)` | `rows_expect_300000` = 300,000 | the count over the paged scan sees every row; pre-fix this read 262,144 |
| aggregates | 12–14 | `min(k)`, `max(k)`, `sum(v)` | `min_expect_1` = 1, `max_expect_300000` = 300,000, `sum_expect_135000450000` = 135,000,450,000 | the sum is the strong check: 3 × Σ(1..300000) = 3 × 45,000,150,000. A scan that dropped, duplicated, or corrupted ANY page would almost certainly miss this exact value — count alone could be right while contents are wrong; max alone could be right while middle pages are missing. |
| verdict | 16–19 | `CASE WHEN count = 300000` | `PASS: full scan returns all rows (no truncation)` | greppable verdict line (the second of only two tests with one) |

Note the typo artifact in this primer's table ("spans至 least") does not
exist in the SQL — the test file is clean ASCII.

This file also doubles as the snapshot-consistency regression hook named in
Master Plan contract 3: the paged snapshot's correctness depends on the
worker holding the `bulk_lock` across the freeze (no commit may interleave
between pages). Any locking change must extend this test.

### 8. rt_pushdown.sql — runtime-parameter pushdown and the safe fallback (manual)

**Purpose.** Verify the qual-pushdown surface for the *realistic
application shape*: a prepared statement with a bound array parameter
(`k = ANY($1)`) must be pushed to the GPU as one batched find, and the
dangerous shape — an array produced by a subplan
(`k = ANY(ARRAY(SELECT ...))`) — must NOT crash and must fall back to the
snapshot path. Not part of the gated suite; run manually after touching the
planner/scan-begin code. `\timing on` (line 1) so the operator can also see
the fast/slow split.

**Setup** (line 2): inserts 100k rows into an existing `kv_rgi` (assumes
catalog objects exist and the index is clean — restart first).

**Cases.**

| Case | Lines | Script | Expected | Guarantee pinned |
|------|-------|--------|----------|------------------|
| bound-param multi-get | 4–7 | `PREPARE q(bigint[]) AS SELECT count(*) ... WHERE k = ANY($1);` then two `EXECUTE`s (5 keys; 15 keys) | counts 5 and 15; fast (~sub-ms to low ms with timing on) | the **pushed** runtime-parameter path: the FDW resolves `$1` at scan time and ships one batched GPU find. This is the path the demo and the workload-fit story lean on ("bound params ARE pushed"). |
| subquery array | 9–10 | `SELECT count(*) ... WHERE k = ANY(ARRAY(SELECT generate_series(1,100000,100)));` | count 1000; SLOW (snapshot fallback); **must not crash** | the safety guard. The array here is computed by a subplan; evaluating subplans inside the FDW's scan-begin caused a backend **segfault** pre-fix. The fix was to detect the subplan shape and route it to the snapshot path instead of evaluating it. This case pins "no crash + correct answer," accepting slowness by design. The ~430 ms fig6 row is this exact shape. |
| plain forms | 12–14 | point `k = 50000`; `k IN (1,2,3,5,8,13,21)` | v = 350000; count 7 | the const-equality and literal IN-list pushdowns still work after any planner change |

### 9. arraytest.sql — literal-array pushdown (manual)

**Purpose.** The smaller sibling of `rt_pushdown.sql`: verify that
*compile-time-constant* array predicates push down. Useful as a quick check
after touching the qual-extraction code, without the prepared-statement
machinery.

**Setup** (lines 1–2): `\timing on`; load 100k rows (same assumption: clean
index, existing catalog).

**Cases.**

| Case | Lines | Script | Expected | Guarantee pinned |
|------|-------|--------|----------|------------------|
| const array | 3–5 | `k = ANY('{1,100,500,50000,99999}'::bigint[])` count; then a 3-key version returning rows `ORDER BY k` | count 5; rows (1,7),(100,700),(50000,350000) | a literal typed array constant is extracted and pushed as one batched find, and the row-returning form (not just count) maps values correctly |
| IN list | 6–7 | `k IN (2,4,8,16,32,64)` | count 6 | Postgres rewrites IN-lists to `= ANY(const array)` internally; this confirms the rewrite lands in the pushed path |

### 10–11. svc_a.sql / svc_b.sql — cross-session worker visibility, raw helper level (manual pair)

**Purpose.** The minimal proof that the GPU index is owned by ONE shared
background worker rather than per-connection state: session A writes through
the raw service helpers, a **different** connection (session B) reads the
same values back. This bypasses the FDW entirely and exercises only the
backend→worker shared-memory channel.

**Protocol.** Run `svc_a.sql` in one psql connection, then `svc_b.sql` in a
*separate* psql connection (a second terminal, or sequential `psql -f`
invocations — each `psql -f` is its own connection, which suffices).

| File | Lines | Script | Expected |
|------|-------|--------|----------|
| svc_a.sql | 1–5 | `SELECT gpu_svc_insert(42, 1234); gpu_svc_insert(7, 70); gpu_svc_insert(100000, 999);` | three inserts acknowledged |
| svc_b.sql | 1–6 | `gpu_svc_lookup(42/7/100000/55555)` | `k42_expect_1234` = 1234, `k7_expect_70` = 70, `k100000_expect_999` = 999, `missing_expect_null` = NULL |

**Critical caveat** (documented in `PROJECT_SUMMARY.md` §7):
`gpu_svc_insert` / `gpu_svc_lookup` are **test-only** SQL helpers. The
single-row write helper goes through the worker's lock but **bypasses PK
validation and transaction semantics entirely**. They exist to demo/debug
the shared-worker channel, not as a supported write path — production-shaped
writes go through SQL DML on `kv_rgi`. A subagent must never use
`gpu_svc_insert` to set up state for a transactional test (it can create
duplicates the SQL path would have rejected).

### 12–13. table_a.sql / table_b.sql — cross-session visibility at the SQL level (manual pair)

**Purpose.** The same two-session experiment one layer up: session A writes
through normal SQL DML on the shared foreign table, session B (different
connection) reads it through SQL, including a pushed multi-key lookup, and
then proves the PK is shared too (B's duplicate insert of A's key fails).

| File | Lines | Script | Expected |
|------|-------|--------|----------|
| table_a.sql | 1–4 | `INSERT ... generate_series(1,1000)` (v = k*7); `INSERT (123456, 42)` | both commit |
| table_b.sql | 1–5 | count; point reads; 5-key ANY | `rows_expect_1001` = 1001, `k500_expect_3500` = 3500, `k123456_expect_42` = 42, `anyof5_expect_5` = 5 |
| table_b.sql | 6–7 | `INSERT INTO kv_rgi VALUES (500, 999);` | ERROR: unique violation — **the shared PK**: session B's commit-time validation sees session A's committed key |

The last case is the one that distinguishes "shared data" from "shared
constraints": a per-connection index would happily accept B's (500, 999).

### 14. rgi_smoke.sql — five-statement smoke for the RGI FDW (manual)

**Purpose.** The fastest end-to-end sanity check after a rebuild/reinstall:
create the catalog objects, push 200k rows through the batched insert path,
read via snapshot and pushed lookups, update, delete. `ON_ERROR_STOP on` and
`\timing on` (lines 1–2) — any error fails it, and the operator gets a feel
for the timings.

| Step | Lines | Script | Expected |
|------|-------|--------|----------|
| setup | 3–7 | extension + server + fresh `kv_rgi` | objects exist |
| bulk insert | 9–10 | 200,000 rows, v = k*7 | "one warp-cooperative flush on GPU" (header comment) |
| reads | 12–14 | `count(*)`; `WHERE k IN (1, 100000, 200000) ORDER BY k` | 200,000; three rows (7; 700000; 1400000) |
| update | 16–18 | `SET v = 999 WHERE k = 100000`, re-read | 999 |
| delete | 19–20 | `DELETE WHERE k = 1`, count | 199,999 |

This is the file to run first when "is the install sane at all?" is the
question — before running the gated suite.

### 15. reload.sql — extension reset helper (setup, not a test)

Four statements (lines 1–4): `DROP EXTENSION IF EXISTS pg_rgi_fdw CASCADE;
CREATE EXTENSION pg_rgi_fdw; CREATE SERVER IF NOT EXISTS rgi ...;
CREATE FOREIGN TABLE IF NOT EXISTS kv_rgi ...`. Used after `make install`
of a new `.so` to force Postgres to re-resolve the extension and recreate
the catalog objects from scratch (the `CASCADE` drops dependent foreign
tables/servers, then they are rebuilt). Note this resets the **catalog**
side; the **data** side still requires a cluster restart, because the worker
and its CUDA context only reload with the postmaster. A sibling
`pg_gpu_fdw\reload.sql` does the same for the toy FDW (server name
`gpu_oltp`).

### 16. enable_worker.sql — one-time worker enablement (setup, not a test)

A single statement (line 1):
`ALTER SYSTEM SET shared_preload_libraries = 'pg_rgi_fdw';`
Writes the setting to `postgresql.auto.conf`; a **cluster restart** is
required to take effect (background workers can only be registered at
postmaster start). Without this, nothing works: there is no GPU-service
worker, the shared-memory channel is absent, and every `kv_rgi` operation
errors with "worker not running" (troubleshooting steps in
`comp_arch_db_explainer\POSTGRES_USER_GUIDE.md` §15). This is run once per
cluster, ever — it is in the repository so a fresh environment can be stood
up from files alone.

<!-- SECTION07_CONTINUES -->
