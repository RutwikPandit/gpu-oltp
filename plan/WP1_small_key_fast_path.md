# WP1 — Small-Key Fast Path (Suffix-Node Elimination)

**Effort:** 1–2 days + benchmark re-run. **Priority:** first technical
task (quick win). **Result served:** strengthens all R-results by lifting
the engine ceilings; deliverable doubles as a finding for the RGI authors.

## Objective

Eliminate the suffix-node indirection for keys that fit in 32 bits, by
storing them as single-slice keys. Expected effect: remove one dependent
128 B load per find and one slab allocation per insert for the common
case, and cut per-entry space substantially.

## Background (the finding this fixes)

Discovered by source reading, confirmed in `gpu_chainhashtable.hpp`:
any key with `key_length > 1` takes RGI's suffix path — the bucket node
stores only a 32-bit hash tag, and the actual key slices plus the value
live in a separately slab-allocated 128 B suffix node. Our wrapper maps
every SQL `bigint` to two uint32 slices (`max_key_length=2`,
`key_lengths=nullptr`), so **every key today pays the suffix hop**:
- find: bucket node load → suffix node load (`streq`) → value
- insert: one slab allocation per key
- space: ~134 B per 16 B key/value pair

All published numbers include this cost; removing it is found headroom.

## Design

Canonical representation rule, applied identically on every path:

```text
if (key >> 32) == 0:  store/lookup as ONE slice  (key_length = 1)
else:                 store/lookup as TWO slices (key_length = 2)
```

RGI natively supports per-request variable lengths via the
`d_key_lengths` array (currently passed as `nullptr` = fixed length).
The change is wrapper-only — RGI source untouched:

1. In `engine/rgi_oltp_engine.cu`, every chunk helper
   (`insert_chunk`, `find_chunk`, `erase_chunk`) builds and passes a
   device `key_lengths` array computed by the rule above. Keys are
   still staged as 2 uint32 slots per entry (layout unchanged —
   `max_key_length` stays 2); only the per-key length varies.
2. Same rule in `rgi_persist_engine.cu` (compute length per request
   in the kernel from the loaded slices: `len = (hi == 0) ? 1 : 2`).
3. The staging/commit/validate paths and the live-set are
   key-value-level and unaffected.

**Correctness argument to verify, not assume:** a key stored with
length 1 and probed with length 1 hashes identically (same
`compute_hash` over one slice). The rule is a pure function of the key
value, so store and probe lengths always agree. The dangerous case —
same value stored once as 1-slice and once as 2-slice — is impossible
under the rule. Write this in the commit message.

## Tasks

1. Implement the rule in the three chunk helpers + the persistent
   kernel (≈40 lines total).
2. Extend the engine microbench (`BUILD_BENCH` in `rgi_oltp_engine.cu`)
   to print bytes/entry (RGI's `validate()` task prints space stats —
   call it before/after).
3. Re-run: `rgi_sweep` (floor/ceiling), `rgi_bench`, the ncu pair
   (`rgi_prof`, fig8 numbers), and the full SQL suite via
   `run_tests.sh`. Mixed-range key test: insert keys straddling 2^32
   (e.g., `4294967290..4294967310`) and verify finds/deletes across
   the boundary — add this as a SQL test (`bigkey_test.sql`).
4. Record before/after in `bench/smallkey_result.md`: find Mop/s,
   insert Mop/s, bytes/entry, DRAM/L1 SOL deltas. Label `[measured]`.
5. Update: explainer §13/§37 (the finding now has a fix + numbers),
   `PROJECT_SUMMARY.md`, memory file.

## Contracts

- RGI source unmodified. If results justify an upstream RGI inline-
  two-slice path, write it up as a one-page proposal for the RGI
  author instead (separate deliverable, do not implement).
- C ABI unchanged (`rgi_*` signatures identical).
- Full test suite green before/after.

## Acceptance criteria

- [ ] Find and insert ceilings improve (any regression > 2% on either
      is a stop-and-investigate).
- [ ] Bytes/entry drops materially for ≤32-bit keys (expect roughly
      2× better; record exact).
- [ ] `bigkey_test.sql` passes; full suite green; differential
      `correctness.sql` still 0 diffs.

## Risks / pitfalls

- The hash-tag template parameter: with `key_length==1`, RGI takes the
  `more_key=false` branch (no tag, direct slice match). Confirm the
  branch by reading `cooperative_find` — do not infer from behavior.
- Erase path must use the same rule (a 1-slice-stored key erased with
  length 2 would miss silently). The mixed-range SQL test covers this;
  make sure it includes DELETE.
- ncu comparison must hold batch size constant (use B=500k as before).
