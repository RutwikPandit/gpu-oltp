# WP0 — Repository and Test Hygiene

**Effort:** 0.5–1 day. **Priority:** first; everything else builds on it.
**Result served:** none directly; removes a catastrophic-loss risk and
makes every later WP reviewable.

## Objective

Put the project under version control, make the test suite a one-command
gate, and sweep stale claims out of the documents, so that every later
work package produces reviewable diffs and cannot silently regress.

## Current state

- The project is NOT a git repository. It lives on OneDrive at
  `gpu_oltp/` with the RGI dependency as a sibling
  (`../RobustGPUIndexing/`). The persistent-kernel engine was added as
  parallel files precisely because branching was unavailable.
- Tests exist and pass (`run_tests.sh` runs 7 SQL test files against a
  restarted cluster) but are WSL-side scripts invoked manually.
- Build entry points: `build_all.sh` (engine .so + FDW), `build_bench.sh`
  (benchmark binaries), `build_persist.sh` (persistent engine + figure
  regeneration).
- Known stale-risk documents: `PROJECT_SUMMARY.md`, `PRESENTATION.md`
  (superseded by `SLIDES_1HR.md`), `bench/bench_report.md`.

## Tasks

1. `git init` in `gpu_oltp/`. Write `.gitignore` covering: `*.so`,
   `*.o`, build outputs, `bench/*.png` is a judgment call — RECOMMEND
   committing PNGs (they are results with provenance) but ignoring any
   `*.log`, `~/gpu_bench` is outside the tree already.
2. Initial commit of the current state, tagged `post-presentation`.
   Commit message should state the date and that all benchmarks were
   re-measured 2026-06-08/09.
3. Decide with the owner whether to add a private remote (GitHub/CMU
   GitLab). Do not push anywhere without explicit confirmation.
4. Make `run_tests.sh` exit nonzero on any test failure (currently it
   streams psql output; add grep-based PASS/FAIL assertions per file and
   a final summary line). This is the merge gate for every later WP.
5. Stale-document sweep:
   - `PRESENTATION.md`: add a banner line pointing to `SLIDES_1HR.md`
     as the current deck.
   - `PROJECT_SUMMARY.md`: add the persistent-kernel v1 result
     (launch 12–23 µs vs doorbell 75–90 µs on WDDM, validated correct)
     and the figure-regeneration date.
   - Verify no document still claims per-connection data or
     pre-pushdown behavior (grep for "per-backend", "per connection",
     "snapshots the whole table").
6. Optional, recommended: a `Makefile` or `make.sh` at repo root with
   targets `build`, `test`, `bench`, `figures` wrapping the existing
   scripts, so subagents have one entry point.

## Contracts

- Do not modify any source semantics in this WP. Hygiene only.
- Do not push to any remote without explicit owner approval.

## Acceptance criteria

- [ ] `git log` shows an initial tagged commit; `git status` clean.
- [ ] `bash run_tests.sh` prints a per-file PASS/FAIL table and exits
      nonzero if any test fails (verify by deliberately breaking one
      expectation and watching it fail, then restoring).
- [ ] Stale-claim grep returns no hits in the four named documents.

## Risks / pitfalls

- OneDrive + git coexist fine for a repo this size, but avoid
  committing from WSL and Windows interchangeably with different
  line-ending configs: set `core.autocrlf=false` and add a
  `.gitattributes` forcing LF for `*.sh` (CRLF in shell scripts has
  bitten this project repeatedly).
- `run_tests.sh` restarts the Postgres cluster per test file (GPU
  memory is the reset mechanism); keep that — tests are order-dependent
  without it.
