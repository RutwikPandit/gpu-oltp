# Master Plan — GPU-OLTP Next Phase

**Status:** post-presentation (June 2026). Prototype validated; feasibility,
harness, and crossover model delivered. This plan converts the prototype
into research results.

**How to use this folder:** each `WPn_*.md` is a self-contained work
package: objective, current state with file pointers, task breakdown,
contracts that must not be broken, acceptance criteria, risks, effort.
Each is written to be handed to a subagent (or a person) with no other
context. This file owns sequencing, priorities, and risks.

---

## 1. The three results that matter

Everything below is justified by one of these. If a task serves none of
them, it is polish and should yield.

| # | Result | Why it is the bar | Work packages |
|---|--------|-------------------|---------------|
| R1 | **Measured multi-client SQL scaling** — throughput vs. number of concurrent connections, through real SQL | Today all bulk ops serialize behind one lock; the "implicit batching" story is designed, not demonstrated. A reviewer can dismiss every SQL number until this exists. | WP2 |
| R2 | **Defensible isolation** — OCC with write-write conflict detection | "Transactional" currently requires a scoped caveat (lost updates possible). OCC removes the caveat and is itself a publishable GPU-validation design. | WP3, WP4 |
| R3 | **Hardware-validated dispatch model** — doorbell cost measured on bare-metal PCIe now, C2C when available; crossover recomputed with on-target numbers | The thesis is gated on one projected number. Measurement converts the projection into the headline figure — or kills it honestly. | WP5, WP7 |

## 2. Work-package index

| WP | Title | Effort | Type | Depends on |
|----|-------|--------|------|------------|
| WP0 | Repository and test hygiene | 0.5–1 day | infra | — |
| WP1 | Small-key fast path (suffix-node elimination) | 1–2 days | quick win | WP0 |
| WP2 | Coalescer: concurrent reads + multi-client benchmark | 1–2 weeks | **R1, critical path** | WP0 |
| WP3 | OCC isolation + row-identity transaction buffer | 2–3 weeks | **R2, critical path** | WP2 (protocol), can design in parallel |
| WP4 | Subtransaction / SAVEPOINT support | 2–4 days | R2 completeness | WP3 (buffer redesign) |
| WP5 | Persistent-kernel v2 + bare-metal PCIe campaign | 1 week + HW access | **R3** | hardware (external) |
| WP6 | GPU-side scan + aggregate pushdown | 1–2 weeks | capability | WP2 helpful, not required |
| WP7 | GB-class (NVLink-C2C) measurement campaign | 1–2 weeks on HW | **R3, thesis-deciding** | WP5 artifacts; GB-class access (external) |
| WP8 | TAM migration, row store, durability design | fall semester | architecture | WP3, WP6 |
| WP9 | Paper / thesis chapter + artifact freeze | continuous | output | feeds on all |

## 3. Sequencing

```text
Week 0      WP0 (hygiene) ──► WP1 (quick win, first solo CUDA change)
Weeks 1-3   WP2 coalescer ──────────────► R1: multi-client curve
Weeks 3-6   WP3 OCC (+ buffer redesign) ─► R2: lost-update test passes
Week 6      WP4 subtransactions
Weeks 7-8   WP6 GPU scan + aggregate pushdown
parallel    WP5 the moment a bare-metal Linux GPU is available
parallel    WP9 paper thread: outline now, figures freeze per-WP
gated       WP7 the moment GB-class hardware is available
fall        WP8 TAM / row store / durability
```

Dependency notes:
- WP2 before any multi-client benchmark claims (R1 blocks on it).
- WP3's buffer redesign (row-identity op history) must precede WP4;
  doing WP4 against the key-collapsed buffer would be rework.
- WP5 and WP7 are externally gated on hardware; their *preparation*
  (binaries, scripts, recording templates) is not — prepare early so
  hardware days are spent measuring, not building.
- WP1 is deliberately first real task: small, self-contained,
  engine-level, and it is the project owner's home-turf CUDA work.

## 4. Standing contracts (every WP must honor these)

1. **The C ABI is the stability boundary.** `rgi_oltp_engine.h` may be
   extended, never broken. The FDW/worker layers must keep building
   against it.
2. **RGI source is not modified** without explicit coordination with
   its author. Wrapper-level solutions first; RGI changes proposed as
   patches, separately.
3. **The serialized-channel invariants** (until WP2 replaces them):
   commit atomicity and snapshot consistency both lean on the single
   `bulk_lock` hold. Any change to locking must re-derive both
   properties and extend `atomic_test.sql` / `snap_page_test.sql`.
4. **The `find<concurrent=false>` coupling**: the engine launches
   finds on RGI's non-concurrent path, valid only under serialization.
   WP2 and WP5 must flip this to `true` when reads can overlap
   mutations, and say so in the commit message.
5. **Measured vs. projected labeling** in every figure and document.
   A number without provenance is a regression.
6. **Tests are the gate.** The full suite
   (`bash run_tests.sh`: correctness, txn, pk, atomic, keyupd,
   keyswap, snap_page) must pass before any WP is declared done.
   New behavior ships with new tests, preferably adversarial ones.

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GB-class doorbell ≥ ~5 µs | low-med | kills headline thesis | This is a *result* either way; WP7 defines the kill criterion up front. Fallback narrative: bare-metal PCIe + coalescer scaling + OCC are publishable without C2C. |
| Grace CPU (72 cores) closes the index-throughput gap | medium | weakens crossover | WP7 measures *both* sides on-target (port `cpu_sweep` to Grace). The RGI paper's 8.8× on server hardware suggests the gap survives; verify, don't assume. |
| Coalescer introduces concurrency bugs | medium | delays R1 | Staged rollout (reads first, commits stay exclusive); differential tests under concurrency; keep the serialized path behind a flag for A/B. |
| OCC read-set capture costs dominate | low-med | weakens R2 perf story | Scope to point/multi-get read-sets (decided); measure overhead explicitly; abort-rate-vs-skew is a *finding*, not a failure. |
| Single-person bandwidth | high | schedule slip | WPs are sized and independent; cut order if needed: WP6 → WP4 → WP1 (capability before completeness before polish). R1/R2/R3 are not cuttable. |
| OneDrive file corruption / no version control | medium | catastrophic, cheap to prevent | WP0 first. |

## 6. Decisions needed from advisors (carry to next meeting)

1. **Hardware:** access to a bare-metal Linux GPU box (lab machine)
   now; expected GB-class/GH200 access path and timing.
2. **TAM timing:** confirm fall scope (WP8) or pull forward.
3. **Venue:** thesis chapter is the floor; target workshop/conference
   for the systems paper (WP9 contains the venue analysis; DaMoN-shaped
   vs. CIDR-shaped framing changes which results to prioritize).
4. **RGI coordination:** small-key fast path (WP1) and the suffix-node
   finding — wrapper-level fix vs. upstream RGI patch.

## 7. Definition of done for the phase

By end of summer, all of:
- [ ] R1: throughput-vs-connections curve through SQL, coalescer on,
      published in `bench/` with the standard provenance labeling.
- [ ] R2: two concurrent sessions racing read-modify-write produce one
      `serialization_failure`, zero lost updates; full suite green.
- [ ] R3 (partial): doorbell + floor measured on bare-metal Linux PCIe;
      crossover re-derived from those numbers.
- [ ] All documents (PROJECT_SUMMARY, explainer, deck) updated to the
      post-WP state; no stale claims.
- [ ] Paper outline with frozen figure list (WP9).
