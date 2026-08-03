# WP9 — Paper / Thesis Chapter + Artifact Freeze

**Effort:** continuous thread, ~0.5 day/week + writing sprints after
each R-result lands. **Priority:** the output everything feeds.
**Depends on:** consumes results from every other WP as they land.

## Objective

Turn the work into (a) a thesis chapter (the floor, guaranteed) and
(b) a submission-shaped systems paper, with figures and claims frozen
under the measured-vs-projected discipline.

## Venue analysis (to confirm with advisors — Master Plan §6.3)

| Venue shape | Fit | What it would demand |
|---|---|---|
| **DaMoN** (workshop @ SIGMOD, ~Feb/Mar deadline) | strongest fit: hardware-conscious DB systems, measurement-driven, 6 pages | R1 + R3(bare-metal) suffice; GB-class numbers make it strong |
| **CIDR** (Jan conference, ~Aug deadline) | fits the "system + position" framing (dispatch economics of coherent GPUs) | the crossover story + working system; less benchmark pressure |
| **VLDB/SIGMOD full** | only after WP7 + WP8 mature | on-target C2C results + OCC under contention + row store |

Recommendation to carry into the advisor meeting: thesis chapter
regardless; target DaMoN-shaped first (deadline pressure is useful and
the page budget matches what is measured), with the full-paper decision
deferred until WP7's outcome is known — the kill-criterion result
changes which venue the work belongs in.

## The claims table (maintain from day one)

A single file, `plan/claims.md`, listing every claim the paper will
make, each tagged `[measured-laptop]`, `[measured-baremetal]`,
`[measured-c2c]`, `[projected]`, or `[design]`, with the figure/test
that backs it. A claim without a backing artifact does not enter the
draft. Seed it from the deck's Part 3 and the explainer's §39.

## Paper skeleton (seed from existing material — most prose exists)

1. Introduction — the dispatch-economics framing
   (explainer Parts I–II; deck Act 1).
2. Background — PG-Strom write path; RGI substrate (credit cleanly).
3. System — stack, worker, commit protocol, pushdown
   (explainer Part III–IV; the design-decision ledger is the
   differentiator — papers rarely show their alternatives).
4. The dispatch model + crossover (explainer Part II §7–8).
5. Evaluation — floor/ceiling sweep, CPU baseline, multi-client
   (WP2), OCC under skew (WP3), persistent-vs-launch (WP5/WP7),
   on-target crossover (WP7).
6. Limitations (explainer Part VIII — keep the candor; it is the
   house style and reviewers reward it).
7. Related work (explainer Part VII table, prose-ified).

## Artifact freeze discipline

- After each WP lands: regenerate affected figures via the scripts,
  commit PNGs + the numbers file together, update `claims.md`.
- Figures carry machine + date in the caption (already the
  convention; enforce in review).
- A `repro/` README: one command per figure (the scripts exist:
  `build_bench.sh`, `build_persist.sh`, `run_bench_sql.sh`,
  `bench/make_plots*.py`) — this is most of an artifact-evaluation
  submission already.

## Tasks

1. Create `plan/claims.md` seeded from current measured results
   (one sitting).
2. Draft §1–2 + §4 now (no dependency on remaining WPs; the prose
   exists in the explainer — compress, do not rewrite).
3. After WP2: draft §5.multi-client; after WP3: §5.occ; after
   WP5/WP7: §5.hardware + rewrite the abstract around the
   decision-rule outcome.
4. Thesis chapter assembly: the explainer is already
   chapter-shaped — restructure Parts I–IX into chapter sections
   once the summer results land.
5. Venue decision memo (1 page) for the advisor meeting where WP7's
   outcome is known.

## Acceptance criteria

- [ ] `claims.md` exists and gates the draft (no orphan claims).
- [ ] §1–2, §4 drafted by mid-summer independent of pending results.
- [ ] Every figure in the draft regenerable by one named command.
- [ ] Venue memo delivered when WP7 lands.

## Risks / pitfalls

- Writing always loses to building for this profile of work; the
  0.5 day/week floor is the mitigation — calendar it.
- Do not let the paper absorb unmeasured GB-class enthusiasm; the
  claims table is the firewall.
- Co-authorship/credit: RGI is the substrate and its author is an
  obvious co-author conversation — raise early with advisors, not at
  submission time.
