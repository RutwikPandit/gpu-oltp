# WP7 — GB-Class (NVLink-C2C) Measurement Campaign

**Effort:** 1–2 weeks on hardware; preparation days beforehand.
**Priority:** R3 (second half) — the thesis-deciding measurements.
**Depends on:** GB-class/GH200 access (external); WP5's campaign
scripts and v2 persistent binding (reuse directly).

## Objective

Replace the single projected number in the crossover model with
measurements on coherent hardware, then run the C2C-specific
design-space studies. Either outcome — confirmation or refutation —
is the headline result of the project.

## Kill criterion (written before measuring, per the standing
discipline)

The thesis claims practicality when the dispatch floor reaches
sub-microsecond scale. Decision boundaries, stated now:

- **D_c2c ≤ ~1 µs:** crossover lands ≤ ~400 in-flight ops →
  thesis confirmed for ordinary transactional serving.
- **1 µs < D_c2c ≤ ~3 µs:** crossover ~400–1,200 → viable for
  high-throughput serving; "ordinary OLTP" claim must be narrowed.
- **D_c2c ≥ ~5 µs:** crossover ≥ ~2,000 → thesis fails for serving
  workloads; report as the finding, pivot the paper to the measured
  PCIe/coalescer/OCC contributions.

Both sides of the crossover MUST be re-measured on-target: Grace has
72 Neoverse cores (R_cpu rises) and the GPU side has 132 SMs + HBM3
(R_gpu rises). Laptop constants are void on GH200.

## The experiment list, in priority order

1. **Doorbell microbenchmark** (the decisive number). Port the toy
   engine's Control-block handshake to C2C allocations: host writes a
   64 B request line in (a) GPU memory, (b) CPU memory; resident
   kernel polls; measure round trip both placements. Compare against
   `cudaHostAllocMapped` and plain `malloc` (Grace-Hopper ATS allows
   system-allocated memory). Cross-check against the published GH200
   ping-pong figures (Fusco et al.) — agreement validates method.
2. **On-target baselines:** `cpu_sweep` on Grace (72 cores; also
   record 16-core slice for laptop comparability); `rgi_sweep`
   (launch binding) and `rgi_persist` (doorbell binding) on the GPU.
   Recompute B\* from these four numbers only.
3. **Queue placement 2×2:** submission queue and completion queue
   each in {GPU memory, CPU memory} — measure rendezvous latency for
   all four. Expectation from the GH200 literature: place each queue
   in its *consumer's* memory; verify, don't assume.
4. **Payload-in-line vs. pointer-to-payload:** request slots carrying
   the 16 B key/value inline vs. a pointer the kernel dereferences
   over C2C. Latency and bandwidth per batch size.
5. **Persistent vs. launch binding under identical load** — the WP5
   comparison, now where it matters.
6. **Contention under skew:** zipfian (θ ∈ {0, 0.9, 0.99}) mixed
   workloads through the RGI path; per-bucket lock behavior across
   the coherent link is the architecture question (the toy engine's
   three lock schemes are the controlled-experiment vehicle if RGI's
   results need explanation).
7. **End-to-end SQL on GH200** (stretch, only if Postgres can be
   installed): the multi-client coalescer curve (WP2's benchmark)
   with the persistent binding — the full-stack demonstration.

Items 1–2 settle the thesis. 3–5 are the systems-paper design space.
6–7 are stretch.

## Preparation (do before access; no hardware needed)

- Generalize `bench/baremetal_campaign.sh` (WP5) into
  `bench/c2c_campaign.sh`: arch detection (`sm_90`), allocation-mode
  flags for the doorbell bench, all seven experiments as numbered
  phases, one provenance-headed results file, signal-trapped cleanup
  (never leave a resident kernel on a shared superchip).
- Pre-build a static request-trace file for item 6 so skew runs are
  reproducible.
- Dry-run the entire script on the laptop (WDDM numbers, wrong but
  shaped right) — borrowed GH200 hours must be measurement-only.

## Contracts

- Decision boundaries above are frozen; results reported against them
  verbatim.
- Every figure regenerated from on-target numbers gets a new
  provenance label; laptop-derived figures are never silently
  replaced (keep both, dated).

## Acceptance criteria

- [ ] Items 1–2 complete: doorbell cost (both placements), all four
      model constants on-target, recomputed crossover, decision-rule
      outcome stated in one sentence.
- [ ] Items 3–4 complete with one table each.
- [ ] Results file + updated figures + a 1-page summary suitable for
      forwarding to advisors unedited.

## Risks / pitfalls

- GH200 software stack differences (driver, CUDA version, ATS
  behavior) — the campaign script's provenance header exists so
  results are interpretable later; record `numactl --hardware` too.
- Grace-side `cpu_sweep` uses OpenMP — pin threads
  (`OMP_PROC_BIND=close OMP_PLACES=cores`) or the 72-core scaling
  number will be noise.
- Do not let item 7 (Postgres install on a borrowed superchip) eat
  the access window; it is last for a reason.
- If access is cloud-metered (e.g., Lambda/CoreWeave GH200), the
  campaign script's unattended end-to-end property is the cost
  control — test it twice on the laptop.
