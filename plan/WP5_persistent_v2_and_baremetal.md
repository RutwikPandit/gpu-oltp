# WP5 — Persistent Kernel v2 + Bare-Metal PCIe Campaign

**Effort:** ~1 week of work + externally-gated hardware access.
**Priority:** R3 (first half). Preparation is NOT gated — do it early so
hardware days are spent measuring.
**Depends on:** hardware (a bare-metal Linux GPU box, e.g., a lab
A100/H100/4090 machine). WP1 should land first (ceilings final).

## Objective

(a) Complete the persistent-kernel binding (erase + reclamation), and
(b) run the measurement campaign that the WSL/WDDM laptop cannot:
doorbell cost and dispatch floors on bare-metal Linux PCIe — the first
hardware checkpoint of the thesis mechanism, available without any
GB-class access.

## Current state

- `engine/rgi_persist_engine.cu` (v1, built 2026-06-09, additive):
  216-resident-block kernel, two-level doorbell (block 0 polls the
  mapped line, republishes {batch, count} to device memory; stop
  sentinel), atomic arrival counter, calls RGI's public device API.
  Validated 1024/1024 on persistent FIND and INSERT.
- Measured on WSL2/WDDM: doorbell rendezvous 75–90 µs vs. kernel
  launch 13–28 µs — persistent is 5–6× worse THERE; the deck/docs
  state the bare-metal expectation only conditionally.
- v1 scope gap: no ERASE (RGI's DEBRA reclamation drains limbo bags at
  kernel exit; a never-exiting kernel needs restructured drains).
- Known measurement artifact: large-B persistent numbers include
  payload H2D each rendezvous; launch-mode sweep pre-stages keys.

## Design — v2 completeness (do on the laptop, before hardware)

1. **Erase support:** construct the DEBRA reclaimer context in the
   persistent kernel (mirror `batch_kernel`'s preamble: shmem buffer,
   `begin/end_critical_section` around each batch iteration). Dispatch
   `request_type_erase` through `cooperative_erase`.
2. **Epoch drains at quiescent points:** between batches, all blocks
   already rendezvous at the arrival counter — that is a global
   quiescent point. Every K batches (start K=16), run the drain path
   (`drain_all`) with all blocks participating before signaling done.
   Read `simple_debra_reclaim.hpp`'s drain signature first; verify it
   tolerates being called repeatedly mid-kernel (it is called with
   block and warp tiles + allocator — confirm no exit-only assumption).
3. **Fair large-B comparison:** add a mode where payload is
   pre-staged device-side for BOTH bindings, so the floor comparison
   isolates dispatch (small B) and the ceiling comparison isolates
   execution (large B). Keep the H2D-inclusive numbers too (they are
   the honest end-to-end story).
4. Validation: persistent INSERT+FIND+ERASE round-trip test; run the
   engine's existing differential pattern (insert N, erase N/13, find
   all, compare against host model). Slab accounting before/after a
   delete-heavy run shows reclamation actually returning slabs
   (RGI's allocator print shows allocated-slab counts).

## The bare-metal campaign (script everything now; run on access)

One script (`bench/baremetal_campaign.sh`) that produces one results
file, in order:

1. `nvidia-smi` inventory + driver/CUDA versions (provenance header).
2. **Doorbell microbenchmark:** the toy engine's single-op round trip
   (CPU→GPU→CPU through mapped memory) — the WDDM-confounded number;
   expectation: low single-digit µs on Linux. THE key number.
3. **Launch floor:** `rgi_sweep` (launch binding), B = 1 … 1 M.
4. **Persistent floor:** `rgi_persist` (doorbell binding), same sweep.
5. **CPU baseline:** `cpu_sweep` on that machine's cores (the
   crossover must use same-box numbers, not laptop numbers).
6. Recompute B\*: with measured D_launch, D_doorbell, R_gpu, R_cpu.
   Auto-emit a small markdown table.

**Decision rule, written before measuring:** if D_doorbell < D_launch
on bare-metal PCIe, the persistent binding becomes the preferred
binding on Linux and the thesis mechanism (floor reduction via
resident polling) is demonstrated pre-C2C. If not, that is a finding
about mapped-memory polling costs and the C2C question stands on the
GH200 published numbers alone — say which outcome occurred, plainly.

## Contracts

- All additive; the launch binding stays the default until the
  campaign justifies switching per-platform.
- Per-block co-residency invariant (grid = exact occupancy) must be
  re-derived on the new GPU (different SM count / occupancy).
- No other kernels may launch while the persistent grid is resident —
  campaign script must order phases accordingly (launch sweep first).

## Acceptance criteria

- [ ] v2 passes insert+find+erase validation with reclamation
      verified (slab counts return).
- [ ] Campaign script runs end-to-end unattended on the laptop (with
      WDDM numbers) — proving it needs zero babysitting on borrowed
      hardware time.
- [ ] On bare-metal: one results file, all five measurements, the
      recomputed crossover, and the decision-rule outcome stated.

## Risks / pitfalls

- DEBRA drain mid-kernel is the real unknown; if `drain_all` assumes
  kernel exit (e.g., frees shmem-resident bags), fall back to:
  persistent kernel uses the dummy reclaimer + a periodic STOP/RELAUNCH
  epoch (drain via kernel exit every N seconds) and document the
  compromise. Do not ship subtle reclamation races to hit a date.
- Borrowed-machine etiquette: the persistent kernel pins SMs at 100%;
  coordinate time slots; the script must trap signals and set the stop
  sentinel on exit (never leave a spinning kernel on a shared box).
- If the box has a different arch (e.g., sm_80/sm_90), build flags
  change (`-arch`); the campaign script should detect and rebuild.
