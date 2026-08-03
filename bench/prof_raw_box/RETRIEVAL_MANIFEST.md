# GH200 Retrieval Manifest — instance cleared for termination

**Date:** 2026-07-07. Executed per `GH200_RETRIEVAL.md` against
ubuntu@<GH200_HOST> (Lambda GH200 480GB).

## Verified present in repo (checksum or content check)

- **Measurement sources, box↔repo md5 IDENTICAL (6/6):**
  `engine/rgi_persist2.cu` (incl. `cross` statistics mode),
  `engine/membench.cu`, `engine/rgi_profile_one.cu`,
  `engine/doorbell.cu`, `engine/cpu_sweep.cpp`, `bench/make_plots9.py`.
- **Profiling artifacts** (`bench/prof_raw_box/prof/`, 33 MB): membench
  text outputs (read/write/copy/triad/rand32/64/128/latency), Nsight
  Compute reports `ncu_rgi_{32k,262k,1m}.ncu-rep` + CSV/summary/stdout
  extractions, `nsys_rgi_persist2.qdstrm`.
- **Run logs:** `rerun_1..4.log` (full-sweep repeats; the core-0
  pinning outlier analysis provenance).
- **RGI measurement patch:** `upstream/kernels_gh200_patched.hpp` +
  `upstream/rgi_launch_geometry_cache.patch` (37 lines; for the
  upstream report on the ~875 µs per-launch `cudaGetDeviceProperties`).
- **All measured numbers with statistics** transcribed in
  `bench/gh200_campaign_results.md` (§4b: 40-rep crossover table,
  B ≈ 640 canonical; §7 characterization).

## Box delta inventory result

Every file on the box newer than the v1 engine baseline is either a
repo-identical source (verified by md5), an already-pulled `prof/`
artifact, or the archived RGI patch. No box-only content remains.

## Pre-kill state

- `nvidia-smi` compute apps: none. GPU utilization: 0%.
- No benchmarks running; no persistent kernels resident.

## Deliberately left behind

`~/work/bin/*` binaries (rebuild from source per `HANDOFF_GH200.md` §3),
populated GPU table state, driver/toolkit provisioning.

**VERDICT: nothing of value exists only on the instance. Safe to
terminate.**
