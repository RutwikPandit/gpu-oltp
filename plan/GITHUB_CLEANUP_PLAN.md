# GitHub Upload — Cleanup Plan

**Goal:** get the repo into a state where teammates can pull the latest deck
+ buildable code, without leaking secrets, bloating the repo, or
re-publishing other people's licensed work.

**Status: EXECUTED 2026-07-08.** Done: Tier 0 (VULTR_API.txt deleted —
*revoke the key in the Vultr console if it was ever live*; SSH IP scrubbed;
sudo-password default removed from all scripts), Tier 1 (`.gitignore` added
for artifacts/profiler-blobs/pptx), Tier 2 (RGI full-file copy deleted, only
the patch + `upstream/README.md` attribution kept; `CPU-GPU_paper.txt`
deleted), Tier 3 (dead `win_*.bat`/`diag.cu`/`test_min.cu` deleted; superseded
docs + chat log moved to `archive/`, which is gitignored), Tier 4
(Vultr→Lambda fixed across data/figure/deck files). **Remaining:** Tier 5
(relativize hardcoded `/mnt/c/...` paths in build scripts — needs a test
build) and `git init` + first commit. Git-tracked footprint is now ~3.7 MB
(from 53 MB).

**Scope note:** the repo is `gpu_oltp/`. It is **not yet a git repo**
(`git init` pending). Total size **53 MB**, but ~42 MB of that is
build artifacts + profiler blobs that should never be committed. The real
source + docs are ~2–3 MB.

---

## TIER 0 — SECRETS (do FIRST; blockers for any upload, public or private)

These must be handled before `git init`, because once committed they live
in history even if deleted later.

| Item | Where | Action |
|---|---|---|
| **Vultr API key (leftover from an abandoned plan)** | `VULTR_API.txt` (36-char credential) | Vultr was the *original* cloud plan; the actual GH200 ran on **Lambda**, so this key was likely never deployed — but it is still a real credential in a file. **DELETE it, and revoke it in the Vultr console if it is a live key.** Never commit. |
| **GH200 SSH IP `<GH200_HOST>` + key name `rutwik_ed25519`** | `HANDOFF_GH200.md`, `HANDOFF_PERF_NSIGHT.md`, `GH200_RETRIEVAL.md`, `bench/prof_raw_box/RETRIEVAL_MANIFEST.md` — **and embedded inside the binary `*.ncu-rep` / `*.qdstrm` profiler files** | Instance is being killed so the IP is ephemeral, but still scrub it: replace with `ubuntu@<GH200_HOST>`. The binary profiler files embed the hostname too — another reason Tier 1 excludes them. |
| **sudo password `<sudo-pw>`** | default in `run_tests.sh`, `run_gpu_db.sh`, `run_bench_sql.sh` (`PW="${PGSUDO_PW:-<sudo-pw>}"`); also in `demo/`, `pg_*/‌*.sql`, `comp_arch_db_explainer/*`, chat export | Low severity (local WSL dev box), but scrub it: change the default to empty so it must come from `$PGSUDO_PW` — `PW="${PGSUDO_PW:?set PGSUDO_PW}"`. Remove literal `<sudo-pw>` from docs/SQL. |

After scrubbing, grep must return clean:
`grep -rn "<GH200_HOST>\|rutwik_ed25519\|<sudo-pw>" .` and
`ls VULTR_API.txt` → not found.

---

## TIER 1 — Build artifacts & profiler blobs (exclude via `.gitignore`)

~42 MB of the 53 MB. All regenerable; none belong in git.

| Pattern / path | Size | Why exclude |
|---|---|---|
| `bench/prof_raw_box/prof/nsys_rgi_persist2.qdstrm` | **29 MB** | binary Nsight Systems capture; half the repo; embeds SSH host |
| `bench/prof_raw_box/prof/*.ncu-rep` (×3) | ~3.4 MB | binary Nsight Compute reports; embed SSH host; the *extracted* CSV/txt summaries in `bench/prof_raw/` are enough |
| `*.so` (`librgioltp.so`, `libgpuoltp.so`, `pg_*/*.so`) | ~3.5 MB | compiled; rebuild from source |
| `*.o`, `*.bc` (`pg_*/`) | ~1 MB | compiled objects; also **predate current sources** (Jun 8 vs Jul sources) — stale |
| `oltp_bench` and any bare-name binaries | ~1 MB | compiled benchmark |
| `SLIDES_1HR.pptx` | **9 MB** | see Tier 3 — regenerable from the `.md`; use Git LFS only if the pptx is the artifact of record |

**Decision needed:** keep the small extracted profiler summaries
(`bench/prof_raw/*.txt`, `*.csv`) as committed evidence (recommended — they
are the numbers behind the figures), and drop the heavy binary
`prof_raw_box/prof/` reports. Or, if teammates want the GUI-openable
reports, put `prof_raw_box/` behind Git LFS.

---

## TIER 2 — Licensing / dependency (RGI is not ours to publish)

`RobustGPUIndexing` (Hyoungjoo Kim, CMU; Apache-2.0) is a **sibling
directory dependency**, not part of this repo. Two concrete issues:

1. **Do NOT vendor RGI into this repo.** Teammates get it by cloning it
   separately. The README (Tier 6) must say: clone RGI, then point the
   build at it via an `RGI_INCLUDE` variable (Tier 5 relativization).
   Consider a git **submodule** if the team wants one-command checkout.
2. **`upstream/kernels_gh200_patched.hpp` is a full copy of Hyoungjoo's
   Apache-licensed source file** (40 KB). Copying his file into your repo
   is an attribution/licensing gray area. **Keep only the diff**
   (`upstream/rgi_launch_geometry_cache.patch`, 2 KB) plus a short note
   crediting RGI and describing the ~875 µs `cudaGetDeviceProperties`
   launch bug. Delete the full-file copy. (This patch is also the
   deliverable to report upstream to Hyoungjoo.)

Also: `CPU-GPU_paper.txt` (60 KB) is the **full text of Fusco et al.'s
paper** — copyrighted, do not republish. Delete it; cite the arXiv link
(2408.11556) in the doc that references it.

---

## TIER 3 — Dead / superseded / internal files

### 3a. Delete (dead toolchain-war + throwaways)
| File(s) | Why |
|---|---|
| `win_probe.bat`, `win_get_vs.bat`, `win_install_vs.bat`, `win_build.bat`, `win_run.bat`, `win_test.bat` | early native-Windows/MSVC toolchain attempts, abandoned for WSL; dead |
| `diag.cu`, `test_min.cu` | one-off CUDA/toolchain sanity throwaways |
| `cursor_cursor_assistance_request_docume.md` (251 KB) | raw AI-assistant chat export from project start; not documentation, should not be public |

### 3b. Superseded by current versions
| File | Superseded by | Action |
|---|---|---|
| `PRESENTATION.md` | `SLIDES_1HR.md` | delete |
| `PROJECT_SUMMARY.md` | the GH200 campaign; it is **entirely pre-campaign** (projected C2C, 5,600→176 crossover, no pointer to results) | delete, or rewrite to point at `bench/gh200_campaign_results.md` |
| `SLIDES_1HR.pptx` / `.pdf` | `SLIDES_GH200_UPDATE.*` is the latest talk | keep one rendered `.pdf` of each; drop the 9 MB `.pptx` (regenerate from `.md` via Marp when needed) |
| `pg_gpu_fdw/` (toy per-session FDW) | `pg_rgi_fdw/` | keep ONLY if you still want the OLAP bandwidth-scan demo; otherwise move to `attic/`. It has no sharing/pushdown/PK. Label clearly as superseded either way. |

### 3c. Internal / working docs — move to a gitignored `internal/` (not public)
These are operational notes with the SSH host / ops details, valuable to
you but not part of the published project:
`HANDOFF_GH200.md`, `HANDOFF_PERF_NSIGHT.md`, `GH200_RETRIEVAL.md`,
`bench/prof_raw_box/RETRIEVAL_MANIFEST.md`, `PRESENTATION_PLAN_GH200_UPDATE.md`,
`Summary.txt`, `VULTR_API.txt` (already deleted in Tier 0).

### 3d. Incomplete WIP
`plan/primer_sections/` — six partial files (~450 KB) from the never-finished
PRIMER (interrupted by session limits); no assembled `PRIMER.md` exists.
Either finish + assemble, or exclude from the upload and keep on a WIP
branch. Recommend: exclude for now.

### 3e. Ambiguous — confirm with owner
`Prelim.pdf`, `Final.pdf` (~920 KB each) look like course deliverables, not
code artifacts. Confirm whether they belong in this repo; likely exclude.

---

## TIER 4 — Fix stale content in KEPT docs (from the verified fact-base ledger)

These ship to teammates, so the errors matter:

1. **Cloud provider: the box was Lambda, not Vultr.** `SLIDES_GH200_UPDATE.md`
   ("Lambda Cloud") is **correct**. The error is the other direction: **8
   files wrongly say "Vultr"** and must be changed to **Lambda** —
   `bench/gh200_campaign_results.md` (provenance §), `bench/make_plots8.py`,
   `bench/make_plots9.py`, `HANDOFF_GH200.md` (§3), `GH200_RETRIEVAL.md`,
   `bench/prof_raw_box/RETRIEVAL_MANIFEST.md`, `SLIDES_1HR.md`, and this
   plan. (The `nsys` "Lambda apt repos" note in the results file is right —
   Lambda Stack ships apt repos.) Global find/replace Vultr→Lambda.
2. **`bench/gh200_campaign_results.md` §4b bolds the B=512 row** as if it
   were the crossover; every claim elsewhere says **B≈640** and B=512 is
   explicitly not claimed. De-bold.
3. **`comp_arch_db_explainer/FULL_PROJECT_EXPLAINER.md` is internally
   inconsistent**: only §8 has the measured B≈640 update; TL;DR, §33, Q&A
   Q5, fig-guide, appendix still say doorbell is *projected* 0.5 µs /
   crossover 194. Reconcile globally before sharing, or add a top banner
   pointing to the campaign results as the source of truth.
4. Minor: two old docs disagree on the *superseded model* crossover
   (5,600/176 vs 6,200/194) — harmless once PROJECT_SUMMARY is
   deleted/rewritten.

---

## TIER 5 — Relativize hardcoded paths (so teammates can build)

20 files hardcode `/mnt/c/Users/rutwi/OneDrive/...` or `C:\Users\rutwi\...`.
The build/run scripts must work on any clone:

- `build_all.sh`, `build_bench.sh`, `build_persist.sh`, `bench/regen_figs.sh`,
  `run_*.sh`: replace the hardcoded `ROOT=/mnt/c/Users/rutwi/...` with
  `ROOT="$(cd "$(dirname "$0")/.." && pwd)"` and take RGI via
  `RGI_INCLUDE="${RGI_INCLUDE:-../RobustGPUIndexing/include}"`.
- Docs (`plan/00_MASTER_PLAN.md`, HANDOFF*, etc.): the absolute paths are
  narrative, lower priority — fix opportunistically or leave in the
  gitignored `internal/` set.

---

## TIER 6 — Proposed final repo layout + README

```
gpu-oltp/                         (rename from gpu_oltp for GitHub)
├── README.md                     REWRITE: what it is, how to build, where the
│                                 deck + results live, RGI dependency setup
├── LICENSE                       add one (MIT/Apache) — currently none
├── .gitignore                    (Tier 1 patterns)
├── Makefile
├── engine/                       KEEP all .cu/.h/.cpp (real code)
├── pg_rgi_fdw/                   KEEP (FDW + worker + tests); scrub artifacts+pw
├── bench/
│   ├── make_plots*.py, *.sh      KEEP (figure generators)
│   ├── fig*.png                  KEEP (results figures)
│   ├── gh200_campaign_results.md KEEP (authoritative results; fix Tier 4)
│   └── prof_raw/                 KEEP small text/CSV summaries (drop prof_raw_box/)
├── comp_arch_db_explainer/       KEEP (fix Tier 4 staleness, scrub pw)
├── demo/                         KEEP (scrub pw)
├── docs/
│   ├── SLIDES_GH200_UPDATE.md/pdf   LATEST DECK (what teammates asked for)
│   ├── SLIDES_1HR.md/pdf            full talk
│   └── upstream_rgi_patch/          the .patch + attribution note (Tier 2)
├── plan/                         KEEP master plan + WP0-9 (relativize paths)
└── internal/   (GITIGNORED)      HANDOFF*, RETRIEVAL, VULTR_API(gone), pptx, WIP
```

**README must include:** one-paragraph what-it-is; the RGI-clone-and-set-
`RGI_INCLUDE` step; `bash build_all.sh` + `bash run_tests.sh`; a pointer to
`docs/SLIDES_GH200_UPDATE.pdf` (latest deck) and
`bench/gh200_campaign_results.md` (numbers); the honest-scope one-liner
(memory-resident high-throughput serving, B≈640 measured).

---

## TIER 7 — Recommended `.gitignore`

```gitignore
# build artifacts
*.so
*.o
*.bc
oltp_bench
rgi_sweep
rgi_persist*
doorbell*
cpu_sweep
membench
rgi_profile_one
# profiler binaries (keep extracted summaries in bench/prof_raw/)
*.ncu-rep
*.qdstrm
bench/prof_raw_box/prof/
# large regenerable exports
*.pptx
# secrets / internal (defense in depth)
VULTR_API.txt
internal/
# os / editor
.DS_Store
```

---

## Execution order (checklist)

1. **Tier 0** — delete `VULTR_API.txt`, **revoke the Vultr key**, scrub IP + password. *(do before git init)*
2. **Tier 2** — drop the RGI full-file copy + `CPU-GPU_paper.txt`; keep the patch with attribution.
3. **Tier 3** — delete dead `.bat`/`diag`/`test_min`/chat-export; move internal docs to `internal/`; decide on `pg_gpu_fdw/`, `Prelim/Final.pdf`, `primer_sections/`.
4. **Tier 5** — relativize build/run scripts.
5. **Tier 4** — fix Vultr→Lambda (8 files), B=512 bolding, explainer inconsistency.
6. **Tier 6** — reorganize into the layout, write README + LICENSE.
7. **Tier 7** — add `.gitignore`.
8. `git init`, verify `git status` shows no artifacts/secrets, first commit, push to a **private** repo first, re-scan, then make public if desired.

---

## What teammates actually asked for (fast path)

If they need the deck + code *today* and the full cleanup can wait, the
minimum-safe subset is: **Tier 0 (secrets) is non-negotiable**, then share
`docs/SLIDES_GH200_UPDATE.pdf` + the `engine/`, `pg_rgi_fdw/`, `bench/`
(scripts+figs), `build_*.sh` (relativized), and
`bench/gh200_campaign_results.md`. Everything else is polish.
