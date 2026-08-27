> 👉 **New here? Read [`START_HERE.md`](START_HERE.md)** — the single source
> of truth (architecture, measured results, design decisions, how to build/run,
> current status). Latest talk: [`SLIDES_GH200_UPDATE.md`](SLIDES_GH200_UPDATE.md).

# gpu_oltp - GPU-resident OLTP engine + Postgres integration

A GPU-resident OLTP prototype for PostgreSQL: a persistent-kernel toy engine,
an RGI-backed warp-cooperative GPU index, and a writeable Postgres FDW backed by
a shared GPU-service background worker. Real SQL `INSERT`/`SELECT`/`UPDATE`/
`DELETE` runs against the GPU-resident index, with simple key qual pushdown,
PK/UNIQUE enforcement, and transaction buffering.

> **Hardware note:** development and SQL validation use an RTX 4060 laptop
> (Ada, PCIe). The C2C doorbell and standalone persistent-runtime results were
> measured on a Lambda GH200 on 2026-07-07; full SQL-over-C2C remains unmeasured.

## Layout

```text
engine/gpu_oltp_engine.cu   toy index + persistent kernel + queue + lock-study microbench
engine/rgi_oltp_engine.cu   RGI-backed batched GPU index wrapper
pg_rgi_fdw/                 writeable FDW + shared GPU-service background worker
pg_gpu_fdw/                 earlier toy-engine FDW + aggregate-scan demo
bench/                      benchmark scripts, report, and figures
```

## Environment

Postgres + the FDW build cleanly on Linux. On this laptop use WSL2 Ubuntu:

```bash
# 1. CUDA toolkit for WSL2
nvcc --version
nvidia-smi

# 2. Postgres dev headers (validated with PostgreSQL 14.x)
sudo apt-get install -y postgresql-14 postgresql-server-dev-14 build-essential
```

## Build & Run

```bash
cd gpu_oltp
make

# microbench: <capacity> <nkeys> <nops> <scheme> <zipf_theta> <write%>
# scheme: 0=lockfree 1=bucketlock 2=globallock; theta 0=uniform
./oltp_bench 16777216 4194304 4194304 0 0.0 50

# locking study sweep
make sweep
```

Expected output per run: preload throughput, workload throughput, amortized
ns/op, and single-op PCIe round-trip latency.

## Postgres FDW + Shared GPU Service

The current SQL integration is `pg_rgi_fdw/`. It links the RGI engine and
exposes a shared GPU-resident table:

```sql
CREATE EXTENSION pg_rgi_fdw;
CREATE SERVER rgi FOREIGN DATA WRAPPER pg_rgi_fdw;
CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;

INSERT INTO kv_rgi VALUES (42, 1234);
SELECT v FROM kv_rgi WHERE k = 42;
```

For shared cross-connection state, enable:

```conf
shared_preload_libraries = 'pg_rgi_fdw'
```

Then restart Postgres and use the scripts in `pg_rgi_fdw/`, especially
`enable_worker.sql`, `rgi_smoke.sql`, `correctness.sql`, `txn_test.sql`,
`pk_test.sql`, and `rt_pushdown.sql`.

## Validation Checklist

- `make` clean (set `ARCH` if not on `sm_89`).
- `pg_rgi_fdw/correctness.sql` passes against a Postgres heap oracle.
- `pg_rgi_fdw/txn_test.sql` verifies commit, rollback, and read-your-writes.
- `pg_rgi_fdw/atomic_test.sql` verifies all-or-nothing commit (multi-chunk dup,
  delete-then-failed-insert) — each test needs a clean GPU index (restart).
- `pg_rgi_fdw/keyupd_test.sql` verifies key-changing UPDATE is PK-validated and
  delete-then-reinsert in one txn succeeds.
- `pg_rgi_fdw/keyswap_test.sql` verifies multi-row key swaps/chains are rejected
  cleanly (unsupported) while non-overlapping renames succeed.
- `pg_rgi_fdw/snap_page_test.sql` verifies the paged scan returns all rows (>262k).
- `pg_rgi_fdw/pk_test.sql` verifies duplicate-key rejection.
- `pg_rgi_fdw/rt_pushdown.sql` verifies pushed point and bound-param multi-get.
- `compute-sanitizer ./oltp_bench ...` is clean before trusting microbenchmarks.

## Known Limitations

- **Atomicity is solid, isolation is not.** Commit is all-or-nothing
  (validate-the-whole-write-set-before-mutating, then no-fail apply), with
  rollback and read-your-writes. But there is **no write-write conflict
  detection** (≈ Read-Committed + RYW; concurrent writers to the same key are
  last-committer-wins) until OCC validation is added, and **no
  subtransactions/SAVEPOINT** (top-level xact callback only).
- This is best described as a **GPU-resident index / access-method** prototype,
  not a relational tuple store: RGI values are 32-bit row ids, so wide/
  multi-column rows need a side row-store.
- No durability/WAL/recovery; HBM state is lost on postmaster restart.
- Multi-row key-changing UPDATEs with overlapping old/new keys (swaps like
  `1<->2`, chains like `k=k+1`) are rejected (`ERRCODE_FEATURE_NOT_SUPPORTED`):
  the key-collapsed transaction buffer cannot represent row identity through
  overlapping renames. Single-row and non-overlapping renames are supported.
- No joins, secondary indexes, range indexes, aggregate pushdown, triggers, or
  full constraint support beyond PK/UNIQUE.
- The shared service is a **single worker** with one `bulk_lock`; bulk requests
  and commits serialize (it is a scheduler/coalescer and must be measured as a
  bottleneck before any concurrent-scaling claim). The full-table scan is paged
  (no truncation).
