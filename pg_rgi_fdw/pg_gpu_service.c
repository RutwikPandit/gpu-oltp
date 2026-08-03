/* pg_gpu_service.c — GPU OLTP background worker (shared, multi-user engine).
 *
 * One background worker owns the GPU engine (a single CUDA context + one RGI
 * index shared by ALL backends). Backends post requests into shared memory and
 * wait on their latch; the worker processes them against the shared index. This
 * gives cross-connection shared data + persistence for the postmaster lifetime.
 *
 * Two transports:
 *  - a slot ring for single-row ops (gpu_svc_insert/lookup test functions);
 *  - a lock-guarded BULK region (one op at a time) for the FDW table:
 *    SNAPSHOT, FIND_MANY (pushdown/multi-get), INSERT_MANY (PK-checked),
 *    UPDATE_MANY, DELETE_MANY.
 *
 * Requires: shared_preload_libraries = 'pg_rgi_fdw'. Built for PostgreSQL 14.
 */
#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "port/atomics.h"
#include "utils/builtins.h"

#include "rgi_oltp_engine.h"
#include "pg_gpu_service.h"

#define GPU_SVC_NSLOTS     2048
#define GPU_SVC_CAPACITY   (1u << 22)
#define GPU_SVC_FILL       2.0f
#define GPU_SVC_POOL       0.20f

enum { ST_FREE = 0, ST_POSTED = 1, ST_DONE = 2 };

typedef struct GpuReqSlot
{
    pg_atomic_uint32 state;
    int       op;
    uint64    key, value, out_value;
    int       found;
    Latch    *waiter;
} GpuReqSlot;

typedef struct GpuServiceShmem
{
    Latch    *worker_latch;
    pid_t     worker_pid;
    LWLock   *alloc_lock;          /* protects slot allocation */
    LWLock   *bulk_lock;           /* serializes bulk ops */
    /* bulk request channel */
    pg_atomic_uint32 bulk_state;   /* ST_FREE/POSTED/DONE */
    int       bulk_op;
    uint32    bulk_count;          /* in: #inputs; out: #result rows */
    int       bulk_ok;             /* out: 1 ok, 0 PK violation / has-more (snapshot) */
    uint64    bulk_dup;            /* out: offending key on PK violation */
    uint64    bulk_off;            /* in: snapshot page offset */
    int       bulk_req_pid;        /* in: requesting backend pid (stage owner) */
    Latch    *bulk_waiter;
    uint64    bulk_keys[GPU_SVC_BULK_CAP];
    uint64    bulk_vals[GPU_SVC_BULK_CAP];
    /* single-row slot ring */
    GpuReqSlot slots[GPU_SVC_NSLOTS];
} GpuServiceShmem;

static GpuServiceShmem *gpu_svc = NULL;
static shmem_startup_hook_type prev_shmem_startup_hook = NULL;
static volatile sig_atomic_t got_sigterm = false;

PG_FUNCTION_INFO_V1(gpu_svc_insert);
PG_FUNCTION_INFO_V1(gpu_svc_lookup);

void _PG_init(void);
PGDLLEXPORT void gpu_service_main(Datum main_arg);

/* ------------------------------- shmem -------------------------------- */
static Size
gpu_svc_shmem_size(void)
{
    return MAXALIGN(sizeof(GpuServiceShmem));
}

static void
gpu_svc_shmem_startup(void)
{
    bool found;
    if (prev_shmem_startup_hook) prev_shmem_startup_hook();
    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
    gpu_svc = (GpuServiceShmem *) ShmemInitStruct("gpu_service",
                                                  gpu_svc_shmem_size(), &found);
    if (!found)
    {
        int i;
        LWLockPadded *tr = GetNamedLWLockTranche("gpu_service");
        gpu_svc->worker_latch = NULL;
        gpu_svc->worker_pid = 0;
        gpu_svc->alloc_lock = &tr[0].lock;
        gpu_svc->bulk_lock  = &tr[1].lock;
        pg_atomic_init_u32(&gpu_svc->bulk_state, ST_FREE);
        gpu_svc->bulk_waiter = NULL;
        for (i = 0; i < GPU_SVC_NSLOTS; i++)
        {
            pg_atomic_init_u32(&gpu_svc->slots[i].state, ST_FREE);
            gpu_svc->slots[i].waiter = NULL;
        }
    }
    LWLockRelease(AddinShmemInitLock);
}

static void
handle_sigterm(SIGNAL_ARGS)
{
    int save_errno = errno;
    got_sigterm = true;
    if (MyProc) SetLatch(&MyProc->procLatch);
    errno = save_errno;
}

/* ------------------------- worker-side handlers ----------------------- */
/* Worker-private state (only the single worker thread touches these). */
static uint64 snap_total = 0;       /* rows in the frozen snapshot set      */
static uint64 snap_off   = 0;       /* next snapshot page offset            */
static int    stage_owner = 0;      /* pid that opened the current staging  */

/* Emit one snapshot page into the bulk region; sets bulk_count + has-more. */
static void
worker_snapshot_page(RgiEngine *engine)
{
    uint32 m = rgi_snapshot_page(engine, snap_off, GPU_SVC_BULK_CAP,
                                 gpu_svc->bulk_keys, gpu_svc->bulk_vals);
    uint64 span = (snap_total - snap_off > (uint64) GPU_SVC_BULK_CAP)
                      ? (uint64) GPU_SVC_BULK_CAP : (snap_total - snap_off);
    snap_off += span;
    gpu_svc->bulk_count = m;
    gpu_svc->bulk_ok = (snap_off < snap_total) ? 1 : 0;   /* 1 = more pages */
}

static void
worker_do_bulk(RgiEngine *engine)
{
    int op = gpu_svc->bulk_op;
    uint32 n = gpu_svc->bulk_count;
    uint32 i;

    gpu_svc->bulk_ok = 1;
    gpu_svc->bulk_dup = 0;

    switch (op)
    {
        case SVC_SNAPSHOT:                        /* begin + first page */
            snap_total = rgi_snapshot_begin(engine);
            snap_off = 0;
            worker_snapshot_page(engine);
            break;
        case SVC_SNAPSHOT_NEXT:                   /* subsequent pages */
            worker_snapshot_page(engine);
            break;
        case SVC_FIND_MANY:
        {
            /* in: n keys in bulk_keys; out: compacted found (key,val) pairs */
            uint64 *outv = (uint64 *) palloc(sizeof(uint64) * (n ? n : 1));
            int    *fnd  = (int *)    palloc(sizeof(int) * (n ? n : 1));
            uint32  m = 0;
            rgi_find_many(engine, gpu_svc->bulk_keys, outv, fnd, n);
            for (i = 0; i < n; i++)
                if (fnd[i]) { gpu_svc->bulk_keys[m] = gpu_svc->bulk_keys[i];
                              gpu_svc->bulk_vals[m] = outv[i]; m++; }
            pfree(outv); pfree(fnd);
            gpu_svc->bulk_count = m;
            break;
        }
        /* ---- atomic transaction commit (stage -> validate -> apply) ---- */
        case SVC_TXN_BEGIN:
            /* A fresh BEGIN always drops any staging left by a prior (possibly
             * crashed) backend, so stale stage state cannot accumulate. */
            rgi_stage_begin(engine);
            stage_owner = gpu_svc->bulk_req_pid;
            break;
        case SVC_TXN_STAGE_DEL:
            if (stage_owner != gpu_svc->bulk_req_pid) { gpu_svc->bulk_ok = 0; break; }
            rgi_stage_del(engine, gpu_svc->bulk_keys, n);
            break;
        case SVC_TXN_STAGE_UPD:
            if (stage_owner != gpu_svc->bulk_req_pid) { gpu_svc->bulk_ok = 0; break; }
            rgi_stage_upd(engine, gpu_svc->bulk_keys, gpu_svc->bulk_vals, n);
            break;
        case SVC_TXN_STAGE_INS:
            if (stage_owner != gpu_svc->bulk_req_pid) { gpu_svc->bulk_ok = 0; break; }
            rgi_stage_ins(engine, gpu_svc->bulk_keys, gpu_svc->bulk_vals, n);
            break;
        case SVC_TXN_COMMIT:
        {
            uint64 dup = 0;
            if (stage_owner != gpu_svc->bulk_req_pid) { gpu_svc->bulk_ok = 0; break; }
            if (rgi_stage_commit(engine, &dup)) { gpu_svc->bulk_ok = 0; gpu_svc->bulk_dup = dup; }
            stage_owner = 0;
            break;
        }
        case SVC_TXN_ABORT:
            rgi_stage_abort(engine);
            stage_owner = 0;
            break;
        /* ---- legacy single-pass bulk write ops (kept for compatibility) ---- */
        case SVC_INSERT_MANY:
        {
            uint64 dup = 0;
            for (i = 0; i < n; i++) rgi_insert(engine, gpu_svc->bulk_keys[i], gpu_svc->bulk_vals[i]);
            if (rgi_flush_unique(engine, &dup)) { gpu_svc->bulk_ok = 0; gpu_svc->bulk_dup = dup; }
            break;
        }
        case SVC_UPDATE_MANY:
            for (i = 0; i < n; i++) rgi_update(engine, gpu_svc->bulk_keys[i], gpu_svc->bulk_vals[i]);
            rgi_flush(engine);
            break;
        case SVC_DELETE_MANY:
            for (i = 0; i < n; i++) rgi_delete(engine, gpu_svc->bulk_keys[i]);
            break;
        default:
            gpu_svc->bulk_ok = 0;
            break;
    }
}

void
gpu_service_main(Datum main_arg)
{
    RgiEngine *engine;

    pqsignal(SIGTERM, handle_sigterm);
    BackgroundWorkerUnblockSignals();

    gpu_svc->worker_latch = &MyProc->procLatch;
    gpu_svc->worker_pid = MyProcPid;

    engine = rgi_create(GPU_SVC_CAPACITY, GPU_SVC_FILL, GPU_SVC_POOL);
    if (!engine)
    {
        ereport(LOG, (errmsg("pg_gpu_service: failed to create GPU engine; exiting")));
        proc_exit(1);
    }
    ereport(LOG, (errmsg("pg_gpu_service: worker started (pid %d), GPU engine ready", MyProcPid)));

    while (!got_sigterm)
    {
        int i;
        /* single-row slot ring */
        for (i = 0; i < GPU_SVC_NSLOTS; i++)
        {
            GpuReqSlot *s = &gpu_svc->slots[i];
            if (pg_atomic_read_u32(&s->state) != ST_POSTED) continue;
            switch (s->op)
            {
                case SVC_INSERT: rgi_insert(engine, s->key, s->value); rgi_flush(engine); s->found = 1; break;
                case SVC_UPDATE: rgi_update(engine, s->key, s->value); rgi_flush(engine); s->found = 1; break;
                case SVC_DELETE: rgi_delete(engine, s->key); s->found = 1; break;
                default:
                {
                    uint64 v = 0; s->found = rgi_lookup(engine, s->key, &v); s->out_value = v; break;
                }
            }
            pg_write_barrier();
            pg_atomic_write_u32(&s->state, ST_DONE);
            if (s->waiter) SetLatch(s->waiter);
        }
        /* bulk channel */
        if (pg_atomic_read_u32(&gpu_svc->bulk_state) == ST_POSTED)
        {
            worker_do_bulk(engine);
            pg_write_barrier();
            pg_atomic_write_u32(&gpu_svc->bulk_state, ST_DONE);
            if (gpu_svc->bulk_waiter) SetLatch(gpu_svc->bulk_waiter);
        }
        (void) WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                         50L, PG_WAIT_EXTENSION);
        ResetLatch(MyLatch);
    }
    rgi_destroy(engine);
    proc_exit(0);
}

/* ------------------------------ client API ---------------------------- */
bool
gpu_svc_available(void)
{
    return (gpu_svc != NULL && gpu_svc->worker_pid != 0 && gpu_svc->worker_latch != NULL);
}

int
gpu_svc_dispatch(int op, uint64 key, uint64 value, uint64 *out_value)
{
    GpuReqSlot *s = NULL;
    int i, found;

    if (!gpu_svc_available())
        ereport(ERROR, (errmsg("pg_gpu_service: worker not running "
                               "(is 'pg_rgi_fdw' in shared_preload_libraries?)")));

    /* Hold bulk_lock for the whole single-row op so it cannot interleave with a
     * staged transaction commit or a paged snapshot (both also hold bulk_lock).
     * Without this, a single-row write could mutate the engine between staged
     * chunks / snapshot pages and break commit atomicity / snapshot freshness.
     * NB: the single-row WRITE helpers (SVC_INSERT/UPDATE/DELETE) bypass PK and
     * transaction semantics and exist ONLY for the cross-session demo/tests
     * (gpu_svc_insert); they are not the supported write path (use SQL DML). */
    LWLockAcquire(gpu_svc->bulk_lock, LW_EXCLUSIVE);

    LWLockAcquire(gpu_svc->alloc_lock, LW_EXCLUSIVE);
    for (i = 0; i < GPU_SVC_NSLOTS; i++)
        if (pg_atomic_read_u32(&gpu_svc->slots[i].state) == ST_FREE)
        { s = &gpu_svc->slots[i]; pg_atomic_write_u32(&s->state, ST_DONE); break; }
    LWLockRelease(gpu_svc->alloc_lock);
    if (!s) ereport(ERROR, (errmsg("pg_gpu_service: no free request slots")));

    s->op = op; s->key = key; s->value = value; s->out_value = 0; s->found = 0; s->waiter = MyLatch;
    pg_write_barrier();
    pg_atomic_write_u32(&s->state, ST_POSTED);
    SetLatch(gpu_svc->worker_latch);

    while (pg_atomic_read_u32(&s->state) != ST_DONE)
    {
        (void) WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH, 1000L, PG_WAIT_EXTENSION);
        ResetLatch(MyLatch);
        CHECK_FOR_INTERRUPTS();
    }
    pg_read_barrier();
    found = s->found;
    if (out_value) *out_value = s->out_value;
    pg_atomic_write_u32(&s->state, ST_FREE);
    LWLockRelease(gpu_svc->bulk_lock);
    return found;
}

/* Post one bulk op and block until the worker finishes it. Assumes the caller
 * already holds bulk_lock; inputs go in/out via the shared bulk region. */
static void
bulk_post_locked(int op, const uint64 *in_keys, const uint64 *in_vals, uint32 n, uint64 off)
{
    if (in_keys) memcpy(gpu_svc->bulk_keys, in_keys, sizeof(uint64) * n);
    if (in_vals) memcpy(gpu_svc->bulk_vals, in_vals, sizeof(uint64) * n);
    gpu_svc->bulk_op = op;
    gpu_svc->bulk_count = n;
    gpu_svc->bulk_off = off;
    gpu_svc->bulk_req_pid = MyProcPid;
    gpu_svc->bulk_ok = 1;
    gpu_svc->bulk_dup = 0;
    gpu_svc->bulk_waiter = MyLatch;
    pg_write_barrier();
    pg_atomic_write_u32(&gpu_svc->bulk_state, ST_POSTED);
    SetLatch(gpu_svc->worker_latch);

    while (pg_atomic_read_u32(&gpu_svc->bulk_state) != ST_DONE)
    {
        (void) WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH, 5000L, PG_WAIT_EXTENSION);
        ResetLatch(MyLatch);
        CHECK_FOR_INTERRUPTS();
    }
    pg_read_barrier();
    /* leave results in the shared region for the caller; mark slot reusable */
    pg_atomic_write_u32(&gpu_svc->bulk_state, ST_FREE);
}

void
gpu_svc_bulk(int op, const uint64 *in_keys, const uint64 *in_vals, uint32 *count,
             uint64 *out_keys, uint64 *out_vals, int *ok, uint64 *dup_key)
{
    uint32 n = count ? *count : 0;

    if (!gpu_svc_available())
        ereport(ERROR, (errmsg("pg_gpu_service: worker not running "
                               "(is 'pg_rgi_fdw' in shared_preload_libraries?)")));
    if (n > GPU_SVC_BULK_CAP)
        ereport(ERROR, (errmsg("pg_gpu_service: bulk request of %u exceeds cap %u",
                               n, GPU_SVC_BULK_CAP)));

    LWLockAcquire(gpu_svc->bulk_lock, LW_EXCLUSIVE);
    bulk_post_locked(op, in_keys, in_vals, n, 0);
    if (count) *count = gpu_svc->bulk_count;
    if (ok) *ok = gpu_svc->bulk_ok;
    if (dup_key) *dup_key = gpu_svc->bulk_dup;
    if (out_keys) memcpy(out_keys, gpu_svc->bulk_keys, sizeof(uint64) * gpu_svc->bulk_count);
    if (out_vals) memcpy(out_vals, gpu_svc->bulk_vals, sizeof(uint64) * gpu_svc->bulk_count);
    LWLockRelease(gpu_svc->bulk_lock);
}

/* Stream one kind of staged rows in <=CAP chunks (lock already held). Returns 0
 * if the worker rejected a chunk (protocol/owner error), 1 on success. */
static int
stage_stream_locked(int op, const uint64 *keys, const uint64 *vals, uint32 n)
{
    uint32 off;
    for (off = 0; off < n; off += GPU_SVC_BULK_CAP)
    {
        uint32 c = (n - off > GPU_SVC_BULK_CAP) ? GPU_SVC_BULK_CAP : (n - off);
        bulk_post_locked(op, keys + off, vals ? vals + off : NULL, c, 0);
        if (!gpu_svc->bulk_ok) return 0;       /* worker rejected this stage chunk */
    }
    return 1;
}

void
gpu_svc_txn_commit(const uint64 *del_k, uint32 nd,
                   const uint64 *upd_k, const uint64 *upd_v, uint32 nu,
                   const uint64 *ins_k, const uint64 *ins_v, uint32 ni,
                   int *ok, uint64 *dup_key)
{
    int staged_ok;

    if (!gpu_svc_available())
        ereport(ERROR, (errmsg("pg_gpu_service: worker not running "
                               "(is 'pg_rgi_fdw' in shared_preload_libraries?)")));

    /* One lock hold for the whole stage->commit: no other backend can interleave
     * staging, so the staged set is unambiguously owned by this commit. */
    LWLockAcquire(gpu_svc->bulk_lock, LW_EXCLUSIVE);
    bulk_post_locked(SVC_TXN_BEGIN, NULL, NULL, 0, 0);
    staged_ok = gpu_svc->bulk_ok
             && stage_stream_locked(SVC_TXN_STAGE_DEL, del_k, NULL,  nd)
             && stage_stream_locked(SVC_TXN_STAGE_UPD, upd_k, upd_v, nu)
             && stage_stream_locked(SVC_TXN_STAGE_INS, ins_k, ins_v, ni);
    if (!staged_ok)
    {
        /* a stage step failed: drop the partial staging, never apply */
        bulk_post_locked(SVC_TXN_ABORT, NULL, NULL, 0, 0);
        LWLockRelease(gpu_svc->bulk_lock);
        ereport(ERROR, (errmsg("pg_gpu_service: transaction staging failed (protocol error)")));
    }
    bulk_post_locked(SVC_TXN_COMMIT, NULL, NULL, 0, 0);
    if (ok) *ok = gpu_svc->bulk_ok;
    if (dup_key) *dup_key = gpu_svc->bulk_dup;
    LWLockRelease(gpu_svc->bulk_lock);
}

void
gpu_svc_snapshot_all(uint64 **out_keys, uint64 **out_vals, uint32 *count)
{
    uint32 cap = GPU_SVC_BULK_CAP;
    uint32 m = 0;
    uint64 *ks, *vs;
    int more;

    if (!gpu_svc_available())
        ereport(ERROR, (errmsg("pg_gpu_service: worker not running "
                               "(is 'pg_rgi_fdw' in shared_preload_libraries?)")));

    ks = (uint64 *) palloc(sizeof(uint64) * cap);
    vs = (uint64 *) palloc(sizeof(uint64) * cap);

    LWLockAcquire(gpu_svc->bulk_lock, LW_EXCLUSIVE);
    bulk_post_locked(SVC_SNAPSHOT, NULL, NULL, 0, 0);
    do {
        uint32 pc = gpu_svc->bulk_count;
        more = gpu_svc->bulk_ok;                 /* 1 => more pages remain */
        if (m + pc > cap) {
            while (m + pc > cap) cap *= 2;
            ks = (uint64 *) repalloc(ks, sizeof(uint64) * cap);
            vs = (uint64 *) repalloc(vs, sizeof(uint64) * cap);
        }
        memcpy(ks + m, gpu_svc->bulk_keys, sizeof(uint64) * pc);
        memcpy(vs + m, gpu_svc->bulk_vals, sizeof(uint64) * pc);
        m += pc;
        if (more) bulk_post_locked(SVC_SNAPSHOT_NEXT, NULL, NULL, 0, 0);
    } while (more);
    LWLockRelease(gpu_svc->bulk_lock);

    *out_keys = ks; *out_vals = vs; *count = m;
}

Datum
gpu_svc_insert(PG_FUNCTION_ARGS)
{
    gpu_svc_dispatch(SVC_INSERT, (uint64) PG_GETARG_INT64(0), (uint64) PG_GETARG_INT64(1), NULL);
    PG_RETURN_VOID();
}

Datum
gpu_svc_lookup(PG_FUNCTION_ARGS)
{
    uint64 v = 0;
    int found = gpu_svc_dispatch(SVC_LOOKUP, (uint64) PG_GETARG_INT64(0), 0, &v);
    if (!found) PG_RETURN_NULL();
    PG_RETURN_INT64((int64) v);
}

/* ------------------------------- init --------------------------------- */
void
_PG_init(void)
{
    BackgroundWorker worker;
    if (!process_shared_preload_libraries_in_progress) return;

    RequestAddinShmemSpace(gpu_svc_shmem_size());
    RequestNamedLWLockTranche("gpu_service", 2);   /* [0]=alloc, [1]=bulk */
    prev_shmem_startup_hook = shmem_startup_hook;
    shmem_startup_hook = gpu_svc_shmem_startup;

    memset(&worker, 0, sizeof(worker));
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
    worker.bgw_restart_time = 5;
    snprintf(worker.bgw_name, BGW_MAXLEN, "pg_gpu_service");
    snprintf(worker.bgw_type, BGW_MAXLEN, "pg_gpu_service");
    snprintf(worker.bgw_library_name, BGW_MAXLEN, "pg_rgi_fdw");
    snprintf(worker.bgw_function_name, BGW_MAXLEN, "gpu_service_main");
    worker.bgw_notify_pid = 0;
    RegisterBackgroundWorker(&worker);
}
