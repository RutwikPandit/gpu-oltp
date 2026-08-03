/* pg_gpu_fdw.c — writeable Postgres FDW backed by the GPU OLTP engine.
 *
 * A foreign table kv(k bigint, v bigint) maps to a GPU-resident hash index
 * served by a persistent kernel (see ../engine/gpu_oltp_engine.cu). SQL writes
 * (INSERT/UPDATE/DELETE) and reads (SELECT) execute on the GPU:
 *
 *   - SELECT  -> gpu_oltp_snapshot()  (full table copied from GPU; PG applies WHERE)
 *   - INSERT  -> gpu_oltp_insert(k, v)
 *   - UPDATE  -> gpu_oltp_update / (delete old + insert new if key changed)
 *   - DELETE  -> gpu_oltp_delete(k)
 *
 * Scope (sprint scaffold): exactly two int8 columns (key, value); one engine
 * instance per backend process (data is visible within a session). Multi-backend
 * sharing via a GPU-owning background worker is future work, mirroring PG-Strom's
 * GPU Service model.
 *
 * Built for PostgreSQL 14 (note the PG14 AddForeignUpdateTargets signature).
 */
#include "postgres.h"

#include "access/htup_details.h"
#include "access/sysattr.h"
#include "catalog/pg_type.h"
#include "executor/executor.h"
#include "foreign/fdwapi.h"
#include "foreign/foreign.h"
#include "nodes/makefuncs.h"
#include "optimizer/appendinfo.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "storage/ipc.h"
#include "utils/builtins.h"
#include "utils/rel.h"

#include "gpu_oltp_engine.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(pg_gpu_fdw_handler);
PG_FUNCTION_INFO_V1(pg_gpu_fdw_validator);

/* bandwidth-scan SQL entry points (data resident in HBM; aggregate on GPU) */
PG_FUNCTION_INFO_V1(gpu_scan_load);
PG_FUNCTION_INFO_V1(gpu_scan_agg_sum);
PG_FUNCTION_INFO_V1(gpu_scan_agg_count_lt);
PG_FUNCTION_INFO_V1(gpu_scan_kernel_ms);

/* Hash-table capacity for the per-backend engine (rounded up to pow2 by engine). */
#define GPU_FDW_CAPACITY (1u << 20)

/* ----- per-backend GPU engine (lazily created) ------------------------- */
static GpuOltpEngine *g_engine = NULL;
static GpuScanCol    *g_scan = NULL;   /* HBM-resident column for BW aggregates */

static void
gpu_fdw_atexit(int code, Datum arg)
{
    if (g_engine)
    {
        gpu_oltp_destroy(g_engine);
        g_engine = NULL;
    }
    if (g_scan)
    {
        gpu_scan_free(g_scan);
        g_scan = NULL;
    }
}

static GpuOltpEngine *
gpu_fdw_get_engine(void)
{
    if (g_engine == NULL)
    {
        g_engine = gpu_oltp_create(GPU_FDW_CAPACITY, 1024);
        if (g_engine == NULL)
            ereport(ERROR,
                    (errmsg("pg_gpu_fdw: failed to create GPU OLTP engine")));
        on_proc_exit(gpu_fdw_atexit, (Datum) 0);
    }
    return g_engine;
}

/* ----- scan state ------------------------------------------------------ */
typedef struct GpuScanState
{
    uint64_t   *keys;
    uint64_t   *vals;
    uint64_t    n;
    uint64_t    cur;
} GpuScanState;

/* ----- modify state ---------------------------------------------------- */
typedef struct GpuModifyState
{
    AttrNumber  key_junk;   /* resjunk attno of the old key (UPDATE/DELETE) */
} GpuModifyState;

/* ====================== scan callbacks ================================= */
static void
gpuGetForeignRelSize(PlannerInfo *root, RelOptInfo *baserel, Oid foreigntableid)
{
    baserel->rows = 1000;   /* rough placeholder estimate */
    baserel->fdw_private = NULL;
}

static void
gpuGetForeignPaths(PlannerInfo *root, RelOptInfo *baserel, Oid foreigntableid)
{
    Cost startup_cost = 1.0;
    Cost total_cost = startup_cost + baserel->rows;

    add_path(baserel, (Path *)
             create_foreignscan_path(root, baserel,
                                     NULL,                 /* default pathtarget */
                                     baserel->rows,
                                     startup_cost, total_cost,
                                     NIL,                  /* no pathkeys */
                                     NULL,                 /* no outer rel */
                                     NULL,                 /* no extra plan */
                                     NIL));                /* no fdw_private */
}

static ForeignScan *
gpuGetForeignPlan(PlannerInfo *root, RelOptInfo *baserel, Oid foreigntableid,
                  ForeignPath *best_path, List *tlist, List *scan_clauses,
                  Plan *outer_plan)
{
    /* We push nothing down: let PG re-check all quals on returned rows. */
    scan_clauses = extract_actual_clauses(scan_clauses, false);

    return make_foreignscan(tlist, scan_clauses, baserel->relid,
                            NIL,    /* fdw_exprs */
                            NIL,    /* fdw_private */
                            NIL,    /* fdw_scan_tlist */
                            NIL,    /* fdw_recheck_quals */
                            outer_plan);
}

static void
gpuBeginForeignScan(ForeignScanState *node, int eflags)
{
    GpuScanState *st;

    if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
        return;

    st = (GpuScanState *) palloc0(sizeof(GpuScanState));
    st->n = gpu_oltp_snapshot(gpu_fdw_get_engine(), &st->keys, &st->vals);
    st->cur = 0;
    node->fdw_state = (void *) st;
}

static TupleTableSlot *
gpuIterateForeignScan(ForeignScanState *node)
{
    GpuScanState   *st = (GpuScanState *) node->fdw_state;
    TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;

    ExecClearTuple(slot);
    if (st == NULL || st->cur >= st->n)
        return slot;                       /* end of scan */

    /* Column 0 = key, column 1 = value (both int8). */
    slot->tts_values[0] = Int64GetDatum((int64) st->keys[st->cur]);
    slot->tts_isnull[0] = false;
    if (slot->tts_tupleDescriptor->natts > 1)
    {
        slot->tts_values[1] = Int64GetDatum((int64) st->vals[st->cur]);
        slot->tts_isnull[1] = false;
    }
    st->cur++;
    ExecStoreVirtualTuple(slot);
    return slot;
}

static void
gpuReScanForeignScan(ForeignScanState *node)
{
    GpuScanState *st = (GpuScanState *) node->fdw_state;
    if (st)
        st->cur = 0;
}

static void
gpuEndForeignScan(ForeignScanState *node)
{
    GpuScanState *st = (GpuScanState *) node->fdw_state;
    if (st)
    {
        if (st->keys) free(st->keys);
        if (st->vals) free(st->vals);
    }
}

/* ====================== modify callbacks =============================== */
static void
gpuAddForeignUpdateTargets(PlannerInfo *root, Index rtindex,
                           RangeTblEntry *target_rte, Relation target_relation)
{
    /* Row identity = the key column (attno 1), exposed as a resjunk "gpu_key". */
    Var *var = makeVar(rtindex, 1, INT8OID, -1, InvalidOid, 0);
    add_row_identity_var(root, var, rtindex, "gpu_key");
}

static List *
gpuPlanForeignModify(PlannerInfo *root, ModifyTable *plan,
                     Index resultRelation, int subplan_index)
{
    return NIL;
}

static void
gpuBeginForeignModify(ModifyTableState *mtstate, ResultRelInfo *rinfo,
                      List *fdw_private, int subplan_index, int eflags)
{
    GpuModifyState *st = (GpuModifyState *) palloc0(sizeof(GpuModifyState));
    CmdType         op = mtstate->operation;

    if (op == CMD_UPDATE || op == CMD_DELETE)
    {
        Plan *subplan = outerPlanState(mtstate)->plan;
        st->key_junk = ExecFindJunkAttributeInTlist(subplan->targetlist, "gpu_key");
        if (!AttributeNumberIsValid(st->key_junk))
            ereport(ERROR,
                    (errmsg("pg_gpu_fdw: could not find row-identity junk column")));
    }

    rinfo->ri_FdwState = (void *) st;
    (void) gpu_fdw_get_engine();
}

/* Pull the two int8 columns out of a result slot. */
static void
gpu_fdw_get_kv(TupleTableSlot *slot, int64 *k, int64 *v)
{
    bool isnull;
    Datum dk = slot_getattr(slot, 1, &isnull);
    *k = isnull ? 0 : DatumGetInt64(dk);
    if (slot->tts_tupleDescriptor->natts > 1)
    {
        Datum dv = slot_getattr(slot, 2, &isnull);
        *v = isnull ? 0 : DatumGetInt64(dv);
    }
    else
        *v = 0;
}

static TupleTableSlot *
gpuExecForeignInsert(EState *estate, ResultRelInfo *rinfo,
                     TupleTableSlot *slot, TupleTableSlot *planSlot)
{
    int64 k, v;
    gpu_fdw_get_kv(slot, &k, &v);
    gpu_oltp_insert(gpu_fdw_get_engine(), (uint64_t) k, (uint64_t) v);
    return slot;
}

static TupleTableSlot *
gpuExecForeignUpdate(EState *estate, ResultRelInfo *rinfo,
                     TupleTableSlot *slot, TupleTableSlot *planSlot)
{
    GpuModifyState *st = (GpuModifyState *) rinfo->ri_FdwState;
    GpuOltpEngine  *e = gpu_fdw_get_engine();
    bool   isnull;
    int64  old_k, new_k, new_v;

    old_k = DatumGetInt64(ExecGetJunkAttribute(planSlot, st->key_junk, &isnull));
    gpu_fdw_get_kv(slot, &new_k, &new_v);

    if (!isnull && old_k != new_k)
    {
        gpu_oltp_delete(e, (uint64_t) old_k);          /* key changed: move it */
        gpu_oltp_insert(e, (uint64_t) new_k, (uint64_t) new_v);
    }
    else
    {
        gpu_oltp_update(e, (uint64_t) new_k, (uint64_t) new_v);
    }
    return slot;
}

static TupleTableSlot *
gpuExecForeignDelete(EState *estate, ResultRelInfo *rinfo,
                     TupleTableSlot *slot, TupleTableSlot *planSlot)
{
    GpuModifyState *st = (GpuModifyState *) rinfo->ri_FdwState;
    bool   isnull;
    int64  old_k = DatumGetInt64(ExecGetJunkAttribute(planSlot, st->key_junk, &isnull));

    if (!isnull)
        gpu_oltp_delete(gpu_fdw_get_engine(), (uint64_t) old_k);
    return slot;
}

static void
gpuEndForeignModify(EState *estate, ResultRelInfo *rinfo)
{
    /* engine persists for the backend's lifetime; nothing to do here */
}

static int
gpuIsForeignRelUpdatable(Relation rel)
{
    return (1 << CMD_INSERT) | (1 << CMD_UPDATE) | (1 << CMD_DELETE);
}

/* ====================== handler / validator ============================ */
Datum
pg_gpu_fdw_handler(PG_FUNCTION_ARGS)
{
    FdwRoutine *r = makeNode(FdwRoutine);

    /* scan */
    r->GetForeignRelSize = gpuGetForeignRelSize;
    r->GetForeignPaths = gpuGetForeignPaths;
    r->GetForeignPlan = gpuGetForeignPlan;
    r->BeginForeignScan = gpuBeginForeignScan;
    r->IterateForeignScan = gpuIterateForeignScan;
    r->ReScanForeignScan = gpuReScanForeignScan;
    r->EndForeignScan = gpuEndForeignScan;

    /* modify */
    r->AddForeignUpdateTargets = gpuAddForeignUpdateTargets;
    r->PlanForeignModify = gpuPlanForeignModify;
    r->BeginForeignModify = gpuBeginForeignModify;
    r->ExecForeignInsert = gpuExecForeignInsert;
    r->ExecForeignUpdate = gpuExecForeignUpdate;
    r->ExecForeignDelete = gpuExecForeignDelete;
    r->EndForeignModify = gpuEndForeignModify;
    r->IsForeignRelUpdatable = gpuIsForeignRelUpdatable;

    PG_RETURN_POINTER(r);
}

Datum
pg_gpu_fdw_validator(PG_FUNCTION_ARGS)
{
    /* Accept any options for now. */
    PG_RETURN_VOID();
}

/* ============== bandwidth-scan SQL functions ========================== */
/* gpu_scan_load(n) : allocate + fill an n-element int8 column in GPU HBM. */
Datum
gpu_scan_load(PG_FUNCTION_ARGS)
{
    int64 n = PG_GETARG_INT64(0);
    static bool atexit_registered = false;

    if (n <= 0)
        ereport(ERROR, (errmsg("pg_gpu_fdw: scan size must be positive")));
    if (g_scan)
    {
        gpu_scan_free(g_scan);
        g_scan = NULL;
    }
    g_scan = gpu_scan_alloc((uint64_t) n);
    if (!atexit_registered)
    {
        on_proc_exit(gpu_fdw_atexit, (Datum) 0);
        atexit_registered = true;
    }
    PG_RETURN_INT64((int64) gpu_scan_count(g_scan));
}

/* gpu_sum() : GPU reduction sum(v) over the resident column. */
Datum
gpu_scan_agg_sum(PG_FUNCTION_ARGS)
{
    if (!g_scan)
        ereport(ERROR, (errmsg("pg_gpu_fdw: no resident column; call gpu_load(n) first")));
    PG_RETURN_INT64((int64) gpu_scan_sum(g_scan));
}

/* gpu_count_lt(thr) : GPU reduction count(v < thr) over the resident column. */
Datum
gpu_scan_agg_count_lt(PG_FUNCTION_ARGS)
{
    int64 thr = PG_GETARG_INT64(0);
    if (!g_scan)
        ereport(ERROR, (errmsg("pg_gpu_fdw: no resident column; call gpu_load(n) first")));
    PG_RETURN_INT64((int64) gpu_scan_count_lt(g_scan, (uint64_t) thr));
}

/* gpu_kernel_ms() : wall time of the last aggregate kernel (for GB/s calc). */
Datum
gpu_scan_kernel_ms(PG_FUNCTION_ARGS)
{
    PG_RETURN_FLOAT8(gpu_scan_last_ms());
}
