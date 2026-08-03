/* pg_rgi_fdw.c — writeable Postgres FDW backed by the RobustGPUIndexing engine.
 *
 * A foreign table kv(k bigint, v bigint) maps to an RGI GPUChainHashtable served
 * on the GPU. Writes are BUFFERED and flushed as one batched RGI launch in
 * EndForeignModify -> so a multi-row statement (INSERT .. SELECT) becomes one
 * warp-cooperative batch (high throughput), while single-row autocommit
 * statements are batch-of-1 (low latency, low throughput). That contrast is the
 * latency/throughput tradeoff.
 *
 *   INSERT -> rgi_insert (buffered) ; flushed at end of statement
 *   SELECT -> rgi_snapshot (batched find of all live keys on GPU)
 *   UPDATE -> rgi_update (+ delete/insert if key changes)
 *   DELETE -> rgi_delete
 *
 * Built for PostgreSQL 14. One engine per backend (session-local data).
 */
#include "postgres.h"

#include "access/htup_details.h"
#include "access/xact.h"
#include "catalog/pg_type.h"
#include "executor/executor.h"
#include "foreign/fdwapi.h"
#include "foreign/foreign.h"
#include "nodes/makefuncs.h"
#include "optimizer/appendinfo.h"
#include "optimizer/clauses.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "storage/ipc.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/errcodes.h"
#include "utils/hsearch.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "rgi_oltp_engine.h"
#include "pg_gpu_service.h"   /* dispatch to the shared GPU-service worker */

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(pg_rgi_fdw_handler);
PG_FUNCTION_INFO_V1(pg_rgi_fdw_validator);

/* ----- scan / modify state -------------------------------------------- */
typedef struct RgiScanState { uint64_t *keys; uint64_t *vals; uint64_t n, cur; } RgiScanState;
typedef struct RgiModifyState { AttrNumber key_junk; CmdType op; } RgiModifyState;

/* ===== transaction buffer (deferred apply, ROLLBACK, read-your-writes) =====
 * Writes in a transaction are buffered here and applied to the shared GPU index
 * atomically at PRE_COMMIT (discarded on ABORT). Reads overlay this buffer so a
 * txn sees its own uncommitted writes; other txns see only committed state (no
 * dirty reads). Write-write conflict detection (full serializability) is future
 * work. */
typedef struct TxnEntry
{
    int64 key;       /* hash key */
    int64 value;
    bool  deleted;   /* tombstone */
    bool  inserted;  /* created in this txn (PK-checked at commit) */
    bool  replace;   /* this key was deleted earlier in THIS txn, then re-created
                      * (delete-then-reinsert / rename-into) -> treat as an upsert
                      * at commit, NOT a fresh insert, so it must not PK-fail. */
    bool  kc_new;    /* this key is the NEW key of a key-changing UPDATE */
    bool  kc_old;    /* this key is the OLD key tombstoned by a key-changing UPDATE */
    bool  seen;      /* transient: matched a worker row during a scan */
} TxnEntry;

/* A key-collapsed per-key buffer cannot represent row identity when a single
 * statement renames keys that overlap (swaps like 1<->2, chains like k=k+1):
 * the new-key write and the old-key tombstone for two different rows land on the
 * same buffer entry and clobber each other. We detect that exact case (old/new
 * key sets overlap) and reject it rather than silently corrupt. Non-overlapping
 * multi-row renames (e.g. k=k+100) and single-row renames are fine. */

static HTAB *txn_buf = NULL;
static bool  xact_cb_registered = false;

static HTAB *
txn_get_buf(void)
{
    if (txn_buf == NULL)
    {
        HASHCTL ctl;
        memset(&ctl, 0, sizeof(ctl));
        ctl.keysize = sizeof(int64);
        ctl.entrysize = sizeof(TxnEntry);
        ctl.hcxt = TopTransactionContext;
        txn_buf = hash_create("rgi txn buffer", 1024, &ctl,
                              HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
    }
    return txn_buf;
}

/* Apply the buffered transaction to the shared index at PRE_COMMIT.
 *
 * The whole write set (deletes + updates + inserts) is staged in the worker,
 * then VALIDATED (PK/UNIQUE) before any mutation, then applied with operations
 * that have no expected failure path. So a duplicate key raises the error with
 * the GPU index left UNCHANGED — the commit is all-or-nothing, never partial. */
static void
rgi_txn_apply(void)
{
    HASH_SEQ_STATUS seq;
    TxnEntry *e;
    uint64_t *ins_k, *ins_v, *upd_k, *upd_v, *del_k;
    uint32_t  ni = 0, nu = 0, nd = 0, cnt;
    int       ok = 1;
    uint64_t  dup = 0;

    if (txn_buf == NULL) return;
    cnt = (uint32_t) hash_get_num_entries(txn_buf);
    if (cnt == 0) { txn_buf = NULL; return; }

    ins_k = palloc(sizeof(uint64_t) * cnt); ins_v = palloc(sizeof(uint64_t) * cnt);
    upd_k = palloc(sizeof(uint64_t) * cnt); upd_v = palloc(sizeof(uint64_t) * cnt);
    del_k = palloc(sizeof(uint64_t) * cnt);
    hash_seq_init(&seq, txn_buf);
    while ((e = (TxnEntry *) hash_seq_search(&seq)) != NULL)
    {
        if (e->deleted && e->inserted) continue;                 /* created+deleted in txn */
        else if (e->deleted)  del_k[nd++] = (uint64_t) e->key;
        else if (e->inserted && !e->replace)                     /* fresh key -> PK-checked */
                              { ins_k[ni] = (uint64_t) e->key; ins_v[ni] = (uint64_t) e->value; ni++; }
        else                  /* value update, or delete-then-reinsert -> upsert (no PK) */
                              { upd_k[nu] = (uint64_t) e->key; upd_v[nu] = (uint64_t) e->value; nu++; }
    }

    /* Stage + validate + apply atomically in the worker. */
    gpu_svc_txn_commit(del_k, nd, upd_k, upd_v, nu, ins_k, ins_v, ni, &ok, &dup);
    txn_buf = NULL;   /* HTAB freed with TopTransactionContext */

    if (!ok)
        ereport(ERROR,
                (errcode(ERRCODE_UNIQUE_VIOLATION),
                 errmsg("duplicate key value violates unique constraint on GPU table"),
                 errdetail("Key (k)=(%lld) already exists.", (long long) dup)));
}

static void
rgi_xact_cb(XactEvent event, void *arg)
{
    switch (event)
    {
        case XACT_EVENT_PRE_COMMIT:
            rgi_txn_apply();          /* a PK violation here aborts the commit */
            break;
        case XACT_EVENT_ABORT:
        case XACT_EVENT_PARALLEL_ABORT:
        case XACT_EVENT_COMMIT:
        case XACT_EVENT_PARALLEL_COMMIT:
            txn_buf = NULL;           /* buffer lived in TopTransactionContext */
            break;
        default:
            break;
    }
}

static void
rgi_ensure_xact_cb(void)
{
    if (!xact_cb_registered)
    {
        RegisterXactCallback(rgi_xact_cb, NULL);
        xact_cb_registered = true;
    }
}

/* ====================== scan callbacks ================================= */
static void
rgiGetForeignRelSize(PlannerInfo *root, RelOptInfo *baserel, Oid foreigntableid)
{
    baserel->rows = 1000;
    baserel->fdw_private = NULL;
}

static void
rgiGetForeignPaths(PlannerInfo *root, RelOptInfo *baserel, Oid foreigntableid)
{
    Cost startup = 1.0;
    add_path(baserel, (Path *)
             create_foreignscan_path(root, baserel, NULL, baserel->rows,
                                     startup, startup + baserel->rows,
                                     NIL, NULL, NULL, NIL));
}

/* Append an int8 key Const to *keys (helper). */
static void rgi_push_key(List **keys, int64 v)
{
    *keys = lappend(*keys, makeConst(INT8OID, -1, InvalidOid, sizeof(int64),
                                     Int64GetDatum(v), false, true));
}

/* Read an int2/int4/int8 Const into int64. Returns false for other types/NULL. */
static bool const_to_int64(Const *c, int64 *out)
{
    if (c->constisnull) return false;
    switch (c->consttype)
    {
        case INT2OID: *out = (int64) DatumGetInt16(c->constvalue); return true;
        case INT4OID: *out = (int64) DatumGetInt32(c->constvalue); return true;
        case INT8OID: *out =          DatumGetInt64(c->constvalue); return true;
        default:      return false;
    }
}

/* Is this opno the "=" operator? (the key column uses int84eq / int8eq etc.) */
static bool is_equal_op(Oid opno)
{
    char *name = get_opname(opno);
    bool  eq = (name && strcmp(name, "=") == 0);
    if (name) pfree(name);
    return eq;
}

/* Extract pushable keys from an equality clause on the key column (attno 1):
 *   k = <int const>            (OpExpr, "=")
 *   k = ANY(<int const array>) (ScalarArrayOpExpr, useOr, "=")
 * Integer literals may be int2/int4/int8 (cross-type ops keep them un-widened). */
static bool
rgi_extract_keys(Expr *clause, List **keys)
{
    if (IsA(clause, OpExpr))
    {
        OpExpr *op = (OpExpr *) clause;
        Node *a, *b; int64 v;
        if (list_length(op->args) != 2 || !is_equal_op(op->opno)) return false;
        a = (Node *) linitial(op->args); b = (Node *) lsecond(op->args);
        if (IsA(b, Var) && IsA(a, Const)) { Node *t=a; a=b; b=t; }  /* normalize Var,Const */
        if (IsA(a, Var) && ((Var *) a)->varattno == 1 &&
            IsA(b, Const) && const_to_int64((Const *) b, &v))
        {
            rgi_push_key(keys, v);
            return true;
        }
        return false;
    }
    if (IsA(clause, ScalarArrayOpExpr))
    {
        ScalarArrayOpExpr *saop = (ScalarArrayOpExpr *) clause;
        Node *a, *arr; ArrayType *at; Oid et;
        int16 typlen; bool byval; char align;
        Datum *elems; bool *nulls; int nelems, i;
        if (!saop->useOr || list_length(saop->args) != 2 || !is_equal_op(saop->opno))
            return false;
        a = (Node *) linitial(saop->args); arr = (Node *) lsecond(saop->args);
        if (!(IsA(a, Var) && ((Var *) a)->varattno == 1 &&
              IsA(arr, Const) && !((Const *) arr)->constisnull))
            return false;
        at = DatumGetArrayTypeP(((Const *) arr)->constvalue);
        et = ARR_ELEMTYPE(at);
        if (et != INT2OID && et != INT4OID && et != INT8OID) return false;
        get_typlenbyvalalign(et, &typlen, &byval, &align);
        deconstruct_array(at, et, typlen, byval, align, &elems, &nulls, &nelems);
        for (i = 0; i < nelems; i++)
        {
            if (nulls[i]) continue;
            if (et == INT2OID)      rgi_push_key(keys, (int64) DatumGetInt16(elems[i]));
            else if (et == INT4OID) rgi_push_key(keys, (int64) DatumGetInt32(elems[i]));
            else                    rgi_push_key(keys,          DatumGetInt64(elems[i]));
        }
        return true;
    }
    return false;
}

static ForeignScan *
rgiGetForeignPlan(PlannerInfo *root, RelOptInfo *baserel, Oid foreigntableid,
                  ForeignPath *best_path, List *tlist, List *scan_clauses, Plan *outer_plan)
{
    List     *pushed = NIL;        /* int8 Const list of keys to look up on GPU */
    ListCell *lc;

    foreach(lc, baserel->baserestrictinfo)
    {
        RestrictInfo *ri = lfirst_node(RestrictInfo, lc);
        (void) rgi_extract_keys(ri->clause, &pushed);  /* best-effort; PG still rechecks */
    }

    scan_clauses = extract_actual_clauses(scan_clauses, false);
    /* keep scan_clauses as qpqual so Postgres rechecks; pushed keys ride fdw_private */
    return make_foreignscan(tlist, scan_clauses, baserel->relid,
                            NIL, pushed, NIL, NIL, outer_plan);
}

/* Runtime pushdown: if a qual is k = ANY(<array expr>) where the array is a
 * Param / subplan / non-const (evaluated only at execution), evaluate it now
 * and return the keys. Handles the real multi-get case `k = ANY($1::bigint[])`.
 * On success allocates *out_keys (malloc) and sets *out_n; returns true. */
static bool
rgi_runtime_array_keys(ForeignScanState *node, uint64_t **out_keys, uint32_t *out_n)
{
    List     *qual = node->ss.ps.plan->qual;
    ListCell *lc;
    foreach(lc, qual)
    {
        Expr *clause = (Expr *) lfirst(lc);
        ScalarArrayOpExpr *saop;
        Node *a, *arrexpr;
        ExprState *es;
        ExprContext *econ;
        Datum d; bool isnull;
        ArrayType *at; Oid et; int16 typlen; bool byval; char align;
        Datum *elems; bool *nulls; int nelems, i;
        uint64_t *kk; uint32_t m = 0;

        if (!IsA(clause, ScalarArrayOpExpr)) continue;
        saop = (ScalarArrayOpExpr *) clause;
        if (!saop->useOr || list_length(saop->args) != 2 || !is_equal_op(saop->opno)) continue;
        a = (Node *) linitial(saop->args); arrexpr = (Node *) lsecond(saop->args);
        if (!(IsA(a, Var) && ((Var *) a)->varattno == 1)) continue;
        /* Only evaluate EXTERNAL params ($1) standalone here — the real multi-get
         * case. InitPlan/PARAM_EXEC, SubPlans, etc. depend on executor state that
         * isn't ready in BeginForeignScan, so fall back to snapshot for those. */
        if (!(IsA(arrexpr, Param) && ((Param *) arrexpr)->paramkind == PARAM_EXTERN))
            continue;

        econ = node->ss.ps.ps_ExprContext;
        es = ExecInitExpr((Expr *) arrexpr, &node->ss.ps);
        d = ExecEvalExpr(es, econ, &isnull);
        if (isnull) continue;
        at = DatumGetArrayTypeP(d);
        et = ARR_ELEMTYPE(at);
        if (et != INT2OID && et != INT4OID && et != INT8OID) continue;
        get_typlenbyvalalign(et, &typlen, &byval, &align);
        deconstruct_array(at, et, typlen, byval, align, &elems, &nulls, &nelems);
        if (nelems <= 0) continue;
        kk = (uint64_t *) malloc(sizeof(uint64_t) * nelems);
        for (i = 0; i < nelems; i++)
        {
            if (nulls[i]) continue;
            if (et == INT2OID)      kk[m++] = (uint64_t) DatumGetInt16(elems[i]);
            else if (et == INT4OID) kk[m++] = (uint64_t) DatumGetInt32(elems[i]);
            else                    kk[m++] = (uint64_t) DatumGetInt64(elems[i]);
        }
        *out_keys = kk; *out_n = m;
        return true;
    }
    return false;
}

static void
rgiBeginForeignScan(ForeignScanState *node, int eflags)
{
    RgiScanState *st;
    ForeignScan  *fsplan = (ForeignScan *) node->ss.ps.plan;
    List         *pushed = fsplan->fdw_private;
    uint64_t     *kk = NULL;
    uint32_t      n = 0;
    bool          have_keys = false;

    if (eflags & EXEC_FLAG_EXPLAIN_ONLY) return;
    st = (RgiScanState *) palloc0(sizeof(RgiScanState));

    if (pushed != NIL)                          /* plan-time const keys */
    {
        int i = 0; ListCell *lc;
        n = (uint32_t) list_length(pushed);
        kk = (uint64_t *) malloc(sizeof(uint64_t) * n);
        foreach(lc, pushed)
            kk[i++] = (uint64_t) DatumGetInt64(((Const *) lfirst(lc))->constvalue);
        have_keys = true;
    }
    else                                        /* runtime k = ANY(param/subplan) */
    {
        have_keys = rgi_runtime_array_keys(node, &kk, &n);
    }

    if (have_keys)
    {
        /* point lookups: read-your-writes overlay, then worker for the rest */
        uint32_t  i, c, nq = 0;
        uint64_t  m = 0;
        uint64_t *res_k = (uint64_t *) palloc(sizeof(uint64_t) * (n ? n : 1));
        uint64_t *res_v = (uint64_t *) palloc(sizeof(uint64_t) * (n ? n : 1));
        uint64_t *wq    = (uint64_t *) palloc(sizeof(uint64_t) * (n ? n : 1));
        for (i = 0; i < n; i++)
        {
            TxnEntry *e = NULL;
            if (txn_buf) { int64 kkey = (int64) kk[i]; e = (TxnEntry *) hash_search(txn_buf, &kkey, HASH_FIND, NULL); }
            if (e) { if (!e->deleted) { res_k[m] = kk[i]; res_v[m] = (uint64_t) e->value; m++; } }
            else   { wq[nq++] = kk[i]; }
        }
        if (nq)
        {
            uint64_t *fk = (uint64_t *) palloc(sizeof(uint64_t) * nq);
            uint64_t *fv = (uint64_t *) palloc(sizeof(uint64_t) * nq);
            c = nq;
            gpu_svc_bulk(SVC_FIND_MANY, wq, NULL, &c, fk, fv, NULL, NULL);
            for (i = 0; i < c; i++) { res_k[m] = fk[i]; res_v[m] = fv[i]; m++; }
        }
        if (kk) free(kk);
        st->keys = res_k; st->vals = res_v; st->n = m;
    }
    else
    {
        /* full-table snapshot from the worker (paged, no truncation), overlaid
         * with the txn buffer */
        uint32_t  c = 0, i;
        uint64_t  m = 0;
        uint64_t  nbuf = txn_buf ? (uint64_t) hash_get_num_entries(txn_buf) : 0;
        uint64_t *ok = NULL, *ov = NULL;
        uint64_t  cap2;
        uint64_t *res_k, *res_v;
        HASH_SEQ_STATUS sq; TxnEntry *e;
        gpu_svc_snapshot_all((uint64 **) &ok, (uint64 **) &ov, &c);
        cap2 = (uint64_t) c + nbuf;
        res_k = (uint64_t *) palloc(sizeof(uint64_t) * (cap2 ? cap2 : 1));
        res_v = (uint64_t *) palloc(sizeof(uint64_t) * (cap2 ? cap2 : 1));
        if (txn_buf) { hash_seq_init(&sq, txn_buf); while ((e = hash_seq_search(&sq)) != NULL) e->seen = false; }
        for (i = 0; i < c; i++)
        {
            TxnEntry *te = NULL;
            if (txn_buf) { int64 kk2 = (int64) ok[i]; te = (TxnEntry *) hash_search(txn_buf, &kk2, HASH_FIND, NULL); }
            if (te) { if (!te->deleted) { res_k[m] = ok[i]; res_v[m] = (uint64_t) te->value; te->seen = true; m++; } }
            else    { res_k[m] = ok[i]; res_v[m] = ov[i]; m++; }
        }
        if (txn_buf) { hash_seq_init(&sq, txn_buf);
            while ((e = hash_seq_search(&sq)) != NULL)
                if (!e->deleted && !e->seen) { res_k[m] = (uint64_t) e->key; res_v[m] = (uint64_t) e->value; m++; } }
        st->keys = res_k; st->vals = res_v; st->n = m;
    }
    st->cur = 0;
    node->fdw_state = (void *) st;
}

static TupleTableSlot *
rgiIterateForeignScan(ForeignScanState *node)
{
    RgiScanState   *st = (RgiScanState *) node->fdw_state;
    TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
    ExecClearTuple(slot);
    if (st == NULL || st->cur >= st->n) return slot;
    slot->tts_values[0] = Int64GetDatum((int64) st->keys[st->cur]);
    slot->tts_isnull[0] = false;
    if (slot->tts_tupleDescriptor->natts > 1) {
        slot->tts_values[1] = Int64GetDatum((int64) st->vals[st->cur]);
        slot->tts_isnull[1] = false;
    }
    st->cur++;
    ExecStoreVirtualTuple(slot);
    return slot;
}

static void rgiReScanForeignScan(ForeignScanState *node)
{
    RgiScanState *st = (RgiScanState *) node->fdw_state;
    if (st) st->cur = 0;
}
static void rgiEndForeignScan(ForeignScanState *node)
{
    /* scan result arrays are palloc'd in the executor context; nothing to free */
}

/* ====================== modify callbacks =============================== */
static void
rgiAddForeignUpdateTargets(PlannerInfo *root, Index rtindex,
                           RangeTblEntry *target_rte, Relation target_relation)
{
    Var *var = makeVar(rtindex, 1, INT8OID, -1, InvalidOid, 0);
    add_row_identity_var(root, var, rtindex, "rgi_key");
}

static List *
rgiPlanForeignModify(PlannerInfo *root, ModifyTable *plan, Index resultRelation, int subplan_index)
{
    return NIL;
}

static void
rgiBeginForeignModify(ModifyTableState *mtstate, ResultRelInfo *rinfo,
                      List *fdw_private, int subplan_index, int eflags)
{
    RgiModifyState *st = (RgiModifyState *) palloc0(sizeof(RgiModifyState));
    CmdType op = mtstate->operation;
    st->op = op;
    if (op == CMD_UPDATE || op == CMD_DELETE) {
        Plan *subplan = outerPlanState(mtstate)->plan;
        st->key_junk = ExecFindJunkAttributeInTlist(subplan->targetlist, "rgi_key");
        if (!AttributeNumberIsValid(st->key_junk))
            ereport(ERROR, (errmsg("pg_rgi_fdw: missing row-identity junk column")));
    }
    rgi_ensure_xact_cb();
    rinfo->ri_FdwState = (void *) st;
}

static void
rgi_get_kv(TupleTableSlot *slot, int64 *k, int64 *v)
{
    bool isnull;
    Datum dk = slot_getattr(slot, 1, &isnull);
    *k = isnull ? 0 : DatumGetInt64(dk);
    if (slot->tts_tupleDescriptor->natts > 1) {
        Datum dv = slot_getattr(slot, 2, &isnull);
        *v = isnull ? 0 : DatumGetInt64(dv);
    } else *v = 0;
}

static TupleTableSlot *
rgiExecForeignInsert(EState *estate, ResultRelInfo *rinfo, TupleTableSlot *slot, TupleTableSlot *planSlot)
{
    HTAB *b = txn_get_buf();
    bool found; TxnEntry *e;
    int64 k, v, kk;
    rgi_get_kv(slot, &k, &v);
    kk = k;
    e = (TxnEntry *) hash_search(b, &kk, HASH_ENTER, &found);
    if (found && !e->deleted)                       /* live row already in this txn */
        ereport(ERROR,
                (errcode(ERRCODE_UNIQUE_VIOLATION),
                 errmsg("duplicate key value violates unique constraint on GPU table"),
                 errdetail("Key (k)=(%lld) already exists.", (long long) k)));
    /* If the key was deleted earlier in this txn, re-inserting it is a legal
     * replace (upsert at commit), not a fresh insert that must PK-fail. */
    e->replace = (found && e->deleted);
    if (!found) { e->kc_new = false; e->kc_old = false; }
    e->value = v; e->deleted = false; e->inserted = true; e->seen = false;
    return slot;
}

static TupleTableSlot *
rgiExecForeignUpdate(EState *estate, ResultRelInfo *rinfo, TupleTableSlot *slot, TupleTableSlot *planSlot)
{
    RgiModifyState *st = (RgiModifyState *) rinfo->ri_FdwState;
    HTAB *b = txn_get_buf();
    bool isnull, found; TxnEntry *e;
    int64 old_k = DatumGetInt64(ExecGetJunkAttribute(planSlot, st->key_junk, &isnull));
    int64 new_k, new_v, nk;
    rgi_get_kv(slot, &new_k, &new_v);
    nk = new_k;

    if (isnull || old_k == new_k)
    {
        /* value-only update of an existing row: no PK check needed */
        e = (TxnEntry *) hash_search(b, &nk, HASH_ENTER, &found);
        if (!found) { e->inserted = false; e->replace = false; e->kc_new = false; e->kc_old = false; }
        e->value = new_v; e->deleted = false; e->seen = false;
    }
    else
    {
        /* KEY-CHANGING update: the NEW key appears and is validated like an
         * INSERT (it can collide with another row); the OLD key is tombstoned.
         * Reject swaps/chains (overlapping old/new keys) the collapsed buffer
         * cannot represent, instead of corrupting. */
        bool found_o; TxnEntry *eo; int64 ok2 = old_k;

        e = (TxnEntry *) hash_search(b, &nk, HASH_ENTER, &found);
        if (found && e->kc_old)                    /* new key is another row's tombstoned old key */
            ereport(ERROR,
                    (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                     errmsg("key-changing UPDATE with key swaps/chains is not supported on GPU foreign tables"),
                     errdetail("Key (k)=(%lld) is both renamed away and renamed into within one transaction.",
                               (long long) new_k)));
        if (found && !e->deleted)                  /* new key already live in this txn */
            ereport(ERROR,
                    (errcode(ERRCODE_UNIQUE_VIOLATION),
                     errmsg("duplicate key value violates unique constraint on GPU table"),
                     errdetail("Key (k)=(%lld) already exists.", (long long) new_k)));
        e->replace = (found && e->deleted);        /* renamed into a key freed earlier in txn */
        if (!found) { e->inserted = true; e->kc_old = false; }  /* brand-new key -> PK-checked at commit */
        e->kc_new = true;
        e->value = new_v; e->deleted = false; e->seen = false;

        eo = (TxnEntry *) hash_search(b, &ok2, HASH_ENTER, &found_o);
        if (found_o && eo->kc_new)                 /* old key is another row's new key */
            ereport(ERROR,
                    (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                     errmsg("key-changing UPDATE with key swaps/chains is not supported on GPU foreign tables"),
                     errdetail("Key (k)=(%lld) is both renamed into and renamed away within one transaction.",
                               (long long) old_k)));
        if (!found_o) { eo->inserted = false; eo->replace = false; eo->kc_new = false; }
        eo->kc_old = true;
        eo->deleted = true; eo->value = 0; eo->seen = false;
    }
    return slot;
}

static TupleTableSlot *
rgiExecForeignDelete(EState *estate, ResultRelInfo *rinfo, TupleTableSlot *slot, TupleTableSlot *planSlot)
{
    RgiModifyState *st = (RgiModifyState *) rinfo->ri_FdwState;
    HTAB *b = txn_get_buf();
    bool isnull, found; TxnEntry *e;
    int64 old_k = DatumGetInt64(ExecGetJunkAttribute(planSlot, st->key_junk, &isnull));
    int64 kk;
    if (isnull) return slot;
    kk = old_k;
    e = (TxnEntry *) hash_search(b, &kk, HASH_ENTER, &found);
    if (!found) { e->inserted = false; e->replace = false; e->kc_new = false; e->kc_old = false; }
    e->deleted = true; e->seen = false;
    return slot;
}

static void
rgiEndForeignModify(EState *estate, ResultRelInfo *rinfo)
{
    /* Writes are buffered per-transaction and applied atomically at COMMIT
     * (rgi_txn_apply via rgi_xact_cb). Nothing to do at statement end. */
}

static int
rgiIsForeignRelUpdatable(Relation rel)
{
    return (1 << CMD_INSERT) | (1 << CMD_UPDATE) | (1 << CMD_DELETE);
}

/* ====================== handler / validator ============================ */
Datum
pg_rgi_fdw_handler(PG_FUNCTION_ARGS)
{
    FdwRoutine *r = makeNode(FdwRoutine);
    r->GetForeignRelSize = rgiGetForeignRelSize;
    r->GetForeignPaths = rgiGetForeignPaths;
    r->GetForeignPlan = rgiGetForeignPlan;
    r->BeginForeignScan = rgiBeginForeignScan;
    r->IterateForeignScan = rgiIterateForeignScan;
    r->ReScanForeignScan = rgiReScanForeignScan;
    r->EndForeignScan = rgiEndForeignScan;
    r->AddForeignUpdateTargets = rgiAddForeignUpdateTargets;
    r->PlanForeignModify = rgiPlanForeignModify;
    r->BeginForeignModify = rgiBeginForeignModify;
    r->ExecForeignInsert = rgiExecForeignInsert;
    r->ExecForeignUpdate = rgiExecForeignUpdate;
    r->ExecForeignDelete = rgiExecForeignDelete;
    r->EndForeignModify = rgiEndForeignModify;
    r->IsForeignRelUpdatable = rgiIsForeignRelUpdatable;
    PG_RETURN_POINTER(r);
}

Datum
pg_rgi_fdw_validator(PG_FUNCTION_ARGS)
{
    PG_RETURN_VOID();
}
