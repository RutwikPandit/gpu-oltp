/* pg_gpu_service.h — client interface to the GPU OLTP background worker. */
#ifndef PG_GPU_SERVICE_H
#define PG_GPU_SERVICE_H

#include "postgres.h"

/* request opcodes */
enum {
    SVC_LOOKUP = 0, SVC_INSERT = 1, SVC_UPDATE = 2, SVC_DELETE = 3,   /* single-row */
    SVC_SNAPSHOT = 4, SVC_FIND_MANY = 5,                              /* bulk read  */
    SVC_INSERT_MANY = 6, SVC_UPDATE_MANY = 7, SVC_DELETE_MANY = 8,    /* bulk write (legacy) */
    SVC_SNAPSHOT_NEXT = 9,                                            /* next snapshot page */
    SVC_TXN_BEGIN = 10, SVC_TXN_STAGE_DEL = 11, SVC_TXN_STAGE_UPD = 12,
    SVC_TXN_STAGE_INS = 13, SVC_TXN_COMMIT = 14, SVC_TXN_ABORT = 15   /* atomic commit */
};

/* Max rows per bulk request/response (bounded shared result buffer). */
#define GPU_SVC_BULK_CAP 262144

/* true if the worker is running and reachable */
extern bool gpu_svc_available(void);

/* Single-row op (used by the gpu_svc_insert/lookup test functions).
 * Returns found (1/0); *out_value set for lookups. */
extern int gpu_svc_dispatch(int op, uint64 key, uint64 value, uint64 *out_value);

/* Bulk op via the shared region (one at a time, lock-guarded):
 *  - in_keys/in_vals (length *count) are the inputs for INSERT/UPDATE/DELETE/FIND_MANY
 *    (in_vals may be NULL for FIND_MANY/DELETE_MANY/SNAPSHOT).
 *  - on return *count = number of result rows in out_keys/out_vals
 *    (SNAPSHOT and FIND_MANY); out_keys/out_vals may be NULL if not needed.
 *  - *ok = 0 with *dup_key set if an INSERT_MANY hits a PK/UNIQUE violation. */
extern void gpu_svc_bulk(int op,
                         const uint64 *in_keys, const uint64 *in_vals,
                         uint32 *count,
                         uint64 *out_keys, uint64 *out_vals,
                         int *ok, uint64 *dup_key);

/* Atomic transaction commit: stages the entire write set (del + upd + ins) into
 * the worker under ONE lock hold, then validates all PK/UNIQUE conditions before
 * any mutation and applies only if validation passes. On a PK/UNIQUE violation
 * *ok is set to 0 and *dup_key to the offending key, and the GPU index is left
 * UNCHANGED (no partial application). Any of the arrays may be NULL when its
 * count is 0. */
extern void gpu_svc_txn_commit(const uint64 *del_k, uint32 nd,
                               const uint64 *upd_k, const uint64 *upd_v, uint32 nu,
                               const uint64 *ins_k, const uint64 *ins_v, uint32 ni,
                               int *ok, uint64 *dup_key);

/* Full snapshot with NO truncation: pages through the frozen live set under one
 * lock hold and returns the complete result in out_keys / out_vals (palloc'd in
 * the caller's context, grown as needed) and *count rows. */
extern void gpu_svc_snapshot_all(uint64 **out_keys, uint64 **out_vals, uint32 *count);

#endif /* PG_GPU_SERVICE_H */
