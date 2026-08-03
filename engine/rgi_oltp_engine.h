/* rgi_oltp_engine.h — plain C ABI for the RobustGPUIndexing-backed OLTP engine.
 *
 * This is the linking boundary between the RGI CUDA wrapper (rgi_oltp_engine.cu,
 * built as librgioltp.so) and the Postgres FDW (pg_rgi_fdw), which is compiled
 * as plain C and never sees RGI's CUDA templates.
 *
 * Storage: RGI GPUChainHashtable (warp-cooperative, concurrent, reclaiming).
 * KV mapping: 8-byte key -> two uint32 RGI key-slices; value -> uint32 row id.
 *
 * Batching: inserts/updates are buffered host-side and flushed as ONE RGI
 * batch_kernel launch (rgi_flush, or implicitly before a read). This is what
 * realizes the GPU's throughput; the batch size is the latency/throughput knob.
 */
#ifndef RGI_OLTP_ENGINE_H
#define RGI_OLTP_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RgiEngine RgiEngine;

/* capacity = expected #keys (sizes the table); fill_factor/pool_ratio tune RGI. */
RgiEngine *rgi_create(uint32_t capacity, float fill_factor, float pool_ratio);
void       rgi_destroy(RgiEngine *e);

/* Buffered write ops (no GPU work until a flush / read). value truncated to 32b. */
void rgi_insert(RgiEngine *e, uint64_t key, uint64_t value);
void rgi_update(RgiEngine *e, uint64_t key, uint64_t value);
void rgi_delete(RgiEngine *e, uint64_t key);

/* Flush buffered writes to the GPU index (one batched launch per chunk). */
void rgi_flush(RgiEngine *e);

/* Flush buffered INSERTs with UNIQUE/PK enforcement. Checks (a) duplicate keys
 * within the batch and (b) keys already present in the index, BEFORE applying.
 * If a duplicate is found, nothing is applied, *dup_key is set, and 1 is
 * returned. On success the batch is applied and 0 is returned. */
int  rgi_flush_unique(RgiEngine *e, uint64_t *dup_key);

/* Point lookup (flushes first). Returns 1 if found (and sets *out_value), else 0. */
int  rgi_lookup(RgiEngine *e, uint64_t key, uint64_t *out_value);

/* Batched point lookup (flushes first): for each i, found[i]=1/0 and
 * out_values[i]=value when found. One GPU find launch (chunked). Used for
 * SQL qual pushdown (k = const, k = ANY(array)). */
void rgi_find_many(RgiEngine *e, const uint64_t *keys, uint64_t *out_values,
                   int *found, uint32_t n);

/* Full snapshot for SELECT *: flushes, batched-finds all live keys on the GPU.
 * Returns #rows; out_keys / out_values are malloc'd (caller frees with free()). */
uint64_t rgi_snapshot(RgiEngine *e, uint64_t **out_keys, uint64_t **out_values);

/* ===== Atomic transaction commit (validate-then-apply) =====================
 * A commit is staged in full, then validated, then applied. Atomicity comes
 * from validating ALL error conditions (PK/UNIQUE) BEFORE any mutation, then
 * applying with operations that have no expected failure path (erase is a
 * no-op when absent; insert-with-update_if_exists never fails). It is NOT a
 * property of "one launch"; partial application is impossible because apply
 * cannot raise an expected error after validation succeeds.
 *
 * Usage: rgi_stage_begin; rgi_stage_{del,upd,ins}* (any order, chunked);
 *        rgi_stage_commit (validates inserts, then applies del+upd+ins) OR
 *        rgi_stage_abort (drops the staged set, index untouched). */
void rgi_stage_begin(RgiEngine *e);
void rgi_stage_del(RgiEngine *e, const uint64_t *keys, uint32_t n);
void rgi_stage_upd(RgiEngine *e, const uint64_t *keys, const uint64_t *vals, uint32_t n);
void rgi_stage_ins(RgiEngine *e, const uint64_t *keys, const uint64_t *vals, uint32_t n);
/* Validate staged INSERTs (intra-set dup + already-present), applying nothing
 * on conflict; on success apply del+upd+ins. Returns 1 (and sets *dup_key) on a
 * PK/UNIQUE violation with the index left UNCHANGED; returns 0 on success.
 * Clears the staged set either way. */
int  rgi_stage_commit(RgiEngine *e, uint64_t *dup_key);
void rgi_stage_abort(RgiEngine *e);

/* ===== Paged snapshot (no silent truncation) ===============================
 * rgi_snapshot_begin flushes and freezes the live-key set, returning the total
 * row count. rgi_snapshot_page emits up to `max` rows of that frozen set
 * starting at `off` (off advances by the SPAN = min(max, total-off), which the
 * caller controls); it returns the number of live rows actually written (<=
 * span, since keys erased after freezing are skipped). */
uint64_t rgi_snapshot_begin(RgiEngine *e);
uint32_t rgi_snapshot_page(RgiEngine *e, uint64_t off, uint32_t max,
                           uint64_t *out_keys, uint64_t *out_vals);

#ifdef __cplusplus
}
#endif

#endif /* RGI_OLTP_ENGINE_H */
