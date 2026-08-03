/* gpu_oltp_engine.h — extern "C" host API for the GPU OLTP engine.
 *
 * This is the linking boundary between the CUDA engine (gpu_oltp_engine.cu,
 * built as libgpuoltp.so) and consumers such as the Postgres FDW (pg_gpu_fdw)
 * or the standalone microbench.
 *
 * Design: a single persistent kernel stays resident on the GPU and consumes
 * request batches from a mapped (zero-copy) ring. The host submits a batch and
 * blocks until the kernel signals completion. On PCIe (RTX 4060) the handshake
 * rides mapped pinned memory; on GB-class the same API can sit on coherent C2C
 * memory with a doorbell over C2C atomics.
 */
#ifndef GPU_OLTP_ENGINE_H
#define GPU_OLTP_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Request opcodes. Numeric order matters: see store classification in the .cu. */
enum gpu_oltp_op {
    GPU_OLTP_LOOKUP = 0,
    GPU_OLTP_INSERT = 1,
    GPU_OLTP_UPDATE = 2,
    GPU_OLTP_DELETE = 3
};

/* Locking scheme for write ops (runtime-selectable for the locking study). */
enum gpu_oltp_scheme {
    GPU_OLTP_LOCKFREE   = 0,
    GPU_OLTP_BUCKETLOCK = 1,
    GPU_OLTP_GLOBALLOCK = 2
};

/* Per-request result status. */
enum gpu_oltp_status {
    GPU_OLTP_OK        = 0,  /* found (lookup) or success (write)      */
    GPU_OLTP_NOTFOUND  = 1,  /* key not present                        */
    GPU_OLTP_FULL      = 2   /* table full (insert)                    */
};

typedef struct GpuOltpEngine GpuOltpEngine;

/* Create/destroy. capacity is rounded up to a power of two. */
GpuOltpEngine *gpu_oltp_create(uint64_t capacity, int threads_per_block);
void           gpu_oltp_destroy(GpuOltpEngine *e);

/* Select the locking scheme used for subsequent write ops. */
void gpu_oltp_set_scheme(GpuOltpEngine *e, int scheme);

/* Submit a batch of n operations and block until complete.
 * types/keys/values are inputs of length n.
 * out_status (length n) and out_values (length n) receive results; either may
 * be NULL if the caller does not need them. */
void gpu_oltp_submit(GpuOltpEngine *e,
                     const uint32_t *types,
                     const uint64_t *keys,
                     const uint64_t *values,
                     uint32_t       *out_status,
                     uint64_t       *out_values,
                     uint32_t        n);

/* Convenience single-op wrappers (each is a batch of 1). */
int gpu_oltp_insert(GpuOltpEngine *e, uint64_t key, uint64_t value);
int gpu_oltp_lookup(GpuOltpEngine *e, uint64_t key, uint64_t *out_value);
int gpu_oltp_update(GpuOltpEngine *e, uint64_t key, uint64_t value);
int gpu_oltp_delete(GpuOltpEngine *e, uint64_t key);

/* Snapshot all live (key,value) pairs for a full-table scan. Copies the
 * GPU-resident table to the host and compacts out empty/tombstone slots.
 * Returns the number of live entries; *out_keys and *out_values (if non-NULL)
 * are malloc'd to that length and must be freed by the caller with free().
 * Intended for single-session use: call when no write batch is in flight. */
uint64_t gpu_oltp_snapshot(GpuOltpEngine *e, uint64_t **out_keys, uint64_t **out_values);

/* ---- Bandwidth-bound scan/aggregate (OLAP) over a GPU-resident column ----
 * Models the "data already in HBM, CPU only issues the query" case: a large
 * column lives in GPU memory; aggregates run on-GPU (streaming HBM) and return
 * only a scalar over PCIe. These are independent of the persistent-kernel
 * engine (no doorbell), so they use ordinary launch+sync. */
typedef struct GpuScanCol GpuScanCol;
GpuScanCol *gpu_scan_alloc(uint64_t n);                  /* HBM column v[i]=i+1     */
uint64_t    gpu_scan_count(GpuScanCol *s);               /* element count           */
uint64_t    gpu_scan_sum(GpuScanCol *s);                 /* GPU reduction: sum(v)   */
uint64_t    gpu_scan_count_lt(GpuScanCol *s, uint64_t threshold); /* count(v<thr)   */
double      gpu_scan_last_ms(void);                      /* last kernel time (ms)   */
void        gpu_scan_free(GpuScanCol *s);

#ifdef __cplusplus
}
#endif

#endif /* GPU_OLTP_ENGINE_H */
