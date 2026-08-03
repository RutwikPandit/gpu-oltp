/* gpu_oltp_engine.cu — GPU-resident OLTP engine.
 *
 *   - Open-addressing (linear-probe) hash index, GPU-resident.
 *   - Persistent kernel: launched once, stays resident, consumes request
 *     batches from a mapped (zero-copy) ring. No per-op kernel launch.
 *   - CPU<->GPU handshake over mapped pinned memory (PCIe on RTX 4060).
 *     The same protocol drops onto coherent C2C memory on GB-class.
 *   - Three write-locking schemes (lock-free / bucket-lock / global-lock) for
 *     the contention study.
 *
 * Build:
 *   shared lib : nvcc -O3 -arch=sm_89 -Xcompiler -fPIC -shared -o libgpuoltp.so gpu_oltp_engine.cu
 *   microbench : nvcc -O3 -arch=sm_89 -DBUILD_BENCH -o oltp_bench gpu_oltp_engine.cu
 *
 * NOTE: scaffold. Compile, run the microbench, and validate before wiring the
 * FDW. Single-block persistent kernel (robust __syncthreads handshake); scale
 * to a multi-block cooperative-groups grid once correctness is confirmed.
 */
#include "gpu_oltp_engine.h"

#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define EMPTY_KEY     0xFFFFFFFFFFFFFFFFULL
#define TOMBSTONE_KEY 0xFFFFFFFFFFFFFFFEULL
#define MAX_BATCH     (1u << 20)   /* mapped ring capacity (requests/batch) */

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                 \
            abort();                                                           \
        }                                                                      \
    } while (0)

/* ---- mapped control block: the CPU<->GPU doorbell ---------------------- */
struct Control {
    volatile uint32_t batch_id;   /* host bumps to publish a new batch       */
    volatile uint32_t done_id;    /* kernel sets to batch_id when finished    */
    volatile uint32_t count;      /* #requests in the current batch           */
    volatile uint32_t scheme;     /* gpu_oltp_scheme for write ops            */
    volatile uint32_t stop;       /* host sets to 1 to retire the kernel      */
    uint32_t _pad[11];            /* keep fields off one cache line each      */
};

/* ---- device-side hash table view -------------------------------------- */
struct Table {
    uint64_t *keys;     /* device, length capacity, init EMPTY_KEY            */
    uint64_t *values;   /* device, length capacity                           */
    int      *locks;    /* device, length capacity (bucket-lock scheme)      */
    int      *glock;    /* device, length 1 (global-lock scheme)             */
    uint64_t  capacity; /* power of two                                      */
    uint64_t  mask;     /* capacity - 1                                      */
};

struct GpuOltpEngine {
    Table     tbl;
    Control  *h_ctrl,  *d_ctrl;   /* mapped: host ptr + device ptr           */
    uint32_t *h_types, *d_types;
    uint64_t *h_keys,  *d_keys;
    uint64_t *h_vals,  *d_vals;
    uint32_t *h_stat,  *d_stat;
    uint64_t *h_oval,  *d_oval;
    cudaStream_t stream;
    uint32_t  cur_batch;          /* host-side mirror of batch_id            */
    int       tpb;
};

/* ---------------------------- device helpers --------------------------- */
__device__ __forceinline__ uint64_t hash64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

/* Lock-free lookup: linear probe; EMPTY ends the chain, TOMBSTONE skips. */
__device__ uint32_t ht_lookup(const Table &t, uint64_t key, uint64_t *out) {
    uint64_t h = hash64(key) & t.mask;
    for (uint64_t i = 0; i < t.capacity; ++i) {
        uint64_t slot = (h + i) & t.mask;
        uint64_t cur  = ((volatile uint64_t *)t.keys)[slot];
        if (cur == key)       { *out = ((volatile uint64_t *)t.values)[slot]; return GPU_OLTP_OK; }
        if (cur == EMPTY_KEY) return GPU_OLTP_NOTFOUND;
    }
    return GPU_OLTP_NOTFOUND;
}

/* Lock-free insert/update: claim EMPTY via CAS; if key present, overwrite. */
__device__ uint32_t ht_insert_lockfree(const Table &t, uint64_t key, uint64_t val) {
    uint64_t h = hash64(key) & t.mask;
    for (uint64_t i = 0; i < t.capacity; ++i) {
        uint64_t slot = (h + i) & t.mask;
        uint64_t cur  = ((volatile uint64_t *)t.keys)[slot];
        if (cur == key) { t.values[slot] = val; __threadfence(); return GPU_OLTP_OK; }
        if (cur == EMPTY_KEY || cur == TOMBSTONE_KEY) {
            uint64_t prev = atomicCAS((unsigned long long *)&t.keys[slot],
                                      (unsigned long long)cur,
                                      (unsigned long long)key);
            if (prev == cur || prev == key) {
                t.values[slot] = val; __threadfence(); return GPU_OLTP_OK;
            }
            /* lost the race; re-evaluate this same slot */
            --i;
        }
    }
    return GPU_OLTP_FULL;
}

/* Spinlock helpers. NOTE the SIMT hazard: threads of one warp contending for
 * the same lock can stall the holder. Acceptable for the contention *study*;
 * lock-free / warp-cooperative is the production answer. */
__device__ __forceinline__ void lock_acquire(int *l)  { while (atomicCAS(l, 0, 1) != 0) { __nanosleep(64); } }
__device__ __forceinline__ void lock_release(int *l)  { __threadfence(); atomicExch(l, 0); }

__device__ uint32_t ht_insert_bucketlock(const Table &t, uint64_t key, uint64_t val) {
    uint64_t home = hash64(key) & t.mask;
    lock_acquire(&t.locks[home]);
    uint32_t rc = GPU_OLTP_FULL;
    for (uint64_t i = 0; i < t.capacity; ++i) {
        uint64_t slot = (home + i) & t.mask;
        uint64_t cur  = t.keys[slot];
        if (cur == key)                              { t.values[slot] = val; rc = GPU_OLTP_OK; break; }
        if (cur == EMPTY_KEY || cur == TOMBSTONE_KEY){ t.keys[slot] = key; t.values[slot] = val; rc = GPU_OLTP_OK; break; }
    }
    lock_release(&t.locks[home]);
    return rc;
}

__device__ uint32_t ht_insert_globallock(const Table &t, uint64_t key, uint64_t val) {
    lock_acquire(t.glock);
    uint32_t rc = GPU_OLTP_FULL;
    uint64_t h = hash64(key) & t.mask;
    for (uint64_t i = 0; i < t.capacity; ++i) {
        uint64_t slot = (h + i) & t.mask;
        uint64_t cur  = t.keys[slot];
        if (cur == key)                              { t.values[slot] = val; rc = GPU_OLTP_OK; break; }
        if (cur == EMPTY_KEY || cur == TOMBSTONE_KEY){ t.keys[slot] = key; t.values[slot] = val; rc = GPU_OLTP_OK; break; }
    }
    lock_release(t.glock);
    return rc;
}

__device__ uint32_t ht_delete(const Table &t, uint64_t key) {
    uint64_t h = hash64(key) & t.mask;
    for (uint64_t i = 0; i < t.capacity; ++i) {
        uint64_t slot = (h + i) & t.mask;
        uint64_t cur  = t.keys[slot];
        if (cur == key)       { atomicExch((unsigned long long *)&t.keys[slot],
                                           (unsigned long long)TOMBSTONE_KEY);
                                __threadfence(); return GPU_OLTP_OK; }
        if (cur == EMPTY_KEY) return GPU_OLTP_NOTFOUND;
    }
    return GPU_OLTP_NOTFOUND;
}

__device__ uint32_t do_write(const Table &t, uint32_t scheme,
                             uint64_t key, uint64_t val) {
    switch (scheme) {
        case GPU_OLTP_BUCKETLOCK: return ht_insert_bucketlock(t, key, val);
        case GPU_OLTP_GLOBALLOCK: return ht_insert_globallock(t, key, val);
        default:                  return ht_insert_lockfree(t, key, val);
    }
}

/* ---------------------------- persistent kernel ------------------------ */
/* Single block. tid 0 spins on the doorbell, broadcasts via __syncthreads. */
__global__ void persistent_kernel(Control *c, Table t,
                                  const uint32_t *types, const uint64_t *keys,
                                  const uint64_t *vals, uint32_t *stat,
                                  uint64_t *oval) {
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    uint32_t seen = 0;

    for (;;) {
        if (tid == 0) {
            while (c->batch_id == seen && c->stop == 0) { __nanosleep(128); }
        }
        __syncthreads();
        if (c->stop != 0) return;

        /* Capture the batch we are about to process. It is stable here: the
         * host cannot advance batch_id until it observes done_id == this value,
         * and it is blocked spinning on that. Using this captured id (instead of
         * re-reading c->batch_id after signalling done) avoids a lost-doorbell
         * race where the host's next batch_id write is mistaken for "already
         * seen", which deadlocks under rapid back-to-back submits. */
        __threadfence_system();   /* acquire: count/payload written before the doorbell are now visible */
        const uint32_t my_batch = c->batch_id;
        const uint32_t count    = c->count;
        const uint32_t scheme   = c->scheme;

        for (uint32_t i = tid; i < count; i += nt) {
            uint64_t v = 0;
            uint32_t rc;
            switch (types[i]) {
                case GPU_OLTP_LOOKUP: rc = ht_lookup(t, keys[i], &v);          break;
                case GPU_OLTP_INSERT: rc = do_write(t, scheme, keys[i], vals[i]); break;
                case GPU_OLTP_UPDATE: rc = do_write(t, scheme, keys[i], vals[i]); break;
                case GPU_OLTP_DELETE: rc = ht_delete(t, keys[i]);              break;
                default:              rc = GPU_OLTP_NOTFOUND;                  break;
            }
            if (stat) stat[i] = rc;
            if (oval) oval[i] = v;
        }

        __threadfence_system();   /* publish results before signalling done   */
        __syncthreads();
        if (tid == 0) { c->done_id = my_batch; __threadfence_system(); }
        __syncthreads();
        seen = my_batch;          /* mark exactly the batch we processed       */
        __syncthreads();
    }
}

/* ------------------------------ host API ------------------------------- */
static uint64_t round_pow2(uint64_t x) {
    uint64_t p = 1; while (p < x) p <<= 1; return p;
}

extern "C" GpuOltpEngine *gpu_oltp_create(uint64_t capacity, int threads_per_block) {
    GpuOltpEngine *e = (GpuOltpEngine *)calloc(1, sizeof(GpuOltpEngine));
    e->tpb = threads_per_block > 0 ? threads_per_block : 1024;

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    capacity = round_pow2(capacity);
    e->tbl.capacity = capacity;
    e->tbl.mask     = capacity - 1;
    CUDA_CHECK(cudaMalloc(&e->tbl.keys,   capacity * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&e->tbl.values, capacity * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&e->tbl.locks,  capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&e->tbl.glock,  sizeof(int)));
    CUDA_CHECK(cudaMemset(e->tbl.keys,  0xFF, capacity * sizeof(uint64_t))); /* EMPTY */
    CUDA_CHECK(cudaMemset(e->tbl.locks, 0,    capacity * sizeof(int)));
    CUDA_CHECK(cudaMemset(e->tbl.glock, 0,    sizeof(int)));

    /* mapped (zero-copy) control + request/response rings */
    auto map_alloc = [](void **hp, void **dp, size_t bytes) {
        CUDA_CHECK(cudaHostAlloc(hp, bytes, cudaHostAllocMapped));
        memset(*hp, 0, bytes);
        CUDA_CHECK(cudaHostGetDevicePointer(dp, *hp, 0));
    };
    map_alloc((void **)&e->h_ctrl,  (void **)&e->d_ctrl,  sizeof(Control));
    map_alloc((void **)&e->h_types, (void **)&e->d_types, MAX_BATCH * sizeof(uint32_t));
    map_alloc((void **)&e->h_keys,  (void **)&e->d_keys,  MAX_BATCH * sizeof(uint64_t));
    map_alloc((void **)&e->h_vals,  (void **)&e->d_vals,  MAX_BATCH * sizeof(uint64_t));
    map_alloc((void **)&e->h_stat,  (void **)&e->d_stat,  MAX_BATCH * sizeof(uint32_t));
    map_alloc((void **)&e->h_oval,  (void **)&e->d_oval,  MAX_BATCH * sizeof(uint64_t));

    e->h_ctrl->scheme = GPU_OLTP_LOCKFREE;
    e->cur_batch = 0;

    /* Non-blocking stream: the persistent kernel runs forever, so it must NOT
     * be synchronized-with by the legacy default stream. Otherwise a later
     * default-stream cudaMemcpy (e.g. gpu_oltp_snapshot) would implicitly wait
     * for the never-completing kernel and deadlock. */
    CUDA_CHECK(cudaStreamCreateWithFlags(&e->stream, cudaStreamNonBlocking));
    persistent_kernel<<<1, e->tpb, 0, e->stream>>>(
        e->d_ctrl, e->tbl, e->d_types, e->d_keys, e->d_vals, e->d_stat, e->d_oval);
    CUDA_CHECK(cudaGetLastError());

    /* WDDM (Windows / WSL2) batches launches in a command queue that is only
     * flushed at a synchronization point. Without this, the persistent kernel
     * sits in the queue and never starts, so the host spins on done_id forever.
     * Recording + querying an event forces the queue to flush and the kernel to
     * launch. No-op cost on Linux/TCC. */
    cudaEvent_t launch_ev;
    CUDA_CHECK(cudaEventCreate(&launch_ev));
    CUDA_CHECK(cudaEventRecord(launch_ev, e->stream));
    cudaEventQuery(launch_ev);   /* deliberately not CUDA_CHECK: returns cudaErrorNotReady, which is expected and fine */
    CUDA_CHECK(cudaEventDestroy(launch_ev));
    return e;
}

extern "C" void gpu_oltp_set_scheme(GpuOltpEngine *e, int scheme) {
    e->h_ctrl->scheme = (uint32_t)scheme;
}

extern "C" void gpu_oltp_submit(GpuOltpEngine *e,
                                const uint32_t *types, const uint64_t *keys,
                                const uint64_t *values, uint32_t *out_status,
                                uint64_t *out_values, uint32_t n) {
    uint32_t off = 0;
    while (off < n) {
        uint32_t chunk = n - off; if (chunk > MAX_BATCH) chunk = MAX_BATCH;
        memcpy(e->h_types, types  + off, chunk * sizeof(uint32_t));
        memcpy(e->h_keys,  keys   + off, chunk * sizeof(uint64_t));
        if (values) memcpy(e->h_vals, values + off, chunk * sizeof(uint64_t));
        e->h_ctrl->count = chunk;

        std::atomic_thread_fence(std::memory_order_release);
        e->cur_batch += 1;
        e->h_ctrl->batch_id = e->cur_batch;     /* doorbell */

        while (e->h_ctrl->done_id != e->cur_batch) { /* spin */ }
        std::atomic_thread_fence(std::memory_order_acquire);

        if (out_status) memcpy(out_status + off, e->h_stat, chunk * sizeof(uint32_t));
        if (out_values) memcpy(out_values + off, e->h_oval, chunk * sizeof(uint64_t));
        off += chunk;
    }
}

extern "C" int gpu_oltp_insert(GpuOltpEngine *e, uint64_t key, uint64_t value) {
    uint32_t t = GPU_OLTP_INSERT, s; gpu_oltp_submit(e, &t, &key, &value, &s, nullptr, 1); return (int)s;
}
extern "C" int gpu_oltp_update(GpuOltpEngine *e, uint64_t key, uint64_t value) {
    uint32_t t = GPU_OLTP_UPDATE, s; gpu_oltp_submit(e, &t, &key, &value, &s, nullptr, 1); return (int)s;
}
extern "C" int gpu_oltp_lookup(GpuOltpEngine *e, uint64_t key, uint64_t *out_value) {
    uint32_t t = GPU_OLTP_LOOKUP, s; uint64_t v = 0;
    gpu_oltp_submit(e, &t, &key, nullptr, &s, &v, 1);
    if (out_value) *out_value = v; return (int)s;
}
extern "C" int gpu_oltp_delete(GpuOltpEngine *e, uint64_t key) {
    uint32_t t = GPU_OLTP_DELETE, s; gpu_oltp_submit(e, &t, &key, nullptr, &s, nullptr, 1); return (int)s;
}

/* ===================== bandwidth scan / aggregate ====================== */
struct GpuScanCol { uint64_t *d; uint64_t n; };
static double g_scan_last_ms = 0.0;

static inline int grid_for(uint64_t n, int tpb) {
    uint64_t b = (n + tpb - 1) / tpb;
    return (b > 65535ULL) ? 65535 : (b < 1ULL ? 1 : (int)b);
}

__global__ void scan_fill_kernel(uint64_t *d, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) d[i] = i + 1;
}
#define SCAN_TPB 256
/* Block-reduction in shared memory, then ONE atomicAdd per block (not per
 * thread). With a capped grid this is bandwidth-bound, not atomic-bound. */
__global__ void scan_sum_kernel(const uint64_t *__restrict__ d, uint64_t n,
                                unsigned long long *out) {
    __shared__ unsigned long long sm[SCAN_TPB];
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long s = 0;
    for (; i < n; i += stride) s += d[i];
    sm[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sm[0]);
}
__global__ void scan_countlt_kernel(const uint64_t *__restrict__ d, uint64_t n,
                                    uint64_t thr, unsigned long long *out) {
    __shared__ unsigned long long sm[SCAN_TPB];
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long c = 0;
    for (; i < n; i += stride) if (d[i] < thr) c++;
    sm[threadIdx.x] = c;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) sm[threadIdx.x] += sm[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sm[0]);
}

extern "C" GpuScanCol *gpu_scan_alloc(uint64_t n) {
    GpuScanCol *s = (GpuScanCol *)calloc(1, sizeof(GpuScanCol));
    s->n = n;
    CUDA_CHECK(cudaMalloc(&s->d, n * sizeof(uint64_t)));
    const int tpb = 256;
    scan_fill_kernel<<<grid_for(n, tpb), tpb>>>(s->d, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    return s;
}

extern "C" uint64_t gpu_scan_count(GpuScanCol *s) { return s ? s->n : 0; }

static unsigned long long scan_reduce(GpuScanCol *s, bool is_count, uint64_t thr) {
    unsigned long long *d_out, h_out = 0;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_out, &h_out, sizeof(h_out), cudaMemcpyHostToDevice));
    const int tpb = SCAN_TPB;
    int blocks = grid_for(s->n, tpb);
    if (blocks > 4096) blocks = 4096;   /* cap grid; grid-stride covers the rest */
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    if (is_count) scan_countlt_kernel<<<blocks, tpb>>>(s->d, s->n, thr, d_out);
    else          scan_sum_kernel    <<<blocks, tpb>>>(s->d, s->n, d_out);
    cudaEventRecord(b);
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b); g_scan_last_ms = (double)ms;
    cudaEventDestroy(a); cudaEventDestroy(b);
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    return h_out;
}

extern "C" uint64_t gpu_scan_sum(GpuScanCol *s)              { return (uint64_t)scan_reduce(s, false, 0); }
extern "C" uint64_t gpu_scan_count_lt(GpuScanCol *s, uint64_t thr) { return (uint64_t)scan_reduce(s, true, thr); }
extern "C" double   gpu_scan_last_ms(void)                  { return g_scan_last_ms; }

extern "C" void gpu_scan_free(GpuScanCol *s) {
    if (s) { if (s->d) cudaFree(s->d); free(s); }
}

extern "C" uint64_t gpu_oltp_snapshot(GpuOltpEngine *e,
                                      uint64_t **out_keys, uint64_t **out_values) {
    const uint64_t cap = e->tbl.capacity;
    uint64_t *hk = (uint64_t *)malloc(cap * sizeof(uint64_t));
    uint64_t *hv = (uint64_t *)malloc(cap * sizeof(uint64_t));
    CUDA_CHECK(cudaMemcpy(hk, e->tbl.keys,   cap * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hv, e->tbl.values, cap * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t n = 0;
    for (uint64_t i = 0; i < cap; ++i) {
        const uint64_t k = hk[i];
        if (k == EMPTY_KEY || k == TOMBSTONE_KEY) continue;
        hk[n] = k; hv[n] = hv[i]; ++n;   /* compact live entries to the front */
    }
    if (out_keys)   *out_keys   = hk; else free(hk);
    if (out_values) *out_values = hv; else free(hv);
    return n;
}

extern "C" void gpu_oltp_destroy(GpuOltpEngine *e) {
    if (!e) return;
    e->h_ctrl->stop = 1;
    std::atomic_thread_fence(std::memory_order_release);
    e->h_ctrl->batch_id = e->cur_batch + 1;   /* wake the spinner so it exits */
    cudaStreamSynchronize(e->stream);
    cudaFree(e->tbl.keys); cudaFree(e->tbl.values);
    cudaFree(e->tbl.locks); cudaFree(e->tbl.glock);
    cudaFreeHost(e->h_ctrl); cudaFreeHost(e->h_types); cudaFreeHost(e->h_keys);
    cudaFreeHost(e->h_vals); cudaFreeHost(e->h_stat);  cudaFreeHost(e->h_oval);
    cudaStreamDestroy(e->stream);
    free(e);
}

/* ============================ microbenchmark =========================== */
#ifdef BUILD_BENCH
#include <chrono>
#include <random>
#include <vector>
#include <cmath>
#include <string>

/* Simple zipfian generator (Hartley/rejection-free precomputed CDF-lite). */
struct Zipf {
    std::vector<double> cdf; std::mt19937_64 rng;
    Zipf(uint64_t n, double theta, uint64_t seed) : rng(seed) {
        cdf.resize(n); double s = 0;
        for (uint64_t i = 1; i <= n; ++i) s += 1.0 / std::pow((double)i, theta);
        double c = 0;
        for (uint64_t i = 1; i <= n; ++i) { c += (1.0 / std::pow((double)i, theta)) / s; cdf[i - 1] = c; }
    }
    uint64_t next() {
        double u = std::uniform_real_distribution<double>(0, 1)(rng);
        uint64_t lo = 0, hi = cdf.size() - 1;
        while (lo < hi) { uint64_t m = (lo + hi) / 2; if (cdf[m] < u) lo = m + 1; else hi = m; }
        return lo;
    }
};

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    uint64_t cap      = (argc > 1) ? strtoull(argv[1], 0, 10) : (1ull << 24);
    uint64_t nkeys    = (argc > 2) ? strtoull(argv[2], 0, 10) : (1ull << 22);
    uint64_t nops     = (argc > 3) ? strtoull(argv[3], 0, 10) : (1ull << 22);
    int      scheme   = (argc > 4) ? atoi(argv[4]) : GPU_OLTP_LOCKFREE;
    double   theta    = (argc > 5) ? atof(argv[5]) : 0.0;   /* 0 = uniform     */
    int      writepct = (argc > 6) ? atoi(argv[6]) : 50;    /* % write ops     */

    printf("cap=%llu nkeys=%llu nops=%llu scheme=%d theta=%.2f write%%=%d\n",
           (unsigned long long)cap, (unsigned long long)nkeys,
           (unsigned long long)nops, scheme, theta, writepct);

    GpuOltpEngine *e = gpu_oltp_create(cap, 1024);
    gpu_oltp_set_scheme(e, scheme);

    /* preload */
    std::vector<uint32_t> ty(nkeys, GPU_OLTP_INSERT);
    std::vector<uint64_t> ke(nkeys), va(nkeys);
    for (uint64_t i = 0; i < nkeys; ++i) { ke[i] = i + 1; va[i] = (i + 1) * 7; }
    double t0 = now_s();
    gpu_oltp_submit(e, ty.data(), ke.data(), va.data(), nullptr, nullptr, (uint32_t)nkeys);
    double t1 = now_s();
    printf("preload: %.3f Mops/s\n", nkeys / (t1 - t0) / 1e6);

    /* workload */
    std::mt19937_64 rng(12345);
    Zipf zipf(nkeys, theta, 999);
    std::vector<uint32_t> wty(nops); std::vector<uint64_t> wke(nops), wva(nops);
    for (uint64_t i = 0; i < nops; ++i) {
        uint64_t k = (theta > 0.0) ? (zipf.next() + 1)
                                   : (rng() % nkeys) + 1;
        bool w = (int)(rng() % 100) < writepct;
        wty[i] = w ? GPU_OLTP_UPDATE : GPU_OLTP_LOOKUP;
        wke[i] = k; wva[i] = k * 11;
    }
    std::vector<uint32_t> st(nops); std::vector<uint64_t> ov(nops);

    t0 = now_s();
    gpu_oltp_submit(e, wty.data(), wke.data(), wva.data(), st.data(), ov.data(), (uint32_t)nops);
    t1 = now_s();
    printf("workload: %.3f Mops/s (%.0f ns/op amortized)\n",
           nops / (t1 - t0) / 1e6, (t1 - t0) / nops * 1e9);

    /* single-op round-trip latency (the PCIe baseline that C2C will cut) */
    const int RT = 2000; uint64_t dummy;
    double tl0 = now_s();
    for (int i = 0; i < RT; ++i) gpu_oltp_lookup(e, (i % nkeys) + 1, &dummy);
    double tl1 = now_s();
    printf("single-op RT latency: %.2f us (PCIe baseline)\n", (tl1 - tl0) / RT * 1e6);

    gpu_oltp_destroy(e);
    return 0;
}
#endif /* BUILD_BENCH */
