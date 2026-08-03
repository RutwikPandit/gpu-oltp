/* rgi_persist_engine.cu — v1 PERSISTENT-KERNEL binding for the RGI chain
 * hashtable, built ALONGSIDE the existing launch-per-batch engine (additive;
 * nothing in rgi_oltp_engine.cu / the FDW / the worker changes).
 *
 * Architecture (the GB-class dispatch model, prototyped on PCIe):
 *   - ONE resident kernel, sized to exact full occupancy, that never exits.
 *   - Host -> GPU: a 64B mapped (zero-copy) control line as the doorbell
 *     (batch_id / count / stop), request payload staged H2D on a copy stream
 *     BEFORE the doorbell rings (NVMe-style: payload DMA, doorbell MMIO).
 *   - TWO-LEVEL doorbell: only block 0 polls the mapped line over PCIe; it
 *     republishes {batch_id, count} into device memory (L2) that the other
 *     blocks poll cheaply. Stop = sentinel batch_id 0xFFFFFFFF.
 *   - Completion: each block atomicAdd's an arrival counter; block 0 waits for
 *     gridDim arrivals, resets it, fences, writes done_id to the mapped line.
 *   - Per-request execution calls RGI's PUBLIC device API
 *     (cooperative_insert / cooperative_find) with the same ballot-queue
 *     pattern as RGI's own batch_kernel. RGI source is NOT modified.
 *
 * v1 scope: INSERT + FIND only. (Erase needs DEBRA epoch drains restructured
 * around batch quiescent points — the kernel never exits, so the end-of-kernel
 * drain_all in batch_kernel never runs. v2 item, documented.)
 *
 * Toy-engine lessons carried over verbatim:
 *   1. WDDM launch flush (event record+query after launch).
 *   2. Capture the batch id ONCE per batch (lost-doorbell race fix),
 *      generalized: per-block 'seen', shared-memory broadcast.
 *   3. Non-blocking streams ONLY (a blocking-stream memcpy would deadlock
 *      against the never-ending kernel).
 *
 * KNOWN HAZARD: while the persistent kernel is resident it occupies every SM
 * block slot, so NO other kernel can launch (it would deadlock). The bench
 * therefore runs the launch-mode sweep FIRST, then starts the resident kernel.
 *
 * Build (WSL):
 *   nvcc -std=c++17 -arch=sm_89 --expt-extended-lambda --expt-relaxed-constexpr \
 *        -maxrregcount=64 -I<RGI>/include rgi_persist_engine.cu -o rgi_persist
 */
#include <gpu_chainhashtable.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <atomic>

using slab_t  = simple_slab_allocator<128>;
using debra_t = simple_debra_reclaimer<>;
using table_t = GpuHashtable::gpu_chainhashtable<slab_t, debra_t, 16>;
using alloc_ctx_t  = table_t::device_allocator_context_type;
using alloc_inst_t = slab_t::device_instance_type;

#define TILE        16
#define BLOCK_SIZE  128
#define PMAX_BATCH  (1u << 20)
#define STOP_SENTINEL 0xFFFFFFFFu

#define CK(call)                                                              \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                        \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

/* 64B mapped control line: the only thing the host and block 0 share. */
struct PersistCtl {
  volatile uint32_t batch_id;   /* host bumps to publish a batch          */
  volatile uint32_t done_id;    /* block 0 sets when the grid finished it */
  volatile uint32_t count;      /* requests in this batch                 */
  volatile uint32_t stop;       /* host sets 1 to retire the kernel       */
  uint32_t pad[12];
};

enum { POP_FIND = 0, POP_INSERT = 1 };

/* ------------------------------- kernel -------------------------------- */
__global__ void __launch_bounds__(BLOCK_SIZE, 8)
rgi_persistent_kernel(table_t table, alloc_inst_t alloc_inst,
                      PersistCtl *ctl,
                      uint32_t *g_pub,        /* device: [0]=batch|sentinel, [1]=count */
                      unsigned int *g_arrive, /* device: arrival counter               */
                      const uint8_t  *d_types,
                      const uint32_t *d_keys,  /* 2 slices per key */
                      const uint32_t *d_vals,
                      uint32_t       *d_out)
{
  __shared__ cg::block_tile_memory<BLOCK_SIZE> tile_shmem;
  auto block = cg::this_thread_block(tile_shmem);
  auto tile  = cg::tiled_partition<TILE>(block);
  alloc_ctx_t allocator{alloc_inst, tile};

  __shared__ uint32_t s_batch, s_count;
  uint32_t seen = 0;

  for (;;) {
    /* ---- level 1: only block 0 touches the mapped line (PCIe) ---- */
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (ctl->batch_id == seen && ctl->stop == 0) { __nanosleep(128); }
      __threadfence_system();                  /* acquire: payload H2D landed before doorbell */
      uint32_t b = ctl->stop ? STOP_SENTINEL : ctl->batch_id;
      ((volatile uint32_t *)g_pub)[1] = ctl->count;
      __threadfence();                         /* count visible before batch id */
      ((volatile uint32_t *)g_pub)[0] = b;
    }
    /* ---- level 2: every block polls device memory (L2-cheap) ---- */
    if (threadIdx.x == 0) {
      uint32_t b;
      do { b = ((volatile uint32_t *)g_pub)[0]; if (b == STOP_SENTINEL) break; __nanosleep(64); }
      while (b == seen);
      s_batch = b;
      s_count = ((volatile uint32_t *)g_pub)[1];
    }
    __syncthreads();
    const uint32_t my_batch = s_batch;         /* captured ONCE (race-fix pattern) */
    if (my_batch == STOP_SENTINEL) return;
    const uint32_t count = s_count;

    /* ---- ballot-queue over the batch, grid-stride (batch_kernel pattern) ---- */
    const uint32_t span = ((count + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    for (uint32_t tid = threadIdx.x + blockIdx.x * BLOCK_SIZE; tid < span;
         tid += gridDim.x * BLOCK_SIZE) {
      bool task = (tid < count);
      const uint32_t *key  = task ? d_keys + 2ull * tid : nullptr;
      uint32_t        type = task ? (uint32_t)d_types[tid] : 0;
      uint32_t        val  = task ? d_vals[tid] : 0;
      uint32_t        out  = 0xFFFFFFFFu;
      auto wq = tile.ballot(task);
      while (wq) {
        int  r  = __ffs(wq) - 1;
        auto ck = tile.shfl(key, r);
        auto ct = tile.shfl(type, r);
        if (ct == POP_INSERT) {
          auto cv = tile.shfl(val, r);
          table.cooperative_insert<true>(ck, 2, cv, tile, allocator, /*update_if_exists=*/true);
        } else {
          auto v = table.cooperative_find<true, true>(ck, 2, tile, allocator);
          if (tile.thread_rank() == r) out = v;
        }
        if (tile.thread_rank() == r) task = false;
        wq = tile.ballot(task);
      }
      if (tid < count) d_out[tid] = out;
    }

    /* ---- completion: arrival counter; block 0 signals the host ---- */
    __threadfence();                            /* publish d_out before arrival */
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(g_arrive, 1u);
      if (blockIdx.x == 0) {
        while (atomicAdd(g_arrive, 0u) < gridDim.x) { __nanosleep(64); }
        atomicExch(g_arrive, 0u);
        __threadfence_system();                 /* results + reset visible before done */
        ctl->done_id = my_batch;
      }
    }
    __syncthreads();
    seen = my_batch;
  }
}

/* ------------------------------ host side ------------------------------ */
struct PersistEngine {
  PersistCtl  *h_ctl, *d_ctl;
  uint32_t    *g_pub;
  unsigned int*g_arrive;
  uint8_t     *d_types;
  uint32_t    *d_keys, *d_vals, *d_out;
  cudaStream_t kstream, cstream;   /* both NON-BLOCKING (toy lesson 3) */
  uint32_t     cur_batch = 0;
  int          grid = 0;
};

static void persist_start(PersistEngine &p, table_t &table, slab_t &ha) {
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  CK(cudaHostAlloc((void **)&p.h_ctl, sizeof(PersistCtl), cudaHostAllocMapped));
  memset((void *)p.h_ctl, 0, sizeof(PersistCtl));
  CK(cudaHostGetDevicePointer((void **)&p.d_ctl, (void *)p.h_ctl, 0));
  CK(cudaMalloc(&p.g_pub, 2 * sizeof(uint32_t)));
  CK(cudaMemset(p.g_pub, 0, 2 * sizeof(uint32_t)));
  CK(cudaMalloc(&p.g_arrive, sizeof(unsigned int)));
  CK(cudaMemset(p.g_arrive, 0, sizeof(unsigned int)));
  CK(cudaMalloc(&p.d_types, PMAX_BATCH));
  CK(cudaMalloc(&p.d_keys, sizeof(uint32_t) * 2 * (size_t)PMAX_BATCH));
  CK(cudaMalloc(&p.d_vals, sizeof(uint32_t) * (size_t)PMAX_BATCH));
  CK(cudaMalloc(&p.d_out,  sizeof(uint32_t) * (size_t)PMAX_BATCH));
  CK(cudaStreamCreateWithFlags(&p.kstream, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&p.cstream, cudaStreamNonBlocking));

  /* exact full-occupancy grid: every block must be co-resident or we deadlock */
  int bpm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &bpm, rgi_persistent_kernel, BLOCK_SIZE, 0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  p.grid = bpm * prop.multiProcessorCount;
  printf("persistent kernel: %d blocks/SM x %d SMs = %d resident blocks\n",
         bpm, prop.multiProcessorCount, p.grid);

  rgi_persistent_kernel<<<p.grid, BLOCK_SIZE, 0, p.kstream>>>(
      table, ha.get_device_instance(), p.d_ctl, p.g_pub, p.g_arrive,
      p.d_types, p.d_keys, p.d_vals, p.d_out);
  CK(cudaGetLastError());
  /* WDDM flush (toy lesson 1): force the launch out of the command queue */
  cudaEvent_t ev;
  CK(cudaEventCreate(&ev));
  CK(cudaEventRecord(ev, p.kstream));
  cudaEventQuery(ev);
  CK(cudaEventDestroy(ev));
}

/* Submit one batch through the doorbell; results land in out (may be NULL). */
static void persist_submit(PersistEngine &p, const uint8_t *types,
                           const uint64_t *keys, const uint32_t *vals,
                           uint32_t *out, uint32_t n) {
  for (uint32_t off = 0; off < n; off += PMAX_BATCH) {
    uint32_t c = (n - off > PMAX_BATCH) ? PMAX_BATCH : (n - off);
    CK(cudaMemcpyAsync(p.d_types, types + off, c, cudaMemcpyHostToDevice, p.cstream));
    CK(cudaMemcpyAsync(p.d_keys, keys + off, sizeof(uint64_t) * c, cudaMemcpyHostToDevice, p.cstream));
    if (vals) CK(cudaMemcpyAsync(p.d_vals, vals + off, sizeof(uint32_t) * c, cudaMemcpyHostToDevice, p.cstream));
    CK(cudaStreamSynchronize(p.cstream));      /* payload resident BEFORE doorbell */
    p.h_ctl->count = c;
    std::atomic_thread_fence(std::memory_order_release);
    p.cur_batch += 1;
    p.h_ctl->batch_id = p.cur_batch;           /* ring */
    while (p.h_ctl->done_id != p.cur_batch) { /* spin */ }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (out) {
      CK(cudaMemcpyAsync(out + off, p.d_out, sizeof(uint32_t) * c, cudaMemcpyDeviceToHost, p.cstream));
      CK(cudaStreamSynchronize(p.cstream));
    }
  }
}

static void persist_stop(PersistEngine &p) {
  p.h_ctl->stop = 1;
  std::atomic_thread_fence(std::memory_order_release);
  p.h_ctl->batch_id = p.cur_batch + 1;         /* wake the block-0 spinner */
  CK(cudaStreamSynchronize(p.kstream));
}

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/* -------------------------------- bench -------------------------------- */
int main(int argc, char **argv) {
  uint32_t N = (argc > 1) ? (uint32_t)strtoul(argv[1], 0, 10) : 2000000u;

  slab_t ha(0.4f);
  debra_t hr;
  table_t table(ha, hr, (std::size_t)N, 2.0f);

  /* base population + device buffers for LAUNCH-mode (existing path) */
  std::vector<uint64_t> keys(N);
  std::vector<uint32_t> vals(N);
  for (uint32_t i = 0; i < N; i++) { keys[i] = (uint64_t)i + 1; vals[i] = i + 1; }
  uint32_t *dk, *dv, *dout;
  CK(cudaMalloc(&dk, sizeof(uint32_t) * 2 * (size_t)N));
  CK(cudaMalloc(&dv, sizeof(uint32_t) * (size_t)N));
  CK(cudaMalloc(&dout, sizeof(uint32_t) * (size_t)N));
  CK(cudaMemcpy(dk, keys.data(), sizeof(uint64_t) * N, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv, vals.data(), sizeof(uint32_t) * N, cudaMemcpyHostToDevice));
  table.insert<true>(dk, 2, nullptr, dv, N, 0, true);
  CK(cudaDeviceSynchronize());
  printf("populated %u keys\n", N);

  const uint32_t batches[] = {1, 8, 64, 512, 4096, 32768, 262144, 1048576};
  const int NB = sizeof(batches) / sizeof(batches[0]);
  double launch_us[NB], persist_us[NB];

  /* ---- phase 1: LAUNCH-mode floor (must run BEFORE the resident kernel) ---- */
  for (int bi = 0; bi < NB; bi++) {
    uint32_t B = batches[bi]; if (B > N) { launch_us[bi] = -1; continue; }
    int iters = (B <= 512) ? 2000 : (B <= 32768 ? 200 : 20);
    table.find<false, true>(dk, 2, nullptr, dout, B);  CK(cudaDeviceSynchronize());
    double t0 = now_s();
    for (int it = 0; it < iters; ++it) table.find<false, true>(dk, 2, nullptr, dout, B);
    CK(cudaDeviceSynchronize());
    launch_us[bi] = (now_s() - t0) / iters * 1e6;
  }
  printf("launch-mode sweep done\n");

  /* ---- phase 2: PERSISTENT mode ---- */
  PersistEngine p;
  persist_start(p, table, ha);

  /* correctness: persistent FIND of known keys */
  {
    uint32_t M = 1024;
    std::vector<uint8_t>  t(M, POP_FIND);
    std::vector<uint64_t> k(M);
    std::vector<uint32_t> o(M);
    for (uint32_t i = 0; i < M; i++) k[i] = (uint64_t)(i * 7 % N) + 1;
    persist_submit(p, t.data(), k.data(), nullptr, o.data(), M);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < M; i++) if (o[i] != (uint32_t)k[i]) bad++;
    printf("persistent FIND validate: %u/%u correct\n", M - bad, M);
    /* persistent INSERT of fresh keys, then persistent FIND of them */
    std::vector<uint8_t>  ti(M, POP_INSERT);
    std::vector<uint64_t> ki(M);
    std::vector<uint32_t> vi(M);
    for (uint32_t i = 0; i < M; i++) { ki[i] = (uint64_t)N + 1 + i; vi[i] = 7777u + i; }
    persist_submit(p, ti.data(), ki.data(), vi.data(), nullptr, M);
    persist_submit(p, t.data(), ki.data(), nullptr, o.data(), M);
    bad = 0;
    for (uint32_t i = 0; i < M; i++) if (o[i] != 7777u + i) bad++;
    printf("persistent INSERT+FIND validate: %u/%u correct\n", M - bad, M);
  }

  /* persistent-mode floor sweep (find batches via doorbell) */
  {
    std::vector<uint8_t>  t(PMAX_BATCH, POP_FIND);
    std::vector<uint32_t> o(PMAX_BATCH);
    for (int bi = 0; bi < NB; bi++) {
      uint32_t B = batches[bi]; if (B > N) { persist_us[bi] = -1; continue; }
      int iters = (B <= 512) ? 1000 : (B <= 32768 ? 100 : 10);
      persist_submit(p, t.data(), keys.data(), nullptr, o.data(), B);   /* warm */
      double t0 = now_s();
      for (int it = 0; it < iters; ++it)
        persist_submit(p, t.data(), keys.data(), nullptr, nullptr, B);  /* no D2H in timed loop */
      persist_us[bi] = (now_s() - t0) / iters * 1e6;
    }
  }
  persist_stop(p);

  /* ---- report ---- */
  printf("\n%-10s %-16s %-16s %-18s %-18s\n",
         "batchB", "launch(us)", "persist(us)", "launch(Mop/s)", "persist(Mop/s)");
  for (int bi = 0; bi < NB; bi++) {
    uint32_t B = batches[bi]; if (B > N) continue;
    printf("%-10u %-16.2f %-16.2f %-18.1f %-18.1f\n", B,
           launch_us[bi], persist_us[bi],
           B / launch_us[bi], B / persist_us[bi]);
  }
  printf("\nNOTE: persist(us) includes the H2D payload stage (copy-stream sync) per\n"
         "rendezvous; the doorbell+completion round trip dominates at small B.\n");
  return 0;
}
