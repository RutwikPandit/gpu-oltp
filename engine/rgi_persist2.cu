/* rgi_persist2.cu — v2 PERSISTENT-KERNEL binding for the RGI chain hashtable.
 * ADDITIVE file: v1 (rgi_persist_engine.cu) stays intact as the published
 * baseline. This file exists to turn the fig16 "B* ~ 1,100" MODEL into a
 * MEASURED end-to-end number on GH200 (see HANDOFF_GH200.md sec.5 and
 * bench/gh200_campaign_results.md sec.4/5).
 *
 * ---------------------------------------------------------------------------
 * DESIGN (v2, final measured configuration — consumer-side placement
 * everywhere; every stage below was chosen by measurement on the box)
 * ---------------------------------------------------------------------------
 * The v1 bottlenecks (diagnosed on GH200 2026-07-07) and what replaced them:
 *
 * 1. Payload staging (cudaMemcpyAsync+sync per batch; a zero-copy mapped
 *    variant made EVERY key deref a ~0.9 us C2C round trip instead)
 *    -> requests (types+keys) AND results (out) live in MANAGED MEMORY
 *       PREFERRED ON HBM. The host writes requests over C2C (posted
 *       stores, excluded from the timed loop like the CPU baseline's
 *       prewritten arrays); the GPU reads and writes locally. Moving out[]
 *       from mapped-host to HBM alone moved the pipelined crossover from
 *       B=16k to B=1k: per-op C2C stores were the last per-op serial cost.
 *
 * 2. Dispatch. Three designs measured, in order:
 *    (a) two-level doorbell (v1): block 0 polls mapped line over C2C,
 *        republishes ONE word every leader consumes serially — sync floor
 *        3.8-5 us, but pipelining is limited: every block must observe
 *        every batch (~0.2 us L2 read each, serial per leader).
 *    (b) host-written per-block mailboxes: host pays nact C2C RFOs per
 *        post — 7 us/batch at nact=128; the HOST becomes the bottleneck.
 *    (c) FINAL: RING + GPU DISPATCHER SHARDS + PER-BLOCK MAILBOXES.
 *        Host posts ONE packed word {count|offset|gen|uniq} into
 *        g_ring[b % RING_K] (managed HBM, one posted 8 B store, no cuda
 *        calls). NDISP=8 dispatcher blocks shard batches round-robin
 *        ((b-1) % NDISP); a dispatcher's leader polls the ring from L2 and
 *        its 128 threads fan the word out to the participants' private
 *        mailbox lines in parallel (ordinals via atomicAdd on disp_k so
 *        cross-dispatcher order is safe; a block consumes in ordinal
 *        order). A serving leader polls ONLY its own line: zero traffic
 *        for batches it doesn't serve. Measured dispatch cadence floor:
 *        0.23 us/batch pipelined (count=0), 6 us sync (the extra hop
 *        costs sync latency; launch-comparable, and pipelining is what
 *        the serving story needs).
 *
 * 3. Completion. Also three designs measured: grid-wide arrival (v1);
 *    per-participant done bytes scanned by the host (host scan of
 *    C2C-invalidated lines ~1.3 us/batch at nact=128 — host became the
 *    pacer); FINAL: LAST-ARRIVER POSTS PER-GENERATION — participants of
 *    batch b bump the padded device counter of generation b % WINDOW_K
 *    (system fence first); the last arriver resets it and posts ONE tag
 *    byte to the generation's mapped-host done line. The RMW chain
 *    (~15 ns x nact) adds batch latency but runs in parallel across
 *    in-flight batches; the host pays exactly one local read per batch.
 *
 * 4. Request->tile mapping (v1: consecutive-128 chunks = 16 SERIAL ops per
 *    16-lane tile ~1.4 us each = the 24 us plateau)
 *    -> strided tiles + PER-BATCH BLOCK ROTATION: batch b uses serving
 *       blocks [(b*nact) % sgrid ...), so consecutive pipelined batches
 *       land on DISJOINT block sets and execute concurrently. At
 *       B <= 8*sgrid every tile has at most ONE op: kernel time ~ one
 *       probe, in parallel. Lane 0 stages each request via shuffle so
 *       RGI's repeated key derefs hit registers (measured worth ~25%).
 *
 * PIPELINING: the fig16 model line (D/B + 0.2 ns/op) implicitly assumes the
 * GPU is kept busy — i.e. batches OVERLAP (the saturated-CPU side gets the
 * same courtesy: 64 threads with full queues). The sync rendezvous (W=1)
 * cannot beat the model at small B because one random HBM probe chain alone
 * is ~1-2 us of latency. So the bench measures BOTH:
 *   - W=1   sync rendezvous latency (the honest per-batch latency number)
 *   - W<=64 windowed submission (the honest peak-serving throughput number;
 *           host posts ahead while older batches execute, waits b-W)
 *
 * Toy-engine lessons kept: capture-batch-word-once, non-blocking streams,
 * WDDM flush, exact full-occupancy grid (all blocks co-resident or deadlock).
 *
 * Workload (what makes this real, not a ping-pong): 64M-key RGI chain table
 * in HBM populated via the launch path BEFORE the resident grid starts; the
 * 1M-entry request array prefilled ONCE with uniform-random keys in [1,N]
 * (YCSB-C read shape: real probes, real suffix derefs, real HBM); offset
 * rotates by B each batch so L2 can't turn it into a cache benchmark.
 * Payload materialization is EXCLUDED and symmetric: the CPU baseline reads
 * prewritten key arrays from its local DRAM; the GPU reads prewritten
 * request words from its local HBM. Timed = doorbell + probes + completion.
 *
 * Build (GH200, aarch64 + sm_90):
 *   nvcc -std=c++17 -arch=sm_90 --expt-extended-lambda --expt-relaxed-constexpr \
 *        -maxrregcount=64 -I ~/work/RobustGPUIndexing/include \
 *        rgi_persist2.cu -o ~/work/bin/rgi_persist2
 */
#include <gpu_chainhashtable.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <random>
#include <atomic>
#ifdef __linux__
#include <sched.h>
#endif

using slab_t  = simple_slab_allocator<128>;
using debra_t = simple_debra_reclaimer<>;
using table_t = GpuHashtable::gpu_chainhashtable<slab_t, debra_t, 16>;
using alloc_ctx_t  = table_t::device_allocator_context_type;
using alloc_inst_t = slab_t::device_instance_type;

#define TILE        16
#define BLOCK_SIZE  128
#define TILES_PER_BLOCK (BLOCK_SIZE / TILE)      /* 8 */
#define PMAX_BATCH  (1u << 20)                   /* 1M request slots */
/* Ring depth: the dispatchers lag the host by at most the in-flight window,
 * but slots must also survive until consumed — 4096 is orders beyond it. */
#define RING_K      4096u
#define NDISP       8u    /* dispatcher blocks; batch b -> dispatcher (b-1)%NDISP,
                             so the ~1.6 us serial fan-out stage is sharded */
#define HOST_WINDOW 64u   /* max batches in flight (pipelined mode) */
/* completion generations = 2x the window: post(b) may precede wait(b-W),
 * so generation b%K must not collide with any incomplete batch — K=2W
 * guarantees batch b-2W was waited (counter reset, done consumed) before
 * b is posted. Must divide 2048 (the batchlo field period). */
#define WINDOW_K    (2u * HOST_WINDOW)
/* mailbox slots per block: a block's consecutive assignments alternate
 * through NSLOT slots, so up to NSLOT of its batches can be posted but
 * unconsumed. Bounds the window when batches share blocks:
 * W_eff = min(HOST_WINDOW, NSLOT * (sgrid / nact)) — NSLOT=32 keeps the
 * full window even at nact = sgrid (B >= ~10k). */
#define NSLOT       32u

#define CK(call)                                                              \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                        \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

enum { POP_FIND = 0, POP_INSERT = 1, POP_NOOP = 2 /* diagnostic: no probe */ };

/* Packed batch word: count(21) | offset(21) | gen(7) | uniq(15).
 * gen = batch % WINDOW_K selects the completion generation; uniq (15 bits
 * of batch >> 7) makes the done tag differ from the previous occupant of
 * the same generation (collision would need 4M+ batches in flight). */
static __host__ __device__ inline unsigned long long
pub_pack(unsigned long long batch, uint32_t count, uint32_t offset) {
  return (count & 0x1FFFFFull) |
         ((unsigned long long)(offset & 0x1FFFFF) << 21) |
         ((batch & 127ull) << 42) |
         (((batch >> 7) & 0x7FFFull) << 49);
}

/* Per-(block, slot) mailbox line: the dispatcher writes rel + payload,
 * fences, then writes seq = the block's assignment ordinal (1, 2, 3, ...).
 * A serving block's leader polls ONLY its own next slot: batches it does
 * not serve cost it nothing (the fix for the shared-word serial-consume
 * bottleneck), so pipelined batches dispatch independently. */
struct Mailbox {
  volatile unsigned long long seq;      /* assignment ordinal; ~0ull = stop */
  volatile unsigned long long payload;  /* pub_pack word */
  volatile unsigned long long rel;      /* this block's rank within the batch */
  uint32_t pad[10];
};

/* Done-byte tag for batch b: nonzero, differs from the previous occupant
 * of the same generation (b vs b - WINDOW_K differ in uniq). */
static __host__ __device__ inline uint8_t done_tag(unsigned long long w) {
  return (uint8_t)(0x80u | ((uint32_t)(w >> 49) & 0x7Fu));
}

/* Arrival counters padded to 128 B: concurrent batches (distinct
 * generations) must not serialize their completion RMWs on a shared line. */
struct ArriveCtr {
  unsigned int n;
  uint32_t pad[31];
};

static __host__ __device__ inline uint32_t
active_blocks(uint32_t count, uint32_t grid) {
  uint32_t n = (count + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;
  if (n == 0) n = 1;                 /* count=0 floor probe still completes */
  return n < grid ? n : grid;
}

/* ------------------------------- kernel -------------------------------- */
__global__ void __launch_bounds__(BLOCK_SIZE, 8)
rgi_persistent2_kernel(table_t table, alloc_inst_t alloc_inst,
                       const unsigned long long *g_ring, /* managed HBM, RING_K */
                       Mailbox *mbox,                    /* device, grid*NSLOT */
                       unsigned long long *disp_k,       /* device, per-block ordinal */
                       volatile uint8_t *doneb,          /* mapped host, WINDOW_K lines */
                       ArriveCtr *g_arrive,              /* device, WINDOW_K padded ctrs */
                       const uint8_t  *d_types,          /* managed, HBM  */
                       const uint32_t *d_keys,           /* managed, HBM, 2 slices/key */
                       uint32_t       *d_out)            /* device (HBM), like launch mode */
{
  __shared__ cg::block_tile_memory<BLOCK_SIZE> tile_shmem;
  auto block = cg::this_thread_block(tile_shmem);
  auto tile  = cg::tiled_partition<TILE>(block);
  alloc_ctx_t allocator{alloc_inst, tile};

  const uint32_t sgrid = gridDim.x - NDISP;  /* serving blocks: NDISP.. */
  __shared__ unsigned long long s_pub;
  __shared__ uint32_t s_rel;

  if (blockIdx.x < NDISP) {
    /* ================= DISPATCHER BLOCKS =================
     * Dispatcher d serves batches b with (b-1) % NDISP == d: leader polls
     * the host's ring word from L2, then all 128 threads fan it out to
     * the participating blocks' mailboxes in parallel. Concurrent batches
     * fan out from different dispatchers; per-block assignment ordinals
     * are taken with atomicAdd (a serving block consumes in ordinal
     * order, so cross-batch write order does not matter). */
    unsigned long long expect = blockIdx.x + 1;
    for (;;) {
      if (threadIdx.x == 0) {
        const unsigned long long want = expect & 0x3FFFFFull;
        unsigned long long v;
        for (;;) {
          v = *(volatile unsigned long long *)&g_ring[expect % RING_K];
          if (v == ~0ull || (v >> 42) == want) break;
          __nanosleep(32);
        }
        __threadfence_system();  /* acquire: host request writes precede ring */
        s_pub = v;
      }
      __syncthreads();
      const unsigned long long w = s_pub;
      if (w == ~0ull) {
        /* dispatcher 0 fans out stop to every mailbox slot of every block
         * (the host drains all batches before stopping, so no dispatcher
         * is mid-fan-out here) */
        if (blockIdx.x == 0) {
          for (uint32_t i = threadIdx.x; i < gridDim.x * NSLOT; i += BLOCK_SIZE)
            mbox[i].seq = ~0ull;
          __threadfence();
        }
        return;
      }
      const uint32_t nact  = active_blocks((uint32_t)(w & 0x1FFFFF), sgrid);
      const uint32_t start = (uint32_t)((expect * nact) % sgrid);
      for (uint32_t j = threadIdx.x; j < nact; j += BLOCK_SIZE) {
        const uint32_t blk = NDISP + (start + j) % sgrid;
        const unsigned long long k = atomicAdd((unsigned long long *)&disp_k[blk], 1ull) + 1;
        Mailbox *mb = &mbox[(size_t)blk * NSLOT + ((k - 1) & (NSLOT - 1))];
        mb->payload = w;
        mb->rel = j;
        __threadfence();               /* payload+rel visible before seq */
        mb->seq = k;
      }
      expect += NDISP;
      __syncthreads();
    }
  }

  /* ================= SERVING BLOCKS ================= */
  unsigned long long myk = 1;          /* leader-only: next assignment ordinal */
  for (;;) {
    /* ---- doorbell: the leader polls ITS OWN next mailbox slot only.
     * seq is written by the dispatcher AFTER rel+payload (fenced), so a
     * seq match means the slot is complete. ---- */
    if (threadIdx.x == 0) {
      Mailbox *mb = &mbox[(size_t)blockIdx.x * NSLOT + ((myk - 1) & (NSLOT - 1))];
      unsigned long long s;
      /* exponential backoff: ~1,300 idle leaders polling every 64 ns is
       * ~130 GB/s of L2 read traffic that starves the actual probes */
      uint32_t bo = 32;
      for (;;) {
        s = mb->seq;
        if (s == myk || s == ~0ull) break;
        __nanosleep(bo);
        if (bo < 1024) bo <<= 1;
      }
      if (s == ~0ull) { s_pub = ~0ull; }
      else {
        __threadfence_system();  /* acquire payload + host request writes */
        s_pub = mb->payload;
        s_rel = (uint32_t)mb->rel;
        myk++;
      }
    }
    __syncthreads();
    const unsigned long long w = s_pub;   /* captured ONCE per block */
    if (w == ~0ull) return;
    const uint32_t count = (uint32_t)(w & 0x1FFFFF);
    const uint32_t off   = (uint32_t)((w >> 21) & 0x1FFFFF);
    const uint32_t gen   = (uint32_t)((w >> 42) & (WINDOW_K - 1));
    const uint32_t nact  = active_blocks(count, sgrid);
    const uint32_t rel   = s_rel;
    {
      /* strided tile mapping: at B <= nact*8 each tile has <= 1 op.
       * Lane 0 loads the request once; shuffle-broadcast so RGI's repeated
       * key derefs hit registers (measured faster than direct pointer). */
      const uint32_t T2 = nact * TILES_PER_BLOCK;
      for (uint32_t r = rel * TILES_PER_BLOCK + threadIdx.x / TILE;
           r < count; r += T2) {
        const uint32_t idx = off + r;
        uint32_t k0 = 0, k1 = 0, ty = 0;
        if (tile.thread_rank() == 0) {
          k0 = d_keys[2ull * idx];
          k1 = d_keys[2ull * idx + 1];
          ty = d_types[idx];
        }
        k0 = tile.shfl(k0, 0);
        k1 = tile.shfl(k1, 0);
        ty = tile.shfl(ty, 0);
        uint32_t kloc[2] = {k0, k1};
        if (ty == POP_INSERT) {
          /* value convention = low key slice (population sets val==key) */
          table.cooperative_insert<true>(kloc, 2, k0, tile, allocator,
                                         /*update_if_exists=*/true);
        } else if (ty == POP_NOOP) {
          if (tile.thread_rank() == 0) d_out[r] = k0;   /* diag: skip probe */
        } else {
          uint32_t v = table.cooperative_find<true, true>(kloc, 2, tile, allocator);
          if (tile.thread_rank() == 0) d_out[r] = v;
        }
      }
      /* ---- completion: last arriver posts the generation's tag byte.
       * Each participant system-fences (its d_out writes become host-
       * ordered for a later readback) then bumps the generation counter;
       * whoever sees nact resets it and posts the single done byte. ---- */
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence_system();
        if (atomicAdd(&g_arrive[gen].n, 1u) + 1 == nact) {
          g_arrive[gen].n = 0;       /* safe: no writer until gen+WINDOW_K */
          __threadfence_system();    /* reset ordered before done post */
          doneb[(size_t)gen * 64] = done_tag(w);
        }
      }
    }
  }
}

/* ------------------------------ host side ------------------------------ */
struct PersistEngine2 {
  unsigned long long *g_ring;        /* managed, preferred HBM; RING_K words */
  Mailbox     *mbox;                 /* device; grid*NSLOT lines */
  unsigned long long *disp_k;        /* device; per-block assignment ordinal */
  uint8_t     *h_doneb, *d_doneb;    /* mapped host, WINDOW_K 64B lines */
  ArriveCtr   *g_arrive;             /* device, WINDOW_K padded counters */
  uint8_t     *d_types;              /* managed, preferred HBM */
  uint32_t    *d_keys;               /* managed, preferred HBM */
  uint32_t    *d_out;                /* managed, preferred HBM (host-readable) */
  cudaStream_t kstream, cstream;     /* NON-BLOCKING (toy lesson 3) */
  unsigned long long cur_batch = 0;
  unsigned long long posted[WINDOW_K];        /* packed word per generation */
  int          grid = 0;             /* total blocks incl. dispatcher */
  uint32_t     sgrid = 0;            /* serving blocks = grid - 1 */
};

/* Effective pipeline depth for batches of nact blocks: a block's slot is
 * reused every NSLOT of its assignments, which are >= floor(sgrid/nact)
 * batches apart, so at most NSLOT*floor(sgrid/nact) batches may be in
 * flight before an unconsumed slot could be overwritten. */
static inline uint32_t window_for(uint32_t count, uint32_t sgrid) {
  uint32_t nact = active_blocks(count, sgrid);
  unsigned long long lim = (unsigned long long)NSLOT * (sgrid / nact);
  if (lim < 1) lim = 1;
  return (uint32_t)(lim < HOST_WINDOW ? lim : HOST_WINDOW);
}

static void *managed_hbm(size_t bytes) {
  void *p;
  CK(cudaMallocManaged(&p, bytes));
  int dev; CK(cudaGetDevice(&dev));
  cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, dev);
  cudaMemAdvise(p, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
  return p;
}

/* Allocation only — safe to call before the launch-mode phase. */
static void persist2_alloc(PersistEngine2 &p) {
  /* grid size decided up front (occupancy query launches nothing) */
  int bpm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &bpm, rgi_persistent2_kernel, BLOCK_SIZE, 0));
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  p.grid = bpm * sms;

  p.sgrid = (uint32_t)p.grid - NDISP;
  p.g_ring = (unsigned long long *)managed_hbm(RING_K * sizeof(unsigned long long));
  memset((void *)p.g_ring, 0, RING_K * sizeof(unsigned long long));
  CK(cudaMalloc(&p.mbox, (size_t)p.grid * NSLOT * sizeof(Mailbox)));
  CK(cudaMemset(p.mbox, 0, (size_t)p.grid * NSLOT * sizeof(Mailbox)));
  CK(cudaMalloc(&p.disp_k, (size_t)p.grid * sizeof(unsigned long long)));
  CK(cudaMemset(p.disp_k, 0, (size_t)p.grid * sizeof(unsigned long long)));
  p.d_types = (uint8_t  *)managed_hbm(PMAX_BATCH);
  p.d_keys  = (uint32_t *)managed_hbm(sizeof(uint32_t) * 2 * (size_t)PMAX_BATCH);
  memset(p.d_types, POP_FIND, PMAX_BATCH);

  const size_t done_bytes = (size_t)WINDOW_K * 64;
  CK(cudaHostAlloc((void **)&p.h_doneb, done_bytes, cudaHostAllocMapped));
  memset(p.h_doneb, 0, done_bytes);
  CK(cudaHostGetDevicePointer((void **)&p.d_doneb, p.h_doneb, 0));
  CK(cudaMalloc(&p.g_arrive, WINDOW_K * sizeof(ArriveCtr)));
  CK(cudaMemset(p.g_arrive, 0, WINDOW_K * sizeof(ArriveCtr)));

  /* results in HBM like the launch path (the CPU baseline reduces its
   * results locally too — remote materialization would be asymmetric);
   * host reads them over C2C for validation */
  p.d_out = (uint32_t *)managed_hbm(sizeof(uint32_t) * (size_t)PMAX_BATCH);

  CK(cudaStreamCreateWithFlags(&p.kstream, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&p.cstream, cudaStreamNonBlocking));
}

/* Launch the resident grid — AFTER this no other kernel can run. */
static void persist2_launch_grid(PersistEngine2 &p, table_t &table, slab_t &ha) {
  /* request arrays are prefilled by now; make sure they are HBM-resident */
  CK(cudaMemPrefetchAsync(p.d_types, PMAX_BATCH, 0, p.cstream));
  CK(cudaMemPrefetchAsync(p.d_keys, sizeof(uint32_t) * 2 * (size_t)PMAX_BATCH, 0, p.cstream));
  CK(cudaMemPrefetchAsync((void *)p.g_ring, RING_K * sizeof(unsigned long long), 0, p.cstream));
  CK(cudaStreamSynchronize(p.cstream));

  printf("persistent v2: %d blocks (%u dispatchers + %u serving, %u tiles), "
         "%u slots/block, window<=%u\n",
         p.grid, NDISP, p.sgrid, p.sgrid * TILES_PER_BLOCK, NSLOT, HOST_WINDOW);

  rgi_persistent2_kernel<<<p.grid, BLOCK_SIZE, 0, p.kstream>>>(
      table, ha.get_device_instance(), p.g_ring, p.mbox, p.disp_k,
      p.d_doneb, p.g_arrive, p.d_types, p.d_keys, p.d_out);
  CK(cudaGetLastError());
  /* WDDM flush (toy lesson 1); harmless no-op cost on Linux */
  cudaEvent_t ev;
  CK(cudaEventCreate(&ev));
  CK(cudaEventRecord(ev, p.kstream));
  cudaEventQuery(ev);
  CK(cudaEventDestroy(ev));
}

/* Post one batch: a single 8-byte store into the HBM ring. NO cuda calls. */
static inline void persist2_post(PersistEngine2 &p, uint32_t count, uint32_t offset) {
  p.cur_batch += 1;
  const unsigned long long w = pub_pack(p.cur_batch, count, offset);
  p.posted[p.cur_batch % WINDOW_K] = w;
  std::atomic_thread_fence(std::memory_order_release);
  ((volatile unsigned long long *)p.g_ring)[p.cur_batch % RING_K] = w;
}

/* Wait for batch b: ONE local read of its generation's done byte. */
static inline void persist2_wait(PersistEngine2 &p, unsigned long long b) {
  const uint8_t tag = done_tag(p.posted[b % WINDOW_K]);
  volatile uint8_t *db = p.h_doneb + (size_t)(b % WINDOW_K) * 64;
  while (*db != tag) { }             /* tight spin (pinned thread) */
  std::atomic_thread_fence(std::memory_order_acquire);
}

/* Synchronous rendezvous = post + wait. */
static inline void persist2_submit(PersistEngine2 &p, uint32_t count, uint32_t offset) {
  persist2_post(p, count, offset);
  persist2_wait(p, p.cur_batch);
}

static void persist2_stop(PersistEngine2 &p) {
  std::atomic_thread_fence(std::memory_order_release);
  for (uint32_t s = 0; s < RING_K; ++s)
    ((volatile unsigned long long *)p.g_ring)[s] = ~0ull;
  CK(cudaStreamSynchronize(p.kstream));
}

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/* rotating offset: successive batches touch different keys (defeats L2) */
static inline void advance_offset(uint32_t &off, uint32_t B) {
  off += B;
  if (off + B > PMAX_BATCH) off = 0;
}

/* One pipelined measurement: `iters` batches of size B, returns ns/op. */
static double pipe_rep(PersistEngine2 &p, uint32_t B, int iters) {
  const uint32_t W = window_for(B, p.sgrid);
  uint32_t off = 0;
  for (int it = 0; it < (int)W; ++it) {          /* fill the window (warm) */
    persist2_post(p, B, off); advance_offset(off, B);
    if (p.cur_batch > W) persist2_wait(p, p.cur_batch - W);
  }
  for (unsigned long long b = p.cur_batch - (W - 1); b <= p.cur_batch; b++)
    persist2_wait(p, b);
  double t0 = now_s();
  for (int it = 0; it < iters; ++it) {
    persist2_post(p, B, off); advance_offset(off, B);
    if (p.cur_batch > W) persist2_wait(p, p.cur_batch - W);
  }
  for (unsigned long long b = p.cur_batch - (W - 1); b <= p.cur_batch; b++)
    persist2_wait(p, b);
  return (now_s() - t0) / ((double)iters * B) * 1e9;
}

/* -------------------------------- bench -------------------------------- */
int main(int argc, char **argv) {
  uint32_t N = (argc > 1) ? (uint32_t)strtoul(argv[1], 0, 10) : 67108864u; /* 64M */
  const bool cross_mode = (argc > 2) && (strcmp(argv[2], "cross") == 0);

#ifdef __linux__
  /* pin the submitting thread: measured worth ~0.5 us on the doorbell
   * (unpinned spinning on 64-core Grace adds scheduler jitter). NOT core
   * 0 — kernel housekeeping and IRQs land there and preempt the spin
   * (bimodal reps observed: 0.78 vs 3+ ns/op). Core 32 is quiet. */
  cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(32, &cs);
  sched_setaffinity(0, sizeof(cs), &cs);
#endif
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));

  slab_t ha(0.4f);
  debra_t hr;
  table_t table(ha, hr, (std::size_t)N, 2.0f);

  /* ---- populate via the LAUNCH path (must precede the resident grid) ---- */
  {
    std::vector<uint64_t> keys(N);
    std::vector<uint32_t> vals(N);
    for (uint32_t i = 0; i < N; i++) { keys[i] = (uint64_t)i + 1; vals[i] = i + 1; }
    uint32_t *dk, *dv;
    CK(cudaMalloc(&dk, sizeof(uint32_t) * 2 * (size_t)N));
    CK(cudaMalloc(&dv, sizeof(uint32_t) * (size_t)N));
    CK(cudaMemcpy(dk, keys.data(), sizeof(uint64_t) * (size_t)N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, vals.data(), sizeof(uint32_t) * (size_t)N, cudaMemcpyHostToDevice));
    table.insert<true>(dk, 2, nullptr, dv, N, 0, true);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(dk)); CK(cudaFree(dv));
    printf("populated %u keys (val == low key slice)\n", N);
  }

  /* ---- workload: 1M uniform-random keys in [1,N], prefilled ONCE.
   * Host writes cross C2C into HBM; excluded from the timed loop
   * (symmetric: the CPU baseline reads prewritten local-DRAM arrays). ---- */
  PersistEngine2 p;
  persist2_alloc(p);
  {
    std::mt19937_64 rng(0x5eed5eedULL);
    std::uniform_int_distribution<uint32_t> dist(1u, N);
    for (uint32_t i = 0; i < PMAX_BATCH; i++) {
      uint64_t k = (uint64_t)dist(rng);          /* 2-slice key, high slice 0 */
      p.d_keys[2 * i]     = (uint32_t)k;
      p.d_keys[2 * i + 1] = (uint32_t)(k >> 32);
    }
    printf("prefilled %u uniform-random request keys in [1, %u]\n", PMAX_BATCH, N);
  }

  const uint32_t batches[] = {64, 256, 512, 768, 1024, 2048, 4096, 8192, 16384,
                              32768, 65536, 131072, 262144, 1048576};
  const int NB = sizeof(batches) / sizeof(batches[0]);
  double launch_us[NB], sync_us[NB], pipe_ns[NB];

  /* ---- phase 1: LAUNCH-mode baseline on the SAME random request stream
   *      (must run BEFORE the resident grid — it blocks all launches) ---- */
  if (!cross_mode) {
    uint32_t *dout;
    CK(cudaMalloc(&dout, sizeof(uint32_t) * (size_t)PMAX_BATCH));
    for (int bi = 0; bi < NB; bi++) {
      uint32_t B = batches[bi];
      int iters = (B <= 8192) ? 200 : (B <= 32768 ? 100 : 20);
      uint32_t off = 0;
      table.find<false, true>(p.d_keys, 2, nullptr, dout, B);
      CK(cudaDeviceSynchronize());
      double t0 = now_s();
      for (int it = 0; it < iters; ++it) {
        table.find<false, true>(p.d_keys + 2ull * off, 2, nullptr, dout, B);
        advance_offset(off, B);
      }
      CK(cudaDeviceSynchronize());
      launch_us[bi] = (now_s() - t0) / iters * 1e6;
    }
    CK(cudaFree(dout));
    printf("launch-mode baseline done\n");
  }

  /* ---- phase 2: start the resident grid ---- */
  persist2_launch_grid(p, table, ha);

  /* ---- validation (before timing) ----
   * Random keys are all in [1,N] and population set val == low key slice,
   * so a correct find returns the key itself. Validate offset 0, an
   * offset-wrap window, and the INSERT branch. */
  {
    uint32_t bad = 0, M = 4096;
    persist2_submit(p, M, 0);
    for (uint32_t i = 0; i < M; i++)
      if (p.d_out[i] != p.d_keys[2 * i]) bad++;
    printf("v2 FIND validate (offset 0):    %u/%u correct\n", M - bad, M);
    if (bad) { fprintf(stderr, "VALIDATION FAILED\n"); persist2_stop(p); return 1; }

    uint32_t off2 = PMAX_BATCH - M;              /* last window before wrap */
    persist2_submit(p, M, off2);
    bad = 0;
    for (uint32_t i = 0; i < M; i++)
      if (p.d_out[i] != p.d_keys[2 * (off2 + i)]) bad++;
    printf("v2 FIND validate (offset wrap): %u/%u correct\n", M - bad, M);
    if (bad) { fprintf(stderr, "VALIDATION FAILED\n"); persist2_stop(p); return 1; }

    /* INSERT path: update 1024 existing keys through the doorbell (insert
     * with update_if_exists rewrites val = low slice — idempotent), then
     * re-find them. Host writes types over C2C; post() fences before ring. */
    memset(p.d_types, POP_INSERT, 1024);
    persist2_submit(p, 1024, 0);
    memset(p.d_types, POP_FIND, 1024);
    persist2_submit(p, 1024, 0);
    bad = 0;
    for (uint32_t i = 0; i < 1024; i++)
      if (p.d_out[i] != p.d_keys[2 * i]) bad++;
    printf("v2 INSERT+FIND validate:        %u/%u correct\n", 1024 - bad, 1024);
    if (bad) { fprintf(stderr, "VALIDATION FAILED\n"); persist2_stop(p); return 1; }

    /* pipelined-path validation: post a W-deep window of FIND batches over
     * DISTINCT offsets, wait them all, check the last one's outs. Batches
     * share out[] indexing by r, so check via a final sync batch. */
    for (uint32_t i = 0; i < 8; i++) persist2_post(p, 512, 512 * i);
    for (unsigned long long b = p.cur_batch - 7; b <= p.cur_batch; b++)
      persist2_wait(p, b);
    persist2_submit(p, 512, 512 * 9);
    bad = 0;
    for (uint32_t i = 0; i < 512; i++)
      if (p.d_out[i] != p.d_keys[2 * (512 * 9 + i)]) bad++;
    printf("v2 pipelined validate:          %u/%u correct\n", 512 - bad, 512);
    if (bad) { fprintf(stderr, "VALIDATION FAILED\n"); persist2_stop(p); return 1; }
  }

  /* ---- CROSS MODE: statistical resolution of the pipelined crossover.
   * R independent repetitions per B (each its own warm window fill +
   * timed run of `iters` batches), interleaved round-robin across the B
   * values so slow drift (clocks, thermals) spreads over all points
   * rather than biasing one. Reports mean/stddev/min/max ns/op. ---- */
  if (cross_mode) {
    const uint32_t cbat[] = {256, 384, 512, 640, 768, 1024, 1536, 2048};
    const int NC = sizeof(cbat) / sizeof(cbat[0]);
    const int R = 20, ITERS = 50000;
    static double rep[8][20];
    for (int r = 0; r < R; ++r)
      for (int ci = 0; ci < NC; ci++)
        rep[ci][r] = pipe_rep(p, cbat[ci], ITERS);
    persist2_stop(p);

    const double GRACE64_NS = 0.842;
    printf("\ncrossover region, pipelined, %d reps x %d batches each "
           "[measured GH200]\n", R, ITERS);
    printf("%-7s %-10s %-10s %-10s %-10s %-10s %-9s %s\n",
           "B", "mean ns/op", "median", "stddev", "min", "max", "reps<CPU",
           "verdict vs 64T Grace (0.842 ns/op)");
    for (int ci = 0; ci < NC; ci++) {
      double s = 0, s2 = 0, mn = 1e9, mx = 0;
      int wins = 0;
      double v[20];
      for (int r = 0; r < R; ++r) {
        v[r] = rep[ci][r];
        s += v[r]; s2 += v[r] * v[r];
        if (v[r] < mn) mn = v[r];
        if (v[r] > mx) mx = v[r];
        if (v[r] < GRACE64_NS) wins++;
      }
      std::sort(v, v + R);
      double med = (v[R / 2 - 1] + v[R / 2]) / 2;
      double mean = s / R;
      double sd = sqrt(s2 / R - mean * mean);
      printf("%-7u %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %2d/%-6d %s\n",
             cbat[ci], mean, med, sd, mn, mx, wins, R,
             (mean + 2 * sd < GRACE64_NS) ? "BEATS (mean+2sd below)" :
             (med < GRACE64_NS)           ? "beats (median below)" : "-");
    }
    printf("\ntotal ops per point: %d x %d x B; every rep is an independent\n"
           "window-fill + timed run; B values interleaved across reps.\n", R, ITERS);
    fflush(stdout);
    _exit(0);
  }

  /* ---- protocol floor: count=0 rendezvous (doorbell + done, one block,
   * zero table work) — isolates dispatch from probe cost ---- */
  {
    const int FI = 2000;
    for (int i = 0; i < 200; i++) persist2_submit(p, 0, 0);
    double t0 = now_s();
    for (int i = 0; i < FI; i++) persist2_submit(p, 0, 0);
    printf("protocol floor (count=0 sync rendezvous): %.2f us\n",
           (now_s() - t0) / FI * 1e6);
    /* pipelined floor: dispatch cadence with W in flight, zero work —
     * the serial-stage throughput of the dispatch path itself */
    const uint32_t WF = window_for(0, p.sgrid);
    for (int i = 0; i < 200; i++) {
      persist2_post(p, 0, 0);
      if (p.cur_batch > WF) persist2_wait(p, p.cur_batch - WF);
    }
    for (unsigned long long b = p.cur_batch - (WF - 1); b <= p.cur_batch; b++)
      persist2_wait(p, b);
    t0 = now_s();
    for (int i = 0; i < FI; i++) {
      persist2_post(p, 0, 0);
      if (p.cur_batch > WF) persist2_wait(p, p.cur_batch - WF);
    }
    for (unsigned long long b = p.cur_batch - (WF - 1); b <= p.cur_batch; b++)
      persist2_wait(p, b);
    printf("protocol floor (count=0 pipelined, W=%u): %.2f us/batch\n",
           WF, (now_s() - t0) / FI * 1e6);
  }

  /* ---- phase 3a: SYNC sweep (W=1): per-batch rendezvous latency ---- */
  for (int bi = 0; bi < NB; bi++) {
    uint32_t B = batches[bi];
    int iters = (B <= 8192) ? 400 : (B <= 32768 ? 100 : 20);
    uint32_t off = 0;
    persist2_submit(p, B, off); advance_offset(off, B);   /* warm */
    double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
      persist2_submit(p, B, off);
      advance_offset(off, B);
    }
    sync_us[bi] = (now_s() - t0) / iters * 1e6;
  }

  /* ---- diagnostic: pipelined NOOP sweep (dispatch+wake+out+completion,
   * zero probes) — separates protocol cadence from probe throughput ---- */
  {
    memset(p.d_types, POP_NOOP, PMAX_BATCH);
    printf("\npipelined NOOP cadence (no probes):\n");
    for (int bi = 0; bi < NB; bi++) {
      uint32_t B = batches[bi];
      const uint32_t W = window_for(B, p.sgrid);
      int iters = (B <= 8192) ? 2000 : 200;
      uint32_t off = 0;
      for (int it = 0; it < (int)W; ++it) {
        persist2_post(p, B, off); advance_offset(off, B);
        if (p.cur_batch > W) persist2_wait(p, p.cur_batch - W);
      }
      for (unsigned long long b = p.cur_batch - (W - 1); b <= p.cur_batch; b++)
        persist2_wait(p, b);
      double t0 = now_s();
      for (int it = 0; it < iters; ++it) {
        persist2_post(p, B, off); advance_offset(off, B);
        if (p.cur_batch > W) persist2_wait(p, p.cur_batch - W);
      }
      for (unsigned long long b = p.cur_batch - (W - 1); b <= p.cur_batch; b++)
        persist2_wait(p, b);
      double us = (now_s() - t0) / iters * 1e6;
      printf("  B=%-8u %8.2f us/batch  %8.3f ns/op\n", B, us, us * 1e3 / B);
    }
    memset(p.d_types, POP_FIND, PMAX_BATCH);
  }

  /* ---- phase 3b: PIPELINED sweep: peak-serving throughput. Batches
   * overlap on disjoint block sets; amortized ns/op is the number the
   * fig16 model line describes. Window depth adapts to B (slot reuse). ---- */
  for (int bi = 0; bi < NB; bi++) {
    uint32_t B = batches[bi];
    const uint32_t W = window_for(B, p.sgrid);
    int iters = (B <= 8192) ? 2000 : (B <= 65536 ? 400 : 40);
    uint32_t off = 0;
    for (int it = 0; it < 16; ++it) {            /* warm */
      persist2_post(p, B, off); advance_offset(off, B);
      if (p.cur_batch > W) persist2_wait(p, p.cur_batch - W);
    }
    for (unsigned long long b = p.cur_batch - (W - 1); b <= p.cur_batch; b++)
      persist2_wait(p, b);
    double t0 = now_s();
    for (int it = 0; it < iters; ++it) {
      persist2_post(p, B, off); advance_offset(off, B);
      if (p.cur_batch > W) persist2_wait(p, p.cur_batch - W);
    }
    for (unsigned long long b = p.cur_batch - (W - 1); b <= p.cur_batch; b++)
      persist2_wait(p, b);
    pipe_ns[bi] = (now_s() - t0) / ((double)iters * B) * 1e9;
  }
  persist2_stop(p);

  /* ---- report ---- */
  const double GRACE64_NS = 0.842;   /* [measured] 64T Grace, 1.6GB table, 2026-07-07 */
  const double GRACE1T_NS = 23.3;    /* [measured] 1T Grace, same table */
  printf("\nuniform-random FIND against %u-key table [measured GH200]\n", N);
  printf("%-9s %-12s %-11s %-12s %-13s %-14s %s\n",
         "B", "launch(us)", "sync(us)", "sync ns/op", "pipe ns/op", "pipe Mop/s",
         "vs Grace64T(0.842) [pipe]");
  int cross64s = -1, cross64p = -1, cross1s = -1;
  for (int bi = 0; bi < NB; bi++) {
    uint32_t B = batches[bi];
    double s_nsop = sync_us[bi] * 1e3 / B;
    if (cross64s < 0 && s_nsop < GRACE64_NS) cross64s = bi;
    if (cross1s  < 0 && s_nsop < GRACE1T_NS) cross1s = bi;
    if (cross64p < 0 && pipe_ns[bi] < GRACE64_NS) cross64p = bi;
    printf("%-9u %-12.2f %-11.2f %-12.3f %-13.3f %-14.1f %s\n", B,
           launch_us[bi], sync_us[bi], s_nsop, pipe_ns[bi], 1e3 / pipe_ns[bi],
           pipe_ns[bi] < GRACE64_NS ? "BEATS-64T" : "");
  }
  printf("\n[measured end-to-end] crossovers:\n");
  if (cross1s >= 0)
    printf("  sync vs 1T Grace (23.3 ns/op):        B = %u\n", batches[cross1s]);
  if (cross64s >= 0)
    printf("  sync vs 64T Grace (0.842 ns/op):      B = %u\n", batches[cross64s]);
  else
    printf("  sync vs 64T Grace (0.842 ns/op):      none in sweep\n");
  if (cross64p >= 0)
    printf("  pipelined(W=%u) vs 64T Grace (0.842):  B = %u\n", HOST_WINDOW, batches[cross64p]);
  else
    printf("  pipelined(W=%u) vs 64T Grace (0.842):  none in sweep\n", HOST_WINDOW);
  printf("\ntimed = doorbell + probes + completion; payload prewritten (symmetric\n"
         "with the CPU baseline). sync = one batch in flight (latency); pipe =\n"
         "up to %u batches in flight on disjoint block sets (peak serving).\n",
         HOST_WINDOW);
  /* The resident kernel has exited (persist2_stop synced), but CUDA context
   * teardown was observed to hang once on this stack with the mapped+managed
   * mix, leaving a zombie at 100%% GPU — metering hazard. Results are out;
   * skip teardown and let the driver reclaim on process exit. */
  fflush(stdout);
  _exit(0);
}
