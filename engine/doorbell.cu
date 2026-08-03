/* doorbell.cu — bare CPU<->GPU doorbell round-trip microbenchmark (WP7 exp 1).
 *
 * Measures the primitive the whole thesis prices: host writes a 4-byte request
 * flag, a resident single-thread kernel spins on it and writes a 4-byte ack,
 * host spins on the ack. One iteration = one full CPU->GPU->CPU rendezvous.
 * No payload, no syncs, no sleep quantization — the floor itself.
 *
 * Allocation/placement modes (the GH200 question is WHERE the flags live):
 *   mapped  : both flags in cudaHostAllocMapped pinned host memory
 *             (what the toy engine uses today)
 *   sysmem  : both flags in plain malloc() system memory
 *             (Grace-Hopper ATS coherent access; invalid on x86+dGPU)
 *   managed : both flags in cudaMallocManaged
 *   split   : req flag preferred-located on the GPU (its consumer),
 *             ack flag preferred-located on the host (its consumer),
 *             via managed memory + cudaMemAdvise — the consumer-side
 *             placement the GH200 ping-pong literature recommends.
 *
 * Usage: doorbell <mode> [iters]      (default 20000 iters, 2000 warmup)
 * Output: one line  MODE=<m> ITERS=<n> MEAN_RT_US=<x>
 *
 * Build: nvcc -O3 -arch=sm_90 doorbell.cu -o doorbell
 */
#include <cuda_runtime.h>
#include <cuda/std/atomic>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#ifdef __linux__
#include <sched.h>
#endif

#define CK(call)                                                              \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                        \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

__global__ void pong_kernel(volatile unsigned *req, volatile unsigned *ack,
                            unsigned iters) {
  for (unsigned i = 1; i <= iters; ++i) {
    while (*req < i) { }          /* tight spin: honest poll cost, no sleep */
    __threadfence_system();
    *ack = i;
    __threadfence_system();
  }
}

/* Replication of Fusco et al. (GH200 characterization) Fig. 13 ping-pong:
 * ONE single-byte system-scope atomic flag, CAS both sides. PONG=0, PING=1.
 * The device thread flips PING->PONG; the host flips PONG->PING. One full
 * exchange = both flips. Their measured Grace0<->Hopper0, flag in LPDDR:
 * 833 ns per full exchange. */
using sysflag = cuda::std::atomic<unsigned char>;

/* Orders match gh_benchmark (github.com/luigifusco/gh_benchmark,
 * src/atomic_benchmarks.cuh): relaxed on success AND failure. */
__global__ void pong_cas_kernel(sysflag *f, unsigned iters) {
  for (unsigned i = 0; i < iters; ++i) {
    unsigned char expected = 1;                       /* PING */
    while (!f->compare_exchange_strong(expected, 0,   /* -> PONG */
                                       cuda::std::memory_order_relaxed,
                                       cuda::std::memory_order_relaxed)) {
      expected = 1;
    }
  }
}

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  const char *mode = (argc > 1) ? argv[1] : "mapped";
  unsigned iters   = (argc > 2) ? (unsigned)strtoul(argv[2], 0, 10) : 20000u;
  const unsigned warm = 2000;
  const unsigned total = iters + warm;

  volatile unsigned *h_req = nullptr, *h_ack = nullptr;  /* host-side views  */
  unsigned *d_req = nullptr, *d_ack = nullptr;           /* device-side views */

  /* ---- Fusco et al. replication modes: single CAS flag ---- */
  if (strncmp(mode, "cas_", 4) == 0) {
#ifdef __linux__
    /* gh_benchmark pins the host thread (pthread_setaffinity_np); match it. */
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
    sysflag *hf = nullptr, *df = nullptr;
    if (strcmp(mode, "cas_ddr") == 0) {          /* flag in Grace LPDDR5X */
      void *buf = aligned_alloc(256, 256);
      memset(buf, 0, 256);                        /* first-touch on host */
      hf = new (buf) sysflag(0);
      df = hf;                                    /* ATS: same pointer */
    } else if (strcmp(mode, "cas_hbm") == 0) {   /* flag in Hopper HBM3 */
      void *buf;
      CK(cudaMallocManaged(&buf, 256));
      int dev; CK(cudaGetDevice(&dev));
      cudaMemAdvise(buf, 256, cudaMemAdviseSetPreferredLocation, dev);
      cudaMemAdvise(buf, 256, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
      hf = new (buf) sysflag(0);
      df = hf;
    } else if (strcmp(mode, "cas_mapped") == 0) { /* pinned host, mapped */
      void *buf;
      CK(cudaSetDeviceFlags(cudaDeviceMapHost));
      CK(cudaHostAlloc(&buf, 256, cudaHostAllocMapped));
      memset(buf, 0, 256);
      hf = new (buf) sysflag(0);
      void *dbuf; CK(cudaHostGetDevicePointer(&dbuf, buf, 0));
      df = (sysflag *)dbuf;
    } else {
      fprintf(stderr, "unknown cas mode %s (cas_ddr|cas_hbm|cas_mapped)\n", mode);
      return 2;
    }
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    pong_cas_kernel<<<1, 1, 0, st>>>(df, total);
    CK(cudaGetLastError());
    cudaEvent_t ev; CK(cudaEventCreate(&ev));
    CK(cudaEventRecord(ev, st)); cudaEventQuery(ev); CK(cudaEventDestroy(ev));

    auto host_ping = [&](unsigned n) {
      for (unsigned i = 0; i < n; ++i) {
        unsigned char e = 0;                      /* PONG -> PING */
        while (!hf->compare_exchange_strong(e, 1,
                                            cuda::std::memory_order_relaxed,
                                            cuda::std::memory_order_relaxed))
          e = 0;
      }
    };
    host_ping(warm);
    double t0 = now_s();
    host_ping(iters);
    double t1 = now_s();
    CK(cudaStreamSynchronize(st));
    printf("MODE=%-10s ITERS=%u MEAN_EXCHANGE_US=%.3f  (Fusco Fig.13 G0-H0 ref: 0.833 us, flag in LPDDR)\n",
           mode, iters, (t1 - t0) / iters * 1e6);
    return 0;
  }

  if (strcmp(mode, "mapped") == 0) {
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    unsigned *p, *q;
    CK(cudaHostAlloc(&p, 128, cudaHostAllocMapped));   /* separate lines */
    CK(cudaHostAlloc(&q, 128, cudaHostAllocMapped));
    *p = 0; *q = 0;
    CK(cudaHostGetDevicePointer((void **)&d_req, p, 0));
    CK(cudaHostGetDevicePointer((void **)&d_ack, q, 0));
    h_req = p; h_ack = q;
  } else if (strcmp(mode, "sysmem") == 0) {
    /* plain malloc: requires Grace-Hopper style full coherence (ATS). */
    unsigned *p = (unsigned *)aligned_alloc(128, 128);
    unsigned *q = (unsigned *)aligned_alloc(128, 128);
    *p = 0; *q = 0;
    h_req = p; h_ack = q; d_req = p; d_ack = q;
  } else if (strcmp(mode, "managed") == 0 || strcmp(mode, "split") == 0) {
    unsigned *p, *q;
    CK(cudaMallocManaged(&p, 128));
    CK(cudaMallocManaged(&q, 128));
    *p = 0; *q = 0;
    if (strcmp(mode, "split") == 0) {
      int dev; CK(cudaGetDevice(&dev));
      /* req consumed by GPU -> prefer GPU; ack consumed by host -> prefer CPU */
      cudaMemAdvise(p, 128, cudaMemAdviseSetPreferredLocation, dev);
      cudaMemAdvise(q, 128, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
      cudaMemAdvise(p, 128, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
      cudaMemAdvise(q, 128, cudaMemAdviseSetAccessedBy, dev);
    }
    h_req = p; h_ack = q; d_req = p; d_ack = q;
  } else {
    fprintf(stderr, "unknown mode %s (mapped|sysmem|managed|split)\n", mode);
    return 2;
  }

  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  pong_kernel<<<1, 1, 0, st>>>(d_req, d_ack, total);
  CK(cudaGetLastError());
  /* flush launch (harmless on Linux, required on WDDM) */
  cudaEvent_t ev; CK(cudaEventCreate(&ev));
  CK(cudaEventRecord(ev, st)); cudaEventQuery(ev); CK(cudaEventDestroy(ev));

  /* warmup */
  for (unsigned i = 1; i <= warm; ++i) {
    std::atomic_thread_fence(std::memory_order_release);
    *h_req = i;
    while (*h_ack < i) { }
  }
  /* timed */
  double t0 = now_s();
  for (unsigned i = warm + 1; i <= total; ++i) {
    std::atomic_thread_fence(std::memory_order_release);
    *h_req = i;
    while (*h_ack < i) { }
  }
  double t1 = now_s();
  std::atomic_thread_fence(std::memory_order_acquire);

  CK(cudaStreamSynchronize(st));
  printf("MODE=%-8s ITERS=%u MEAN_RT_US=%.3f\n", mode, iters,
         (t1 - t0) / iters * 1e6);
  return 0;
}
