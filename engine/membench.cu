/* membench.cu - GH200/Hopper memory characterization microbenchmarks.
 *
 * Additive benchmark for HANDOFF_PERF_NSIGHT.md:
 *   stream read/write/copy/triad, dependent pointer-chase latency, and
 *   random-access bandwidth at 32/64/128 B useful granules.
 *
 * Build on GH200:
 *   nvcc -O3 -std=c++17 -arch=sm_90 engine/membench.cu -o ~/work/bin/membench
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(call) do { \
  cudaError_t e__ = (call); \
  if (e__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e__)); \
    exit(1); \
  } \
} while (0)

static constexpr int THREADS = 1024;
static constexpr uint64_t DEFAULT_BYTES = 4ull << 30;

__device__ unsigned long long g_sink64;
__device__ float g_sinkf;

__global__ void stream_read_kernel(const ulonglong2 *__restrict__ a, size_t nvec) {
  ulonglong2 acc{0, 0};
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
    ulonglong2 v = a[i];
    acc.x ^= v.x;
    acc.y ^= v.y;
  }
  unsigned long long x = acc.x ^ acc.y;
  for (int off = 16; off; off >>= 1) x ^= __shfl_xor_sync(0xffffffff, x, off);
  if ((threadIdx.x & 31) == 0) atomicXor(&g_sink64, x);
}

__global__ void stream_write_kernel(ulonglong2 *__restrict__ a, size_t nvec, unsigned long long seed) {
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) {
    a[i] = make_ulonglong2(seed + i, seed ^ i);
  }
}

__global__ void stream_copy_kernel(ulonglong2 *__restrict__ dst, const ulonglong2 *__restrict__ src, size_t nvec) {
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nvec; i += stride) dst[i] = src[i];
}

__global__ void stream_triad_kernel(float4 *__restrict__ a, const float4 *__restrict__ b,
                                    const float4 *__restrict__ c, size_t n4, float s) {
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 bv = b[i], cv = c[i];
    a[i] = make_float4(bv.x + s * cv.x, bv.y + s * cv.y, bv.z + s * cv.z, bv.w + s * cv.w);
  }
}

__global__ void ptr_chase_kernel(const uint32_t *__restrict__ next, uint32_t iters, uint32_t *out) {
  uint32_t p = 0;
  unsigned long long t0 = clock64();
  for (uint32_t i = 0; i < iters; ++i) {
    asm volatile("" ::: "memory");
    p = next[p];
  }
  unsigned long long t1 = clock64();
  out[0] = p;
  reinterpret_cast<unsigned long long *>(out)[1] = t1 - t0;
}

template<int GRANULE>
__global__ void rand_bw_kernel(const uint8_t *__restrict__ buf, size_t slots, uint32_t iters) {
  uint64_t x = 0x9e3779b97f4a7c15ull ^ ((uint64_t)blockIdx.x << 32) ^ threadIdx.x;
  unsigned long long acc = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    x = x * 2862933555777941757ULL + 3037000493ULL;
    size_t slot = (x >> 16) % slots;
    const uint8_t *p = buf + slot * GRANULE;
    if constexpr (GRANULE == 32) {
      const ulonglong4 *q = reinterpret_cast<const ulonglong4 *>(p);
      ulonglong4 v = *q;
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    } else if constexpr (GRANULE == 64) {
      const ulonglong4 *q = reinterpret_cast<const ulonglong4 *>(p);
      ulonglong4 v0 = q[0], v1 = q[1];
      acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w;
    } else {
      const ulonglong4 *q = reinterpret_cast<const ulonglong4 *>(p);
      ulonglong4 v0 = q[0], v1 = q[1], v2 = q[2], v3 = q[3];
      acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w ^
             v2.x ^ v2.y ^ v2.z ^ v2.w ^ v3.x ^ v3.y ^ v3.z ^ v3.w;
    }
  }
  for (int off = 16; off; off >>= 1) acc ^= __shfl_xor_sync(0xffffffff, acc, off);
  if ((threadIdx.x & 31) == 0) atomicXor(&g_sink64, acc);
}

static double elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.0f;
  CK(cudaEventElapsedTime(&ms, a, b));
  return (double)ms;
}

template <typename F>
static double time_kernel(F launch, int reps = 10) {
  cudaEvent_t s, e;
  CK(cudaEventCreate(&s));
  CK(cudaEventCreate(&e));
  launch();
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(s));
  for (int i = 0; i < reps; ++i) launch();
  CK(cudaEventRecord(e));
  CK(cudaEventSynchronize(e));
  double ms = elapsed_ms(s, e) / reps;
  CK(cudaEventDestroy(s));
  CK(cudaEventDestroy(e));
  return ms;
}

static void fill_bytes(void *p, size_t n) {
  CK(cudaMemset(p, 0x5a, n));
}

static void run_stream(const std::string &mode, size_t bytes) {
  int blocks_list[] = {33, 66, 132, 264, 528, 1056};
  void *a = nullptr, *b = nullptr, *c = nullptr;
  CK(cudaMalloc(&a, bytes));
  CK(cudaMalloc(&b, bytes));
  CK(cudaMalloc(&c, bytes));
  fill_bytes(a, bytes);
  fill_bytes(b, bytes);
  fill_bytes(c, bytes);
  size_t nvec = bytes / sizeof(ulonglong2);
  size_t n4 = bytes / sizeof(float4);
  for (int blocks : blocks_list) {
    double ms = 0.0, moved = 0.0;
    if (mode == "read") {
      ms = time_kernel([&] { stream_read_kernel<<<blocks, THREADS>>>((const ulonglong2 *)a, nvec); });
      moved = (double)bytes;
    } else if (mode == "write") {
      ms = time_kernel([&] { stream_write_kernel<<<blocks, THREADS>>>((ulonglong2 *)a, nvec, 123); });
      moved = (double)bytes;
    } else if (mode == "copy") {
      ms = time_kernel([&] { stream_copy_kernel<<<blocks, THREADS>>>((ulonglong2 *)a, (const ulonglong2 *)b, nvec); });
      moved = 2.0 * (double)bytes;
    } else if (mode == "triad") {
      ms = time_kernel([&] { stream_triad_kernel<<<blocks, THREADS>>>((float4 *)a, (const float4 *)b, (const float4 *)c, n4, 1.25f); });
      moved = 3.0 * (double)bytes;
    }
    CK(cudaGetLastError());
    printf("mode=%s,bytes=%llu,blocks=%d,threads=%d,ms=%.3f,GBps=%.1f\n",
           mode.c_str(), (unsigned long long)bytes, blocks, THREADS, ms, moved / (ms * 1e6));
    fflush(stdout);
  }
  cudaFree(a); cudaFree(b); cudaFree(c);
}

static void make_perm(std::vector<uint32_t> &next) {
  std::vector<uint32_t> p(next.size());
  for (uint32_t i = 0; i < p.size(); ++i) p[i] = i;
  std::mt19937_64 rng(0xC2C20260707ull + next.size());
  std::shuffle(p.begin(), p.end(), rng);
  for (size_t i = 0; i < p.size(); ++i) next[p[i]] = p[(i + 1) % p.size()];
}

static void run_lat(size_t max_bytes) {
  uint32_t *d = nullptr, *out = nullptr;
  CK(cudaMalloc(&out, sizeof(uint64_t) * 2));
  for (size_t bytes = 16ull << 10; bytes <= max_bytes; bytes <<= 1) {
    size_t n = bytes / sizeof(uint32_t);
    std::vector<uint32_t> h(n);
    make_perm(h);
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    uint32_t iters = (uint32_t)std::min<size_t>(n * 4, 200000000ull);
    if (iters < 1000000u) iters = 1000000u;
    ptr_chase_kernel<<<1,1>>>(d, iters, out);
    CK(cudaDeviceSynchronize());
    uint64_t h_out[2] = {};
    CK(cudaMemcpy(h_out, out, sizeof(h_out), cudaMemcpyDeviceToHost));
    double khz = 0.0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    khz = (double)prop.clockRate;
    double ns = (double)h_out[1] / khz * 1000000.0 / iters;
    printf("mode=lat,bytes=%llu,iters=%u,cycles=%llu,ns_per_access=%.2f,last=%u\n",
           (unsigned long long)bytes, iters, (unsigned long long)h_out[1], ns, (uint32_t)h_out[0]);
    fflush(stdout);
    cudaFree(d);
  }
  cudaFree(out);
}

template<int GRANULE>
static void run_rand(size_t bytes) {
  int blocks_list[] = {33, 66, 132, 264, 528, 1056};
  uint8_t *buf = nullptr;
  CK(cudaMalloc(&buf, bytes));
  fill_bytes(buf, bytes);
  size_t slots = bytes / GRANULE;
  for (int blocks : blocks_list) {
    uint32_t iters = 256;
    double ms = time_kernel([&] { rand_bw_kernel<GRANULE><<<blocks, THREADS>>>(buf, slots, iters); }, 5);
    CK(cudaGetLastError());
    double useful = (double)blocks * THREADS * iters * GRANULE;
    printf("mode=rand%d,bytes=%llu,blocks=%d,threads=%d,iters=%u,ms=%.3f,GBps=%.1f\n",
           GRANULE, (unsigned long long)bytes, blocks, THREADS, iters, ms, useful / (ms * 1e6));
    fflush(stdout);
  }
  cudaFree(buf);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s read|write|copy|triad|lat|rand32|rand64|rand128 [bytes]\n", argv[0]);
    return 2;
  }
  std::string mode = argv[1];
  size_t bytes = argc > 2 ? strtoull(argv[2], nullptr, 0) : DEFAULT_BYTES;
  int dev = 0;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, dev));
  CK(cudaSetDevice(dev));
  printf("provenance=GH200_H100,device=\"%s\",cc=%d.%d,mode=%s,date=2026-07-07\n",
         p.name, p.major, p.minor, mode.c_str());
  if (mode == "read" || mode == "write" || mode == "copy" || mode == "triad") run_stream(mode, bytes);
  else if (mode == "lat") run_lat(bytes);
  else if (mode == "rand32") run_rand<32>(bytes);
  else if (mode == "rand64") run_rand<64>(bytes);
  else if (mode == "rand128") run_rand<128>(bytes);
  else {
    fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
  }
  return 0;
}
