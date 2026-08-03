/* rgi_profile_one.cu - single-batch RGI find harness for Nsight Compute.
 *
 * Additive profiling helper: builds the same RGI chain table and runs one
 * chosen find batch repeatedly so ncu can collect a specific B without
 * launch-skip gymnastics through rgi_sweep.
 *
 * Build on GH200:
 *   nvcc -std=c++17 -arch=sm_90 --expt-extended-lambda --expt-relaxed-constexpr \
 *        -maxrregcount=64 -I ~/work/RobustGPUIndexing/include \
 *        engine/rgi_profile_one.cu -o ~/work/bin/rgi_profile_one
 */
#include <gpu_chainhashtable.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using slab_t  = simple_slab_allocator<128>;
using debra_t = simple_debra_reclaimer<>;
using table_t = GpuHashtable::gpu_chainhashtable<slab_t, debra_t, 16>;

#define CK(c) do { \
  cudaError_t e__ = (c); \
  if (e__ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e__)); \
    return 1; \
  } \
} while (0)

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  uint32_t N = (argc > 1) ? (uint32_t)strtoul(argv[1], nullptr, 10) : 67108864u;
  uint32_t B = (argc > 2) ? (uint32_t)strtoul(argv[2], nullptr, 10) : 1048576u;
  uint32_t iters = (argc > 3) ? (uint32_t)strtoul(argv[3], nullptr, 10) : 20u;
  if (B > N) B = N;

  slab_t ha(0.4f);
  debra_t hr;
  table_t table(ha, hr, (std::size_t)N, 2.0f);

  std::vector<uint64_t> keys(N);
  std::vector<uint32_t> vals(N);
  for (uint32_t i = 0; i < N; ++i) {
    keys[i] = (uint64_t)i + 1;
    vals[i] = i + 1;
  }

  uint32_t *dk = nullptr, *dv = nullptr, *dout = nullptr;
  CK(cudaMalloc(&dk, sizeof(uint32_t) * 2ull * N));
  CK(cudaMalloc(&dv, sizeof(uint32_t) * (uint64_t)N));
  CK(cudaMalloc(&dout, sizeof(uint32_t) * (uint64_t)N));
  CK(cudaMemcpy(dk, keys.data(), sizeof(uint64_t) * (uint64_t)N, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv, vals.data(), sizeof(uint32_t) * (uint64_t)N, cudaMemcpyHostToDevice));
  table.insert<true>(dk, 2, nullptr, dv, N, 0, true);
  CK(cudaDeviceSynchronize());

  table.find<false, true>(dk, 2, nullptr, dout, B);
  CK(cudaDeviceSynchronize());

  double t0 = now_s();
  for (uint32_t i = 0; i < iters; ++i) {
    table.find<false, true>(dk, 2, nullptr, dout, B);
  }
  CK(cudaDeviceSynchronize());
  double t1 = now_s();
  double us = (t1 - t0) * 1e6 / iters;
  printf("profile_one,N=%u,B=%u,iters=%u,latency_us=%.2f,Mops=%.1f\n",
         N, B, iters, us, B / (us / 1e6) / 1e6);

  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dout);
  return 0;
}
