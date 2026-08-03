/* rgi_sweep.cu — latency vs throughput tradeoff for batched RGI lookups.
 * Inserts N keys into an RGI chain hashtable, then issues point-lookup batches
 * of increasing size B and reports per-batch latency (us) and throughput
 * (Mop/s). Small B = low latency / low throughput; large B = high throughput /
 * higher latency. This is the dispatch tradeoff the C2C queue targets. */
#include <gpu_chainhashtable.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <chrono>

using slab_t  = simple_slab_allocator<128>;
using debra_t = simple_debra_reclaimer<>;
using table_t = GpuHashtable::gpu_chainhashtable<slab_t, debra_t, 16>;

#define CK(c) do{cudaError_t e=(c); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}}while(0)
static double now_s(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}

int main(int argc, char** argv){
  uint32_t N = (argc>1)?(uint32_t)strtoul(argv[1],0,10):2000000u;
  slab_t ha(0.4f); debra_t hr; table_t table(ha, hr, (std::size_t)N, 2.0f);

  std::vector<uint64_t> keys(N); std::vector<uint32_t> vals(N);
  for(uint32_t i=0;i<N;i++){keys[i]=(uint64_t)i+1; vals[i]=i+1;}
  uint32_t *dk,*dv,*dout;
  CK(cudaMalloc(&dk,sizeof(uint32_t)*2*N)); CK(cudaMalloc(&dv,sizeof(uint32_t)*N)); CK(cudaMalloc(&dout,sizeof(uint32_t)*N));
  CK(cudaMemcpy(dk,keys.data(),sizeof(uint64_t)*N,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv,vals.data(),sizeof(uint32_t)*N,cudaMemcpyHostToDevice));
  table.insert<true>(dk,2,nullptr,dv,N,0,true); CK(cudaDeviceSynchronize());

  printf("%-10s %-14s %-16s\n","batchB","latency(us)","throughput(Mop/s)");
  uint32_t batches[] = {1,8,64,512,4096,32768,262144,1048576};
  for(uint32_t bi=0; bi<sizeof(batches)/sizeof(batches[0]); ++bi){
    uint32_t B = batches[bi]; if(B>N) break;
    int iters = (B<=512)?2000:(B<=32768?200:20);
    /* warmup */
    table.find<false,true>(dk,2,nullptr,dout,B); CK(cudaDeviceSynchronize());
    double t0=now_s();
    for(int it=0; it<iters; ++it){ table.find<false,true>(dk,2,nullptr,dout,B); }
    CK(cudaDeviceSynchronize());
    double t1=now_s();
    double per_batch_us = (t1-t0)/iters*1e6;
    double mops = B/ (per_batch_us/1e6) /1e6;
    printf("%-10u %-14.2f %-16.1f\n", B, per_batch_us, mops);
  }
  cudaFree(dk);cudaFree(dv);cudaFree(dout);
  return 0;
}
