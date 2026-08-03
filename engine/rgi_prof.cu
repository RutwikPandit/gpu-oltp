/* rgi_prof.cu — minimal profiling target: insert N keys, then ONE find(N) into
 * an RGI chain hashtable. Two batch_kernel launches (insert, find) for ncu SOL
 * analysis of the OLTP index path. */
#include <gpu_chainhashtable.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>
#include <cstdint>
#include <cstdio>
#include <vector>

using slab_t  = simple_slab_allocator<128>;
using debra_t = simple_debra_reclaimer<>;
using table_t = GpuHashtable::gpu_chainhashtable<slab_t, debra_t, 16>;
#define CK(c) do{cudaError_t e=(c); if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

int main(int argc, char** argv){
  uint32_t N = (argc>1)?(uint32_t)strtoul(argv[1],0,10):500000u;
  slab_t ha(0.4f); debra_t hr; table_t table(ha,hr,(std::size_t)N,2.0f);
  std::vector<uint64_t> keys(N); std::vector<uint32_t> vals(N);
  for(uint32_t i=0;i<N;i++){keys[i]=(uint64_t)i+1; vals[i]=i+1;}
  uint32_t *dk,*dv,*dout;
  CK(cudaMalloc(&dk,sizeof(uint32_t)*2*N)); CK(cudaMalloc(&dv,sizeof(uint32_t)*N)); CK(cudaMalloc(&dout,sizeof(uint32_t)*N));
  CK(cudaMemcpy(dk,keys.data(),sizeof(uint64_t)*N,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv,vals.data(),sizeof(uint32_t)*N,cudaMemcpyHostToDevice));
  table.insert<true>(dk,2,nullptr,dv,N,0,true); CK(cudaDeviceSynchronize());   // batch_kernel #1
  table.find<false,true>(dk,2,nullptr,dout,N);  CK(cudaDeviceSynchronize());   // batch_kernel #2
  cudaFree(dk);cudaFree(dv);cudaFree(dout);
  return 0;
}
