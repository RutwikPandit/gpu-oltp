/* rgi_prof2.cu — parameterized single-op profiling target for the DRAM-BW schmoo.
 *   usage: rgi_prof2 <find|insert> <N>
 * For 'find': populate a 2M-key base, then ONE find(N). For 'insert': ONE
 * insert(N) into an empty table. ncu profiles the matching *_device_func. */
#include <gpu_chainhashtable.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using slab_t  = simple_slab_allocator<128>;
using debra_t = simple_debra_reclaimer<>;
using table_t = GpuHashtable::gpu_chainhashtable<slab_t, debra_t, 16>;
#define CK(c) do{cudaError_t e=(c); if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

int main(int argc, char** argv){
  const char* op = (argc>1)?argv[1]:"find";
  uint32_t N = (argc>2)?(uint32_t)strtoul(argv[2],0,10):1000000u;
  const uint32_t BASE = 2000000u;
  uint32_t cap = (N>BASE?N:BASE);
  slab_t ha(0.4f); debra_t hr; table_t table(ha,hr,(std::size_t)cap,2.0f);

  std::vector<uint64_t> keys(cap); std::vector<uint32_t> vals(cap);
  for(uint32_t i=0;i<cap;i++){keys[i]=(uint64_t)i+1; vals[i]=i+1;}
  uint32_t *dk,*dv,*dout;
  CK(cudaMalloc(&dk,sizeof(uint32_t)*2*cap)); CK(cudaMalloc(&dv,sizeof(uint32_t)*cap)); CK(cudaMalloc(&dout,sizeof(uint32_t)*cap));
  CK(cudaMemcpy(dk,keys.data(),sizeof(uint64_t)*cap,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv,vals.data(),sizeof(uint32_t)*cap,cudaMemcpyHostToDevice));

  if(strcmp(op,"find")==0){
    table.insert<true>(dk,2,nullptr,dv,BASE,0,true); CK(cudaDeviceSynchronize());  // base populate
    table.find<false,true>(dk,2,nullptr,dout,N);     CK(cudaDeviceSynchronize());  // MEASURED
  } else {
    table.insert<true>(dk,2,nullptr,dv,N,0,true);    CK(cudaDeviceSynchronize());  // MEASURED
  }
  cudaFree(dk);cudaFree(dv);cudaFree(dout);
  return 0;
}
