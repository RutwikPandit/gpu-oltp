/* cpu_sweep.cpp — CPU baseline for the point-lookup latency/throughput curve.
 * A flat open-addressing hash index (uint64 key -> uint32 value), same class of
 * structure as the GPU index, queried single- and multi-threaded. Reports
 * per-op latency (single thread) and aggregate throughput vs thread count.
 * Build: g++ -O3 -march=native -fopenmp cpu_sweep.cpp -o cpu_sweep         */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

static inline uint64_t hash64(uint64_t x){ x^=x>>33; x*=0xff51afd7ed558ccdULL; x^=x>>33; x*=0xc4ceb9fe1a85ec53ULL; x^=x>>33; return x; }
static double now_s(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}

struct HT {
  std::vector<uint64_t> keys;   // EMPTY = 0
  std::vector<uint32_t> vals;
  uint64_t mask;
  HT(uint64_t cap){ uint64_t p=1; while(p<cap) p<<=1; keys.assign(p,0); vals.assign(p,0); mask=p-1; }
  void insert(uint64_t k, uint32_t v){ uint64_t h=hash64(k)&mask; while(keys[h]!=0){ if(keys[h]==k){vals[h]=v;return;} h=(h+1)&mask;} keys[h]=k; vals[h]=v; }
  inline uint32_t find(uint64_t k) const { uint64_t h=hash64(k)&mask; for(;;){ uint64_t cur=keys[h]; if(cur==k) return vals[h]; if(cur==0) return 0xFFFFFFFFu; h=(h+1)&mask;} }
};

int main(int argc, char** argv){
  uint32_t N = (argc>1)?(uint32_t)strtoul(argv[1],0,10):2000000u;
  uint64_t M = (argc>2)?strtoull(argv[2],0,10):40000000ull;   // total lookups
  HT ht((uint64_t)(N/0.5));   // load factor 0.5
  for(uint32_t i=0;i<N;i++) ht.insert((uint64_t)i+1, i+1);

  // lookup key stream (uniform random over inserted keys)
  std::mt19937_64 rng(12345);
  std::vector<uint64_t> q(M);
  for(uint64_t i=0;i<M;i++) q[i]=(rng()%N)+1;

  printf("CPU open-addressing HT: N=%u keys, table=%.0f MB, M=%llu lookups\n",
         N, double(ht.keys.size())*(8+4)/1e6, (unsigned long long)M);
  printf("%-8s %-16s %-14s\n","threads","throughput(Mop/s)","ns/op");

  int thread_counts[] = {1,2,4,8,16,32,64};
  int max_procs = omp_get_num_procs();
  for(int ti=0; ti<7; ++ti){
    int T = thread_counts[ti];
    if(T > max_procs) break;   /* skip thread counts beyond available cores */
    omp_set_num_threads(T);
    volatile uint64_t sink=0;
    // warmup
    #pragma omp parallel for reduction(+:sink) schedule(static)
    for(long long i=0;i<(long long)M;i++) sink += ht.find(q[i]);
    double t0=now_s();
    uint64_t s=0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for(long long i=0;i<(long long)M;i++) s += ht.find(q[i]);
    double t1=now_s();
    sink += s;
    double mops = M/(t1-t0)/1e6;
    double nsop = (t1-t0)/M*1e9 * T;   // per-core ns/op (latency proxy at T=1)
    printf("%-8d %-16.1f %-14.2f\n", T, mops, (T==1)?(t1-t0)/M*1e9:nsop);
    if(sink==0xdeadbeef) printf("");
  }
  return 0;
}
