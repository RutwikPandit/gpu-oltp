/* rgi_oltp_engine.cu — batched GPU OLTP engine backed by RobustGPUIndexing.
 *
 * Storage = RGI GPUChainHashtable (warp-cooperative, var-len keys, concurrent
 * mixed ops, linearizable, with memory reclamation), driven through RGI's
 * batch_kernel "ballot queue". Writes are buffered host-side and flushed as one
 * batched launch -> the batch size is the latency/throughput knob.
 *
 * KV mapping: 8-byte key = two uint32 RGI key-slices (max_key_length=2);
 * value = uint32 row id (what a DB index stores). Values are truncated to 32b.
 *
 * Build shared lib:
 *   nvcc -std=c++17 -arch=sm_89 --expt-extended-lambda --expt-relaxed-constexpr \
 *        -maxrregcount=64 -Xcompiler -fPIC -shared -I<RGI>/include \
 *        rgi_oltp_engine.cu -o librgioltp.so
 */
#include "rgi_oltp_engine.h"

#include <gpu_chainhashtable.hpp>
#include <simple_slab_alloc.hpp>
#include <simple_debra_reclaim.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_set>

#define RGI_CK(call)                                                           \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                         \
      abort();                                                                 \
    }                                                                          \
  } while (0)

using slab_t  = simple_slab_allocator<128>;
using debra_t = simple_debra_reclaimer<>;
using table_t = GpuHashtable::gpu_chainhashtable<slab_t, debra_t, 16>;

static constexpr uint32_t BATCH_MAX = 1u << 20;   /* device staging capacity */
static constexpr uint32_t RGI_INVALID = 0xFFFFFFFFu;

struct RgiEngine {
  slab_t   *host_alloc;
  debra_t  *host_reclaim;
  table_t  *table;
  uint32_t *d_keys;   /* 2*BATCH_MAX uint32 */
  uint32_t *d_vals;   /* BATCH_MAX uint32   */
  uint32_t *d_out;    /* BATCH_MAX uint32   */
  /* host-side staging */
  std::vector<uint64_t> pend_k;   /* buffered insert/update keys   */
  std::vector<uint32_t> pend_v;   /* buffered insert/update values */
  std::unordered_set<uint64_t> live;  /* live keys, for snapshot enumeration */
  /* transaction staging (validate-then-apply commit) */
  std::vector<uint64_t> stg_del_k;
  std::vector<uint64_t> stg_upd_k, stg_upd_v;
  std::vector<uint64_t> stg_ins_k, stg_ins_v;
  /* frozen live-key set for a paged snapshot */
  std::vector<uint64_t> snap_cache;
};

/* ---- internal helpers ------------------------------------------------- */
static void insert_chunk(RgiEngine *e, const uint64_t *keys, const uint32_t *vals, uint32_t n) {
  RGI_CK(cudaMemcpy(e->d_keys, keys, sizeof(uint64_t) * n, cudaMemcpyHostToDevice));
  RGI_CK(cudaMemcpy(e->d_vals, vals, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));
  e->table->insert<true>(e->d_keys, 2, nullptr, e->d_vals, n, 0, /*update_if_exists=*/true);
  RGI_CK(cudaDeviceSynchronize());
}

static void find_chunk(RgiEngine *e, const uint64_t *keys, uint32_t *out, uint32_t n) {
  RGI_CK(cudaMemcpy(e->d_keys, keys, sizeof(uint64_t) * n, cudaMemcpyHostToDevice));
  e->table->find<false, true>(e->d_keys, 2, nullptr, e->d_out, n);
  RGI_CK(cudaDeviceSynchronize());
  RGI_CK(cudaMemcpy(out, e->d_out, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost));
}

/* ---- public C ABI ----------------------------------------------------- */
extern "C" RgiEngine *rgi_create(uint32_t capacity, float fill_factor, float pool_ratio) {
  RgiEngine *e   = new RgiEngine();
  e->host_alloc  = new slab_t(pool_ratio);
  e->host_reclaim = new debra_t();
  e->table       = new table_t(*e->host_alloc, *e->host_reclaim,
                               (std::size_t)capacity, fill_factor);
  RGI_CK(cudaMalloc(&e->d_keys, sizeof(uint32_t) * 2 * (size_t)BATCH_MAX));
  RGI_CK(cudaMalloc(&e->d_vals, sizeof(uint32_t) * (size_t)BATCH_MAX));
  RGI_CK(cudaMalloc(&e->d_out,  sizeof(uint32_t) * (size_t)BATCH_MAX));
  return e;
}

extern "C" void rgi_flush(RgiEngine *e) {
  uint32_t total = (uint32_t)e->pend_k.size();
  for (uint32_t off = 0; off < total; off += BATCH_MAX) {
    uint32_t chunk = total - off; if (chunk > BATCH_MAX) chunk = BATCH_MAX;
    insert_chunk(e, e->pend_k.data() + off, e->pend_v.data() + off, chunk);
  }
  for (uint32_t i = 0; i < total; ++i) e->live.insert(e->pend_k[i]);
  e->pend_k.clear();
  e->pend_v.clear();
}

extern "C" int rgi_flush_unique(RgiEngine *e, uint64_t *dup_key) {
    uint32_t total = (uint32_t) e->pend_k.size();
    if (total == 0) return 0;

    /* (a) duplicate keys within the pending batch */
    std::unordered_set<uint64_t> seen;
    seen.reserve(total * 2);
    for (uint32_t i = 0; i < total; ++i)
        if (!seen.insert(e->pend_k[i]).second) {
            if (dup_key) *dup_key = e->pend_k[i];
            e->pend_k.clear(); e->pend_v.clear();   /* discard the failed batch */
            return 1;
        }

    /* (b) keys already present in the index (batched find, nothing applied yet) */
    std::vector<uint32_t> tmp(total);
    for (uint32_t off = 0; off < total; off += BATCH_MAX) {
        uint32_t chunk = (total - off > BATCH_MAX) ? BATCH_MAX : (total - off);
        find_chunk(e, e->pend_k.data() + off, tmp.data() + off, chunk);
    }
    for (uint32_t i = 0; i < total; ++i)
        if (tmp[i] != RGI_INVALID) {
            if (dup_key) *dup_key = e->pend_k[i];
            e->pend_k.clear(); e->pend_v.clear();   /* discard the failed batch */
            return 1;
        }

    /* no conflict: apply the batch */
    rgi_flush(e);
    return 0;
}

extern "C" void rgi_insert(RgiEngine *e, uint64_t key, uint64_t value) {
  e->pend_k.push_back(key);
  e->pend_v.push_back((uint32_t)value);
}

extern "C" void rgi_update(RgiEngine *e, uint64_t key, uint64_t value) {
  /* RGI insert with update_if_exists overwrites; live-set add is idempotent. */
  e->pend_k.push_back(key);
  e->pend_v.push_back((uint32_t)value);
}

extern "C" void rgi_delete(RgiEngine *e, uint64_t key) {
  rgi_flush(e);                          /* make sure key exists in the index */
  uint64_t k = key;
  RGI_CK(cudaMemcpy(e->d_keys, &k, sizeof(uint64_t), cudaMemcpyHostToDevice));
  e->table->erase<true, true>(e->d_keys, 2, nullptr, 1);
  RGI_CK(cudaDeviceSynchronize());
  e->live.erase(key);
}

extern "C" int rgi_lookup(RgiEngine *e, uint64_t key, uint64_t *out_value) {
  rgi_flush(e);
  uint32_t v = RGI_INVALID;
  find_chunk(e, &key, &v, 1);
  if (v == RGI_INVALID) return 0;
  if (out_value) *out_value = (uint64_t)v;
  return 1;
}

extern "C" void rgi_find_many(RgiEngine *e, const uint64_t *keys,
                              uint64_t *out_values, int *found, uint32_t n) {
    rgi_flush(e);
    std::vector<uint32_t> tmp(n ? n : 1);
    for (uint32_t off = 0; off < n; off += BATCH_MAX) {
        uint32_t chunk = (n - off > BATCH_MAX) ? BATCH_MAX : (n - off);
        find_chunk(e, keys + off, tmp.data() + off, chunk);
    }
    for (uint32_t i = 0; i < n; ++i) {
        found[i]      = (tmp[i] != RGI_INVALID);
        out_values[i] = (uint64_t) tmp[i];
    }
}

extern "C" uint64_t rgi_snapshot(RgiEngine *e, uint64_t **out_keys, uint64_t **out_values) {
  rgi_flush(e);
  uint64_t n = (uint64_t)e->live.size();
  uint64_t *ks = (uint64_t *)malloc(sizeof(uint64_t) * (n ? n : 1));
  uint64_t *vs = (uint64_t *)malloc(sizeof(uint64_t) * (n ? n : 1));
  std::vector<uint64_t> keys(e->live.begin(), e->live.end());
  std::vector<uint32_t> tmp(keys.size());
  for (uint64_t off = 0; off < n; off += BATCH_MAX) {
    uint32_t chunk = (uint32_t)((n - off > BATCH_MAX) ? BATCH_MAX : (n - off));
    find_chunk(e, keys.data() + off, tmp.data() + off, chunk);
  }
  uint64_t live_n = 0;
  for (uint64_t i = 0; i < n; ++i) {
    if (tmp[i] == RGI_INVALID) continue;   /* erased after enumeration */
    ks[live_n] = keys[i];
    vs[live_n] = (uint64_t)tmp[i];
    live_n++;
  }
  if (out_keys)   *out_keys   = ks; else free(ks);
  if (out_values) *out_values = vs; else free(vs);
  return live_n;
}

/* ---- transaction staging (validate-then-apply) ----------------------- */
static void erase_chunk(RgiEngine *e, const uint64_t *keys, uint32_t n) {
  RGI_CK(cudaMemcpy(e->d_keys, keys, sizeof(uint64_t) * n, cudaMemcpyHostToDevice));
  e->table->erase<true, true>(e->d_keys, 2, nullptr, n);
  RGI_CK(cudaDeviceSynchronize());
}

extern "C" void rgi_stage_begin(RgiEngine *e) {
  e->stg_del_k.clear();
  e->stg_upd_k.clear(); e->stg_upd_v.clear();
  e->stg_ins_k.clear(); e->stg_ins_v.clear();
}

extern "C" void rgi_stage_abort(RgiEngine *e) { rgi_stage_begin(e); }

extern "C" void rgi_stage_del(RgiEngine *e, const uint64_t *keys, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) e->stg_del_k.push_back(keys[i]);
}

extern "C" void rgi_stage_upd(RgiEngine *e, const uint64_t *keys, const uint64_t *vals, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) { e->stg_upd_k.push_back(keys[i]); e->stg_upd_v.push_back(vals[i]); }
}

extern "C" void rgi_stage_ins(RgiEngine *e, const uint64_t *keys, const uint64_t *vals, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) { e->stg_ins_k.push_back(keys[i]); e->stg_ins_v.push_back(vals[i]); }
}

extern "C" int rgi_stage_commit(RgiEngine *e, uint64_t *dup_key) {
  /* make sure prior buffered writes are visible before we validate against the index */
  rgi_flush(e);

  const uint32_t ni = (uint32_t) e->stg_ins_k.size();

  /* ---- VALIDATE (no mutation): every error condition is checked here ---- */
  /* (a) duplicate insert keys within this commit set */
  if (ni) {
    std::unordered_set<uint64_t> seen;
    seen.reserve(ni * 2);
    for (uint32_t i = 0; i < ni; ++i)
      if (!seen.insert(e->stg_ins_k[i]).second) {
        if (dup_key) *dup_key = e->stg_ins_k[i];
        rgi_stage_abort(e);
        return 1;
      }
    /* (b) insert keys already present in the index */
    std::vector<uint32_t> tmp(ni);
    for (uint32_t off = 0; off < ni; off += BATCH_MAX) {
      uint32_t chunk = (ni - off > BATCH_MAX) ? BATCH_MAX : (ni - off);
      find_chunk(e, e->stg_ins_k.data() + off, tmp.data() + off, chunk);
    }
    for (uint32_t i = 0; i < ni; ++i)
      if (tmp[i] != RGI_INVALID) {
        if (dup_key) *dup_key = e->stg_ins_k[i];
        rgi_stage_abort(e);
        return 1;
      }
  }

  /* ---- APPLY (no expected failure path) -------------------------------- */
  /* deletes: erase is a no-op for absent keys */
  {
    const uint32_t nd = (uint32_t) e->stg_del_k.size();
    for (uint32_t off = 0; off < nd; off += BATCH_MAX) {
      uint32_t chunk = (nd - off > BATCH_MAX) ? BATCH_MAX : (nd - off);
      erase_chunk(e, e->stg_del_k.data() + off, chunk);
    }
    for (uint32_t i = 0; i < nd; ++i) e->live.erase(e->stg_del_k[i]);
  }
  /* updates + inserts: insert with update_if_exists never fails (validated) */
  {
    std::vector<uint64_t> ak; std::vector<uint32_t> av;
    ak.reserve(e->stg_upd_k.size() + ni);
    av.reserve(e->stg_upd_k.size() + ni);
    for (size_t i = 0; i < e->stg_upd_k.size(); ++i) { ak.push_back(e->stg_upd_k[i]); av.push_back((uint32_t) e->stg_upd_v[i]); }
    for (uint32_t i = 0; i < ni; ++i)                { ak.push_back(e->stg_ins_k[i]); av.push_back((uint32_t) e->stg_ins_v[i]); }
    const uint32_t na = (uint32_t) ak.size();
    for (uint32_t off = 0; off < na; off += BATCH_MAX) {
      uint32_t chunk = (na - off > BATCH_MAX) ? BATCH_MAX : (na - off);
      insert_chunk(e, ak.data() + off, av.data() + off, chunk);
    }
    for (uint32_t i = 0; i < na; ++i) e->live.insert(ak[i]);
  }

  rgi_stage_abort(e);   /* clear staged set */
  return 0;
}

/* ---- paged snapshot --------------------------------------------------- */
extern "C" uint64_t rgi_snapshot_begin(RgiEngine *e) {
  rgi_flush(e);
  e->snap_cache.assign(e->live.begin(), e->live.end());
  return (uint64_t) e->snap_cache.size();
}

extern "C" uint32_t rgi_snapshot_page(RgiEngine *e, uint64_t off, uint32_t max,
                                      uint64_t *out_keys, uint64_t *out_vals) {
  uint64_t total = (uint64_t) e->snap_cache.size();
  if (off >= total) return 0;
  uint32_t span = (uint32_t) ((total - off > (uint64_t) max) ? max : (total - off));
  std::vector<uint32_t> tmp(span);
  find_chunk(e, e->snap_cache.data() + off, tmp.data(), span);
  uint32_t m = 0;
  for (uint32_t i = 0; i < span; ++i) {
    if (tmp[i] == RGI_INVALID) continue;     /* erased after the set was frozen */
    out_keys[m] = e->snap_cache[off + i];
    out_vals[m] = (uint64_t) tmp[i];
    m++;
  }
  return m;
}

extern "C" void rgi_destroy(RgiEngine *e) {
  if (!e) return;
  cudaFree(e->d_keys); cudaFree(e->d_vals); cudaFree(e->d_out);
  delete e->table; delete e->host_reclaim; delete e->host_alloc; delete e;
}

/* ============================ microbenchmark =========================== */
#ifdef BUILD_BENCH
#include <chrono>
static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}
int main(int argc, char **argv) {
  uint32_t n = (argc > 1) ? (uint32_t)strtoul(argv[1], 0, 10) : 5000000u;
  RgiEngine *e = rgi_create(n, 2.0f, 0.5f);
  double t0 = now_s();
  for (uint32_t i = 0; i < n; ++i) rgi_insert(e, (uint64_t)i + 1, i + 1);
  rgi_flush(e);
  double t1 = now_s();
  printf("buffered insert+flush: %.1f Mop/s\n", n / (t1 - t0) / 1e6);
  uint64_t *ks, *vs;
  t0 = now_s();
  uint64_t m = rgi_snapshot(e, &ks, &vs);
  t1 = now_s();
  printf("snapshot %llu rows: %.1f Mop/s\n", (unsigned long long)m, m / (t1 - t0) / 1e6);
  uint64_t v; int f = rgi_lookup(e, 12345, &v);
  printf("lookup 12345 -> found=%d value=%llu\n", f, (unsigned long long)v);
  free(ks); free(vs);
  rgi_destroy(e);
  return 0;
}
#endif
