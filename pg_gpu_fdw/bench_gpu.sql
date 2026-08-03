-- YCSB-shaped GPU run: same statements as bench_cpu.sql against the GPU FDW.
-- MUST run in a single psql connection (one persistent-kernel engine per backend).
\timing on
\set N 50000

\echo ===== GPU: setup (foreign table on persistent-kernel engine) =====
DROP FOREIGN TABLE IF EXISTS kv_gpu;
CREATE FOREIGN TABLE kv_gpu (k bigint, v bigint) SERVER gpu_oltp;

\echo ===== GPU: LOAD (:N inserts, one engine round-trip per row) =====
INSERT INTO kv_gpu SELECT g, g*7 FROM generate_series(1, :N) g;

\echo ===== GPU: full scan count (GPU snapshot) =====
SELECT count(*) FROM kv_gpu;
\echo ===== GPU: full scan aggregate (sum) =====
SELECT sum(v) FROM kv_gpu;

\echo ===== GPU: point SELECT x3 (NO pushdown -> full snapshot each) =====
SELECT v FROM kv_gpu WHERE k = 1;
SELECT v FROM kv_gpu WHERE k = 25000;
SELECT v FROM kv_gpu WHERE k = 50000;

\echo ===== GPU: full-table UPDATE (:N updates) =====
UPDATE kv_gpu SET v = v + 1;

\echo ===== GPU: DELETE half =====
DELETE FROM kv_gpu WHERE k <= 25000;
SELECT count(*) FROM kv_gpu;
