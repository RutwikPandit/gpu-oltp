-- YCSB-shaped CPU baseline: native Postgres heap table + btree primary key.
-- Same statements as bench_gpu.sql; run single-connection for apples-to-apples.
\timing on
\set N 50000

\echo ===== CPU: setup (heap table + btree PK) =====
DROP TABLE IF EXISTS kv_cpu;
CREATE TABLE kv_cpu (k bigint PRIMARY KEY, v bigint);

\echo ===== CPU: LOAD (:N inserts) =====
INSERT INTO kv_cpu SELECT g, g*7 FROM generate_series(1, :N) g;

\echo ===== CPU: full scan count =====
SELECT count(*) FROM kv_cpu;
\echo ===== CPU: full scan aggregate (sum) =====
SELECT sum(v) FROM kv_cpu;

\echo ===== CPU: point SELECT x3 (btree probe) =====
SELECT v FROM kv_cpu WHERE k = 1;
SELECT v FROM kv_cpu WHERE k = 25000;
SELECT v FROM kv_cpu WHERE k = 50000;

\echo ===== CPU: full-table UPDATE (:N updates) =====
UPDATE kv_cpu SET v = v + 1;

\echo ===== CPU: DELETE half =====
DELETE FROM kv_cpu WHERE k <= 25000;
SELECT count(*) FROM kv_cpu;
