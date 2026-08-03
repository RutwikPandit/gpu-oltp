-- Bandwidth-bound aggregate scan, CPU baseline (native Postgres heap table).
-- Data is "already loaded" (build is not the measured op); warm runs read from
-- the OS page cache / RAM, so this measures CPU DRAM-bound scan throughput.
-- Parallel seq scan enabled to give the CPU all the bandwidth it can use.
\timing on
SET max_parallel_workers_per_gather = 8;
SET max_parallel_workers = 8;
SET work_mem = '256MB';

\echo ===== CPU: build 50M-row column (LOAD, not measured) =====
DROP TABLE IF EXISTS scan_cpu;
CREATE TABLE scan_cpu (v bigint);
INSERT INTO scan_cpu SELECT g FROM generate_series(1, 50000000) g;
VACUUM (ANALYZE) scan_cpu;
SELECT pg_size_pretty(pg_relation_size('scan_cpu')) AS heap_size;

\echo ===== CPU: SUM(v)  (run twice; 2nd = warm/cached) =====
SELECT sum(v) FROM scan_cpu;
SELECT sum(v) FROM scan_cpu;

\echo ===== CPU: COUNT(v < 25,000,000)  (run twice; 2nd = warm) =====
SELECT count(*) FROM scan_cpu WHERE v < 25000000;
SELECT count(*) FROM scan_cpu WHERE v < 25000000;
