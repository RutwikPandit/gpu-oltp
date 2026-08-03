\set ON_ERROR_STOP on
\pset pager off
\if :{?demo_timing}
  \timing :demo_timing
\else
  \timing off
\endif

-- Live demo script for pg_rgi_fdw.
--
-- Run through demo/run_demo.sh for a clean GPU service worker, or run manually:
--   sudo -u postgres psql -X -f demo/demo_fast_path.sql
--
-- This script intentionally highlights fast paths:
--   1. equality predicate pushdown
--   2. bound-parameter multi-key lookup
--   3. batched transactional writes
--   4. validate-then-apply primary-key failure
--
-- It avoids the known slow path:
--   k = ANY(ARRAY(SELECT ...)) falls back to snapshot scan.

\echo
\echo '============================================================'
\echo 'GPU-OLTP demo: Postgres SQL -> FDW -> GPU service -> RGI'
\echo '============================================================'
\echo
\echo 'Setup completed quietly: extension, server, CPU table, GPU foreign table.'

\echo
\echo '1) The GPU service worker is alive'
\echo '   These helper functions are test-only, but they prove the shared worker owns the GPU index.'
\prompt 'Press Enter to call the GPU service helper functions...' demo_pause
\echo 'postgres=# SELECT gpu_svc_insert(900000001, 12345);'
SELECT gpu_svc_insert(900000001, 12345);
\echo 'postgres=# SELECT gpu_svc_lookup(900000001) AS gpu_service_lookup_result;'
SELECT gpu_svc_lookup(900000001) AS gpu_service_lookup_result;

\echo
\echo '2) Bulk load the same 100k rows into CPU heap and GPU foreign table'
\echo '   GPU path buffers rows and batch-applies them to the RGI index at commit.'
\prompt 'Press Enter to bulk-load 100k rows into CPU and GPU tables...' demo_pause
\echo '-- CPU heap load'
\echo 'postgres=# INSERT INTO kv_ref SELECT g, g * 7 FROM generate_series(1, 100000) AS g;'
INSERT INTO kv_ref SELECT g, g * 7 FROM generate_series(1, 100000) AS g;
\echo '-- GPU RGI load'
\echo 'postgres=# INSERT INTO kv_rgi SELECT g, g * 7 FROM generate_series(1, 100000) AS g;'
INSERT INTO kv_rgi SELECT g, g * 7 FROM generate_series(1, 100000) AS g;

\echo
\echo '3) Point lookup: predicate pushdown turns WHERE k = 50000 into one GPU index find'
\echo '-- Postgres still sees this as a normal query over a foreign table'
\prompt 'Press Enter to show the plan and run a point lookup...' demo_pause
\echo 'postgres=# EXPLAIN (COSTS OFF) SELECT v FROM kv_rgi WHERE k = 50000;'
EXPLAIN (COSTS OFF) SELECT v FROM kv_rgi WHERE k = 50000;
\echo '-- CPU heap + btree'
\echo 'postgres=# SELECT v FROM kv_ref WHERE k = 50000;'
SELECT v FROM kv_ref WHERE k = 50000;
\echo '-- GPU RGI FDW'
\echo 'postgres=# SELECT v FROM kv_rgi WHERE k = 50000;'
SELECT v FROM kv_rgi WHERE k = 50000;

\echo
\echo '4) Literal IN list: many keys become one batched GPU find'
\prompt 'Press Enter to run a literal IN-list multi-key lookup...' demo_pause
\echo 'postgres=# SELECT count(*) AS hits, sum(v) AS value_sum FROM kv_rgi WHERE k IN (...);'
SELECT count(*) AS hits, sum(v) AS value_sum
FROM kv_rgi
WHERE k IN (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233);

\echo
\echo '5) Bound-parameter multi-get: realistic app shape, still pushed down'
\prompt 'Press Enter to prepare and execute k = ANY($1) multi-get queries...' demo_pause
\echo 'postgres=# PREPARE demo_multiget(bigint[]) AS SELECT count(*), sum(v) FROM kv_rgi WHERE k = ANY($1);'
PREPARE demo_multiget(bigint[]) AS
  SELECT count(*) AS hits, sum(v) AS value_sum
  FROM kv_rgi
  WHERE k = ANY($1);

\echo 'postgres=# EXPLAIN (COSTS OFF) EXECUTE demo_multiget(...);'
EXPLAIN (COSTS OFF) EXECUTE demo_multiget('{1,100,500,50000,99999}');
\echo 'postgres=# EXECUTE demo_multiget(''{1,100,500,50000,99999}'');'
EXECUTE demo_multiget('{1,100,500,50000,99999}');
\echo 'postgres=# EXECUTE demo_multiget(''{2,4,6,8,10,12,14,16,18,20,22,24,26,28,30}'');'
EXECUTE demo_multiget('{2,4,6,8,10,12,14,16,18,20,22,24,26,28,30}');

\echo
\echo '6) Bulk update: Postgres identifies rows; commit batch-applies writes to the GPU index'
\prompt 'Press Enter to run the CPU and GPU bulk updates...' demo_pause
\echo '-- CPU heap'
\echo 'postgres=# UPDATE kv_ref SET v = v + 1 WHERE k % 7 = 0;'
UPDATE kv_ref SET v = v + 1 WHERE k % 7 = 0;
\echo '-- GPU RGI FDW'
\echo 'postgres=# UPDATE kv_rgi SET v = v + 1 WHERE k % 7 = 0;'
UPDATE kv_rgi SET v = v + 1 WHERE k % 7 = 0;

\echo
\echo '7) Read-your-writes and rollback'
\prompt 'Press Enter to demonstrate BEGIN, read-your-writes, and ROLLBACK...' demo_pause
\echo 'postgres=# BEGIN;'
BEGIN;
\echo 'postgres=# UPDATE kv_rgi SET v = 424242 WHERE k = 42;'
UPDATE kv_rgi SET v = 424242 WHERE k = 42;
\echo 'postgres=# SELECT v AS inside_txn_sees_own_write FROM kv_rgi WHERE k = 42;'
SELECT v AS inside_txn_sees_own_write FROM kv_rgi WHERE k = 42;
\echo 'postgres=# ROLLBACK;'
ROLLBACK;
\echo 'postgres=# SELECT v AS after_rollback_gpu_index_unchanged FROM kv_rgi WHERE k = 42;'
SELECT v AS after_rollback_gpu_index_unchanged FROM kv_rgi WHERE k = 42;

\echo
\echo '8) Atomic commit / PK validation: failing insert must not leak the delete'
\prompt 'Press Enter to run the expected failing COMMIT...' demo_pause
\echo 'postgres=# BEGIN;'
BEGIN;
\echo 'postgres=# DELETE FROM kv_rgi WHERE k = 2;'
DELETE FROM kv_rgi WHERE k = 2;
\echo 'postgres=# INSERT INTO kv_rgi VALUES (7, 70);'
INSERT INTO kv_rgi VALUES (7, 70);
\echo '   COMMIT should fail because key 7 already exists.'
\echo '   After the error, key 2 must still exist.'
\set ON_ERROR_STOP off
\echo 'postgres=# COMMIT;'
COMMIT;
\set ON_ERROR_STOP on

\echo '-- Verify the failed commit did not apply the delete.'
\echo 'postgres=# SELECT v AS key_2_still_present FROM kv_rgi WHERE k = 2;'
SELECT v AS key_2_still_present FROM kv_rgi WHERE k = 2;
\echo 'postgres=# SELECT v AS key_7_original_value FROM kv_rgi WHERE k = 7;'
SELECT v AS key_7_original_value FROM kv_rgi WHERE k = 7;

\echo
\echo '9) Optional GPU pulse: one normal SQL statement that batch-applies 100k writes'
\echo '   This is not the main benchmark; it gives nvidia-smi something easier to catch.'
\prompt 'Press Enter to run UPDATE kv_rgi SET v = v + 1 over the whole table...' demo_pause
\echo 'postgres=# UPDATE kv_rgi SET v = v + 1;'
UPDATE kv_rgi SET v = v + 1;
\echo 'postgres=# SELECT count(*) AS rows_after_gpu_pulse FROM kv_rgi;'
SELECT count(*) AS rows_after_gpu_pulse FROM kv_rgi;

\echo
\echo '10) Fast-path demo complete'
\echo '   What they saw: real psql, GPU service alive, pushed key lookups,'
\echo '   batched writes, rollback, and validate-before-mutate atomic commit.'
