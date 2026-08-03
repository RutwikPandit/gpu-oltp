-- Differential correctness test: GPU (kv_rgi) vs traditional Postgres heap (kv_ref).
-- The SAME ops are applied to both; results are compared with set-difference.
-- A correct implementation yields ZERO diffs in both directions.
\set ON_ERROR_STOP on
\echo === setup: heap oracle + GPU foreign table ===
CREATE EXTENSION IF NOT EXISTS pg_rgi_fdw;
CREATE SERVER IF NOT EXISTS rgi FOREIGN DATA WRAPPER pg_rgi_fdw;
DROP TABLE IF EXISTS kv_ref;
CREATE TABLE kv_ref (k bigint PRIMARY KEY, v bigint);
DROP FOREIGN TABLE IF EXISTS kv_rgi;
CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;

\echo === op 1: bulk INSERT 100k rows into BOTH ===
INSERT INTO kv_ref  SELECT g, g*7 FROM generate_series(1,100000) g;
INSERT INTO kv_rgi  SELECT g, g*7 FROM generate_series(1,100000) g;

\echo === op 2: UPDATE every 7th key in BOTH ===
UPDATE kv_ref SET v = v + 1 WHERE k % 7 = 0;
UPDATE kv_rgi SET v = v + 1 WHERE k % 7 = 0;

\echo === op 3: DELETE every 13th key in BOTH ===
DELETE FROM kv_ref WHERE k % 13 = 0;
DELETE FROM kv_rgi WHERE k % 13 = 0;

\echo === compare row counts (must be equal) ===
SELECT (SELECT count(*) FROM kv_ref) AS ref_rows,
       (SELECT count(*) FROM kv_rgi) AS gpu_rows;

\echo === set-difference both directions (BOTH must be 0) ===
SELECT count(*) AS gpu_minus_ref FROM (SELECT * FROM kv_rgi EXCEPT SELECT * FROM kv_ref) d;
SELECT count(*) AS ref_minus_gpu FROM (SELECT * FROM kv_ref EXCEPT SELECT * FROM kv_rgi) d;

\echo === point-query spot checks (GPU = heap) ===
SELECT (SELECT v FROM kv_rgi WHERE k = 7)   AS gpu_k7,   (SELECT v FROM kv_ref WHERE k = 7)   AS ref_k7;
SELECT (SELECT v FROM kv_rgi WHERE k = 13)  AS gpu_k13,  (SELECT v FROM kv_ref WHERE k = 13)  AS ref_k13;  -- both NULL (deleted)
SELECT (SELECT v FROM kv_rgi WHERE k = 100) AS gpu_k100, (SELECT v FROM kv_ref WHERE k = 100) AS ref_k100;

\echo === VERDICT ===
SELECT CASE
  WHEN (SELECT count(*) FROM (SELECT * FROM kv_rgi EXCEPT SELECT * FROM kv_ref) d) = 0
   AND (SELECT count(*) FROM (SELECT * FROM kv_ref EXCEPT SELECT * FROM kv_rgi) d) = 0
   AND (SELECT count(*) FROM kv_ref) = (SELECT count(*) FROM kv_rgi)
  THEN 'PASS: GPU matches traditional Postgres'
  ELSE 'FAIL: divergence detected'
END AS result;
