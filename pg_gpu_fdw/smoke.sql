\set ON_ERROR_STOP on
\echo === create extension + foreign table ===
CREATE EXTENSION IF NOT EXISTS pg_gpu_fdw;
CREATE SERVER IF NOT EXISTS gpu_oltp FOREIGN DATA WRAPPER pg_gpu_fdw;
DROP FOREIGN TABLE IF EXISTS kv;
CREATE FOREIGN TABLE kv (k bigint, v bigint) SERVER gpu_oltp;

\echo === INSERT (executes on GPU) ===
INSERT INTO kv VALUES (42, 1234), (7, 70), (100, 9999), (256, 1);

\echo === SELECT * (GPU snapshot) ===
SELECT * FROM kv ORDER BY k;

\echo === point SELECT WHERE k = 42 ===
SELECT v FROM kv WHERE k = 42;

\echo === UPDATE k=42 -> v=5555 ===
UPDATE kv SET v = 5555 WHERE k = 42;
SELECT v FROM kv WHERE k = 42;

\echo === DELETE k=7 ===
DELETE FROM kv WHERE k = 7;
SELECT * FROM kv ORDER BY k;

\echo === count ===
SELECT count(*) AS rows_on_gpu FROM kv;
