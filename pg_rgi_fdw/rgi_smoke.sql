\set ON_ERROR_STOP on
\timing on
\echo === create RGI-backed extension + foreign table ===
CREATE EXTENSION IF NOT EXISTS pg_rgi_fdw;
CREATE SERVER IF NOT EXISTS rgi FOREIGN DATA WRAPPER pg_rgi_fdw;
DROP FOREIGN TABLE IF EXISTS kv_rgi;
CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;

\echo === batched INSERT ... SELECT (one warp-cooperative flush on GPU) ===
INSERT INTO kv_rgi SELECT g, g*7 FROM generate_series(1, 200000) g;

\echo === SELECT * sanity (GPU snapshot via RGI batched find) ===
SELECT count(*) AS rows_on_gpu FROM kv_rgi;
SELECT * FROM kv_rgi WHERE k IN (1, 100000, 200000) ORDER BY k;

\echo === UPDATE + DELETE on GPU ===
UPDATE kv_rgi SET v = 999 WHERE k = 100000;
SELECT v FROM kv_rgi WHERE k = 100000;
DELETE FROM kv_rgi WHERE k = 1;
SELECT count(*) AS after_delete FROM kv_rgi;
