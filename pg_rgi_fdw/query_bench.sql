-- Per-statement-type benchmark: CPU heap (kv_ref) vs GPU RGI (kv_rgi), same SQL.
\set ON_ERROR_STOP on
\timing on
CREATE EXTENSION IF NOT EXISTS pg_rgi_fdw;
CREATE SERVER IF NOT EXISTS rgi FOREIGN DATA WRAPPER pg_rgi_fdw;
DROP TABLE IF EXISTS kv_ref;
CREATE TABLE kv_ref (k bigint PRIMARY KEY, v bigint);
DROP FOREIGN TABLE IF EXISTS kv_rgi;
CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;

\echo ============ BULK INSERT 100k ============
\echo -- CPU
INSERT INTO kv_ref SELECT g, g*7 FROM generate_series(1,100000) g;
\echo -- GPU
INSERT INTO kv_rgi SELECT g, g*7 FROM generate_series(1,100000) g;

\echo ============ POINT SELECT (single row) ============
\echo -- CPU
SELECT v FROM kv_ref WHERE k = 50000;
\echo -- GPU
SELECT v FROM kv_rgi WHERE k = 50000;

\echo ============ BATCHED POINT SELECT (k = ANY of 1000) ============
\echo -- CPU
SELECT count(v) FROM kv_ref WHERE k = ANY (ARRAY(SELECT generate_series(1,100000,100)));
\echo -- GPU
SELECT count(v) FROM kv_rgi WHERE k = ANY (ARRAY(SELECT generate_series(1,100000,100)));

\echo ============ FULL SCAN count(*) ============
\echo -- CPU
SELECT count(*) FROM kv_ref;
\echo -- GPU
SELECT count(*) FROM kv_rgi;

\echo ============ SINGLE-ROW UPDATE ============
\echo -- CPU
UPDATE kv_ref SET v = 1 WHERE k = 50000;
\echo -- GPU
UPDATE kv_rgi SET v = 1 WHERE k = 50000;

\echo ============ BULK UPDATE (every 7th row) ============
\echo -- CPU
UPDATE kv_ref SET v = v + 1 WHERE k % 7 = 0;
\echo -- GPU
UPDATE kv_rgi SET v = v + 1 WHERE k % 7 = 0;

\echo ============ SINGLE-ROW DELETE ============
\echo -- CPU
DELETE FROM kv_ref WHERE k = 1;
\echo -- GPU
DELETE FROM kv_rgi WHERE k = 1;
