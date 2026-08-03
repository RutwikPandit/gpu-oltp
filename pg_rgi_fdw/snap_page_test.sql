-- Paged snapshot test: SELECT * must not silently truncate at GPU_SVC_BULK_CAP.
\set ON_ERROR_STOP on
DROP FOREIGN TABLE IF EXISTS kv_rgi;
CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;

\echo === insert 300000 rows (> 262144 bulk cap => multiple snapshot pages) ===
INSERT INTO kv_rgi SELECT g, g*3 FROM generate_series(1,300000) g;

\echo === full scan count must equal 300000 (was capped at 262144 before paging) ===
SELECT count(*) AS rows_expect_300000 FROM kv_rgi;

\echo === aggregates over the full (paged) scan ===
SELECT min(k) AS min_expect_1, max(k) AS max_expect_300000 FROM kv_rgi;
SELECT sum(v) AS sum_expect_135000450000 FROM kv_rgi;   -- 3 * sum(1..300000)

\echo === VERDICT ===
SELECT CASE WHEN (SELECT count(*) FROM kv_rgi) = 300000
            THEN 'PASS: full scan returns all rows (no truncation)'
            ELSE 'FAIL: scan truncated' END AS result;
