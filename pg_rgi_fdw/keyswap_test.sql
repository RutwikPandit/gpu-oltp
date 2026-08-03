-- Multi-row key-changing UPDATE: swaps/chains are rejected (not silently wrong);
-- non-overlapping renames succeed. ON_ERROR_STOP off to see each outcome.
DROP FOREIGN TABLE IF EXISTS kv_rgi;
CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;
INSERT INTO kv_rgi VALUES (1,10),(2,20);
SELECT count(*) AS start_expect_2 FROM kv_rgi;

\echo === G: key SWAP (1<->2) must be REJECTED cleanly, rows unchanged ===
BEGIN;
UPDATE kv_rgi SET k = CASE WHEN k = 1 THEN 2 ELSE 1 END WHERE k IN (1,2);  -- expect ERROR
COMMIT;
SELECT k, v FROM kv_rgi ORDER BY k;            -- expect (1,10),(2,20) unchanged

\echo === H: key CHAIN (k=k+1) must be REJECTED cleanly, rows unchanged ===
BEGIN;
UPDATE kv_rgi SET k = k + 1 WHERE k IN (1,2);  -- expect ERROR (2 is both old and new)
COMMIT;
SELECT k, v FROM kv_rgi ORDER BY k;            -- expect (1,10),(2,20) unchanged

\echo === I: non-overlapping multi-row rename (k=k+100) must SUCCEED ===
BEGIN;
UPDATE kv_rgi SET k = k + 100 WHERE k IN (1,2);
COMMIT;
SELECT k, v FROM kv_rgi ORDER BY k;            -- expect (101,10),(102,20)
SELECT count(*) AS rows_expect_2 FROM kv_rgi;
