-- Key-changing UPDATE + delete-then-reinsert (the reviewer's gate cases).
-- ON_ERROR_STOP off so the expected unique violation in D doesn't stop the script.
DROP FOREIGN TABLE IF EXISTS kv_rgi;
CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;
INSERT INTO kv_rgi VALUES (1,10),(2,20);
SELECT count(*) AS start_expect_2 FROM kv_rgi;

\echo === D: UPDATE SET k=2 WHERE k=1 must FAIL (unique); both rows preserved ===
UPDATE kv_rgi SET k = 2 WHERE k = 1;          -- expected: ERROR (k=2 already exists)
SELECT k, v FROM kv_rgi ORDER BY k;            -- expect (1,10),(2,20) unchanged
SELECT count(*) AS rows_expect_2 FROM kv_rgi;

\echo === E: DELETE k=1 then INSERT (1,99) in one txn must SUCCEED (final v=99) ===
BEGIN;
DELETE FROM kv_rgi WHERE k = 1;
INSERT INTO kv_rgi VALUES (1, 99);
COMMIT;
SELECT v AS k1_expect_99 FROM kv_rgi WHERE k = 1;
SELECT count(*) AS rows_expect_2b FROM kv_rgi;

\echo === F: legal key change to a FREE key must SUCCEED ===
UPDATE kv_rgi SET k = 3 WHERE k = 2;           -- k=3 is free -> ok
SELECT k, v FROM kv_rgi ORDER BY k;            -- expect (1,99),(3,20)
