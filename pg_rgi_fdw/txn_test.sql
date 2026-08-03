-- Transaction test (single session). ON_ERROR_STOP off so T4 continues after the expected error.
SELECT count(*) AS start_rows FROM kv_rgi;

\echo === T1: ROLLBACK discards writes ===
BEGIN;
INSERT INTO kv_rgi VALUES (1,10),(2,20);
SELECT count(*) AS in_txn_expect_2 FROM kv_rgi;          -- read-your-writes
ROLLBACK;
SELECT count(*) AS after_rollback_expect_0 FROM kv_rgi;

\echo === T2: COMMIT applies atomically ===
BEGIN;
INSERT INTO kv_rgi VALUES (1,10),(2,20),(3,30);
SELECT v AS k2_in_txn_expect_20 FROM kv_rgi WHERE k=2;
COMMIT;
SELECT count(*) AS after_commit_expect_3 FROM kv_rgi;

\echo === T3: read-your-writes for UPDATE/DELETE, then ROLLBACK ===
BEGIN;
UPDATE kv_rgi SET v=999 WHERE k=1;
SELECT v AS k1_in_txn_expect_999 FROM kv_rgi WHERE k=1;
DELETE FROM kv_rgi WHERE k=2;
SELECT count(*) AS rows_in_txn_expect_2 FROM kv_rgi;
ROLLBACK;
SELECT v AS k1_after_rollback_expect_10 FROM kv_rgi WHERE k=1;
SELECT count(*) AS rows_after_rollback_expect_3 FROM kv_rgi;

\echo === T4: PK violation at COMMIT aborts the transaction ===
BEGIN;
INSERT INTO kv_rgi VALUES (1, 111);
COMMIT;
SELECT v AS k1_expect_10 FROM kv_rgi WHERE k=1;
