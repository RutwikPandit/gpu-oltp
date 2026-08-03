-- Commit atomicity tests (the bug fixed by stage->validate->apply).
-- ON_ERROR_STOP off so the script continues past the expected PK errors.
DROP FOREIGN TABLE IF EXISTS kv_rgi;
CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;
INSERT INTO kv_rgi SELECT g, g*7 FROM generate_series(1,5) g;
SELECT count(*) AS start_expect_5 FROM kv_rgi;

\echo === A: delete-then-failed-insert must NOT apply the delete ===
-- Old code applied deletes before validating inserts, so the delete of k=2
-- leaked even though the commit aborted. Correct behavior: k=2 survives.
BEGIN;
DELETE FROM kv_rgi WHERE k = 2;          -- staged delete
INSERT INTO kv_rgi VALUES (3, 999);      -- k=3 already exists -> PK violation at COMMIT
COMMIT;                                   -- expected: ERROR, nothing applied
SELECT v AS k2_expect_14 FROM kv_rgi WHERE k = 2;     -- must still be 14 (delete rolled back)
SELECT count(*) AS rows_expect_5 FROM kv_rgi;          -- unchanged

\echo === B: multi-chunk commit with a duplicate must be all-or-nothing ===
-- >262144 fresh rows (spans several GPU_SVC_BULK_CAP chunks) plus one duplicate.
-- None of the 300k may be applied if the commit aborts.
BEGIN;
INSERT INTO kv_rgi SELECT g, g FROM generate_series(1000, 301000) g;   -- 300001 fresh keys
INSERT INTO kv_rgi VALUES (4, 123);                                    -- k=4 already exists -> dup
COMMIT;                                   -- expected: ERROR, nothing applied
SELECT count(*) AS rows_expect_5 FROM kv_rgi;          -- still 5, the 300k were discarded

\echo === C: same large commit WITHOUT a duplicate commits fully ===
BEGIN;
INSERT INTO kv_rgi SELECT g, g FROM generate_series(1000, 301000) g;   -- 300001 fresh keys
COMMIT;
SELECT count(*) AS rows_expect_300006 FROM kv_rgi;     -- 5 + 300001
