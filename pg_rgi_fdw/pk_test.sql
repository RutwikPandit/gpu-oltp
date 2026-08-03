-- PK/UNIQUE enforcement test (ON_ERROR_STOP off so we see each outcome)
DROP FOREIGN TABLE IF EXISTS kv_rgi;
CREATE FOREIGN TABLE kv_rgi (k bigint, v bigint) SERVER rgi;
INSERT INTO kv_rgi SELECT g, g*7 FROM generate_series(1,1000) g;
\echo == duplicate of an existing key (expect UNIQUE violation) ==
INSERT INTO kv_rgi VALUES (500, 999);
\echo == duplicate within one batch (expect UNIQUE violation) ==
INSERT INTO kv_rgi VALUES (5000,1),(5000,2);
\echo == brand-new keys (expect success) ==
INSERT INTO kv_rgi VALUES (2000, 11),(2001, 22);
SELECT count(*) AS rows_after_inserts FROM kv_rgi;
\echo == UPDATE still overwrites (no error) ==
UPDATE kv_rgi SET v = 42 WHERE k = 500;
SELECT v AS v_at_500 FROM kv_rgi WHERE k = 500;
\echo == after a failed insert, session still healthy (insert new key) ==
INSERT INTO kv_rgi VALUES (3000, 33);
SELECT count(*) AS final_rows FROM kv_rgi;
