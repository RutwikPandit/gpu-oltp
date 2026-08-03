CREATE EXTENSION IF NOT EXISTS pg_rgi_fdw;
CREATE SERVER IF NOT EXISTS rgi FOREIGN DATA WRAPPER pg_rgi_fdw;

CREATE TABLE IF NOT EXISTS kv_ref (k bigint PRIMARY KEY, v bigint);
TRUNCATE kv_ref;
CREATE FOREIGN TABLE IF NOT EXISTS kv_rgi (k bigint, v bigint) SERVER rgi;

INSERT INTO kv_ref SELECT g, g * 7 FROM generate_series(1, 100000) AS g;
INSERT INTO kv_rgi SELECT g, g * 7 FROM generate_series(1, 100000) AS g;

EXPLAIN (COSTS OFF) SELECT v FROM kv_rgi WHERE k = 50000;
SELECT v FROM kv_ref WHERE k = 50000;
SELECT v FROM kv_rgi WHERE k = 50000;

SELECT count(*) AS hits, sum(v) AS value_sum
FROM kv_rgi
WHERE k IN (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233);

PREPARE demo_multiget(bigint[]) AS
  SELECT count(*) AS hits, sum(v) AS value_sum
  FROM kv_rgi
  WHERE k = ANY($1);

EXPLAIN (COSTS OFF) EXECUTE demo_multiget('{1,100,500,50000,99999}');
EXECUTE demo_multiget('{1,100,500,50000,99999}');
EXECUTE demo_multiget('{2,4,6,8,10,12,14,16,18,20,22,24,26,28,30}');

UPDATE kv_ref SET v = v + 1 WHERE k % 7 = 0;
UPDATE kv_rgi SET v = v + 1 WHERE k % 7 = 0;

BEGIN;
UPDATE kv_rgi SET v = 424242 WHERE k = 42;
SELECT v AS inside_txn_sees_own_write FROM kv_rgi WHERE k = 42;
ROLLBACK;
SELECT v AS after_rollback_gpu_index_unchanged FROM kv_rgi WHERE k = 42;

BEGIN;
DELETE FROM kv_rgi WHERE k = 2;
INSERT INTO kv_rgi VALUES (7, 70);
COMMIT;

SELECT v AS key_2_still_present FROM kv_rgi WHERE k = 2;
SELECT v AS key_7_original_value FROM kv_rgi WHERE k = 7;

UPDATE kv_rgi SET v = v + 1;
SELECT count(*) AS rows_after_gpu_pulse FROM kv_rgi;
