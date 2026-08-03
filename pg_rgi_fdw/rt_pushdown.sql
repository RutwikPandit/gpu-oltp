\timing on
INSERT INTO kv_rgi SELECT g, g*7 FROM generate_series(1,100000) g;

\echo == bound-parameter multi-get: k = ANY($1)  (realistic case -> pushed) ==
PREPARE q(bigint[]) AS SELECT count(*) FROM kv_rgi WHERE k = ANY($1);
EXECUTE q('{1,100,500,50000,99999}');
EXECUTE q('{2,4,6,8,10,12,14,16,18,20,22,24,26,28,30}');

\echo == subquery array: k = ANY(ARRAY(SELECT ...))  (must NOT crash; falls back) ==
SELECT count(*) FROM kv_rgi WHERE k = ANY(ARRAY(SELECT generate_series(1,100000,100)));

\echo == plain point + const-array still work ==
SELECT v FROM kv_rgi WHERE k = 50000;
SELECT count(*) FROM kv_rgi WHERE k IN (1,2,3,5,8,13,21);
