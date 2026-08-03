\timing on
INSERT INTO kv_rgi SELECT g, g*7 FROM generate_series(1,100000) g;
\echo -- k = ANY(literal const array)  -> should push down (fast)
SELECT count(*) FROM kv_rgi WHERE k = ANY('{1,100,500,50000,99999}'::bigint[]);
SELECT k,v FROM kv_rgi WHERE k = ANY('{1,100,50000}'::bigint[]) ORDER BY k;
\echo -- k IN (literal list) -> should push down (fast)
SELECT count(*) FROM kv_rgi WHERE k IN (2,4,8,16,32,64);
