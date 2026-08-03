\echo [session B - DIFFERENT connection] reading the SHARED kv_rgi table
SELECT count(*)                         AS rows_expect_1001 FROM kv_rgi;
SELECT v                                AS k500_expect_3500 FROM kv_rgi WHERE k = 500;
SELECT v                                AS k123456_expect_42 FROM kv_rgi WHERE k = 123456;
SELECT count(*) AS anyof5_expect_5 FROM kv_rgi WHERE k = ANY('{1,2,3,500,1000}'::bigint[]);
\echo [session B] duplicate insert of an existing key -> expect UNIQUE violation (shared PK)
INSERT INTO kv_rgi VALUES (500, 999);
