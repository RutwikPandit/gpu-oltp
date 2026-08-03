\echo [session B - a DIFFERENT connection] reading what session A wrote
SELECT gpu_svc_lookup(42)     AS k42_expect_1234;
SELECT gpu_svc_lookup(7)      AS k7_expect_70;
SELECT gpu_svc_lookup(100000) AS k100000_expect_999;
SELECT gpu_svc_lookup(55555)  AS missing_expect_null;
