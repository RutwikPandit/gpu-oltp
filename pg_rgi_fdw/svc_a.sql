\echo [session A] inserting via GPU service worker
SELECT gpu_svc_insert(42, 1234);
SELECT gpu_svc_insert(7, 70);
SELECT gpu_svc_insert(100000, 999);
