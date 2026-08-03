-- Bandwidth-bound aggregate scan, GPU path. Column resides in GPU HBM
-- (gpu_load = the "already loaded" step, not the measured op). Aggregates run
-- on the GPU streaming HBM; only a scalar crosses PCIe. Run in ONE psql session.
\timing on

\echo ===== GPU: load 50M-element column into HBM (LOAD, not measured) =====
SELECT gpu_load(50000000) AS n_resident;

\echo ===== GPU: SUM(v)  (run twice; report 2nd) =====
SELECT gpu_sum();
SELECT gpu_sum();
SELECT gpu_kernel_ms() AS sum_kernel_ms;

\echo ===== GPU: COUNT(v < 25,000,000)  (run twice; report 2nd) =====
SELECT gpu_count_lt(25000000);
SELECT gpu_count_lt(25000000);
SELECT gpu_kernel_ms() AS count_kernel_ms;
