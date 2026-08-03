\echo Use "CREATE EXTENSION pg_gpu_fdw" to load this file. \quit

CREATE FUNCTION pg_gpu_fdw_handler()
RETURNS fdw_handler AS 'MODULE_PATHNAME' LANGUAGE C STRICT;

CREATE FUNCTION pg_gpu_fdw_validator(text[], oid)
RETURNS void AS 'MODULE_PATHNAME' LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER pg_gpu_fdw
  HANDLER pg_gpu_fdw_handler
  VALIDATOR pg_gpu_fdw_validator;

-- Bandwidth-scan API: data resident in GPU HBM, aggregate executed on the GPU.
CREATE FUNCTION gpu_load(bigint)        RETURNS bigint AS 'MODULE_PATHNAME','gpu_scan_load'        LANGUAGE C STRICT VOLATILE;
CREATE FUNCTION gpu_sum()               RETURNS bigint AS 'MODULE_PATHNAME','gpu_scan_agg_sum'     LANGUAGE C VOLATILE;
CREATE FUNCTION gpu_count_lt(bigint)    RETURNS bigint AS 'MODULE_PATHNAME','gpu_scan_agg_count_lt' LANGUAGE C STRICT VOLATILE;
CREATE FUNCTION gpu_kernel_ms()         RETURNS double precision AS 'MODULE_PATHNAME','gpu_scan_kernel_ms' LANGUAGE C VOLATILE;
