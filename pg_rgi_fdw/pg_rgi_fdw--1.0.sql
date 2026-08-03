\echo Use "CREATE EXTENSION pg_rgi_fdw" to load this file. \quit

CREATE FUNCTION pg_rgi_fdw_handler()
RETURNS fdw_handler AS 'MODULE_PATHNAME' LANGUAGE C STRICT;

CREATE FUNCTION pg_rgi_fdw_validator(text[], oid)
RETURNS void AS 'MODULE_PATHNAME' LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER pg_rgi_fdw
  HANDLER pg_rgi_fdw_handler
  VALIDATOR pg_rgi_fdw_validator;

-- GPU-service worker test API (shared, multi-user index owned by the bgworker)
CREATE FUNCTION gpu_svc_insert(bigint, bigint) RETURNS void
  AS 'MODULE_PATHNAME','gpu_svc_insert' LANGUAGE C STRICT VOLATILE;
CREATE FUNCTION gpu_svc_lookup(bigint) RETURNS bigint
  AS 'MODULE_PATHNAME','gpu_svc_lookup' LANGUAGE C STRICT VOLATILE;
