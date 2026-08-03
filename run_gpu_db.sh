#!/usr/bin/env bash
# Single-command launcher for the GPU OLTP Postgres server (WSL/Ubuntu).
#   Usage (from a WSL shell):     bash run_gpu_db.sh
#   Or from Windows PowerShell:   wsl -d Ubuntu -- bash <path>/run_gpu_db.sh
#
# Starts the Postgres 14 cluster, ensures the GPU FDW extensions + a foreign
# table exist, then drops you into psql. SQL you type executes on the GPU.
#
# NOTE: the GPU index is SHARED across connections (owned by the pg_gpu_service
# background worker; requires shared_preload_libraries='pg_rgi_fdw'). Data is
# visible to all sessions and persists for the postmaster's lifetime; it does
# NOT survive a cluster restart (GPU memory is volatile; no WAL yet).
set -e
PW="${PGSUDO_PW:?set PGSUDO_PW to your sudo password}"

echo ">> starting postgres 14 cluster ..."
echo "$PW" | sudo -S -k pg_ctlcluster 14 main start 2>/dev/null || true
echo "$PW" | sudo -S -k pg_lsclusters

echo ">> ensuring GPU OLTP extension + foreign table ..."
echo "$PW" | sudo -S -k -u postgres psql -X -v ON_ERROR_STOP=0 <<'SQL'
CREATE EXTENSION IF NOT EXISTS pg_rgi_fdw;
CREATE SERVER IF NOT EXISTS rgi FOREIGN DATA WRAPPER pg_rgi_fdw;
CREATE FOREIGN TABLE IF NOT EXISTS kv_rgi (k bigint, v bigint) SERVER rgi;
SQL

echo ">> opening psql (GPU OLTP). Try:"
echo "     INSERT INTO kv_rgi SELECT g, g*7 FROM generate_series(1,100000) g;"
echo "     SELECT * FROM kv_rgi WHERE k IN (1, 50000, 100000);"
echo "   (data is SHARED across sessions via the GPU service worker.)"
exec sudo -u postgres psql -X
