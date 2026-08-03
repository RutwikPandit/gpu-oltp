#!/usr/bin/env bash
# Run the Postgres per-query benchmark (fig6) and bandwidth scan (fig3) on a
# fresh GPU index. Assumes the cluster is online and pg_rgi_fdw preloaded.
PW="${PGSUDO_PW:?set PGSUDO_PW to your sudo password}"
ROOT="/mnt/c/Users/rutwi/OneDrive/Documents/CMU/Research_spring26/Research_spring26"
RGI="$ROOT/gpu_oltp/pg_rgi_fdw"
GPU="$ROOT/gpu_oltp/pg_gpu_fdw"

echo ">> restart cluster for a clean GPU index"
echo "$PW" | sudo -S -k pg_ctlcluster 14 main restart >/dev/null 2>&1
sleep 3

echo ""
echo "################## PER-QUERY (fig6): CPU heap vs GPU RGI ##################"
echo "$PW" | sudo -S -k -u postgres psql -X -f "$RGI/query_bench.sql" 2>&1

echo ""
echo "################## BANDWIDTH SCAN (fig3): GPU path ##################"
echo "$PW" | sudo -S -k -u postgres psql -X -c "CREATE EXTENSION IF NOT EXISTS pg_gpu_fdw;" 2>&1
echo "$PW" | sudo -S -k -u postgres psql -X -f "$GPU/bench_bw_gpu.sql" 2>&1

echo ""
echo "################## BANDWIDTH SCAN (fig3): CPU baseline ##################"
echo "$PW" | sudo -S -k -u postgres psql -X -f "$GPU/bench_bw_cpu.sql" 2>&1

echo ">> SQL BENCH DONE"
