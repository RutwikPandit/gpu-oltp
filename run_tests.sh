#!/usr/bin/env bash
# Install the freshly built FDW/worker, restart the cluster (reloads the GPU
# worker), and run the correctness + atomicity + paging test suite.
export PATH="/usr/local/cuda/bin:$PATH"
PW="${PGSUDO_PW:?set PGSUDO_PW to your sudo password}"
ROOT="/mnt/c/Users/rutwi/OneDrive/Documents/CMU/Research_spring26/Research_spring26"
FDW="$ROOT/gpu_oltp/pg_rgi_fdw"

cd "$FDW"
echo ">> make install ..."
echo "$PW" | sudo -S -k make install 2>&1 | tail -5

# Each test needs a clean GPU index. HBM is volatile, so a cluster restart
# (which restarts the GPU worker and recreates an empty engine) is the reset.
run() {
  echo ""
  echo "============================================================"
  echo ">> $1   (fresh GPU index via cluster restart)"
  echo "============================================================"
  echo "$PW" | sudo -S -k pg_ctlcluster 14 main restart >/dev/null 2>&1 || true
  sleep 2
  echo "$PW" | sudo -S -k -u postgres psql -X -v ON_ERROR_STOP=0 -f "$FDW/$1" 2>&1
}

run correctness.sql
run txn_test.sql
run pk_test.sql
run atomic_test.sql
run keyupd_test.sql
run keyswap_test.sql
run snap_page_test.sql
echo ""
echo ">> TESTS DONE"
