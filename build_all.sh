#!/usr/bin/env bash
# Build the RGI engine shared lib + the FDW/worker extension (WSL/Ubuntu).
set -e
export PATH="/usr/local/cuda/bin:$PATH"

ROOT="/mnt/c/Users/rutwi/OneDrive/Documents/CMU/Research_spring26/Research_spring26"
GPU="$ROOT/gpu_oltp"
RGI="$ROOT/RobustGPUIndexing/include"

echo ">> building librgioltp.so ..."
cd "$GPU"
rm -f librgioltp.so
nvcc -std=c++17 -arch=sm_89 --expt-extended-lambda --expt-relaxed-constexpr \
     -maxrregcount=64 -Xcompiler -fPIC -shared -I"$RGI" \
     engine/rgi_oltp_engine.cu -o librgioltp.so
ls -la librgioltp.so

echo ">> building pg_rgi_fdw (PGXS) ..."
cd "$GPU/pg_rgi_fdw"
make clean >/dev/null 2>&1 || true
make 2>&1 | tail -50
ls -la pg_rgi_fdw.so

echo ">> BUILD OK"
