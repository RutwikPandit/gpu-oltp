#!/usr/bin/env bash
# Build all standalone benchmark binaries into ~/gpu_bench (WSL home, not OneDrive).
set -e
export PATH="/usr/local/cuda/bin:$PATH"
ROOT="/mnt/c/Users/rutwi/OneDrive/Documents/CMU/Research_spring26/Research_spring26"
RGI="$ROOT/RobustGPUIndexing/include"
ENG="$ROOT/gpu_oltp/engine"
NVF="-std=c++17 -arch=sm_89 --expt-extended-lambda --expt-relaxed-constexpr -maxrregcount=64"
OUT="$HOME/gpu_bench"
mkdir -p "$OUT"
cd "$OUT"

echo ">> rgi_sweep"
nvcc $NVF -I"$RGI" "$ENG/rgi_sweep.cu" -o rgi_sweep

echo ">> rgi_bench (engine -DBUILD_BENCH)"
nvcc $NVF -DBUILD_BENCH -I"$RGI" "$ENG/rgi_oltp_engine.cu" -o rgi_bench

echo ">> rgi_prof (SOL target)"
nvcc $NVF -I"$RGI" "$ENG/rgi_prof.cu" -o rgi_prof

echo ">> rgi_prof2 (schmoo target)"
nvcc $NVF -I"$RGI" "$ENG/rgi_prof2.cu" -o rgi_prof2

echo ">> oltp_bench (toy engine, bandwidth scan + persistent kernel)"
nvcc -O3 -arch=sm_89 -DBUILD_BENCH "$ENG/gpu_oltp_engine.cu" -o oltp_bench

echo ">> cpu_sweep"
g++ -O3 -march=native -fopenmp "$ENG/cpu_sweep.cpp" -o cpu_sweep

echo ">> ALL BUILT"
ls -la "$OUT"
