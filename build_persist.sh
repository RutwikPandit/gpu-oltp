#!/usr/bin/env bash
# Build + run the v1 persistent-kernel RGI engine bench, and regenerate figures.
set -e
export PATH="/usr/local/cuda/bin:$PATH"
ROOT="/mnt/c/Users/rutwi/OneDrive/Documents/CMU/Research_spring26/Research_spring26"
RGI="$ROOT/RobustGPUIndexing/include"
ENG="$ROOT/gpu_oltp/engine"
BENCH="$ROOT/gpu_oltp/bench"
OUT="$HOME/gpu_bench"
mkdir -p "$OUT"

echo ">> building rgi_persist ..."
nvcc -std=c++17 -arch=sm_89 --expt-extended-lambda --expt-relaxed-constexpr \
     -maxrregcount=64 -I"$RGI" "$ENG/rgi_persist_engine.cu" -o "$OUT/rgi_persist"
echo ">> BUILD OK"

echo ">> running rgi_persist (2M keys) ..."
timeout 600 "$OUT/rgi_persist" 2000000
echo ">> RUN OK"

echo ">> regenerating figures ..."
cd "$BENCH"
for s in make_plots.py make_plots2.py make_plots3.py make_plots4.py make_plots5.py make_plots6.py make_plots7.py; do
  python3 "$s" 2>&1 | tail -1
done
cp "$OUT/fig9_dram_schmoo.png" "$BENCH/" 2>/dev/null && echo "fig9 copied"
ls -la "$BENCH"/fig*.png | head -15
echo ">> ALL DONE"
