#!/usr/bin/env bash
# Regenerate all figures from the (freshly updated) plot scripts.
set -e
cd "$(dirname "$0")"
for s in make_plots.py make_plots2.py make_plots3.py make_plots4.py make_plots5.py make_plots6.py make_plots7.py; do
  echo "== $s =="
  python3 "$s" 2>&1 | tail -4
done
# fig9 (schmoo) was generated with fresh ncu data in ~/gpu_bench; bring it here.
if [ -f "$HOME/gpu_bench/fig9_dram_schmoo.png" ]; then
  cp "$HOME/gpu_bench/fig9_dram_schmoo.png" .
  echo "fig9 copied from ~/gpu_bench"
fi
echo "== DONE =="
ls -la fig*.png | sort -k9
