#!/usr/bin/env bash
# Regenerate all benchmark figures from the (freshly updated) plot scripts.
set -e
BENCH="/mnt/c/Users/rutwi/OneDrive/Documents/CMU/Research_spring26/Research_spring26/gpu_oltp/bench"
cd "$BENCH"
for s in make_plots.py make_plots2.py make_plots3.py make_plots4.py make_plots5.py make_plots6.py make_plots7.py; do
  echo "--- $s ---"
  python3 "$s"
done
if [ -f "$HOME/gpu_bench/fig9_dram_schmoo.png" ]; then
  cp "$HOME/gpu_bench/fig9_dram_schmoo.png" "$BENCH/"
  echo "fig9 copied from ~/gpu_bench"
fi
echo "--- PNG timestamps ---"
ls -l --time-style=+"%Y-%m-%d %H:%M" "$BENCH"/*.png
