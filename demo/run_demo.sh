#!/usr/bin/env bash
set -euo pipefail

# Live demo runner for the fast pg_rgi_fdw path.
#
# Usage from WSL:
#   bash demo/run_demo.sh
#
# Optional:
#   PGSUDO_PW=your_password bash demo/run_demo.sh
#
# The restart is intentional: the GPU service worker owns an in-memory index.
# Restarting Postgres gives the demo a clean GPU index and avoids duplicate-key
# leftovers from previous runs.

PW="${PGSUDO_PW:?set PGSUDO_PW to your sudo password}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL="$SCRIPT_DIR/demo_clean.sql"
TIMING="off"
DEMO_DELAY="${DEMO_DELAY:-0.8}"
DEMO_COLOR="${DEMO_COLOR:-1}"

case "${1:-}" in
  --time|-t)
    TIMING="on"
    ;;
  "" )
    ;;
  * )
    echo "Usage: bash demo/run_demo.sh [--time]"
    exit 2
    ;;
esac

printf '%s\n' "$PW" | sudo -S -p '' -k -v >/dev/null
sudo -n pg_ctlcluster 14 main restart >/dev/null
sleep 3

PSQL_ENV=(env PGOPTIONS="-c client_min_messages=warning")

feed_sql() {
  local file="$1"
  while IFS= read -r line || [[ -n "$line" ]]; do
    printf '%s\n' "$line"
    if [[ "$line" =~ \;[[:space:]]*$ ]]; then
      sleep "$DEMO_DELAY"
    fi
  done < "$file"
}

colorize_psql() {
  if [[ "$DEMO_COLOR" != "1" ]]; then
    cat
    return
  fi

  awk '
    BEGIN {
      cyan = "\033[36m";
      red = "\033[31m";
      reset = "\033[0m";
    }
    /^ERROR:/ || /^DETAIL:/ {
      print red $0 reset;
      next;
    }
    /^[[:space:]]*(SELECT|INSERT|UPDATE|DELETE|BEGIN|COMMIT|ROLLBACK|CREATE|DROP|TRUNCATE|PREPARE|EXECUTE|EXPLAIN|FROM|WHERE|SET)[[:space:];(]/ {
      print cyan $0 reset;
      next;
    }
    {
      print;
    }
  '
}

if [[ "$TIMING" == "on" ]]; then
  { printf '\\timing on\n'; sleep "$DEMO_DELAY"; feed_sql "$SQL"; } | sudo -n -u postgres "${PSQL_ENV[@]}" psql -X -a -P pager=off -v ON_ERROR_STOP=0 | colorize_psql
else
  feed_sql "$SQL" | sudo -n -u postgres "${PSQL_ENV[@]}" psql -X -a -P pager=off -v ON_ERROR_STOP=0 | colorize_psql
fi

echo
echo "Demo complete."
