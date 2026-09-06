#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <ingress-binary> <comparator-binary> <work-directory>" >&2
  exit 2
fi

ingress_binary=$1
comparator_binary=$2
work_directory=$3
baseline="$work_directory/baseline.json"

mkdir -p "$work_directory"

"$ingress_binary" \
  --backend mock \
  --warmup 8 \
  --iterations 64 \
  --json-out "$baseline" \
  >/dev/null

comparison=$(
  "$comparator_binary" \
    --baseline "$baseline" \
    --candidate "$baseline" \
    --scope e2e \
    --metric p99 \
    --max-regression-percent 0
)

case "$comparison" in
  *'"status":"pass"'*) ;;
  *)
    echo "comparator did not return a passing JSON result: $comparison" >&2
    exit 1
    ;;
esac

set +e
"$comparator_binary" \
  --baseline "$baseline" \
  --candidate "$baseline" \
  --scope device \
  --metric p99 \
  >/dev/null 2>&1
device_status=$?
set -e

if [[ $device_status -ne 2 ]]; then
  echo "expected unavailable mock device metrics to return 2, got $device_status" >&2
  exit 1
fi

echo "metrics regression integration: pass"
