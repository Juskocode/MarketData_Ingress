#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
BIN="${BUILD_DIR}/bin/marketdata_ingress"

"${BIN}" --backend mock --iterations 64 --warmup 8 --json-out "${BUILD_DIR}/marketdata_metrics_pr_64.json" --target-us 50000
"${BIN}" --backend mock --iterations 32 --input-size 128 --warmup 4 --json-out "${BUILD_DIR}/marketdata_metrics_pr_128.json"
"${BIN}" --backend mock --iterations 32 --input-size 1024 --warmup 4 --json-out "${BUILD_DIR}/marketdata_metrics_pr_1024.json"

echo "[pr] tests passed"
